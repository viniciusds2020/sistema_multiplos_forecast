"""Pagina 5: Comparacao e avaliacao de modelos."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from dashboard.components import (
    kpi_card, chart_container, section_divider, create_plotly_layout,
    plot_metrics_comparison, plot_radar_chart,
)
from config import COLORS, MODEL_COLORS, AVAILABLE_MODELS
from pipeline.forecasting_pipeline import run_forecast_pipeline
from evaluation.metrics import calculate_all_metrics


def render(df: pd.DataFrame):
    """Renderiza pagina de comparacao de modelos."""
    st.markdown("# Comparacao de Modelos")
    st.markdown("Avalie e compare o desempenho dos modelos de forecast em todos os SKUs")
    section_divider()

    with st.sidebar:
        st.markdown("### Configuracoes")
        available_models = ["XGBoost", "LightGBM", "AutoARIMA", "Prophet"]
        try:
            import torch
            available_models.append("Chronos")
        except ImportError:
            pass
        available_models.append("CrostonSBA")

        selected_models = st.multiselect(
            "Modelos para Comparacao",
            available_models,
            default=["XGBoost", "LightGBM", "AutoARIMA"],
            key="comp_models",
        )
        horizon = st.slider("Horizonte", 7, 90, 30, key="comp_horizon")
        test_days = st.slider("Dias de teste", 30, 120, 60, key="comp_test_days")

        selected_skus = st.multiselect(
            "SKUs para Avaliar",
            sorted(df["sku_id"].unique()),
            default=sorted(df["sku_id"].unique())[:6],
            key="comp_skus",
        )

    if st.button("Executar Comparacao Completa", type="primary", use_container_width=True):
        all_results = {}
        progress = st.progress(0)

        for idx, sku_id in enumerate(selected_skus):
            progress.progress((idx + 1) / len(selected_skus), f"SKU {sku_id}...")
            result = run_forecast_pipeline(
                df, sku_id, selected_models, test_days, horizon,
            )
            all_results[sku_id] = result

        progress.progress(1.0, "Concluido!")
        st.session_state["comparison_results"] = all_results

    if "comparison_results" not in st.session_state:
        st.info("Clique em 'Executar Comparacao Completa' para avaliar os modelos.")
        return

    all_results = st.session_state["comparison_results"]

    # Montar dataframe de metricas
    metrics_rows = []
    for sku_id, models in all_results.items():
        sku_name = df[df["sku_id"] == sku_id]["sku_name"].iloc[0]
        profile = df[df["sku_id"] == sku_id]["demand_profile"].iloc[0]
        for model_name, res in models.items():
            if res["metrics"] is not None:
                metrics_rows.append({
                    "SKU": sku_id,
                    "Nome": sku_name,
                    "Perfil": profile,
                    "Model": model_name,
                    **res["metrics"],
                })

    if not metrics_rows:
        st.warning("Nenhum resultado disponivel.")
        return

    metrics_df = pd.DataFrame(metrics_rows)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Ranking Geral", "Comparacao Visual",
        "Analise por SKU", "Analise de Residuos",
    ])

    with tab1:
        _render_ranking(metrics_df)
    with tab2:
        _render_visual_comparison(metrics_df)
    with tab3:
        _render_per_sku_analysis(metrics_df, all_results, df, test_days)
    with tab4:
        _render_residual_analysis(all_results, df, test_days)


def _render_ranking(metrics_df: pd.DataFrame):
    """Ranking geral de modelos."""
    chart_container("Ranking Geral dos Modelos")

    # Media por modelo
    model_avg = metrics_df.groupby("Model")[["MAE", "RMSE", "MAPE", "WAPE"]].mean().round(2)
    model_avg = model_avg.reset_index()

    # Destacar melhores
    st.dataframe(model_avg, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)

    with col1:
        chart_container("Melhor Modelo por Metrica")
        best_models = {}
        for col in ["MAE", "RMSE", "MAPE", "WAPE"]:
            best_idx = model_avg[col].idxmin()
            best_models[col] = model_avg.loc[best_idx, "Model"]

        best_df = pd.DataFrame([
            {"Metrica": k, "Melhor Modelo": v} for k, v in best_models.items()
        ])
        st.dataframe(best_df, use_container_width=True, hide_index=True)

    with col2:
        chart_container("Melhor Modelo por SKU (WAPE)")
        best_per_sku = metrics_df.loc[metrics_df.groupby("SKU")["WAPE"].idxmin()]
        best_summary = best_per_sku["Model"].value_counts().reset_index()
        best_summary.columns = ["Modelo", "Vezes Melhor"]

        fig = go.Figure(go.Bar(
            x=best_summary["Modelo"],
            y=best_summary["Vezes Melhor"],
            marker=dict(
                color=[MODEL_COLORS.get(m, COLORS["primary"]) for m in best_summary["Modelo"]],
            ),
            text=best_summary["Vezes Melhor"],
            textposition="outside",
        ))
        fig.update_layout(**create_plotly_layout("", 350))
        st.plotly_chart(fig, use_container_width=True)


def _render_visual_comparison(metrics_df: pd.DataFrame):
    """Comparacao visual com graficos."""
    model_avg = metrics_df.groupby("Model")[["MAE", "RMSE", "MAPE", "WAPE"]].mean().round(2).reset_index()

    # Radar Chart
    chart_container("Radar de Metricas (media)")
    fig = plot_radar_chart(model_avg, ["MAE", "RMSE", "MAPE", "WAPE"])
    st.plotly_chart(fig, use_container_width=True)

    section_divider()

    # Box plots por metrica
    metric_choice = st.selectbox("Metrica para Box Plot", ["WAPE", "MAE", "RMSE", "MAPE"])

    chart_container(f"Distribuicao de {metric_choice} por Modelo")
    fig = px.box(
        metrics_df, x="Model", y=metric_choice, color="Model",
        color_discrete_map=MODEL_COLORS,
        points="all",
    )
    fig.update_layout(**create_plotly_layout("", 450))
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    section_divider()

    # Grouped bar chart
    chart_container(f"{metric_choice} por SKU e Modelo")
    fig = px.bar(
        metrics_df, x="Nome", y=metric_choice, color="Model",
        barmode="group",
        color_discrete_map=MODEL_COLORS,
    )
    fig.update_layout(**create_plotly_layout("", 450))
    fig.update_layout(xaxis=dict(tickangle=-45))
    st.plotly_chart(fig, use_container_width=True)


def _render_per_sku_analysis(metrics_df, all_results, df, test_days):
    """Analise detalhada por SKU."""
    chart_container("Analise Detalhada por SKU")

    sku_choice = st.selectbox(
        "Selecionar SKU",
        metrics_df["SKU"].unique(),
        key="comp_sku_detail",
    )

    sku_metrics = metrics_df[metrics_df["SKU"] == sku_choice]
    sku_name = sku_metrics["Nome"].iloc[0]

    st.markdown(f"**{sku_name}** - Perfil: {sku_metrics['Perfil'].iloc[0]}")

    # Tabela metricas
    display_cols = ["Model", "MAE", "RMSE", "MAPE", "WAPE", "Bias"]
    available = [c for c in display_cols if c in sku_metrics.columns]
    st.dataframe(sku_metrics[available], use_container_width=True, hide_index=True)

    # Forecast overlay
    if sku_choice in all_results:
        sku_data = df[df["sku_id"] == sku_choice].sort_values("date")
        train = sku_data.iloc[:-test_days]
        test = sku_data.iloc[-test_days:]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=train["date"], y=train["demand"],
            mode="lines", name="Historico",
            line=dict(color=COLORS["text"], width=1.5),
        ))
        fig.add_trace(go.Scatter(
            x=test["date"], y=test["demand"],
            mode="lines", name="Real",
            line=dict(color=COLORS["dark"], width=2, dash="dot"),
        ))

        for model_name, res in all_results[sku_choice].items():
            if res["forecast"] is not None:
                fc = res["forecast"]
                color = MODEL_COLORS.get(model_name, COLORS["primary"])
                fig.add_trace(go.Scatter(
                    x=fc["ds"], y=fc["yhat"],
                    mode="lines", name=model_name,
                    line=dict(color=color, width=2),
                ))

        fig.update_layout(**create_plotly_layout(f"Forecast - {sku_name}", 450))
        st.plotly_chart(fig, use_container_width=True)


def _render_residual_analysis(all_results, df, test_days):
    """Analise de residuos."""
    chart_container("Analise de Residuos")

    skus_available = list(all_results.keys())
    sku_choice = st.selectbox("SKU", skus_available, key="residual_sku")

    models_available = [
        m for m, r in all_results[sku_choice].items() if r["forecast"] is not None
    ]
    model_choice = st.selectbox("Modelo", models_available, key="residual_model")

    if model_choice and sku_choice:
        res = all_results[sku_choice][model_choice]
        if res["forecast"] is not None:
            sku_data = df[df["sku_id"] == sku_choice].sort_values("date")
            test = sku_data.iloc[-test_days:]
            fc = res["forecast"]

            y_true = test["demand"].values[:len(fc)]
            y_pred = fc["yhat"].values[:len(y_true)]
            residuals = y_true - y_pred
            dates = test["date"].values[:len(residuals)]

            col1, col2 = st.columns(2)

            with col1:
                chart_container("Residuos ao Longo do Tempo")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=dates, y=residuals,
                    mode="lines+markers",
                    line=dict(color=COLORS["primary"]),
                    marker=dict(size=4),
                ))
                fig.add_hline(y=0, line_dash="dash", line_color=COLORS["danger"])
                fig.update_layout(**create_plotly_layout("", 350))
                fig.update_layout(yaxis=dict(title="Residuo"))
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                chart_container("Distribuicao dos Residuos")
                fig = go.Figure(go.Histogram(
                    x=residuals,
                    nbinsx=30,
                    marker=dict(color=COLORS["primary"], opacity=0.7),
                ))
                fig.add_vline(x=0, line_dash="dash", line_color=COLORS["danger"])
                fig.update_layout(**create_plotly_layout("", 350))
                fig.update_layout(xaxis=dict(title="Residuo"), yaxis=dict(title="Frequencia"))
                st.plotly_chart(fig, use_container_width=True)

            # ACF dos residuos
            chart_container("Autocorrelacao dos Residuos (ACF)")
            max_lag = min(30, len(residuals) - 1)
            acf_values = []
            for lag in range(max_lag + 1):
                if lag == 0:
                    acf_values.append(1.0)
                else:
                    acf_values.append(np.corrcoef(residuals[lag:], residuals[:-lag])[0, 1])

            fig = go.Figure(go.Bar(
                x=list(range(max_lag + 1)),
                y=acf_values,
                marker=dict(color=COLORS["primary"]),
            ))
            # Limites de significancia
            sig = 1.96 / np.sqrt(len(residuals))
            fig.add_hline(y=sig, line_dash="dash", line_color=COLORS["danger"], opacity=0.5)
            fig.add_hline(y=-sig, line_dash="dash", line_color=COLORS["danger"], opacity=0.5)
            fig.update_layout(**create_plotly_layout("", 350))
            fig.update_layout(xaxis=dict(title="Lag"), yaxis=dict(title="ACF"))
            st.plotly_chart(fig, use_container_width=True)
