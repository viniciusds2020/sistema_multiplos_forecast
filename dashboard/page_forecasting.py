"""Pagina 4: Forecasting - treinar modelos e gerar previsoes."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from dashboard.components import (
    kpi_card, chart_container, section_divider, create_plotly_layout,
    plot_forecast_chart,
)
from config import COLORS, MODEL_COLORS, AVAILABLE_MODELS
from pipeline.forecasting_pipeline import run_forecast_pipeline
from pipeline.cluster_pipeline import run_cluster_forecast_pipeline, run_cluster_analysis


def render(df: pd.DataFrame):
    """Renderiza pagina de forecasting."""
    st.markdown("# Forecasting")
    st.markdown("Treine modelos e gere previsoes por SKU individual ou por cluster agregado")
    section_divider()

    # Controles sidebar
    with st.sidebar:
        st.markdown("### Configuracoes de Forecast")

        mode = st.radio(
            "Modo de Forecast",
            ["Individual (por SKU)", "Agregado (por Cluster)"],
        )

        available_models = ["XGBoost", "LightGBM", "AutoARIMA", "Prophet"]

        # Chronos
        try:
            import torch
            available_models.append("Chronos")
        except ImportError:
            pass

        available_models.append("CrostonSBA")

        selected_models = st.multiselect(
            "Modelos",
            available_models,
            default=["XGBoost", "LightGBM", "AutoARIMA"],
        )

        horizon = st.slider("Horizonte (dias)", 7, 90, 30)
        test_days = st.slider("Dias de teste", 30, 120, 60)

    if mode == "Individual (por SKU)":
        _render_individual_mode(df, selected_models, horizon, test_days)
    else:
        _render_aggregated_mode(df, selected_models, horizon, test_days)


def _render_individual_mode(df, selected_models, horizon, test_days):
    """Modo de forecast individual por SKU."""
    sku_names = sorted(df["sku_name"].unique())
    selected_sku_name = st.selectbox("Selecionar SKU", sku_names)

    sku_id = df[df["sku_name"] == selected_sku_name]["sku_id"].iloc[0]
    demand_profile = df[df["sku_id"] == sku_id]["demand_profile"].iloc[0]

    col1, col2, col3 = st.columns(3)
    with col1:
        kpi_card("SKU", sku_id, border_color=COLORS["primary"])
    with col2:
        kpi_card("Perfil", demand_profile, border_color=COLORS["info"])
    with col3:
        avg = df[df["sku_id"] == sku_id]["demand"].mean()
        kpi_card("Media Demanda", f"{avg:.0f}", border_color=COLORS["success"])

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Executar Forecast", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def progress_cb(pct, msg):
            progress_bar.progress(min(pct, 1.0))
            status_text.text(msg)

        with st.spinner("Treinando modelos..."):
            results = run_forecast_pipeline(
                df, sku_id, selected_models, test_days, horizon,
            )

        progress_bar.progress(1.0)
        status_text.text("Concluido!")

        st.session_state["individual_results"] = results
        st.session_state["individual_sku_id"] = sku_id

    # Exibir resultados
    if "individual_results" in st.session_state:
        results = st.session_state["individual_results"]
        sku_data = df[df["sku_id"] == st.session_state.get("individual_sku_id")].sort_values("date")

        section_divider()

        tab_forecast, tab_metrics, tab_details = st.tabs([
            "Previsoes", "Metricas", "Detalhes dos Modelos",
        ])

        with tab_forecast:
            _render_forecast_charts(sku_data, results, test_days)

        with tab_metrics:
            _render_metrics_table(results)

        with tab_details:
            _render_model_details(results)


def _render_forecast_charts(sku_data, results, test_days):
    """Renderiza graficos de forecast."""
    train_data = sku_data.iloc[:-test_days]
    test_data = sku_data.iloc[-test_days:]

    # Overlay de todos os modelos
    chart_container("Comparacao de Forecasts")
    fig = go.Figure()

    # Historico
    fig.add_trace(go.Scatter(
        x=train_data["date"], y=train_data["demand"],
        mode="lines", name="Historico",
        line=dict(color=COLORS["text"], width=1.5),
    ))

    # Real (teste)
    fig.add_trace(go.Scatter(
        x=test_data["date"], y=test_data["demand"],
        mode="lines", name="Real (Teste)",
        line=dict(color=COLORS["dark"], width=2, dash="dot"),
    ))

    # Forecasts
    for model_name, res in results.items():
        if res["forecast"] is not None:
            fc = res["forecast"]
            color = MODEL_COLORS.get(model_name, COLORS["primary"])
            fig.add_trace(go.Scatter(
                x=fc["ds"], y=fc["yhat"],
                mode="lines", name=model_name,
                line=dict(color=color, width=2.5),
            ))

            # Intervalo de confianca
            fig.add_trace(go.Scatter(
                x=list(fc["ds"]) + list(fc["ds"][::-1]),
                y=list(fc["yhat_upper"]) + list(fc["yhat_lower"][::-1]),
                fill="toself",
                fillcolor=f"{color}10",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
            ))

    fig.update_layout(**create_plotly_layout("", 500))
    st.plotly_chart(fig, use_container_width=True)

    # Graficos individuais por modelo
    for model_name, res in results.items():
        if res["forecast"] is not None:
            with st.expander(f"Detalhe: {model_name}"):
                fig = plot_forecast_chart(
                    train_data["date"], train_data["demand"],
                    res["forecast"],
                    model_name=model_name,
                    actual_test=test_data["demand"].values,
                    test_dates=test_data["date"].values,
                )
                st.plotly_chart(fig, use_container_width=True)


def _render_metrics_table(results):
    """Renderiza tabela de metricas."""
    chart_container("Metricas de Avaliacao")

    rows = []
    for model_name, res in results.items():
        if res["metrics"] is not None:
            row = {"Model": model_name, **res["metrics"]}
            rows.append(row)
        elif res["error"]:
            rows.append({"Model": model_name, "error": res["error"]})

    if rows:
        metrics_df = pd.DataFrame(rows)

        # Destacar melhor modelo
        metric_cols = ["MAE", "RMSE", "MAPE", "WAPE"]
        for col in metric_cols:
            if col in metrics_df.columns:
                metrics_df[col] = pd.to_numeric(metrics_df[col], errors="coerce")

        st.dataframe(
            metrics_df,
            use_container_width=True,
            hide_index=True,
        )

        # Bar chart de WAPE
        if "WAPE" in metrics_df.columns:
            valid = metrics_df.dropna(subset=["WAPE"])
            if not valid.empty:
                fig = go.Figure(go.Bar(
                    x=valid["Model"],
                    y=valid["WAPE"],
                    marker=dict(
                        color=[MODEL_COLORS.get(m, COLORS["primary"]) for m in valid["Model"]],
                    ),
                    text=valid["WAPE"].round(1),
                    textposition="outside",
                ))
                fig.update_layout(**create_plotly_layout("WAPE por Modelo (%)", 400))
                st.plotly_chart(fig, use_container_width=True)


def _render_model_details(results):
    """Renderiza detalhes dos modelos."""
    for model_name, res in results.items():
        with st.expander(f"{model_name}"):
            if res["error"]:
                st.error(f"Erro: {res['error']}")
            else:
                st.json(res["params"])

                # Feature importance para ML models
                if model_name in ("XGBoost", "LightGBM"):
                    try:
                        from models.model_registry import get_model
                        # Se temos o modelo no resultado, mostrar importancia
                        st.markdown("*Feature importance disponivel apos treino*")
                    except Exception:
                        pass


def _render_aggregated_mode(df, selected_models, horizon, test_days):
    """Modo de forecast agregado por cluster."""
    st.markdown("### Forecast Agregado por Cluster")

    with st.sidebar:
        metric = st.selectbox(
            "Metrica de Similaridade",
            ["pearson", "euclidean", "dtw"],
            index=0,
            key="agg_metric",
        )
        weight_method = st.selectbox(
            "Metodo de Rateio",
            ["rolling", "static"],
            index=0,
        )

    if st.button("Executar Forecast Agregado", type="primary", use_container_width=True):
        with st.spinner("Executando analise de clusters e forecast..."):
            # Clustering
            cluster_info = run_cluster_analysis(df, metric=metric)

            # Forecast
            results = run_cluster_forecast_pipeline(
                df, cluster_info, selected_models,
                test_days=test_days, horizon=horizon,
                weight_method=weight_method,
            )

            st.session_state["cluster_forecast_results"] = results
            st.session_state["cluster_info_forecast"] = cluster_info

    # Exibir resultados
    if "cluster_forecast_results" in st.session_state:
        results = st.session_state["cluster_forecast_results"]
        cluster_info = st.session_state["cluster_info_forecast"]

        section_divider()

        # KPIs
        n_clusters = cluster_info["n_clusters"]
        col1, col2, col3 = st.columns(3)
        with col1:
            kpi_card("Clusters", str(n_clusters), border_color=COLORS["primary"])
        with col2:
            kpi_card("Modelos", str(len(selected_models)), border_color=COLORS["info"])
        with col3:
            kpi_card("Horizonte", f"{horizon} dias", border_color=COLORS["success"])

        st.markdown("<br>", unsafe_allow_html=True)

        tab_agg, tab_rateio, tab_weights = st.tabs([
            "Forecast Agregado", "Rateio por SKU", "Pesos de Rateio",
        ])

        with tab_agg:
            _render_aggregated_forecasts(df, results, cluster_info, test_days)

        with tab_rateio:
            _render_disaggregated_results(df, results, cluster_info)

        with tab_weights:
            _render_weights(results, cluster_info)


def _render_aggregated_forecasts(df, results, cluster_info, test_days):
    """Renderiza forecasts agregados por cluster."""
    from similarity.aggregation import aggregate_cluster_demand

    cluster_data = aggregate_cluster_demand(
        df, cluster_info["sku_ids"], cluster_info["labels"]
    )

    for cluster_id in sorted(results["cluster_forecasts"].keys()):
        with st.expander(f"Cluster {cluster_id}", expanded=(cluster_id == 1)):
            agg_df = cluster_data[cluster_id].sort_values("date")

            fig = go.Figure()

            # Historico
            train = agg_df.iloc[:-test_days]
            test = agg_df.iloc[-test_days:]

            fig.add_trace(go.Scatter(
                x=train["date"], y=train["demand_agg"],
                mode="lines", name="Historico",
                line=dict(color=COLORS["text"], width=1.5),
            ))

            fig.add_trace(go.Scatter(
                x=test["date"], y=test["demand_agg"],
                mode="lines", name="Real",
                line=dict(color=COLORS["dark"], width=2, dash="dot"),
            ))

            for model_name, fc in results["cluster_forecasts"][cluster_id].items():
                if fc is not None:
                    color = MODEL_COLORS.get(model_name, COLORS["primary"])
                    fig.add_trace(go.Scatter(
                        x=fc["ds"], y=fc["yhat"],
                        mode="lines", name=model_name,
                        line=dict(color=color, width=2.5),
                    ))

            fig.update_layout(**create_plotly_layout(f"Cluster {cluster_id} - Forecast Agregado", 400))
            st.plotly_chart(fig, use_container_width=True)

            # Metricas
            if cluster_id in results["metrics_agg"]:
                metrics = results["metrics_agg"][cluster_id]
                rows = [{"Model": m, **v} for m, v in metrics.items() if "error" not in v]
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_disaggregated_results(df, results, cluster_info):
    """Renderiza resultados desagregados (rateio)."""
    chart_container("Forecast Desagregado por SKU (Rateio Proporcional)")

    for cluster_id in sorted(results["disaggregated"].keys()):
        with st.expander(f"Cluster {cluster_id}"):
            for model_name in results["disaggregated"][cluster_id]:
                sku_forecasts = results["disaggregated"][cluster_id][model_name]
                if sku_forecasts is None:
                    continue

                st.markdown(f"**{model_name}**")

                # Pequenos graficos por SKU
                cols = st.columns(min(3, len(sku_forecasts)))
                for i, (sku_id, fc) in enumerate(sku_forecasts.items()):
                    with cols[i % len(cols)]:
                        sku_name = df[df["sku_id"] == sku_id]["sku_name"].iloc[0]
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=fc["ds"], y=fc["yhat"],
                            mode="lines", fill="tozeroy",
                            fillcolor=f"{COLORS['primary']}15",
                            line=dict(color=COLORS["primary"], width=2),
                        ))
                        fig.update_layout(
                            height=200, margin=dict(l=30, r=10, t=30, b=30),
                            title=dict(text=sku_name, font=dict(size=11)),
                            paper_bgcolor="white", plot_bgcolor="white",
                            xaxis=dict(showgrid=False),
                            yaxis=dict(showgrid=True, gridcolor="#f0f0f5"),
                        )
                        st.plotly_chart(fig, use_container_width=True)


def _render_weights(results, cluster_info):
    """Renderiza pesos de rateio."""
    chart_container("Pesos de Rateio Proporcional")

    for cluster_id, weights in results["weights"].items():
        with st.expander(f"Cluster {cluster_id}"):
            weights_df = pd.DataFrame([
                {"SKU": k, "Peso": round(v, 4), "Peso (%)": round(v * 100, 2)}
                for k, v in weights.items()
            ]).sort_values("Peso", ascending=False)

            col1, col2 = st.columns([1, 1])
            with col1:
                st.dataframe(weights_df, use_container_width=True, hide_index=True)

            with col2:
                fig = go.Figure(go.Pie(
                    labels=weights_df["SKU"],
                    values=weights_df["Peso"],
                    hole=0.4,
                    textinfo="label+percent",
                ))
                fig.update_layout(height=300, paper_bgcolor="white", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
