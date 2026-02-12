"""Componentes reutilizaveis do dashboard."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from config import COLORS, MODEL_COLORS


def kpi_card(label: str, value: str, delta: str = None, delta_type: str = "neutral", border_color: str = None):
    """Renderiza card de KPI."""
    color = border_color or COLORS["primary"]
    delta_class = f"kpi-delta-{delta_type}"
    delta_html = f'<div class="kpi-delta {delta_class}">{delta}</div>' if delta else ""

    st.markdown(f"""
    <div class="kpi-card" style="border-left: 4px solid {color};">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def chart_container(title: str):
    """Renderiza container estilizado para graficos."""
    st.markdown(f"""
    <div class="chart-title">{title}</div>
    """, unsafe_allow_html=True)


def section_divider():
    """Renderiza divisor de secao."""
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)


def sidebar_logo():
    """Renderiza logo no sidebar."""
    st.markdown("""
    <div class="logo-container">
        <div class="logo-title">Forecast Pro</div>
        <div class="logo-subtitle">Sistema Multi-Produto</div>
    </div>
    """, unsafe_allow_html=True)


def status_badge(text: str, badge_type: str = "info"):
    """Renderiza badge de status."""
    return f'<span class="status-badge badge-{badge_type}">{text}</span>'


def create_plotly_layout(title: str = "", height: int = 450) -> dict:
    """Layout padrao para graficos Plotly."""
    return dict(
        title=dict(text=title, font=dict(size=16, color=COLORS["text"], family="Inter")),
        font=dict(family="Inter", size=12, color=COLORS["text"]),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=height,
        margin=dict(l=50, r=30, t=50, b=50),
        xaxis=dict(
            showgrid=True, gridcolor="#f0f0f5", gridwidth=1,
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=True, gridcolor="#f0f0f5", gridwidth=1,
            zeroline=False,
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(size=11),
        ),
        hoverlabel=dict(
            bgcolor="white", font_size=12, font_family="Inter",
            bordercolor="#e0e0e0",
        ),
    )


def plot_time_series(df, date_col, value_cols, names=None, colors=None, title="", height=450):
    """Plota series temporais com Plotly."""
    fig = go.Figure()

    if isinstance(value_cols, str):
        value_cols = [value_cols]

    if names is None:
        names = value_cols

    if colors is None:
        default_colors = list(MODEL_COLORS.values())
        colors = default_colors[:len(value_cols)]

    for i, (col, name) in enumerate(zip(value_cols, names)):
        fig.add_trace(go.Scatter(
            x=df[date_col], y=df[col],
            mode="lines",
            name=name,
            line=dict(color=colors[i % len(colors)], width=2),
        ))

    fig.update_layout(**create_plotly_layout(title, height))
    return fig


def plot_forecast_chart(
    train_dates, train_values, forecast_df,
    model_name="Modelo", actual_test=None, test_dates=None,
    height=500,
):
    """Plota grafico de forecast com historico e intervalos de confianca."""
    fig = go.Figure()

    # Historico
    fig.add_trace(go.Scatter(
        x=train_dates, y=train_values,
        mode="lines", name="Historico",
        line=dict(color=COLORS["text"], width=2),
    ))

    # Actual test (se disponivel)
    if actual_test is not None and test_dates is not None:
        fig.add_trace(go.Scatter(
            x=test_dates, y=actual_test,
            mode="lines", name="Real",
            line=dict(color=COLORS["dark"], width=2, dash="dot"),
        ))

    # Forecast
    color = MODEL_COLORS.get(model_name, COLORS["primary"])
    fig.add_trace(go.Scatter(
        x=forecast_df["ds"], y=forecast_df["yhat"],
        mode="lines", name=f"Forecast ({model_name})",
        line=dict(color=color, width=2.5),
    ))

    # Intervalo de confianca
    if "yhat_lower" in forecast_df.columns and "yhat_upper" in forecast_df.columns:
        fig.add_trace(go.Scatter(
            x=list(forecast_df["ds"]) + list(forecast_df["ds"][::-1]),
            y=list(forecast_df["yhat_upper"]) + list(forecast_df["yhat_lower"][::-1]),
            fill="toself",
            fillcolor=f"{color}15",
            line=dict(color="rgba(0,0,0,0)"),
            name="IC 90%",
            showlegend=True,
        ))

    fig.update_layout(**create_plotly_layout(f"Forecast - {model_name}", height))
    return fig


def plot_metrics_comparison(metrics_df, metric_col="WAPE", title=""):
    """Plota comparacao de metricas entre modelos."""
    fig = px.bar(
        metrics_df.sort_values(metric_col),
        x="Model", y=metric_col,
        color="Model",
        color_discrete_map=MODEL_COLORS,
        title=title or f"Comparacao - {metric_col}",
    )
    fig.update_layout(**create_plotly_layout(title or f"Comparacao - {metric_col}", 400))
    fig.update_layout(showlegend=False)
    return fig


def plot_radar_chart(metrics_df, metrics_cols, title="Radar de Metricas"):
    """Plota radar chart para comparacao multi-metrica."""
    fig = go.Figure()

    for _, row in metrics_df.iterrows():
        model = row["Model"]
        values = []
        for col in metrics_cols:
            val = row[col]
            if val is not None and not (isinstance(val, float) and val != val):
                values.append(val)
            else:
                values.append(0)
        values.append(values[0])  # fechar radar

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics_cols + [metrics_cols[0]],
            name=model,
            line=dict(color=MODEL_COLORS.get(model, COLORS["primary"])),
            fill="toself",
            fillcolor=f"{MODEL_COLORS.get(model, COLORS['primary'])}15",
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, showticklabels=True, gridcolor="#f0f0f5"),
            angularaxis=dict(gridcolor="#f0f0f5"),
            bgcolor="white",
        ),
        font=dict(family="Inter", size=12),
        title=dict(text=title, font=dict(size=16)),
        height=450,
        paper_bgcolor="white",
    )
    return fig
