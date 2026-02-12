"""Pagina 2: Data Explorer - explorar dados e features."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from dashboard.components import (
    chart_container, section_divider, create_plotly_layout,
)
from config import COLORS, MODEL_COLORS


def render(df: pd.DataFrame):
    """Renderiza pagina de exploracao de dados."""
    st.markdown("# Data Explorer")
    st.markdown("Explore as series temporais, dados climaticos e features")
    section_divider()

    # Sidebar controls
    with st.sidebar:
        st.markdown("### Filtros")
        selected_skus = st.multiselect(
            "Selecionar SKUs",
            options=sorted(df["sku_name"].unique()),
            default=sorted(df["sku_name"].unique())[:4],
        )

        date_range = st.date_input(
            "Periodo",
            value=(df["date"].min().date(), df["date"].max().date()),
            min_value=df["date"].min().date(),
            max_value=df["date"].max().date(),
        )

    # Filtrar dados
    if selected_skus:
        df_filtered = df[df["sku_name"].isin(selected_skus)].copy()
    else:
        df_filtered = df.copy()

    if len(date_range) == 2:
        df_filtered = df_filtered[
            (df_filtered["date"].dt.date >= date_range[0]) &
            (df_filtered["date"].dt.date <= date_range[1])
        ]

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Series de Demanda", "Dados Climaticos",
        "Distribuicoes", "Dados Brutos",
    ])

    with tab1:
        _render_demand_tab(df_filtered)

    with tab2:
        _render_climate_tab(df_filtered)

    with tab3:
        _render_distributions_tab(df_filtered)

    with tab4:
        _render_raw_data_tab(df_filtered)


def _render_demand_tab(df: pd.DataFrame):
    """Tab de series de demanda."""
    chart_container("Series de Demanda por SKU")

    fig = go.Figure()
    colors = list(MODEL_COLORS.values())

    for i, sku in enumerate(df["sku_name"].unique()):
        sku_data = df[df["sku_name"] == sku].sort_values("date")
        fig.add_trace(go.Scatter(
            x=sku_data["date"],
            y=sku_data["demand"],
            mode="lines",
            name=sku,
            line=dict(color=colors[i % len(colors)], width=1.5),
            opacity=0.85,
        ))

    fig.update_layout(**create_plotly_layout("", 500))
    st.plotly_chart(fig, use_container_width=True)

    # Media movel por SKU
    st.markdown("<br>", unsafe_allow_html=True)
    chart_container("Media Movel 28 dias")

    fig2 = go.Figure()
    for i, sku in enumerate(df["sku_name"].unique()):
        sku_data = df[df["sku_name"] == sku].sort_values("date")
        ma = sku_data["demand"].rolling(28).mean()
        fig2.add_trace(go.Scatter(
            x=sku_data["date"],
            y=ma,
            mode="lines",
            name=sku,
            line=dict(color=colors[i % len(colors)], width=2),
        ))

    fig2.update_layout(**create_plotly_layout("", 400))
    st.plotly_chart(fig2, use_container_width=True)


def _render_climate_tab(df: pd.DataFrame):
    """Tab de dados climaticos."""
    # Pegar uma serie unica de clima (sao compartilhados)
    climate_df = df.drop_duplicates(subset="date").sort_values("date")

    col1, col2 = st.columns(2)

    with col1:
        chart_container("Temperatura (C)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=climate_df["date"], y=climate_df["temperature"],
            mode="lines", fill="tozeroy",
            fillcolor=f"{COLORS['danger']}10",
            line=dict(color=COLORS["danger"], width=1.5),
        ))
        fig.update_layout(**create_plotly_layout("", 350))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        chart_container("Precipitacao (mm)")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=climate_df["date"], y=climate_df["rainfall"],
            marker=dict(color=COLORS["info"], opacity=0.7),
        ))
        fig.update_layout(**create_plotly_layout("", 350))
        st.plotly_chart(fig, use_container_width=True)

    chart_container("Umidade Relativa (%)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=climate_df["date"], y=climate_df["humidity"],
        mode="lines", fill="tozeroy",
        fillcolor=f"{COLORS['success']}10",
        line=dict(color=COLORS["success"], width=1.5),
    ))
    fig.update_layout(**create_plotly_layout("", 350))
    st.plotly_chart(fig, use_container_width=True)


def _render_distributions_tab(df: pd.DataFrame):
    """Tab de distribuicoes."""
    col1, col2 = st.columns(2)

    with col1:
        chart_container("Demanda por Estacao")
        fig = px.box(
            df, x="season", y="demand", color="season",
            color_discrete_map={
                "verao": COLORS["danger"],
                "outono": COLORS["warning"],
                "inverno": COLORS["info"],
                "primavera": COLORS["success"],
            },
            category_orders={"season": ["verao", "outono", "inverno", "primavera"]},
        )
        fig.update_layout(**create_plotly_layout("", 400))
        fig.update_layout(showlegend=False, xaxis_title="Estacao", yaxis_title="Demanda")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        chart_container("Demanda por Dia da Semana")
        dow_names = {0: "Seg", 1: "Ter", 2: "Qua", 3: "Qui", 4: "Sex", 5: "Sab", 6: "Dom"}
        df_dow = df.copy()
        df_dow["dow_name"] = df_dow["day_of_week"].map(dow_names)
        fig = px.box(df_dow, x="dow_name", y="demand",
                     category_orders={"dow_name": list(dow_names.values())})
        fig.update_traces(marker_color=COLORS["primary"])
        fig.update_layout(**create_plotly_layout("", 400))
        fig.update_layout(xaxis_title="Dia", yaxis_title="Demanda")
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        chart_container("Demanda por Fase Safra (Soja)")
        fig = px.box(
            df, x="safra_soja", y="demand", color="safra_soja",
            category_orders={"safra_soja": ["plantio", "crescimento", "colheita", "entressafra"]},
        )
        fig.update_layout(**create_plotly_layout("", 400))
        fig.update_layout(showlegend=False, xaxis_title="Fase", yaxis_title="Demanda")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        chart_container("Demanda por Perfil")
        fig = px.violin(
            df, x="demand_profile", y="demand", color="demand_profile",
            box=True,
        )
        fig.update_layout(**create_plotly_layout("", 400))
        fig.update_layout(showlegend=False, xaxis_title="Perfil", yaxis_title="Demanda")
        st.plotly_chart(fig, use_container_width=True)


def _render_raw_data_tab(df: pd.DataFrame):
    """Tab de dados brutos."""
    chart_container("Dados Brutos")
    st.markdown(f"**{len(df):,} registros** | {df['sku_id'].nunique()} SKUs")

    st.dataframe(
        df.sort_values(["sku_id", "date"]).head(500),
        use_container_width=True,
        hide_index=True,
    )

    # Download
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        csv,
        "forecast_data.csv",
        "text/csv",
    )
