"""Pagina 1: Dashboard Overview com KPIs."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from dashboard.components import (
    kpi_card, chart_container, section_divider,
    create_plotly_layout,
)
from config import COLORS


def render(df: pd.DataFrame):
    """Renderiza pagina de overview."""
    st.markdown("# Dashboard Overview")
    st.markdown("Visao geral do sistema de forecast multi-produto SKU")
    section_divider()

    # KPIs Row
    col1, col2, col3, col4 = st.columns(4)

    total_skus = df["sku_id"].nunique()
    total_records = len(df)
    date_range_days = (df["date"].max() - df["date"].min()).days
    avg_demand = df["demand"].mean()

    with col1:
        kpi_card("Total SKUs", str(total_skus), border_color=COLORS["primary"])
    with col2:
        kpi_card("Registros", f"{total_records:,}", border_color=COLORS["success"])
    with col3:
        kpi_card("Cobertura", f"{date_range_days} dias", border_color=COLORS["info"])
    with col4:
        kpi_card("Demanda Media", f"{avg_demand:.0f} un/dia", border_color=COLORS["warning"])

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2: Heatmap + Top SKUs
    col_left, col_right = st.columns([3, 2])

    with col_left:
        chart_container("Heatmap de Demanda - SKU x Mes")

        # Preparar dados heatmap
        df_month = df.copy()
        df_month["year_month"] = df_month["date"].dt.to_period("M").astype(str)
        pivot = df_month.pivot_table(
            index="sku_name", columns="year_month", values="demand", aggfunc="sum"
        ).fillna(0)

        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale=[
                [0, "#f8f9fc"],
                [0.25, "#c7d2fe"],
                [0.5, "#818cf8"],
                [0.75, "#4f46e5"],
                [1, "#1e1b4b"],
            ],
            hoverongaps=False,
            colorbar=dict(title="Demanda"),
        ))
        fig.update_layout(
            **create_plotly_layout("", 500),
            xaxis=dict(title="Mes", tickangle=-45),
            yaxis=dict(title=""),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        chart_container("Top 10 SKUs por Volume")

        top_skus = (
            df.groupby(["sku_id", "sku_name"])["demand"]
            .sum()
            .reset_index()
            .sort_values("demand", ascending=True)
            .tail(10)
        )

        fig = go.Figure(go.Bar(
            x=top_skus["demand"],
            y=top_skus["sku_name"],
            orientation="h",
            marker=dict(
                color=top_skus["demand"],
                colorscale=[[0, "#818cf8"], [1, "#4361ee"]],
            ),
        ))
        fig.update_layout(**create_plotly_layout("", 500))
        fig.update_layout(yaxis=dict(title=""), xaxis=dict(title="Demanda Total"))
        st.plotly_chart(fig, use_container_width=True)

    section_divider()

    # Row 3: Demanda Agregada + Perfis
    col_left, col_right = st.columns([3, 2])

    with col_left:
        chart_container("Demanda Total Agregada ao Longo do Tempo")
        daily_total = df.groupby("date")["demand"].sum().reset_index()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_total["date"],
            y=daily_total["demand"],
            mode="lines",
            fill="tozeroy",
            fillcolor=f"{COLORS['primary']}15",
            line=dict(color=COLORS["primary"], width=2),
            name="Demanda Total",
        ))

        # Media movel 28 dias
        daily_total["ma28"] = daily_total["demand"].rolling(28).mean()
        fig.add_trace(go.Scatter(
            x=daily_total["date"],
            y=daily_total["ma28"],
            mode="lines",
            line=dict(color=COLORS["danger"], width=2.5, dash="dash"),
            name="MM 28 dias",
        ))

        fig.update_layout(**create_plotly_layout("", 400))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        chart_container("Distribuicao por Perfil de Demanda")
        profile_counts = df.groupby("demand_profile")["sku_id"].nunique().reset_index()
        profile_counts.columns = ["Perfil", "SKUs"]

        colors_pie = [COLORS["primary"], COLORS["success"], COLORS["warning"],
                      COLORS["danger"], COLORS["info"]]

        fig = go.Figure(go.Pie(
            labels=profile_counts["Perfil"],
            values=profile_counts["SKUs"],
            marker=dict(colors=colors_pie[:len(profile_counts)]),
            hole=0.4,
            textinfo="label+percent",
            textfont=dict(size=12, family="Inter"),
        ))
        fig.update_layout(
            height=400, paper_bgcolor="white",
            font=dict(family="Inter"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Row 4: Estatisticas por perfil
    section_divider()
    chart_container("Estatisticas por Perfil de Demanda")

    stats = df.groupby("demand_profile").agg(
        skus=("sku_id", "nunique"),
        mean_demand=("demand", "mean"),
        std_demand=("demand", "std"),
        max_demand=("demand", "max"),
        zero_pct=("demand", lambda x: f"{(x == 0).mean() * 100:.1f}%"),
    ).reset_index()

    stats.columns = ["Perfil", "SKUs", "Media", "Desvio", "Maximo", "% Zeros"]
    stats["Media"] = stats["Media"].round(1)
    stats["Desvio"] = stats["Desvio"].round(1)

    st.dataframe(stats, use_container_width=True, hide_index=True)
