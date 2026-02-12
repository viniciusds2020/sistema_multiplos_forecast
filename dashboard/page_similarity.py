"""Pagina 3: Analise de Similaridade e Clustering."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from dashboard.components import (
    kpi_card, chart_container, section_divider, create_plotly_layout,
)
from config import COLORS
from similarity.clustering import (
    build_demand_matrix, compute_distance_matrix,
    find_optimal_clusters, cluster_series, compute_mds_projection,
    get_cluster_summary,
)


def render(df: pd.DataFrame):
    """Renderiza pagina de similaridade."""
    st.markdown("# Analise de Similaridade")
    st.markdown("Identifique padroes semelhantes entre SKUs e agrupe-os em clusters")
    section_divider()

    # Controles
    with st.sidebar:
        st.markdown("### Configuracoes de Clustering")
        metric = st.selectbox(
            "Metrica de Distancia",
            ["pearson", "euclidean", "dtw"],
            index=0,
        )
        auto_k = st.checkbox("K automatico (silhouette)", value=True)
        manual_k = st.slider("Numero de clusters (k)", 2, 8, 4, disabled=auto_k)

    # Executar clustering
    with st.spinner("Calculando distancias e clusters..."):
        matrix, sku_ids = build_demand_matrix(df)
        dist_matrix = compute_distance_matrix(matrix, metric=metric)

        if auto_k:
            n_clusters, silhouette_scores = find_optimal_clusters(matrix, dist_matrix)
        else:
            n_clusters = manual_k
            _, silhouette_scores = find_optimal_clusters(matrix, dist_matrix)

        labels, Z = cluster_series(dist_matrix, n_clusters)
        mds_coords = compute_mds_projection(dist_matrix)
        summary = get_cluster_summary(df, sku_ids, labels)

    # Salvar no session state
    st.session_state["cluster_info"] = {
        "matrix": matrix,
        "sku_ids": sku_ids,
        "dist_matrix": dist_matrix,
        "labels": labels,
        "linkage": Z,
        "n_clusters": n_clusters,
        "silhouette_scores": silhouette_scores,
        "cluster_summary": summary,
    }

    # KPIs
    col1, col2, col3 = st.columns(3)
    with col1:
        kpi_card("Clusters", str(n_clusters), border_color=COLORS["primary"])
    with col2:
        best_score = max(silhouette_scores.values()) if silhouette_scores else 0
        kpi_card("Melhor Silhouette", f"{best_score:.3f}", border_color=COLORS["success"])
    with col3:
        kpi_card("Total SKUs", str(len(sku_ids)), border_color=COLORS["info"])

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 1: Heatmap + Silhouette
    col1, col2 = st.columns(2)

    with col1:
        chart_container("Matriz de Distancia")

        # Reordenar por cluster
        order = np.argsort(labels)
        ordered_dist = dist_matrix[order][:, order]
        ordered_names = [sku_ids[i] for i in order]

        fig = go.Figure(data=go.Heatmap(
            z=ordered_dist,
            x=ordered_names,
            y=ordered_names,
            colorscale=[
                [0, "#06d6a0"],
                [0.3, "#ffd166"],
                [0.6, "#ef476f"],
                [1, "#1a1a2e"],
            ],
            colorbar=dict(title="Distancia"),
        ))
        fig.update_layout(**create_plotly_layout("", 500))
        fig.update_layout(
            xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
            yaxis=dict(tickfont=dict(size=9)),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        chart_container("Silhouette Score por K")

        ks = sorted(silhouette_scores.keys())
        scores = [silhouette_scores[k] for k in ks]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[str(k) for k in ks],
            y=scores,
            marker=dict(
                color=[COLORS["primary"] if k != n_clusters else COLORS["success"] for k in ks],
            ),
            text=[f"{s:.3f}" for s in scores],
            textposition="outside",
        ))
        fig.update_layout(**create_plotly_layout("", 500))
        fig.update_layout(
            xaxis=dict(title="Numero de Clusters (k)"),
            yaxis=dict(title="Silhouette Score"),
        )
        st.plotly_chart(fig, use_container_width=True)

    section_divider()

    # Row 2: MDS + Dendrograma
    col1, col2 = st.columns(2)

    with col1:
        chart_container("Projecao MDS 2D")

        cluster_colors = px.colors.qualitative.Set2[:n_clusters]
        color_map = {i+1: cluster_colors[i] for i in range(n_clusters)}

        fig = go.Figure()
        for c in sorted(set(labels)):
            mask = labels == c
            idxs = np.where(mask)[0]
            fig.add_trace(go.Scatter(
                x=mds_coords[idxs, 0],
                y=mds_coords[idxs, 1],
                mode="markers+text",
                name=f"Cluster {c}",
                text=[sku_ids[i] for i in idxs],
                textposition="top center",
                textfont=dict(size=9),
                marker=dict(size=12, color=color_map.get(c, COLORS["primary"])),
            ))

        fig.update_layout(**create_plotly_layout("", 500))
        fig.update_layout(xaxis=dict(title="MDS 1"), yaxis=dict(title="MDS 2"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        chart_container("Dendrograma Hierarquico")

        # Plotar dendrograma com plotly
        from scipy.cluster.hierarchy import dendrogram as scipy_dendro
        dendro = scipy_dendro(Z, labels=sku_ids, no_plot=True)

        fig = go.Figure()
        for i, (xs, ys) in enumerate(zip(dendro["icoord"], dendro["dcoord"])):
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(color=COLORS["primary"], width=1.5),
                showlegend=False,
            ))

        fig.update_layout(**create_plotly_layout("", 500))
        fig.update_layout(
            xaxis=dict(
                title="SKUs",
                ticktext=dendro["ivl"],
                tickvals=list(range(5, len(dendro["ivl"]) * 10 + 5, 10)),
                tickangle=-45, tickfont=dict(size=9),
            ),
            yaxis=dict(title="Distancia"),
        )
        st.plotly_chart(fig, use_container_width=True)

    section_divider()

    # Row 3: Detalhe por cluster
    chart_container("Detalhe dos Clusters")
    selected_cluster = st.selectbox(
        "Selecionar Cluster",
        sorted(set(labels)),
        format_func=lambda x: f"Cluster {x}",
    )

    cluster_mask = labels == selected_cluster
    cluster_sku_ids = [sku_ids[i] for i in np.where(cluster_mask)[0]]

    # Overlay de series do cluster
    fig = go.Figure()
    colors_cycle = px.colors.qualitative.Set2
    for i, sku_id in enumerate(cluster_sku_ids):
        sku_data = df[df["sku_id"] == sku_id].sort_values("date")
        fig.add_trace(go.Scatter(
            x=sku_data["date"],
            y=sku_data["demand"],
            mode="lines",
            name=sku_data["sku_name"].iloc[0],
            line=dict(color=colors_cycle[i % len(colors_cycle)], width=1.5),
        ))

    fig.update_layout(**create_plotly_layout(f"Series do Cluster {selected_cluster}", 400))
    st.plotly_chart(fig, use_container_width=True)

    # Tabela resumo do cluster
    cluster_summary = summary[summary["cluster"] == selected_cluster]
    display_cols = ["sku_id", "sku_name", "demand_profile", "mean_demand", "std_demand", "cv", "zero_pct"]
    available = [c for c in display_cols if c in cluster_summary.columns]
    st.dataframe(
        cluster_summary[available].round(2),
        use_container_width=True,
        hide_index=True,
    )
