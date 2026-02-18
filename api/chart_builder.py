"""All Plotly chart construction logic extracted from app.py."""

import json

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import plotly.express as px

from config import MODEL_COLORS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fig_to_json(fig: go.Figure) -> str:
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def _hex_to_rgba(hex_color: str, alpha: float = 0.1) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _dates_to_str(series: pd.Series) -> list:
    try:
        return series.dt.strftime("%Y-%m-%d").tolist()
    except Exception:
        return series.astype(str).tolist()


def base_layout(title: str = "", height: int = 440) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=15, color="#2b2d42", family="Inter")),
        font=dict(family="Inter, system-ui, sans-serif", size=12, color="#2b2d42"),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        height=height,
        margin=dict(l=48, r=24, t=48, b=48),
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Inter", bordercolor="#e2e8f0"),
    )


# ---------------------------------------------------------------------------
# Overview charts
# ---------------------------------------------------------------------------
def build_overview_charts(df: pd.DataFrame) -> dict:
    charts = {}

    # 1 - Heatmap SKU x Mes
    dfm = df.copy()
    dfm["ym"] = dfm["date"].dt.to_period("M").astype(str)
    pivot = dfm.pivot_table(index="sku_name", columns="ym", values="demand", aggfunc="sum").fillna(0)
    fig = go.Figure(go.Heatmap(
        z=pivot.values.tolist(), x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale=[[0, "#f8fafc"], [.25, "#c7d2fe"], [.5, "#818cf8"], [.75, "#4f46e5"], [1, "#1e1b4b"]],
        colorbar=dict(title="Demanda"),
    ))
    fig.update_layout(**base_layout("", 480))
    fig.update_xaxes(tickangle=-45)
    charts["heatmap"] = fig_to_json(fig)

    # 2 - Top 10 SKUs
    top = df.groupby(["sku_id", "sku_name"])["demand"].sum().reset_index().sort_values("demand", ascending=True).tail(10)
    fig = go.Figure(go.Bar(
        x=top["demand"].tolist(), y=top["sku_name"].tolist(), orientation="h",
        marker=dict(color=top["demand"].tolist(), colorscale=[[0, "#818cf8"], [1, "#4361ee"]])))
    fig.update_layout(**base_layout("", 420))
    fig.update_xaxes(title="Demanda Total")
    fig.update_yaxes(title="")
    charts["top_skus"] = fig_to_json(fig)

    # 3 - Demanda total agregada
    daily = df.groupby("date")["demand"].sum().reset_index()
    daily["ma28"] = daily["demand"].rolling(28).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=_dates_to_str(daily["date"]), y=daily["demand"].tolist(),
        mode="lines", fill="tozeroy", fillcolor="rgba(67,97,238,.08)",
        line=dict(color="#4361ee", width=1.5), name="Demanda Total"))
    fig.add_trace(go.Scatter(
        x=_dates_to_str(daily["date"]), y=daily["ma28"].tolist(),
        mode="lines", line=dict(color="#ef476f", width=2.5, dash="dash"), name="MM 28d"))
    fig.update_layout(**base_layout("", 380))
    charts["total_demand"] = fig_to_json(fig)

    # 4 - Perfis donut
    pc = df.groupby("demand_profile")["sku_id"].nunique().reset_index()
    pc.columns = ["profile", "count"]
    colors_pie = ["#4361ee", "#06d6a0", "#ffd166", "#ef476f", "#118ab2"]
    fig = go.Figure(go.Pie(
        labels=pc["profile"].tolist(), values=pc["count"].tolist(),
        marker=dict(colors=colors_pie[:len(pc)]), hole=.45,
        textinfo="label+percent", textfont=dict(size=12)))
    fig.update_layout(height=380, paper_bgcolor="white", showlegend=False, font=dict(family="Inter"))
    charts["profiles"] = fig_to_json(fig)

    # 5 - Tabela estatisticas
    stats = df.groupby("demand_profile").agg(
        skus=("sku_id", "nunique"), media=("demand", "mean"),
        desvio=("demand", "std"), maximo=("demand", "max"),
        zero_pct=("demand", lambda x: round(float((x == 0).mean() * 100), 1)),
    ).reset_index()
    stats["media"] = stats["media"].round(1).astype(float)
    stats["desvio"] = stats["desvio"].round(1).astype(float)
    stats["maximo"] = stats["maximo"].astype(int)
    stats["skus"] = stats["skus"].astype(int)
    charts["stats_table"] = json.loads(stats.to_json(orient="records"))

    return charts


# ---------------------------------------------------------------------------
# Explorer charts
# ---------------------------------------------------------------------------
def build_demand_chart(df: pd.DataFrame, skus: list[str]) -> str:
    filtered = df[df["sku_name"].isin(skus)].sort_values("date")
    colors = list(MODEL_COLORS.values())
    fig = go.Figure()
    for i, sku in enumerate(skus):
        s = filtered[filtered["sku_name"] == sku]
        if s.empty:
            continue
        fig.add_trace(go.Scatter(
            x=_dates_to_str(s["date"]), y=s["demand"].tolist(),
            mode="lines", name=sku,
            line=dict(color=colors[i % len(colors)], width=1.5)))
    fig.update_layout(**base_layout("", 460))
    return fig_to_json(fig)


def build_climate_charts(df: pd.DataFrame) -> dict:
    c = df.drop_duplicates("date").sort_values("date")
    dates = _dates_to_str(c["date"])

    fig_temp = go.Figure(go.Scatter(
        x=dates, y=c["temperature"].tolist(), mode="lines",
        fill="tozeroy", fillcolor="rgba(239,71,111,.08)",
        line=dict(color="#ef476f", width=1.5)))
    fig_temp.update_layout(**base_layout("Temperatura (C)", 320))

    fig_rain = go.Figure(go.Bar(
        x=dates, y=c["rainfall"].tolist(),
        marker=dict(color="#118ab2", opacity=.7)))
    fig_rain.update_layout(**base_layout("Precipitacao (mm)", 320))

    fig_hum = go.Figure(go.Scatter(
        x=dates, y=c["humidity"].tolist(), mode="lines",
        fill="tozeroy", fillcolor="rgba(6,214,160,.08)",
        line=dict(color="#06d6a0", width=1.5)))
    fig_hum.update_layout(**base_layout("Umidade (%)", 320))

    return {
        "temperature": fig_to_json(fig_temp),
        "rainfall": fig_to_json(fig_rain),
        "humidity": fig_to_json(fig_hum),
    }


def build_distribution_charts(df: pd.DataFrame) -> dict:
    fig_season = px.box(df, x="season", y="demand", color="season",
                         color_discrete_map={"verao": "#ef476f", "outono": "#ffd166",
                                             "inverno": "#118ab2", "primavera": "#06d6a0"},
                         category_orders={"season": ["verao", "outono", "inverno", "primavera"]})
    fig_season.update_layout(**base_layout("Demanda por Estacao", 380), showlegend=False,
                              xaxis_title="Estacao", yaxis_title="Demanda")

    dow = {0: "Seg", 1: "Ter", 2: "Qua", 3: "Qui", 4: "Sex", 5: "Sab", 6: "Dom"}
    dfd = df.copy()
    dfd["dow"] = dfd["day_of_week"].map(dow)
    fig_dow = px.box(dfd, x="dow", y="demand", category_orders={"dow": list(dow.values())})
    fig_dow.update_traces(marker_color="#4361ee")
    fig_dow.update_layout(**base_layout("Demanda por Dia da Semana", 380),
                           xaxis_title="Dia", yaxis_title="Demanda")

    fig_safra = px.box(df, x="safra_soja", y="demand", color="safra_soja",
                        category_orders={"safra_soja": ["plantio", "crescimento", "colheita", "entressafra"]})
    fig_safra.update_layout(**base_layout("Demanda por Fase Safra (Soja)", 380),
                             showlegend=False, xaxis_title="Fase", yaxis_title="Demanda")

    fig_profile = px.violin(df, x="demand_profile", y="demand", color="demand_profile", box=True)
    fig_profile.update_layout(**base_layout("Demanda por Perfil", 380),
                               showlegend=False, xaxis_title="Perfil", yaxis_title="Demanda")

    return {
        "season": fig_to_json(fig_season),
        "dow": fig_to_json(fig_dow),
        "safra": fig_to_json(fig_safra),
        "profile": fig_to_json(fig_profile),
    }


# ---------------------------------------------------------------------------
# Similarity charts
# ---------------------------------------------------------------------------
def build_similarity_charts(
    dist_matrix: np.ndarray,
    sku_ids: list,
    labels: np.ndarray,
    n_clusters: int,
    sil_scores: dict,
    mds: np.ndarray,
    Z: np.ndarray,
    df: pd.DataFrame,
) -> dict:
    charts = {}

    # Heatmap de distancia (reordenado por cluster)
    order = np.argsort(labels)
    od = dist_matrix[order][:, order]
    on = [sku_ids[i] for i in order]
    fig = go.Figure(go.Heatmap(
        z=od.tolist(), x=on, y=on,
        colorscale=[[0, "#06d6a0"], [.3, "#ffd166"], [.6, "#ef476f"], [1, "#1a1a2e"]],
        colorbar=dict(title="Dist")))
    fig.update_layout(**base_layout("", 460))
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=9))
    fig.update_yaxes(tickfont=dict(size=9))
    charts["dist_heatmap"] = fig_to_json(fig)

    # Silhouette bar
    ks = sorted(sil_scores.keys())
    sc = [float(sil_scores[k]) for k in ks]
    fig = go.Figure(go.Bar(
        x=[str(k) for k in ks], y=sc,
        marker=dict(color=["#4361ee" if k != n_clusters else "#06d6a0" for k in ks]),
        text=[f"{s:.3f}" for s in sc], textposition="outside"))
    fig.update_layout(**base_layout("", 420), xaxis_title="K", yaxis_title="Silhouette")
    charts["silhouette"] = fig_to_json(fig)

    # MDS 2D scatter
    cl_colors = px.colors.qualitative.Set2[:n_clusters]
    labels_min = int(np.min(labels)) if len(labels) else 0
    fig = go.Figure()
    for c_val in sorted(set(labels)):
        idxs = np.where(labels == c_val)[0]
        color_idx = int(c_val) - 1 if labels_min == 1 else int(c_val)
        fig.add_trace(go.Scatter(
            x=mds[idxs, 0].tolist(), y=mds[idxs, 1].tolist(), mode="markers+text",
            name=f"Cluster {c_val}",
            text=[sku_ids[i] for i in idxs], textposition="top center",
            textfont=dict(size=9),
            marker=dict(size=12, color=cl_colors[color_idx] if 0 <= color_idx < len(cl_colors) else "#4361ee")))
    fig.update_layout(**base_layout("", 460), xaxis_title="MDS 1", yaxis_title="MDS 2")
    charts["mds"] = fig_to_json(fig)

    # Dendrograma
    from scipy.cluster.hierarchy import dendrogram as scipy_dendro
    dd = scipy_dendro(Z, labels=sku_ids, no_plot=True)
    fig = go.Figure()
    for xs, ys in zip(dd["icoord"], dd["dcoord"]):
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                                  line=dict(color="#4361ee", width=1.5), showlegend=False))
    fig.update_layout(**base_layout("", 440))
    fig.update_xaxes(title="SKUs", ticktext=dd["ivl"],
                     tickvals=list(range(5, len(dd["ivl"]) * 10 + 5, 10)),
                     tickangle=-45, tickfont=dict(size=9))
    fig.update_yaxes(title="Distancia")
    charts["dendro"] = fig_to_json(fig)

    # Series por cluster
    cluster_series_charts = {}
    colors_cycle = px.colors.qualitative.Set2
    for c_val in sorted(set(labels)):
        c_ids = [sku_ids[i] for i in np.where(labels == c_val)[0]]
        fig = go.Figure()
        for i, sid in enumerate(c_ids):
            sd = df[df["sku_id"] == sid].sort_values("date")
            if sd.empty:
                continue
            fig.add_trace(go.Scatter(
                x=_dates_to_str(sd["date"]), y=sd["demand"].tolist(),
                mode="lines", name=sd["sku_name"].iloc[0],
                line=dict(color=colors_cycle[i % len(colors_cycle)], width=1.5)))
        fig.update_layout(**base_layout(f"Cluster {c_val}", 360))
        cluster_series_charts[str(int(c_val))] = fig_to_json(fig)

    charts["cluster_series"] = cluster_series_charts
    return charts


# ---------------------------------------------------------------------------
# Forecast charts
# ---------------------------------------------------------------------------
def build_forecast_overlay(df: pd.DataFrame, sku_id: str, results: dict, test_days: int):
    """Build overlay chart, individual charts, and metrics for a SKU."""
    sku_data = df[df["sku_id"] == sku_id].sort_values("date")
    n = len(sku_data)
    td = min(test_days, n - 30)
    if td < 1:
        td = max(1, n // 4)

    train = sku_data.iloc[:-td]
    test = sku_data.iloc[-td:]

    fig = go.Figure()
    if not train.empty:
        fig.add_trace(go.Scatter(
            x=_dates_to_str(train["date"]), y=train["demand"].tolist(),
            mode="lines", name="Historico", line=dict(color="#2b2d42", width=1.5)))
    if not test.empty:
        fig.add_trace(go.Scatter(
            x=_dates_to_str(test["date"]), y=test["demand"].tolist(),
            mode="lines", name="Real (Teste)", line=dict(color="#1a1a2e", width=2, dash="dot")))

    metrics_rows = []
    detail_charts = {}

    for mname, res in results.items():
        if res.get("forecast") is not None:
            fc = res["forecast"]
            color = MODEL_COLORS.get(mname, "#4361ee")
            ds_str = _dates_to_str(fc["ds"])
            yhat = fc["yhat"].tolist()

            fig.add_trace(go.Scatter(x=ds_str, y=yhat, mode="lines", name=mname,
                                      line=dict(color=color, width=2.5)))

            if "yhat_upper" in fc.columns and "yhat_lower" in fc.columns:
                fig.add_trace(go.Scatter(
                    x=ds_str + ds_str[::-1],
                    y=fc["yhat_upper"].tolist() + fc["yhat_lower"].tolist()[::-1],
                    fill="toself", fillcolor=_hex_to_rgba(color, 0.07),
                    line=dict(color="rgba(0,0,0,0)"), showlegend=False))

            # Chart individual
            figi = go.Figure()
            if not train.empty:
                figi.add_trace(go.Scatter(
                    x=_dates_to_str(train["date"]), y=train["demand"].tolist(),
                    mode="lines", name="Historico", line=dict(color="#2b2d42", width=1.5)))
            if not test.empty:
                figi.add_trace(go.Scatter(
                    x=_dates_to_str(test["date"]), y=test["demand"].tolist(),
                    mode="lines", name="Real", line=dict(color="#1a1a2e", width=2, dash="dot")))
            figi.add_trace(go.Scatter(x=ds_str, y=yhat, mode="lines", name=mname,
                                       line=dict(color=color, width=2.5)))
            if "yhat_upper" in fc.columns and "yhat_lower" in fc.columns:
                figi.add_trace(go.Scatter(
                    x=ds_str + ds_str[::-1],
                    y=fc["yhat_upper"].tolist() + fc["yhat_lower"].tolist()[::-1],
                    fill="toself", fillcolor=_hex_to_rgba(color, 0.1),
                    line=dict(color="rgba(0,0,0,0)"), showlegend=False))
            figi.update_layout(**base_layout(mname, 360))
            detail_charts[mname] = fig_to_json(figi)

        if res.get("metrics"):
            metrics_rows.append({"Model": mname, **res["metrics"]})

    fig.update_layout(**base_layout("Comparacao de Forecasts", 500))

    # WAPE bar
    if metrics_rows:
        mdf = pd.DataFrame(metrics_rows)
        fig_wape = go.Figure(go.Bar(
            x=mdf["Model"].tolist(), y=mdf["WAPE"].tolist(),
            marker=dict(color=[MODEL_COLORS.get(m, "#4361ee") for m in mdf["Model"]]),
            text=[round(float(w), 1) for w in mdf["WAPE"]], textposition="outside"))
        fig_wape.update_layout(**base_layout("WAPE por Modelo (%)", 360))
    else:
        fig_wape = go.Figure()
        fig_wape.update_layout(**base_layout("Sem resultados", 360))

    return fig, fig_wape, detail_charts, metrics_rows


# ---------------------------------------------------------------------------
# Aggregated forecast charts
# ---------------------------------------------------------------------------
def build_aggregated_charts(
    df: pd.DataFrame,
    cluster_data: dict,
    cluster_forecasts: dict,
    test_days: int,
) -> dict:
    charts = {}
    for cid in sorted(cluster_forecasts.keys()):
        agg = cluster_data[cid].sort_values("date")
        n = len(agg)
        td = min(test_days, n - 30)
        if td < 1:
            td = max(1, n // 4)

        tr = agg.iloc[:-td]
        te = agg.iloc[-td:]
        fig = go.Figure()
        if not tr.empty:
            fig.add_trace(go.Scatter(
                x=_dates_to_str(tr["date"]), y=tr["demand_agg"].tolist(),
                mode="lines", name="Historico", line=dict(color="#2b2d42", width=1.5)))
        if not te.empty:
            fig.add_trace(go.Scatter(
                x=_dates_to_str(te["date"]), y=te["demand_agg"].tolist(),
                mode="lines", name="Real", line=dict(color="#1a1a2e", width=2, dash="dot")))
        for mname, fc in cluster_forecasts[cid].items():
            if fc is not None:
                color = MODEL_COLORS.get(mname, "#4361ee")
                fig.add_trace(go.Scatter(
                    x=_dates_to_str(fc["ds"]), y=fc["yhat"].tolist(),
                    mode="lines", name=mname, line=dict(color=color, width=2.5)))
        fig.update_layout(**base_layout(f"Cluster {cid}", 380))
        charts[str(int(cid))] = fig_to_json(fig)
    return charts


# ---------------------------------------------------------------------------
# Comparison charts
# ---------------------------------------------------------------------------
def build_comparison_charts(mdf: pd.DataFrame) -> dict:
    charts = {}

    # Radar
    avg = mdf.groupby("Model")[["MAE", "RMSE", "MAPE", "WAPE"]].mean().round(2).reset_index()
    fig = go.Figure()
    cols = ["MAE", "RMSE", "MAPE", "WAPE"]
    for _, row in avg.iterrows():
        vals = [float(row[c]) for c in cols] + [float(row[cols[0]])]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cols + [cols[0]], name=str(row["Model"]),
            fill="toself", fillcolor=_hex_to_rgba(MODEL_COLORS.get(row['Model'], '#4361ee'), 0.1),
            line=dict(color=MODEL_COLORS.get(row["Model"], "#4361ee"))))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, gridcolor="#f1f5f9"),
                    angularaxis=dict(gridcolor="#f1f5f9"), bgcolor="white"),
        font=dict(family="Inter", size=12), height=440, paper_bgcolor="white")
    charts["radar"] = fig_to_json(fig)

    # Box WAPE
    fig = px.box(mdf, x="Model", y="WAPE", color="Model", color_discrete_map=MODEL_COLORS, points="all")
    fig.update_layout(**base_layout("Distribuicao WAPE", 400), showlegend=False)
    charts["box_wape"] = fig_to_json(fig)

    # Grouped bar
    fig = px.bar(mdf, x="Nome", y="WAPE", color="Model", barmode="group", color_discrete_map=MODEL_COLORS)
    fig.update_layout(**base_layout("WAPE por SKU", 420))
    fig.update_xaxes(tickangle=-45)
    charts["grouped_bar"] = fig_to_json(fig)

    # Best model per SKU
    mdf_wape = mdf[mdf["WAPE"].notna()].copy()
    if not mdf_wape.empty:
        best_per = mdf_wape.loc[mdf_wape.groupby("SKU")["WAPE"].idxmin()]
        bcounts = best_per["Model"].value_counts().reset_index()
        bcounts.columns = ["Model", "count"]
        fig = go.Figure(go.Bar(
            x=bcounts["Model"].tolist(), y=bcounts["count"].tolist(),
            marker=dict(color=[MODEL_COLORS.get(m, "#4361ee") for m in bcounts["Model"]]),
            text=bcounts["count"].tolist(), textposition="outside"))
        fig.update_layout(**base_layout("Vezes Melhor Modelo", 340))
        charts["best_count"] = fig_to_json(fig)
    else:
        fig = go.Figure()
        fig.update_layout(**base_layout("Sem dados", 340))
        charts["best_count"] = fig_to_json(fig)

    return charts, avg


def build_comparison_sku_chart(
    df: pd.DataFrame,
    sid: str,
    results: dict,
    test_days: int,
) -> str:
    sku_data = df[df["sku_id"] == sid].sort_values("date")
    n = len(sku_data)
    td = min(test_days, n - 30)
    if td < 1:
        td = max(1, n // 4)

    tr = sku_data.iloc[:-td]
    te = sku_data.iloc[-td:]
    sname = str(sku_data["sku_name"].iloc[0])

    fig = go.Figure()
    if not tr.empty:
        fig.add_trace(go.Scatter(
            x=_dates_to_str(tr["date"]), y=tr["demand"].tolist(),
            mode="lines", name="Historico", line=dict(color="#2b2d42", width=1.5)))
    if not te.empty:
        fig.add_trace(go.Scatter(
            x=_dates_to_str(te["date"]), y=te["demand"].tolist(),
            mode="lines", name="Real", line=dict(color="#1a1a2e", width=2, dash="dot")))

    for mname, r in results.items():
        if r.get("forecast") is not None:
            fc = r["forecast"]
            color = MODEL_COLORS.get(mname, "#4361ee")
            fig.add_trace(go.Scatter(
                x=_dates_to_str(fc["ds"]), y=fc["yhat"].tolist(),
                mode="lines", name=mname, line=dict(color=color, width=2)))
    fig.update_layout(**base_layout(sname, 340))
    return fig_to_json(fig)
