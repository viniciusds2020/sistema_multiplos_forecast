"""Dashboard Flask - Sistema de Forecast Multi-Produto SKU. ForecastForge"""

import sys
import json
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import plotly.express as px
from flask import Flask, render_template, jsonify, request

from config import COLORS, MODEL_COLORS
from data.synthetic_generator import generate_synthetic_data
from similarity.clustering import (
    build_demand_matrix, compute_distance_matrix,
    find_optimal_clusters, cluster_series, compute_mds_projection,
    get_cluster_summary,
)
from similarity.aggregation import (
    aggregate_cluster_demand, compute_disaggregation_weights,
    disaggregate_forecast,
)
from pipeline.forecasting_pipeline import run_forecast_pipeline
from pipeline.cluster_pipeline import run_cluster_analysis, run_cluster_forecast_pipeline

app = Flask(__name__)


# ---------------------------------------------------------------------------
# JSON Encoder customizado para tipos numpy/pandas
# ---------------------------------------------------------------------------
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, pd.Timestamp):
            return obj.strftime("%Y-%m-%d")
        return super().default(obj)


# Flask 3.x JSON provider customizado
from flask.json.provider import DefaultJSONProvider

class NumpyJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, pd.Timestamp):
            return obj.strftime("%Y-%m-%d")
        return super().default(obj)

app.json_provider_class = NumpyJSONProvider
app.json = NumpyJSONProvider(app)


def safe_jsonify(data):
    """jsonify seguro que converte tipos numpy automaticamente."""
    cleaned = json.loads(json.dumps(data, cls=NumpyEncoder))
    return jsonify(cleaned)


# ---------------------------------------------------------------------------
# Cache global de dados
# ---------------------------------------------------------------------------
_cache: dict = {}
_cache_lock = None


def get_data() -> pd.DataFrame:
    global _cache_lock
    if _cache_lock is None:
        import threading
        _cache_lock = threading.Lock()
    if "df" not in _cache:
        with _cache_lock:
            if "df" not in _cache:
                _cache["df"] = generate_synthetic_data()
    return _cache["df"]


# ---------------------------------------------------------------------------
# Helpers Plotly -> JSON
# ---------------------------------------------------------------------------
def fig_to_json(fig: go.Figure) -> str:
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def _hex_to_rgba(hex_color: str, alpha: float = 0.1) -> str:
    """Converte cor hex para rgba string."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _dates_to_str(series: pd.Series) -> list:
    """Converte datas para strings de forma segura."""
    try:
        return series.dt.strftime("%Y-%m-%d").tolist()
    except Exception:
        return series.astype(str).tolist()


def base_layout(title="", height=440):
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


# ===================================================================
# ROTAS DE PAGINA
# ===================================================================
@app.route("/")
def page_overview():
    return render_template("overview.html")


@app.route("/explorer")
def page_explorer():
    return render_template("explorer.html")


@app.route("/similarity")
def page_similarity():
    return render_template("similarity.html")


@app.route("/forecasting")
def page_forecasting():
    return render_template("forecasting.html")


@app.route("/comparison")
def page_comparison():
    return render_template("comparison.html")


# ===================================================================
# API - DADOS GERAIS
# ===================================================================
@app.route("/api/models")
def api_models():
    """Retorna lista de modelos disponiveis (que podem ser importados)."""
    from models.model_registry import list_models, list_available_models
    return safe_jsonify({
        "all": list_models(),
        "available": list_available_models(),
    })


@app.route("/api/kpis")
def api_kpis():
    df = get_data()
    return safe_jsonify({
        "total_skus": int(df["sku_id"].nunique()),
        "total_records": int(len(df)),
        "coverage_days": int((df["date"].max() - df["date"].min()).days),
        "avg_demand": round(float(df["demand"].mean()), 1),
        "date_start": str(df["date"].min().date()),
        "date_end": str(df["date"].max().date()),
    })


@app.route("/api/skus")
def api_skus():
    df = get_data()
    skus = (
        df.groupby(["sku_id", "sku_name", "demand_profile"])
        .agg(mean_demand=("demand", "mean"), total_demand=("demand", "sum"))
        .reset_index()
    )
    skus["mean_demand"] = skus["mean_demand"].round(1)
    skus["total_demand"] = skus["total_demand"].astype(int)
    return safe_jsonify(skus.to_dict(orient="records"))


@app.route("/api/overview/charts")
def api_overview_charts():
    try:
        df = get_data()
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

        return safe_jsonify(charts)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===================================================================
# API - DATA EXPLORER
# ===================================================================
@app.route("/api/explorer/demand")
def api_explorer_demand():
    df = get_data()
    skus = request.args.getlist("skus") or list(df["sku_name"].unique()[:4])
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
    return jsonify({"chart": fig_to_json(fig)})


@app.route("/api/explorer/climate")
def api_explorer_climate():
    df = get_data()
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

    return jsonify({
        "temperature": fig_to_json(fig_temp),
        "rainfall": fig_to_json(fig_rain),
        "humidity": fig_to_json(fig_hum),
    })


@app.route("/api/explorer/distributions")
def api_explorer_distributions():
    df = get_data()

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

    return jsonify({
        "season": fig_to_json(fig_season),
        "dow": fig_to_json(fig_dow),
        "safra": fig_to_json(fig_safra),
        "profile": fig_to_json(fig_profile),
    })


@app.route("/api/explorer/raw")
def api_explorer_raw():
    df = get_data()
    sample = df.sort_values(["sku_id", "date"]).head(300).copy()
    sample["date"] = sample["date"].dt.strftime("%Y-%m-%d")
    return safe_jsonify(sample.to_dict(orient="records"))


# ===================================================================
# API - SIMILARIDADE
# ===================================================================
@app.route("/api/similarity/run", methods=["POST"])
def api_similarity_run():
    try:
        df = get_data()
        body = request.get_json(silent=True) or {}
        metric = body.get("metric", "pearson")
        auto_k = body.get("auto_k", True)
        manual_k = body.get("manual_k", 4)

        matrix, sku_ids = build_demand_matrix(df)

        if matrix.shape[0] < 2:
            return jsonify({"error": "Sao necessarios pelo menos 2 SKUs para clustering"}), 400

        dist_matrix = compute_distance_matrix(matrix, metric=metric)

        if auto_k:
            n_clusters, sil_scores = find_optimal_clusters(matrix, dist_matrix)
        else:
            n_clusters = manual_k
            _, sil_scores = find_optimal_clusters(matrix, dist_matrix)

        labels, Z = cluster_series(dist_matrix, n_clusters)
        mds = compute_mds_projection(dist_matrix)
        summary = get_cluster_summary(df, sku_ids, labels)

        # Guardar no cache
        _cache["cluster_info"] = {
            "sku_ids": sku_ids,
            "labels": [int(x) for x in labels],
            "linkage": [[float(v) for v in row] for row in Z],
            "n_clusters": int(n_clusters),
        }

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

        # Tabela resumo
        display_cols = ["cluster", "sku_id", "sku_name", "demand_profile", "mean_demand", "cv", "zero_pct"]
        avail = [c for c in display_cols if c in summary.columns]
        display = summary[avail].copy()
        for col in ["mean_demand", "cv", "zero_pct"]:
            if col in display.columns:
                display[col] = display[col].round(3).astype(float)
        display["cluster"] = display["cluster"].astype(int)

        return safe_jsonify({
            "charts": charts,
            "n_clusters": int(n_clusters),
            "sil_best": round(float(max(sil_scores.values())), 3) if sil_scores else 0,
            "total_skus": len(sku_ids),
            "summary": json.loads(display.to_json(orient="records")),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ===================================================================
# Helpers para montar charts de forecast
# ===================================================================
def _build_forecast_overlay(df, sku_id, results, test_days):
    """Monta chart overlay, charts individuais, metricas para um SKU."""
    sku_data = df[df["sku_id"] == sku_id].sort_values("date")
    n = len(sku_data)
    td = min(test_days, n - 30)  # garantir treino minimo de 30
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


# ===================================================================
# API - FORECASTING
# ===================================================================
@app.route("/api/forecast/individual", methods=["POST"])
def api_forecast_individual():
    try:
        df = get_data()
        body = request.get_json(silent=True) or {}
        sku_id = body.get("sku_id")
        models = body.get("models", ["XGBoost", "LightGBM", "AutoARIMA"])
        horizon = body.get("horizon", 30)
        test_days = body.get("test_days", 60)

        if not sku_id:
            return jsonify({"error": "sku_id obrigatorio"}), 400

        sku_data = df[df["sku_id"] == sku_id]
        if sku_data.empty:
            return jsonify({"error": f"SKU '{sku_id}' nao encontrado"}), 404

        results = run_forecast_pipeline(df, sku_id, models, test_days, horizon)

        fig, fig_wape, detail_charts, metrics_rows = _build_forecast_overlay(
            df, sku_id, results, test_days)

        return safe_jsonify({
            "overlay_chart": fig_to_json(fig),
            "wape_chart": fig_to_json(fig_wape),
            "detail_charts": detail_charts,
            "metrics": metrics_rows,
            "params": {m: r.get("params", {}) for m, r in results.items()},
            "errors": {m: r["error"] for m, r in results.items() if r.get("error")},
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/forecast/aggregated", methods=["POST"])
def api_forecast_aggregated():
    try:
        df = get_data()
        body = request.get_json(silent=True) or {}
        models = body.get("models", ["XGBoost", "LightGBM", "AutoARIMA"])
        horizon = body.get("horizon", 30)
        test_days = body.get("test_days", 60)
        metric = body.get("metric", "pearson")
        weight_method = body.get("weight_method", "rolling")

        cluster_info = run_cluster_analysis(df, metric=metric)
        results = run_cluster_forecast_pipeline(df, cluster_info, models,
                                                test_days=test_days, horizon=horizon,
                                                weight_method=weight_method)

        cluster_data = aggregate_cluster_demand(df, cluster_info["sku_ids"], cluster_info["labels"])
        charts = {}

        for cid in sorted(results["cluster_forecasts"].keys()):
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
            for mname, fc in results["cluster_forecasts"][cid].items():
                if fc is not None:
                    color = MODEL_COLORS.get(mname, "#4361ee")
                    fig.add_trace(go.Scatter(
                        x=_dates_to_str(fc["ds"]), y=fc["yhat"].tolist(),
                        mode="lines", name=mname, line=dict(color=color, width=2.5)))
            fig.update_layout(**base_layout(f"Cluster {cid}", 380))
            charts[str(int(cid))] = fig_to_json(fig)

        # Pesos + metricas
        weights_out = {}
        for cid, w in results["weights"].items():
            weights_out[str(int(cid))] = [
                {"sku": k, "peso": round(float(v), 4), "pct": round(float(v * 100), 2)}
                for k, v in w.items()
            ]

        metrics_agg = {}
        for cid, mdata in results["metrics_agg"].items():
            rows = []
            for m, v in mdata.items():
                if "error" not in v:
                    rows.append({"Model": m, **{k: float(val) if isinstance(val, (np.floating, float)) else val for k, val in v.items()}})
            metrics_agg[str(int(cid))] = rows

        return safe_jsonify({
            "charts": charts,
            "weights": weights_out,
            "metrics": metrics_agg,
            "n_clusters": int(cluster_info["n_clusters"]),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ===================================================================
# API - COMPARACAO
# ===================================================================
@app.route("/api/comparison/run", methods=["POST"])
def api_comparison_run():
    try:
        df = get_data()
        body = request.get_json(silent=True) or {}
        models = body.get("models", ["XGBoost", "LightGBM", "AutoARIMA"])
        horizon = body.get("horizon", 30)
        test_days = body.get("test_days", 60)
        sku_ids = body.get("sku_ids") or sorted(df["sku_id"].unique().tolist())[:6]

        all_metrics = []
        sku_charts = {}

        for sid in sku_ids:
            sku_check = df[df["sku_id"] == sid]
            if sku_check.empty:
                continue

            sname = str(sku_check["sku_name"].iloc[0])
            profile = str(sku_check["demand_profile"].iloc[0])

            res = run_forecast_pipeline(df, sid, models, test_days, horizon)

            sku_data = sku_check.sort_values("date")
            n = len(sku_data)
            td = min(test_days, n - 30)
            if td < 1:
                td = max(1, n // 4)

            tr = sku_data.iloc[:-td]
            te = sku_data.iloc[-td:]

            fig = go.Figure()
            if not tr.empty:
                fig.add_trace(go.Scatter(
                    x=_dates_to_str(tr["date"]), y=tr["demand"].tolist(),
                    mode="lines", name="Historico", line=dict(color="#2b2d42", width=1.5)))
            if not te.empty:
                fig.add_trace(go.Scatter(
                    x=_dates_to_str(te["date"]), y=te["demand"].tolist(),
                    mode="lines", name="Real", line=dict(color="#1a1a2e", width=2, dash="dot")))

            for mname, r in res.items():
                if r.get("metrics"):
                    row = {"SKU": sid, "Nome": sname, "Perfil": profile, "Model": mname}
                    for k, v in r["metrics"].items():
                        row[k] = float(v) if isinstance(v, (np.floating, float, np.integer)) else v
                    all_metrics.append(row)
                if r.get("forecast") is not None:
                    fc = r["forecast"]
                    color = MODEL_COLORS.get(mname, "#4361ee")
                    fig.add_trace(go.Scatter(
                        x=_dates_to_str(fc["ds"]), y=fc["yhat"].tolist(),
                        mode="lines", name=mname, line=dict(color=color, width=2)))
            fig.update_layout(**base_layout(sname, 340))
            sku_charts[sid] = fig_to_json(fig)

        if not all_metrics:
            return jsonify({"error": "Nenhum resultado obtido. Verifique os modelos selecionados."}), 400

        mdf = pd.DataFrame(all_metrics)
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
        if mdf_wape.empty:
            return jsonify({"error": "Nenhum WAPE valido para comparar."}), 400
        best_per = mdf_wape.loc[mdf_wape.groupby("SKU")["WAPE"].idxmin()]
        bcounts = best_per["Model"].value_counts().reset_index()
        bcounts.columns = ["Model", "count"]
        fig = go.Figure(go.Bar(
            x=bcounts["Model"].tolist(), y=bcounts["count"].tolist(),
            marker=dict(color=[MODEL_COLORS.get(m, "#4361ee") for m in bcounts["Model"]]),
            text=bcounts["count"].tolist(), textposition="outside"))
        fig.update_layout(**base_layout("Vezes Melhor Modelo", 340))
        charts["best_count"] = fig_to_json(fig)

        return safe_jsonify({
            "metrics": all_metrics,
            "avg": json.loads(avg.to_json(orient="records")),
            "charts": charts,
            "sku_charts": sku_charts,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ===================================================================
if __name__ == "__main__":
    print("Gerando dados sinteticos...")
    get_data()
    print("Dados prontos. Iniciando servidor em http://localhost:5000")
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
