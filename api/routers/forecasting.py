"""Forecasting endpoints: individual and aggregated."""

import json
import traceback
import asyncio

import numpy as np
import pandas as pd

from fastapi import APIRouter

from api.dependencies import get_data, executor
from api.json_utils import NumpyEncoder, NumpyJSONResponse
from api.schemas.forecasting import IndividualForecastRequest, AggregatedForecastRequest
from api.chart_builder import (
    build_forecast_overlay, fig_to_json,
    build_aggregated_charts,
)

router = APIRouter(tags=["forecasting"])


def _run_individual(body: IndividualForecastRequest):
    from pipeline.forecasting_pipeline import run_forecast_pipeline
    from config import MODEL_COLORS

    df = get_data()
    sku_data = df[df["sku_id"] == body.sku_id]
    if sku_data.empty:
        return {"error": f"SKU '{body.sku_id}' nao encontrado"}

    results = run_forecast_pipeline(df, body.sku_id, body.models, body.test_days, body.horizon)

    fig, fig_wape, detail_charts, metrics_rows = build_forecast_overlay(
        df, body.sku_id, results, body.test_days)

    return json.loads(json.dumps({
        "overlay_chart": fig_to_json(fig),
        "wape_chart": fig_to_json(fig_wape),
        "detail_charts": detail_charts,
        "metrics": metrics_rows,
        "params": {m: r.get("params", {}) for m, r in results.items()},
        "errors": {m: r["error"] for m, r in results.items() if r.get("error")},
    }, cls=NumpyEncoder))


def _run_aggregated(body: AggregatedForecastRequest):
    from pipeline.cluster_pipeline import run_cluster_analysis, run_cluster_forecast_pipeline
    from similarity.aggregation import aggregate_cluster_demand

    df = get_data()
    cluster_info = run_cluster_analysis(df, metric=body.metric)
    results = run_cluster_forecast_pipeline(
        df, cluster_info, body.models,
        test_days=body.test_days, horizon=body.horizon,
        weight_method=body.weight_method)

    cluster_data = aggregate_cluster_demand(df, cluster_info["sku_ids"], cluster_info["labels"])
    charts = build_aggregated_charts(df, cluster_data, results["cluster_forecasts"], body.test_days)

    # Weights
    weights_out = {}
    for cid, w in results["weights"].items():
        weights_out[str(int(cid))] = [
            {"sku": k, "peso": round(float(v), 4), "pct": round(float(v * 100), 2)}
            for k, v in w.items()
        ]

    # Metrics
    metrics_agg = {}
    for cid, mdata in results["metrics_agg"].items():
        rows = []
        for m, v in mdata.items():
            if "error" not in v:
                rows.append({"Model": m, **{
                    k: float(val) if isinstance(val, (np.floating, float)) else val
                    for k, val in v.items()
                }})
        metrics_agg[str(int(cid))] = rows

    return json.loads(json.dumps({
        "charts": charts,
        "weights": weights_out,
        "metrics": metrics_agg,
        "n_clusters": int(cluster_info["n_clusters"]),
    }, cls=NumpyEncoder))


@router.post("/forecast/individual", response_class=NumpyJSONResponse)
async def api_forecast_individual(body: IndividualForecastRequest):
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, _run_individual, body)
        return result
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


@router.post("/forecast/aggregated", response_class=NumpyJSONResponse)
async def api_forecast_aggregated(body: AggregatedForecastRequest):
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, _run_aggregated, body)
        return result
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}
