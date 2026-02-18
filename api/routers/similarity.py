"""Similarity / clustering endpoint."""

import json
import traceback
import asyncio

import numpy as np

from fastapi import APIRouter

from api.dependencies import get_data, set_cluster_info, executor
from api.json_utils import NumpyEncoder, NumpyJSONResponse
from api.schemas.similarity import SimilarityRequest
from api.chart_builder import build_similarity_charts

router = APIRouter(tags=["similarity"])


def _run_clustering(body: SimilarityRequest):
    from similarity.clustering import (
        build_demand_matrix, compute_distance_matrix,
        find_optimal_clusters, cluster_series, compute_mds_projection,
        get_cluster_summary,
    )

    df = get_data()
    matrix, sku_ids = build_demand_matrix(df)

    if matrix.shape[0] < 2:
        return {"error": "Sao necessarios pelo menos 2 SKUs para clustering"}

    dist_matrix = compute_distance_matrix(matrix, metric=body.metric)

    if body.auto_k:
        n_clusters, sil_scores = find_optimal_clusters(matrix, dist_matrix)
    else:
        n_clusters = body.manual_k
        _, sil_scores = find_optimal_clusters(matrix, dist_matrix)

    labels, Z = cluster_series(dist_matrix, n_clusters)
    mds = compute_mds_projection(dist_matrix)
    summary = get_cluster_summary(df, sku_ids, labels)

    # Store in cache for aggregated forecast
    set_cluster_info({
        "sku_ids": sku_ids,
        "labels": [int(x) for x in labels],
        "linkage": [[float(v) for v in row] for row in Z],
        "n_clusters": int(n_clusters),
    })

    charts = build_similarity_charts(
        dist_matrix, sku_ids, labels, n_clusters, sil_scores, mds, Z, df
    )

    # Summary table
    display_cols = ["cluster", "sku_id", "sku_name", "demand_profile", "mean_demand", "cv", "zero_pct"]
    avail = [c for c in display_cols if c in summary.columns]
    display = summary[avail].copy()
    for col in ["mean_demand", "cv", "zero_pct"]:
        if col in display.columns:
            display[col] = display[col].round(3).astype(float)
    display["cluster"] = display["cluster"].astype(int)

    return json.loads(json.dumps({
        "charts": charts,
        "n_clusters": int(n_clusters),
        "sil_best": round(float(max(sil_scores.values())), 3) if sil_scores else 0,
        "total_skus": len(sku_ids),
        "summary": json.loads(display.to_json(orient="records")),
    }, cls=NumpyEncoder))


@router.post("/similarity/run", response_class=NumpyJSONResponse)
async def api_similarity_run(body: SimilarityRequest):
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, _run_clustering, body)
        return result
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}
