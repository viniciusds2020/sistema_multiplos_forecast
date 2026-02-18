"""General API endpoints: models, kpis, skus, cache."""

import json

from fastapi import APIRouter

from api.dependencies import get_data, cached_endpoint, clear_cache
from api.json_utils import NumpyEncoder, NumpyJSONResponse

router = APIRouter(tags=["general"])


@router.get("/models", response_class=NumpyJSONResponse)
@cached_endpoint(ttl=3600)
def api_models():
    from models.model_registry import list_models, list_available_models
    return {"all": list_models(), "available": list_available_models()}


@router.get("/kpis", response_class=NumpyJSONResponse)
@cached_endpoint(ttl=600)
def api_kpis():
    df = get_data()
    return json.loads(json.dumps({
        "total_skus": int(df["sku_id"].nunique()),
        "total_records": int(len(df)),
        "coverage_days": int((df["date"].max() - df["date"].min()).days),
        "avg_demand": round(float(df["demand"].mean()), 1),
        "date_start": str(df["date"].min().date()),
        "date_end": str(df["date"].max().date()),
    }, cls=NumpyEncoder))


@router.get("/skus", response_class=NumpyJSONResponse)
@cached_endpoint(ttl=600)
def api_skus():
    df = get_data()
    skus = (
        df.groupby(["sku_id", "sku_name", "demand_profile"])
        .agg(mean_demand=("demand", "mean"), total_demand=("demand", "sum"))
        .reset_index()
    )
    skus["mean_demand"] = skus["mean_demand"].round(1)
    skus["total_demand"] = skus["total_demand"].astype(int)
    return json.loads(json.dumps(skus.to_dict(orient="records"), cls=NumpyEncoder))


@router.post("/cache/clear")
def api_cache_clear():
    clear_cache()
    return {"status": "cache cleared"}
