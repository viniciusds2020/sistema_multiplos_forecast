"""Explorer endpoints: demand, climate, distributions, raw data."""

from fastapi import APIRouter, Query

from api.dependencies import get_data, cached_endpoint
from api.json_utils import NumpyJSONResponse, safe_serialize
from api.chart_builder import build_demand_chart, build_climate_charts, build_distribution_charts

router = APIRouter(tags=["explorer"])


@router.get("/explorer/demand", response_class=NumpyJSONResponse)
def api_explorer_demand(skus: list[str] = Query(default=[])):
    df = get_data()
    sku_list = skus if skus else list(df["sku_name"].unique()[:4])
    chart = build_demand_chart(df, sku_list)
    return {"chart": chart}


@router.get("/explorer/climate", response_class=NumpyJSONResponse)
@cached_endpoint(ttl=600)
def api_explorer_climate():
    df = get_data()
    return build_climate_charts(df)


@router.get("/explorer/distributions", response_class=NumpyJSONResponse)
@cached_endpoint(ttl=600)
def api_explorer_distributions():
    df = get_data()
    return build_distribution_charts(df)


@router.get("/explorer/raw", response_class=NumpyJSONResponse)
@cached_endpoint(ttl=600)
def api_explorer_raw():
    df = get_data()
    sample = df.sort_values(["sku_id", "date"]).head(300).copy()
    sample["date"] = sample["date"].dt.strftime("%Y-%m-%d")
    return safe_serialize(sample.to_dict(orient="records"))
