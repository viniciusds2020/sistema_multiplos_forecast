"""Overview charts endpoint."""

from fastapi import APIRouter

from api.dependencies import get_data, cached_endpoint
from api.json_utils import NumpyJSONResponse, safe_serialize
from api.chart_builder import build_overview_charts

router = APIRouter(tags=["overview"])


@router.get("/overview/charts", response_class=NumpyJSONResponse)
@cached_endpoint(ttl=600)
def api_overview_charts():
    try:
        df = get_data()
        charts = build_overview_charts(df)
        return safe_serialize(charts)
    except Exception as e:
        return {"error": str(e)}
