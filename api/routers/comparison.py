"""Comparison endpoint."""

import json
import traceback
import asyncio

import numpy as np
import pandas as pd

from fastapi import APIRouter

from api.dependencies import get_data, executor
from api.json_utils import NumpyEncoder, NumpyJSONResponse
from api.schemas.comparison import ComparisonRequest
from api.chart_builder import build_comparison_charts, build_comparison_sku_chart

router = APIRouter(tags=["comparison"])


def _run_comparison(body: ComparisonRequest):
    from pipeline.forecasting_pipeline import run_forecast_pipeline
    from config import MODEL_COLORS

    df = get_data()
    sku_ids = body.sku_ids if body.sku_ids else sorted(df["sku_id"].unique().tolist())[:6]

    all_metrics = []
    sku_charts = {}

    for sid in sku_ids:
        sku_check = df[df["sku_id"] == sid]
        if sku_check.empty:
            continue

        sname = str(sku_check["sku_name"].iloc[0])
        profile = str(sku_check["demand_profile"].iloc[0])

        res = run_forecast_pipeline(df, sid, body.models, body.test_days, body.horizon)

        sku_charts[sid] = build_comparison_sku_chart(df, sid, res, body.test_days)

        for mname, r in res.items():
            if r.get("metrics"):
                row = {"SKU": sid, "Nome": sname, "Perfil": profile, "Model": mname}
                for k, v in r["metrics"].items():
                    row[k] = float(v) if isinstance(v, (np.floating, float, np.integer)) else v
                all_metrics.append(row)

    if not all_metrics:
        return {"error": "Nenhum resultado obtido. Verifique os modelos selecionados."}

    mdf = pd.DataFrame(all_metrics)
    charts, avg = build_comparison_charts(mdf)

    return json.loads(json.dumps({
        "metrics": all_metrics,
        "avg": json.loads(avg.to_json(orient="records")),
        "charts": charts,
        "sku_charts": sku_charts,
    }, cls=NumpyEncoder))


@router.post("/comparison/run", response_class=NumpyJSONResponse)
async def api_comparison_run(body: ComparisonRequest):
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, _run_comparison, body)
        return result
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}
