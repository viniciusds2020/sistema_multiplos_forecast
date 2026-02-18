"""JSON utilities for numpy/pandas serialization in FastAPI."""

import json
from typing import Any

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
from fastapi.responses import JSONResponse


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
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


class NumpyJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        return json.dumps(content, cls=NumpyEncoder).encode("utf-8")


def fig_to_json(fig: go.Figure) -> str:
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def safe_serialize(data: Any) -> Any:
    """Convert numpy/pandas types to JSON-safe Python types."""
    return json.loads(json.dumps(data, cls=NumpyEncoder))
