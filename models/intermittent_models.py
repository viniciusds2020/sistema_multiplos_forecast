"""Modelos para demanda intermitente (Croston SBA)."""

import numpy as np
import pandas as pd
from models.base_model import BaseForecaster


class CrostonSBAForecaster(BaseForecaster):
    name = "CrostonSBA"
    supports_exogenous = False
    supports_intermittent = True

    def __init__(self):
        self.model = None
        self._last_y = None
        self._fitted_values = None

    def fit(self, y: pd.Series, X: pd.DataFrame = None) -> "CrostonSBAForecaster":
        from statsforecast import StatsForecast
        from statsforecast.models import CrostonSBA

        self._last_y = y.copy()

        sf_df = pd.DataFrame({
            "unique_id": "sku",
            "ds": y.index,
            "y": y.values.astype(float),
        })

        self.model = StatsForecast(
            models=[CrostonSBA()],
            freq="D",
        )
        self.model.fit(sf_df)
        return self

    def predict(self, horizon: int, X_future: pd.DataFrame = None) -> pd.DataFrame:
        forecast = self.model.predict(h=horizon)

        yhat = forecast["CrostonSBA"].values
        yhat = np.maximum(0, yhat)

        # Croston produz forecasts planos - estimar intervalos via residuos
        fitted = self.model.fitted_[0]["CrostonSBA"].values if hasattr(self.model, 'fitted_') else None
        if fitted is not None:
            residual_std = np.std(self._last_y.values[-len(fitted):] - fitted)
        else:
            residual_std = np.std(self._last_y.values) * 0.5

        future_dates = pd.date_range(
            start=self._last_y.index[-1] + pd.Timedelta(days=1),
            periods=horizon,
            freq="D",
        )

        return pd.DataFrame({
            "ds": future_dates[:len(yhat)],
            "yhat": yhat,
            "yhat_lower": np.maximum(0, yhat - 1.96 * residual_std),
            "yhat_upper": yhat + 1.96 * residual_std,
        })

    def get_params(self) -> dict:
        return {"method": "CrostonSBA"}
