"""Auto ARIMA model wrapper para forecast."""

import numpy as np
import pandas as pd
from models.base_model import BaseForecaster


class AutoARIMAForecaster(BaseForecaster):
    name = "AutoARIMA"
    supports_exogenous = True

    def __init__(self, seasonal=True, m=7, max_p=5, max_q=5):
        self.seasonal = seasonal
        self.m = m
        self.max_p = max_p
        self.max_q = max_q
        self.model = None
        self._last_y = None

    def fit(self, y: pd.Series, X: pd.DataFrame = None) -> "AutoARIMAForecaster":
        import pmdarima as pm

        self._last_y = y.copy()
        exog = X.values if X is not None else None

        self.model = pm.auto_arima(
            y.values,
            exogenous=exog,
            seasonal=self.seasonal,
            m=self.m,
            max_p=self.max_p,
            max_q=self.max_q,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            trace=False,
        )
        return self

    def predict(self, horizon: int, X_future: pd.DataFrame = None) -> pd.DataFrame:
        exog_future = X_future.values if X_future is not None else None

        preds, conf_int = self.model.predict(
            n_periods=horizon,
            exogenous=exog_future,
            return_conf_int=True,
            alpha=0.05,
        )

        preds = np.maximum(0, preds)
        conf_int[:, 0] = np.maximum(0, conf_int[:, 0])

        last_date = self._last_y.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D"
        )

        return pd.DataFrame({
            "ds": future_dates,
            "yhat": preds,
            "yhat_lower": conf_int[:, 0],
            "yhat_upper": conf_int[:, 1],
        })

    def get_params(self) -> dict:
        if self.model is None:
            return {}
        return {
            "order": self.model.order,
            "seasonal_order": self.model.seasonal_order,
            "aic": round(self.model.aic(), 2),
        }
