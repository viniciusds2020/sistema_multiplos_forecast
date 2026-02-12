"""Prophet model wrapper para forecast."""

import numpy as np
import pandas as pd
from models.base_model import BaseForecaster


class ProphetForecaster(BaseForecaster):
    name = "Prophet"
    supports_exogenous = True

    def __init__(self, yearly_seasonality=True, weekly_seasonality=True,
                 changepoint_prior_scale=0.05):
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.model = None
        self._regressors = []
        self._last_date = None

    def fit(self, y: pd.Series, X: pd.DataFrame = None) -> "ProphetForecaster":
        from prophet import Prophet

        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale,
            daily_seasonality=False,
        )
        self.model.add_country_holidays(country_name="BR")

        # Preparar dataframe Prophet
        prophet_df = pd.DataFrame({
            "ds": y.index,
            "y": y.values,
        })

        # Adicionar regressores exogenos
        if X is not None:
            self._regressors = list(X.columns)
            for col in self._regressors:
                self.model.add_regressor(col)
                prophet_df[col] = X[col].values

        self.model.fit(prophet_df)
        self._last_date = y.index[-1]
        return self

    def predict(self, horizon: int, X_future: pd.DataFrame = None) -> pd.DataFrame:
        future = self.model.make_future_dataframe(periods=horizon, freq="D")
        future = future.tail(horizon).reset_index(drop=True)

        if X_future is not None and self._regressors:
            for col in self._regressors:
                if col in X_future.columns:
                    future[col] = X_future[col].values[:horizon]

        forecast = self.model.predict(future)

        return pd.DataFrame({
            "ds": forecast["ds"].values,
            "yhat": np.maximum(0, forecast["yhat"].values),
            "yhat_lower": np.maximum(0, forecast["yhat_lower"].values),
            "yhat_upper": forecast["yhat_upper"].values,
        })

    def get_params(self) -> dict:
        return {
            "yearly_seasonality": self.yearly_seasonality,
            "weekly_seasonality": self.weekly_seasonality,
            "changepoint_prior_scale": self.changepoint_prior_scale,
            "regressors": self._regressors,
        }
