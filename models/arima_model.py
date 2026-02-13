"""Auto ARIMA model wrapper para forecast usando statsmodels."""

import warnings
import itertools
import numpy as np
import pandas as pd
from models.base_model import BaseForecaster


class AutoARIMAForecaster(BaseForecaster):
    name = "AutoARIMA"
    supports_exogenous = True

    def __init__(self, seasonal=True, m=7, max_p=3, max_q=3):
        self.seasonal = seasonal
        self.m = m
        self.max_p = max_p
        self.max_q = max_q
        self.model = None
        self._last_y = None
        self._best_order = None
        self._best_seasonal = None

    def _auto_select(self, y, exog=None):
        """Seleciona melhor (p,d,q) via AIC com busca stepwise."""
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        from statsmodels.tsa.stattools import adfuller

        # Determinar d via teste ADF
        d = 0
        try:
            adf_p = adfuller(y, maxlag=min(14, len(y) // 3))[1]
            if adf_p > 0.05:
                d = 1
                diff_y = np.diff(y)
                adf_p2 = adfuller(diff_y, maxlag=min(14, len(diff_y) // 3))[1]
                if adf_p2 > 0.05:
                    d = 2
        except Exception:
            d = 1

        # Busca stepwise simplificada
        best_aic = np.inf
        best_order = (1, d, 1)
        best_seasonal = (0, 0, 0, 0)

        p_range = range(0, min(self.max_p + 1, 4))
        q_range = range(0, min(self.max_q + 1, 4))

        for p, q in itertools.product(p_range, q_range):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = SARIMAX(y, exog=exog, order=(p, d, q),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
                    result = model.fit(disp=False, maxiter=50)
                    if result.aic < best_aic:
                        best_aic = result.aic
                        best_order = (p, d, q)
            except Exception:
                continue

        # Tentar adicionar sazonalidade se configurado
        if self.seasonal and self.m > 1 and len(y) >= 2 * self.m:
            for P, Q in [(1, 0), (0, 1), (1, 1)]:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = SARIMAX(y, exog=exog, order=best_order,
                                        seasonal_order=(P, 0, Q, self.m),
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
                        result = model.fit(disp=False, maxiter=50)
                        if result.aic < best_aic:
                            best_aic = result.aic
                            best_seasonal = (P, 0, Q, self.m)
                except Exception:
                    continue

        return best_order, best_seasonal, best_aic

    def fit(self, y: pd.Series, X: pd.DataFrame = None) -> "AutoARIMAForecaster":
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        self._last_y = y.copy()
        exog = X.values if X is not None else None

        order, seasonal, _ = self._auto_select(y.values, exog)
        self._best_order = order
        self._best_seasonal = seasonal

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(y.values, exog=exog, order=order,
                            seasonal_order=seasonal if seasonal != (0, 0, 0, 0) else None,
                            enforce_stationarity=False,
                            enforce_invertibility=False)
            self.model = model.fit(disp=False, maxiter=200)

        return self

    def predict(self, horizon: int, X_future: pd.DataFrame = None) -> pd.DataFrame:
        exog_future = X_future.values if X_future is not None else None

        forecast = self.model.get_forecast(steps=horizon, exog=exog_future)
        preds = forecast.predicted_mean
        conf_int = forecast.conf_int(alpha=0.05)

        preds = np.maximum(0, preds)
        lower = np.maximum(0, conf_int.iloc[:, 0].values) if hasattr(conf_int, 'iloc') else np.maximum(0, conf_int[:, 0])
        upper = conf_int.iloc[:, 1].values if hasattr(conf_int, 'iloc') else conf_int[:, 1]

        last_date = self._last_y.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D"
        )

        return pd.DataFrame({
            "ds": future_dates,
            "yhat": preds,
            "yhat_lower": lower,
            "yhat_upper": upper,
        })

    def get_params(self) -> dict:
        if self.model is None:
            return {}
        return {
            "order": self._best_order,
            "seasonal_order": self._best_seasonal,
            "aic": round(float(self.model.aic), 2),
        }
