"""Modelos ML (XGBoost e LightGBM) para forecast de series temporais."""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from models.base_model import BaseForecaster


class XGBoostForecaster(BaseForecaster):
    name = "XGBoost"
    supports_exogenous = True

    def __init__(self, n_estimators=200, max_depth=6, learning_rate=0.1):
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
        }
        self.model = None
        self._last_y = None
        self._last_X = None

    def fit(self, y: pd.Series, X: pd.DataFrame = None) -> "XGBoostForecaster":
        import xgboost as xgb

        self._last_y = y.copy()
        self._last_X = X.copy() if X is not None else None

        if X is None:
            raise ValueError("XGBoost requer features exogenas (X).")

        self.model = xgb.XGBRegressor(
            **self.params,
            random_state=42,
            verbosity=0,
        )
        self.model.fit(X.values, y.values)
        return self

    def predict(self, horizon: int, X_future: pd.DataFrame = None) -> pd.DataFrame:
        if X_future is None:
            raise ValueError("XGBoost requer features futuras (X_future).")

        preds = self.model.predict(X_future.values)
        preds = np.maximum(0, preds)

        # Estimativa de intervalos via residuos do treino
        train_preds = self.model.predict(self._last_X.values)
        residual_std = np.std(self._last_y.values - train_preds)

        last_date = self._last_y.index[-1] if hasattr(self._last_y.index, 'freq') else self._last_y.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

        return pd.DataFrame({
            "ds": future_dates[:len(preds)],
            "yhat": preds,
            "yhat_lower": np.maximum(0, preds - 1.96 * residual_std),
            "yhat_upper": preds + 1.96 * residual_std,
        })

    def get_params(self) -> dict:
        return self.params

    def feature_importance(self) -> pd.Series:
        if self.model is None:
            return pd.Series()
        return pd.Series(
            self.model.feature_importances_,
            index=self._last_X.columns if self._last_X is not None else None,
        ).sort_values(ascending=False)


class LightGBMForecaster(BaseForecaster):
    name = "LightGBM"
    supports_exogenous = True

    def __init__(self, n_estimators=200, max_depth=6, learning_rate=0.1):
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
        }
        self.model = None
        self._last_y = None
        self._last_X = None

    def fit(self, y: pd.Series, X: pd.DataFrame = None) -> "LightGBMForecaster":
        import lightgbm as lgb

        self._last_y = y.copy()
        self._last_X = X.copy() if X is not None else None

        if X is None:
            raise ValueError("LightGBM requer features exogenas (X).")

        self.model = lgb.LGBMRegressor(
            **self.params,
            random_state=42,
            verbose=-1,
        )
        self.model.fit(X.values, y.values)
        return self

    def predict(self, horizon: int, X_future: pd.DataFrame = None) -> pd.DataFrame:
        if X_future is None:
            raise ValueError("LightGBM requer features futuras (X_future).")

        preds = self.model.predict(X_future.values)
        preds = np.maximum(0, preds)

        train_preds = self.model.predict(self._last_X.values)
        residual_std = np.std(self._last_y.values - train_preds)

        last_date = self._last_y.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

        return pd.DataFrame({
            "ds": future_dates[:len(preds)],
            "yhat": preds,
            "yhat_lower": np.maximum(0, preds - 1.96 * residual_std),
            "yhat_upper": preds + 1.96 * residual_std,
        })

    def get_params(self) -> dict:
        return self.params

    def feature_importance(self) -> pd.Series:
        if self.model is None:
            return pd.Series()
        return pd.Series(
            self.model.feature_importances_,
            index=self._last_X.columns if self._last_X is not None else None,
        ).sort_values(ascending=False)
