"""Chronos (Amazon) foundation model wrapper para forecast."""

import numpy as np
import pandas as pd
from models.base_model import BaseForecaster


class ChronosForecaster(BaseForecaster):
    name = "Chronos"
    supports_exogenous = False

    def __init__(self, model_id="amazon/chronos-bolt-small", device="cpu"):
        self.model_id = model_id
        self.device = device
        self.pipeline = None
        self._context = None
        self._last_date = None

    def fit(self, y: pd.Series, X: pd.DataFrame = None) -> "ChronosForecaster":
        try:
            import torch
            try:
                from chronos import ChronosBoltPipeline
                self.pipeline = ChronosBoltPipeline.from_pretrained(
                    self.model_id,
                    device_map=self.device,
                    torch_dtype=torch.float32,
                )
            except (ImportError, Exception):
                from chronos import ChronosPipeline
                self.pipeline = ChronosPipeline.from_pretrained(
                    self.model_id,
                    device_map=self.device,
                    torch_dtype=torch.float32,
                )

            self._context = torch.tensor(y.values, dtype=torch.float32)
            self._last_date = y.index[-1]
        except Exception as e:
            raise RuntimeError(f"Chronos nao disponivel: {e}. Instale com: pip install chronos-forecasting torch")
        return self

    def predict(self, horizon: int, X_future: pd.DataFrame = None) -> pd.DataFrame:
        if self.pipeline is None:
            raise RuntimeError("Modelo Chronos nao foi treinado. Chame fit() primeiro.")

        import torch

        forecast = self.pipeline.predict(
            self._context.unsqueeze(0),
            prediction_length=horizon,
            num_samples=20,
        )

        # forecast shape: (1, num_samples, horizon)
        forecast_np = forecast.numpy()[0] if hasattr(forecast, 'numpy') else np.array(forecast)[0]

        yhat = np.median(forecast_np, axis=0)
        yhat_lower = np.percentile(forecast_np, 10, axis=0)
        yhat_upper = np.percentile(forecast_np, 90, axis=0)

        future_dates = pd.date_range(
            start=self._last_date + pd.Timedelta(days=1),
            periods=horizon,
            freq="D",
        )

        return pd.DataFrame({
            "ds": future_dates,
            "yhat": np.maximum(0, yhat),
            "yhat_lower": np.maximum(0, yhat_lower),
            "yhat_upper": yhat_upper,
        })

    def get_params(self) -> dict:
        return {
            "model_id": self.model_id,
            "device": self.device,
        }
