"""Registry de modelos de forecast disponíveis."""

from models.ml_models import XGBoostForecaster, LightGBMForecaster
from models.arima_model import AutoARIMAForecaster
from models.prophet_model import ProphetForecaster
from models.chronos_model import ChronosForecaster
from models.intermittent_models import CrostonSBAForecaster

MODEL_REGISTRY = {
    "XGBoost": XGBoostForecaster,
    "LightGBM": LightGBMForecaster,
    "AutoARIMA": AutoARIMAForecaster,
    "Prophet": ProphetForecaster,
    "Chronos": ChronosForecaster,
    "CrostonSBA": CrostonSBAForecaster,
}


def get_model(name: str, **kwargs):
    """Instancia um modelo pelo nome."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Modelo '{name}' nao encontrado. Disponiveis: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)


def list_models() -> list[str]:
    """Lista nomes dos modelos disponiveis."""
    return list(MODEL_REGISTRY.keys())
