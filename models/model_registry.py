"""Registry de modelos de forecast disponíveis com imports lazy."""

import importlib

# Mapeamento: nome -> (modulo, classe)
_MODEL_MAP = {
    "XGBoost": ("models.ml_models", "XGBoostForecaster"),
    "LightGBM": ("models.ml_models", "LightGBMForecaster"),
    "AutoARIMA": ("models.arima_model", "AutoARIMAForecaster"),
    "Prophet": ("models.prophet_model", "ProphetForecaster"),
    "Chronos": ("models.chronos_model", "ChronosForecaster"),
    "CrostonSBA": ("models.intermittent_models", "CrostonSBAForecaster"),
}

# Cache de classes ja importadas
_loaded = {}


def get_model(name: str, **kwargs):
    """Instancia um modelo pelo nome com import lazy."""
    if name not in _MODEL_MAP:
        raise ValueError(f"Modelo '{name}' nao encontrado. Disponiveis: {list(_MODEL_MAP.keys())}")

    if name not in _loaded:
        module_path, class_name = _MODEL_MAP[name]
        try:
            mod = importlib.import_module(module_path)
            _loaded[name] = getattr(mod, class_name)
        except Exception as e:
            raise ImportError(f"Erro ao carregar modelo '{name}': {e}")

    return _loaded[name](**kwargs)


def list_models() -> list:
    """Lista nomes dos modelos disponiveis."""
    return list(_MODEL_MAP.keys())


def list_available_models() -> list:
    """Lista modelos que podem ser importados sem erro."""
    available = []
    for name in _MODEL_MAP:
        try:
            get_model(name)
            available.append(name)
        except Exception:
            pass
    return available
