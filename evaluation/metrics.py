"""Metricas de avaliacao de modelos de forecast."""

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (ignora zeros no denominador)."""
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error."""
    total = np.sum(np.abs(y_true))
    if total == 0:
        return np.nan
    return float(np.sum(np.abs(y_true - y_pred)) / total * 100)


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Bias medio (positivo = sobre-previsao)."""
    return float(np.mean(y_pred - y_true))


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcula todas as metricas de uma vez."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": round(mae(y_true, y_pred), 2),
        "RMSE": round(rmse(y_true, y_pred), 2),
        "MAPE": round(mape(y_true, y_pred), 2),
        "WAPE": round(wape(y_true, y_pred), 2),
        "Bias": round(bias(y_true, y_pred), 2),
    }


def evaluate_forecasts(
    results: dict[str, pd.DataFrame],
    y_test: pd.Series,
) -> pd.DataFrame:
    """Avalia multiplos modelos contra dados de teste.

    Args:
        results: Dict {model_name: forecast_df} onde forecast_df tem coluna 'yhat'.
        y_test: Serie real de teste.

    Returns:
        DataFrame com metricas por modelo.
    """
    rows = []
    y_true = y_test.values

    for model_name, forecast_df in results.items():
        y_pred = forecast_df["yhat"].values[:len(y_true)]
        metrics = calculate_all_metrics(y_true, y_pred)
        metrics["Model"] = model_name
        rows.append(metrics)

    df = pd.DataFrame(rows)
    df = df[["Model", "MAE", "RMSE", "MAPE", "WAPE", "Bias"]]
    return df
