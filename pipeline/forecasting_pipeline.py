"""Pipeline de forecasting individual por SKU."""

import pandas as pd
import numpy as np
from data.feature_engineering import (
    prepare_ml_features,
    get_ml_feature_columns,
    get_exogenous_feature_columns,
)
from models.model_registry import get_model
from evaluation.metrics import calculate_all_metrics


# Modelos que so devem receber variaveis verdadeiramente exogenas.
# ARIMA e Prophet ja modelam autocorrelacao internamente — passar lags
# causa multicolinearidade e piora a performance.
_EXOGENOUS_ONLY_MODELS = {"AutoARIMA", "Prophet"}


def split_train_test(
    df_sku: pd.DataFrame, test_days: int = 60
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide dados de um SKU em treino e teste."""
    df_sku = df_sku.sort_values("date").reset_index(drop=True)
    n = len(df_sku)
    if n == 0:
        return df_sku.copy(), df_sku.copy()
    if test_days < 1:
        test_days = 1
    if test_days >= n:
        test_days = max(1, n - 1)
    split_idx = n - test_days
    return df_sku.iloc[:split_idx].copy(), df_sku.iloc[split_idx:].copy()


def _select_feature_cols(model_name: str) -> list[str]:
    """Retorna o conjunto de features adequado para cada modelo."""
    if model_name in _EXOGENOUS_ONLY_MODELS:
        return get_exogenous_feature_columns()
    return get_ml_feature_columns()


def run_single_model(
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    horizon: int,
    feature_cols: list[str] = None,
) -> dict:
    """Executa um modelo para um SKU.

    Returns:
        Dict com: model_name, forecast_df, metrics, params, error
    """
    result = {
        "model_name": model_name,
        "forecast": None,
        "metrics": None,
        "params": {},
        "error": None,
    }

    try:
        model = get_model(model_name)

        if len(test_df) == 0:
            raise ValueError("Conjunto de teste vazio. Ajuste test_days.")
        if horizon > len(test_df):
            horizon = len(test_df)

        y_train = pd.Series(
            train_df["demand"].values,
            index=pd.DatetimeIndex(train_df["date"]),
        )
        y_test = pd.Series(
            test_df["demand"].values[:horizon],
            index=pd.DatetimeIndex(test_df["date"].values[:horizon]),
        )

        # Seleciona features adequadas ao modelo
        X_train = None
        X_future = None

        if model.supports_exogenous and feature_cols:
            cols = _select_feature_cols(model_name)
            available = [c for c in cols if c in train_df.columns]
            if available:
                X_train  = train_df[available].reset_index(drop=True)
                X_future = test_df[available].iloc[:horizon].reset_index(drop=True)

        model.fit(y_train, X_train)
        forecast = model.predict(horizon, X_future)
        result["forecast"] = forecast
        result["params"]   = model.get_params()

        y_pred = forecast["yhat"].values[:len(y_test)]
        y_true = y_test.values[:len(y_pred)]
        result["metrics"] = calculate_all_metrics(y_true, y_pred)

    except Exception as e:
        result["error"] = str(e)

    return result


def run_forecast_pipeline(
    df: pd.DataFrame,
    sku_id: str,
    model_names: list[str],
    test_days: int = 60,
    horizon: int = 30,
) -> dict:
    """Executa pipeline completo de forecast para um SKU.

    Returns:
        Dict com resultados por modelo.
    """
    df_sku = df[df["sku_id"] == sku_id].copy()
    if df_sku.empty:
        return {}

    df_sku = prepare_ml_features(df_sku)
    # feature_cols passado apenas como sinalizador — selecao real e feita
    # em _select_feature_cols() dentro de run_single_model()
    feature_cols = get_ml_feature_columns()

    n = len(df_sku)
    if n < 60:
        return {m: {"model_name": m, "forecast": None, "metrics": None, "params": {},
                     "error": f"Dados insuficientes ({n} registros, minimo 60)"} for m in model_names}

    safe_test_days = min(test_days, n - 30)
    safe_test_days = max(1, min(safe_test_days, n - 1))
    train_df, test_df = split_train_test(df_sku, safe_test_days)

    results = {}
    for model_name in model_names:
        res = run_single_model(model_name, train_df, test_df, horizon, feature_cols)
        results[model_name] = res

    return results


def run_all_skus_pipeline(
    df: pd.DataFrame,
    model_names: list[str],
    test_days: int = 60,
    horizon: int = 30,
    progress_callback=None,
) -> dict:
    """Executa forecast para todos os SKUs.

    Returns:
        Dict {sku_id: {model_name: result}}
    """
    sku_ids = df["sku_id"].unique()
    all_results = {}

    for idx, sku_id in enumerate(sku_ids):
        if progress_callback:
            progress_callback(idx / len(sku_ids), f"Processando {sku_id}...")

        all_results[sku_id] = run_forecast_pipeline(
            df, sku_id, model_names, test_days, horizon
        )

    if progress_callback:
        progress_callback(1.0, "Concluido!")

    return all_results
