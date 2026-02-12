"""Pipeline de forecasting individual por SKU."""

import pandas as pd
import numpy as np
from data.feature_engineering import prepare_ml_features, get_feature_columns
from models.model_registry import get_model
from evaluation.metrics import calculate_all_metrics


def split_train_test(
    df_sku: pd.DataFrame, test_days: int = 60
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide dados de um SKU em treino e teste."""
    df_sku = df_sku.sort_values("date").reset_index(drop=True)
    split_idx = len(df_sku) - test_days
    return df_sku.iloc[:split_idx].copy(), df_sku.iloc[split_idx:].copy()


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

        # Preparar series
        y_train = pd.Series(
            train_df["demand"].values,
            index=pd.DatetimeIndex(train_df["date"]),
        )
        y_test = pd.Series(
            test_df["demand"].values[:horizon],
            index=pd.DatetimeIndex(test_df["date"].values[:horizon]),
        )

        # Preparar features
        X_train = None
        X_future = None

        if model.supports_exogenous and feature_cols:
            available_cols = [c for c in feature_cols if c in train_df.columns]
            if available_cols:
                X_train = train_df[available_cols].reset_index(drop=True)
                X_future = test_df[available_cols].iloc[:horizon].reset_index(drop=True)

        model.fit(y_train, X_train)
        forecast = model.predict(horizon, X_future)
        result["forecast"] = forecast
        result["params"] = model.get_params()

        # Metricas
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

    # Feature engineering
    df_sku = prepare_ml_features(df_sku)
    feature_cols = get_feature_columns()

    train_df, test_df = split_train_test(df_sku, test_days)

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
