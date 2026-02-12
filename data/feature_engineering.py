"""Engenharia de features para modelos de forecast."""

import numpy as np
import pandas as pd


def add_lag_features(df: pd.DataFrame, lags: list[int] = None) -> pd.DataFrame:
    """Adiciona features de lag por SKU."""
    if lags is None:
        lags = [1, 7, 14, 28]

    df = df.sort_values(["sku_id", "date"]).copy()
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby("sku_id")["demand"].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame, windows: list[int] = None
) -> pd.DataFrame:
    """Adiciona features de media e desvio movel por SKU."""
    if windows is None:
        windows = [7, 14, 28]

    df = df.sort_values(["sku_id", "date"]).copy()
    for w in windows:
        df[f"rolling_mean_{w}"] = (
            df.groupby("sku_id")["demand"]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        )
        df[f"rolling_std_{w}"] = (
            df.groupby("sku_id")["demand"]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).std())
        )
    df = df.fillna(0)
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Codifica variaveis categoricas para modelos ML."""
    df = df.copy()

    season_map = {"verao": 0, "outono": 1, "inverno": 2, "primavera": 3}
    safra_map = {"plantio": 0, "crescimento": 1, "colheita": 2, "entressafra": 3}

    df["season_encoded"] = df["season"].map(season_map).fillna(0).astype(int)

    for col in ["safra_soja", "safra_milho", "safra_cana"]:
        df[f"{col}_encoded"] = df[col].map(safra_map).fillna(0).astype(int)

    return df


def prepare_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline completo de features para modelos ML."""
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = encode_categorical(df)
    return df


def get_feature_columns() -> list[str]:
    """Retorna lista de colunas de features para ML."""
    return [
        "day_of_week", "month", "year",
        "temperature", "rainfall", "humidity",
        "season_encoded",
        "safra_soja_encoded", "safra_milho_encoded", "safra_cana_encoded",
        "lag_1", "lag_7", "lag_14", "lag_28",
        "rolling_mean_7", "rolling_mean_14", "rolling_mean_28",
        "rolling_std_7", "rolling_std_14", "rolling_std_28",
    ]
