"""Engenharia de features para modelos de forecast."""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Construtores de features
# ---------------------------------------------------------------------------

def add_lag_features(df: pd.DataFrame, lags: list[int] = None) -> pd.DataFrame:
    """Adiciona features de lag por SKU.

    Lags curtos (2-6) capturam padroes de dias adjacentes.
    Lags medios (7, 14) capturam sazonalidade semanal/quinzenal.
    Lag longo (28) captura padrao mensal.
    """
    if lags is None:
        lags = [1, 2, 3, 4, 5, 6, 7, 14, 28]

    df = df.sort_values(["sku_id", "date"]).copy()
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby("sku_id")["demand"].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, windows: list[int] = None) -> pd.DataFrame:
    """Adiciona media, desvio, max e min moveis por SKU.

    Min/max capturam amplitude da demanda na janela (pico e vale).
    """
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
        df[f"rolling_max_{w}"] = (
            df.groupby("sku_id")["demand"]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).max())
        )
        df[f"rolling_min_{w}"] = (
            df.groupby("sku_id")["demand"]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).min())
        )
    df = df.fillna(0)
    return df


def add_ewm_features(df: pd.DataFrame, spans: list[int] = None) -> pd.DataFrame:
    """Adiciona Exponential Weighted Mean por SKU.

    EWM da mais peso a observacoes recentes via decaimento exponencial.
    Mais reativo a mudancas de nivel do que rolling mean simples.
    span=7  -> decaimento rapido (sensivel a variacao recente)
    span=28 -> decaimento lento (tendencia de longo prazo)
    """
    if spans is None:
        spans = [7, 14, 28]

    df = df.sort_values(["sku_id", "date"]).copy()
    for span in spans:
        df[f"ewm_{span}"] = (
            df.groupby("sku_id")["demand"]
            .transform(lambda x: x.shift(1).ewm(span=span, min_periods=1).mean())
        )
    df = df.fillna(0)
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona features de calendario enriquecidas.

    is_weekend     -> demanda diferente em finais de semana
    week_of_year   -> captura sazonalidade anual granular
    quarter        -> padrao trimestral / fiscal
    day_of_year    -> posicao no ano (continuua)
    is_month_start -> efeito de inicio de mes (compras, reposicao)
    is_month_end   -> efeito de fim de mes (fechamento de pedidos)
    """
    df = df.copy()
    df["is_weekend"]     = (df["day_of_week"] >= 5).astype(int)
    df["week_of_year"]   = df["date"].dt.isocalendar().week.astype(int)
    df["quarter"]        = df["date"].dt.quarter.astype(int)
    df["day_of_year"]    = df["date"].dt.dayofyear.astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"]   = df["date"].dt.is_month_end.astype(int)
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Codifica variaveis categoricas para modelos ML."""
    df = df.copy()

    season_map = {"verao": 0, "outono": 1, "inverno": 2, "primavera": 3}
    safra_map  = {"plantio": 0, "crescimento": 1, "colheita": 2, "entressafra": 3}

    df["season_encoded"] = df["season"].map(season_map).fillna(0).astype(int)

    for col in ["safra_soja", "safra_milho", "safra_cana"]:
        df[f"{col}_encoded"] = df[col].map(safra_map).fillna(0).astype(int)

    return df


# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------

def prepare_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline completo de features — aplicado a todos os modelos.

    Cada modelo recebe apenas as colunas relevantes para seu tipo
    via get_ml_feature_columns() ou get_exogenous_feature_columns().
    """
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_ewm_features(df)
    df = add_calendar_features(df)
    df = encode_categorical(df)
    return df


# ---------------------------------------------------------------------------
# Conjuntos de features por tipo de modelo
# ---------------------------------------------------------------------------

def get_ml_feature_columns() -> list[str]:
    """Features completas para modelos ML (XGBoost, LightGBM).

    Inclui lags e estatisticas autoregressivas porque esses modelos
    nao modelam autocorrelacao internamente — precisam dessas features
    para aprender dependencias temporais.

    Total: ~40 features
    """
    calendar = [
        "day_of_week", "month", "year",
        "is_weekend", "week_of_year", "quarter",
        "day_of_year", "is_month_start", "is_month_end",
    ]
    climate = ["temperature", "rainfall", "humidity"]
    categorical = [
        "season_encoded",
        "safra_soja_encoded", "safra_milho_encoded", "safra_cana_encoded",
    ]
    lags    = [f"lag_{l}" for l in [1, 2, 3, 4, 5, 6, 7, 14, 28]]
    rolling = (
        [f"rolling_mean_{w}" for w in [7, 14, 28]] +
        [f"rolling_std_{w}"  for w in [7, 14, 28]] +
        [f"rolling_max_{w}"  for w in [7, 28]] +
        [f"rolling_min_{w}"  for w in [7, 28]]
    )
    ewm = [f"ewm_{s}" for s in [7, 14, 28]]

    return calendar + climate + categorical + lags + rolling + ewm


def get_exogenous_feature_columns() -> list[str]:
    """Features exogenas para ARIMA e Prophet.

    Exclui lags e estatisticas autoregressivas: esses modelos ja
    capturam autocorrelacao internamente (parametros p/q no ARIMA,
    componentes de tendencia/sazonalidade no Prophet). Passar lags
    como regressores causa multicolinearidade severa.

    Usa apenas variaveis verdadeiramente exogenas (externas a serie).

    Total: ~16 features
    """
    return [
        # Calendario
        "day_of_week", "month", "year",
        "is_weekend", "week_of_year", "quarter",
        "day_of_year", "is_month_start", "is_month_end",
        # Clima (exogeno real — independente da demanda)
        "temperature", "rainfall", "humidity",
        # Sazonalidade / safra
        "season_encoded",
        "safra_soja_encoded", "safra_milho_encoded", "safra_cana_encoded",
    ]


def get_feature_columns() -> list[str]:
    """Alias para compatibilidade — retorna features ML completas."""
    return get_ml_feature_columns()
