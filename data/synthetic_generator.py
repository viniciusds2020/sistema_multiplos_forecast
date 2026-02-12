"""Geracao de dados sinteticos para o sistema de forecast multi-produto."""

import numpy as np
import pandas as pd
from config import (
    NUM_SKUS, DATE_START, DATE_END, SKU_CATEGORIES,
    SAFRA_CALENDARS, RANDOM_SEED,
)


def _get_season(month: int) -> str:
    """Retorna a estacao do ano (hemisferio sul)."""
    if month in (12, 1, 2):
        return "verao"
    elif month in (3, 4, 5):
        return "outono"
    elif month in (6, 7, 8):
        return "inverno"
    else:
        return "primavera"


def _get_safra_phase(month: int, crop: str) -> str:
    """Retorna a fase da safra para um mes e cultura."""
    cal = SAFRA_CALENDARS[crop]
    for phase, (start, end) in cal.items():
        if start <= end:
            if start <= month <= end:
                return phase
        else:
            if month >= start or month <= end:
                return phase
    return "entressafra"


def _generate_climate(dates: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    """Gera dados climaticos sinteticos (Ribeirao Preto, SP)."""
    n = len(dates)
    day_of_year = dates.dayofyear.values.astype(float)

    # Temperatura: sinusoidal com pico em janeiro (hemisferio sul)
    temperature = 24.0 + 6.0 * np.sin(2 * np.pi * (day_of_year - 30) / 365.25)
    temperature += rng.normal(0, 2.0, n)
    temperature = np.clip(temperature, 5, 42)

    # Chuva: sazonal - mais chuva no verao (nov-mar)
    month = dates.month.values
    lambda_rain = np.where(
        (month >= 10) | (month <= 3), 0.65, 0.20
    )
    rain_occurs = rng.binomial(1, lambda_rain)
    rain_amount = rng.gamma(2.0, 8.0, n)
    rainfall = rain_occurs * rain_amount
    rainfall = np.round(rainfall, 1)

    # Umidade: correlacionada com chuva e temperatura
    humidity = 60.0 + 0.15 * rainfall - 0.5 * (temperature - 24.0)
    humidity += rng.normal(0, 5.0, n)
    humidity = np.clip(humidity, 20, 100)

    return pd.DataFrame({
        "temperature": np.round(temperature, 1),
        "rainfall": rainfall,
        "humidity": np.round(humidity, 1),
    })


def _generate_sku_demand(
    dates: pd.DatetimeIndex,
    sku_name: str,
    profile: str,
    rng: np.random.Generator,
    climate: pd.DataFrame,
) -> np.ndarray:
    """Gera serie de demanda para um SKU especifico."""
    n = len(dates)
    t = np.arange(n, dtype=float)
    day_of_year = dates.dayofyear.values.astype(float)
    month = dates.month.values

    if profile == "estavel":
        baseline = rng.lognormal(4.3, 0.15)  # ~80
        trend = 0.0
        season_amp = baseline * 0.10
        noise_std = baseline * 0.12

    elif profile == "sazonal":
        baseline = rng.lognormal(4.0, 0.2)  # ~55
        trend = 0.0
        season_amp = baseline * 0.60
        noise_std = baseline * 0.15

    elif profile == "tendencia_alta":
        baseline = rng.lognormal(3.5, 0.2)  # ~33
        trend = rng.uniform(0.05, 0.15)
        season_amp = baseline * 0.15
        noise_std = baseline * 0.18

    elif profile == "tendencia_baixa":
        baseline = rng.lognormal(4.5, 0.2)  # ~90
        trend = rng.uniform(-0.12, -0.04)
        season_amp = baseline * 0.10
        noise_std = baseline * 0.15

    elif profile == "intermitente":
        baseline = rng.lognormal(3.0, 0.3)  # ~20
        trend = 0.0
        season_amp = baseline * 0.05
        noise_std = baseline * 0.25
    else:
        baseline = 50.0
        trend = 0.0
        season_amp = 5.0
        noise_std = 8.0

    # Componentes
    trend_component = trend * t
    seasonality = season_amp * np.sin(2 * np.pi * (day_of_year - 30) / 365.25)
    seasonality += (season_amp * 0.3) * np.sin(2 * np.pi * day_of_year / 182.625)

    # Efeito safra (leve boost durante colheita)
    crops = list(SAFRA_CALENDARS.keys())
    crop = rng.choice(crops)
    safra_effect = np.zeros(n)
    for i, m in enumerate(month):
        phase = _get_safra_phase(m, crop)
        if phase == "colheita":
            safra_effect[i] = baseline * 0.08
        elif phase == "plantio":
            safra_effect[i] = baseline * 0.04

    # Efeito clima
    climate_effect = (
        0.3 * (climate["temperature"].values - 24.0)
        + 0.1 * (climate["rainfall"].values - 5.0)
    )

    # Ruido
    noise = rng.normal(0, noise_std, n)

    # Spikes de promocao (2% dos dias)
    promo_mask = rng.binomial(1, 0.02, n)
    promo_effect = promo_mask * baseline * rng.uniform(0.5, 1.5, n)

    # Composicao
    demand = baseline + trend_component + seasonality + safra_effect + climate_effect + noise + promo_effect

    # Intermitente: mascara Bernoulli para zeros
    if profile == "intermitente":
        arrival_prob = rng.uniform(0.25, 0.45)
        mask = rng.binomial(1, arrival_prob, n)
        demand = demand * mask

    demand = np.maximum(0, np.round(demand)).astype(int)
    return demand


def generate_synthetic_data(
    num_skus: int = NUM_SKUS,
    date_start=DATE_START,
    date_end=DATE_END,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Gera dataset sintetico completo com multiplos SKUs.

    Returns:
        DataFrame com colunas: date, sku_id, sku_name, demand, temperature,
        rainfall, humidity, day_of_week, month, year, season,
        safra_soja, safra_milho, safra_cana, demand_profile
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=date_start, end=date_end, freq="D")
    climate = _generate_climate(dates, rng)

    # Montar lista de SKUs
    all_skus = []
    for profile, names in SKU_CATEGORIES.items():
        for name in names:
            all_skus.append((name, profile))

    # Limitar ao numero desejado
    all_skus = all_skus[:num_skus]

    records = []
    for idx, (sku_name, profile) in enumerate(all_skus):
        demand = _generate_sku_demand(dates, sku_name, profile, rng, climate)

        for i, date in enumerate(dates):
            m = date.month
            records.append({
                "date": date,
                "sku_id": f"SKU_{idx+1:03d}",
                "sku_name": sku_name,
                "demand": demand[i],
                "temperature": climate.iloc[i]["temperature"],
                "rainfall": climate.iloc[i]["rainfall"],
                "humidity": climate.iloc[i]["humidity"],
                "day_of_week": date.dayofweek,
                "month": m,
                "year": date.year,
                "season": _get_season(m),
                "safra_soja": _get_safra_phase(m, "soja"),
                "safra_milho": _get_safra_phase(m, "milho_safrinha"),
                "safra_cana": _get_safra_phase(m, "cana"),
                "demand_profile": profile,
            })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df


if __name__ == "__main__":
    df = generate_synthetic_data()
    print(f"Shape: {df.shape}")
    print(f"SKUs: {df['sku_id'].nunique()}")
    print(f"Periodo: {df['date'].min()} a {df['date'].max()}")
    print(f"\nPerfis:\n{df.groupby('demand_profile')['sku_id'].nunique()}")
    print(f"\nAmostra:\n{df.head(10)}")
