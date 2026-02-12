"""Agregacao de series por cluster e rateio proporcional."""

import numpy as np
import pandas as pd


def aggregate_cluster_demand(
    df: pd.DataFrame, sku_ids: list[str], labels: np.ndarray
) -> dict[int, pd.DataFrame]:
    """Agrega demanda por cluster (soma diaria).

    Returns:
        Dict {cluster_id: DataFrame com colunas [date, demand_agg, + features]}
    """
    cluster_map = dict(zip(sku_ids, labels))
    df_temp = df[df["sku_id"].isin(sku_ids)].copy()
    df_temp["cluster"] = df_temp["sku_id"].map(cluster_map)

    cluster_data = {}
    for cluster_id in sorted(set(labels)):
        cluster_df = df_temp[df_temp["cluster"] == cluster_id]

        agg = cluster_df.groupby("date").agg(
            demand_agg=("demand", "sum"),
            temperature=("temperature", "first"),
            rainfall=("rainfall", "first"),
            humidity=("humidity", "first"),
            day_of_week=("day_of_week", "first"),
            month=("month", "first"),
            year=("year", "first"),
        ).reset_index()

        agg = agg.sort_values("date").reset_index(drop=True)
        cluster_data[cluster_id] = agg

    return cluster_data


def compute_disaggregation_weights(
    df: pd.DataFrame, sku_ids: list[str], labels: np.ndarray,
    method: str = "rolling", window: int = 28,
) -> dict[int, dict[str, float]]:
    """Calcula pesos de rateio proporcional por SKU dentro de cada cluster.

    Args:
        method: 'static' (proporcao historica total) ou 'rolling' (ultimos N dias).
        window: Janela em dias para metodo rolling.

    Returns:
        Dict {cluster_id: {sku_id: peso}}
    """
    cluster_map = dict(zip(sku_ids, labels))
    df_temp = df[df["sku_id"].isin(sku_ids)].copy()
    df_temp["cluster"] = df_temp["sku_id"].map(cluster_map)

    weights = {}
    for cluster_id in sorted(set(labels)):
        cluster_df = df_temp[df_temp["cluster"] == cluster_id]
        skus_in_cluster = cluster_df["sku_id"].unique()

        if method == "static":
            totals = cluster_df.groupby("sku_id")["demand"].sum()
            cluster_total = totals.sum()
            if cluster_total == 0:
                w = {sku: 1.0 / len(skus_in_cluster) for sku in skus_in_cluster}
            else:
                w = (totals / cluster_total).to_dict()

        elif method == "rolling":
            max_date = cluster_df["date"].max()
            cutoff = max_date - pd.Timedelta(days=window)
            recent = cluster_df[cluster_df["date"] > cutoff]
            totals = recent.groupby("sku_id")["demand"].sum()
            cluster_total = totals.sum()
            if cluster_total == 0:
                w = {sku: 1.0 / len(skus_in_cluster) for sku in skus_in_cluster}
            else:
                w = (totals / cluster_total).to_dict()
        else:
            raise ValueError(f"Metodo '{method}' nao suportado. Use 'static' ou 'rolling'.")

        weights[cluster_id] = w

    return weights


def disaggregate_forecast(
    forecast_agg: pd.DataFrame,
    weights: dict[str, float],
) -> dict[str, pd.DataFrame]:
    """Desagrega forecast agregado para SKUs individuais via rateio.

    Args:
        forecast_agg: DataFrame com colunas [ds, yhat, yhat_lower, yhat_upper].
        weights: {sku_id: peso} para o cluster.

    Returns:
        Dict {sku_id: DataFrame [ds, yhat, yhat_lower, yhat_upper]}
    """
    result = {}
    for sku_id, weight in weights.items():
        sku_forecast = forecast_agg.copy()
        sku_forecast["yhat"] = np.maximum(0, sku_forecast["yhat"] * weight)
        sku_forecast["yhat_lower"] = np.maximum(0, sku_forecast["yhat_lower"] * weight)
        sku_forecast["yhat_upper"] = sku_forecast["yhat_upper"] * weight
        result[sku_id] = sku_forecast

    return result
