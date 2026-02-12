"""Pipeline de forecast agregado por cluster com rateio proporcional."""

import pandas as pd
import numpy as np
from data.feature_engineering import prepare_ml_features, get_feature_columns, add_lag_features, add_rolling_features
from models.model_registry import get_model
from evaluation.metrics import calculate_all_metrics
from similarity.clustering import (
    build_demand_matrix, compute_distance_matrix,
    find_optimal_clusters, cluster_series, get_cluster_summary,
)
from similarity.aggregation import (
    aggregate_cluster_demand, compute_disaggregation_weights,
    disaggregate_forecast,
)


def run_cluster_analysis(
    df: pd.DataFrame, metric: str = "dtw", n_clusters: int = None,
) -> dict:
    """Executa analise de similaridade e clustering.

    Returns:
        Dict com: matrix, sku_ids, dist_matrix, labels, linkage, n_clusters,
        silhouette_scores, cluster_summary
    """
    matrix, sku_ids = build_demand_matrix(df)
    dist_matrix = compute_distance_matrix(matrix, metric=metric)

    if n_clusters is None:
        best_k, scores = find_optimal_clusters(matrix, dist_matrix)
        n_clusters = best_k
    else:
        _, scores = find_optimal_clusters(matrix, dist_matrix)

    labels, Z = cluster_series(dist_matrix, n_clusters)
    summary = get_cluster_summary(df, sku_ids, labels)

    return {
        "matrix": matrix,
        "sku_ids": sku_ids,
        "dist_matrix": dist_matrix,
        "labels": labels,
        "linkage": Z,
        "n_clusters": n_clusters,
        "silhouette_scores": scores,
        "cluster_summary": summary,
    }


def _prepare_agg_features(agg_df: pd.DataFrame) -> pd.DataFrame:
    """Prepara features para serie agregada."""
    agg_df = agg_df.copy()
    agg_df = agg_df.rename(columns={"demand_agg": "demand"})
    agg_df["sku_id"] = "agg"

    # Adicionar features encodadas
    season_map = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    agg_df["season_encoded"] = agg_df["month"].map(season_map).fillna(0).astype(int)
    agg_df["safra_soja_encoded"] = 0
    agg_df["safra_milho_encoded"] = 0
    agg_df["safra_cana_encoded"] = 0

    agg_df = add_lag_features(agg_df)
    agg_df = add_rolling_features(agg_df)
    return agg_df


def run_cluster_forecast_pipeline(
    df: pd.DataFrame,
    cluster_info: dict,
    model_names: list[str],
    test_days: int = 60,
    horizon: int = 30,
    weight_method: str = "rolling",
    progress_callback=None,
) -> dict:
    """Executa pipeline de forecast agregado por cluster com rateio.

    Returns:
        Dict com:
            cluster_forecasts: {cluster_id: {model_name: forecast_df}}
            disaggregated: {cluster_id: {model_name: {sku_id: forecast_df}}}
            weights: {cluster_id: {sku_id: peso}}
            metrics_agg: {cluster_id: {model_name: metrics}}
            metrics_disagg: {cluster_id: {model_name: {sku_id: metrics}}}
    """
    sku_ids = cluster_info["sku_ids"]
    labels = cluster_info["labels"]

    # Agregar demanda por cluster
    cluster_data = aggregate_cluster_demand(df, sku_ids, labels)
    weights = compute_disaggregation_weights(df, sku_ids, labels, method=weight_method)

    cluster_forecasts = {}
    disaggregated = {}
    metrics_agg = {}
    metrics_disagg = {}

    total_clusters = len(cluster_data)
    feature_cols = get_feature_columns()

    for idx, (cluster_id, agg_df) in enumerate(cluster_data.items()):
        if progress_callback:
            progress_callback(
                idx / total_clusters,
                f"Cluster {cluster_id}/{total_clusters}..."
            )

        # Preparar features para serie agregada
        agg_prepared = _prepare_agg_features(agg_df)

        # Split treino/teste
        agg_prepared = agg_prepared.sort_values("date").reset_index(drop=True)
        split_idx = len(agg_prepared) - test_days
        train = agg_prepared.iloc[:split_idx]
        test = agg_prepared.iloc[split_idx:]

        cluster_forecasts[cluster_id] = {}
        disaggregated[cluster_id] = {}
        metrics_agg[cluster_id] = {}
        metrics_disagg[cluster_id] = {}

        for model_name in model_names:
            try:
                model = get_model(model_name)

                y_train = pd.Series(
                    train["demand"].values,
                    index=pd.DatetimeIndex(train["date"]),
                )

                X_train = None
                X_future = None
                if model.supports_exogenous:
                    available = [c for c in feature_cols if c in train.columns]
                    if available:
                        X_train = train[available].reset_index(drop=True)
                        X_future = test[available].iloc[:horizon].reset_index(drop=True)

                model.fit(y_train, X_train)
                forecast = model.predict(horizon, X_future)

                cluster_forecasts[cluster_id][model_name] = forecast

                # Metricas agregadas
                y_test_agg = test["demand"].values[:horizon]
                y_pred_agg = forecast["yhat"].values[:len(y_test_agg)]
                metrics_agg[cluster_id][model_name] = calculate_all_metrics(
                    y_test_agg, y_pred_agg
                )

                # Desagregar (rateio)
                sku_forecasts = disaggregate_forecast(
                    forecast, weights[cluster_id]
                )
                disaggregated[cluster_id][model_name] = sku_forecasts

                # Metricas desagregadas por SKU
                metrics_disagg[cluster_id][model_name] = {}
                for sku_id, sku_fc in sku_forecasts.items():
                    sku_test = df[
                        (df["sku_id"] == sku_id) &
                        (df["date"] >= test["date"].iloc[0])
                    ]["demand"].values[:horizon]
                    if len(sku_test) > 0:
                        sku_pred = sku_fc["yhat"].values[:len(sku_test)]
                        metrics_disagg[cluster_id][model_name][sku_id] = (
                            calculate_all_metrics(sku_test, sku_pred)
                        )

            except Exception as e:
                cluster_forecasts[cluster_id][model_name] = None
                metrics_agg[cluster_id][model_name] = {"error": str(e)}

    if progress_callback:
        progress_callback(1.0, "Concluido!")

    return {
        "cluster_forecasts": cluster_forecasts,
        "disaggregated": disaggregated,
        "weights": weights,
        "metrics_agg": metrics_agg,
        "metrics_disagg": metrics_disagg,
    }
