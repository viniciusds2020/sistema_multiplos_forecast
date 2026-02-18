"""
Sistema de Forecast Multi-Produto SKU
=====================================

Orquestrador principal que integra:
- Geracao de dados sinteticos (20 SKUs, 2.5 anos)
- Modelos: XGBoost, LightGBM, AutoARIMA, Prophet, Chronos, CrostonSBA
- Analise de similaridade (DTW, Pearson, Euclidean)
- Clustering de series temporais
- Forecast agregado por cluster com rateio proporcional
- Dashboard interativo (Flask)

Uso:
    python app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from data.synthetic_generator import generate_synthetic_data
from models.model_registry import list_models
from pipeline.forecasting_pipeline import run_forecast_pipeline
from pipeline.cluster_pipeline import run_cluster_analysis, run_cluster_forecast_pipeline


def run_demo():
    """Executa demonstracao completa do sistema via CLI."""
    print("=" * 70)
    print("  SISTEMA DE FORECAST MULTI-PRODUTO SKU")
    print("=" * 70)

    # 1. Gerar dados
    print("\n[1/6] Gerando dados sinteticos...")
    df = generate_synthetic_data()
    print(f"  -> {df.shape[0]:,} registros | {df['sku_id'].nunique()} SKUs")
    print(f"  -> Periodo: {df['date'].min().date()} a {df['date'].max().date()}")
    print(f"  -> Perfis: {dict(df.groupby('demand_profile')['sku_id'].nunique())}")

    # 2. Analise de similaridade
    print("\n[2/6] Executando analise de similaridade...")
    cluster_info = run_cluster_analysis(df, metric="pearson")
    n_clusters = cluster_info["n_clusters"]
    labels = cluster_info["labels"]
    labels_min = int(np.min(labels)) if len(labels) else 0
    print(f"  -> Clusters encontrados: {n_clusters}")
    print(f"  -> Silhouette scores: {cluster_info['silhouette_scores']}")

    # Mostrar composicao dos clusters
    for c in range(n_clusters):
        label_val = c + 1 if labels_min == 1 else c
        mask = labels == label_val
        skus = [cluster_info["sku_ids"][i] for i in np.where(mask)[0]]
        print(f"  -> Cluster {label_val}: {skus}")

    # 3. Forecast individual (amostra)
    print("\n[3/6] Executando forecast individual (3 SKUs de amostra)...")
    sample_skus = df["sku_id"].unique()[:3]
    models_to_use = ["XGBoost", "LightGBM", "AutoARIMA"]

    for sku_id in sample_skus:
        print(f"\n  SKU: {sku_id}")
        results = run_forecast_pipeline(df, sku_id, models_to_use, test_days=60, horizon=30)
        for model_name, res in results.items():
            if res["metrics"]:
                wape = res["metrics"].get("WAPE", "N/A")
                mae = res["metrics"].get("MAE", "N/A")
                wape_str = f"{float(wape):.2f}" if isinstance(wape, (int, float, np.floating)) else str(wape)
                mae_str = f"{float(mae):.2f}" if isinstance(mae, (int, float, np.floating)) else str(mae)
                print(f"    {model_name:15s} | WAPE: {wape_str:>8} | MAE: {mae_str:>8}")
            elif res["error"]:
                print(f"    {model_name:15s} | ERRO: {res['error'][:50]}")

    # 4. Forecast agregado por cluster
    print("\n[4/6] Executando forecast agregado por cluster...")
    cluster_results = run_cluster_forecast_pipeline(
        df, cluster_info, models_to_use,
        test_days=60, horizon=30, weight_method="rolling",
    )

    for cluster_id in sorted(cluster_results["metrics_agg"].keys()):
        print(f"\n  Cluster {cluster_id}:")
        for model_name, metrics in cluster_results["metrics_agg"][cluster_id].items():
            if "error" not in metrics:
                wape = metrics.get("WAPE", "N/A")
                mae = metrics.get("MAE", "N/A")
                wape_str = f"{float(wape):.2f}" if isinstance(wape, (int, float, np.floating)) else str(wape)
                mae_str = f"{float(mae):.2f}" if isinstance(mae, (int, float, np.floating)) else str(mae)
                print(f"    {model_name:15s} | WAPE: {wape_str:>8} | MAE: {mae_str:>8}")

    # 5. Pesos de rateio
    print("\n[5/6] Pesos de rateio proporcional:")
    for cluster_id, weights in cluster_results["weights"].items():
        print(f"\n  Cluster {cluster_id}:")
        for sku_id, w in sorted(weights.items(), key=lambda x: -x[1]):
            print(f"    {sku_id}: {w:.4f} ({w*100:.1f}%)")

    # 6. Resumo
    print("\n[6/6] Resumo Final")
    print(f"  -> Modelos disponveis: {list_models()}")
    print(f"  -> SKUs processados: {df['sku_id'].nunique()}")
    print(f"  -> Clusters: {n_clusters}")
    print(f"\nPara o dashboard interativo, execute:")
    print(f"  python app.py")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
