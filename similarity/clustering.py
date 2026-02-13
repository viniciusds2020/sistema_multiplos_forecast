"""Analise de similaridade e clustering de series temporais."""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler


def build_demand_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Constroi matriz SKU x tempo de demanda.

    Returns:
        (matrix, sku_ids) onde matrix tem shape (n_skus, n_timesteps)
    """
    pivot = df.pivot_table(
        index="sku_id", columns="date", values="demand", aggfunc="sum"
    ).fillna(0)
    return pivot.values, list(pivot.index)


def compute_distance_matrix(
    matrix: np.ndarray, metric: str = "dtw"
) -> np.ndarray:
    """Computa matriz de distancia entre series temporais.

    Args:
        matrix: Shape (n_skus, n_timesteps).
        metric: 'dtw', 'pearson', ou 'euclidean'.

    Returns:
        Matriz de distancia (n_skus, n_skus).
    """
    n = matrix.shape[0]
    if n < 2:
        raise ValueError(f"Necessario ao menos 2 SKUs para calcular distancia, mas encontrou {n}.")

    # Z-normalizar
    scaler = StandardScaler()
    normalized = scaler.fit_transform(matrix.T).T  # normaliza cada serie

    if metric == "dtw":
        try:
            from tslearn.metrics import dtw as ts_dtw
            dist = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    d = ts_dtw(normalized[i], normalized[j])
                    dist[i, j] = d
                    dist[j, i] = d
        except ImportError:
            # Fallback para euclidean
            from scipy.spatial.distance import cdist
            dist = cdist(normalized, normalized, metric="euclidean")

    elif metric == "pearson":
        corr = np.corrcoef(normalized)
        dist = 1.0 - corr
        dist = np.clip(dist, 0, 2)

    elif metric == "euclidean":
        from scipy.spatial.distance import cdist
        dist = cdist(normalized, normalized, metric="euclidean")
    else:
        raise ValueError(f"Metrica '{metric}' nao suportada.")

    np.fill_diagonal(dist, 0)
    return dist


def find_optimal_clusters(
    matrix: np.ndarray, dist_matrix: np.ndarray,
    min_k: int = 2, max_k: int = 8,
) -> tuple[int, dict[int, float]]:
    """Encontra numero otimo de clusters via silhouette score.

    Returns:
        (best_k, scores_dict) onde scores_dict mapeia k -> silhouette_score.
    """
    from sklearn.metrics import silhouette_score

    scores = {}
    for k in range(min_k, min(max_k + 1, matrix.shape[0])):
        try:
            condensed = squareform(dist_matrix, checks=False)
            Z = linkage(condensed, method="ward")
            labels = fcluster(Z, t=k, criterion="maxclust")
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(dist_matrix, labels, metric="precomputed")
            scores[k] = round(score, 4)
        except Exception:
            continue

    if not scores:
        return 2, {2: 0.0}

    best_k = max(scores, key=scores.get)
    return best_k, scores


def cluster_series(
    dist_matrix: np.ndarray, n_clusters: int
) -> tuple[np.ndarray, np.ndarray]:
    """Clusteriza series usando hierarchical clustering.

    Returns:
        (labels, linkage_matrix)
    """
    condensed = squareform(dist_matrix, checks=False)
    Z = linkage(condensed, method="ward")
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    return labels, Z


def compute_mds_projection(dist_matrix: np.ndarray) -> np.ndarray:
    """Projeta distancias em 2D via MDS para visualizacao.

    Returns:
        Coordenadas (n_skus, 2).
    """
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, normalized_stress="auto")
    coords = mds.fit_transform(dist_matrix)
    return coords


def get_cluster_summary(
    df: pd.DataFrame, sku_ids: list[str], labels: np.ndarray
) -> pd.DataFrame:
    """Resume informacoes por cluster."""
    cluster_map = dict(zip(sku_ids, labels))
    df_temp = df.copy()
    df_temp["cluster"] = df_temp["sku_id"].map(cluster_map)

    summary = df_temp.groupby(["cluster", "sku_id", "sku_name", "demand_profile"]).agg(
        mean_demand=("demand", "mean"),
        std_demand=("demand", "std"),
        total_demand=("demand", "sum"),
        zero_pct=("demand", lambda x: (x == 0).mean() * 100),
    ).reset_index()

    summary["cv"] = (summary["std_demand"] / summary["mean_demand"].replace(0, np.nan)).round(3)
    # Arredondar colunas numericas para evitar problemas de serializacao
    for col in ["mean_demand", "std_demand", "total_demand", "zero_pct"]:
        if col in summary.columns:
            summary[col] = summary[col].round(2)
    return summary
