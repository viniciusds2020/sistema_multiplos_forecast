// General
export interface KpisResponse {
  total_skus: number;
  total_records: number;
  coverage_days: number;
  avg_demand: number;
  date_start: string;
  date_end: string;
}

export interface SkuInfo {
  sku_id: string;
  sku_name: string;
  demand_profile: string;
  mean_demand: number;
  total_demand: number;
}

export interface ModelsResponse {
  all: string[];
  available: string[];
}

// Overview
export interface OverviewCharts {
  heatmap: string;
  top_skus: string;
  total_demand: string;
  profiles: string;
  stats_table: StatsRow[];
}

export interface StatsRow {
  demand_profile: string;
  skus: number;
  media: number;
  desvio: number;
  maximo: number;
  zero_pct: number;
}

// Explorer
export interface DemandResponse {
  chart: string;
}

export interface ClimateResponse {
  temperature: string;
  rainfall: string;
  humidity: string;
}

export interface DistributionsResponse {
  season: string;
  dow: string;
  safra: string;
  profile: string;
}

// Similarity
export interface SimilarityRequest {
  metric: string;
  auto_k: boolean;
  manual_k: number;
}

export interface SimilarityResponse {
  charts: {
    dist_heatmap: string;
    silhouette: string;
    mds: string;
    dendro: string;
    cluster_series: Record<string, string>;
  };
  n_clusters: number;
  sil_best: number;
  total_skus: number;
  summary: ClusterSummaryRow[];
  error?: string;
}

export interface ClusterSummaryRow {
  cluster: number;
  sku_id: string;
  sku_name: string;
  demand_profile: string;
  mean_demand: number;
  cv: number;
  zero_pct: number;
}

// Forecasting
export interface IndividualForecastRequest {
  sku_id: string;
  models: string[];
  horizon: number;
  test_days: number;
}

export interface IndividualForecastResponse {
  overlay_chart: string;
  wape_chart: string;
  detail_charts: Record<string, string>;
  metrics: MetricsRow[];
  params: Record<string, Record<string, unknown>>;
  errors: Record<string, string>;
  error?: string;
}

export interface MetricsRow {
  Model: string;
  MAE: number;
  RMSE: number;
  MAPE: number;
  WAPE: number;
  Bias: number;
}

export interface AggregatedForecastRequest {
  models: string[];
  horizon: number;
  test_days: number;
  metric: string;
  weight_method: string;
}

export interface AggregatedForecastResponse {
  charts: Record<string, string>;
  weights: Record<string, WeightEntry[]>;
  metrics: Record<string, MetricsRow[]>;
  n_clusters: number;
  error?: string;
}

export interface WeightEntry {
  sku: string;
  peso: number;
  pct: number;
}

// Comparison
export interface ComparisonRequest {
  models: string[];
  sku_ids: string[];
  horizon: number;
  test_days: number;
}

export interface ComparisonMetricRow extends MetricsRow {
  SKU: string;
  Nome: string;
  Perfil: string;
}

export interface ComparisonAvgRow {
  Model: string;
  MAE: number;
  RMSE: number;
  MAPE: number;
  WAPE: number;
}

export interface ComparisonResponse {
  metrics: ComparisonMetricRow[];
  avg: ComparisonAvgRow[];
  charts: {
    radar: string;
    box_wape: string;
    grouped_bar: string;
    best_count: string;
  };
  sku_charts: Record<string, string>;
  error?: string;
}
