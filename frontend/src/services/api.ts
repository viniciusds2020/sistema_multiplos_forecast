import axios from 'axios';
import type {
  KpisResponse, SkuInfo, ModelsResponse, OverviewCharts,
  DemandResponse, ClimateResponse, DistributionsResponse,
  SimilarityRequest, SimilarityResponse,
  IndividualForecastRequest, IndividualForecastResponse,
  AggregatedForecastRequest, AggregatedForecastResponse,
  ComparisonRequest, ComparisonResponse,
} from '../types/api';

const api = axios.create({
  baseURL: '/api',
  timeout: 300000,
});

// General
export const fetchModels = () => api.get<ModelsResponse>('/models').then(r => r.data);
export const fetchKpis = () => api.get<KpisResponse>('/kpis').then(r => r.data);
export const fetchSkus = () => api.get<SkuInfo[]>('/skus').then(r => r.data);
export const clearCache = () => api.post('/cache/clear');

// Overview
export const fetchOverviewCharts = () => api.get<OverviewCharts>('/overview/charts').then(r => r.data);

// Explorer
export const fetchExplorerDemand = (skus: string[]) => {
  const params = new URLSearchParams();
  skus.forEach(s => params.append('skus', s));
  return api.get<DemandResponse>('/explorer/demand', { params }).then(r => r.data);
};
export const fetchExplorerClimate = () => api.get<ClimateResponse>('/explorer/climate').then(r => r.data);
export const fetchExplorerDistributions = () => api.get<DistributionsResponse>('/explorer/distributions').then(r => r.data);
export const fetchExplorerRaw = () => api.get<Record<string, unknown>[]>('/explorer/raw').then(r => r.data);

// Similarity
export const runSimilarity = (body: SimilarityRequest) =>
  api.post<SimilarityResponse>('/similarity/run', body).then(r => r.data);

// Forecasting
export const runIndividualForecast = (body: IndividualForecastRequest) =>
  api.post<IndividualForecastResponse>('/forecast/individual', body).then(r => r.data);

export const runAggregatedForecast = (body: AggregatedForecastRequest) =>
  api.post<AggregatedForecastResponse>('/forecast/aggregated', body).then(r => r.data);

// Comparison
export const runComparison = (body: ComparisonRequest) =>
  api.post<ComparisonResponse>('/comparison/run', body).then(r => r.data);
