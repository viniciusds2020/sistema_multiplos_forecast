import { useState, useEffect, useMemo } from 'react';
import Select from 'react-select';
import PageHeader from '../components/common/PageHeader';
import KpiCard from '../components/common/KpiCard';
import PlotlyChart from '../components/common/PlotlyChart';
import MetricsTable from '../components/common/MetricsTable';
import ModelCheckboxes from '../components/common/ModelCheckboxes';
import TabGroup from '../components/common/TabGroup';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorMessage from '../components/common/ErrorMessage';
import GradientButton from '../components/common/GradientButton';
import { useModels } from '../hooks/useModels';
import { useSkus } from '../hooks/useSkus';
import { useIndividualForecast, useAggregatedForecast } from '../hooks/useForecast';
import type { IndividualForecastResponse, AggregatedForecastResponse } from '../types/api';

const MODE_TABS = [
  { key: 'individual', label: 'Individual (por SKU)' },
  { key: 'aggregated', label: 'Agregado (Cluster)' },
];

export default function ForecastingPage() {
  const modelsQuery = useModels();
  const skusQuery = useSkus();

  const allModels = modelsQuery.data?.all ?? ['XGBoost', 'LightGBM', 'AutoARIMA', 'Prophet', 'CrostonSBA'];
  const availModels = modelsQuery.data?.available ?? allModels;

  const [mode, setMode] = useState('individual');

  // Individual state
  const [skuId, setSkuId] = useState('');
  const [indModels, setIndModels] = useState<string[]>([]);
  const [horizon, setHorizon] = useState(30);
  const [testDays, setTestDays] = useState(60);

  // Aggregated state
  const [aggModels, setAggModels] = useState<string[]>([]);
  const [aggMetric, setAggMetric] = useState('pearson');
  const [weightMethod, setWeightMethod] = useState('rolling');

  const indMutation = useIndividualForecast();
  const aggMutation = useAggregatedForecast();

  const indData = indMutation.data as IndividualForecastResponse | undefined;
  const aggData = aggMutation.data as AggregatedForecastResponse | undefined;

  // Initialize model selection when available models load
  useEffect(() => {
    if (availModels.length && indModels.length === 0) {
      const init = availModels.slice(0, 3);
      setIndModels(init);
      setAggModels(init);
    }
  }, [availModels]);

  // SKU options
  const skuOptions = useMemo(() => {
    if (!skusQuery.data) return [];
    return skusQuery.data.map(s => ({
      value: s.sku_id,
      label: `${s.sku_name} (${s.demand_profile})`,
    }));
  }, [skusQuery.data]);

  useEffect(() => {
    if (skuOptions.length && !skuId) setSkuId(skuOptions[0].value);
  }, [skuOptions]);

  const runIndividual = () => {
    if (!skuId || !indModels.length) return;
    indMutation.mutate({ sku_id: skuId, models: indModels, horizon, test_days: testDays });
  };

  const runAggregated = () => {
    if (!aggModels.length) return;
    aggMutation.mutate({
      models: aggModels, horizon: 30, test_days: 60,
      metric: aggMetric, weight_method: weightMethod,
    });
  };

  // Best WAPE for individual
  const bestWape = indData?.metrics?.length ? Math.min(...indData.metrics.map(m => m.WAPE)) : null;
  const bestModel = bestWape !== null ? indData?.metrics.find(m => m.WAPE === bestWape)?.Model : '-';

  return (
    <div className="fade-in">
      <PageHeader title="Previsão" subtitle="Treine modelos e gere previsões por SKU individual ou por cluster agregado" />

      <div className="max-w-md">
        <TabGroup tabs={MODE_TABS} active={mode} onChange={setMode} />
      </div>

      {/* ========== INDIVIDUAL MODE ========== */}
      {mode === 'individual' && (
        <>
          <div className="card p-4 mb-6">
            <div className="flex flex-wrap items-end gap-4">
              <div>
                <label className="text-xs font-semibold text-gray-400 uppercase tracking-wider">SKU</label>
                <div className="mt-1 w-56">
                  <Select
                    options={skuOptions}
                    value={skuOptions.find(o => o.value === skuId)}
                    onChange={(o) => o && setSkuId(o.value)}
                    placeholder="Buscar SKU..."
                    className="text-sm"
                  />
                </div>
              </div>
              <div>
                <label className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Modelos</label>
                <div className="mt-1">
                  <ModelCheckboxes models={allModels} available={availModels} selected={indModels} onChange={setIndModels} />
                </div>
              </div>
              <div>
                <label className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Horizonte</label>
                <input type="number" value={horizon} min={7} max={90} onChange={e => setHorizon(+e.target.value)}
                  className="mt-1 block w-20 rounded-lg border border-gray-200 px-2 py-2 text-sm text-center" />
              </div>
              <div>
                <label className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Teste (dias)</label>
                <input type="number" value={testDays} min={30} max={120} onChange={e => setTestDays(+e.target.value)}
                  className="mt-1 block w-20 rounded-lg border border-gray-200 px-2 py-2 text-sm text-center" />
              </div>
              <GradientButton onClick={runIndividual} disabled={indMutation.isPending}>
                Executar Previsão
              </GradientButton>
            </div>
          </div>

          {indMutation.isPending && <LoadingSpinner message="Treinando modelos..." />}
          {indData?.error && <ErrorMessage message={indData.error} />}

          {indData && !indData.error && (
            <>
              <div className="grid grid-cols-3 gap-4 mb-6">
                <KpiCard label="Modelos" value={indModels.length} color="#4361ee" />
                <KpiCard label="Melhor WAPE" value={bestWape !== null ? `${bestWape.toFixed(1)}%` : '-'} color="#06d6a0" />
                <KpiCard label="Melhor Modelo" value={bestModel ?? '-'} color="#118ab2" />
              </div>

              <div className="card p-5 mb-6">
                <h3 className="text-sm font-bold text-gray-700 mb-3 pb-2 border-b border-gray-100">Comparação de Previsões</h3>
                <PlotlyChart chartJson={indData.overlay_chart} />
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                <div className="card p-5">
                  <h3 className="text-sm font-bold text-gray-700 mb-3 pb-2 border-b border-gray-100">Métricas de Avaliação</h3>
                  <MetricsTable rows={indData.metrics} />
                </div>
                <div className="card p-5">
                  <h3 className="text-sm font-bold text-gray-700 mb-3 pb-2 border-b border-gray-100">WAPE por Modelo</h3>
                  <PlotlyChart chartJson={indData.wape_chart} />
                </div>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {Object.entries(indData.detail_charts || {}).map(([mname, chartJson]) => (
                  <div key={mname} className="card p-5">
                    <h3 className="text-sm font-bold text-gray-700 mb-3 pb-2 border-b border-gray-100">{mname}</h3>
                    <PlotlyChart chartJson={chartJson} />
                  </div>
                ))}
              </div>
            </>
          )}
        </>
      )}

      {/* ========== AGGREGATED MODE ========== */}
      {mode === 'aggregated' && (
        <>
          <div className="card p-4 mb-6">
            <div className="flex flex-wrap items-end gap-4">
              <div>
                <label className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Metrica de Similaridade</label>
                <select value={aggMetric} onChange={e => setAggMetric(e.target.value)}
                  className="mt-1 block w-36 rounded-lg border border-gray-200 px-3 py-2 text-sm">
                  <option value="pearson">Pearson</option>
                  <option value="euclidean">Euclidean</option>
                  <option value="dtw">DTW</option>
                </select>
              </div>
              <div>
                <label className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Rateio</label>
                <select value={weightMethod} onChange={e => setWeightMethod(e.target.value)}
                  className="mt-1 block w-32 rounded-lg border border-gray-200 px-3 py-2 text-sm">
                  <option value="rolling">Rolling</option>
                  <option value="static">Static</option>
                </select>
              </div>
              <div>
                <label className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Modelos</label>
                <div className="mt-1">
                  <ModelCheckboxes models={allModels} available={availModels} selected={aggModels} onChange={setAggModels} />
                </div>
              </div>
              <GradientButton onClick={runAggregated} disabled={aggMutation.isPending}>
                Executar Previsão Agregada
              </GradientButton>
            </div>
          </div>

          {aggMutation.isPending && <LoadingSpinner message="Clusterizando e treinando..." />}
          {aggData?.error && <ErrorMessage message={aggData.error} />}

          {aggData && !aggData.error && (
            <>
              <div className="grid grid-cols-3 gap-4 mb-6">
                <KpiCard label="Clusters" value={aggData.n_clusters} color="#4361ee" />
                <KpiCard label="Modelos" value={aggModels.length} color="#06d6a0" />
                <KpiCard label="Horizonte" value="30 dias" color="#118ab2" />
              </div>

              {Object.entries(aggData.charts).map(([cid, chartJson]) => (
                <div key={cid} className="card p-5 mb-6">
                  <h3 className="text-sm font-bold text-gray-700 mb-3 pb-2 border-b border-gray-100">Cluster {cid}</h3>
                  <PlotlyChart chartJson={chartJson} />
                  {aggData.metrics[cid]?.length > 0 && (
                    <div className="mt-4 overflow-x-auto">
                      <MetricsTable rows={aggData.metrics[cid]} showBias={false} />
                    </div>
                  )}
                </div>
              ))}

              <div className="card p-5 mt-6">
                <h3 className="text-sm font-bold text-gray-700 mb-3 pb-2 border-b border-gray-100">Pesos de Rateio Proporcional</h3>
                <div className="space-y-4">
                  {Object.entries(aggData.weights).map(([cid, weights]) => (
                    <div key={cid} className="border border-gray-100 rounded-lg p-3">
                      <h4 className="text-xs font-bold text-gray-500 uppercase mb-2">Cluster {cid}</h4>
                      <div className="flex flex-wrap gap-2">
                        {weights.map(w => (
                          <div key={w.sku} className="bg-gray-50 rounded-lg px-3 py-2 text-xs">
                            <span className="font-semibold text-gray-700">{w.sku}</span>
                            <span className="ml-2 text-brand-500 font-bold">{w.pct}%</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
}
