import { useState, useEffect, useMemo } from 'react';
import Select from 'react-select';
import PageHeader from '../components/common/PageHeader';
import PlotlyChart from '../components/common/PlotlyChart';
import MetricsTable from '../components/common/MetricsTable';
import ModelCheckboxes from '../components/common/ModelCheckboxes';
import TabGroup from '../components/common/TabGroup';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorMessage from '../components/common/ErrorMessage';
import GradientButton from '../components/common/GradientButton';
import { MODEL_COLORS } from '../constants/colors';
import { useModels } from '../hooks/useModels';
import { useSkus } from '../hooks/useSkus';
import { useComparison } from '../hooks/useComparison';
import type { ComparisonResponse } from '../types/api';

const RESULT_TABS = [
  { key: 'ranking', label: 'Ranking Geral' },
  { key: 'visual', label: 'Comparação Visual' },
  { key: 'skus', label: 'Por SKU' },
];

export default function ComparisonPage() {
  const modelsQuery = useModels();
  const skusQuery = useSkus();

  const allModels = modelsQuery.data?.all ?? ['XGBoost', 'LightGBM', 'AutoARIMA', 'Prophet', 'CrostonSBA'];
  const availModels = modelsQuery.data?.available ?? allModels;

  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [selectedSkuIds, setSelectedSkuIds] = useState<string[]>([]);
  const [resultTab, setResultTab] = useState('ranking');
  const [detailSku, setDetailSku] = useState('');

  const mutation = useComparison();
  const data = mutation.data as ComparisonResponse | undefined;

  // Initialize model selection
  useEffect(() => {
    if (availModels.length && selectedModels.length === 0) {
      setSelectedModels(availModels.slice(0, 3));
    }
  }, [availModels]);

  // SKU options
  const skuOptions = useMemo(() => {
    if (!skusQuery.data) return [];
    return skusQuery.data.map(s => ({ value: s.sku_id, label: s.sku_name }));
  }, [skusQuery.data]);

  // Initialize selected SKUs
  useEffect(() => {
    if (skuOptions.length && selectedSkuIds.length === 0) {
      setSelectedSkuIds(skuOptions.slice(0, 6).map(o => o.value));
    }
  }, [skuOptions]);

  const run = () => {
    if (!selectedModels.length) return;
    mutation.mutate({
      models: selectedModels,
      sku_ids: selectedSkuIds,
      horizon: 30,
      test_days: 60,
    }, {
      onSuccess: (result) => {
        if (result.metrics?.length) {
          const uniqueSkus = [...new Set(result.metrics.map(m => m.SKU))];
          if (uniqueSkus.length) setDetailSku(uniqueSkus[0]);
        }
      },
    });
  };

  // Unique SKUs from results
  const resultSkus = useMemo(() => {
    if (!data?.metrics) return [];
    return [...new Set(data.metrics.map(m => m.SKU))];
  }, [data]);

  // Detail SKU metrics
  const detailRows = useMemo(() => {
    if (!data?.metrics || !detailSku) return [];
    return data.metrics.filter(m => m.SKU === detailSku);
  }, [data, detailSku]);

  const detailSkuName = data?.metrics?.find(m => m.SKU === detailSku)?.Nome ?? detailSku;

  return (
    <div className="fade-in">
      <PageHeader title="Comparação de Modelos" subtitle="Avalie e compare o desempenho dos modelos em todos os SKUs" />

      {/* Controls */}
      <div className="card p-4 mb-6">
        <div className="flex flex-wrap items-end gap-4">
          <div>
            <label className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Modelos</label>
            <div className="mt-1">
              <ModelCheckboxes models={allModels} available={availModels} selected={selectedModels} onChange={setSelectedModels} />
            </div>
          </div>
          <div>
            <label className="text-xs font-semibold text-gray-400 uppercase tracking-wider">SKUs</label>
            <div className="mt-1 w-80">
              <Select
                isMulti
                options={skuOptions}
                value={skuOptions.filter(o => selectedSkuIds.includes(o.value))}
                onChange={(selected) => setSelectedSkuIds(selected ? selected.map(s => s.value) : [])}
                placeholder="Buscar SKUs..."
                className="text-sm"
              />
            </div>
          </div>
          <GradientButton onClick={run} disabled={mutation.isPending}>
            Executar Comparação
          </GradientButton>
        </div>
      </div>

      {mutation.isPending && <LoadingSpinner message="Avaliando modelos em todos os SKUs..." />}
      {data?.error && <ErrorMessage message={data.error} />}

      {data && !data.error && (
        <>
          <TabGroup tabs={RESULT_TABS} active={resultTab} onChange={setResultTab} />

          {/* Ranking tab */}
          {resultTab === 'ranking' && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
              <div className="card p-5">
                <h3 className="text-sm font-bold text-gray-700 mb-3 pb-2 border-b border-gray-100">Média por Modelo</h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-left text-xs font-semibold text-gray-400 uppercase border-b">
                        <th className="pb-2 pr-4">Modelo</th>
                        <th className="pb-2 pr-4">MAE</th>
                        <th className="pb-2 pr-4">RMSE</th>
                        <th className="pb-2 pr-4">MAPE</th>
                        <th className="pb-2">WAPE</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.avg.map(r => {
                        const minWape = Math.min(...data.avg.map(a => a.WAPE));
                        const best = r.WAPE === minWape;
                        return (
                          <tr key={r.Model} className={`border-b border-gray-50 ${best ? 'bg-green-50/50' : ''}`}>
                            <td className="py-2.5 pr-4 font-semibold" style={{ color: MODEL_COLORS[r.Model] || '#333' }}>{r.Model}</td>
                            <td className="py-2.5 pr-4">{r.MAE}</td>
                            <td className="py-2.5 pr-4">{r.RMSE}</td>
                            <td className="py-2.5 pr-4">{r.MAPE}</td>
                            <td className={`py-2.5 font-bold ${best ? 'text-green-600' : ''}`}>{r.WAPE}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
              <div className="card p-5">
                <h3 className="text-sm font-bold text-gray-700 mb-3 pb-2 border-b border-gray-100">Vezes Melhor Modelo</h3>
                <PlotlyChart chartJson={data.charts.best_count} />
              </div>
            </div>
          )}

          {/* Visual tab */}
          {resultTab === 'visual' && (
            <>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                <div className="card p-5">
                  <h3 className="text-sm font-bold text-gray-700 mb-3 pb-2 border-b border-gray-100">Radar de Métricas</h3>
                  <PlotlyChart chartJson={data.charts.radar} />
                </div>
                <div className="card p-5">
                  <h3 className="text-sm font-bold text-gray-700 mb-3 pb-2 border-b border-gray-100">Distribuição WAPE</h3>
                  <PlotlyChart chartJson={data.charts.box_wape} />
                </div>
              </div>
              <div className="card p-5">
                <h3 className="text-sm font-bold text-gray-700 mb-3 pb-2 border-b border-gray-100">WAPE por SKU e Modelo</h3>
                <PlotlyChart chartJson={data.charts.grouped_bar} />
              </div>
            </>
          )}

          {/* Per SKU tab */}
          {resultTab === 'skus' && (
            <>
              <div className="card p-4 mb-4">
                <label className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Selecionar SKU</label>
                <div className="mt-1 w-56">
                  <Select
                    options={resultSkus.map(s => {
                      const row = data.metrics.find(m => m.SKU === s);
                      return { value: s, label: row?.Nome || s };
                    })}
                    value={{ value: detailSku, label: detailSkuName }}
                    onChange={(o) => o && setDetailSku(o.value)}
                    className="text-sm"
                  />
                </div>
              </div>
              <div className="card p-5 mb-6">
                <PlotlyChart chartJson={data.sku_charts[detailSku]} />
              </div>
              <div className="card p-5">
                <h3 className="text-sm font-bold text-gray-700 mb-3 pb-2 border-b border-gray-100">Métricas do SKU</h3>
                <MetricsTable rows={detailRows} />
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
}
