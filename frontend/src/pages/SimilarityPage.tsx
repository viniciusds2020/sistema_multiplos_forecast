import { useState } from 'react';
import Select from 'react-select';
import PageHeader from '../components/common/PageHeader';
import KpiCard from '../components/common/KpiCard';
import PlotlyChart from '../components/common/PlotlyChart';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorMessage from '../components/common/ErrorMessage';
import GradientButton from '../components/common/GradientButton';
import { useSimilarity } from '../hooks/useSimilarity';
import type { SimilarityResponse } from '../types/api';

const METRIC_OPTIONS = [
  { value: 'pearson', label: 'Pearson' },
  { value: 'euclidean', label: 'Euclidean' },
  { value: 'dtw', label: 'DTW' },
];

export default function SimilarityPage() {
  const [metric, setMetric] = useState('pearson');
  const [autoK, setAutoK] = useState(true);
  const [manualK, setManualK] = useState(4);
  const [selectedCluster, setSelectedCluster] = useState<string>('');

  const mutation = useSimilarity();
  const data = mutation.data as SimilarityResponse | undefined;

  const run = () => {
    mutation.mutate({ metric, auto_k: autoK, manual_k: manualK }, {
      onSuccess: (result) => {
        if (result.charts?.cluster_series) {
          const keys = Object.keys(result.charts.cluster_series);
          if (keys.length) setSelectedCluster(keys[0]);
        }
      },
    });
  };

  const clusterKeys = data?.charts?.cluster_series ? Object.keys(data.charts.cluster_series) : [];
  const clusterRows = data?.summary?.filter(r => String(r.cluster) === selectedCluster) ?? [];

  return (
    <div className="fade-in">
      <PageHeader title="Analise de Similaridade" subtitle="Identifique padroes semelhantes entre SKUs e agrupe-os em clusters" />

      {/* Controls */}
      <div className="card p-4 mb-6">
        <div className="flex flex-wrap items-center gap-6">
          <div>
            <label className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Metrica</label>
            <div className="mt-1 w-44">
              <Select
                options={METRIC_OPTIONS}
                value={METRIC_OPTIONS.find(o => o.value === metric)}
                onChange={(o) => o && setMetric(o.value)}
                className="text-sm"
              />
            </div>
          </div>
          <div className="flex items-center gap-3">
            <label className="flex items-center gap-2 text-sm font-medium text-gray-600">
              <input
                type="checkbox"
                checked={autoK}
                onChange={(e) => setAutoK(e.target.checked)}
                className="rounded border-gray-300 text-brand-500 focus:ring-brand-500"
              />
              K automatico
            </label>
            <input
              type="number"
              min={2}
              max={8}
              value={manualK}
              disabled={autoK}
              onChange={(e) => setManualK(parseInt(e.target.value) || 4)}
              className="w-16 rounded-lg border border-gray-200 px-2 py-2 text-sm text-center disabled:opacity-40"
            />
          </div>
          <GradientButton onClick={run} disabled={mutation.isPending}>
            Executar Clustering
          </GradientButton>
        </div>
      </div>

      {/* KPIs */}
      {data && !data.error && (
        <div className="grid grid-cols-3 gap-4 mb-8">
          <KpiCard label="Clusters" value={data.n_clusters} color="#4361ee" />
          <KpiCard label="Silhouette" value={data.sil_best} color="#06d6a0" />
          <KpiCard label="Total SKUs" value={data.total_skus} color="#118ab2" />
        </div>
      )}

      {mutation.isPending && <LoadingSpinner message="Calculando clusters..." />}
      {data?.error && <ErrorMessage message={data.error} />}

      {data && !data.error && data.charts && (
        <>
          {/* Row 1: Heatmap + Silhouette */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <div className="card p-5">
              <h3 className="text-sm font-bold text-gray-700 mb-3 pb-2 border-b border-gray-100">Matriz de Distância</h3>
              <PlotlyChart chartJson={data.charts.dist_heatmap} />
            </div>
            <div className="card p-5">
              <h3 className="text-sm font-bold text-gray-700 mb-3 pb-2 border-b border-gray-100">Silhouette Score por K</h3>
              <PlotlyChart chartJson={data.charts.silhouette} />
            </div>
          </div>

          {/* Row 2: MDS + Dendro */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <div className="card p-5">
              <h3 className="text-sm font-bold text-gray-700 mb-3 pb-2 border-b border-gray-100">Projeção MDS 2D</h3>
              <PlotlyChart chartJson={data.charts.mds} />
            </div>
            <div className="card p-5">
              <h3 className="text-sm font-bold text-gray-700 mb-3 pb-2 border-b border-gray-100">Dendrograma Hierárquico</h3>
              <PlotlyChart chartJson={data.charts.dendro} />
            </div>
          </div>

          {/* Cluster detail */}
          <div className="card p-5 mb-8">
            <div className="flex items-center gap-4 mb-4">
              <h3 className="text-sm font-bold text-gray-700">Detalhe do Cluster</h3>
              <select
                value={selectedCluster}
                onChange={(e) => setSelectedCluster(e.target.value)}
                className="rounded-lg border border-gray-200 px-3 py-1.5 text-sm"
              >
                {clusterKeys.map(c => (
                  <option key={c} value={c}>Cluster {c}</option>
                ))}
              </select>
            </div>
            <PlotlyChart chartJson={data.charts.cluster_series[selectedCluster]} />
            {clusterRows.length > 0 && (
              <div className="overflow-x-auto mt-4">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-xs font-semibold text-gray-400 uppercase tracking-wider border-b">
                      <th className="pb-2 pr-4">SKU</th>
                      <th className="pb-2 pr-4">Nome</th>
                      <th className="pb-2 pr-4">Perfil</th>
                      <th className="pb-2 pr-4">Média</th>
                      <th className="pb-2 pr-4">CV</th>
                      <th className="pb-2">% Zeros</th>
                    </tr>
                  </thead>
                  <tbody>
                    {clusterRows.map(r => (
                      <tr key={r.sku_id} className="border-b border-gray-50 hover:bg-gray-50/50">
                        <td className="py-2 pr-4 font-mono text-xs">{r.sku_id}</td>
                        <td className="py-2 pr-4 font-medium">{r.sku_name}</td>
                        <td className="py-2 pr-4">
                          <span className="px-2 py-0.5 rounded-full text-[10px] font-bold uppercase bg-brand-50 text-brand-600">
                            {r.demand_profile}
                          </span>
                        </td>
                        <td className="py-2 pr-4">{r.mean_demand}</td>
                        <td className="py-2 pr-4">{r.cv}</td>
                        <td className="py-2">{r.zero_pct}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
