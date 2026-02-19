import PageHeader from '../components/common/PageHeader';
import KpiCard from '../components/common/KpiCard';
import PlotlyChart from '../components/common/PlotlyChart';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorMessage from '../components/common/ErrorMessage';
import { useKpis } from '../hooks/useKpis';
import { useOverviewCharts } from '../hooks/useOverviewCharts';

export default function OverviewPage() {
  const kpis = useKpis();
  const charts = useOverviewCharts();

  return (
    <div className="fade-in">
      <PageHeader title="Visão Geral do Dashboard" subtitle="Visão geral do sistema de forecast multi-produto SKU" />

      {/* KPIs */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        {kpis.data && (
          <>
            <KpiCard label="Total SKUs" value={kpis.data.total_skus} color="#4361ee" />
            <KpiCard label="Registros" value={kpis.data.total_records.toLocaleString()} color="#06d6a0" />
            <KpiCard label="Cobertura" value={`${kpis.data.coverage_days} dias`} color="#118ab2" />
            <KpiCard label="Demanda Média" value={`${kpis.data.avg_demand} un/dia`} color="#ffd166" />
          </>
        )}
      </div>

      {charts.isLoading && <LoadingSpinner />}
      {charts.error && <ErrorMessage message="Erro ao carregar graficos" />}

      {charts.data && (
        <>
          {/* Heatmap + Top SKUs */}
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-6 mb-8">
            <div className="lg:col-span-3 card p-5">
              <h3 className="text-sm font-bold text-gray-700 mb-3 pb-2 border-b border-gray-100">Heatmap de Demanda - SKU x Mes</h3>
              <PlotlyChart chartJson={charts.data.heatmap} />
            </div>
            <div className="lg:col-span-2 card p-5">
              <h3 className="text-sm font-bold text-gray-700 mb-3 pb-2 border-b border-gray-100">Top 10 SKUs por Volume</h3>
              <PlotlyChart chartJson={charts.data.top_skus} />
            </div>
          </div>

          {/* Total demand + Profiles */}
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-6 mb-8">
            <div className="lg:col-span-3 card p-5">
              <h3 className="text-sm font-bold text-gray-700 mb-3 pb-2 border-b border-gray-100">Demanda Total Agregada</h3>
              <PlotlyChart chartJson={charts.data.total_demand} />
            </div>
            <div className="lg:col-span-2 card p-5">
              <h3 className="text-sm font-bold text-gray-700 mb-3 pb-2 border-b border-gray-100">Distribuição por Perfil</h3>
              <PlotlyChart chartJson={charts.data.profiles} />
            </div>
          </div>

          {/* Stats table */}
          <div className="card p-5">
            <h3 className="text-sm font-bold text-gray-700 mb-3 pb-2 border-b border-gray-100">Estatísticas por Perfil de Demanda</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-xs font-semibold text-gray-400 uppercase tracking-wider border-b border-gray-100">
                    <th className="pb-3 pr-4">Perfil</th>
                    <th className="pb-3 pr-4">SKUs</th>
                    <th className="pb-3 pr-4">Média</th>
                    <th className="pb-3 pr-4">Desvio</th>
                    <th className="pb-3 pr-4">Máximo</th>
                    <th className="pb-3">% Zeros</th>
                  </tr>
                </thead>
                <tbody>
                  {charts.data.stats_table.map(r => (
                    <tr key={r.demand_profile} className="border-b border-gray-50 hover:bg-gray-50/50">
                      <td className="py-2.5 pr-4 font-medium">{r.demand_profile}</td>
                      <td className="py-2.5 pr-4">{r.skus}</td>
                      <td className="py-2.5 pr-4">{r.media}</td>
                      <td className="py-2.5 pr-4">{r.desvio}</td>
                      <td className="py-2.5 pr-4">{r.maximo}</td>
                      <td className="py-2.5">{r.zero_pct}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
