import { useState, useMemo } from 'react';
import Select from 'react-select';
import PageHeader from '../components/common/PageHeader';
import PlotlyChart from '../components/common/PlotlyChart';
import TabGroup from '../components/common/TabGroup';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorMessage from '../components/common/ErrorMessage';
import GradientButton from '../components/common/GradientButton';
import { useSkus } from '../hooks/useSkus';
import { useExplorerDemand, useExplorerClimate, useExplorerDistributions, useExplorerRaw } from '../hooks/useExplorer';

const TABS = [
  { key: 'demand', label: 'Séries de Demanda' },
  { key: 'climate', label: 'Dados Climáticos' },
  { key: 'dist', label: 'Distribuicoes' },
  { key: 'raw', label: 'Dados Brutos' },
];

export default function ExplorerPage() {
  const skus = useSkus();
  const [activeTab, setActiveTab] = useState('demand');
  const [selectedSkus, setSelectedSkus] = useState<string[]>([]);
  const [demandSkus, setDemandSkus] = useState<string[]>([]);

  // Initialize selected SKUs once data loads
  const skuOptions = useMemo(() => {
    if (!skus.data) return [];
    const opts = skus.data.map(s => ({ value: s.sku_name, label: s.sku_name }));
    if (selectedSkus.length === 0 && opts.length > 0) {
      const initial = opts.slice(0, 4).map(o => o.value);
      setSelectedSkus(initial);
      setDemandSkus(initial);
    }
    return opts;
  }, [skus.data]);

  const demand = useExplorerDemand(demandSkus);
  const climate = useExplorerClimate(activeTab === 'climate');
  const distributions = useExplorerDistributions(activeTab === 'dist');
  const raw = useExplorerRaw(activeTab === 'raw');

  return (
    <div className="fade-in">
      <PageHeader title="Exploração de Dados" subtitle="Explore as séries temporais, dados climáticos e features" />

      {/* SKU selector */}
      <div className="card p-4 mb-6">
        <div className="flex flex-wrap items-center gap-3">
          <label className="text-sm font-semibold text-gray-600">SKUs:</label>
          <div className="w-96">
            <Select
              isMulti
              options={skuOptions}
              value={skuOptions.filter(o => selectedSkus.includes(o.value))}
              onChange={(selected) => setSelectedSkus(selected ? selected.map(s => s.value) : [])}
              placeholder="Buscar SKUs..."
              className="text-sm"
            />
          </div>
          <GradientButton onClick={() => setDemandSkus([...selectedSkus])}>
            Atualizar
          </GradientButton>
        </div>
      </div>

      <TabGroup tabs={TABS} active={activeTab} onChange={setActiveTab} />

      {/* Demand tab */}
      {activeTab === 'demand' && (
        <div className="card p-5">
          <h3 className="text-sm font-bold text-gray-700 mb-3 pb-2 border-b border-gray-100">Séries de Demanda por SKU</h3>
          {demand.isLoading && <LoadingSpinner />}
          {demand.error && <ErrorMessage message="Erro ao carregar dados" />}
          {demand.data && <PlotlyChart chartJson={demand.data.chart} />}
        </div>
      )}

      {/* Climate tab */}
      {activeTab === 'climate' && (
        <>
          {climate.isLoading && <LoadingSpinner />}
          {climate.error && <ErrorMessage message="Erro ao carregar dados climaticos" />}
          {climate.data && (
            <>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                <div className="card p-5"><PlotlyChart chartJson={climate.data.temperature} /></div>
                <div className="card p-5"><PlotlyChart chartJson={climate.data.rainfall} /></div>
              </div>
              <div className="card p-5"><PlotlyChart chartJson={climate.data.humidity} /></div>
            </>
          )}
        </>
      )}

      {/* Distributions tab */}
      {activeTab === 'dist' && (
        <>
          {distributions.isLoading && <LoadingSpinner />}
          {distributions.error && <ErrorMessage message="Erro ao carregar distribuicoes" />}
          {distributions.data && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="card p-5"><PlotlyChart chartJson={distributions.data.season} /></div>
              <div className="card p-5"><PlotlyChart chartJson={distributions.data.dow} /></div>
              <div className="card p-5"><PlotlyChart chartJson={distributions.data.safra} /></div>
              <div className="card p-5"><PlotlyChart chartJson={distributions.data.profile} /></div>
            </div>
          )}
        </>
      )}

      {/* Raw data tab */}
      {activeTab === 'raw' && (
        <div className="card p-5">
          <h3 className="text-sm font-bold text-gray-700 mb-3 pb-2 border-b border-gray-100">Dados Brutos (primeiros 300 registros)</h3>
          {raw.isLoading && <LoadingSpinner />}
          {raw.error && <ErrorMessage message="Erro ao carregar dados brutos" />}
          {raw.data && raw.data.length > 0 && (
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-left text-[10px] font-semibold text-gray-400 uppercase tracking-wider border-b">
                    {Object.keys(raw.data[0]).map(col => (
                      <th key={col} className="pb-2 pr-3 whitespace-nowrap">{col}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {raw.data.map((row, i) => (
                    <tr key={i} className="border-b border-gray-50 hover:bg-gray-50/50">
                      {Object.values(row).map((val, j) => (
                        <td key={j} className="py-1.5 pr-3 whitespace-nowrap">{String(val ?? '')}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
