import { useMemo } from 'react';
import Plot from 'react-plotly.js';

interface Props {
  chartJson: string | undefined | null;
  className?: string;
}

export default function PlotlyChart({ chartJson, className = '' }: Props) {
  const parsed = useMemo(() => {
    if (!chartJson) return null;
    try {
      return JSON.parse(chartJson);
    } catch {
      return null;
    }
  }, [chartJson]);

  if (!parsed) {
    return (
      <div className="flex items-center justify-center py-12 text-sm text-gray-400">
        Sem dados para exibir
      </div>
    );
  }

  return (
    <div className={`plotly-chart ${className}`}>
      <Plot
        data={parsed.data}
        layout={{ ...parsed.layout, autosize: true }}
        config={{ responsive: true, displayModeBar: false }}
        useResizeHandler
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  );
}
