import { MODEL_COLORS } from '../../constants/colors';
import type { MetricsRow } from '../../types/api';

interface Props {
  rows: MetricsRow[];
  showBias?: boolean;
}

export default function MetricsTable({ rows, showBias = true }: Props) {
  if (!rows.length) return null;

  const minWape = Math.min(...rows.map(r => r.WAPE));

  return (
    <table className="w-full text-sm">
      <thead>
        <tr className="text-left text-xs font-semibold text-gray-400 uppercase border-b">
          <th className="pb-2 pr-4">Modelo</th>
          <th className="pb-2 pr-4">MAE</th>
          <th className="pb-2 pr-4">RMSE</th>
          <th className="pb-2 pr-4">MAPE</th>
          <th className="pb-2 pr-4">WAPE</th>
          {showBias && <th className="pb-2">Bias</th>}
        </tr>
      </thead>
      <tbody>
        {rows.map(r => {
          const isBest = r.WAPE === minWape;
          return (
            <tr key={r.Model} className={`border-b border-gray-50 ${isBest ? 'bg-green-50/50' : 'hover:bg-gray-50/50'}`}>
              <td className="py-2 pr-4 font-semibold" style={{ color: MODEL_COLORS[r.Model] || '#333' }}>{r.Model}</td>
              <td className="py-2 pr-4">{r.MAE}</td>
              <td className="py-2 pr-4">{r.RMSE}</td>
              <td className="py-2 pr-4">{r.MAPE}</td>
              <td className={`py-2 pr-4 font-bold ${isBest ? 'text-green-600' : ''}`}>{r.WAPE}</td>
              {showBias && <td className="py-2">{r.Bias}</td>}
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
