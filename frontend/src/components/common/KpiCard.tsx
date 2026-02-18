interface Props {
  label: string;
  value: string | number;
  color: string;
}

export default function KpiCard({ label, value, color }: Props) {
  return (
    <div className="card kpi-card p-5" style={{ '--kpi-color': color } as React.CSSProperties}>
      <div className="text-xs font-semibold text-gray-400 uppercase tracking-wider">{label}</div>
      <div className="text-2xl font-extrabold text-gray-900 mt-1">{value}</div>
    </div>
  );
}
