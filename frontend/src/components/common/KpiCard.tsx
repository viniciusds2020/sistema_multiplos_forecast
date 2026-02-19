interface Props {
  label: string;
  value: string | number;
  color: string;
}

export default function KpiCard({ label, value, color }: Props) {
  return (
    <div className="card kpi-card p-5" style={{ '--kpi-color': color } as React.CSSProperties}>
      <div className="text-[11px] font-semibold text-gray-500 uppercase tracking-[.3em]">{label}</div>
      <div className="text-2xl md:text-3xl font-semibold text-brand-900 mt-2 font-serif">{value}</div>
    </div>
  );
}
