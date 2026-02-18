interface Tab {
  key: string;
  label: string;
}

interface Props {
  tabs: Tab[];
  active: string;
  onChange: (key: string) => void;
}

export default function TabGroup({ tabs, active, onChange }: Props) {
  return (
    <div className="flex gap-1 bg-gray-100 rounded-xl p-1 mb-6">
      {tabs.map(tab => (
        <button
          key={tab.key}
          onClick={() => onChange(tab.key)}
          className={`tab-btn flex-1 py-2.5 text-sm font-semibold rounded-lg text-gray-500 ${active === tab.key ? 'active' : ''}`}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}
