interface Props {
  models: string[];
  available: string[];
  selected: string[];
  onChange: (selected: string[]) => void;
}

export default function ModelCheckboxes({ models, available, selected, onChange }: Props) {
  const toggle = (model: string) => {
    if (selected.includes(model)) {
      onChange(selected.filter(m => m !== model));
    } else {
      onChange([...selected, model]);
    }
  };

  return (
    <div className="flex flex-wrap gap-2">
      {models.map(m => {
        const avail = available.includes(m);
        return (
          <label
            key={m}
            className={`flex items-center gap-1.5 text-xs font-medium ${avail ? 'text-gray-600' : 'text-gray-300 line-through'}`}
          >
            <input
              type="checkbox"
              checked={selected.includes(m)}
              disabled={!avail}
              onChange={() => toggle(m)}
              className="rounded border-gray-300 text-brand-500 focus:ring-brand-500"
            />
            {m}{!avail && ' (indisponivel)'}
          </label>
        );
      })}
    </div>
  );
}
