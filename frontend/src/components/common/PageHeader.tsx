interface Props {
  title: string;
  subtitle: string;
}

export default function PageHeader({ title, subtitle }: Props) {
  return (
    <>
      <h1 className="text-2xl font-extrabold text-gray-900 tracking-tight">{title}</h1>
      <p className="text-sm text-gray-400 mt-1">{subtitle}</p>
      <div className="h-px bg-gradient-to-r from-brand-500/40 to-transparent mt-4 mb-6" />
    </>
  );
}
