interface Props {
  title: string;
  subtitle: string;
}

export default function PageHeader({ title, subtitle }: Props) {
  return (
    <>
      <h1 className="text-3xl md:text-4xl font-semibold text-brand-900 tracking-tight font-serif">
        {title}
      </h1>
      <p className="text-sm text-gray-500 mt-2 max-w-2xl">{subtitle}</p>
      <div className="h-px bg-gradient-to-r from-brand-500/60 via-brand-200 to-transparent mt-5 mb-8" />
    </>
  );
}
