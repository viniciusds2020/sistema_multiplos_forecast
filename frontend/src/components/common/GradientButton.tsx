interface Props {
  onClick: () => void;
  children: React.ReactNode;
  disabled?: boolean;
}

export default function GradientButton({ onClick, children, disabled }: Props) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className="px-5 py-2.5 bg-gradient-to-r from-brand-500 to-brand-600 text-white text-sm font-semibold rounded-lg shadow-md shadow-brand-500/30 hover:shadow-lg hover:-translate-y-0.5 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
    >
      {children}
    </button>
  );
}
