interface Props {
  message?: string;
}

export default function LoadingSpinner({ message = 'Carregando...' }: Props) {
  return (
    <div className="flex items-center justify-center py-20">
      <div className="spinner" />
      <span className="ml-3 text-gray-400">{message}</span>
    </div>
  );
}
