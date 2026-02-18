import { useQuery } from '@tanstack/react-query';
import { fetchSkus } from '../services/api';

export function useSkus() {
  return useQuery({
    queryKey: ['skus'],
    queryFn: fetchSkus,
    staleTime: 10 * 60 * 1000,
  });
}
