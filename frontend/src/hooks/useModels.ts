import { useQuery } from '@tanstack/react-query';
import { fetchModels } from '../services/api';

export function useModels() {
  return useQuery({
    queryKey: ['models'],
    queryFn: fetchModels,
    staleTime: 60 * 60 * 1000,
  });
}
