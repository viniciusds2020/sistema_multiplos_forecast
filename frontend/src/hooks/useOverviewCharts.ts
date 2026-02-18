import { useQuery } from '@tanstack/react-query';
import { fetchOverviewCharts } from '../services/api';

export function useOverviewCharts() {
  return useQuery({
    queryKey: ['overview-charts'],
    queryFn: fetchOverviewCharts,
    staleTime: 5 * 60 * 1000,
  });
}
