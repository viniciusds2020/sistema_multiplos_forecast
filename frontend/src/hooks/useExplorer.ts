import { useQuery } from '@tanstack/react-query';
import {
  fetchExplorerDemand, fetchExplorerClimate,
  fetchExplorerDistributions, fetchExplorerRaw,
} from '../services/api';

export function useExplorerDemand(skus: string[]) {
  return useQuery({
    queryKey: ['explorer-demand', skus],
    queryFn: () => fetchExplorerDemand(skus),
    enabled: skus.length > 0,
    staleTime: 5 * 60 * 1000,
  });
}

export function useExplorerClimate(enabled: boolean) {
  return useQuery({
    queryKey: ['explorer-climate'],
    queryFn: fetchExplorerClimate,
    enabled,
    staleTime: 10 * 60 * 1000,
  });
}

export function useExplorerDistributions(enabled: boolean) {
  return useQuery({
    queryKey: ['explorer-distributions'],
    queryFn: fetchExplorerDistributions,
    enabled,
    staleTime: 10 * 60 * 1000,
  });
}

export function useExplorerRaw(enabled: boolean) {
  return useQuery({
    queryKey: ['explorer-raw'],
    queryFn: fetchExplorerRaw,
    enabled,
    staleTime: 10 * 60 * 1000,
  });
}
