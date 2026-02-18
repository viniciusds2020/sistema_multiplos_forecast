import { useMutation } from '@tanstack/react-query';
import { runIndividualForecast, runAggregatedForecast } from '../services/api';

export function useIndividualForecast() {
  return useMutation({
    mutationFn: runIndividualForecast,
  });
}

export function useAggregatedForecast() {
  return useMutation({
    mutationFn: runAggregatedForecast,
  });
}
