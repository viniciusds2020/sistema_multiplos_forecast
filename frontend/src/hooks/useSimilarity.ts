import { useMutation } from '@tanstack/react-query';
import { runSimilarity } from '../services/api';

export function useSimilarity() {
  return useMutation({
    mutationFn: runSimilarity,
  });
}
