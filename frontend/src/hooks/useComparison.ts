import { useMutation } from '@tanstack/react-query';
import { runComparison } from '../services/api';

export function useComparison() {
  return useMutation({
    mutationFn: runComparison,
  });
}
