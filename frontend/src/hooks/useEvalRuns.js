/**
 * hooks/useEvalRuns.js - React Query hooks for eval runs + regression data.
 */
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { listEvalRuns, getRegressions, startSampleEval, deleteEvalRun, getEvalStatus } from '../api/client.js'

export const EVAL_RUNS_KEY = ['eval-runs']
export const REGRESSIONS_KEY = ['regressions']

export function useEvalRuns({ limit = 50 } = {}) {
  return useQuery({
    queryKey: [...EVAL_RUNS_KEY, limit],
    queryFn: () => listEvalRuns({ limit }),
    staleTime: 15_000,
    refetchInterval: ({ data } = {}) => {
      const hasRunning = data?.some?.((r) => r.status === 'running')
      return hasRunning ? 8_000 : 30_000
    },
    retry: 1,
  })
}

export function useRegressions() {
  return useQuery({
    queryKey: REGRESSIONS_KEY,
    queryFn: getRegressions,
    refetchInterval: 30_000,
    retry: 1,
  })
}

export function useEvalStatus() {
  return useQuery({
    queryKey: ['eval-status'],
    queryFn: getEvalStatus,
    refetchInterval: 10_000,
    retry: false,
  })
}

export function useStartSampleEval({ onSuccess, onError } = {}) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: () => startSampleEval(),
    onSuccess: (data) => {
      qc.invalidateQueries({ queryKey: EVAL_RUNS_KEY })
      onSuccess?.(data)
    },
    onError: (err) => onError?.(err),
  })
}

export function useDeleteEvalRun() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (runId) => deleteEvalRun(runId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: EVAL_RUNS_KEY })
      qc.invalidateQueries({ queryKey: REGRESSIONS_KEY })
    },
  })
}
