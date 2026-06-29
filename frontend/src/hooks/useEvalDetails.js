/**
 * hooks/useEvalDetails.js â€” React Query hooks for a single eval run + its cases.
 * TanStack Query v5: refetchInterval receives { data } not data directly.
 */
import { useQuery } from '@tanstack/react-query'
import { getEvalRun, getEvalCases } from '../api/client.js'

export function useEvalRun(runId) {
  return useQuery({
    queryKey: ['eval-run', runId],
    queryFn: () => getEvalRun(runId),
    enabled: !!runId,
    // v5 API: destructure { data } from the query object
    refetchInterval: ({ data } = {}) => (data?.status === 'running' ? 5_000 : false),
  })
}

export function useEvalCases(runId, { page = 1, pageSize = 20 } = {}) {
  return useQuery({
    queryKey: ['eval-cases', runId, page, pageSize],
    queryFn: () => getEvalCases(runId, { page, pageSize }),
    enabled: !!runId,
    placeholderData: (prev) => prev,
  })
}