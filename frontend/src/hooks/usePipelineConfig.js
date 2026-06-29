import { useQuery } from '@tanstack/react-query'
import { getPipelineConfig, getPipelineStats } from '../api/client.js'

export function usePipelineConfig() {
  return useQuery({
    queryKey: ['pipeline-config'],
    queryFn: getPipelineConfig,
    staleTime: 300_000,
    retry: 1,
  })
}

export function usePipelineStats() {
  return useQuery({
    queryKey: ['pipeline-stats'],
    queryFn: getPipelineStats,
    staleTime: 30_000,
    retry: 1,
  })
}
