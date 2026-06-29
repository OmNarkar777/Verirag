/**
 * hooks/usePipeline.js — React Query hooks for pipeline ingest and query.
 */
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { getPipelineStats, listDocuments, ingestFile, ingestText, queryPipeline } from '../api/client.js'

export function usePipelineStats() {
  return useQuery({
    queryKey: ['pipeline-stats'],
    queryFn: getPipelineStats,
    staleTime: 60_000,
  })
}

export function useDocuments() {
  return useQuery({
    queryKey: ['documents'],
    queryFn: listDocuments,
    staleTime: 30_000,
  })
}

export function useIngestFile() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ file, collectionName }) => ingestFile(file, collectionName),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['pipeline-stats'] })
      qc.invalidateQueries({ queryKey: ['documents'] })
    },
  })
}

export function useIngestText() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ text, filename }) => ingestText(text, filename),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['pipeline-stats'] })
      qc.invalidateQueries({ queryKey: ['documents'] })
    },
  })
}

export function useQueryPipeline() {
  return useMutation({
    mutationFn: ({ question, topK, useMmr, fetchK, mmrLambda }) =>
      queryPipeline(question, { topK, useMmr, fetchK, mmrLambda }),
  })
}
