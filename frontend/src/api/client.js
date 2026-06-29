/**
 * api/client.js — Axios instance and all typed API functions.
 *
 * Single source of truth for API calls. Every component imports from here —
 * never writes inline fetch(). This makes mocking in tests trivial and
 * keeps base URL / auth headers in one place.
 */
import axios from 'axios'

const api = axios.create({
  baseURL: '/api/v1',
  timeout: 30_000,
  headers: { 'Content-Type': 'application/json' },
})

api.interceptors.response.use(
  (res) => res,
  (err) => {
    const msg = err.response?.data?.detail || err.message || 'Unknown error'
    console.error(`[API] ${err.config?.method?.toUpperCase()} ${err.config?.url} → ${msg}`)
    return Promise.reject(new Error(msg))
  },
)

// ── Eval ──────────────────────────────────────────────────────────────────────
export const startEvalRun = (payload) =>
  api.post('/eval/run', payload).then((r) => r.data)

export const startSampleEval = () =>
  api.post('/eval/run/sample').then((r) => r.data)

export const listEvalRuns = ({ limit = 50, offset = 0 } = {}) =>
  api.get('/eval/runs', { params: { limit, offset } }).then((r) => r.data)

export const getEvalRun = (runId) =>
  api.get(`/eval/runs/${runId}`).then((r) => r.data)

export const getEvalCases = (runId, { page = 1, pageSize = 20 } = {}) =>
  api.get(`/eval/runs/${runId}/cases`, { params: { page, page_size: pageSize } }).then((r) => r.data)

export const getRegressions = () =>
  api.get('/eval/regressions').then((r) => r.data)

export const getEvalStatus = () =>
  api.get('/eval/status').then((r) => r.data)

export const deleteEvalRun = (runId) =>
  api.delete(`/eval/runs/${runId}`)

// ── Pipeline ──────────────────────────────────────────────────────────────────
export const ingestFile = (file, collectionName) => {
  const form = new FormData()
  form.append('file', file)
  if (collectionName) form.append('collection_name', collectionName)
  return api.post('/pipeline/ingest', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 120_000,
  }).then((r) => r.data)
}

export const ingestText = (text, filename) => {
  const form = new FormData()
  form.append('text', text)
  form.append('filename', filename)
  return api.post('/pipeline/ingest/text', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  }).then((r) => r.data)
}

export const queryPipeline = (question, topK = 5) =>
  api.post('/pipeline/query', { question, top_k: topK }).then((r) => r.data)

export const getPipelineStats = () =>
  api.get('/pipeline/stats').then((r) => r.data)

export const getPipelineConfig = () =>
  api.get('/pipeline/config').then((r) => r.data)

export const listDocuments = () =>
  api.get('/pipeline/documents').then((r) => r.data)

// ── Health ────────────────────────────────────────────────────────────────────
export const getHealth = () =>
  axios.get('/health').then((r) => r.data)

export default api
