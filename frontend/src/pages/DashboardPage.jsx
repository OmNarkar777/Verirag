/**
 * DashboardPage - metric cards, trend chart, regression alert, runs table.
 */
import { useState } from 'react'
import { useEvalRuns, useStartSampleEval } from '../hooks/useEvalRuns.js'
import MetricCard from '../components/dashboard/MetricCard.jsx'
import MetricTrendChart from '../components/dashboard/MetricTrendChart.jsx'
import EvalRunsTable from '../components/dashboard/EvalRunsTable.jsx'
import RegressionAlert from '../components/dashboard/RegressionAlert.jsx'

const METRICS = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']

function ApiErrorBanner({ message }) {
  return (
    <div className="rounded-xl border border-red-500/30 bg-red-500/5 px-5 py-4">
      <p className="text-sm font-medium text-red-400 mb-1">Unable to reach backend API</p>
      <p className="text-xs text-slate-500">{message}</p>
      <p className="text-xs text-slate-600 mt-2">
        If you just deployed, check that <code className="text-slate-400">DATABASE_URL</code> and{' '}
        <code className="text-slate-400">GROQ_API_KEY</code> are set in your Vercel environment variables.
      </p>
    </div>
  )
}

export default function DashboardPage() {
  const { data: runs = [], isLoading, isError, error } = useEvalRuns()
  const startSample = useStartSampleEval()

  const completed = runs.filter((r) => r.status === 'completed' && r.scores)
  const latest = completed[0] ?? null
  const previous = completed[1] ?? null

  const getDelta = (metric) => {
    if (!latest?.scores?.[metric] || !previous?.scores?.[metric]) return null
    return latest.scores[metric] - previous.scores[metric]
  }

  const handleRunSample = async () => {
    const tag = `v${Date.now().toString(36)}-sample`
    try {
      await startSample.mutateAsync(tag)
    } catch (e) {
      console.error('Failed to start sample eval:', e.message)
    }
  }

  const activeCount = runs.filter((r) => r.status === 'running').length

  return (
    <div className="p-6 max-w-7xl space-y-6">
      {isError && <ApiErrorBanner message={error?.message} />}

      <RegressionAlert />

      {activeCount > 0 && (
        <div className="rounded-xl border border-blue-500/20 bg-blue-500/5 px-4 py-2.5 flex items-center gap-2.5">
          <span className="w-2 h-2 rounded-full bg-blue-400 animate-pulse shrink-0" />
          <p className="text-xs text-blue-300">
            {activeCount} evaluation{activeCount > 1 ? 's' : ''} running — scores will appear when complete
          </p>
        </div>
      )}

      {/* Metric cards */}
      <div>
        <h2 className="text-xs font-medium text-slate-500 uppercase tracking-wider mb-3">
          Latest Scores
          {latest && (
            <span className="ml-2 text-slate-600 normal-case font-normal">
              from <span className="font-mono text-slate-500">{latest.version_tag}</span>
            </span>
          )}
        </h2>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          {METRICS.map((m) => (
            <MetricCard
              key={m}
              metric={m}
              score={latest?.scores?.[m] ?? null}
              delta={getDelta(m)}
              loading={isLoading}
            />
          ))}
        </div>
      </div>

      {/* Trend chart */}
      <div className="glass rounded-xl p-5">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-sm font-semibold text-slate-200">Metric Trends</h2>
            <p className="text-xs text-slate-500 mt-0.5">Last 10 completed evaluation runs</p>
          </div>
          {completed.length > 0 && (
            <span className="text-xs text-slate-600">{completed.length} completed run{completed.length !== 1 ? 's' : ''}</span>
          )}
        </div>
        <MetricTrendChart runs={runs} loading={isLoading} />
      </div>

      <EvalRunsTable
        runs={runs}
        loading={isLoading}
        onRunSample={handleRunSample}
      />
    </div>
  )
}
