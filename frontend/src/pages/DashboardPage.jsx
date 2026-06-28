/**
 * DashboardPage — metric overview, trend chart, regression alerts, run history.
 * Shows a product intro banner when data is present so recruiters immediately
 * understand what VeriRAG does and why it matters.
 */
import { useState } from 'react'
import { useEvalRuns, useStartSampleEval } from '../hooks/useEvalRuns.js'
import MetricCard from '../components/dashboard/MetricCard.jsx'
import MetricTrendChart from '../components/dashboard/MetricTrendChart.jsx'
import EvalRunsTable from '../components/dashboard/EvalRunsTable.jsx'
import RegressionAlert from '../components/dashboard/RegressionAlert.jsx'

const METRICS = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']

const METRIC_DESCRIPTIONS = {
  faithfulness: 'Is every answer claim grounded in retrieved context? Measures hallucination.',
  answer_relevancy: 'Does the answer address what was actually asked?',
  context_precision: 'Are the most useful chunks ranked highest in retrieval?',
  context_recall: 'Does the retrieved context cover all information needed for a correct answer?',
}

function ApiErrorBanner({ message }) {
  return (
    <div className="rounded-xl border border-red-500/30 bg-red-500/5 px-5 py-4">
      <p className="text-sm font-medium text-red-400 mb-1">Unable to reach backend</p>
      <p className="text-xs text-slate-500">{message}</p>
      <p className="text-xs text-slate-600 mt-2">
        Check that <code className="text-slate-400">DATABASE_URL</code> and{' '}
        <code className="text-slate-400">GROQ_API_KEY</code> are configured in Vercel.
      </p>
    </div>
  )
}

function ProductHeroBanner({ latestRun, totalRuns, avgFaithfulness }) {
  return (
    <div className="glass rounded-xl p-6 border-brand-500/10 bg-gradient-to-br from-slate-900/80 to-brand-500/5">
      <div className="flex items-start justify-between gap-6">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-xs font-medium text-brand-400 uppercase tracking-widest">Production RAG Evaluation</span>
          </div>
          <h2 className="text-lg font-bold text-slate-100 mb-2">
            Continuous quality measurement for RAG pipelines
          </h2>
          <p className="text-sm text-slate-400 max-w-xl leading-relaxed">
            VeriRAG runs{' '}
            <a href="https://docs.ragas.io" target="_blank" rel="noreferrer" className="text-brand-400 hover:text-brand-300">
              RAGAS
            </a>{' '}
            evaluations against your RAG pipeline — measuring faithfulness, answer relevancy,
            context precision, and recall across every version. Regressions are automatically
            detected and surfaced before they reach production.
          </p>
          <div className="flex flex-wrap items-center gap-3 mt-4">
            <StatPill label="Eval runs" value={totalRuns} />
            <StatPill label="Avg faithfulness" value={avgFaithfulness != null ? `${(avgFaithfulness * 100).toFixed(1)}%` : '—'} highlight={avgFaithfulness > 0.85} />
            <StatPill label="Pipeline" value="llama-3.3-70b / MMR" />
            <StatPill label="Embeddings" value="all-MiniLM-L6-v2" />
          </div>
        </div>
        <div className="hidden lg:flex flex-col items-end gap-2 shrink-0">
          <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          <span className="text-xs text-slate-600 writing-vertical">Live</span>
        </div>
      </div>
    </div>
  )
}

function StatPill({ label, value, highlight }) {
  return (
    <div className={`flex items-center gap-1.5 text-xs rounded-full px-3 py-1 border ${
      highlight
        ? 'bg-emerald-500/10 border-emerald-500/25 text-emerald-300'
        : 'bg-slate-800 border-slate-700 text-slate-400'
    }`}>
      <span className="text-slate-500">{label}:</span>
      <span className="font-semibold">{value}</span>
    </div>
  )
}

function MetricTooltip({ metric }) {
  const [show, setShow] = useState(false)
  const desc = METRIC_DESCRIPTIONS[metric]
  if (!desc) return null
  return (
    <div className="relative inline-block">
      <button
        onMouseEnter={() => setShow(true)}
        onMouseLeave={() => setShow(false)}
        className="w-4 h-4 rounded-full border border-slate-700 bg-slate-800 text-slate-500 text-xs flex items-center justify-center hover:border-slate-600 hover:text-slate-400 transition-colors"
      >
        ?
      </button>
      {show && (
        <div className="absolute left-6 top-0 z-50 w-52 bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-300 shadow-xl">
          {desc}
        </div>
      )}
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
  const avgFaithfulness = completed.length > 0
    ? completed.slice(0, 5).reduce((sum, r) => sum + (r.scores?.faithfulness ?? 0), 0) / Math.min(completed.length, 5)
    : null

  return (
    <div className="p-6 max-w-7xl space-y-5">
      {isError && <ApiErrorBanner message={error?.message} />}

      {/* Hero banner — shown when data exists so recruiters see product value immediately */}
      {!isLoading && completed.length > 0 && (
        <ProductHeroBanner
          latestRun={latest}
          totalRuns={runs.length}
          avgFaithfulness={avgFaithfulness}
        />
      )}

      <RegressionAlert />

      {activeCount > 0 && (
        <div className="rounded-xl border border-blue-500/20 bg-blue-500/5 px-4 py-2.5 flex items-center gap-2.5">
          <span className="w-2 h-2 rounded-full bg-blue-400 animate-pulse shrink-0" />
          <p className="text-xs text-blue-300">
            {activeCount} evaluation{activeCount > 1 ? 's' : ''} running — results appear automatically when complete
          </p>
        </div>
      )}

      {/* Metric cards */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-xs font-medium text-slate-500 uppercase tracking-wider">
            Latest Scores
            {latest && (
              <span className="ml-2 text-slate-600 normal-case font-normal">
                from <span className="font-mono text-slate-500">{latest.version_tag}</span>
              </span>
            )}
          </h2>
        </div>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          {METRICS.map((m) => (
            <div key={m} className="relative group">
              <MetricCard
                metric={m}
                score={latest?.scores?.[m] ?? null}
                delta={getDelta(m)}
                loading={isLoading}
              />
              <div className="absolute top-4 right-10 opacity-0 group-hover:opacity-100 transition-opacity">
                <MetricTooltip metric={m} />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Trend chart */}
      <div className="glass rounded-xl p-5">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-sm font-semibold text-slate-200">Metric Trends</h2>
            <p className="text-xs text-slate-500 mt-0.5">Score progression across evaluation runs</p>
          </div>
          {completed.length > 0 && (
            <span className="text-xs text-slate-600 tabular-nums">
              {completed.length} completed run{completed.length !== 1 ? 's' : ''}
            </span>
          )}
        </div>
        <MetricTrendChart runs={runs} loading={isLoading} />
      </div>

      {/* Runs table */}
      <EvalRunsTable
        runs={runs}
        loading={isLoading}
        onRunSample={handleRunSample}
      />

      {/* How it works — shown when no data yet */}
      {!isLoading && runs.length === 0 && (
        <div className="glass rounded-xl p-8 text-center space-y-4">
          <div className="w-12 h-12 rounded-xl bg-brand-500/15 border border-brand-500/25 flex items-center justify-center mx-auto">
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
              <path d="M3 14L8 9L12 13L17 7" stroke="#7494f9" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
              <circle cx="3" cy="14" r="1.5" fill="#7494f9"/>
              <circle cx="8" cy="9" r="1.5" fill="#7494f9"/>
              <circle cx="12" cy="13" r="1.5" fill="#7494f9"/>
              <circle cx="17" cy="7" r="1.5" fill="#7494f9"/>
            </svg>
          </div>
          <div>
            <h3 className="text-base font-semibold text-slate-200 mb-1">No evaluations yet</h3>
            <p className="text-sm text-slate-500 max-w-sm mx-auto">
              Run the sample evaluation to see RAGAS scores, regression detection, and metric trends.
            </p>
          </div>
          <button
            onClick={handleRunSample}
            disabled={startSample.isPending}
            className="mx-auto block px-5 py-2.5 rounded-lg bg-brand-500 hover:bg-brand-600 text-white text-sm font-medium transition-colors disabled:opacity-50"
          >
            {startSample.isPending ? 'Starting evaluation…' : 'Run Sample Evaluation'}
          </button>
        </div>
      )}
    </div>
  )
}
