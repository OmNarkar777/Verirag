/**
 * DashboardPage — metric overview, trend chart, regression alerts, run history.
 *
 * Performance: uses /eval/dashboard (combined endpoint) — one DB round trip
 * instead of two. Module-level cache on backend means warm requests return
 * in <100ms. All sections render skeletons immediately; nothing blocks on data.
 */
import { useState } from 'react'
import clsx from 'clsx'
import { useDashboard, useStartSampleEval, DASHBOARD_KEY } from '../hooks/useEvalRuns.js'
import { useQueryClient } from '@tanstack/react-query'
import MetricCard from '../components/dashboard/MetricCard.jsx'
import MetricTrendChart from '../components/dashboard/MetricTrendChart.jsx'
import EvalRunsTable from '../components/dashboard/EvalRunsTable.jsx'
import { scoreColorClass, fmtScore } from '../utils/scoreColor.js'

const METRICS = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']

const METRIC_LABELS = {
  faithfulness: 'Faithfulness',
  answer_relevancy: 'Answer Relevancy',
  context_precision: 'Context Precision',
  context_recall: 'Context Recall',
}

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

function avgScore(run) {
  const s = run.scores
  if (!s) return 0
  const vals = [s.faithfulness, s.answer_relevancy, s.context_precision, s.context_recall]
    .filter((v) => v != null)
  return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0
}

function BestPipelineBanner({ runs }) {
  const completed = runs.filter((r) => r.status === 'completed' && r.scores)
  if (completed.length < 2) return null

  const best = completed.reduce((a, b) => avgScore(a) >= avgScore(b) ? a : b)
  const bestAvg = avgScore(best)

  return (
    <div className="rounded-xl border border-brand-500/25 bg-brand-500/5 px-5 py-4">
      <div className="flex items-center gap-3">
        <span className="text-base">&#127942;</span>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <p className="text-sm font-semibold text-slate-200">Best Performing Pipeline</p>
            <code className="text-xs text-brand-400 font-mono">{best.version_tag}</code>
            <span className="text-xs text-slate-500">
              avg score: <span className="text-slate-300 font-mono">{(bestAvg * 100).toFixed(1)}%</span>
            </span>
          </div>
          <div className="flex flex-wrap gap-3 mt-1.5">
            {METRICS.map((m) => best.scores?.[m] != null && (
              <span key={m} className="text-xs text-slate-500">
                {METRIC_LABELS[m].split(' ')[0]}:{' '}
                <span className={clsx('font-mono font-medium', scoreColorClass(best.scores[m], 'text'))}>
                  {fmtScore(best.scores[m])}
                </span>
              </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

function RegressionBanner({ regression }) {
  if (!regression) return null
  const flaggedMetrics = Object.entries(regression.regression_details || {})
    .filter(([, v]) => v?.is_regression)
  if (flaggedMetrics.length === 0) return null

  return (
    <div className="rounded-xl border border-red-500/40 bg-red-500/10 p-4">
      <div className="flex items-start gap-3">
        <div className="mt-0.5 w-5 h-5 rounded-full bg-red-500/20 border border-red-500/40 flex items-center justify-center shrink-0">
          <span className="text-red-400 text-xs">!</span>
        </div>
        <div className="flex-1">
          <div className="flex items-center gap-2 flex-wrap">
            <p className="text-sm font-semibold text-red-300">
              Regression detected in {regression.version_tag}
            </p>
            <a href={`/runs/${regression.id}`} className="text-xs text-red-400/70 hover:text-red-300 underline underline-offset-2">
              View run →
            </a>
          </div>
          <div className="mt-2 flex flex-wrap gap-3">
            {flaggedMetrics.map(([metric, details]) => (
              <div key={metric} className="text-xs bg-red-900/30 border border-red-500/20 rounded-lg px-3 py-1.5">
                <span className="text-red-300 font-medium">{METRIC_LABELS[metric]}</span>
                <span className="text-slate-400 mx-1">
                  {fmtScore(details.previous)} → {fmtScore(details.current)}
                </span>
                <span className="text-red-400 font-semibold">
                  ({(details.delta * 100).toFixed(1)}%)
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

function RunComparisonPanel({ runA, runB, onClose }) {
  if (!runA || !runB) return null

  return (
    <div className="glass rounded-xl p-5">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-semibold text-slate-200">Run Comparison</h2>
        <button onClick={onClose} className="text-xs text-slate-500 hover:text-slate-300 transition-colors">
          Clear
        </button>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-800 text-xs text-slate-500 uppercase tracking-wider">
              <th className="text-left py-2 pr-4 font-medium">Metric</th>
              <th className="text-right py-2 px-4 font-medium font-mono text-slate-400">{runA.version_tag}</th>
              <th className="text-right py-2 px-4 font-medium font-mono text-slate-400">{runB.version_tag}</th>
              <th className="text-right py-2 pl-4 font-medium">Change</th>
            </tr>
          </thead>
          <tbody>
            {METRICS.map((m) => {
              const scoreA = runA.scores?.[m]
              const scoreB = runB.scores?.[m]
              const delta = scoreA != null && scoreB != null ? scoreB - scoreA : null
              return (
                <tr key={m} className="border-b border-slate-800/60">
                  <td className="py-2.5 pr-4 text-xs text-slate-400">{METRIC_LABELS[m]}</td>
                  <td className={clsx('py-2.5 px-4 text-right font-mono text-xs tabular-nums', scoreColorClass(scoreA, 'text'))}>
                    {fmtScore(scoreA)}
                  </td>
                  <td className={clsx('py-2.5 px-4 text-right font-mono text-xs tabular-nums', scoreColorClass(scoreB, 'text'))}>
                    {fmtScore(scoreB)}
                  </td>
                  <td className="py-2.5 pl-4 text-right">
                    {delta != null ? (
                      <span className={clsx(
                        'text-xs font-mono font-semibold tabular-nums',
                        delta > 0.01 ? 'text-emerald-400' : delta < -0.01 ? 'text-red-400' : 'text-slate-500',
                      )}>
                        {delta > 0 ? '+' : ''}{(delta * 100).toFixed(1)}%
                      </span>
                    ) : (
                      <span className="text-xs text-slate-700">—</span>
                    )}
                  </td>
                </tr>
              )
            })}
            <tr>
              <td className="pt-2.5 pr-4 text-xs text-slate-500 font-medium">Avg Score</td>
              <td className="pt-2.5 px-4 text-right font-mono text-xs text-slate-300">{(avgScore(runA) * 100).toFixed(1)}%</td>
              <td className="pt-2.5 px-4 text-right font-mono text-xs text-slate-300">{(avgScore(runB) * 100).toFixed(1)}%</td>
              <td className="pt-2.5 pl-4 text-right">
                {(() => {
                  const d = avgScore(runB) - avgScore(runA)
                  return (
                    <span className={clsx('text-xs font-mono font-semibold', d > 0.01 ? 'text-emerald-400' : d < -0.01 ? 'text-red-400' : 'text-slate-500')}>
                      {d > 0 ? '+' : ''}{(d * 100).toFixed(1)}%
                    </span>
                  )
                })()}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default function DashboardPage() {
  const { data, isLoading, isError, error } = useDashboard()
  const qc = useQueryClient()
  const startSample = useStartSampleEval({
    onSuccess: () => qc.invalidateQueries({ queryKey: DASHBOARD_KEY }),
  })
  const [compareA, setCompareA] = useState(null)
  const [compareB, setCompareB] = useState(null)

  const runs = data?.runs ?? []
  const latestRegression = data?.latest_regression ?? null

  const completed = runs.filter((r) => r.status === 'completed' && r.scores)
  const latest = completed[0] ?? null
  const previous = completed[1] ?? null

  const getDelta = (metric) => {
    if (!latest?.scores?.[metric] || !previous?.scores?.[metric]) return null
    return latest.scores[metric] - previous.scores[metric]
  }

  const handleRunSample = async () => {
    try {
      await startSample.mutateAsync()
    } catch (e) {
      console.error('Failed to start sample eval:', e.message)
    }
  }

  const handleCompare = (run) => {
    if (!compareA) {
      setCompareA(run)
    } else if (!compareB && run.id !== compareA.id) {
      setCompareB(run)
    } else {
      setCompareA(run)
      setCompareB(null)
    }
  }

  const activeCount = runs.filter((r) => r.status === 'running').length

  return (
    <div className="p-6 max-w-7xl space-y-5">
      {isError && <ApiErrorBanner message={error?.message} />}

      {/* Regression alert — shows from combined data, no extra API call */}
      {latestRegression && <RegressionBanner regression={latestRegression} />}

      {activeCount > 0 && (
        <div className="rounded-xl border border-blue-500/20 bg-blue-500/5 px-4 py-2.5 flex items-center gap-2.5">
          <span className="w-2 h-2 rounded-full bg-blue-400 animate-pulse shrink-0" />
          <p className="text-xs text-blue-300">
            {activeCount} evaluation{activeCount > 1 ? 's' : ''} running — results appear automatically when complete
          </p>
        </div>
      )}

      {!isLoading && completed.length >= 2 && (
        <BestPipelineBanner runs={completed} />
      )}

      {/* Metric cards — always render (skeleton when loading) */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-xs font-medium text-slate-500 uppercase tracking-wider">
            Latest Scores
            {latest && !isLoading && (
              <span className="ml-2 text-slate-600 normal-case font-normal">
                from <span className="font-mono text-slate-500">{latest.version_tag}</span>
              </span>
            )}
          </h2>
          {previous && !isLoading && (
            <span className="text-xs text-slate-600">
              vs <span className="font-mono">{previous.version_tag}</span>
            </span>
          )}
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

      {/* Trend chart — always renders (skeleton when loading) */}
      <div className="glass rounded-xl p-5">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-sm font-semibold text-slate-200">Metric Trends</h2>
            <p className="text-xs text-slate-500 mt-0.5">Score progression across evaluation runs</p>
          </div>
          {!isLoading && completed.length > 0 && (
            <span className="text-xs text-slate-600 tabular-nums">
              {completed.length} completed run{completed.length !== 1 ? 's' : ''}
            </span>
          )}
        </div>
        <MetricTrendChart runs={runs} loading={isLoading} />
      </div>

      {/* Run comparison panel */}
      {compareA && compareB && (
        <RunComparisonPanel
          runA={compareA}
          runB={compareB}
          onClose={() => { setCompareA(null); setCompareB(null) }}
        />
      )}

      {compareA && !compareB && (
        <div className="rounded-xl border border-brand-500/25 bg-brand-500/5 px-4 py-2.5 text-xs text-slate-400">
          <span className="text-brand-400 font-medium">{compareA.version_tag}</span> selected —
          click another completed run to compare
        </div>
      )}

      {/* Runs table — always renders (skeleton when loading) */}
      <EvalRunsTable
        runs={runs}
        loading={isLoading}
        onRunSample={handleRunSample}
        onCompare={handleCompare}
        compareA={compareA}
        compareB={compareB}
      />

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
