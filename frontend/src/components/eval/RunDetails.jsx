/**
 * RunDetails — full breakdown page for a single eval run.
 *
 * Sections:
 * 1. Status banner (running/completed/failed with live polling)
 * 2. Aggregate MetricCards (4 scores + delta vs compared-to run)
 * 3. Score distribution histograms (4 mini histograms)
 * 4. Regression details (if any)
 * 5. Per-case CaseTable (paginated)
 * 6. LangSmith trace link
 */
import { useParams, Link } from 'react-router-dom'
import { formatDistanceToNow } from 'date-fns'
import clsx from 'clsx'
import { useEvalRun } from '../../hooks/useEvalDetails.js'
import MetricCard from '../dashboard/MetricCard.jsx'
import ScoreDistribution from './ScoreDistribution.jsx'
import CaseTable from './CaseTable.jsx'
import OptimizationInsights from './OptimizationInsights.jsx'
import { METRIC_LABELS, fmtScore } from '../../utils/scoreColor.js'

const METRICS = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']

function StatusBanner({ run }) {
  if (run.status === 'running') {
    return (
      <div className="rounded-xl border border-blue-500/30 bg-blue-500/10 px-5 py-3 flex items-center gap-3 mb-6">
        <span className="w-2 h-2 rounded-full bg-blue-400 animate-pulse shrink-0" />
        <p className="text-sm text-blue-300">
          Evaluation in progress — auto-refreshing every 5 seconds…
        </p>
      </div>
    )
  }
  if (run.status === 'failed') {
    return (
      <div className="rounded-xl border border-red-500/30 bg-red-500/10 px-5 py-3 mb-6">
        <p className="text-sm text-red-300 font-medium">Evaluation failed</p>
        {run.error_message && (
          <p className="text-xs text-red-400/70 mt-1 font-mono">{run.error_message}</p>
        )}
      </div>
    )
  }
  return null
}

function RegressionBanner({ run }) {
  if (!run.has_regression) return null
  const flagged = Object.entries(run.regression_details || {})
    .filter(([, v]) => v?.is_regression)

  return (
    <div className="rounded-xl border border-red-500/30 bg-red-950/30 px-5 py-4 mb-6">
      <p className="text-sm font-semibold text-red-300 mb-2">
        Regression detected in this run
      </p>
      <div className="flex flex-wrap gap-3">
        {flagged.map(([metric, d]) => (
          <div key={metric} className="text-xs bg-red-900/30 border border-red-500/20 rounded-lg px-3 py-1.5">
            <span className="text-red-300 font-medium">{METRIC_LABELS[metric]}: </span>
            <span className="text-slate-300">
              {fmtScore(d.previous)} → {fmtScore(d.current)}
            </span>
            <span className="text-red-400 ml-1">
              ({(d.delta * 100).toFixed(1)}%)
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function RunDetails() {
  const { id } = useParams()
  const { data: run, isLoading } = useEvalRun(id)

  if (isLoading) {
    return (
      <div className="p-6 space-y-4">
        <div className="skeleton h-8 w-64" />
        <div className="grid grid-cols-4 gap-4">
          {[...Array(4)].map((_, i) => <div key={i} className="skeleton h-28 rounded-xl" />)}
        </div>
        <div className="skeleton h-64 rounded-xl" />
      </div>
    )
  }

  if (!run) {
    return (
      <div className="p-6">
        <p className="text-slate-500">Run not found.</p>
        <Link to="/" className="text-brand-400 text-sm hover:underline mt-2 block">← Back to dashboard</Link>
      </div>
    )
  }

  const scores = run.scores ?? {}
  const hasCases = run.cases?.length > 0

  return (
    <div className="p-6 space-y-6 max-w-7xl">
      {/* Breadcrumb */}
      <div className="flex items-center gap-2 text-sm">
        <Link to="/" className="text-slate-500 hover:text-slate-300 transition-colors">Dashboard</Link>
        <span className="text-slate-700">/</span>
        <span className="text-slate-300 font-mono">{run.version_tag}</span>
        {run.status === 'completed' && run.completed_at && (
          <span className="text-slate-600 text-xs ml-2">
            completed {formatDistanceToNow(new Date(run.completed_at), { addSuffix: true })}
          </span>
        )}
      </div>

      {/* Status banners */}
      <StatusBanner run={run} />
      <RegressionBanner run={run} />

      {/* Metric cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {METRICS.map((m) => (
          <MetricCard
            key={m}
            metric={m}
            score={scores[m]}
            delta={null}
            loading={run.status === 'running'}
          />
        ))}
      </div>

      {/* Score distributions */}
      {hasCases && (
        <div className="glass rounded-xl p-5">
          <h3 className="text-sm font-semibold text-slate-200 mb-4">Score Distributions</h3>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-6">
            {METRICS.map((m) => (
              <ScoreDistribution key={m} cases={run.cases} metric={m} />
            ))}
          </div>
        </div>
      )}

      {/* Run metadata */}
      <div className="glass rounded-xl p-5">
        <h3 className="text-sm font-semibold text-slate-200 mb-3">Run Configuration</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
          {Object.entries(run.run_metadata || {}).slice(0, 8).map(([k, v]) => (
            <div key={k}>
              <p className="text-slate-500 uppercase tracking-wider mb-0.5">{k.replace(/_/g, ' ')}</p>
              <p className="text-slate-300 font-mono truncate">{String(v)}</p>
            </div>
          ))}
        </div>
        {run.langsmith_run_url && (
          <a
            href={run.langsmith_run_url}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1.5 mt-4 text-xs text-slate-400 hover:text-slate-200 border border-slate-700 hover:border-slate-600 px-3 py-1.5 rounded-lg transition-colors"
          >
            <span>View in LangSmith</span>
            <span>↗</span>
          </a>
        )}
      </div>

      {/* Optimization insights — only shown for completed runs */}
      {run.status === 'completed' && (
        <OptimizationInsights scores={scores} />
      )}

      {/* Pipeline config used in this run */}
      {run.run_metadata?.pipeline_config && (
        <div className="glass rounded-xl p-5">
          <h3 className="text-sm font-semibold text-slate-200 mb-3">Pipeline Configuration</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
            {[
              ['Chunk Size', `${run.run_metadata.pipeline_config.chunk_size} tokens`],
              ['Chunk Overlap', `${run.run_metadata.pipeline_config.chunk_overlap} tokens`],
              ['Retrieval', run.run_metadata.pipeline_config.retrieval_strategy],
              ['Top-K', run.run_metadata.pipeline_config.top_k],
              ['MMR Lambda', run.run_metadata.pipeline_config.mmr_lambda],
              ['Embedding', run.run_metadata.pipeline_config.embedding],
              ['LLM', run.run_metadata.pipeline_config.llm],
            ].map(([k, v]) => (
              <div key={k}>
                <p className="text-slate-500 uppercase tracking-wider mb-0.5">{k}</p>
                <p className="text-slate-300 font-mono truncate">{String(v)}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Per-case table */}
      <CaseTable runId={id} />
    </div>
  )
}
