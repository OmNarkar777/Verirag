/**
 * EvalRunsTable — table of all eval runs with scores, status badges, regression flags.
 *
 * Columns: version, pipeline, date, cases, faithfulness, relevancy,
 *          precision, recall, status, regression badge, actions.
 *
 * Regression runs get a red badge + row highlight so they're unmissable.
 */
import { useState } from 'react'
import { Link } from 'react-router-dom'
import { formatDistanceToNow } from 'date-fns'
import clsx from 'clsx'
import { useDeleteEvalRun } from '../../hooks/useEvalRuns.js'
import { scoreColorClass, fmtScore } from '../../utils/scoreColor.js'

function StatusBadge({ status }) {
  const cfg = {
    completed: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
    running:   'bg-blue-500/10 text-blue-400 border-blue-500/20',
    failed:    'bg-red-500/10 text-red-400 border-red-500/20',
    pending:   'bg-slate-700 text-slate-400 border-slate-600',
  }[status] ?? 'bg-slate-700 text-slate-400 border-slate-600'

  return (
    <span className={clsx('text-xs px-2 py-0.5 rounded-full border font-medium', cfg)}>
      {status === 'running' && <span className="mr-1 inline-block w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" />}
      {status}
    </span>
  )
}

function ScoreCell({ score }) {
  return (
    <span className={clsx('font-mono text-xs tabular-nums', scoreColorClass(score, 'text'))}>
      {fmtScore(score)}
    </span>
  )
}

function SkeletonRow() {
  return (
    <tr className="border-b border-slate-800">
      {[...Array(9)].map((_, i) => (
        <td key={i} className="px-4 py-3">
          <div className="skeleton h-3 rounded w-full" />
        </td>
      ))}
    </tr>
  )
}

export default function EvalRunsTable({ runs = [], loading = false, onRunSample, onCompare, compareA, compareB }) {
  const [deleting, setDeleting] = useState(null)
  const [showFailed, setShowFailed] = useState(false)
  const deleteMutation = useDeleteEvalRun()

  const failedCount = runs.filter((r) => r.status === 'failed').length
  const visibleRuns = showFailed ? runs : runs.filter((r) => r.status !== 'failed')

  const handleDelete = async (runId) => {
    if (!confirm('Delete this eval run and all its case results?')) return
    setDeleting(runId)
    try {
      await deleteMutation.mutateAsync(runId)
    } finally {
      setDeleting(null)
    }
  }

  return (
    <div className="glass rounded-xl overflow-hidden">
      {/* Table header */}
      <div className="px-5 py-4 border-b border-slate-800 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-slate-200">Evaluation Runs</h2>
        <div className="flex items-center gap-3">
          {failedCount > 0 && (
            <button
              onClick={() => setShowFailed((v) => !v)}
              className="text-xs text-slate-500 hover:text-slate-300 transition-colors"
            >
              {showFailed ? `Hide ${failedCount} failed` : `Show ${failedCount} failed`}
            </button>
          )}
          <button
            onClick={onRunSample}
            className="text-xs bg-brand-500 hover:bg-brand-600 text-white px-3 py-1.5 rounded-lg font-medium transition-colors"
          >
            + Run Sample Eval
          </button>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-800 text-xs text-slate-500 uppercase tracking-wider">
              {['Version', 'Pipeline', 'Date', 'Cases', 'Faith.', 'Relev.', 'Prec.', 'Recall', 'Status', ''].map((h) => (
                <th key={h} className="px-4 py-3 text-left font-medium whitespace-nowrap">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {loading && [...Array(5)].map((_, i) => <SkeletonRow key={i} />)}

            {!loading && visibleRuns.length === 0 && (
              <tr>
                <td colSpan={10} className="px-4 py-12 text-center text-slate-600 text-sm">
                  No evaluation runs yet.{' '}
                  <button onClick={onRunSample} className="text-brand-400 hover:text-brand-300 underline">
                    Run the sample evaluation
                  </button>{' '}
                  to get started.
                </td>
              </tr>
            )}

            {!loading && visibleRuns.map((run) => {
              const isSelectedA = compareA?.id === run.id
              const isSelectedB = compareB?.id === run.id
              const isSelected = isSelectedA || isSelectedB
              return (
              <tr
                key={run.id}
                className={clsx(
                  'border-b border-slate-800/60 hover:bg-slate-800/30 transition-colors',
                  run.has_regression && 'bg-red-950/20',
                  isSelectedA && 'ring-1 ring-inset ring-brand-500/30 bg-brand-500/5',
                  isSelectedB && 'ring-1 ring-inset ring-emerald-500/30 bg-emerald-500/5',
                )}
              >
                {/* Version */}
                <td className="px-4 py-3">
                  <div className="flex items-center gap-2">
                    <Link
                      to={`/runs/${run.id}`}
                      className="text-brand-400 hover:text-brand-300 font-mono text-xs hover:underline"
                    >
                      {run.version_tag}
                    </Link>
                    {run.has_regression && (
                      <span className="text-xs bg-red-500/15 text-red-400 border border-red-500/20 px-1.5 py-0.5 rounded font-medium">
                        ↓ regression
                      </span>
                    )}
                  </div>
                </td>

                {/* Pipeline */}
                <td className="px-4 py-3 text-slate-400 text-xs max-w-32 truncate">
                  {run.pipeline_name}
                </td>

                {/* Date */}
                <td className="px-4 py-3 text-slate-500 text-xs whitespace-nowrap">
                  {formatDistanceToNow(new Date(run.created_at), { addSuffix: true })}
                </td>

                {/* Cases */}
                <td className="px-4 py-3 text-slate-400 text-xs text-center">
                  {run.total_cases}
                </td>

                {/* Scores */}
                <td className="px-4 py-3"><ScoreCell score={run.scores?.faithfulness} /></td>
                <td className="px-4 py-3"><ScoreCell score={run.scores?.answer_relevancy} /></td>
                <td className="px-4 py-3"><ScoreCell score={run.scores?.context_precision} /></td>
                <td className="px-4 py-3"><ScoreCell score={run.scores?.context_recall} /></td>

                {/* Status */}
                <td className="px-4 py-3"><StatusBadge status={run.status} /></td>

                {/* Actions */}
                <td className="px-4 py-3">
                  <div className="flex items-center gap-2 whitespace-nowrap">
                    <Link
                      to={`/runs/${run.id}`}
                      className="text-xs text-slate-500 hover:text-slate-300 transition-colors"
                    >
                      View
                    </Link>
                    {onCompare && run.status === 'completed' && (
                      <button
                        onClick={() => onCompare(run)}
                        className={clsx(
                          'text-xs transition-colors',
                          isSelected ? 'text-brand-400' : 'text-slate-600 hover:text-brand-400',
                        )}
                      >
                        {isSelected ? 'Selected' : 'Compare'}
                      </button>
                    )}
                    <button
                      onClick={() => handleDelete(run.id)}
                      disabled={deleting === run.id}
                      className="text-xs text-slate-600 hover:text-red-400 transition-colors disabled:opacity-40"
                    >
                      {deleting === run.id ? '...' : 'Delete'}
                    </button>
                  </div>
                </td>
              </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
