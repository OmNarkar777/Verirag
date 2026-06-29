/**
 * QueryPanel — RAG query interface with Experiment Mode controls.
 *
 * Experiment Mode lets engineers change retrieval parameters without code changes:
 * - Top-K: number of chunks returned to the LLM
 * - MMR toggle: Maximal Marginal Relevance vs cosine similarity
 * - fetch_k: candidate pool size for MMR (larger = more diverse candidates)
 * - MMR lambda: 1.0 = pure relevance, 0.0 = pure diversity
 * Each query includes its retrieval config in the response for traceability.
 */
import { useState } from 'react'
import clsx from 'clsx'
import { useQueryPipeline } from '../../hooks/usePipeline.js'
import { scoreColorClass, fmtScore } from '../../utils/scoreColor.js'

const EXAMPLE_QUESTIONS = [
  'What is the key innovation of the Transformer architecture?',
  'How does RAGAS measure faithfulness?',
  'Why does MMR retrieval improve context precision?',
]

const CONFIDENCE_CONFIG = {
  high:   { label: 'High confidence',   className: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/25', dot: 'bg-emerald-400' },
  medium: { label: 'Medium confidence', className: 'text-amber-400 bg-amber-500/10 border-amber-500/25',       dot: 'bg-amber-400'   },
  low:    { label: 'Low confidence',    className: 'text-red-400 bg-red-500/10 border-red-500/25',             dot: 'bg-red-400'     },
}

function ConfidenceBadge({ confidence }) {
  const cfg = CONFIDENCE_CONFIG[confidence] ?? CONFIDENCE_CONFIG.medium
  return (
    <span className={clsx('flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full border font-medium', cfg.className)}>
      <span className={clsx('w-1.5 h-1.5 rounded-full', cfg.dot)} />
      {cfg.label}
    </span>
  )
}

function ExperimentControls({ topK, setTopK, useMmr, setUseMmr, fetchK, setFetchK, mmrLambda, setMmrLambda }) {
  const [open, setOpen] = useState(false)

  return (
    <div className="border border-slate-800 rounded-lg">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between px-3 py-2 text-xs text-slate-500 hover:text-slate-400 transition-colors"
      >
        <span className="flex items-center gap-2">
          <svg className="w-3.5 h-3.5 text-brand-400" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M2 4h12M4 8h8M6 12h4" strokeLinecap="round"/>
          </svg>
          <span className="font-medium">Experiment Mode</span>
          {/* live summary of non-default values */}
          <span className="text-slate-700">
            top_k={topK} · {useMmr ? `MMR (λ={mmrLambda}, pool={fetchK})` : 'cosine'}
          </span>
        </span>
        <svg className={clsx('w-3.5 h-3.5 transition-transform', open && 'rotate-180')} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
          <path d="M4 6l4 4 4-4" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      </button>

      {open && (
        <div className="border-t border-slate-800 px-3 py-3 space-y-4">
          <p className="text-xs text-slate-600">
            Change retrieval parameters per-query without modifying code.
            Each result reflects these exact settings — use this to run controlled experiments.
          </p>

          <div className="grid grid-cols-2 gap-4">
            {/* Top-K */}
            <div>
              <label className="text-xs text-slate-500 block mb-1.5">
                Top-K <span className="text-slate-700">— chunks sent to LLM</span>
              </label>
              <div className="flex items-center gap-2">
                <input
                  type="range" min={1} max={20} step={1} value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                  className="flex-1 accent-brand-500"
                />
                <span className="text-xs text-slate-300 font-mono w-6 text-right">{topK}</span>
              </div>
            </div>

            {/* Retrieval strategy */}
            <div>
              <label className="text-xs text-slate-500 block mb-1.5">Retrieval Strategy</label>
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => setUseMmr(true)}
                  className={clsx(
                    'flex-1 py-1 rounded text-xs font-medium border transition-colors',
                    useMmr
                      ? 'bg-brand-500/20 border-brand-500/50 text-brand-300'
                      : 'bg-slate-800 border-slate-700 text-slate-500 hover:text-slate-400'
                  )}
                >
                  MMR
                </button>
                <button
                  type="button"
                  onClick={() => setUseMmr(false)}
                  className={clsx(
                    'flex-1 py-1 rounded text-xs font-medium border transition-colors',
                    !useMmr
                      ? 'bg-brand-500/20 border-brand-500/50 text-brand-300'
                      : 'bg-slate-800 border-slate-700 text-slate-500 hover:text-slate-400'
                  )}
                >
                  Cosine
                </button>
              </div>
            </div>
          </div>

          {useMmr && (
            <div className="grid grid-cols-2 gap-4">
              {/* MMR Lambda */}
              <div>
                <label className="text-xs text-slate-500 block mb-1.5">
                  MMR Lambda (λ={mmrLambda})
                  <span className="text-slate-700 ml-1">— 1.0=relevance, 0.0=diversity</span>
                </label>
                <input
                  type="range" min={0} max={1} step={0.05} value={mmrLambda}
                  onChange={(e) => setMmrLambda(Number(e.target.value))}
                  className="w-full accent-brand-500"
                />
                <div className="flex justify-between text-xs text-slate-700 mt-0.5">
                  <span>max diversity</span>
                  <span>max relevance</span>
                </div>
              </div>

              {/* fetch_k */}
              <div>
                <label className="text-xs text-slate-500 block mb-1.5">
                  Candidate Pool (fetch_k={fetchK})
                  <span className="text-slate-700 ml-1">— larger = better MMR diversity</span>
                </label>
                <div className="flex items-center gap-2">
                  <input
                    type="range" min={5} max={100} step={5} value={fetchK}
                    onChange={(e) => setFetchK(Number(e.target.value))}
                    className="flex-1 accent-brand-500"
                  />
                  <span className="text-xs text-slate-300 font-mono w-8 text-right">{fetchK}</span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default function QueryPanel({ onEvalCase }) {
  const [question, setQuestion] = useState('')
  const [topK, setTopK] = useState(5)
  const [useMmr, setUseMmr] = useState(true)
  const [fetchK, setFetchK] = useState(20)
  const [mmrLambda, setMmrLambda] = useState(0.5)
  const queryMutation = useQueryPipeline()

  const handleQuery = () => {
    if (!question.trim()) return
    queryMutation.mutate({ question: question.trim(), topK, useMmr, fetchK, mmrLambda })
  }

  const result = queryMutation.data

  const handleUseAsEval = () => {
    if (!result || !onEvalCase) return
    onEvalCase({
      question: result.question,
      answer: result.answer,
      contexts: result.retrieved_chunks.map((c) => c.content),
    })
  }

  return (
    <div className="space-y-4">
      {/* Input */}
      <div className="glass rounded-xl p-4 space-y-3">
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && e.metaKey && handleQuery()}
          placeholder="Ask a question about your ingested documents…"
          rows={3}
          className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 placeholder-slate-600 focus:outline-none focus:border-brand-500 resize-none"
        />

        <ExperimentControls
          topK={topK} setTopK={setTopK}
          useMmr={useMmr} setUseMmr={setUseMmr}
          fetchK={fetchK} setFetchK={setFetchK}
          mmrLambda={mmrLambda} setMmrLambda={setMmrLambda}
        />

        <div className="flex items-center justify-between">
          <span className="text-xs text-slate-600">⌘↵ to submit</span>
          <button
            onClick={handleQuery}
            disabled={!question.trim() || queryMutation.isPending}
            className="px-4 py-2 rounded-lg bg-brand-500 hover:bg-brand-600 text-white text-sm font-medium transition-colors disabled:opacity-50"
          >
            {queryMutation.isPending ? 'Querying…' : 'Query Pipeline'}
          </button>
        </div>

        {/* Example questions */}
        <div className="flex flex-wrap gap-2 pt-1 border-t border-slate-800">
          <span className="text-xs text-slate-600">Try:</span>
          {EXAMPLE_QUESTIONS.map((q) => (
            <button
              key={q}
              onClick={() => setQuestion(q)}
              className="text-xs text-slate-500 hover:text-slate-300 underline underline-offset-2 transition-colors text-left"
            >
              {q.length > 55 ? q.slice(0, 55) + '…' : q}
            </button>
          ))}
        </div>
      </div>

      {/* Error */}
      {queryMutation.isError && (
        <div className="rounded-xl border border-red-500/30 bg-red-500/5 px-4 py-3 text-xs text-red-400">
          {queryMutation.error?.message}
        </div>
      )}

      {/* Result */}
      {result && (
        <div className="space-y-4">
          {/* Answer */}
          <div className="glass rounded-xl p-5">
            <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
              <div className="flex items-center gap-3">
                <h3 className="text-sm font-semibold text-slate-200">Answer</h3>
                <ConfidenceBadge confidence={result.confidence} />
              </div>
              <div className="flex items-center gap-3">
                <span className="text-xs text-slate-600 font-mono">{result.model_used}</span>
                {onEvalCase && (
                  <button
                    onClick={handleUseAsEval}
                    className="text-xs text-brand-400 hover:text-brand-300 border border-brand-500/30 px-2.5 py-1 rounded-lg transition-colors"
                  >
                    + Use as Eval Case
                  </button>
                )}
              </div>
            </div>

            {/* Experiment config used for this result */}
            <div className="mb-3 flex flex-wrap gap-1.5">
              <span className="text-xs bg-slate-800 border border-slate-700 text-slate-500 px-2 py-0.5 rounded font-mono">
                top_k={topK}
              </span>
              <span className="text-xs bg-slate-800 border border-slate-700 text-slate-500 px-2 py-0.5 rounded font-mono">
                {useMmr ? `MMR λ=${mmrLambda} pool=${fetchK}` : 'cosine'}
              </span>
            </div>

            {result.confidence === 'low' && result.retrieved_chunks.length === 0 && (
              <div className="mb-3 rounded-lg border border-amber-500/20 bg-amber-500/5 px-3 py-2 text-xs text-amber-400">
                No documents indexed yet. Upload documents first, or try the pre-loaded AI/ML topics.
              </div>
            )}
            {result.confidence === 'low' && result.retrieved_chunks.length > 0 && (
              <div className="mb-3 rounded-lg border border-amber-500/20 bg-amber-500/5 px-3 py-2 text-xs text-amber-400">
                Low retrieval score — the indexed documents may not fully cover this topic.
              </div>
            )}

            <p className="text-sm text-slate-300 leading-relaxed">{result.answer}</p>
          </div>

          {/* Retrieved chunks */}
          <div className="glass rounded-xl p-5">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold text-slate-200">Retrieved Context</h3>
              <span className="text-xs text-slate-500">
                {result.retrieved_chunks.length} chunks ·{' '}
                <code className="text-slate-400">contexts</code> in RAGAS eval
              </span>
            </div>
            <div className="space-y-3">
              {result.retrieved_chunks.map((chunk, i) => (
                <div key={i} className="border border-slate-800 rounded-lg p-3">
                  <div className="flex items-center justify-between mb-2 gap-2">
                    <div className="flex items-center gap-2 min-w-0">
                      <span className="text-xs text-slate-600 shrink-0 tabular-nums">#{i + 1}</span>
                      <span className="text-xs text-slate-400 font-mono truncate">{chunk.source}</span>
                    </div>
                    <span className={clsx(
                      'text-xs font-mono tabular-nums shrink-0',
                      scoreColorClass(chunk.score, 'text'),
                    )}>
                      {fmtScore(chunk.score)}
                    </span>
                  </div>
                  <p className="text-xs text-slate-400 leading-relaxed">{chunk.content}</p>
                </div>
              ))}
            </div>
          </div>

          {/* LangSmith link */}
          {result.langsmith_trace_url && (
            <p className="text-xs text-slate-600">
              Trace:{' '}
              <a
                href={result.langsmith_trace_url}
                target="_blank"
                rel="noreferrer"
                className="text-slate-400 hover:text-slate-200 underline"
              >
                LangSmith ↗
              </a>
            </p>
          )}
        </div>
      )}
    </div>
  )
}
