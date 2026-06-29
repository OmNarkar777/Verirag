/**
 * PipelineConfigPanel — displays the current RAG pipeline configuration.
 * Shows engineering decisions: chunking, embedding, retrieval strategy, LLM.
 * This is the "Baseline Pipeline" panel of the RAG optimization workflow.
 *
 * EMBEDDING MODE TRANSPARENCY:
 * The panel prominently flags when TF-IDF fallback is active. This is not
 * a cosmetic note — TF-IDF produces keyword-based (not semantic) retrieval,
 * which significantly degrades context quality. Engineers must know this.
 */
import { usePipelineConfig, usePipelineStats } from '../../hooks/usePipelineConfig.js'

const TFIDF_MARKER = 'TF-IDF'

function EmbeddingModeAlert({ model }) {
  const isFallback = model?.includes(TFIDF_MARKER)
  if (!isFallback) return null

  return (
    <div className="rounded-lg border border-amber-500/40 bg-amber-500/8 px-3 py-3 mb-4">
      <div className="flex items-start gap-2.5">
        <svg className="w-4 h-4 text-amber-400 mt-0.5 shrink-0" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
          <path d="M8 2L14 13H2L8 2z" strokeLinejoin="round"/>
          <path d="M8 7v3M8 11.5v.5" strokeLinecap="round"/>
        </svg>
        <div>
          <p className="text-xs font-semibold text-amber-400 mb-1">TF-IDF Fallback Active — Reduced Retrieval Quality</p>
          <p className="text-xs text-amber-300/70 leading-relaxed">
            Semantic embeddings unavailable. Retrieval is keyword-based, not semantic — similar phrasing
            with different words will NOT match. Set <code className="bg-amber-500/15 px-1 rounded font-mono">HF_TOKEN</code> in
            Vercel environment variables to enable <code className="bg-amber-500/15 px-1 rounded font-mono">sentence-transformers/all-MiniLM-L6-v2</code>.
          </p>
          <p className="text-xs text-amber-500/60 mt-1.5">
            Note: retrieval scores range -0.5 to +0.5 (not 0.3–0.95 as with semantic models).
            Confidence badges are calibrated for this range.
          </p>
        </div>
      </div>
    </div>
  )
}

function ConfigSection({ title, children }) {
  return (
    <div>
      <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">{title}</p>
      <div className="space-y-1.5">{children}</div>
    </div>
  )
}

function ConfigRow({ label, value, mono = true }) {
  return (
    <div className="flex items-center justify-between gap-4 py-1 border-b border-slate-800/50 last:border-0">
      <span className="text-xs text-slate-500">{label}</span>
      <span className={`text-xs text-slate-300 truncate max-w-[60%] text-right ${mono ? 'font-mono' : ''}`}>
        {value}
      </span>
    </div>
  )
}

export default function PipelineConfigPanel() {
  const { data: config, isLoading } = usePipelineConfig()
  const { data: stats } = usePipelineStats()

  if (isLoading) {
    return (
      <div className="glass rounded-xl p-5">
        <div className="skeleton h-4 w-40 mb-4 rounded" />
        <div className="space-y-2">
          {[...Array(6)].map((_, i) => <div key={i} className="skeleton h-3 rounded" />)}
        </div>
      </div>
    )
  }

  if (!config) return null

  const docCount = stats?.document_count ?? '—'
  const embeddingModel = config.embedding?.model ?? ''

  return (
    <div className="glass rounded-xl p-5">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-sm font-semibold text-slate-200">Pipeline Configuration</h3>
          <p className="text-xs text-slate-500 mt-0.5">Baseline RAG architecture — saved with each eval run</p>
        </div>
        <div className="text-right">
          <p className="text-lg font-mono font-bold text-brand-400">{docCount}</p>
          <p className="text-xs text-slate-600">chunks indexed</p>
        </div>
      </div>

      <EmbeddingModeAlert model={embeddingModel} />

      <div className="space-y-4">
        <ConfigSection title="Chunking">
          <ConfigRow label="Strategy" value={config.chunking?.strategy} />
          <ConfigRow label="Chunk size" value={`${config.chunking?.chunk_size} tokens`} />
          <ConfigRow label="Overlap" value={`${config.chunking?.chunk_overlap} tokens`} />
        </ConfigSection>

        <ConfigSection title="Embedding">
          <ConfigRow label="Model" value={embeddingModel} mono={false} />
          <ConfigRow label="Dimensions" value={config.embedding?.dimensions} />
          <ConfigRow
            label="Mode"
            value={embeddingModel.includes(TFIDF_MARKER) ? 'TF-IDF (keyword)' : 'Semantic (vector)'}
            mono={false}
          />
        </ConfigSection>

        <ConfigSection title="Retrieval">
          <ConfigRow label="Strategy" value={config.retrieval?.strategy} mono={false} />
          <ConfigRow label="Top-K" value={config.retrieval?.top_k} />
          <ConfigRow label="MMR Lambda" value={config.retrieval?.mmr_lambda} />
          <ConfigRow label="Candidate pool" value={`${config.retrieval?.fetch_k} chunks`} />
        </ConfigSection>

        <ConfigSection title="Generation">
          <ConfigRow label="LLM" value={config.generation?.llm} />
          <ConfigRow label="Provider" value={config.generation?.provider} mono={false} />
          <ConfigRow label="Temperature" value={config.generation?.temperature} />
        </ConfigSection>
      </div>
    </div>
  )
}
