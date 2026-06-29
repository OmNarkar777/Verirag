/**
 * PipelineConfigPanel — displays the current RAG pipeline configuration.
 * Shows engineering decisions: chunking, embedding, retrieval strategy, LLM.
 * This is the "Baseline Pipeline" panel of the RAG optimization workflow.
 */
import { usePipelineConfig, usePipelineStats } from '../../hooks/usePipelineConfig.js'

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

  return (
    <div className="glass rounded-xl p-5">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-sm font-semibold text-slate-200">Pipeline Configuration</h3>
          <p className="text-xs text-slate-500 mt-0.5">Current RAG architecture settings</p>
        </div>
        <div className="text-right">
          <p className="text-lg font-mono font-bold text-brand-400">{docCount}</p>
          <p className="text-xs text-slate-600">chunks indexed</p>
        </div>
      </div>

      <div className="space-y-4">
        <ConfigSection title="Chunking">
          <ConfigRow label="Strategy" value={config.chunking?.strategy} />
          <ConfigRow label="Chunk size" value={`${config.chunking?.chunk_size} tokens`} />
          <ConfigRow label="Overlap" value={`${config.chunking?.chunk_overlap} tokens`} />
        </ConfigSection>

        <ConfigSection title="Embedding">
          <ConfigRow label="Model" value={config.embedding?.model} mono={false} />
          <ConfigRow label="Dimensions" value={config.embedding?.dimensions} />
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

      <div className="mt-4 pt-3 border-t border-slate-800">
        <p className="text-xs text-slate-600">
          Pipeline config is saved with every evaluation run for reproducible comparison.
        </p>
      </div>
    </div>
  )
}
