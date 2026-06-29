/**
 * OptimizationInsights — analyzes eval run scores and surfaces actionable
 * improvement recommendations based on known RAG failure modes.
 *
 * Each insight explains WHY a metric is low and WHAT to change,
 * mapping directly to concrete engineering actions.
 */

const THRESHOLDS = {
  good: 0.75,
  warn: 0.55,
}

function level(score) {
  if (score == null) return 'unknown'
  if (score >= THRESHOLDS.good) return 'good'
  if (score >= THRESHOLDS.warn) return 'warn'
  return 'poor'
}

const INSIGHTS = {
  faithfulness: {
    warn: {
      title: 'Moderate hallucination risk',
      why: 'The LLM is adding claims that go beyond the retrieved context passages. This happens when the system prompt doesn\'t enforce strict grounding, or when the retrieved chunks lack enough detail.',
      actions: [
        { label: 'Tighten grounding rules', detail: 'Strengthen the system prompt: "Answer ONLY from the provided context. Never add information from training knowledge."' },
        { label: 'Increase chunk overlap', detail: 'Higher overlap (100–150 tokens) prevents splitting related sentences, giving the LLM complete thoughts to cite.' },
        { label: 'Add source citation enforcement', detail: 'Require [source: filename] citations per claim — forces the model to check whether context supports each statement.' },
      ],
    },
    poor: {
      title: 'High hallucination detected',
      why: 'Most answer claims are not supported by retrieved context. The LLM is drawing on parametric knowledge instead of the indexed documents. This is the most critical RAG failure mode.',
      actions: [
        { label: 'Rebuild the system prompt', detail: 'Use explicit grounding rules: list what the model CAN and CANNOT do. Include "Do not use external knowledge."' },
        { label: 'Filter low-similarity chunks', detail: 'If chunks with <15% similarity score are reaching the LLM, they confuse the model and invite hallucination. Consider raising the retrieval threshold.' },
        { label: 'Reduce temperature to 0.0', detail: 'Temperature > 0 adds randomness that can cause the model to "elaborate" beyond context. Set temperature=0 for factual RAG.' },
      ],
    },
  },
  answer_relevancy: {
    warn: {
      title: 'Answers partially off-topic',
      why: 'The generated answers sometimes drift from the specific question asked. This often indicates the retrieved context is tangentially related but not directly answering the question.',
      actions: [
        { label: 'Increase top-K', detail: 'Retrieve more chunks (7–10) to increase the chance that the directly relevant passage is included in context.' },
        { label: 'Enable MMR with higher lambda', detail: 'MMR lambda=0.7 weights relevance over diversity — use when the answer quality matters more than context variety.' },
        { label: 'Add question restatement to prompt', detail: 'Include the question in the prompt multiple times: "Always answer this specific question: {question}" to keep the model focused.' },
      ],
    },
    poor: {
      title: 'Answers not addressing the question',
      why: 'The LLM is generating responses that don\'t answer what was asked. This is typically a retrieval failure — wrong chunks retrieved — combined with a prompt that doesn\'t force relevance.',
      actions: [
        { label: 'Use hybrid retrieval (BM25 + dense)', detail: 'Keyword-based BM25 retrieval captures exact term matches; dense retrieval captures semantics. Hybrid finds the best of both.' },
        { label: 'Add query expansion', detail: 'Before retrieval, expand the query with synonyms or rephrasings. Example: "What is RAG?" → also search "retrieval augmented generation definition".' },
        { label: 'Improve the prompt format', detail: 'Use explicit Q&A structure: "QUESTION: {question}\\nAnswer the question directly and completely using the context below."' },
      ],
    },
  },
  context_precision: {
    warn: {
      title: 'Some irrelevant chunks retrieved',
      why: 'The top-K retrieved chunks include passages that don\'t help answer the question. RAGAS penalizes off-topic chunks because they dilute the context and can confuse the LLM.',
      actions: [
        { label: 'Reduce top-K', detail: 'Fewer chunks (3–5) with higher average relevance beats more chunks with mixed relevance. Try top_k=3 for focused queries.' },
        { label: 'Increase MMR lambda', detail: 'lambda=0.7–0.8 prioritizes query relevance over diversity. Lower lambda (0.3–0.5) trades relevance for variety.' },
        { label: 'Add metadata filtering', detail: 'If documents span multiple topics, use metadata filters (e.g., category, date) to pre-filter the search space before vector similarity.' },
      ],
    },
    poor: {
      title: 'Retrieval returning mostly irrelevant context',
      why: 'The vector search is not finding relevant passages. This usually means the embedding model has poor semantic coverage for this domain, or the chunk size is mismatched to the query type.',
      actions: [
        { label: 'Upgrade to semantic embeddings', detail: 'The fallback TF-IDF embedder uses keyword matching. Upgrade to sentence-transformers/all-MiniLM-L6-v2 via HuggingFace API for true semantic similarity.' },
        { label: 'Reduce chunk size to 256 tokens', detail: 'Smaller chunks produce more specific embeddings. A 512-token chunk covering multiple topics will match fewer queries precisely.' },
        { label: 'Try BM25 retrieval for keyword-heavy queries', detail: 'For technical terms (acronyms, product names), BM25 often outperforms dense retrieval because it matches exact strings.' },
      ],
    },
  },
  context_recall: {
    warn: {
      title: 'Partial information coverage',
      why: 'The retrieved context is missing some information needed to fully answer the question. The LLM is working with an incomplete picture.',
      actions: [
        { label: 'Increase chunk overlap', detail: 'Overlap of 100–150 tokens ensures that key statements spanning chunk boundaries are captured in at least one chunk.' },
        { label: 'Increase top-K to 8–10', detail: 'Retrieve more chunks to maximize coverage. The answer may be spread across multiple passages.' },
        { label: 'Use parent-child chunking', detail: 'Index small chunks (128 tokens) for precise retrieval, but return their parent chunk (512 tokens) for full context. Balances precision and recall.' },
      ],
    },
    poor: {
      title: 'Critical information not retrieved',
      why: 'The ground truth answer requires information that the retrieval system is not surfacing. This is a coverage failure — either the document isn\'t indexed or the query embedding doesn\'t match the relevant passage.',
      actions: [
        { label: 'Verify document is indexed', detail: 'Check /pipeline/stats to confirm the document was ingested. Re-upload if chunk_count is 0.' },
        { label: 'Use semantic chunking', detail: 'Split on topic boundaries rather than fixed token counts. Semantic chunkers detect when the text shifts topic, producing more coherent retrievable units.' },
        { label: 'Add query rewriting', detail: 'Rephrase the question from multiple angles before retrieval. Example: "What is RAG?" → ["Define retrieval augmented generation", "How does RAG work?"]' },
      ],
    },
  },
}

function InsightCard({ metric, score, label }) {
  const l = level(score)
  if (l === 'good' || l === 'unknown') return null

  const insight = INSIGHTS[metric]?.[l]
  if (!insight) return null

  const borderColor = l === 'poor' ? 'border-red-500/25' : 'border-amber-500/25'
  const bgColor = l === 'poor' ? 'bg-red-950/20' : 'bg-amber-950/20'
  const titleColor = l === 'poor' ? 'text-red-300' : 'text-amber-300'
  const dotColor = l === 'poor' ? 'bg-red-400' : 'bg-amber-400'
  const badgeColor = l === 'poor'
    ? 'bg-red-900/40 text-red-400 border-red-500/20'
    : 'bg-amber-900/40 text-amber-400 border-amber-500/20'

  return (
    <div className={`rounded-xl border ${borderColor} ${bgColor} p-4`}>
      <div className="flex items-start gap-3 mb-3">
        <span className={`w-2 h-2 rounded-full ${dotColor} mt-1.5 shrink-0`} />
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs font-mono text-slate-500 uppercase tracking-wider">{label}</span>
            <span className={`text-xs px-1.5 py-0.5 rounded border font-medium ${badgeColor}`}>
              {Math.round(score * 100)}%
            </span>
          </div>
          <p className={`text-sm font-semibold ${titleColor}`}>{insight.title}</p>
          <p className="text-xs text-slate-400 mt-1 leading-relaxed">{insight.why}</p>
        </div>
      </div>
      <div className="space-y-2 ml-5">
        {insight.actions.map((action, i) => (
          <div key={i} className="border border-slate-700/50 rounded-lg px-3 py-2">
            <p className="text-xs font-semibold text-slate-300">{action.label}</p>
            <p className="text-xs text-slate-500 mt-0.5 leading-relaxed">{action.detail}</p>
          </div>
        ))}
      </div>
    </div>
  )
}

const METRIC_LABELS = {
  faithfulness: 'Faithfulness',
  answer_relevancy: 'Answer Relevancy',
  context_precision: 'Context Precision',
  context_recall: 'Context Recall',
}

export default function OptimizationInsights({ scores }) {
  if (!scores) return null

  const metrics = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
  const issues = metrics.filter((m) => {
    const s = scores[m]
    return s != null && s < THRESHOLDS.good
  })

  if (issues.length === 0) {
    const avg = metrics.reduce((s, m) => s + (scores[m] ?? 0), 0) / metrics.length
    return (
      <div className="glass rounded-xl p-5">
        <h3 className="text-sm font-semibold text-slate-200 mb-1">Optimization Insights</h3>
        <div className="flex items-center gap-3 mt-3">
          <span className="w-2 h-2 rounded-full bg-emerald-400 shrink-0" />
          <div>
            <p className="text-sm text-emerald-400 font-semibold">All metrics above threshold</p>
            <p className="text-xs text-slate-500 mt-0.5">
              Average score: {Math.round(avg * 100)}%. Pipeline is performing well.
              Continue monitoring across more diverse queries to validate.
            </p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="glass rounded-xl p-5">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-sm font-semibold text-slate-200">Optimization Insights</h3>
          <p className="text-xs text-slate-500 mt-0.5">
            {issues.length} metric{issues.length > 1 ? 's' : ''} below threshold — root causes and recommended fixes
          </p>
        </div>
        <span className="text-xs text-slate-600 border border-slate-700 rounded px-2 py-1">
          {issues.length} issue{issues.length > 1 ? 's' : ''}
        </span>
      </div>
      <div className="space-y-3">
        {metrics.map((m) => (
          <InsightCard
            key={m}
            metric={m}
            score={scores[m]}
            label={METRIC_LABELS[m]}
          />
        ))}
      </div>
    </div>
  )
}
