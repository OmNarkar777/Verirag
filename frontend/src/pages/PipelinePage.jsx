/**
 * PipelinePage â€” two-panel layout: document ingestion + RAG query interface.
 */
import { useState } from 'react'
import { startEvalRun } from '../api/client.js'
import { useStartSampleEval } from '../hooks/useEvalRuns.js'
import IngestPanel from '../components/pipeline/IngestPanel.jsx'
import QueryPanel from '../components/pipeline/QueryPanel.jsx'

export default function PipelinePage() {
  const [evalCase, setEvalCase] = useState(null)
  const [groundTruth, setGroundTruth] = useState('')
  const [versionTag, setVersionTag] = useState('v1.0.0-live')
  const [submitted, setSubmitted] = useState(false)
  const [submitError, setSubmitError] = useState('')
  const [sampleNotice, setSampleNotice] = useState(null) // { type: 'success'|'error', msg }
  const startSample = useStartSampleEval({
    onSuccess: (data) => setSampleNotice({
      type: 'success',
      msg: `RAGAS evaluation complete — run ${data?.eval_run_id?.slice(0, 8) ?? '…'} (${data?.version_tag ?? ''}). Dashboard and chart updated.`,
    }),
    onError: (err) => setSampleNotice({
      type: 'error',
      msg: err?.response?.data?.detail ?? err?.message ?? 'Failed to start evaluation.',
    }),
  })

  const handleEvalCase = (c) => {
    setEvalCase(c)
    setSubmitted(false)
    setSubmitError('')
    setTimeout(() => document.getElementById('eval-form')?.scrollIntoView({ behavior: 'smooth' }), 100)
  }

  const handleSubmitEval = async () => {
    if (!evalCase || !groundTruth.trim()) return
    try {
      await startEvalRun({
        version_tag: versionTag,
        pipeline_name: 'live-query-eval',
        test_cases: [{ ...evalCase, ground_truth: groundTruth }],
        metadata: { source: 'pipeline_query_panel' },
      })
      setSubmitted(true)
      setEvalCase(null)
      setGroundTruth('')
    } catch (e) {
      setSubmitError(e.message)
    }
  }

  return (
    <div className="p-6 max-w-7xl space-y-6">
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <div>
          <h2 className="text-xs font-medium text-slate-500 uppercase tracking-wider mb-3">Document Ingestion</h2>
          <IngestPanel />
        </div>
        <div>
          <h2 className="text-xs font-medium text-slate-500 uppercase tracking-wider mb-3">Query Pipeline</h2>
          <QueryPanel onEvalCase={handleEvalCase} />
        </div>
      </div>

      {evalCase && (
        <div id="eval-form" className="glass rounded-xl p-5">
          <h3 className="text-sm font-semibold text-slate-200 mb-1">Evaluate this Query</h3>
          <p className="text-xs text-slate-500 mb-4">Add a ground truth and submit to create a 1-case eval run.</p>
          <div className="space-y-3">
            <div>
              <label className="text-xs text-slate-500 block mb-1">Question</label>
              <div className="bg-slate-800 rounded-lg px-3 py-2 text-xs text-slate-300 border border-slate-700">{evalCase.question}</div>
            </div>
            <div>
              <label className="text-xs text-slate-500 block mb-1">Ground Truth <span className="text-red-400">*</span></label>
              <textarea
                value={groundTruth}
                onChange={(e) => setGroundTruth(e.target.value)}
                placeholder="The ideal, complete answer to this question..."
                rows={3}
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 placeholder-slate-600 focus:outline-none focus:border-brand-500 resize-none"
              />
            </div>
            <div className="flex items-center gap-3">
              <div className="flex-1">
                <label className="text-xs text-slate-500 block mb-1">Version Tag</label>
                <input
                  value={versionTag}
                  onChange={(e) => setVersionTag(e.target.value)}
                  className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-300 font-mono focus:outline-none focus:border-brand-500"
                />
              </div>
              <button
                onClick={handleSubmitEval}
                disabled={!groundTruth.trim()}
                className="mt-5 px-5 py-2 rounded-lg bg-brand-500 hover:bg-brand-600 text-white text-sm font-medium transition-colors disabled:opacity-50"
              >
                Submit Eval
              </button>
            </div>
            {submitError && <p className="text-xs text-red-400">{submitError}</p>}
          </div>
        </div>
      )}

      {submitted && (
        <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/5 px-4 py-3 text-xs">
          <p className="text-emerald-400 font-medium">Evaluation started</p>
          <p className="text-slate-400 mt-1">Check the <a href="/" className="text-brand-400 hover:underline">Dashboard</a> for results.</p>
        </div>
      )}

      <div className="glass rounded-xl p-5">
        <h3 className="text-sm font-semibold text-slate-200 mb-1">Quick Actions</h3>
        <p className="text-xs text-slate-500 mb-3">
          12 AI/ML reference documents are pre-loaded — query them immediately without uploading anything.
        </p>
        {sampleNotice && (
          <div className={`mb-3 rounded-lg border px-4 py-2.5 text-xs flex items-start justify-between gap-3 ${
            sampleNotice.type === 'success'
              ? 'border-emerald-500/30 bg-emerald-500/5 text-emerald-400'
              : 'border-red-500/30 bg-red-500/5 text-red-400'
          }`}>
            <span>{sampleNotice.msg}</span>
            <button onClick={() => setSampleNotice(null)} className="shrink-0 opacity-60 hover:opacity-100">✕</button>
          </div>
        )}
        <div className="flex flex-wrap gap-3">
          <button
            onClick={() => { setSampleNotice(null); startSample.mutate('v0.0.1-quick') }}
            disabled={startSample.isPending}
            className="text-xs border border-slate-700 hover:border-slate-600 text-slate-300 px-4 py-2 rounded-lg transition-colors disabled:opacity-50 flex items-center gap-2"
          >
            {startSample.isPending && (
              <svg className="animate-spin w-3 h-3 text-brand-400" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/>
              </svg>
            )}
            {startSample.isPending ? 'Running evaluation… (~10s)' : 'Run Sample Evaluation (5 cases, Groq LLM judge)'}
          </button>
          <a href="/docs" target="_blank" rel="noreferrer"
            className="text-xs border border-slate-700 hover:border-slate-600 text-slate-400 px-4 py-2 rounded-lg transition-colors">
            API Docs
          </a>
        </div>
        <div className="mt-4 pt-4 border-t border-slate-800">
          <p className="text-xs text-slate-600 mb-2 uppercase tracking-wider font-medium">Pre-loaded documents</p>
          <div className="flex flex-wrap gap-1.5">
            {[
              'Attention Is All You Need',
              'RAGAS Evaluation Framework',
              'RAG Pipeline Patterns',
              'Vector Databases Comparison',
              'Embeddings & Semantic Search',
              'LangChain Guide',
              'Prompt Engineering',
              'Chunking Strategies',
              'FastAPI for ML',
              'LLM Hallucination & Faithfulness',
              'Production RAG Deployment',
              'Retrieval Evaluation Metrics',
            ].map((doc) => (
              <span key={doc} className="text-xs bg-slate-800/80 border border-slate-700/50 text-slate-500 px-2 py-0.5 rounded">
                {doc}
              </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}