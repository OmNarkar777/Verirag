/**
 * IngestPanel — document ingestion with drag-and-drop file upload.
 */
import { useState, useRef, useCallback } from 'react'
import clsx from 'clsx'
import { useIngestFile, useIngestText, useDocuments, usePipelineStats } from '../../hooks/usePipeline.js'
import { formatDistanceToNow } from 'date-fns'

function UploadIcon({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" y1="3" x2="12" y2="15" />
    </svg>
  )
}

function SpinnerIcon({ className }) {
  return (
    <svg className={clsx('animate-spin', className)} viewBox="0 0 24 24" fill="none">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
    </svg>
  )
}

function FileIcon({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
      <polyline points="14 2 14 8 20 8" />
    </svg>
  )
}

function TextIcon({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <line x1="17" y1="10" x2="3" y2="10" />
      <line x1="21" y1="6" x2="3" y2="6" />
      <line x1="21" y1="14" x2="3" y2="14" />
      <line x1="17" y1="18" x2="3" y2="18" />
    </svg>
  )
}

function DropZone({ isDragging }) {
  return (
    <div className={clsx(
      'border-2 border-dashed rounded-xl p-10 text-center transition-all',
      isDragging
        ? 'border-brand-500 bg-brand-500/5'
        : 'border-slate-700 hover:border-slate-600 hover:bg-slate-800/20',
    )}>
      <div className="flex justify-center mb-4">
        <div className="w-12 h-12 rounded-xl bg-slate-800 border border-slate-700 flex items-center justify-center">
          <UploadIcon className="w-6 h-6 text-slate-400" />
        </div>
      </div>
      <p className="text-sm font-medium text-slate-300 mb-1">
        Drop files here or <span className="text-brand-400">click to browse</span>
      </p>
      <p className="text-xs text-slate-600">Supported: .txt, .md, .pdf</p>
    </div>
  )
}

function ProcessingState() {
  return (
    <div className="border-2 border-dashed border-brand-500/40 rounded-xl p-10 text-center">
      <div className="flex justify-center mb-4">
        <SpinnerIcon className="w-6 h-6 text-brand-400" />
      </div>
      <p className="text-sm font-medium text-brand-300">Processing document…</p>
      <p className="text-xs text-slate-600 mt-1">Chunking and embedding</p>
    </div>
  )
}

export default function IngestPanel() {
  const [isDragging, setIsDragging] = useState(false)
  const [mode, setMode] = useState('file')
  const [pasteText, setPasteText] = useState('')
  const [pasteFilename, setPasteFilename] = useState('document.txt')
  const [lastResult, setLastResult] = useState(null)
  const fileInputRef = useRef(null)

  const ingestFile = useIngestFile()
  const ingestText = useIngestText()
  const { data: docs = [], isLoading: docsLoading } = useDocuments()
  const { data: stats } = usePipelineStats()

  const handleFiles = useCallback(async (files) => {
    setLastResult(null)
    for (const file of files) {
      try {
        const result = await ingestFile.mutateAsync({ file })
        setLastResult(result)
      } catch (e) {
        console.error('Ingest failed:', e.message)
      }
    }
  }, [ingestFile])

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setIsDragging(false)
    handleFiles([...e.dataTransfer.files])
  }, [handleFiles])

  const handlePasteSubmit = async () => {
    if (!pasteText.trim()) return
    setLastResult(null)
    try {
      const result = await ingestText.mutateAsync({ text: pasteText, filename: pasteFilename })
      setLastResult(result)
      setPasteText('')
    } catch (e) {
      console.error('Ingest failed:', e.message)
    }
  }

  const isLoading = ingestFile.isPending || ingestText.isPending

  return (
    <div className="space-y-4">
      {/* Stats bar */}
      {stats && (
        <div className="glass rounded-xl px-5 py-3 flex flex-wrap items-center gap-x-6 gap-y-1 text-xs">
          <div>
            <span className="text-slate-500">Collection </span>
            <span className="text-slate-300 font-mono">{stats.collection_name}</span>
          </div>
          <div>
            <span className="text-slate-500">Chunks </span>
            <span className="text-brand-400 font-semibold tabular-nums">{stats.document_count}</span>
          </div>
          <div>
            <span className="text-slate-500">Model </span>
            <span className="text-slate-400 font-mono text-[11px]">{stats.embedding_model}</span>
          </div>
          <div>
            <span className="text-slate-500">Strategy </span>
            <span className="text-slate-300">MMR top-{stats.top_k}</span>
          </div>
        </div>
      )}

      {/* Mode toggle */}
      <div className="flex rounded-lg overflow-hidden border border-slate-700/80 w-fit">
        <button
          onClick={() => setMode('file')}
          className={clsx(
            'flex items-center gap-1.5 px-4 py-1.5 text-xs font-medium transition-colors',
            mode === 'file' ? 'bg-brand-500 text-white' : 'text-slate-400 hover:text-slate-200',
          )}
        >
          <FileIcon className="w-3.5 h-3.5" />
          File Upload
        </button>
        <button
          onClick={() => setMode('text')}
          className={clsx(
            'flex items-center gap-1.5 px-4 py-1.5 text-xs font-medium transition-colors border-l border-slate-700/80',
            mode === 'text' ? 'bg-brand-500 text-white' : 'text-slate-400 hover:text-slate-200',
          )}
        >
          <TextIcon className="w-3.5 h-3.5" />
          Paste Text
        </button>
      </div>

      {/* File upload */}
      {mode === 'file' && (
        <div
          onDrop={onDrop}
          onDragOver={(e) => { e.preventDefault(); setIsDragging(true) }}
          onDragLeave={() => setIsDragging(false)}
          onClick={() => !isLoading && fileInputRef.current?.click()}
          className="cursor-pointer"
        >
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".txt,.md,.pdf"
            className="hidden"
            onChange={(e) => handleFiles([...e.target.files])}
          />
          {isLoading ? <ProcessingState /> : <DropZone isDragging={isDragging} />}
        </div>
      )}

      {/* Paste text */}
      {mode === 'text' && (
        <div className="glass rounded-xl p-4 space-y-3">
          <input
            type="text"
            value={pasteFilename}
            onChange={(e) => setPasteFilename(e.target.value)}
            placeholder="Document identifier (e.g., my-corpus.txt)"
            className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 placeholder-slate-600 focus:outline-none focus:border-brand-500"
          />
          <textarea
            value={pasteText}
            onChange={(e) => setPasteText(e.target.value)}
            placeholder="Paste document text here…"
            rows={6}
            className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 placeholder-slate-600 focus:outline-none focus:border-brand-500 resize-none"
          />
          <button
            onClick={handlePasteSubmit}
            disabled={!pasteText.trim() || isLoading}
            className="w-full py-2 rounded-lg bg-brand-500 hover:bg-brand-600 text-white text-sm font-medium transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
          >
            {isLoading && <SpinnerIcon className="w-4 h-4" />}
            {isLoading ? 'Ingesting…' : 'Ingest Text'}
          </button>
        </div>
      )}

      {/* Success */}
      {lastResult && !isLoading && (
        <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/5 px-4 py-3 text-xs">
          <p className="text-emerald-400 font-medium mb-0.5">Ingested successfully</p>
          <p className="text-slate-400">
            <span className="font-mono">{lastResult.filename}</span>
            {' → '}
            <span className="text-emerald-300 font-semibold">{lastResult.chunks_created} chunks</span>
            {' in '}
            <span className="font-mono text-slate-300">{lastResult.collection_name}</span>
          </p>
        </div>
      )}

      {/* Error */}
      {(ingestFile.isError || ingestText.isError) && (
        <div className="rounded-xl border border-red-500/30 bg-red-500/5 px-4 py-3 text-xs text-red-400">
          {ingestFile.error?.message ?? ingestText.error?.message}
        </div>
      )}

      {/* Document list */}
      <div className="glass rounded-xl overflow-hidden">
        <div className="px-5 py-3 border-b border-slate-800">
          <h3 className="text-sm font-semibold text-slate-200">Ingested Documents</h3>
        </div>
        {docsLoading ? (
          <div className="p-4 space-y-2">
            {[...Array(3)].map((_, i) => <div key={i} className="skeleton h-8 rounded" />)}
          </div>
        ) : docs.length === 0 ? (
          <p className="px-5 py-6 text-xs text-slate-600 text-center">
            No documents ingested yet.
          </p>
        ) : (
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-slate-800 text-slate-500 uppercase tracking-wider">
                <th className="px-4 py-2.5 text-left font-medium">Filename</th>
                <th className="px-4 py-2.5 text-center font-medium">Chunks</th>
                <th className="px-4 py-2.5 text-left font-medium">Ingested</th>
              </tr>
            </thead>
            <tbody>
              {docs.slice(0, 20).map((doc) => (
                <tr key={doc.id} className="border-b border-slate-800/60 hover:bg-slate-800/20">
                  <td className="px-4 py-2.5 text-slate-300 font-mono">{doc.filename}</td>
                  <td className="px-4 py-2.5 text-center text-slate-400">{doc.chunk_count}</td>
                  <td className="px-4 py-2.5 text-slate-500">
                    {formatDistanceToNow(new Date(doc.ingested_at), { addSuffix: true })}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
