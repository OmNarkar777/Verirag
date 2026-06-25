/**
 * App.jsx - root component: router, layout shell, page routing, global status banner.
 */
import { BrowserRouter, Routes, Route, useLocation } from 'react-router-dom'
import { Component } from 'react'
import { useQuery } from '@tanstack/react-query'
import axios from 'axios'
import Sidebar from './components/layout/Sidebar.jsx'
import Header from './components/layout/Header.jsx'
import DashboardPage from './pages/DashboardPage.jsx'
import RunDetailsPage from './pages/RunDetailsPage.jsx'
import PipelinePage from './pages/PipelinePage.jsx'

class ErrorBoundary extends Component {
  state = { error: null }
  static getDerivedStateFromError(error) { return { error } }
  render() {
    if (this.state.error) {
      return (
        <div className="p-8">
          <p className="text-red-400 font-medium mb-2">Something went wrong</p>
          <p className="text-xs text-slate-500 font-mono mb-4">{this.state.error.message}</p>
          <button
            onClick={() => this.setState({ error: null })}
            className="text-xs text-brand-400 hover:text-brand-300 underline"
          >
            Try again
          </button>
        </div>
      )
    }
    return this.props.children
  }
}

function SetupBanner({ missing }) {
  if (!missing?.length) return null
  return (
    <div className="bg-amber-500/10 border-b border-amber-500/20 px-6 py-2.5 flex items-center gap-3 text-xs shrink-0">
      <div className="w-4 h-4 rounded-full bg-amber-500/20 border border-amber-500/40 flex items-center justify-center shrink-0">
        <span className="text-amber-400 font-bold" style={{ fontSize: 9 }}>!</span>
      </div>
      <span className="text-amber-300 font-medium">Setup required:</span>
      <span className="text-amber-400/80">
        Set{' '}
        {missing.map((v, i) => (
          <span key={v}>
            <code className="text-amber-300 font-mono">{v}</code>
            {i < missing.length - 1 ? ', ' : ''}
          </span>
        ))}
        {' '}in your Vercel environment variables to enable all features.
      </span>
    </div>
  )
}

function useSystemStatus() {
  return useQuery({
    queryKey: ['system-status'],
    queryFn: () => axios.get('/api/v1/system/status').then((r) => r.data),
    staleTime: 60_000,
    retry: false,
  })
}

const PAGE_META = {
  '/': { title: 'Dashboard', subtitle: 'RAGAS metric trends and eval run history' },
  '/pipeline': { title: 'Pipeline', subtitle: 'Ingest documents · Query · Trigger evaluations' },
}

function Layout() {
  const location = useLocation()
  const meta = PAGE_META[location.pathname] ?? {
    title: 'Run Details',
    subtitle: 'Per-case RAGAS scores and regression analysis',
  }

  const { data: status } = useSystemStatus()

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        <Header title={meta.title} subtitle={meta.subtitle} />
        {status && !status.configured && (
          <SetupBanner missing={status.missing_vars} />
        )}
        <main className="flex-1 overflow-y-auto">
          <ErrorBoundary>
            <Routes>
              <Route path="/" element={<DashboardPage />} />
              <Route path="/runs/:id" element={<RunDetailsPage />} />
              <Route path="/pipeline" element={<PipelinePage />} />
              <Route path="*" element={
                <div className="p-8 text-center">
                  <p className="text-slate-500">Page not found</p>
                </div>
              } />
            </Routes>
          </ErrorBoundary>
        </main>
      </div>
    </div>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <Layout />
    </BrowserRouter>
  )
}
