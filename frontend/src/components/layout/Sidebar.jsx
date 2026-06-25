import { NavLink } from 'react-router-dom'
import clsx from 'clsx'

function DashboardIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" className="shrink-0">
      <rect x="1" y="1" width="6" height="6" rx="1.5" stroke="currentColor" strokeWidth="1.5"/>
      <rect x="9" y="1" width="6" height="6" rx="1.5" stroke="currentColor" strokeWidth="1.5"/>
      <rect x="1" y="9" width="6" height="6" rx="1.5" stroke="currentColor" strokeWidth="1.5"/>
      <rect x="9" y="9" width="6" height="6" rx="1.5" stroke="currentColor" strokeWidth="1.5"/>
    </svg>
  )
}

function PipelineIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" className="shrink-0">
      <circle cx="3" cy="8" r="2" stroke="currentColor" strokeWidth="1.5"/>
      <circle cx="13" cy="4" r="2" stroke="currentColor" strokeWidth="1.5"/>
      <circle cx="13" cy="12" r="2" stroke="currentColor" strokeWidth="1.5"/>
      <path d="M5 7.5L11 4.5M5 8.5L11 11.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
    </svg>
  )
}

const NAV = [
  { to: '/',         label: 'Dashboard',  Icon: DashboardIcon },
  { to: '/pipeline', label: 'Pipeline',   Icon: PipelineIcon },
]

export default function Sidebar() {
  return (
    <aside className="w-56 shrink-0 bg-slate-900 border-r border-slate-800 flex flex-col">
      {/* Logo */}
      <div className="px-5 py-5 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 rounded-md bg-brand-500 flex items-center justify-center">
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
              <path d="M2 10L6 2L10 10" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M3.5 7.5H8.5" stroke="white" strokeWidth="1.5" strokeLinecap="round"/>
            </svg>
          </div>
          <span className="text-base font-bold tracking-tight">
            <span className="text-brand-400">Veri</span>
            <span className="text-slate-100">RAG</span>
          </span>
        </div>
        <p className="text-xs text-slate-500 mt-1.5 pl-8">RAG Evaluation Platform</p>
      </div>

      {/* Nav */}
      <nav className="flex-1 px-3 py-4 space-y-0.5">
        {NAV.map(({ to, label, Icon }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              clsx(
                'flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors',
                isActive
                  ? 'bg-brand-500/15 text-brand-400 border border-brand-500/25'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60',
              )
            }
          >
            {({ isActive }) => (
              <>
                <span className={isActive ? 'text-brand-400' : 'text-slate-500'}>
                  <Icon />
                </span>
                {label}
              </>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Footer */}
      <div className="px-5 py-4 border-t border-slate-800 space-y-2">
        <a
          href="/docs"
          target="_blank"
          rel="noreferrer"
          className="flex items-center gap-1.5 text-xs text-slate-500 hover:text-slate-300 transition-colors"
        >
          <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
            <path d="M2 10L10 2M10 2H5.5M10 2V6.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          API Docs
        </a>
        <p className="text-xs text-slate-700">RAGAS · Groq · PostgreSQL</p>
      </div>
    </aside>
  )
}
