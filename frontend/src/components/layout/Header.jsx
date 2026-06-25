import { useEvalStatus } from '../../hooks/useEvalRuns.js'
import { getHealth } from '../../api/client.js'
import { useQuery } from '@tanstack/react-query'

function BackendStatusDot() {
  const { data, isError, isLoading } = useQuery({
    queryKey: ['health'],
    queryFn: getHealth,
    refetchInterval: 30_000,
    retry: false,
    staleTime: 25_000,
  })

  if (isLoading) return null

  const isOk = !isError && data?.status === 'ok'
  const isDegraded = !isError && data?.status === 'degraded'

  return (
    <div className="flex items-center gap-1.5 text-xs">
      <span className={
        isOk ? 'w-1.5 h-1.5 rounded-full bg-emerald-400' :
        isDegraded ? 'w-1.5 h-1.5 rounded-full bg-amber-400' :
        'w-1.5 h-1.5 rounded-full bg-red-400'
      } />
      <span className={
        isOk ? 'text-slate-600' :
        isDegraded ? 'text-amber-600' :
        'text-red-600'
      }>
        {isOk ? 'Connected' : isDegraded ? 'Degraded' : 'Offline'}
      </span>
    </div>
  )
}

export default function Header({ title, subtitle }) {
  const { data: status } = useEvalStatus()

  return (
    <header className="h-14 px-6 border-b border-slate-800 bg-slate-900/60 flex items-center justify-between shrink-0">
      <div>
        <h1 className="text-base font-semibold text-slate-100">{title}</h1>
        {subtitle && <p className="text-xs text-slate-500">{subtitle}</p>}
      </div>

      <div className="flex items-center gap-5">
        {status?.active_evals > 0 && (
          <div className="flex items-center gap-2 text-xs text-amber-400">
            <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" />
            {status.active_evals} eval{status.active_evals > 1 ? 's' : ''} running
          </div>
        )}
        <BackendStatusDot />
      </div>
    </header>
  )
}
