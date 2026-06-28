/**
 * MetricTrendChart — line chart of all 4 RAGAS metrics across the last N runs.
 *
 * WHY RECHARTS:
 * - Works without a build step in browser (no D3 boilerplate)
 * - Composable: LineChart + Line + Tooltip + Legend are separate components
 * - Responsive: ResponsiveContainer fills any parent width
 *
 * X-axis: version_tag (truncated to keep labels readable)
 * Y-axis: 0–1 (RAGAS score range)
 * Lines: one per metric, each with its own color from METRIC_COLORS
 */
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend, ReferenceLine,
} from 'recharts'
import { METRIC_COLORS, METRIC_LABELS, fmtScore } from '../../utils/scoreColor.js'

function SkeletonChart() {
  return <div className="skeleton h-64 w-full rounded-xl" />
}

// Formats the x-axis tick: ISO date → "Jun 28" or "06/28 14:30"
const shortDate = (iso) => {
  if (!iso) return ''
  const d = new Date(iso)
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-slate-800 border border-slate-700 rounded-lg p-3 text-xs shadow-xl">
      <p className="text-slate-300 font-medium mb-1">{payload[0]?.payload?.version ?? label}</p>
      <p className="text-slate-500 mb-2">{shortDate(label)}</p>
      {payload.map((p) => (
        <div key={p.dataKey} className="flex items-center gap-2 mb-1">
          <span className="w-2 h-2 rounded-full" style={{ background: p.color }} />
          <span className="text-slate-400">{METRIC_LABELS[p.dataKey]}:</span>
          <span className="text-slate-100 font-semibold tabular-nums">
            {fmtScore(p.value)}
          </span>
        </div>
      ))}
    </div>
  )
}

export default function MetricTrendChart({ runs = [], loading = false }) {
  if (loading) return <SkeletonChart />

  // Build chart data: completed runs sorted by created_at (oldest→newest = left→right)
  // Deduplicate by version prefix so repeated sample evals don't collapse the view,
  // but keep each distinct run as its own point using created_at as the x key.
  const completed = [...runs]
    .filter((r) => r.status === 'completed' && r.scores)
    .sort((a, b) => new Date(a.created_at) - new Date(b.created_at))
    .slice(-20) // last 20 completed runs
    .map((r) => ({
      version: r.version_tag,
      date: r.created_at,
      faithfulness: r.scores?.faithfulness,
      answer_relevancy: r.scores?.answer_relevancy,
      context_precision: r.scores?.context_precision,
      context_recall: r.scores?.context_recall,
    }))

  if (completed.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-slate-600 text-sm">
        No completed evaluations yet. Run an eval to see trends.
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={260}>
      <LineChart data={completed} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />

        <XAxis
          dataKey="date"
          tickFormatter={shortDate}
          tick={{ fill: '#64748b', fontSize: 11 }}
          axisLine={{ stroke: '#1e293b' }}
          tickLine={false}
        />
        <YAxis
          domain={[0, 1]}
          tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
          tick={{ fill: '#64748b', fontSize: 11 }}
          axisLine={false}
          tickLine={false}
          width={40}
        />

        {/* Warning threshold line */}
        <ReferenceLine y={0.8} stroke="#334155" strokeDasharray="4 4" />
        <ReferenceLine y={0.6} stroke="#292524" strokeDasharray="4 4" />

        <Tooltip content={<CustomTooltip />} />
        <Legend
          formatter={(val) => (
            <span style={{ color: '#94a3b8', fontSize: 11 }}>{METRIC_LABELS[val]}</span>
          )}
        />

        {Object.entries(METRIC_COLORS).map(([key, color]) => (
          <Line
            key={key}
            type="monotone"
            dataKey={key}
            stroke={color}
            strokeWidth={2}
            dot={{ r: 3, fill: color, strokeWidth: 0 }}
            activeDot={{ r: 5 }}
            connectNulls
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  )
}
