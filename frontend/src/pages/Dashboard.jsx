import { useState, useEffect } from 'react'
import { useAuth } from '../context/AuthContext'
import { alertsApi } from '../api/api'
import { Link } from 'react-router-dom'
import {
  Bell, ShieldCheck, ShieldAlert, Activity,
  Video, ArrowRight, Clock, AlertTriangle,
  HardHat, Eye, TrendingUp, Zap
} from 'lucide-react'
import {
  PieChart, Pie, Cell, ResponsiveContainer,
  AreaChart, Area, XAxis, Tooltip, BarChart, Bar, YAxis
} from 'recharts'

// ── PPE icons for Construction role
const PPE_ITEMS = [
  { key: 'No Hardhat',       icon: '⛑️',  label: 'Hardhat'       },
  { key: 'No Safety Vest',   icon: '🦺',  label: 'Safety Vest'   },
  { key: 'No Mask',          icon: '😷',  label: 'Mask'          },
  { key: 'No Gloves',        icon: '🧤',  label: 'Gloves'        },
  { key: 'No Goggles',       icon: '🥽',  label: 'Goggles'       },
  { key: 'No Safety Shoes',  icon: '👟',  label: 'Safety Shoes'  },
]

// Build real hourly alert trend from timestamp list
function buildTrend(recentAlerts) {
  const counts = {}
  for (let h = 0; h < 24; h += 2) counts[`${h}:00`] = 0
  if (!recentAlerts?.length)
    return Object.entries(counts).map(([hour, alerts]) => ({ hour, alerts }))
  recentAlerts.forEach(a => {
    const d = new Date(a.timestamp)
    const h = Math.floor(d.getHours() / 2) * 2
    const key = `${h}:00`
    if (key in counts) counts[key]++
  })
  return Object.entries(counts).map(([hour, alerts]) => ({ hour, alerts }))
}

// Extract top missing items from detected_issue field
function buildViolationSummary(alerts) {
  const freq = {}
  alerts?.forEach(a => {
    if (!a.detected_issue) return
    a.detected_issue.split(',').forEach(item => {
      const k = item.trim()
      if (k) freq[k] = (freq[k] || 0) + 1
    })
  })
  return Object.entries(freq)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5)
    .map(([item, count]) => ({ item, count }))
}

// Count how many times each PPE item appears as missing
function buildPpeBreakdown(alerts) {
  const freq = {}
  PPE_ITEMS.forEach(p => { freq[p.label] = 0 })
  alerts?.forEach(a => {
    if (!a.detected_issue) return
    a.detected_issue.split(',').forEach(raw => {
      const item = raw.trim()
      PPE_ITEMS.forEach(p => {
        if (item.toLowerCase().includes(p.label.toLowerCase()) ||
            item.toLowerCase().includes(p.key.toLowerCase())) {
          freq[p.label]++
        }
      })
    })
  })
  return PPE_ITEMS.map(p => ({ name: p.icon + ' ' + p.label, value: freq[p.label] }))
    .filter(d => d.value > 0)
}

export default function Dashboard() {
  const { user } = useAuth()
  const [stats,   setStats]   = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    alertsApi.stats()
      .then(r => setStats(r.data))
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  const compliancePct    = stats?.compliance_percentage ?? 0
  const isConstruction   = !user?.role || user?.role === 'Construction Worker'
  const roleColor        = isConstruction ? 'var(--accent-construction)' : 'var(--accent-blue)'
  const pieData = [
    { name: 'Compliant',  value: compliancePct },
    { name: 'Violations', value: 100 - compliancePct },
  ]

  const trendData      = buildTrend(stats?.recent_alerts)
  const violationSummary = buildViolationSummary(stats?.recent_alerts)
  const ppeBreakdown   = buildPpeBreakdown(stats?.recent_alerts)
  const complianceColor = compliancePct >= 80 ? '#10b981' : compliancePct >= 60 ? '#f59e0b' : '#ef4444'

  if (loading) return (
    <div className="loading-container">
      <span className="spinner" />
      <span>Loading dashboard…</span>
    </div>
  )

  return (
    <div className="page-container">

      {/* ── Page Header ── */}
      <div className="page-header">
        <div>
          <h1 className="page-title" style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            {isConstruction && <HardHat size={24} color={roleColor} />}
            Occupational Safety Dashboard
          </h1>
          <p className="page-subtitle" style={{ marginTop: 4 }}>
            <span style={{
              display: 'inline-flex', alignItems: 'center', gap: 6,
              padding: '2px 10px', borderRadius: 99,
              background: isConstruction ? 'rgba(249,115,22,0.12)' : 'rgba(59,130,246,0.12)',
              color: roleColor, fontWeight: 700, fontSize: '0.78rem',
            }}>
              {isConstruction ? '🏗️' : '👤'} {user?.role || 'No role set'}
            </span>
            {' '}
            <span style={{ color: 'var(--text-muted)' }}>· Safety Intelligence Overview</span>
          </p>
        </div>
        <div style={{ display: 'flex', gap: 10 }}>
          <Link to="/alerts" className="btn btn-ghost btn-sm">
            <Bell size={14} /> Alerts
          </Link>
          <Link to="/monitor" className="btn btn-primary" style={
            isConstruction ? {
              background: 'linear-gradient(135deg, #f97316, #ea580c)',
              boxShadow: '0 4px 15px rgba(249,115,22,0.3)',
            } : {}
          }>
            <Video size={15} /> Start Monitoring
          </Link>
        </div>
      </div>

      {/* ── KPI Stat Cards ── */}
      <div className="grid-4" style={{ marginBottom: 24 }}>
        <StatCard
          icon={<Bell size={22} />}
          iconClass="stat-icon-red"
          value={stats?.total_alerts ?? 0}
          label="Total Violations"
          delta={`+${stats?.today_alerts ?? 0} today`}
          deltaClass="stat-delta-up"
        />
        <StatCard
          icon={<ShieldAlert size={22} />}
          iconClass="stat-icon-orange"
          value={stats?.critical_alerts ?? 0}
          label="Critical Alerts"
          delta={stats?.critical_alerts > 0 ? '⚠ Immediate action' : '✓ None today'}
          deltaClass={stats?.critical_alerts > 0 ? 'stat-delta-up' : 'stat-delta-down'}
        />
        <StatCard
          icon={<ShieldCheck size={22} />}
          iconClass="stat-icon-green"
          value={`${compliancePct}%`}
          label="Compliance Rate"
          delta={compliancePct >= 80 ? '✓ Target met' : '↓ Below 80% target'}
          deltaClass={compliancePct >= 80 ? 'stat-delta-down' : 'stat-delta-up'}
        />
        <StatCard
          icon={<Activity size={22} />}
          iconClass="stat-icon-construction"
          value={stats?.today_alerts ?? 0}
          label="Alerts Today"
          delta="Real-time detection"
        />
      </div>

      {/* ── Charts Row ── */}
      <div className="grid-2" style={{ marginBottom: 24 }}>

        {/* Compliance Gauge */}
        <div className="card card-p">
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
            <Eye size={16} color={roleColor} />
            <h3 style={{ fontSize: '1rem' }}>Safety Compliance</h3>
            <span style={{
              marginLeft: 'auto', fontSize: '0.72rem', fontWeight: 700, padding: '2px 8px',
              borderRadius: 99, background: compliancePct >= 80
                ? 'rgba(16,185,129,0.15)' : 'rgba(239,68,68,0.15)',
              color: compliancePct >= 80 ? '#34d399' : '#f87171',
            }}>
              {compliancePct >= 80 ? 'ON TARGET' : 'NEEDS ATTENTION'}
            </span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 28 }}>
            <div style={{ width: 150, height: 150, flexShrink: 0 }}>
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie data={pieData} cx="50%" cy="50%"
                    innerRadius={42} outerRadius={64}
                    dataKey="value" startAngle={90} endAngle={-270} stroke="none">
                    <Cell fill={complianceColor} />
                    <Cell fill="rgba(255,255,255,0.06)" />
                  </Pie>
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div>
              <div style={{ fontSize: '2.8rem', fontWeight: 800, color: complianceColor, lineHeight: 1 }}>
                {compliancePct}%
              </div>
              <div style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginTop: 6 }}>
                Compliance Rate
              </div>
              <div style={{ marginTop: 14, display: 'flex', flexDirection: 'column', gap: 6 }}>
                <LegendDot color={complianceColor} label="Compliant personnel" />
                <LegendDot color="#ef4444" label="PPE violations detected" />
              </div>
              {/* Target bar */}
              <div style={{ marginTop: 14 }}>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginBottom: 4 }}>
                  Target: 80%
                </div>
                <div className="progress-bar">
                  <div className="progress-fill" style={{
                    width: `${Math.min(compliancePct, 100)}%`,
                    background: `linear-gradient(90deg, ${complianceColor}, ${complianceColor}88)`,
                  }} />
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Alert Trend */}
        <div className="card card-p">
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
            <TrendingUp size={16} color={roleColor} />
            <h3 style={{ fontSize: '1rem' }}>Alert Activity (24h)</h3>
          </div>
          <ResponsiveContainer width="100%" height={148}>
            <AreaChart data={trendData} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
              <defs>
                <linearGradient id="alertGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor={isConstruction ? '#f97316' : '#3b82f6'} stopOpacity={0.35} />
                  <stop offset="95%" stopColor={isConstruction ? '#f97316' : '#3b82f6'} stopOpacity={0}    />
                </linearGradient>
              </defs>
              <XAxis dataKey="hour" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} tickLine={false} axisLine={false} />
              <YAxis  tick={{ fontSize: 10, fill: 'var(--text-muted)' }} tickLine={false} axisLine={false} allowDecimals={false} />
              <Tooltip contentStyle={{
                background: 'var(--bg-secondary)', border: '1px solid var(--border)',
                borderRadius: 8, fontSize: 12
              }} />
              <Area type="monotone" dataKey="alerts"
                stroke={isConstruction ? '#f97316' : '#3b82f6'} strokeWidth={2}
                fill="url(#alertGrad)" dot={false} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* ── PPE Breakdown Bar chart (Construction) + Violation List ── */}
      <div className={ppeBreakdown.length > 0 ? 'grid-2' : ''} style={{ marginBottom: 24 }}>

        {/* PPE violation breakdown */}
        {ppeBreakdown.length > 0 && (
          <div className="card card-p">
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
              <HardHat size={16} color={roleColor} />
              <h3 style={{ fontSize: '1rem' }}>PPE Violation Breakdown</h3>
            </div>
            <ResponsiveContainer width="100%" height={160}>
              <BarChart data={ppeBreakdown} margin={{ top: 0, right: 10, left: -20, bottom: 0 }}>
                <XAxis dataKey="name" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} tickLine={false} axisLine={false} />
                <YAxis tick={{ fontSize: 10, fill: 'var(--text-muted)' }} tickLine={false} axisLine={false} allowDecimals={false} />
                <Tooltip contentStyle={{
                  background: 'var(--bg-secondary)', border: '1px solid var(--border)',
                  borderRadius: 8, fontSize: 12
                }} />
                <Bar dataKey="value" fill={isConstruction ? '#f97316' : '#3b82f6'} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Top violation ranking */}
        {violationSummary.length > 0 && (
          <div className="card card-p">
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
              <AlertTriangle size={16} color="var(--accent-orange)" />
              <h3 style={{ fontSize: '1rem' }}>Top Violations</h3>
              <span style={{ marginLeft: 'auto', fontSize: '0.72rem', color: 'var(--text-muted)' }}>all time</span>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              {violationSummary.map(({ item, count }, i) => {
                const max = violationSummary[0].count
                const barColor = i === 0 ? '#ef4444' : i === 1 ? '#f59e0b' : '#f97316'
                return (
                  <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                    <div style={{ width: 20, fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 700 }}>
                      #{i + 1}
                    </div>
                    <div style={{ width: 120, fontSize: '0.82rem', color: 'var(--text-secondary)', flexShrink: 0 }}>{item}</div>
                    <div style={{ flex: 1, background: 'var(--bg-tertiary)', borderRadius: 4, overflow: 'hidden', height: 8 }}>
                      <div style={{
                        width: `${(count / max) * 100}%`, height: '100%',
                        background: barColor, borderRadius: 4, transition: 'width 0.4s'
                      }} />
                    </div>
                    <div style={{ width: 28, fontSize: '0.78rem', color: 'var(--text-muted)', textAlign: 'right' }}>{count}×</div>
                  </div>
                )
              })}
            </div>
          </div>
        )}
      </div>

      {/* ── PPE Quick-check grid for Construction ── */}
      {isConstruction && (
        <div className="card card-p" style={{ marginBottom: 24 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 14 }}>
            <Zap size={16} color={roleColor} />
            <h3 style={{ fontSize: '1rem' }}>Required PPE — Construction Worker</h3>
            <span style={{
              marginLeft: 'auto', fontSize: '0.7rem', fontWeight: 700, padding: '2px 8px',
              borderRadius: 99, background: 'rgba(249,115,22,0.12)', color: '#fb923c'
            }}>5 ITEMS MONITORED</span>
          </div>
          <div className="ppe-grid">
            {PPE_ITEMS.map(p => {
              const missCount = (stats?.recent_alerts || []).filter(a =>
                (a.detected_issue || '').toLowerCase().includes(p.key.toLowerCase())
              ).length
              return (
                <div key={p.key} className={`ppe-item ${missCount === 0 ? 'ok' : 'miss'}`}>
                  <span style={{ fontSize: '1.5rem' }}>{p.icon}</span>
                  <span>{p.label}</span>
                  {missCount > 0
                    ? <span style={{ fontSize: '0.65rem' }}>{missCount} violation{missCount > 1 ? 's' : ''}</span>
                    : <span style={{ fontSize: '0.65rem' }}>✓ No issues</span>}
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* ── Recent Alerts Table ── */}
      <div className="card card-p">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 18 }}>
          <h3 style={{ fontSize: '1rem', display: 'flex', alignItems: 'center', gap: 8 }}>
            <Clock size={16} color="var(--text-muted)" /> Recent Violations
          </h3>
          <Link to="/alerts" className="btn btn-ghost btn-sm">
            View All <ArrowRight size={13} />
          </Link>
        </div>
        {!stats?.recent_alerts?.length ? (
          <div className="empty-state" style={{ padding: '30px 0' }}>
            <ShieldCheck size={40} />
            <h3>No violations detected</h3>
            <p>Start live monitoring to begin detecting safety violations.</p>
            <Link to="/monitor" className="btn btn-primary" style={{ marginTop: 8 }}>
              <Video size={14} /> Start Monitoring
            </Link>
          </div>
        ) : (
          <table className="alerts-table">
            <thead>
              <tr>
                <th>Time</th>
                <th>Violation</th>
                <th>Severity</th>
                <th>Missing PPE</th>
                <th>Confidence</th>
              </tr>
            </thead>
            <tbody>
              {stats.recent_alerts.slice(0, 8).map(a => (
                <tr key={a.id}>
                  <td style={{ color: 'var(--text-muted)', fontSize: '0.78rem', whiteSpace: 'nowrap' }}>
                    <Clock size={11} style={{ marginRight: 4, verticalAlign: 'middle' }} />
                    {formatTime(a.timestamp)}
                  </td>
                  <td style={{ color: 'var(--text-primary)', maxWidth: 200 }}>{a.message}</td>
                  <td><span className={`badge badge-${a.severity}`}>{a.severity}</span></td>
                  <td>
                    {a.detected_issue
                      ? a.detected_issue.split(',').map((item, i) => (
                          <span key={i} className="ppe-badge ppe-err" style={{ marginRight: 4 }}>
                            {item.trim()}
                          </span>
                        ))
                      : <span style={{ color: 'var(--text-muted)' }}>—</span>}
                  </td>
                  <td style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>
                    {a.confidence ? `${(a.confidence * 100).toFixed(0)}%` : '—'}
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

/* ─── Sub-components ─── */

function StatCard({ icon, iconClass, value, label, delta, deltaClass }) {
  return (
    <div className="stat-card">
      <div className={`stat-icon ${iconClass}`}>{icon}</div>
      <div>
        <div className="stat-value">{value}</div>
        <div className="stat-label">{label}</div>
        {delta && <div className={`stat-delta ${deltaClass || ''}`}>{delta}</div>}
      </div>
    </div>
  )
}

function LegendDot({ color, label }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
      <span style={{ width: 10, height: 10, borderRadius: '50%', background: color, display: 'inline-block', flexShrink: 0 }} />
      {label}
    </div>
  )
}

function getTimeGreeting() {
  const h = new Date().getHours()
  if (h < 12) return 'morning'
  if (h < 17) return 'afternoon'
  return 'evening'
}

function formatTime(iso) {
  return new Date(iso).toLocaleString('en-IN', { dateStyle: 'short', timeStyle: 'short' })
}
