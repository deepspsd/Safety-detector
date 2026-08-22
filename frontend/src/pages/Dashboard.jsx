import { useState, useEffect, useCallback } from 'react'
import { alertsApi } from '../api/api'
import api from '../api/api'
import { Link } from 'react-router-dom'
import {
  Bell, ShieldCheck, ShieldAlert, Activity,
  Video, ArrowRight, Clock, AlertTriangle,
  HardHat, TrendingUp, Zap, Users, Layers,
  ChefHat, Flame, Package, RefreshCw
} from 'lucide-react'
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer,
  BarChart, Bar, Cell
} from 'recharts'

const FLOORS = [
  { id: 'ground', label: 'Ground Floor', icon: '🏭', color: '#3b82f6', workStart: '8:00 AM', zones: 'Entrance · Dough Mixing · Cutting · Oven · Packing' },
  { id: 'first',  label: 'First Floor',  icon: '🏗️', color: '#8b5cf6', workStart: '6:00 AM', zones: 'Work Tables · Dough Mixing · Lift · Cylinders' },
  { id: 'second', label: 'Second Floor', icon: '🏢', color: '#06b6d4', workStart: '5:00 AM', zones: 'Cooking · Ovens · Stock · Windows' },
  { id: 'shop',   label: 'Shop',         icon: '🛒', color: '#f59e0b', workStart: '8:00 AM', zones: 'Counter · Cashbox · Entrance' },
]

const BAKERY_CHECKS = [
  { key: 'No Head Cap',   icon: '👷', label: 'Head Cap',   desc: 'Mandatory on all floors' },
  { key: 'No Hardhat',    icon: '⛑️',  label: 'Hardhat',   desc: 'Detected by AI model' },
  { key: 'No Mask',       icon: '😷', label: 'Mask',       desc: 'Required near food' },
  { key: 'Bangle',        icon: '🚫', label: 'No Bangles', desc: 'Not allowed near machines' },
  { key: 'Idle',          icon: '⏰', label: 'Idle >5min', desc: 'General idle rule' },
  { key: 'Dress code',    icon: '👔', label: 'Uniform',    desc: 'Mandatory all floors' },
]

function buildTrend(recentAlerts) {
  const counts = {}
  for (let h = 0; h < 24; h += 2) counts[`${h}:00`] = 0
  if (!recentAlerts?.length) return Object.entries(counts).map(([hour, alerts]) => ({ hour, alerts }))
  recentAlerts.forEach(a => {
    const d = new Date(a.timestamp)
    const h = Math.floor(d.getHours() / 2) * 2
    const key = `${h}:00`
    if (key in counts) counts[key]++
  })
  return Object.entries(counts).map(([hour, alerts]) => ({ hour, alerts }))
}

function buildTypeBreakdown(alerts) {
  const freq = {}
  alerts?.forEach(a => {
    if (!a.detected_issue) return
    const k = a.detected_issue.split(',')[0].trim()
    if (k) freq[k] = (freq[k] || 0) + 1
  })
  return Object.entries(freq)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 6)
    .map(([name, value]) => ({ name, value }))
}

const FLOOR_COLORS = { ground: '#3b82f6', first: '#8b5cf6', second: '#06b6d4', shop: '#f59e0b' }

function liveTime() {
  return new Date().toLocaleString('en-IN', { dateStyle: 'medium', timeStyle: 'short' })
}

export default function Dashboard() {
  const [stats,      setStats]      = useState(null)
  const [summary,    setSummary]    = useState(null)
  const [loading,    setLoading]    = useState(true)
  const [now,        setNow]        = useState(liveTime())
  const [refreshing, setRefreshing] = useState(false)

  const load = useCallback(async () => {
    try {
      const [alertRes, sumRes] = await Promise.all([
        alertsApi.stats(),
        api.get('/workflow/summary').catch(() => ({ data: null })),
      ])
      setStats(alertRes.data)
      setSummary(sumRes.data)
    } catch { /* silent */ }
    finally { setLoading(false); setRefreshing(false) }
  }, [])


  // Auto-refresh every 30s
  useEffect(() => { load() }, [load])
  useEffect(() => {
    const t = setInterval(load, 30000)
    return () => clearInterval(t)
  }, [load])

  // Live clock
  useEffect(() => {
    const t = setInterval(() => setNow(liveTime()), 30000)
    return () => clearInterval(t)
  }, [])

  const handleRefresh = () => { setRefreshing(true); load() }

  const compliancePct   = stats?.compliance_percentage ?? 0
  const trendData       = buildTrend(stats?.recent_alerts)
  const typeBreakdown   = buildTypeBreakdown(stats?.recent_alerts)
  const floorCounts     = summary?.floor_counts || {}

  if (loading) return (
    <div className="loading-container">
      <span className="spinner" />
      <span>Loading Bakery Safety Dashboard…</span>
    </div>
  )

  return (
    <div className="page-container">

      {/* ── Header ── */}
      <div className="page-header" style={{ marginBottom: 24 }}>
        <div>
          <h1 className="page-title" style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <ChefHat size={24} color="#f97316" />
            Bakery Safety Monitor
          </h1>
          <p className="page-subtitle" style={{ marginTop: 4, display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{
              display: 'inline-flex', alignItems: 'center', gap: 6,
              padding: '2px 10px', borderRadius: 99,
              background: 'rgba(249,115,22,0.12)', color: '#fb923c',
              fontWeight: 700, fontSize: '0.78rem',
            }}>
              🧁 Bakery Food Safety
            </span>
            <span style={{ color: 'var(--text-muted)', fontSize: '0.82rem' }}>{now}</span>
          </p>
        </div>
        <div style={{ display: 'flex', gap: 10 }}>
          <button className="btn btn-ghost btn-sm" onClick={handleRefresh} disabled={refreshing}>
            <RefreshCw size={14} style={{ animation: refreshing ? 'spin 1s linear infinite' : 'none' }} />
            Refresh
          </button>
          <Link to="/alerts" className="btn btn-ghost btn-sm">
            <Bell size={14} /> Alerts
          </Link>
          <Link to="/monitor" className="btn btn-primary" style={{
            background: 'linear-gradient(135deg, #f97316, #ea580c)',
            boxShadow: '0 4px 15px rgba(249,115,22,0.3)',
          }}>
            <Video size={15} /> Live Monitor
          </Link>
        </div>
      </div>

      {/* ── KPI Cards ── */}
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
          delta={stats?.critical_alerts > 0 ? '⚠ Action needed' : '✓ None today'}
          deltaClass={stats?.critical_alerts > 0 ? 'stat-delta-up' : 'stat-delta-down'}
        />
        <StatCard
          icon={<ShieldCheck size={22} />}
          iconClass="stat-icon-green"
          value={`${compliancePct}%`}
          label="Compliance Rate"
          delta={compliancePct >= 80 ? '✓ On target' : '↓ Below 80%'}
          deltaClass={compliancePct >= 80 ? 'stat-delta-down' : 'stat-delta-up'}
        />
        <StatCard
          icon={<Activity size={22} />}
          iconClass="stat-icon-construction"
          value={summary?.total_today ?? stats?.today_alerts ?? 0}
          label="Workflow Events"
          delta="All types · today"
        />
      </div>

      {/* ── Floor Status Grid ── */}
      <div style={{ marginBottom: 24 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 14 }}>
          <Layers size={16} color="#f97316" />
          <h3 style={{ fontSize: '0.95rem', fontWeight: 700 }}>Floor Status</h3>
          <span style={{ marginLeft: 'auto', fontSize: '0.72rem', color: 'var(--text-muted)' }}>
            alert counts today
          </span>
        </div>
        <div className="grid-4">
          {FLOORS.map(f => {
            const count = floorCounts[f.id] ?? 0
            const hasAlerts = count > 0
            return (
              <Link key={f.id} to={`/floors/${f.id}`} style={{ textDecoration: 'none' }}>
                <div style={{
                  background: 'var(--bg-card)',
                  border: `1px solid ${hasAlerts ? f.color + '60' : 'var(--border)'}`,
                  borderRadius: 'var(--radius-lg)',
                  padding: '16px 18px',
                  cursor: 'pointer',
                  transition: 'all var(--transition)',
                  position: 'relative',
                  overflow: 'hidden',
                }}
                onMouseEnter={e => e.currentTarget.style.transform = 'translateY(-2px)'}
                onMouseLeave={e => e.currentTarget.style.transform = ''}>
                  {/* Color accent bar */}
                  <div style={{
                    position: 'absolute', top: 0, left: 0, right: 0, height: 3,
                    background: `linear-gradient(90deg, ${f.color}, ${f.color}80)`,
                  }} />
                  <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 8 }}>
                    <span style={{ fontSize: '1.4rem' }}>{f.icon}</span>
                    <span style={{
                      fontSize: '1.4rem', fontWeight: 800,
                      color: hasAlerts ? f.color : 'var(--text-muted)',
                    }}>{count}</span>
                  </div>
                  <div style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: 4 }}>{f.label}</div>
                  <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', marginBottom: 6 }}>
                    ⏰ Work starts {f.workStart}
                  </div>
                  <div style={{ fontSize: '0.66rem', color: 'var(--text-muted)', lineHeight: 1.4 }}>{f.zones}</div>
                  {hasAlerts && (
                    <div style={{
                      marginTop: 8, display: 'inline-flex', alignItems: 'center', gap: 4,
                      fontSize: '0.68rem', color: f.color, fontWeight: 700,
                      padding: '2px 8px', background: `${f.color}15`, borderRadius: 99,
                    }}>
                      <span style={{ width: 6, height: 6, borderRadius: '50%', background: f.color, animation: 'pulse 2s infinite' }} />
                      {count} alert{count !== 1 ? 's' : ''}
                    </div>
                  )}
                </div>
              </Link>
            )
          })}
        </div>
      </div>

      {/* ── Charts Row ── */}
      <div className="grid-2" style={{ marginBottom: 24 }}>

        {/* Alert Trend */}
        <div className="card card-p">
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
            <TrendingUp size={16} color="#f97316" />
            <h3 style={{ fontSize: '1rem' }}>Alert Activity (24h)</h3>
          </div>
          <ResponsiveContainer width="100%" height={148}>
            <AreaChart data={trendData} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
              <defs>
                <linearGradient id="alertGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor="#f97316" stopOpacity={0.35} />
                  <stop offset="95%" stopColor="#f97316" stopOpacity={0}    />
                </linearGradient>
              </defs>
              <XAxis dataKey="hour" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} tickLine={false} axisLine={false} />
              <YAxis  tick={{ fontSize: 10, fill: 'var(--text-muted)' }} tickLine={false} axisLine={false} allowDecimals={false} />
              <Tooltip contentStyle={{
                background: 'var(--bg-secondary)', border: '1px solid var(--border)',
                borderRadius: 8, fontSize: 12
              }} />
              <Area type="monotone" dataKey="alerts"
                stroke="#f97316" strokeWidth={2}
                fill="url(#alertGrad)" dot={false} />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Violation Type Breakdown */}
        <div className="card card-p">
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
            <AlertTriangle size={16} color="#f59e0b" />
            <h3 style={{ fontSize: '1rem' }}>Violation Types</h3>
          </div>
          {typeBreakdown.length === 0 ? (
            <div style={{ textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.84rem', padding: '32px 0' }}>
              <ShieldCheck size={32} style={{ marginBottom: 8, opacity: 0.3 }} />
              <div>No violations today</div>
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={148}>
              <BarChart data={typeBreakdown} margin={{ top: 0, right: 10, left: -20, bottom: 0 }}>
                <XAxis dataKey="name" tick={{ fontSize: 9, fill: 'var(--text-muted)' }} tickLine={false} axisLine={false} />
                <YAxis  tick={{ fontSize: 9, fill: 'var(--text-muted)' }} tickLine={false} axisLine={false} allowDecimals={false} />
                <Tooltip contentStyle={{
                  background: 'var(--bg-secondary)', border: '1px solid var(--border)',
                  borderRadius: 8, fontSize: 12
                }} />
                <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                  {typeBreakdown.map((_, i) => (
                    <Cell key={i} fill={['#ef4444','#f59e0b','#f97316','#8b5cf6','#3b82f6','#10b981'][i % 6]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      {/* ── Workflow Summary Cards ── */}
      {summary && (
        <div style={{ marginBottom: 24 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 14 }}>
            <Zap size={16} color="#f97316" />
            <h3 style={{ fontSize: '0.95rem', fontWeight: 700 }}>Today's Workflow Events</h3>
          </div>
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
            {[
              { label: 'Idle Events',    value: summary.idle_events_today,       icon: '⏰', color: '#f59e0b', to: '/workflow' },
              { label: 'Dress Code',     value: summary.dress_code_events_today, icon: '👷', color: '#ef4444', to: '/workflow' },
              { label: 'Cash Zone',      value: summary.cash_events_today,       icon: '💰', color: '#10b981', to: '/workflow' },
              { label: 'Oven/Gas',       value: summary.oven_events_today,       icon: '🔥', color: '#f97316', to: '/workflow' },
              { label: 'Stock Flow',     value: summary.stock_events_today,      icon: '📦', color: '#8b5cf6', to: '/workflow' },
              { label: 'Lift Events',    value: summary.lift_events_today,       icon: '🛗', color: '#3b82f6', to: '/workflow' },
            ].map(({ label, value, icon, color, to }) => (
              <Link key={label} to={to} style={{ textDecoration: 'none', flex: 1, minWidth: 120 }}>
                <div style={{
                  background: 'var(--bg-card)', border: '1px solid var(--border)',
                  borderRadius: 'var(--radius-lg)', padding: '12px 16px',
                  display: 'flex', alignItems: 'center', gap: 10,
                  transition: 'all var(--transition)',
                }}
                onMouseEnter={e => e.currentTarget.style.borderColor = color + '60'}
                onMouseLeave={e => e.currentTarget.style.borderColor = 'var(--border)'}>
                  <span style={{ fontSize: '1.3rem' }}>{icon}</span>
                  <div>
                    <div style={{ fontSize: '1.3rem', fontWeight: 800, color, lineHeight: 1 }}>{value ?? 0}</div>
                    <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', marginTop: 2, textTransform: 'uppercase', letterSpacing: '0.04em' }}>{label}</div>
                  </div>
                </div>
              </Link>
            ))}
          </div>
        </div>
      )}

      {/* ── Bakery Safety Checklist ── */}
      <div className="card card-p" style={{ marginBottom: 24 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 14 }}>
          <HardHat size={16} color="#f97316" />
          <h3 style={{ fontSize: '1rem' }}>Bakery Safety Checklist</h3>
          <span style={{
            marginLeft: 'auto', fontSize: '0.7rem', fontWeight: 700, padding: '2px 8px',
            borderRadius: 99, background: 'rgba(249,115,22,0.12)', color: '#fb923c'
          }}>ALL FLOORS</span>
        </div>
        <div className="ppe-grid">
          {BAKERY_CHECKS.map(p => {
            const missCount = (stats?.recent_alerts || []).filter(a =>
              (a.detected_issue || '').toLowerCase().includes(p.key.toLowerCase()) ||
              (a.message || '').toLowerCase().includes(p.key.toLowerCase())
            ).length
            return (
              <div key={p.key} className={`ppe-item ${missCount === 0 ? 'ok' : 'miss'}`}>
                <span style={{ fontSize: '1.4rem' }}>{p.icon}</span>
                <span style={{ fontWeight: 600 }}>{p.label}</span>
                <span style={{ fontSize: '0.62rem', color: 'var(--text-muted)', lineHeight: 1.3 }}>{p.desc}</span>
                {missCount > 0
                  ? <span style={{ fontSize: '0.65rem', color: '#ef4444', fontWeight: 700 }}>{missCount} violation{missCount > 1 ? 's' : ''}</span>
                  : <span style={{ fontSize: '0.65rem', color: '#10b981' }}>✓ No issues</span>}
              </div>
            )
          })}
        </div>
      </div>

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
                <th>Floor</th>
                <th>Violation</th>
                <th>Severity</th>
                <th>Confidence</th>
              </tr>
            </thead>
            <tbody>
              {stats.recent_alerts.slice(0, 10).map(a => (
                <tr key={a.id}>
                  <td style={{ color: 'var(--text-muted)', fontSize: '0.78rem', whiteSpace: 'nowrap' }}>
                    <Clock size={11} style={{ marginRight: 4, verticalAlign: 'middle' }} />
                    {new Date(a.timestamp).toLocaleString('en-IN', { dateStyle: 'short', timeStyle: 'short' })}
                  </td>
                  <td>
                    {a.floor ? (
                      <span style={{
                        fontSize: '0.7rem', fontWeight: 700, padding: '2px 7px', borderRadius: 99,
                        background: `${FLOOR_COLORS[a.floor] || '#64748b'}18`,
                        color: FLOOR_COLORS[a.floor] || 'var(--text-muted)',
                      }}>
                        {FLOORS.find(f => f.id === a.floor)?.icon || ''} {a.floor}
                      </span>
                    ) : <span style={{ color: 'var(--text-muted)' }}>—</span>}
                  </td>
                  <td style={{ color: 'var(--text-primary)', maxWidth: 220 }}>{a.detected_issue || a.message}</td>
                  <td><span className={`badge badge-${a.severity}`}>{a.severity}</span></td>
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
