import { useState, useEffect, useCallback } from 'react'
import api from '../api/api'
import { useToast } from '../context/ToastContext'
import {
  Briefcase, DollarSign, Package, ArrowUpDown,
  RefreshCw, AlertTriangle, Clock, Wrench,
  HardHat, Flame, UserX, Timer
} from 'lucide-react'

// ── API helpers ─────────────────────────────────────────────────────────────
const workflowApi = {
  summary:    () => api.get('/workflow/summary'),
  cash:       () => api.get('/workflow/cash-events'),
  stock:      () => api.get('/workflow/stock-events'),
  lift:       () => api.get('/workflow/lift-events'),
  packing:    () => api.get('/workflow/packing-events'),
  idle:       () => api.get('/workflow/idle-events'),
  dressCode:  () => api.get('/workflow/dress-code-events'),
  oven:       () => api.get('/workflow/oven-events'),
}

const TABS = [
  { id: 'idle',      label: 'Idle Tracker', icon: Timer,      color: '#f59e0b',           desc: 'All idle alerts · 5-min rule applies all floors' },
  { id: 'dressCode', label: 'Dress Code',   icon: HardHat,    color: '#ef4444',           desc: 'Head cap · Uniform · No Bangles · Mask' },
  { id: 'cash',      label: 'Cash Zone',    icon: DollarSign, color: '#10b981',           desc: 'Alerts when person in cashbox zone unauthorised' },
  { id: 'stock',     label: 'Stock Flow',   icon: Package,    color: '#8b5cf6',           desc: 'Exposed goods detected in stock zone' },
  { id: 'oven',      label: 'Oven / Gas',   icon: Flame,      color: '#f97316',           desc: 'Floor 2 · Unattended fire · Gas waste · Oil idle' },
  { id: 'lift',      label: 'Lift Events',  icon: ArrowUpDown,color: 'var(--accent-blue)',desc: 'Entry/exit tracking across all 3 floors' },
  { id: 'packing',   label: 'Packing',      icon: Wrench,     color: 'var(--accent-green)',desc: 'Packing section hand movement monitoring' },
]

// ── Summary card ────────────────────────────────────────────────────────────
function SummaryCard({ label, value, icon, color }) {
  const Ic = icon
  return (
    <div style={{
      background: 'var(--bg-card)', border: '1px solid var(--border)',
      borderRadius: 'var(--radius-lg)', padding: '14px 20px',
      display: 'flex', alignItems: 'center', gap: 12, flex: 1, minWidth: 140,
    }}>
      <div style={{ width: 40, height: 40, borderRadius: 10, background: `${color}18`, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
        <Ic size={18} color={color} />
      </div>
      <div>
        <div style={{ fontSize: '1.4rem', fontWeight: 800, color, lineHeight: 1 }}>{value ?? '—'}</div>
        <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: 2, textTransform: 'uppercase', letterSpacing: '0.04em' }}>{label}</div>
      </div>
    </div>
  )
}



// ── Severity badge ──────────────────────────────────────────────────────────
function SevBadge({ severity }) {
  const cfg = {
    critical: { bg: 'rgba(239,68,68,0.12)',  color: '#ef4444' },
    high:     { bg: 'rgba(239,68,68,0.12)',  color: '#ef4444' },
    medium:   { bg: 'rgba(249,115,22,0.12)', color: '#f97316' },
    low:      { bg: 'rgba(16,185,129,0.12)', color: 'var(--accent-green)' },
  }[severity] || { bg: 'var(--bg-secondary)', color: 'var(--text-muted)' }
  return (
    <span style={{ fontSize: '0.68rem', fontWeight: 700, padding: '2px 7px', borderRadius: 99, background: cfg.bg, color: cfg.color, textTransform: 'uppercase' }}>
      {severity}
    </span>
  )
}

// ── Floor badge ─────────────────────────────────────────────────────────────
function FloorBadge({ floor }) {
  if (!floor) return null
  const cfg = { ground: '#3b82f6', first: '#8b5cf6', second: '#06b6d4', shop: '#f59e0b' }
  const icon = { ground: '🏭', first: '🏗️', second: '🏢', shop: '🛒' }
  const c = cfg[floor] || '#64748b'
  return (
    <span style={{ fontSize: '0.68rem', fontWeight: 700, padding: '2px 7px', borderRadius: 99,
      background: `${c}18`, color: c, border: `1px solid ${c}40` }}>
      {icon[floor]} {floor}
    </span>
  )
}

// ── Generic alert list ──────────────────────────────────────────────────────
function AlertList({ events, emptyMsg }) {
  if (!events || events.length === 0) return (
    <div style={{ padding: 36, textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.84rem' }}>
      <AlertTriangle size={28} style={{ opacity: 0.2, marginBottom: 8 }} />
      <div>{emptyMsg || 'No events recorded.'}</div>
    </div>
  )
  return (
    <div>
      {events.map((e, i) => (
        <div key={e.id || i} style={{
          padding: '12px 18px', display: 'flex', alignItems: 'flex-start', gap: 10,
          borderBottom: i < events.length - 1 ? '1px solid var(--border)' : 'none',
          background: i % 2 === 0 ? 'transparent' : 'var(--bg-secondary)',
        }}>
          <AlertTriangle size={14} color="#f97316" style={{ flexShrink: 0, marginTop: 2 }} />
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: '0.82rem', color: 'var(--text-primary)', lineHeight: 1.4 }}>
              {e.message || e.detected_issue || '—'}
            </div>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: 4, display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
              <SevBadge severity={e.severity} />
              <FloorBadge floor={e.floor} />
              <Clock size={11} />
              <span>{new Date(e.timestamp).toLocaleString('en-IN', { dateStyle: 'short', timeStyle: 'short' })}</span>
              {e.camera_id && <span>Camera #{e.camera_id}</span>}
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}

// ── Lift events table ────────────────────────────────────────────────────────
function LiftList({ events }) {
  if (!events || events.length === 0) return (
    <div style={{ padding: 36, textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.84rem' }}>
      No lift events recorded. Draw a lift zone polygon on a camera to start tracking.
    </div>
  )
  const typeColor = { entry: 'var(--accent-green)', exit: 'var(--accent-blue)', idle: '#f97316' }
  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr style={{ background: 'var(--bg-secondary)' }}>
            {['Event', 'Track', 'Floor', 'Duration', 'Time'].map(h => (
              <th key={h} style={{ padding: '9px 14px', textAlign: 'left', fontSize: '0.7rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {events.map((e, i) => (
            <tr key={e.id} style={{ borderTop: '1px solid var(--border)', background: i % 2 === 0 ? 'transparent' : 'var(--bg-secondary)' }}>
              <td style={{ padding: '9px 14px' }}>
                <span style={{ fontSize: '0.78rem', fontWeight: 700, color: typeColor[e.event_type] || 'var(--text-muted)', textTransform: 'capitalize' }}>
                  {e.event_type === 'entry' ? '▲ Entry' : e.event_type === 'exit' ? '▼ Exit' : '⏰ Idle'}
                </span>
              </td>
              <td style={{ padding: '9px 14px', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>#{e.track_id}</td>
              <td style={{ padding: '9px 14px', fontSize: '0.8rem', color: 'var(--text-secondary)', textTransform: 'capitalize' }}>
                {e.floor_from || '—'}{e.floor_to ? ` → ${e.floor_to}` : ''}
              </td>
              <td style={{ padding: '9px 14px', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                {e.duration_sec ? `${Math.round(e.duration_sec)}s` : '—'}
              </td>
              <td style={{ padding: '9px 14px', fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                {new Date(e.timestamp).toLocaleString('en-IN', { timeStyle: 'short', dateStyle: 'short' })}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ── Packing panel ────────────────────────────────────────────────────────────
function PackingPanel({ data }) {
  if (!data) return null
  return (
    <div>
      <div style={{ padding: '12px 18px', borderBottom: '1px solid var(--border)', fontSize: '0.84rem', color: 'var(--text-muted)' }}>
        Today's packing zone idle alerts: <strong style={{ color: 'var(--text-primary)' }}>{data.today_count ?? 0}</strong>
        <span style={{ marginLeft: 8, fontSize: '0.72rem' }}>— hands must be in movement</span>
      </div>
      <AlertList events={data.events || []} emptyMsg="No packing zone idle alerts today." />
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function WorkflowMonitor() {
  const { addToast } = useToast()
  const [activeTab, setActiveTab] = useState('idle')
  const [summary,   setSummary]   = useState(null)
  const [data,      setData]      = useState({})
  const [loading,   setLoading]   = useState(false)

  const loadSummary = useCallback(async () => {
    try {
      const res = await workflowApi.summary()
      setSummary(res.data)
    } catch { /* silent */ }
  }, [])

  const loadTab = useCallback(async (tab) => {
    setLoading(true)
    try {
      const fn = workflowApi[tab]
      if (!fn) return
      const res = await fn()
      setData(prev => ({ ...prev, [tab]: res.data }))
    } catch {
      addToast('Failed to load events', '', 'danger')
    } finally {
      setLoading(false)
    }
  }, [addToast])

  useEffect(() => { loadSummary() }, [loadSummary])
  useEffect(() => { loadTab(activeTab) }, [activeTab, loadTab])

  const renderContent = () => {
    const d = data[activeTab]
    if (loading) return (
      <div style={{ padding: 36, textAlign: 'center', color: 'var(--text-muted)' }}>
        <span className="spinner" style={{ marginRight: 8 }} />Loading…
      </div>
    )
    if (!d) return null
    if (activeTab === 'lift')    return <LiftList events={Array.isArray(d) ? d : []} />
    if (activeTab === 'packing') return <PackingPanel data={d} />
    return <AlertList events={Array.isArray(d) ? d : []}
      emptyMsg={`No ${TABS.find(t => t.id === activeTab)?.label || ''} events today.`} />
  }

  const summaryItems = [
    { label: 'Idle Events',  value: summary?.idle_events_today,       icon: Timer,      color: '#f59e0b' },
    { label: 'Dress Code',   value: summary?.dress_code_events_today, icon: HardHat,    color: '#ef4444' },
    { label: 'Cash Events',  value: summary?.cash_events_today,       icon: DollarSign, color: '#10b981' },
    { label: 'Stock Events', value: summary?.stock_events_today,      icon: Package,    color: '#8b5cf6' },
    { label: 'Oven/Gas',     value: summary?.oven_events_today,       icon: Flame,      color: '#f97316' },
    { label: 'Lift Events',  value: summary?.lift_events_today,       icon: ArrowUpDown,color: 'var(--accent-blue)' },
  ]

  return (
    <div className="page-container">
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 24, flexWrap: 'wrap', gap: 12 }}>
        <div>
          <h1 style={{ fontSize: '1.4rem', fontWeight: 800, color: 'var(--text-primary)', margin: 0, display: 'flex', alignItems: 'center', gap: 8 }}>
            <Briefcase size={20} style={{ verticalAlign: 'middle' }} />
            Workflow Monitor
          </h1>
          <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: 3 }}>
            Idle · Dress Code · Cash · Stock · Oven/Gas · Lift · Packing
          </div>
        </div>
        <button className="btn btn-ghost" onClick={() => { loadSummary(); loadTab(activeTab) }}>
          <RefreshCw size={14} />
        </button>
      </div>

      {/* Summary cards */}
      {summary && (
        <div style={{ display: 'flex', gap: 12, marginBottom: 24, flexWrap: 'wrap' }}>
          {summaryItems.map(s => (
            <SummaryCard key={s.label} label={s.label} value={s.value} icon={s.icon} color={s.color} />
          ))}
        </div>
      )}

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 16, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button
            key={t.id}
            onClick={() => setActiveTab(t.id)}
            className={`btn ${activeTab === t.id ? 'btn-primary' : 'btn-ghost'}`}
            style={{ gap: 6, fontSize: '0.82rem',
              ...(activeTab === t.id ? {} : { color: t.color, borderColor: `${t.color}30` })
            }}
          >
            <t.icon size={14} color={activeTab === t.id ? '#fff' : t.color} />
            {t.label}
          </button>
        ))}
      </div>

      {/* Content panel */}
      <div style={{
        background: 'var(--bg-card)', border: '1px solid var(--border)',
        borderRadius: 'var(--radius-lg)', overflow: 'hidden',
      }}>
        {/* Panel header */}
        <div style={{ padding: '12px 18px', borderBottom: '1px solid var(--border)', display: 'flex', alignItems: 'center', gap: 8 }}>
          {(() => {
            const tab = TABS.find(t => t.id === activeTab)
            if (!tab) return null
            return <>
              <tab.icon size={15} color={tab.color} />
              <span style={{ fontWeight: 700, fontSize: '0.88rem', color: 'var(--text-primary)' }}>{tab.label}</span>
              <span style={{ marginLeft: 'auto', fontSize: '0.72rem', color: 'var(--text-muted)' }}>{tab.desc}</span>
            </>
          })()}
        </div>
        {renderContent()}
      </div>
    </div>
  )
}
