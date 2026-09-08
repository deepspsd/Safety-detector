import { useState, useEffect, useCallback } from 'react'
import { alertsApi } from '../api/api'
import { useToast } from '../context/ToastContext'
import {
  Bell, Trash2, Eye, Filter, ChevronLeft,
  ChevronRight, Clock, X, Camera, AlertTriangle,
  ShieldAlert, CheckCircle, ExternalLink, User
} from 'lucide-react'

const SEVERITIES = ['', 'critical', 'high', 'medium', 'low']
const FLOORS_LIST = ['', 'ground', 'first', 'second', 'shop']
const FLOOR_LABELS = { ground: '🏭 Ground', first: '🏗️ First', second: '🏢 Second', shop: '🛒 Shop' }

const SEV_COLORS = {
  critical: { bg: 'rgba(220,38,38,0.12)', border: 'rgba(220,38,38,0.3)', text: '#f87171' },
  high:     { bg: 'rgba(245,158,11,0.12)', border: 'rgba(245,158,11,0.3)', text: '#fbbf24' },
  medium:   { bg: 'rgba(59,130,246,0.12)', border: 'rgba(59,130,246,0.3)', text: '#60a5fa' },
  low:      { bg: 'rgba(16,185,129,0.12)', border: 'rgba(16,185,129,0.3)', text: '#34d399' },
}

// Resolve snapshot image source from the SAME origin as the API.
// In dev, the Vite proxy serves /uploads → localhost:8000, so a relative
// URL works both on localhost AND when accessed via a LAN IP (phone/TV).
// AlertsReview.jsx uses the identical pattern — keep them in sync.
const BASE_URL = import.meta.env.VITE_API_BASE_URL?.replace(/\/api$/, '') || ''

export function formatAlertTimestamp(ts) {
  if (!ts) return '—'
  const d = new Date(ts)
  if (isNaN(d.getTime())) return ts
  return d.toLocaleString('en-IN', {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: true
  })
}

export default function Alerts() {
  const { addToast } = useToast()
  const [alerts,        setAlerts]        = useState([])
  const [total,         setTotal]         = useState(0)
  const [page,          setPage]          = useState(1)
  const [loading,       setLoading]       = useState(false)
  const [selectedAlert, setSelectedAlert] = useState(null)
  const [viewMode,      setViewMode]      = useState('cards')   // 'cards' | 'table'
  const [filters,       setFilters]       = useState({ severity: '', floor: '', date_from: '', date_to: '' })

  const fetchAlerts = useCallback(async () => {
    setLoading(true)
    try {
      const params = {
        page, limit: 12,
        ...Object.fromEntries(Object.entries(filters).filter(([, v]) => v))
      }
      const res = await alertsApi.list(params)
      setAlerts(res.data.alerts)
      setTotal(res.data.total)
    } catch {
      addToast('Failed to load alerts', '', 'danger')
    } finally {
      setLoading(false)
    }
  }, [page, filters, addToast])

  useEffect(() => { fetchAlerts() }, [fetchAlerts])

  // Auto-refresh every 20s
  useEffect(() => {
    const t = setInterval(fetchAlerts, 20000)
    return () => clearInterval(t)
  }, [fetchAlerts])

  const openDetail = async (a) => {
    try {
      const res = await alertsApi.get(a.id)
      setSelectedAlert(res.data)
    } catch { setSelectedAlert(a) }
  }

  const deleteAlert = async (id, e) => {
    e.stopPropagation()
    try {
      await alertsApi.delete(id)
      setAlerts(prev => prev.filter(a => a.id !== id))
      setTotal(t => t - 1)
      addToast('Alert deleted', '', 'success')
    } catch { addToast('Delete failed', '', 'danger') }
  }

  const clearAll = async () => {
    if (!confirm('Delete ALL alerts and their snapshot images? This cannot be undone.')) return
    try {
      await alertsApi.clearAll()
      setAlerts([]); setTotal(0)
      addToast('All alerts cleared', '', 'success')
    } catch { addToast('Failed to clear', '', 'danger') }
  }

  const totalPages = Math.ceil(total / 12)
  const setFilter  = (k, v) => { setFilters(f => ({ ...f, [k]: v })); setPage(1) }

  // Resolve snapshot image source — prefer file URL, fall back to base64
  const snapSrc = (a) => {
    if (a.snapshot_url) return `${BASE_URL}${a.snapshot_url}`
    if (a.snapshot_b64) return a.snapshot_b64
    return null
  }

  return (
    <div className="page-container">
      {/* ── Header ─────────────────────────────────────────── */}
      <div className="page-header">
        <div>
          <h1 className="page-title">Alert History</h1>
          <p className="page-subtitle">{total} violation alerts recorded</p>
        </div>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
          {/* View toggle */}
          <div style={{
            display: 'flex', background: 'var(--bg-tertiary)',
            border: '1px solid var(--border)', borderRadius: 8, overflow: 'hidden'
          }}>
            {[['cards', '⊞ Cards'], ['table', '☰ Table']].map(([m, l]) => (
              <button key={m} onClick={() => setViewMode(m)} style={{
                padding: '6px 14px', fontSize: '0.78rem', fontWeight: 600,
                background: viewMode === m ? 'var(--bg-card)' : 'transparent',
                color: viewMode === m ? 'var(--text-primary)' : 'var(--text-muted)',
                border: 'none', cursor: 'pointer', transition: 'all 0.15s'
              }}>{l}</button>
            ))}
          </div>
          <button className="btn btn-danger btn-sm" onClick={clearAll} disabled={total === 0}>
            <Trash2 size={13} /> Clear All
          </button>
        </div>
      </div>

      {/* ── Filters ─────────────────────────────────────────── */}
      <div className="filters-bar">
        <Filter size={14} style={{ color: 'var(--text-muted)' }} />
        <select className="filter-select" value={filters.floor} onChange={e => setFilter('floor', e.target.value)}>
          <option value="">All Floors</option>
          {FLOORS_LIST.filter(Boolean).map(f => <option key={f} value={f}>{FLOOR_LABELS[f]}</option>)}
        </select>
        <select className="filter-select" value={filters.severity} onChange={e => setFilter('severity', e.target.value)}>
          <option value="">All Severities</option>
          {SEVERITIES.filter(Boolean).map(s => (
            <option key={s} value={s}>{s.charAt(0).toUpperCase() + s.slice(1)}</option>
          ))}
        </select>
        <input type="date" className="filter-select" value={filters.date_from}
          onChange={e => setFilter('date_from', e.target.value)} title="From date" />
        <input type="date" className="filter-select" value={filters.date_to}
          onChange={e => setFilter('date_to', e.target.value)} title="To date" />
        {Object.values(filters).some(Boolean) && (
          <button className="btn btn-ghost btn-sm"
            onClick={() => { setFilters({ severity:'', floor:'', date_from:'', date_to:'' }); setPage(1) }}>
            <X size={13} /> Reset
          </button>
        )}
      </div>

      {/* ── Content ─────────────────────────────────────────── */}
      {loading ? (
        <div className="loading-container"><span className="spinner" /><span>Loading alerts…</span></div>
      ) : alerts.length === 0 ? (
        <div className="card" style={{ padding: '60px 24px', textAlign: 'center' }}>
          <Bell size={48} style={{ opacity: 0.2, margin: '0 auto 16px' }} />
          <h3 style={{ marginBottom: 8 }}>No alerts found</h3>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem' }}>
            No violations detected yet — or no alerts match your filters.
          </p>
        </div>
      ) : viewMode === 'cards' ? (
        <AlertCards alerts={alerts} onOpen={openDetail} onDelete={deleteAlert} snapSrc={snapSrc} />
      ) : (
        <AlertTable alerts={alerts} onOpen={openDetail} onDelete={deleteAlert} snapSrc={snapSrc} />
      )}

      {/* ── Pagination ─────────────────────────────────────── */}
      {totalPages > 1 && (
        <div style={{ display:'flex', justifyContent:'center', alignItems:'center', gap:12, marginTop:20 }}>
          <button className="btn btn-ghost btn-sm" disabled={page <= 1} onClick={() => setPage(p => p-1)}>
            <ChevronLeft size={14} />
          </button>
          <span style={{ fontSize:'0.85rem', color:'var(--text-secondary)' }}>
            Page {page} of {totalPages}
          </span>
          <button className="btn btn-ghost btn-sm" disabled={page >= totalPages} onClick={() => setPage(p => p+1)}>
            <ChevronRight size={14} />
          </button>
        </div>
      )}

      {/* ── Detail Modal ────────────────────────────────────── */}
      {selectedAlert && (
        <AlertModal alert={selectedAlert} snapSrc={snapSrc} onClose={() => setSelectedAlert(null)} />
      )}
    </div>
  )
}

// ── Card Grid View ───────────────────────────────────────────────

function AlertCards({ alerts, onOpen, onDelete, snapSrc }) {
  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fill, minmax(310px, 1fr))',
      gap: 16
    }}>
      {alerts.map(a => {
        const sev   = SEV_COLORS[a.severity] || SEV_COLORS.medium
        const src   = snapSrc(a)
        const tsStr = formatAlertTimestamp(a.timestamp)
        return (
          <div key={a.id}
            onClick={() => onOpen(a)}
            style={{
              background: 'var(--bg-card)', border: `1px solid ${sev.border}`,
              borderRadius: 12, overflow: 'hidden', cursor: 'pointer',
              transition: 'transform 0.15s, box-shadow 0.15s',
              boxShadow: '0 2px 8px rgba(0,0,0,0.18)'
            }}
            onMouseEnter={e => { e.currentTarget.style.transform='translateY(-2px)'; e.currentTarget.style.boxShadow='0 6px 20px rgba(0,0,0,0.28)' }}
            onMouseLeave={e => { e.currentTarget.style.transform=''; e.currentTarget.style.boxShadow='0 2px 8px rgba(0,0,0,0.18)' }}
          >
            {/* ── Snapshot image (annotated frame) ── */}
            <div style={{
              height: 180, background: 'var(--bg-tertiary)', position: 'relative',
              overflow: 'hidden', borderBottom: `1px solid ${sev.border}`
            }}>
              {src ? (
                <img src={src} alt="Violation snapshot"
                  style={{ width:'100%', height:'100%', objectFit:'cover' }}
                  onError={e => { e.target.style.display='none'; e.target.nextSibling.style.display='flex' }}
                />
              ) : null}
              <div style={{
                display: src ? 'none' : 'flex',
                flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
                height: '100%', color: 'var(--text-muted)', gap: 8
              }}>
                <Camera size={28} style={{ opacity: 0.3 }} />
                <span style={{ fontSize: '0.78rem' }}>No snapshot</span>
              </div>
              {/* Severity pill over image */}
              <div style={{
                position: 'absolute', top: 10, left: 10,
                background: sev.bg, border: `1px solid ${sev.border}`,
                backdropFilter: 'blur(6px)',
                color: sev.text, padding: '3px 10px', borderRadius: 99,
                fontSize: '0.70rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.04em'
              }}>
                {a.severity}
              </div>
              {/* Camera icon if snapshot exists */}
              {src && (
                <div style={{
                  position: 'absolute', top: 10, right: 10,
                  background: 'rgba(0,0,0,0.5)', borderRadius: 6, padding: '4px 6px'
                }}>
                  <Camera size={12} color="#fff" />
                </div>
              )}
            </div>

            {/* ── Card body ── */}
            <div style={{ padding: '12px 14px' }}>
              <div style={{ display:'flex', justifyContent:'space-between', alignItems:'flex-start', marginBottom: 8 }}>
                <div style={{ fontSize:'0.82rem', fontWeight:600, color:'var(--text-primary)', lineHeight:1.3, flex:1, paddingRight:8 }}>
                  {a.detected_issue || a.message}
                </div>
                <button className="btn btn-ghost btn-icon btn-sm"
                  onClick={e => { e.stopPropagation(); onDelete(a.id, e) }}
                  style={{ flexShrink:0 }} title="Delete">
                  <Trash2 size={13} />
                </button>
              </div>

              <div style={{ fontSize:'0.75rem', color:'var(--text-muted)', marginBottom:8, lineHeight:1.4 }}>
                {a.message !== a.detected_issue && (
                  <div style={{ marginBottom:4, color:'var(--text-secondary)' }}>{a.message}</div>
                )}
              </div>

              <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center' }}>
                <div style={{ display:'flex', flexDirection:'column', gap:3 }}>
                  <div style={{ display:'flex', alignItems:'center', gap:5, fontSize:'0.72rem', color:'var(--text-muted)' }}>
                    <Clock size={11} />
                    {tsStr}
                  </div>
                  {/* Worker name badge */}
                  {a.worker_name && (
                    <div style={{ display:'flex', alignItems:'center', gap:4, fontSize:'0.72rem', color:'#a78bfa', fontWeight:600 }}>
                      <User size={11} />
                      {a.worker_name}
                    </div>
                  )}
                  {a.role && (
                    <span style={{
                      fontSize:'0.70rem', padding:'2px 8px', borderRadius:99, width:'fit-content',
                      background:'rgba(59,130,246,0.12)', color:'#60a5fa', fontWeight:600
                    }}>{a.role}</span>
                  )}
                </div>
                {a.confidence && (
                  <div style={{
                    fontFamily: 'JetBrains Mono, monospace',
                    fontSize: '0.80rem', color: 'var(--accent-cyan)',
                    background: 'rgba(6,182,212,0.08)', padding: '3px 8px', borderRadius: 6
                  }}>
                    {(a.confidence * 100).toFixed(0)}%
                  </div>
                )}
              </div>
            </div>
          </div>
        )
      })}
    </div>
  )
}

// ── Table View ───────────────────────────────────────────────────

function AlertTable({ alerts, onOpen, onDelete, snapSrc }) {
  return (
    <div className="card" style={{ overflow: 'hidden' }}>
      <div style={{ overflowX: 'auto' }}>
        <table className="alerts-table" style={{ margin: '8px 16px', width: 'calc(100% - 32px)' }}>
          <thead>
            <tr>
              <th>📸</th>
              <th>#</th>
              <th>Timestamp</th>
              <th>Issue</th>
              <th>Worker</th>
              <th>Role</th>
              <th>Severity</th>
              <th>Conf.</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {alerts.map(a => {
              const src = snapSrc(a)
              return (
                <tr key={a.id} style={{ cursor:'pointer' }} onClick={() => onOpen(a)}>
                  <td>
                    {src ? (
                      <img src={src} alt="snap" style={{
                        width: 56, height: 40, objectFit:'cover', borderRadius:4,
                        border:'1px solid var(--border)'
                      }} />
                    ) : (
                      <div style={{
                        width:56, height:40, background:'var(--bg-tertiary)', borderRadius:4,
                        display:'flex', alignItems:'center', justifyContent:'center'
                      }}>
                        <Camera size={14} style={{ opacity:0.3 }} />
                      </div>
                    )}
                  </td>
                  <td style={{ color:'var(--text-muted)', fontSize:'0.75rem' }}>#{a.id}</td>
                  <td style={{ color:'var(--text-muted)', fontSize:'0.78rem', whiteSpace:'nowrap' }}>
                    <Clock size={11} style={{ marginRight:4 }} />
                    {formatAlertTimestamp(a.timestamp)}
                  </td>
                  <td style={{ color:'var(--text-primary)', maxWidth:200, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>
                    {a.detected_issue || a.message}
                  </td>
                  <td>
                    {a.worker_name ? (
                      <span style={{ display:'flex', alignItems:'center', gap:4, fontSize:'0.78rem', color:'#a78bfa', fontWeight:600 }}>
                        <User size={11} />{a.worker_name}
                      </span>
                    ) : <span style={{ color:'var(--text-muted)', fontSize:'0.75rem' }}>—</span>}
                  </td>
                  <td><span className="badge badge-medium">{a.role || '—'}</span></td>
                  <td><span className={`badge badge-${a.severity}`}>{a.severity}</span></td>
                  <td style={{ fontFamily:'JetBrains Mono', fontSize:'0.78rem', color:'var(--accent-cyan)' }}>
                    {a.confidence ? `${(a.confidence*100).toFixed(0)}%` : '—'}
                  </td>
                  <td>
                    <div style={{ display:'flex', gap:4 }}>
                      <button className="btn btn-ghost btn-icon btn-sm"
                        onClick={e => { e.stopPropagation(); onOpen(a) }} title="View">
                        <Eye size={13} />
                      </button>
                      <button className="btn btn-ghost btn-icon btn-sm"
                        onClick={e => onDelete(a.id, e)} title="Delete">
                        <Trash2 size={13} />
                      </button>
                    </div>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ── Detail Modal ─────────────────────────────────────────────────

function AlertModal({ alert: a, snapSrc, onClose }) {
  const src = snapSrc(a)
  const sev = SEV_COLORS[a.severity] || SEV_COLORS.medium

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-box" onClick={e => e.stopPropagation()}
        style={{ maxWidth: 680, width: '95vw' }}>

        {/* Header */}
        <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:16 }}>
          <div style={{ display:'flex', alignItems:'center', gap:10 }}>
            <AlertTriangle size={18} color={sev.text} />
            <h3 style={{ fontSize:'1rem' }}>Alert #{a.id}</h3>
            <span className={`badge badge-${a.severity}`}>{a.severity?.toUpperCase()}</span>
          </div>
          <button className="btn btn-ghost btn-icon btn-sm" onClick={onClose}><X size={15} /></button>
        </div>

        {/* Annotated snapshot — the main feature */}
        {src ? (
          <div style={{ marginBottom:16, borderRadius:10, overflow:'hidden',
            border:`1px solid ${sev.border}`, position:'relative' }}>
            <img src={src} alt="Annotated violation frame"
              style={{ width:'100%', maxHeight:360, objectFit:'contain', display:'block',
                background:'#000' }}
            />
            <div style={{
              position:'absolute', bottom:0, left:0, right:0,
              background:'linear-gradient(transparent, rgba(0,0,0,0.7))',
              padding:'20px 12px 8px',
              fontSize:'0.75rem', color:'rgba(255,255,255,0.7)',
              display:'flex', justifyContent:'space-between', alignItems:'flex-end'
            }}>
              <span>🔴 Red = Violating person · 🟢 Green = Compliant</span>
              {a.snapshot_url && (
                <a href={`${BASE_URL}${a.snapshot_url}`} target="_blank" rel="noreferrer"
                  style={{ color:'rgba(255,255,255,0.8)', display:'flex', alignItems:'center', gap:4 }}
                  onClick={e => e.stopPropagation()}>
                  <ExternalLink size={11} /> Full image
                </a>
              )}
            </div>
          </div>
        ) : (
          <div style={{
            background:'var(--bg-tertiary)', border:'1px dashed var(--border)',
            borderRadius:10, padding:'32px 0', textAlign:'center',
            marginBottom:16, color:'var(--text-muted)'
          }}>
            <Camera size={32} style={{ opacity:0.25, margin:'0 auto 10px', display:'block' }} />
            <div style={{ fontSize:'0.85rem' }}>No snapshot captured</div>
            <div style={{ fontSize:'0.75rem', marginTop:4 }}>Start Live Monitor to enable violation snapshots</div>
          </div>
        )}

        {/* Info grid */}
        <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:12, fontSize:'0.85rem' }}>
          <InfoRow label="Timestamp" value={formatAlertTimestamp(a.timestamp)} />
          <InfoRow label="Role" value={a.role || '—'} />
          {a.worker_name && (
            <InfoRow label="Worker" value={
              <span style={{ display:'flex', alignItems:'center', gap:6, color:'#a78bfa', fontWeight:600 }}>
                <User size={13} />{a.worker_name}
              </span>
            } />
          )}
          <InfoRow label="Severity"
            value={<span className={`badge badge-${a.severity}`}>{a.severity}</span>} />
          <InfoRow label="Confidence"
            value={a.confidence ? `${(a.confidence*100).toFixed(1)}%` : '—'} />
          <InfoRow label="Model / Engine" value={a.model_name || 'yolov8x_coco'} />
          <InfoRow label="Zone" value={a.zone_id ? `#${a.zone_id}` : (a.floor ? `${a.floor.toUpperCase()} Floor` : 'Global')} />
          <div style={{ gridColumn:'1/-1' }}>
            <InfoRow label="Detected Issue" value={a.detected_issue || '—'} />
          </div>
          <div style={{ gridColumn:'1/-1' }}>
            <InfoRow label="Alert Message" value={a.message} />
          </div>
        </div>
      </div>
    </div>
  )
}

function InfoRow({ label, value }) {
  return (
    <div>
      <div style={{
        fontSize:'0.7rem', fontWeight:700, color:'var(--text-muted)',
        textTransform:'uppercase', letterSpacing:'0.5px', marginBottom:4
      }}>{label}</div>
      <div style={{ color:'var(--text-primary)' }}>{value}</div>
    </div>
  )
}
