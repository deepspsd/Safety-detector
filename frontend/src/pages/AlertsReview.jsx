import { useState, useEffect, useCallback, useRef } from 'react'
import { alertsApi } from '../api/api'
import { useToast } from '../context/ToastContext'
import {
  ShieldAlert, CheckCircle, XCircle, Clock, Camera,
  RefreshCw, Filter, AlertTriangle, Eye, Bell, Inbox
} from 'lucide-react'

const SEV_CFG = {
  critical: { color: '#f87171', bg: 'rgba(239,68,68,0.12)',  border: 'rgba(239,68,68,0.3)'  },
  high:     { color: '#fbbf24', bg: 'rgba(245,158,11,0.12)', border: 'rgba(245,158,11,0.3)' },
  medium:   { color: '#60a5fa', bg: 'rgba(59,130,246,0.12)', border: 'rgba(59,130,246,0.3)' },
  low:      { color: '#34d399', bg: 'rgba(16,185,129,0.12)', border: 'rgba(16,185,129,0.3)' },
}

const AUTO_REFRESH_S = 15

function AlertCard({ alert, onConfirm, onDismiss, busy }) {
  const [expanded, setExpanded] = useState(false)
  const sc = SEV_CFG[alert.severity] || SEV_CFG.low
  const BASE = import.meta.env.VITE_API_BASE_URL?.replace(/\/api$/, '') || ''
  const snapSrc = alert.snapshot_url
    ? (alert.snapshot_url.startsWith('http') ? alert.snapshot_url : `${BASE}${alert.snapshot_url}`)
    : alert.snapshot_b64 || null

  return (
    <div className="review-card" style={{ borderLeft: `3px solid ${sc.color}` }}>
      {/* snapshot */}
      {snapSrc && (
        <div className="review-card-img" onClick={() => setExpanded(v => !v)}>
          <img src={snapSrc} alt="alert snapshot" style={{ width: '100%', objectFit: 'cover', borderRadius: 8, cursor: 'zoom-in' }} />
          {expanded && (
            <div style={{
              position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.85)', zIndex: 9999,
              display: 'flex', alignItems: 'center', justifyContent: 'center'
            }} onClick={() => setExpanded(false)}>
              <img src={snapSrc} alt="full" style={{ maxWidth: '90vw', maxHeight: '90vh', borderRadius: 12 }} />
            </div>
          )}
        </div>
      )}

      {/* meta */}
      <div className="review-card-body">
        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 8 }}>
          <div>
            <span style={{
              display: 'inline-block', padding: '2px 9px', borderRadius: 99,
              background: sc.bg, color: sc.color, border: `1px solid ${sc.border}`,
              fontSize: '0.7rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 6
            }}>{alert.severity}</span>
            <div style={{ fontWeight: 700, fontSize: '0.95rem', marginBottom: 2 }}>
              {alert.detected_issue || alert.message || 'Violation detected'}
            </div>
            {alert.message && alert.detected_issue && (
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{alert.message}</div>
            )}
          </div>
          {alert.confidence != null && (
            <div style={{ textAlign: 'right', flexShrink: 0 }}>
              <div style={{ fontSize: '1.1rem', fontWeight: 700, color: sc.color }}>
                {Math.round(alert.confidence * 100)}%
              </div>
              <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Confidence</div>
            </div>
          )}
        </div>

        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginTop: 8, fontSize: '0.77rem', color: 'var(--text-muted)' }}>
          {alert.floor    && <span><Camera size={11} style={{ marginRight: 3 }} />{alert.floor} floor</span>}
          {alert.camera_id && <span>Cam #{alert.camera_id}</span>}
          {alert.role     && <span>Role: {alert.role}</span>}
          <span><Clock size={11} style={{ marginRight: 3 }} />{new Date(alert.timestamp).toLocaleString()}</span>
        </div>

        {/* Action buttons */}
        <div style={{ display: 'flex', gap: 8, marginTop: 14 }}>
          <button
            className="btn btn-sm"
            style={{ flex: 1, background: 'rgba(16,185,129,0.15)', color: '#34d399',
              border: '1px solid rgba(16,185,129,0.3)', gap: 6 }}
            onClick={() => onConfirm(alert.id)}
            disabled={busy}
          >
            <CheckCircle size={14} />
            Confirm &amp; Notify
          </button>
          <button
            className="btn btn-sm"
            style={{ flex: 1, background: 'rgba(239,68,68,0.1)', color: '#f87171',
              border: '1px solid rgba(239,68,68,0.25)', gap: 6 }}
            onClick={() => onDismiss(alert.id)}
            disabled={busy}
          >
            <XCircle size={14} />
            Dismiss
          </button>
        </div>
      </div>
    </div>
  )
}

export default function AlertsReview() {
  const { addToast } = useToast()
  const [pending,    setPending]    = useState([])
  const [loading,    setLoading]    = useState(true)
  const [busy,       setBusy]       = useState(false)
  const [countdown,  setCountdown]  = useState(AUTO_REFRESH_S)
  const timerRef = useRef(null)
  const countRef = useRef(null)

  const fetchPending = useCallback(async () => {
    try {
      const res = await alertsApi.pending()
      // backend returns list directly or { alerts: [...] }
      const list = Array.isArray(res.data) ? res.data : (res.data.alerts || [])
      setPending(list)
    } catch { addToast('Failed to load pending alerts', '', 'danger') }
    finally { setLoading(false); setCountdown(AUTO_REFRESH_S) }
  }, [addToast])

  // Auto-refresh every AUTO_REFRESH_S seconds
  useEffect(() => {
    fetchPending()
    timerRef.current = setInterval(fetchPending, AUTO_REFRESH_S * 1000)
    countRef.current = setInterval(() => setCountdown(c => (c > 0 ? c - 1 : AUTO_REFRESH_S)), 1000)
    return () => { clearInterval(timerRef.current); clearInterval(countRef.current) }
  }, [fetchPending])

  const handleConfirm = async (id) => {
    setBusy(true)
    try {
      await alertsApi.confirm(id)
      setPending(prev => prev.filter(a => a.id !== id))
      addToast('Alert confirmed', 'Status promoted — Telegram notification sent', 'success')
    } catch (e) {
      addToast('Confirm failed', e.response?.data?.detail || '', 'danger')
    } finally { setBusy(false) }
  }

  const handleDismiss = async (id) => {
    setBusy(true)
    try {
      await alertsApi.dismiss(id)
      setPending(prev => prev.filter(a => a.id !== id))
      addToast('Alert dismissed', 'Marked as false-positive — kept for retraining', 'info')
    } catch (e) {
      addToast('Dismiss failed', e.response?.data?.detail || '', 'danger')
    } finally { setBusy(false) }
  }

  const totalPending = pending.length

  return (
    <div className="page-container">
      <div className="page-header">
        <div>
          <h1 className="page-title">
            <ShieldAlert size={22} style={{ marginRight: 10 }} />
            Alert Review Queue
          </h1>
          <p className="page-subtitle">
            {loading ? 'Loading…' : `${totalPending} pending alert${totalPending !== 1 ? 's' : ''} awaiting review`}
          </p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            <Clock size={11} style={{ marginRight: 4 }} />
            Refreshes in {countdown}s
          </span>
          <button className="btn btn-ghost btn-icon" onClick={fetchPending} title="Refresh now">
            <RefreshCw size={15} />
          </button>
        </div>
      </div>

      {/* legend */}
      <div style={{ display: 'flex', gap: 16, marginBottom: 24, flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.78rem', color: 'var(--text-muted)' }}>
          <CheckCircle size={13} color="#34d399" />
          <strong style={{ color: 'var(--text-secondary)' }}>Confirm</strong> — promotes to confirmed, sends Telegram alert
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.78rem', color: 'var(--text-muted)' }}>
          <XCircle size={13} color="#f87171" />
          <strong style={{ color: 'var(--text-secondary)' }}>Dismiss</strong> — marks as false-positive, kept for model retraining
        </div>
      </div>

      {loading ? (
        <div className="empty-state" style={{ marginTop: 80 }}>
          <span className="spinner" style={{ width: 36, height: 36 }} />
          <p>Loading pending alerts…</p>
        </div>
      ) : pending.length === 0 ? (
        <div className="empty-state" style={{ marginTop: 80 }}>
          <Inbox size={52} color="var(--accent-green)" style={{ opacity: 0.7 }} />
          <h3 style={{ color: 'var(--accent-green)' }}>Queue is clear ✓</h3>
          <p>No alerts are waiting for review right now.</p>
          <p style={{ fontSize: '0.75rem', marginTop: 4 }}>Auto-refreshes every {AUTO_REFRESH_S}s</p>
        </div>
      ) : (
        <div className="review-grid">
          {pending.map(alert => (
            <AlertCard
              key={alert.id}
              alert={alert}
              onConfirm={handleConfirm}
              onDismiss={handleDismiss}
              busy={busy}
            />
          ))}
        </div>
      )}
    </div>
  )
}
