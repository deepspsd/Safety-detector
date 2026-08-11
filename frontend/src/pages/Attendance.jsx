import { useState, useEffect, useCallback } from 'react'
import api from '../api/api'
import { useToast } from '../context/ToastContext'
import {
  Users, UserCheck, UserX, Clock, LogIn, LogOut,
  Download, RefreshCw, Calendar, Plus, X, ChevronDown
} from 'lucide-react'

// ── API helpers ────────────────────────────────────────────────────────────────
const attendanceApi = {
  stats:    ()              => api.get('/attendance/stats'),
  list:     (params)        => api.get('/attendance/', { params }),
  clockIn:  (data)          => api.post('/attendance/clock-in', data),
  clockOut: (recordId)      => api.post(`/attendance/clock-out/${recordId}`),
  export:   (date)          => api.get('/attendance/export', { params: date ? { date } : {}, responseType: 'blob' }),
}

// ── Stat card ──────────────────────────────────────────────────────────────────
 
function StatCard({ label, value, icon: StatIcon, color, sub }) {
  const Ic = StatIcon
  return (
    <div style={{
      background: 'var(--bg-card)', border: '1px solid var(--border)',
      borderRadius: 'var(--radius-lg)', padding: '18px 22px',
      display: 'flex', alignItems: 'center', gap: 14, flex: 1, minWidth: 160,
    }}>
      <div style={{
        width: 44, height: 44, borderRadius: 12,
        background: `${color}18`, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
      }}>
        <Ic size={20} color={color} />
      </div>
      <div>
        <div style={{ fontSize: '1.6rem', fontWeight: 800, color, lineHeight: 1 }}>{value}</div>
        <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: 3, textTransform: 'uppercase', letterSpacing: '0.05em' }}>{label}</div>
        {sub && <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: 2 }}>{sub}</div>}
      </div>
    </div>
  )
}

// ── Method badge ───────────────────────────────────────────────────────────────
function MethodBadge({ method }) {
  const cfg = {
    face:   { label: '🤳 Face',   color: 'var(--accent-blue)' },
    manual: { label: '✍️ Manual', color: 'var(--text-muted)'  },
    qr:     { label: '📷 QR',     color: 'var(--accent-green)'},
  }[method] || { label: method, color: 'var(--text-muted)' }

  return (
    <span style={{
      fontSize: '0.7rem', fontWeight: 600, padding: '2px 8px',
      borderRadius: 99, color: cfg.color,
      background: `${cfg.color}18`, border: `1px solid ${cfg.color}40`,
    }}>{cfg.label}</span>
  )
}

// ── Duration formatter ─────────────────────────────────────────────────────────
function fmtDuration(sec) {
  if (!sec) return '—'
  const h = Math.floor(sec / 3600)
  const m = Math.floor((sec % 3600) / 60)
  if (h > 0) return `${h}h ${m}m`
  return `${m}m`
}

function fmtTime(iso) {
  if (!iso) return '—'
  return new Date(iso).toLocaleString('en-IN', { timeStyle: 'short' })
}

// ── Clock-in modal ─────────────────────────────────────────────────────────────
function ClockInModal({ onClose, onSuccess }) {
  const { addToast } = useToast()
  const [employees, setEmployees]   = useState([])
  const [empId, setEmpId]           = useState('')
  const [notes, setNotes]           = useState('')
  const [saving, setSaving]         = useState(false)

  useEffect(() => {
    api.get('/faces/').then(r => setEmployees(Array.isArray(r.data) ? r.data : [])).catch(() => {})
  }, [])

  const handleSubmit = async () => {
    setSaving(true)
    try {
      await attendanceApi.clockIn({
        employee_id: empId ? parseInt(empId) : null,
        method:      'manual',
        notes:       notes || null,
      })
      addToast('Clock-in recorded', '', 'success')
      onSuccess()
      onClose()
    } catch {
      addToast('Clock-in failed', '', 'danger')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="drawer-overlay" onClick={onClose}>
      <div className="drawer-panel" style={{ maxWidth: 420 }} onClick={e => e.stopPropagation()}>
        <div className="drawer-header">
          <h3>Manual Clock-In</h3>
          <button className="btn btn-ghost btn-icon" onClick={onClose}><X size={18} /></button>
        </div>
        <div className="drawer-body" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          <div>
            <label className="form-label">Employee (optional)</label>
            <div style={{ position: 'relative' }}>
              <select
                className="form-input"
                value={empId}
                onChange={e => setEmpId(e.target.value)}
                style={{ appearance: 'none', paddingRight: 32 }}
              >
                <option value="">— Unidentified / Visitor —</option>
                {employees.map(e => (
                  <option key={e.id} value={e.id}>{e.label}</option>
                ))}
              </select>
              <ChevronDown size={14} style={{ position: 'absolute', right: 10, top: '50%', transform: 'translateY(-50%)', pointerEvents: 'none', color: 'var(--text-muted)' }} />
            </div>
          </div>
          <div>
            <label className="form-label">Notes (optional)</label>
            <input
              className="form-input"
              placeholder="Reason, shift, etc."
              value={notes}
              onChange={e => setNotes(e.target.value)}
            />
          </div>
          <button
            className="btn btn-primary"
            style={{ marginTop: 4 }}
            onClick={handleSubmit}
            disabled={saving}
          >
            {saving ? 'Saving…' : <><LogIn size={15} /> Clock In</>}
          </button>
        </div>
      </div>
    </div>
  )
}

// ── Main page ──────────────────────────────────────────────────────────────────
export default function Attendance() {
  const { addToast } = useToast()
  const [stats,     setStats]     = useState(null)
  const [records,   setRecords]   = useState([])
  const [loading,   setLoading]   = useState(true)
  const [showModal, setShowModal] = useState(false)
  const [date,      setDate]      = useState(() => new Date().toISOString().slice(0, 10))

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const [sRes, rRes] = await Promise.all([
        attendanceApi.stats(),
        attendanceApi.list({ date }),
      ])
      setStats(sRes.data)
      setRecords(Array.isArray(rRes.data) ? rRes.data : [])
    } catch {
      addToast('Failed to load attendance', '', 'danger')
    } finally {
      setLoading(false)
    }
  }, [date, addToast])

  useEffect(() => { load() }, [load])

  const handleClockOut = async (recordId) => {
    try {
      await attendanceApi.clockOut(recordId)
      addToast('Clocked out', '', 'success')
      load()
    } catch {
      addToast('Clock-out failed', '', 'danger')
    }
  }

  const handleExport = async () => {
    try {
      const res = await attendanceApi.export(date)
      const url = URL.createObjectURL(new Blob([res.data]))
      const a = document.createElement('a')
      a.href = url; a.download = `attendance_${date}.csv`; a.click()
      URL.revokeObjectURL(url)
    } catch {
      addToast('Export failed', '', 'danger')
    }
  }

  return (
    <div className="page-container">
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 24, flexWrap: 'wrap', gap: 12 }}>
        <div>
          <h1 style={{ fontSize: '1.4rem', fontWeight: 800, color: 'var(--text-primary)', margin: 0 }}>
            <Users size={20} style={{ verticalAlign: 'middle', marginRight: 8 }} />
            Attendance
          </h1>
          <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: 3 }}>
            Employee clock-in / clock-out log
          </div>
        </div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <div style={{ position: 'relative' }}>
            <Calendar size={14} style={{ position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
            <input
              type="date"
              className="form-input"
              style={{ paddingLeft: 30, height: 36, fontSize: '0.82rem' }}
              value={date}
              onChange={e => setDate(e.target.value)}
            />
          </div>
          <button className="btn btn-ghost" style={{ gap: 6 }} onClick={load}>
            <RefreshCw size={14} />
          </button>
          <button className="btn btn-ghost" style={{ gap: 6 }} onClick={handleExport}>
            <Download size={14} /> Export CSV
          </button>
          <button className="btn btn-primary" style={{ gap: 6 }} onClick={() => setShowModal(true)}>
            <Plus size={14} /> Clock In
          </button>
        </div>
      </div>

      {/* Stats */}
      {stats && (
        <div style={{ display: 'flex', gap: 12, marginBottom: 24, flexWrap: 'wrap' }}>
          <StatCard label="Present Today"   value={stats.present_count}   icon={UserCheck} color="var(--accent-green)" sub={`of ${stats.total_employees} employees`} />
          <StatCard label="Absent"          value={stats.absent_count}    icon={UserX}     color="#ef4444" />
          <StatCard label="Late Arrivals"   value={stats.late_count}      icon={Clock}     color="#f97316" sub="clocked in after 09:00" />
          <StatCard label="Currently On-site" value={stats.open_sessions} icon={Users}     color="var(--accent-blue)" />
        </div>
      )}

      {/* Records table */}
      <div style={{
        background: 'var(--bg-card)', border: '1px solid var(--border)',
        borderRadius: 'var(--radius-lg)', overflow: 'hidden',
      }}>
        <div style={{ padding: '14px 18px', borderBottom: '1px solid var(--border)', fontWeight: 700, fontSize: '0.88rem', color: 'var(--text-primary)' }}>
          Records for {new Date(date + 'T12:00:00').toLocaleDateString('en-IN', { dateStyle: 'long' })}
          <span style={{ marginLeft: 8, color: 'var(--text-muted)', fontWeight: 400, fontSize: '0.78rem' }}>
            ({records.length} entries)
          </span>
        </div>
        {loading ? (
          <div style={{ padding: 32, textAlign: 'center', color: 'var(--text-muted)' }}>
            <span className="spinner" style={{ marginRight: 8 }} />Loading…
          </div>
        ) : records.length === 0 ? (
          <div style={{ padding: 40, textAlign: 'center', color: 'var(--text-muted)' }}>
            No attendance records for this date.
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ background: 'var(--bg-secondary)' }}>
                  {['Employee', 'Method', 'Clock In', 'Clock Out', 'Duration', 'Notes', 'Action'].map(h => (
                    <th key={h} style={{
                      padding: '10px 14px', textAlign: 'left', fontSize: '0.72rem',
                      fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase',
                      letterSpacing: '0.05em', whiteSpace: 'nowrap',
                    }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {records.map((r, i) => (
                  <tr key={r.id} style={{
                    borderTop: '1px solid var(--border)',
                    background: i % 2 === 0 ? 'transparent' : 'var(--bg-secondary)',
                  }}>
                    <td style={{ padding: '10px 14px' }}>
                      <div style={{ fontWeight: 600, fontSize: '0.84rem', color: 'var(--text-primary)' }}>
                        {r.employee_name || <span style={{ color: 'var(--text-muted)' }}>Unidentified</span>}
                      </div>
                      {r.employee_id && (
                        <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>ID #{r.employee_id}</div>
                      )}
                    </td>
                    <td style={{ padding: '10px 14px' }}><MethodBadge method={r.method} /></td>
                    <td style={{ padding: '10px 14px', fontSize: '0.82rem', color: 'var(--text-primary)' }}>
                      {fmtTime(r.clock_in)}
                    </td>
                    <td style={{ padding: '10px 14px', fontSize: '0.82rem', color: r.is_open ? 'var(--accent-green)' : 'var(--text-primary)' }}>
                      {r.is_open ? '● On-site' : fmtTime(r.clock_out)}
                    </td>
                    <td style={{ padding: '10px 14px', fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
                      {fmtDuration(r.duration_seconds)}
                    </td>
                    <td style={{ padding: '10px 14px', fontSize: '0.78rem', color: 'var(--text-muted)', maxWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {r.notes || '—'}
                    </td>
                    <td style={{ padding: '10px 14px' }}>
                      {r.is_open && (
                        <button
                          className="btn btn-ghost"
                          style={{ fontSize: '0.72rem', padding: '3px 10px', gap: 4, color: '#ef4444' }}
                          onClick={() => handleClockOut(r.id)}
                        >
                          <LogOut size={12} /> Clock Out
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {showModal && (
        <ClockInModal onClose={() => setShowModal(false)} onSuccess={load} />
      )}
    </div>
  )
}
