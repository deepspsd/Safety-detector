import { useState, useEffect, useCallback, useRef } from 'react'
import api from '../api/api'
import { useToast } from '../context/ToastContext'
import {
  FileText, Upload, CheckCircle, XCircle, RefreshCw,
  Filter, ChevronDown, Eye, EyeOff
} from 'lucide-react'

// ── API helpers ────────────────────────────────────────────────────────────────
const documentsApi = {
  stats:   ()           => api.get('/documents/stats'),
  list:    (params)     => api.get('/documents/', { params }),
  approve: (id, table)  => api.patch(`/documents/${id}/approve?table=${table}`),
  reject:  (id, table)  => api.patch(`/documents/${id}/reject?table=${table}`),
}

// ── Stat card ──────────────────────────────────────────────────────────────────
function Stat({ label, value, color }) {
  return (
    <div style={{
      background: 'var(--bg-card)', border: '1px solid var(--border)',
      borderRadius: 'var(--radius-lg)', padding: '14px 20px', flex: 1, minWidth: 140,
    }}>
      <div style={{ fontSize: '1.5rem', fontWeight: 800, color }}>{value}</div>
      <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginTop: 3 }}>{label}</div>
    </div>
  )
}

// ── Direction badge ────────────────────────────────────────────────────────────
function DirBadge({ direction }) {
  const isIn = direction === 'inward'
  return (
    <span style={{
      fontSize: '0.7rem', fontWeight: 700, padding: '2px 8px', borderRadius: 99,
      color:       isIn ? 'var(--accent-green)' : '#f97316',
      background:  isIn ? 'rgba(16,185,129,0.12)' : 'rgba(249,115,22,0.12)',
      border:      `1px solid ${isIn ? 'rgba(16,185,129,0.3)' : 'rgba(249,115,22,0.3)'}`,
    }}>
      {isIn ? '⬇ Inward' : '⬆ Outward'}
    </span>
  )
}

// ── Status badge ───────────────────────────────────────────────────────────────
function StatusBadge({ approved, ocr_available }) {
  if (!ocr_available) return (
    <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>OCR unavailable</span>
  )
  return approved
    ? <span style={{ fontSize: '0.7rem', color: 'var(--accent-green)', fontWeight: 700 }}><CheckCircle size={11} style={{ verticalAlign: 'middle', marginRight: 3 }} />Approved</span>
    : <span style={{ fontSize: '0.7rem', color: '#ef4444', fontWeight: 700 }}><XCircle size={11} style={{ verticalAlign: 'middle', marginRight: 3 }} />Rejected</span>
}

// ── Manual scan panel ──────────────────────────────────────────────────────────
function ScanPanel({ onScanned }) {
  const { addToast } = useToast()
  const [direction, setDirection] = useState('inward')
  const [file, setFile]           = useState(null)
  const [preview, setPreview]     = useState(null)
  const [scanning, setScanning]   = useState(false)
  const [result, setResult]       = useState(null)
  const fileRef = useRef()

  const handleFile = e => {
    const f = e.target.files?.[0]
    if (!f) return
    setFile(f)
    setResult(null)
    const reader = new FileReader()
    reader.onload = ev => setPreview(ev.target.result)
    reader.readAsDataURL(f)
  }

  const handleScan = async () => {
    if (!file) { addToast('No file selected', 'Upload a document image first', 'warning'); return }
    setScanning(true)
    try {
      const form = new FormData()
      form.append('file', file)
      form.append('direction', direction)
      const res = await api.post('/documents/scan', form, { headers: { 'Content-Type': 'multipart/form-data' } })
      setResult(res.data)
      if (res.data.approved) addToast('Document approved ✅', res.data.raw_text?.slice(0, 60), 'success')
      else addToast('Document rejected ⚠️', 'OCR text did not match expected patterns', 'warning')
      onScanned?.()
    } catch {
      addToast('Scan failed', 'Check backend and OCR setup', 'danger')
    } finally {
      setScanning(false)
    }
  }

  return (
    <div style={{
      background: 'var(--bg-card)', border: '1px solid var(--border)',
      borderRadius: 'var(--radius-lg)', padding: 20, marginBottom: 20,
    }}>
      <div style={{ fontWeight: 700, fontSize: '0.9rem', color: 'var(--text-primary)', marginBottom: 14, display: 'flex', alignItems: 'center', gap: 7 }}>
        <Upload size={15} /> Manual Document Scan
      </div>
      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'flex-start' }}>
        {/* Direction toggle */}
        <div style={{ display: 'flex', gap: 6 }}>
          {['inward', 'outward'].map(d => (
            <button
              key={d}
              className={`btn ${direction === d ? 'btn-primary' : 'btn-ghost'}`}
              style={{ fontSize: '0.78rem', padding: '5px 14px' }}
              onClick={() => setDirection(d)}
            >
              {d === 'inward' ? '⬇ Inward' : '⬆ Outward'}
            </button>
          ))}
        </div>

        {/* File upload */}
        <button className="btn btn-ghost" style={{ gap: 6, fontSize: '0.78rem' }} onClick={() => fileRef.current?.click()}>
          <Upload size={13} /> {file ? file.name : 'Upload Image'}
        </button>
        <input ref={fileRef} type="file" accept="image/*" style={{ display: 'none' }} onChange={handleFile} />

        {/* Scan button */}
        <button className="btn btn-primary" style={{ gap: 6, fontSize: '0.78rem' }} onClick={handleScan} disabled={scanning || !file}>
          {scanning ? 'Scanning…' : <><FileText size={13} /> Run OCR</>}
        </button>
      </div>

      {/* Preview + result */}
      {(preview || result) && (
        <div style={{ display: 'flex', gap: 16, marginTop: 16, flexWrap: 'wrap' }}>
          {preview && (
            <img src={preview} alt="doc" style={{
              maxWidth: 200, maxHeight: 140, objectFit: 'contain',
              borderRadius: 8, border: '1px solid var(--border)',
            }} />
          )}
          {result && (
            <div style={{ flex: 1, minWidth: 200 }}>
              <div style={{
                padding: '10px 14px', borderRadius: 8,
                background: result.approved ? 'rgba(16,185,129,0.07)' : 'rgba(239,68,68,0.07)',
                border: `1px solid ${result.approved ? 'rgba(16,185,129,0.25)' : 'rgba(239,68,68,0.25)'}`,
              }}>
                <div style={{ fontWeight: 700, fontSize: '0.84rem', color: result.approved ? 'var(--accent-green)' : '#ef4444', marginBottom: 6 }}>
                  {result.approved ? '✅ Document Approved' : '❌ Document Rejected'}
                </div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontFamily: 'monospace', maxHeight: 80, overflowY: 'auto', whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                  {result.raw_text || '(No text extracted)'}
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ── Main page ──────────────────────────────────────────────────────────────────
export default function Documents() {
  const { addToast } = useToast()
  const [stats,      setStats]      = useState(null)
  const [records,    setRecords]    = useState([])
  const [loading,    setLoading]    = useState(true)
  const [direction,  setDirection]  = useState('')
  const [approved,   setApproved]   = useState('')
  const [expandedId, setExpandedId] = useState(null)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const params = {}
      if (direction) params.direction = direction
      if (approved !== '') params.approved = approved === 'true'
      const [sRes, rRes] = await Promise.all([
        documentsApi.stats(),
        documentsApi.list(params),
      ])
      setStats(sRes.data)
      setRecords(Array.isArray(rRes.data) ? rRes.data : [])
    } catch {
      addToast('Failed to load documents', '', 'danger')
    } finally {
      setLoading(false)
    }
  }, [direction, approved, addToast])

  useEffect(() => { load() }, [load])

  const handleApprove = async (id, table) => {
    try {
      await documentsApi.approve(id, table)
      addToast('Approved', '', 'success')
      load()
    } catch { addToast('Failed', '', 'danger') }
  }

  const handleReject = async (id, table) => {
    try {
      await documentsApi.reject(id, table)
      addToast('Rejected', '', 'success')
      load()
    } catch { addToast('Failed', '', 'danger') }
  }

  return (
    <div className="page-container">
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 24, flexWrap: 'wrap', gap: 12 }}>
        <div>
          <h1 style={{ fontSize: '1.4rem', fontWeight: 800, color: 'var(--text-primary)', margin: 0 }}>
            <FileText size={20} style={{ verticalAlign: 'middle', marginRight: 8 }} />
            Document Scan Log
          </h1>
          <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: 3 }}>
            Invoice &amp; order-form OCR gate — inward and outward
          </div>
        </div>
        <button className="btn btn-ghost" onClick={load}><RefreshCw size={14} /></button>
      </div>

      {/* Stats */}
      {stats && (
        <div style={{ display: 'flex', gap: 12, marginBottom: 20, flexWrap: 'wrap' }}>
          <Stat label="Today Total"    value={stats.today_total}    color="var(--text-primary)" />
          <Stat label="Inward"         value={stats.today_inward}   color="var(--accent-green)" />
          <Stat label="Outward"        value={stats.today_outward}  color="#f97316" />
          <Stat label="Approved"       value={stats.today_approved} color="var(--accent-green)" />
          <Stat label="Rejected"       value={stats.today_rejected} color="#ef4444" />
        </div>
      )}

      {/* Scan panel */}
      <ScanPanel onScanned={load} />

      {/* Filters */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 14, flexWrap: 'wrap', alignItems: 'center' }}>
        <Filter size={14} color="var(--text-muted)" />
        {['', 'inward', 'outward'].map(d => (
          <button key={d || 'all'} onClick={() => setDirection(d)}
            className={`btn ${direction === d ? 'btn-primary' : 'btn-ghost'}`}
            style={{ fontSize: '0.76rem', padding: '4px 12px' }}>
            {d || 'All'}
          </button>
        ))}
        <div style={{ width: 1, height: 22, background: 'var(--border)', margin: '0 4px' }} />
        {[['', 'All Status'], ['true', 'Approved'], ['false', 'Rejected']].map(([v, label]) => (
          <button key={v || 'all-status'} onClick={() => setApproved(v)}
            className={`btn ${approved === v ? 'btn-primary' : 'btn-ghost'}`}
            style={{ fontSize: '0.76rem', padding: '4px 12px' }}>
            {label}
          </button>
        ))}
      </div>

      {/* Records */}
      <div style={{
        background: 'var(--bg-card)', border: '1px solid var(--border)',
        borderRadius: 'var(--radius-lg)', overflow: 'hidden',
      }}>
        {loading ? (
          <div style={{ padding: 32, textAlign: 'center', color: 'var(--text-muted)' }}>
            <span className="spinner" style={{ marginRight: 8 }} />Loading…
          </div>
        ) : records.length === 0 ? (
          <div style={{ padding: 40, textAlign: 'center', color: 'var(--text-muted)' }}>
            No documents found for current filters.
          </div>
        ) : (
          <div>
            {records.map((r, i) => (
              <div key={`${r.table}-${r.id}`} style={{
                padding: '12px 18px',
                borderBottom: i < records.length - 1 ? '1px solid var(--border)' : 'none',
                background: i % 2 === 0 ? 'transparent' : 'var(--bg-secondary)',
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
                  <DirBadge direction={r.direction} />
                  <StatusBadge approved={r.approved} ocr_available={r.ocr_available} />
                  <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginLeft: 'auto' }}>
                    {new Date(r.timestamp).toLocaleString('en-IN', { dateStyle: 'short', timeStyle: 'short' })}
                  </span>
                  <button
                    className="btn btn-ghost btn-icon"
                    onClick={() => setExpandedId(expandedId === r.id ? null : r.id)}
                    style={{ padding: 4 }}
                  >
                    {expandedId === r.id ? <EyeOff size={14} /> : <Eye size={14} />}
                  </button>
                  {!r.approved && (
                    <button className="btn btn-ghost" style={{ fontSize: '0.7rem', padding: '2px 8px', color: 'var(--accent-green)' }}
                      onClick={() => handleApprove(r.id, r.table)}>Approve</button>
                  )}
                  {r.approved && (
                    <button className="btn btn-ghost" style={{ fontSize: '0.7rem', padding: '2px 8px', color: '#ef4444' }}
                      onClick={() => handleReject(r.id, r.table)}>Reject</button>
                  )}
                </div>
                {expandedId === r.id && (
                  <div style={{ marginTop: 10, padding: '10px 14px', background: 'var(--bg-secondary)', borderRadius: 8, fontSize: '0.78rem', fontFamily: 'monospace', color: 'var(--text-secondary)', whiteSpace: 'pre-wrap', wordBreak: 'break-word', maxHeight: 160, overflowY: 'auto' }}>
                    {r.raw_ocr_text || '(No OCR text)'}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
