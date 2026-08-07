import { useState, useEffect, useRef, useCallback } from 'react'
import { facesApi } from '../api/api'
import { useToast } from '../context/ToastContext'
import {
  Users, Plus, Trash2, Pencil, Camera, Upload,
  X, Save, CheckCircle, UserCircle, Search, RefreshCw
} from 'lucide-react'

const ROLES_HINT = ['Employee', 'Manager', 'Supervisor', 'Security', 'Visitor', 'Owner']

/** Capture modal: webcam or file upload */
function CaptureModal({ onCapture, onClose }) {
  const videoRef  = useRef(null)
  const canvasRef = useRef(null)
  const [stream,  setStream]  = useState(null)
  const [imgSrc,  setImgSrc]  = useState(null)
  const [label,   setLabel]   = useState('')
  const [camErr,  setCamErr]  = useState(false)
  const { addToast } = useToast()

  useEffect(() => {
    navigator.mediaDevices?.getUserMedia({ video: { facingMode: 'user' } })
      .then(s => { setStream(s); if (videoRef.current) videoRef.current.srcObject = s })
      .catch(() => setCamErr(true))
    return () => { stream?.getTracks().forEach(t => t.stop()) }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const capture = () => {
    const video  = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas) return
    canvas.width  = video.videoWidth  || 640
    canvas.height = video.videoHeight || 480
    canvas.getContext('2d').drawImage(video, 0, 0)
    setImgSrc(canvas.toDataURL('image/jpeg', 0.92))
  }

  const handleFile = e => {
    const f = e.target.files?.[0]
    if (!f) return
    const reader = new FileReader()
    reader.onload = ev => setImgSrc(ev.target.result)
    reader.readAsDataURL(f)
  }

  const handleSave = () => {
    if (!label.trim()) { addToast('Name required', 'Enter the employee name', 'warning'); return }
    if (!imgSrc)       { addToast('Photo required', 'Capture or upload a photo', 'warning'); return }
    onCapture(label.trim(), imgSrc)
  }

  return (
    <div className="drawer-overlay" onClick={onClose}>
      <div className="drawer-panel" style={{ maxWidth: 480 }} onClick={e => e.stopPropagation()}>
        <div className="drawer-header">
          <h3>Enroll Employee Face</h3>
          <button className="btn btn-ghost btn-icon" onClick={onClose}><X size={18} /></button>
        </div>
        <div className="drawer-body">
          <label className="form-label">Employee Name / Label *</label>
          <input className="form-input" placeholder="e.g. Ravi Kumar" value={label}
            onChange={e => setLabel(e.target.value)} list="roles-hint" />
          <datalist id="roles-hint">{ROLES_HINT.map(r => <option key={r} value={r} />)}</datalist>

          <div style={{ marginTop: 16, borderRadius: 'var(--radius-md)', overflow: 'hidden',
            border: '1px solid var(--border)', background: '#000', aspectRatio: '4/3',
            display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative' }}>
            {imgSrc
              ? <img src={imgSrc} alt="captured" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
              : camErr
                ? <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8, color: 'var(--text-muted)' }}>
                    <Camera size={32} /><span style={{ fontSize: '0.8rem' }}>Camera unavailable — upload a file instead</span>
                  </div>
                : <video ref={videoRef} autoPlay playsInline muted style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
            }
          </div>
          <canvas ref={canvasRef} style={{ display: 'none' }} />

          <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
            {!camErr && !imgSrc && (
              <button className="btn btn-primary" style={{ flex: 1 }} onClick={capture}>
                <Camera size={14} /> Capture Photo
              </button>
            )}
            {imgSrc && (
              <button className="btn btn-ghost" style={{ flex: 1 }} onClick={() => setImgSrc(null)}>
                <RotateCcw size={14} /> Retake
              </button>
            )}
            <label className="btn btn-ghost" style={{ flex: 1, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 6, justifyContent: 'center' }}>
              <Upload size={14} /> Upload File
              <input type="file" accept="image/*" style={{ display: 'none' }} onChange={handleFile} />
            </label>
          </div>
        </div>
        <div className="drawer-footer">
          <button className="btn btn-ghost" onClick={onClose}>Cancel</button>
          <button className="btn btn-primary" onClick={handleSave}>
            <Save size={14} /> Enroll
          </button>
        </div>
      </div>
    </div>
  )
}

// Missing import fix
function RotateCcw({ size }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
      <polyline points="1 4 1 10 7 10" /><path d="M3.51 15a9 9 0 1 0 .49-4.11" />
    </svg>
  )
}

export default function Employees() {
  const { addToast } = useToast()
  const [faces,       setFaces]       = useState([])
  const [loading,     setLoading]     = useState(true)
  const [search,      setSearch]      = useState('')
  const [showCapture, setShowCapture] = useState(false)
  const [editingId,   setEditingId]   = useState(null)
  const [editLabel,   setEditLabel]   = useState('')
  const [deleting,    setDeleting]    = useState(null)
  const [enrolling,   setEnrolling]   = useState(false)

  const loadFaces = useCallback(async () => {
    try {
      const res = await facesApi.list()
      setFaces(res.data)
    } catch { addToast('Failed to load employees', '', 'danger') }
    finally { setLoading(false) }
  }, [addToast])

  useEffect(() => { loadFaces() }, [loadFaces])

  const handleCapture = async (label, image_b64) => {
    setShowCapture(false)
    setEnrolling(true)
    try {
      await facesApi.register(label, image_b64)
      addToast('Employee enrolled', label, 'success')
      loadFaces()
    } catch (e) {
      addToast('Enrollment failed', e.response?.data?.detail || 'No face detected — use a clear front-facing photo', 'danger')
    } finally { setEnrolling(false) }
  }

  const handleRenameStart = (face) => {
    setEditingId(face.id)
    setEditLabel(face.label)
  }

  const handleRenameSubmit = async (id) => {
    if (!editLabel.trim()) return
    try {
      await facesApi.rename(id, editLabel.trim())
      setFaces(prev => prev.map(f => f.id === id ? { ...f, label: editLabel.trim() } : f))
      addToast('Name updated', '', 'success')
    } catch { addToast('Rename failed', '', 'danger') }
    finally { setEditingId(null) }
  }

  const handleDelete = async (face) => {
    if (!confirm(`Remove "${face.label}" from the system?`)) return
    setDeleting(face.id)
    try {
      await facesApi.delete(face.id)
      setFaces(prev => prev.filter(f => f.id !== face.id))
      addToast('Employee removed', face.label, 'success')
    } catch { addToast('Delete failed', '', 'danger') }
    finally { setDeleting(null) }
  }

  const displayed = faces.filter(f =>
    !search || f.label.toLowerCase().includes(search.toLowerCase())
  )

  return (
    <div className="page-container">
      <div className="page-header">
        <div>
          <h1 className="page-title"><Users size={22} style={{ marginRight: 10 }} />Employee Roster</h1>
          <p className="page-subtitle">
            {faces.length} registered · {loading ? '…' : faces.length + ' face profiles'}
          </p>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button className="btn btn-ghost btn-icon" onClick={loadFaces} title="Refresh"><RefreshCw size={15} /></button>
          <button className="btn btn-primary" onClick={() => setShowCapture(true)} disabled={enrolling}>
            <Plus size={14} />
            {enrolling ? 'Enrolling…' : 'Enroll Employee'}
          </button>
        </div>
      </div>

      {/* Search */}
      <div style={{ position: 'relative', maxWidth: 340, marginBottom: 24 }}>
        <Search size={14} style={{ position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
        <input className="form-input" style={{ paddingLeft: 32 }} placeholder="Search by name…" value={search} onChange={e => setSearch(e.target.value)} />
      </div>

      {/* Grid */}
      {loading ? (
        <div className="empty-state"><span className="spinner" style={{ width: 36, height: 36 }} /><p>Loading…</p></div>
      ) : displayed.length === 0 ? (
        <div className="empty-state" style={{ marginTop: 60 }}>
          <UserCircle size={52} color="var(--text-muted)" />
          <h3>{faces.length === 0 ? 'No employees enrolled' : 'No results'}</h3>
          <p>Click <strong>Enroll Employee</strong> to register a face.</p>
        </div>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(200px,1fr))', gap: 16 }}>
          {displayed.map(face => (
            <div key={face.id} className="card emp-card">
              {/* avatar / thumbnail */}
              <div className="emp-avatar">
                {face.thumbnail_b64
                  ? <img src={face.thumbnail_b64} alt={face.label} />
                  : <UserCircle size={52} color="var(--text-muted)" />
                }
              </div>

              {/* name */}
              {editingId === face.id ? (
                <div style={{ display: 'flex', gap: 6, marginTop: 8 }}>
                  <input className="form-input form-input-sm" value={editLabel}
                    onChange={e => setEditLabel(e.target.value)}
                    onKeyDown={e => { if (e.key === 'Enter') handleRenameSubmit(face.id); if (e.key === 'Escape') setEditingId(null) }}
                    autoFocus />
                  <button className="btn btn-ghost btn-icon btn-sm" onClick={() => handleRenameSubmit(face.id)}><CheckCircle size={13} /></button>
                </div>
              ) : (
                <div style={{ fontWeight: 600, fontSize: '0.9rem', marginTop: 10, textAlign: 'center' }}>{face.label}</div>
              )}

              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: 4, textAlign: 'center' }}>
                {new Date(face.created_at).toLocaleDateString()}
              </div>

              {/* actions */}
              <div style={{ display: 'flex', gap: 6, marginTop: 12 }}>
                <button className="btn btn-ghost btn-sm" style={{ flex: 1 }} onClick={() => handleRenameStart(face)}>
                  <Pencil size={12} /> Rename
                </button>
                <button className="btn btn-ghost btn-sm" style={{ flex: 1, color: '#f87171' }}
                  onClick={() => handleDelete(face)} disabled={deleting === face.id}>
                  <Trash2 size={12} /> Remove
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {showCapture && <CaptureModal onCapture={handleCapture} onClose={() => setShowCapture(false)} />}
    </div>
  )
}
