import { useState, useEffect, useCallback } from 'react'
import { Link } from 'react-router-dom'
import { camerasApi } from '../api/api'
import { useToast } from '../context/ToastContext'
import {
  Camera, Plus, Pencil, Trash2, RefreshCw, Wifi, WifiOff,
  ChevronRight, Zap, Search, X, CheckCircle, AlertCircle,
  MapPin, ScanLine, Save, RotateCcw
} from 'lucide-react'

const FLOORS = ['ground', 'first', 'second', 'shop']
const FLOOR_LABELS = { ground: 'Ground Floor', first: 'First Floor', second: 'Second Floor', shop: 'Shop Floor' }
const ZONE_TYPES = ['general', 'restricted', 'bakery', 'cashbox', 'entrance', 'corridor', '']

const STATUS_CFG = {
  online:  { icon: Wifi,     color: '#34d399', bg: 'rgba(16,185,129,0.12)'  },
  offline: { icon: WifiOff,  color: '#94a3b8', bg: 'rgba(100,116,139,0.12)' },
  error:   { icon: AlertCircle, color: '#f87171', bg: 'rgba(239,68,68,0.12)' },
}

const EMPTY_FORM = {
  name: '', floor: 'ground', rtsp_url: '', zone_type: '',
  camera_code: '', department: '', purpose: '', status: 'offline',
  ai_enabled: true,
}

/** Inline drawer for add / edit */
function CameraDrawer({ camera, onClose, onSaved }) {
  const { addToast } = useToast()
  const [form,    setForm]    = useState(camera ? { ...camera } : { ...EMPTY_FORM })
  const [saving,  setSaving]  = useState(false)
  const [testing, setTesting] = useState(false)
  const isEdit = Boolean(camera?.id)

  const set = (k, v) => setForm(f => ({ ...f, [k]: v }))

  const handleSave = async () => {
    if (!form.name.trim()) { addToast('Name is required', '', 'warning'); return }
    if (!form.floor)       { addToast('Floor is required', '', 'warning'); return }
    setSaving(true)
    try {
      if (isEdit) {
        await camerasApi.update(camera.id, form)
        addToast('Camera updated', form.name, 'success')
      } else {
        await camerasApi.create(form)
        addToast('Camera added', form.name, 'success')
      }
      onSaved()
      onClose()
    } catch (e) {
      addToast(isEdit ? 'Update failed' : 'Create failed', e.response?.data?.detail || '', 'danger')
    } finally { setSaving(false) }
  }

  const handleRestart = async () => {
    if (!isEdit) return
    setTesting(true)
    try { await camerasApi.restart(camera.id); addToast('Camera restarted', '', 'success') }
    catch { addToast('Restart failed', '', 'danger') }
    finally { setTesting(false) }
  }

  return (
    <div className="drawer-overlay" onClick={onClose}>
      <div className="drawer-panel" onClick={e => e.stopPropagation()}>
        <div className="drawer-header">
          <h3>{isEdit ? `Edit — ${camera.name}` : 'Add Camera'}</h3>
          <button className="btn btn-ghost btn-icon" onClick={onClose}><X size={18} /></button>
        </div>

        <div className="drawer-body">
          <label className="form-label">Camera Name *</label>
          <input className="form-input" value={form.name} onChange={e => set('name', e.target.value)} placeholder="e.g. Ground-Entrance-01" />

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
            <div>
              <label className="form-label">Floor *</label>
              <select className="form-select" value={form.floor} onChange={e => set('floor', e.target.value)}>
                {FLOORS.map(f => <option key={f} value={f}>{FLOOR_LABELS[f]}</option>)}
              </select>
            </div>
            <div>
              <label className="form-label">Zone Type</label>
              <select className="form-select" value={form.zone_type || ''} onChange={e => set('zone_type', e.target.value)}>
                <option value="">— none —</option>
                {ZONE_TYPES.filter(Boolean).map(z => <option key={z} value={z}>{z}</option>)}
              </select>
            </div>
          </div>

          <label className="form-label">RTSP / Stream URL</label>
          <input className="form-input" value={form.rtsp_url || ''} onChange={e => set('rtsp_url', e.target.value)}
            placeholder="rtsp://user:pass@192.168.1.100:554/stream" />

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
            <div>
              <label className="form-label">Camera Code</label>
              <input className="form-input" value={form.camera_code || ''} onChange={e => set('camera_code', e.target.value)} placeholder="CAM-001" />
            </div>
            <div>
              <label className="form-label">Department</label>
              <input className="form-input" value={form.department || ''} onChange={e => set('department', e.target.value)} placeholder="Bakery / Security" />
            </div>
          </div>

          <label className="form-label">Purpose / Notes</label>
          <input className="form-input" value={form.purpose || ''} onChange={e => set('purpose', e.target.value)} placeholder="Entrance monitoring, dough table…" />

          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginTop: 8 }}>
            <label className="form-label" style={{ marginBottom: 0 }}>AI Detection</label>
            <button
              type="button"
              className={`toggle-pill ${form.ai_enabled ? 'on' : ''}`}
              onClick={() => set('ai_enabled', !form.ai_enabled)}
            >
              <span className="toggle-thumb" />
            </button>
            <span style={{ fontSize: '0.8rem', color: form.ai_enabled ? 'var(--accent-green)' : 'var(--text-muted)' }}>
              {form.ai_enabled ? 'Enabled' : 'Disabled'}
            </span>
          </div>
        </div>

        <div className="drawer-footer">
          {isEdit && (
            <button className="btn btn-ghost" onClick={handleRestart} disabled={testing}>
              <RotateCcw size={14} />
              {testing ? 'Restarting…' : 'Restart'}
            </button>
          )}
          {isEdit && (
            <Link to="/settings" style={{ textDecoration: 'none' }}>
              <button className="btn btn-ghost">
                <ScanLine size={14} /> Zone Calibration
              </button>
            </Link>
          )}
          <button className="btn btn-primary" onClick={handleSave} disabled={saving} style={{ marginLeft: 'auto' }}>
            <Save size={14} />
            {saving ? 'Saving…' : isEdit ? 'Save Changes' : 'Add Camera'}
          </button>
        </div>
      </div>
    </div>
  )
}

export default function Cameras() {
  const { addToast } = useToast()
  const [cameras,    setCameras]    = useState([])
  const [loading,    setLoading]    = useState(true)
  const [search,     setSearch]     = useState('')
  const [floorFilter,setFloorFilter]= useState('')
  const [drawer,     setDrawer]     = useState(null)  // null | 'new' | camera-obj
  const [deleting,   setDeleting]   = useState(null)
  const [discovering,setDiscovering]= useState(false)

  const loadCameras = useCallback(async () => {
    try {
      const res = await camerasApi.list()
      const list = Array.isArray(res.data) ? res.data : (res.data.cameras || [])
      setCameras(list)
    } catch { addToast('Failed to load cameras', '', 'danger') }
    finally { setLoading(false) }
  }, [addToast])

  useEffect(() => { loadCameras() }, [loadCameras])

  const handleDelete = async (cam) => {
    if (!confirm(`Delete "${cam.name}"? This will stop its reader thread.`)) return
    setDeleting(cam.id)
    try {
      await camerasApi.delete(cam.id)
      setCameras(prev => prev.filter(c => c.id !== cam.id))
      addToast('Camera removed', cam.name, 'success')
    } catch { addToast('Delete failed', '', 'danger') }
    finally { setDeleting(null) }
  }

  const handleDiscover = async () => {
    setDiscovering(true)
    try {
      const res = await camerasApi.discover()
      const found = res.data?.found || 0
      addToast('Discovery complete', `${found} camera(s) found on LAN`, found > 0 ? 'success' : 'info')
      if (found > 0) loadCameras()
    } catch { addToast('Discovery failed', 'Scan requires network access', 'danger') }
    finally { setDiscovering(false) }
  }

  const displayed = cameras.filter(c => {
    const q = search.toLowerCase()
    const matchQ = !q || c.name?.toLowerCase().includes(q) || c.rtsp_url?.toLowerCase().includes(q) || c.camera_code?.toLowerCase().includes(q)
    const matchF = !floorFilter || c.floor === floorFilter
    return matchQ && matchF
  })

  const grouped = FLOORS.reduce((acc, f) => {
    const list = displayed.filter(c => c.floor === f)
    if (list.length) acc[f] = list
    return acc
  }, {})

  return (
    <div className="page-container">
      {/* Header */}
      <div className="page-header">
        <div>
          <h1 className="page-title"><Camera size={22} style={{ marginRight: 10 }} />Camera Management</h1>
          <p className="page-subtitle">{cameras.length} cameras · {cameras.filter(c => c.status === 'online').length} online</p>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button className="btn btn-ghost" onClick={handleDiscover} disabled={discovering}>
            <ScanLine size={14} />
            {discovering ? 'Scanning…' : 'Discover LAN'}
          </button>
          <button className="btn btn-primary" onClick={() => setDrawer('new')}>
            <Plus size={14} /> Add Camera
          </button>
        </div>
      </div>

      {/* Filters */}
      <div style={{ display: 'flex', gap: 10, marginBottom: 20, flexWrap: 'wrap' }}>
        <div style={{ position: 'relative', flex: 1, minWidth: 200 }}>
          <Search size={14} style={{ position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
          <input className="form-input" style={{ paddingLeft: 32 }} placeholder="Search by name, URL, code…" value={search} onChange={e => setSearch(e.target.value)} />
        </div>
        <select className="form-select" style={{ width: 160 }} value={floorFilter} onChange={e => setFloorFilter(e.target.value)}>
          <option value="">All Floors</option>
          {FLOORS.map(f => <option key={f} value={f}>{FLOOR_LABELS[f]}</option>)}
        </select>
        <button className="btn btn-ghost btn-icon" onClick={loadCameras} title="Refresh"><RefreshCw size={15} /></button>
      </div>

      {/* Table */}
      {loading ? (
        <div className="empty-state"><span className="spinner" style={{ width: 36, height: 36 }} /><p>Loading cameras…</p></div>
      ) : cameras.length === 0 ? (
        <div className="empty-state" style={{ marginTop: 60 }}>
          <Camera size={48} color="var(--text-muted)" />
          <h3>No cameras configured</h3>
          <p>Click <strong>Add Camera</strong> or <strong>Discover LAN</strong> to get started.</p>
        </div>
      ) : (
        Object.entries(grouped).map(([floor, cams]) => (
          <div key={floor} style={{ marginBottom: 28 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
              <MapPin size={14} color="var(--text-muted)" />
              <span style={{ fontSize: '0.78rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-muted)' }}>
                {FLOOR_LABELS[floor]}
              </span>
              <span style={{ fontSize: '0.72rem', background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 99, padding: '1px 8px', color: 'var(--text-secondary)' }}>{cams.length}</span>
            </div>
            <div className="card" style={{ overflow: 'hidden', padding: 0 }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ background: 'rgba(255,255,255,0.02)' }}>
                    {['Name', 'Status', 'Zone', 'RTSP URL', 'AI', 'Actions'].map(h => (
                      <th key={h} style={{ padding: '10px 14px', textAlign: 'left', fontSize: '0.72rem', fontWeight: 700,
                        textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)',
                        borderBottom: '1px solid var(--border)' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {cams.map((cam, i) => {
                    const sc = STATUS_CFG[cam.status] || STATUS_CFG.offline
                    const StatusIcon = sc.icon
                    return (
                      <tr key={cam.id} style={{
                        borderBottom: i < cams.length - 1 ? '1px solid var(--border)' : 'none',
                        transition: 'background var(--transition)',
                      }}
                        onMouseEnter={e => e.currentTarget.style.background = 'var(--bg-card-hover)'}
                        onMouseLeave={e => e.currentTarget.style.background = ''}
                      >
                        <td style={{ padding: '12px 14px' }}>
                          <div style={{ fontWeight: 600, fontSize: '0.88rem' }}>{cam.name}</div>
                          {cam.camera_code && <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>{cam.camera_code}</div>}
                        </td>
                        <td style={{ padding: '12px 14px' }}>
                          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5, padding: '3px 9px',
                            borderRadius: 99, background: sc.bg, color: sc.color, fontSize: '0.73rem', fontWeight: 600 }}>
                            <StatusIcon size={11} /> {cam.status}
                          </span>
                        </td>
                        <td style={{ padding: '12px 14px', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                          {cam.zone_type || <span style={{ color: 'var(--text-muted)' }}>—</span>}
                        </td>
                        <td style={{ padding: '12px 14px', maxWidth: 220 }}>
                          {cam.rtsp_url
                            ? <span style={{ fontSize: '0.75rem', fontFamily: 'JetBrains Mono, monospace',
                                color: 'var(--text-muted)', wordBreak: 'break-all' }}>{cam.rtsp_url}</span>
                            : <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>—</span>
                          }
                        </td>
                        <td style={{ padding: '12px 14px' }}>
                          {cam.ai_enabled !== false
                            ? <CheckCircle size={15} color="#34d399" />
                            : <X size={15} color="#f87171" />
                          }
                        </td>
                        <td style={{ padding: '12px 14px' }}>
                          <div style={{ display: 'flex', gap: 4 }}>
                            <button className="btn btn-ghost btn-icon btn-sm" onClick={() => setDrawer(cam)} title="Edit">
                              <Pencil size={13} />
                            </button>
                            <button className="btn btn-ghost btn-icon btn-sm" style={{ color: '#f87171' }}
                              onClick={() => handleDelete(cam)}
                              disabled={deleting === cam.id}
                              title="Delete">
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
        ))
      )}

      {/* Drawer */}
      {drawer && (
        <CameraDrawer
          camera={drawer === 'new' ? null : drawer}
          onClose={() => setDrawer(null)}
          onSaved={loadCameras}
        />
      )}
    </div>
  )
}
