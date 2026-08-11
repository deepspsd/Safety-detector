/* eslint-disable no-unused-vars */
import { useState, useEffect, useRef, useCallback } from 'react'
import { useParams, useNavigate, Link } from 'react-router-dom'
import { camerasApi, alertsApi } from '../api/api'
import { useToast } from '../context/ToastContext'
import {
  Camera, AlertTriangle, CheckCircle, WifiOff, RefreshCw,
  Maximize2, ChevronLeft, Activity, Clock, Layers
} from 'lucide-react'

const FLOORS = [
  { id: 'ground', label: 'Ground Floor', icon: '🏭', color: '#3b82f6' },
  { id: 'first',  label: 'First Floor',  icon: '🏗️', color: '#8b5cf6' },
  { id: 'second', label: 'Second Floor', icon: '🏢', color: '#06b6d4' },
  { id: 'shop',   label: 'Shop Floor',   icon: '🛒', color: '#f59e0b' },
]

const STATUS_COLORS = {
  online:  { bg: 'rgba(16,185,129,0.15)',  border: 'rgba(16,185,129,0.4)',  text: '#34d399' },
  offline: { bg: 'rgba(100,116,139,0.15)', border: 'rgba(100,116,139,0.4)', text: '#94a3b8' },
  error:   { bg: 'rgba(239,68,68,0.15)',   border: 'rgba(239,68,68,0.4)',   text: '#f87171' },
}

const POLL_MS = 2500

/** Single camera tile that polls /cameras/{id}/snapshot every POLL_MS */
function CameraTile({ camera, onSelect }) {
  const [imgSrc,      setImgSrc]      = useState(null)
  const [lastUpdated, setLastUpdated] = useState(null)
  const [pollErr,     setPollErr]     = useState(false)
  const isMounted = useRef(true)
  const BASE = import.meta.env.VITE_API_BASE_URL?.replace(/\/api$/, '') || ''

  useEffect(() => {
    isMounted.current = true
    let tid

    const fetchSnap = async () => {
      if (!camera.id || camera.status === 'offline') return
      try {
        // snapshot returns JSON { frame_b64: "data:image/jpeg;base64,..." }
        const token = localStorage.getItem('token')
        const url = `${BASE}/api/cameras/${camera.id}/snapshot?t=${Date.now()}`
        const res = await fetch(url, { headers: token ? { Authorization: `Bearer ${token}` } : {} })
        if (!res.ok) throw new Error(res.status)
        const data = await res.json()
        if (!isMounted.current) return
        if (data.frame_b64) {
          setImgSrc(prev => { if (prev && prev.startsWith('blob:')) URL.revokeObjectURL(prev); return data.frame_b64 })
          setLastUpdated(new Date())
          setPollErr(false)
        } else {
          setPollErr(true)
        }
      } catch {
        if (isMounted.current) setPollErr(true)
      } finally {
        if (isMounted.current) tid = setTimeout(fetchSnap, POLL_MS)
      }
    }

    fetchSnap()
    return () => { isMounted.current = false; clearTimeout(tid); if (imgSrc && imgSrc.startsWith('blob:')) URL.revokeObjectURL(imgSrc) }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [camera.id, camera.status])

  const sc = STATUS_COLORS[camera.status] || STATUS_COLORS.offline

  return (
    <div
      className="camera-tile"
      onClick={() => onSelect(camera)}
      title={`Open ${camera.name} in Live Monitor`}
    >
      {/* snapshot / placeholder */}
      <div className="camera-tile-img-wrap">
        {imgSrc ? (
          <img src={imgSrc} alt={camera.name} className="camera-tile-img" />
        ) : (
          <div className="camera-tile-no-feed">
            {pollErr
              ? <WifiOff size={28} color="#f87171" />
              : camera.status === 'offline'
                ? <WifiOff size={28} color="#64748b" />
                : <span className="spinner" style={{ width: 28, height: 28 }} />
            }
            <span style={{ marginTop: 8, fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              {pollErr ? 'No signal' : camera.status === 'offline' ? 'Offline' : 'Connecting…'}
            </span>
          </div>
        )}

        {/* overlay: status + expand */}
        <div className="camera-tile-overlay">
          <span className="camera-tile-status" style={{ background: sc.bg, color: sc.text, border: `1px solid ${sc.border}` }}>
            <span className="status-dot" style={{ background: sc.text }} />
            {camera.status}
          </span>
          <Maximize2 size={13} style={{ color: 'rgba(255,255,255,0.7)' }} />
        </div>
      </div>

      {/* footer */}
      <div className="camera-tile-footer">
        <Camera size={13} color="var(--text-muted)" />
        <span className="camera-tile-name">{camera.name}</span>
        {lastUpdated && (
          <span className="camera-tile-time">
            <Clock size={10} />
            {lastUpdated.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
          </span>
        )}
      </div>
    </div>
  )
}

export default function FloorOverview() {
  const { floorId } = useParams()
  const navigate    = useNavigate()
  const { addToast } = useToast()

  const [cameras,  setCameras]  = useState([])
  const [,         setStats]    = useState(null)
  const [loading,  setLoading]  = useState(true)

  const floor = FLOORS.find(f => f.id === floorId) || FLOORS[0]

  const load = useCallback(async () => {
    try {
      const [camRes, statRes] = await Promise.all([
        camerasApi.list(),
        alertsApi.stats().catch(() => null),
      ])
      const all = Array.isArray(camRes.data) ? camRes.data : (camRes.data.cameras || [])
      setCameras(all.filter(c => c.floor === floorId))
      setStats(statRes?.data || null)
    } catch { addToast('Failed to load cameras', '', 'danger') }
    finally { setLoading(false) }
  }, [floorId, addToast])

  useEffect(() => { setLoading(true); load() }, [load])

  const handleSelect = (cam) => {
    // Navigate to live monitor with camera pre-selected via query param
    navigate(`/monitor?camera=${cam.id}`)
  }

  const onlineCams  = cameras.filter(c => c.status === 'online').length
  const totalCams   = cameras.length

  return (
    <div className="page-container">
      {/* Breadcrumb */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 20 }}>
        <Link to="/floors/ground" style={{ color: 'var(--text-muted)', textDecoration: 'none', fontSize: '0.82rem' }}>
          <Layers size={13} style={{ marginRight: 4 }} />Floors
        </Link>
        <span style={{ color: 'var(--text-muted)' }}>/</span>
        <span style={{ color: 'var(--text-primary)', fontSize: '0.82rem', fontWeight: 600 }}>
          {floor.icon} {floor.label}
        </span>
      </div>

      {/* Floor tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 24, flexWrap: 'wrap' }}>
        {FLOORS.map(f => (
          <Link
            key={f.id}
            to={`/floors/${f.id}`}
            style={{
              display: 'flex', alignItems: 'center', gap: 6,
              padding: '7px 16px', borderRadius: 'var(--radius-md)',
              border: `1px solid ${f.id === floorId ? f.color : 'var(--border)'}`,
              background: f.id === floorId ? `${f.color}18` : 'var(--bg-card)',
              color: f.id === floorId ? f.color : 'var(--text-secondary)',
              textDecoration: 'none', fontWeight: 600, fontSize: '0.82rem',
              transition: 'all var(--transition)',
            }}
          >
            <span>{f.icon}</span>{f.label}
          </Link>
        ))}
      </div>

      {/* Stats bar */}
      <div style={{ display: 'flex', gap: 12, marginBottom: 24, flexWrap: 'wrap' }}>
        {[
          { label: 'Total Cameras', value: totalCams, icon: Camera, color: 'var(--accent-blue)' },
          { label: 'Online',        value: onlineCams, icon: Activity, color: 'var(--accent-green)' },
          { label: 'Offline',       value: totalCams - onlineCams, icon: WifiOff, color: 'var(--text-muted)' },
        ].map(({ label, value, icon: Icon, color }) => (
          <div key={label} style={{
            display: 'flex', alignItems: 'center', gap: 10,
            padding: '10px 18px', borderRadius: 'var(--radius-md)',
            background: 'var(--bg-card)', border: '1px solid var(--border)',
          }}>
            <Icon size={16} color={color} />
            <div>
              <div style={{ fontSize: '1.1rem', fontWeight: 700, color }}>{value}</div>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>{label}</div>
            </div>
          </div>
        ))}
      </div>

      {/* Camera grid */}
      {loading ? (
        <div className="empty-state" style={{ marginTop: 80 }}>
          <span className="spinner" style={{ width: 36, height: 36 }} />
          <p>Loading cameras…</p>
        </div>
      ) : cameras.length === 0 ? (
        <div className="empty-state" style={{ marginTop: 80 }}>
          <Camera size={48} color="var(--text-muted)" />
          <h3>No cameras on this floor</h3>
          <p>Add cameras via <Link to="/settings/cameras" style={{ color: 'var(--accent-blue)' }}>Settings → Cameras</Link></p>
        </div>
      ) : (
        <div className="camera-grid">
          {cameras.map(cam => (
            <CameraTile key={cam.id} camera={cam} onSelect={handleSelect} />
          ))}
        </div>
      )}

      {/* Refresh hint */}
      <div style={{ marginTop: 24, display: 'flex', alignItems: 'center', gap: 6, color: 'var(--text-muted)', fontSize: '0.72rem' }}>
        <RefreshCw size={11} />
        Snapshots auto-refresh every {POLL_MS / 1000}s
      </div>
    </div>
  )
}
