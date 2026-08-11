/* eslint-disable react-hooks/set-state-in-effect */
import { useState, useEffect, useRef, useCallback } from 'react'
import { Link } from 'react-router-dom'
import { camerasApi, alertsApi } from '../api/api'
import {
  Shield, Camera, AlertTriangle, WifiOff, Activity, Clock, Layers
} from 'lucide-react'

const FLOORS = [
  { id: 'ground', label: 'Ground',  icon: '🏭' },
  { id: 'first',  label: 'First',   icon: '🏗️' },
  { id: 'second', label: 'Second',  icon: '🏢' },
  { id: 'shop',   label: 'Shop',    icon: '🛒' },
]

const SNAP_POLL_MS  = 3000
const ALERT_POLL_MS = 20000

const BASE = import.meta.env.VITE_API_BASE_URL?.replace(/\/api$/, '') || ''

/** Single camera tile (lightweight — no auth header for TV mode) */
function TVCameraTile({ camera }) {
  const [src,    setSrc]    = useState(null)
  const [err,    setErr]    = useState(false)
  const mounted = useRef(true)

  useEffect(() => {
    mounted.current = true
    let tid

    const poll = async () => {
      if (camera.status === 'offline') { setErr(true); return }
      try {
        const token = localStorage.getItem('token')
        const url   = `${BASE}/api/cameras/${camera.id}/snapshot?t=${Date.now()}`
        const res   = await fetch(url, { headers: token ? { Authorization: `Bearer ${token}` } : {} })
        if (!res.ok) throw new Error()
        const data  = await res.json()
        if (!mounted.current) return
        if (data.frame_b64) { setSrc(data.frame_b64); setErr(false) }
        else setErr(true)
      } catch { if (mounted.current) setErr(true) }
      finally  { if (mounted.current) tid = setTimeout(poll, SNAP_POLL_MS) }
    }

    poll()
    return () => { mounted.current = false; clearTimeout(tid) }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [camera.id])

  return (
    <div className="tv-tile">
      <div className="tv-tile-img">
        {src && !err
          ? <img src={src} alt={camera.name} />
          : (
            <div className="tv-tile-no-feed">
              {camera.status === 'offline'
                ? <WifiOff size={22} color="#475569" />
                : err
                  ? <WifiOff size={22} color="#f87171" />
                  : <span className="spinner" style={{ width: 22, height: 22, borderColor: 'rgba(255,255,255,0.2)', borderTopColor: '#3b82f6' }} />
              }
            </div>
          )
        }
      </div>
      <div className="tv-tile-label">
        <span className={`tv-dot ${camera.status === 'online' ? 'tv-dot-online' : camera.status === 'error' ? 'tv-dot-error' : 'tv-dot-offline'}`} />
        {camera.name}
      </div>
    </div>
  )
}

/** Scrolling violation ticker */
function ViolationTicker({ alerts }) {
  if (!alerts.length) return null
  const doubled = [...alerts, ...alerts]   // seamless scroll loop

  return (
    <div className="tv-ticker-wrap">
      <div className="tv-ticker-label"><AlertTriangle size={12} /> VIOLATIONS</div>
      <div className="tv-ticker-track">
        <div className="tv-ticker-inner">
          {doubled.map((a, i) => (
            <span key={i} className="tv-ticker-item">
              <span className={`tv-sev-dot sev-${a.severity}`} />
              {a.detected_issue || a.message}
              {a.floor && <span className="tv-ticker-floor"> · {a.floor}</span>}
            </span>
          ))}
        </div>
      </div>
    </div>
  )
}

export default function TVDisplay() {
  const [activeFloor, setActiveFloor] = useState('ground')
  const [allCameras,  setAllCameras]  = useState([])
  const [alerts,      setAlerts]      = useState([])
  const [now,         setNow]         = useState(new Date())

  const cameras = allCameras.filter(c => c.floor === activeFloor)

  const loadCameras = useCallback(async () => {
    try {
      const res = await camerasApi.list()
      const list = Array.isArray(res.data) ? res.data : (res.data.cameras || [])
      setAllCameras(list)
    } catch { /* silent */ }
  }, [])

  const loadAlerts = useCallback(async () => {
    try {
      const res = await alertsApi.list({ limit: 20, status: 'confirmed' })
      setAlerts(res.data.alerts || [])
    } catch { /* silent */ }
  }, [])

  useEffect(() => {
    loadCameras()
    loadAlerts()
    const cam = setInterval(loadCameras, 30000)
    const ale = setInterval(loadAlerts, ALERT_POLL_MS)
    const clk = setInterval(() => setNow(new Date()), 1000)
    return () => { clearInterval(cam); clearInterval(ale); clearInterval(clk) }
  }, [loadCameras, loadAlerts])

  const online = allCameras.filter(c => c.status === 'online').length

  return (
    <div className="tv-display">
      {/* Header bar */}
      <header className="tv-header">
        <div className="tv-header-logo">
          <Shield size={20} color="#f97316" />
          <span>OccuSafe</span>
          <span className="tv-badge">LIVE</span>
        </div>

        <div className="tv-floor-tabs">
          {FLOORS.map(f => (
            <button
              key={f.id}
              className={`tv-floor-tab ${f.id === activeFloor ? 'active' : ''}`}
              onClick={() => setActiveFloor(f.id)}
            >
              {f.icon} {f.label}
            </button>
          ))}
        </div>

        <div className="tv-header-right">
          <div className="tv-stat">
            <Activity size={13} color="#34d399" />
            <span style={{ color: '#34d399', fontWeight: 700 }}>{online}</span>
            <span style={{ color: '#64748b' }}>/{allCameras.length} online</span>
          </div>
          <div className="tv-clock">
            <Clock size={13} color="#64748b" />
            {now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
          </div>
        </div>
      </header>

      {/* Camera grid */}
      <main className="tv-main">
        {cameras.length === 0 ? (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12,
            color: '#475569', padding: 80 }}>
            <Camera size={40} />
            <span>No cameras on this floor</span>
          </div>
        ) : (
          <div className="tv-camera-grid" style={{
            gridTemplateColumns: cameras.length <= 2 ? `repeat(${cameras.length},1fr)`
              : cameras.length <= 4 ? 'repeat(2,1fr)'
              : cameras.length <= 6 ? 'repeat(3,1fr)'
              : 'repeat(4,1fr)'
          }}>
            {cameras.map(cam => <TVCameraTile key={cam.id} camera={cam} />)}
          </div>
        )}
      </main>

      {/* Violation ticker */}
      <ViolationTicker alerts={alerts} />
    </div>
  )
}
