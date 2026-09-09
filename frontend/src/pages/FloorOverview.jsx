/* eslint-disable no-unused-vars */
import { useState, useEffect, useRef, useCallback } from 'react'
import { useParams, useNavigate, Link } from 'react-router-dom'
import { camerasApi, alertsApi } from '../api/api'
import api from '../api/api'
import { useToast } from '../context/ToastContext'
import {
  Camera, AlertTriangle, CheckCircle, WifiOff, RefreshCw,
  Maximize2, ChevronLeft, Activity, Clock, Layers
} from 'lucide-react'

const FLOORS = [
  { id: 'ground', label: 'Ground Floor', icon: '🏭', color: '#3b82f6', workStart: '8:00 AM', zones: 'Entrance · Dough · Oven · Packing' },
  { id: 'first',  label: 'First Floor',  icon: '🏗️', color: '#8b5cf6', workStart: '6:00 AM', zones: 'Work Tables · Dough · Lift' },
  { id: 'second', label: 'Second Floor', icon: '🏢', color: '#06b6d4', workStart: '5:00 AM', zones: 'Cooking · Ovens · Stock' },
  { id: 'shop',   label: 'Shop',         icon: '🛒', color: '#f59e0b', workStart: '9:00 AM', zones: 'Counter · Cashbox · Entrance' },
]

const STATUS_COLORS = {
  online:  { bg: 'rgba(16,185,129,0.15)',  border: 'rgba(16,185,129,0.4)',  text: '#34d399' },
  offline: { bg: 'rgba(100,116,139,0.15)', border: 'rgba(100,116,139,0.4)', text: '#94a3b8' },
  error:   { bg: 'rgba(239,68,68,0.15)',   border: 'rgba(239,68,68,0.4)',   text: '#f87171' },
}

import CameraFullscreenModal from '../components/CameraFullscreenModal'

const POLL_MS = 3000

/** Single camera tile that renders live MJPEG stream or snapshot fallback */
function CameraTile({ camera, onSelect }) {
  const [streamError, setStreamError] = useState(false)
  const BASE = import.meta.env.VITE_API_BASE_URL?.replace(/\/api$/, '') || ''
  const streamUrl = `${BASE}/api/cameras/${camera.id}/annotated-stream`

  const sc = STATUS_COLORS[camera.status] || STATUS_COLORS.offline

  return (
    <div
      className="camera-tile"
      onClick={() => onSelect(camera)}
      title={`Inspect ${camera.name}`}
      style={{ cursor: 'pointer' }}
    >
      {/* stream / placeholder */}
      <div className="camera-tile-img-wrap" style={{ position: 'relative', background: '#090d16' }}>
        {camera.status === 'online' && !streamError ? (
          <img
            src={streamUrl}
            alt={camera.name}
            className="camera-tile-img"
            onError={() => setStreamError(true)}
            style={{ width: '100%', height: '100%', objectFit: 'cover' }}
          />
        ) : (
          <div className="camera-tile-no-feed">
            <WifiOff size={28} color={camera.status === 'offline' ? '#64748b' : '#f87171'} />
            <span style={{ marginTop: 8, fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              {camera.status === 'offline' ? 'Offline' : 'Connecting feed…'}
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
        <div style={{ flex: 1, minWidth: 0 }}>
          <div className="camera-tile-name">{camera.name}</div>
          {camera.zone_type && (
            <div style={{ fontSize: '0.6rem', color: 'var(--text-muted)', marginTop: 2, textTransform: 'capitalize', letterSpacing: '0.03em' }}>
              {camera.zone_type.replace(/_/g, ' ')}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default function FloorOverview() {
  const { floorId } = useParams()
  const navigate    = useNavigate()
  const { addToast } = useToast()

  const [cameras,       setCameras]      = useState([])
  const [,              setStats]        = useState(null)
  const [loading,       setLoading]      = useState(true)
  const [floorAlerts,   setFloorAlerts]  = useState({})
  const [selectedCam,   setSelectedCam]  = useState(null)

  const floor = FLOORS.find(f => f.id === floorId) || FLOORS[0]

  const load = useCallback(async () => {
    try {
      const [camRes, statRes, sumRes] = await Promise.all([
        camerasApi.list(),
        alertsApi.stats().catch(() => null),
        api.get('/workflow/summary').catch(() => null),
      ])
      const all = Array.isArray(camRes.data) ? camRes.data : (camRes.data.cameras || [])
      setCameras(all.filter(c => (c.floor || '').toLowerCase() === (floorId || '').toLowerCase()))
      setStats(statRes?.data || null)
      if (sumRes?.data?.floor_counts) setFloorAlerts(sumRes.data.floor_counts)
    } catch { addToast('Failed to load cameras', '', 'danger') }
    finally { setLoading(false) }
  }, [floorId, addToast])

  useEffect(() => { setLoading(true); load() }, [load])

  const handleSelect = (cam) => {
    setSelectedCam(cam)
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

      {/* Floor tabs — with alert count badges */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 24, flexWrap: 'wrap' }}>
        {FLOORS.map(f => {
          const ac = floorAlerts[f.id] ?? 0
          return (
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
              {ac > 0 && (
                <span style={{
                  marginLeft: 2, fontSize: '0.65rem', fontWeight: 800,
                  background: f.color, color: '#fff',
                  padding: '1px 6px', borderRadius: 99,
                }}>{ac}</span>
              )}
            </Link>
          )
        })}
      </div>

      {/* Stats bar — cameras + work-start time + today's alerts */}
      <div style={{ display: 'flex', gap: 12, marginBottom: 24, flexWrap: 'wrap', alignItems: 'center' }}>
        {[
          { label: 'Total Cameras', value: totalCams,              icon: Camera,    color: 'var(--accent-blue)' },
          { label: 'Online',        value: onlineCams,             icon: Activity,  color: 'var(--accent-green)' },
          { label: 'Offline',       value: totalCams - onlineCams, icon: WifiOff,   color: 'var(--text-muted)' },
          { label: 'Alerts Today',  value: floorAlerts[floorId] ?? 0, icon: AlertTriangle, color: (floorAlerts[floorId] ?? 0) > 0 ? '#ef4444' : 'var(--text-muted)' },
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
        {/* Work-start time badge */}
        <div style={{
          marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 6,
          padding: '8px 14px', borderRadius: 'var(--radius-md)',
          background: `${floor.color}12`, border: `1px solid ${floor.color}40`,
        }}>
          <Clock size={14} color={floor.color} />
          <div>
            <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Work Starts</div>
            <div style={{ fontSize: '0.9rem', fontWeight: 700, color: floor.color }}>{floor.workStart}</div>
          </div>
        </div>
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
        Live feeds auto-streaming via on-premise inference pipeline
      </div>

      {/* In-place fullscreen modal */}
      {selectedCam && (
        <CameraFullscreenModal
          camera={selectedCam}
          onClose={() => setSelectedCam(null)}
        />
      )}
    </div>
  )
}
