import { useState, useEffect, useRef, useCallback } from 'react'
import { useAuth } from '../context/AuthContext'
import { useToast } from '../context/ToastContext'
import { videoApi } from '../api/api'
import {
  Video, VideoOff, Upload, Wifi,
  AlertTriangle, CheckCircle, ZapOff, Zap,
  User, ShieldCheck, ShieldAlert, HardHat
} from 'lucide-react'

// WebSocket URL: automatically uses same host as the page, port 8000
// Works from PC (localhost) and mobile (network IP) without any config change
function getWsURL() {
  const envURL = import.meta.env.VITE_API_BASE_URL
  if (envURL && envURL !== 'http://localhost:8000') {
    return envURL.replace(/^http/, 'ws') + '/ws/detect'
  }
  const host = window.location.hostname
  const wsHost = (host === 'localhost' || host === '127.0.0.1') ? 'localhost' : host
  return `ws://${wsHost}:8000/ws/detect`
}
const WS_URL = getWsURL()
const FRAME_INTERVAL = 80 // ms → ~12.5 fps

// Role rule descriptions for the info panel
const ROLE_PPE_RULES = {
  'Doctor':              ['😷 Mask required', '🧤 Gloves required'],
  'Traffic Police':      ['⛑️ Hardhat required'],
  'Construction Worker': ['⛑️ Hardhat required', '🦺 Safety Vest required', '😷 Mask required', '🧤 Gloves required', '🥽 Goggles required', '👟 Safety Shoes required'],
  'College':             ['🪪 ID Card required', 'Uniform required'],
  'Home':                ['🔍 Face recognition — unknown persons trigger alert'],
}

// All filterable PPE classes
const PPE_FILTERS = [
  { id: 'NO-Hardhat',      label: 'Hardhat',      icon: '⛑️' },
  { id: 'NO-Safety Vest',  label: 'Safety Vest',  icon: '🦺' },
  { id: 'NO-Mask',         label: 'Mask',         icon: '😷' },
  { id: 'NO-Gloves',       label: 'Gloves',       icon: '🧤' },
  { id: 'NO-Goggles',      label: 'Goggles',      icon: '🥽' },
  { id: 'NO-Safety Shoes', label: 'Safety Shoes', icon: '👟' },
  { id: 'NO-ID Card',      label: 'ID Card',      icon: '🪪' },
  { id: 'NO-Uniform',      label: 'Uniform',      icon: '👕' },
]

// Severity badge colours
const SEV_CLASS = {
  critical: 'badge-critical',
  high:     'badge-high',
  medium:   'badge-medium',
  low:      'badge-low',
}

export default function LiveMonitor() {
  const { user, customPpeItems, noPhoneZone: savedNoPhoneZone } = useAuth()
  const { addToast } = useToast()

  const [mode,           setMode]           = useState('webcam')
  const [rtspUrl,        setRtspUrl]        = useState('')
  const [streaming,      setStreaming]      = useState(false)
  const [connected,      setConnected]      = useState(false)
  const [currentAlert,   setCurrentAlert]   = useState(null)
  const [detectionInfo,  setDetectionInfo]  = useState(null)
  const [frameCount,     setFrameCount]     = useState(0)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [jobId,          setJobId]          = useState(null)
  const [jobStatus,      setJobStatus]      = useState(null)
  const [modelMode,      setModelMode]      = useState('')
  // Phone detection state — default ON so detection works immediately
  const [noPhoneZone,    setNoPhoneZone]    = useState(savedNoPhoneZone !== undefined ? !!savedNoPhoneZone : true)
  const [phoneStatus,    setPhoneStatus]    = useState('safe')
  // Detection filters — for None role: seed from saved custom PPE config
  const defaultFilters = (user?.role === 'None' && customPpeItems?.length)
    ? customPpeItems
    : PPE_FILTERS.map(f => f.id)
  const [activeFilters,  setActiveFilters]  = useState(defaultFilters)
  const [showFilters,    setShowFilters]    = useState(false)

  // Keep noPhoneZone in a ref so WS callbacks always read latest value
  const noPhoneZoneRef = useRef(noPhoneZone)
  useEffect(() => { noPhoneZoneRef.current = noPhoneZone }, [noPhoneZone])

  const videoRef        = useRef(null)
  const canvasRef       = useRef(null)
  const wsRef           = useRef(null)
  const streamRef       = useRef(null)
  const intervalRef     = useRef(null)
  const pollRef         = useRef(null)
  // Ref always holds the LATEST activeFilters — avoids stale closure in ws callbacks
  const activeFiltersRef = useRef(activeFilters)

  const token = localStorage.getItem('token')

  // Keep the ref in sync with state on every render
  useEffect(() => { activeFiltersRef.current = activeFilters }, [activeFilters])

  // ── WebSocket connection ─────────────────────────────────────
  const connectWs = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return
    const ws = new WebSocket(WS_URL)
    wsRef.current = ws

    // Use ref so onopen always reads the LATEST filter selection (fixes stale closure)
    ws.onopen = () => {
      console.log('[WS] Connection opened. Sending handshake with filters:', activeFiltersRef.current)
      ws.send(JSON.stringify({
        token,
        filters: activeFiltersRef.current,
        no_phone_zone: noPhoneZoneRef.current,
      }))
    }

    ws.onmessage = (e) => {
      const data = JSON.parse(e.data)
      // Ignore status-only messages (connected, filters_updated) without frame data
      if (data.status === 'connected') { setConnected(true); return }
      if (data.status === 'filters_updated') return  // ← was erroneously clearing detectionInfo
      if (data.error) { console.warn('[WS] error:', data.error); return }
      // Only process messages that actually have detection results
      if (data.annotated_frame === undefined && data.is_compliant === undefined) return

      setFrameCount(f => f + 1)
      if (data.model_mode) setModelMode(data.model_mode)
      if (data.phone_status) setPhoneStatus(data.phone_status)

      setDetectionInfo({
        isCompliant:      data.is_compliant,
        missing:          data.missing_items || [],
        detections:       data.detections    || [],
        persons:          data.persons       || [],
        violationsCount:  data.violations_count ?? 0,
        personsCount:     data.persons_count    ?? 0,
        phoneDetected:    data.phone_detected   ?? false,
        phoneStatus:      data.phone_status     || 'safe',
      })

      // Draw annotated frame
      if (data.annotated_frame && canvasRef.current) {
        const img = new Image()
        img.onload = () => {
          const ctx = canvasRef.current?.getContext('2d')
          if (ctx) ctx.drawImage(img, 0, 0, canvasRef.current.width, canvasRef.current.height)
        }
        img.src = data.annotated_frame
      }

      if (!data.is_compliant && data.alert_message) {
        setCurrentAlert({ message: data.alert_message, severity: data.severity })
        if (data.alert_saved) {
          addToast('⚠️ Alert Saved!', data.alert_message, 'danger', 5000)
        }
      } else {
        setCurrentAlert(null)
      }
    }
    ws.onclose = () => setConnected(false)
    ws.onerror = () => setConnected(false)
  }, [token, addToast])  // refs are stable — no need to add activeFiltersRef here

  // ── Webcam ───────────────────────────────────────────────────
  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }
      connectWs()
      setStreaming(true)

      intervalRef.current = setInterval(() => {
        if (!videoRef.current || !canvasRef.current || !wsRef.current) return
        if (wsRef.current.readyState !== WebSocket.OPEN) return
        const vw = videoRef.current.videoWidth  || 640
        const vh = videoRef.current.videoHeight || 480
        canvasRef.current.width  = vw
        canvasRef.current.height = vh
        const ctx = canvasRef.current.getContext('2d')
        ctx.drawImage(videoRef.current, 0, 0)
        const b64 = canvasRef.current.toDataURL('image/jpeg', 0.65)
        const filtersNow = activeFiltersRef.current
        wsRef.current.send(JSON.stringify({
          frame: b64,
          filters: filtersNow,
          no_phone_zone: noPhoneZoneRef.current,
        }))
      }, FRAME_INTERVAL)
    } catch (err) {
      addToast('Camera error', err.message || 'Could not access webcam', 'danger')
    }
  }

  const stopStream = useCallback(() => {
    clearInterval(intervalRef.current)
    clearInterval(pollRef.current)
    streamRef.current?.getTracks().forEach(t => t.stop())
    wsRef.current?.close()
    setStreaming(false)
    setConnected(false)
    setCurrentAlert(null)
    setDetectionInfo(null)
    setFrameCount(0)
    if (canvasRef.current) {
      canvasRef.current.getContext('2d')?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)
    }
  }, [])

  useEffect(() => () => stopStream(), [stopStream])

  // activeFiltersRef keeps the ref in sync — no extra WebSocket message needed
  // Filters are embedded in every frame payload above for zero-latency enforcement

  // ── Upload ───────────────────────────────────────────────────
  const handleUpload = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    setUploadProgress(0); setJobStatus(null)
    try {
      const res = await videoApi.upload(file, setUploadProgress)
      const id  = res.data.job_id
      setJobId(id)
      setJobStatus({ status: 'processing', progress: 0 })
      addToast('Video uploaded', 'Violation scanning started…', 'info')

      pollRef.current = setInterval(async () => {
        const s = await videoApi.status(id)
        setJobStatus(s.data)
        if (s.data.status === 'complete' || s.data.status === 'error') {
          clearInterval(pollRef.current)
          if (s.data.status === 'complete') {
            addToast(
              '✅ Video processed',
              `Found ${s.data.total_alerts} alerts · ${s.data.total_violations ?? 0} total violations`,
              'success'
            )
          }
        }
      }, 1500)
    } catch (err) {
      addToast('Upload failed', err.response?.data?.detail || 'Try again', 'danger')
    }
  }

  // ── Violation badge colours ──────────────────────────────────
  const violationBg = currentAlert
    ? 'rgba(220,38,38,0.12)'
    : 'rgba(16,185,129,0.08)'
  const violationBorder = currentAlert
    ? 'rgba(220,38,38,0.35)'
    : 'rgba(16,185,129,0.25)'

  return (
    <div className="page-container">
      {/* ── Header ─────────────────────────────────────────── */}
      <div className="page-header">
        <div>
          <h1 className="page-title">Occupational Safety Monitoring</h1>
          <p className="page-subtitle">
            Real-time violation detection · Role: <strong style={{ color: 'var(--accent-blue)' }}>{user?.role || 'Not set'}</strong>
            {modelMode && (
              <span style={{ marginLeft: 10, fontSize: '0.72rem', color: 'var(--text-muted)',
                background: 'rgba(255,255,255,0.06)', padding: '2px 8px', borderRadius: 99 }}>
                {modelMode === 'ppe.pt' ? '🎯 ppe.pt' : modelMode === 'simulation' ? '🔵 Simulation' : `⚙ ${modelMode}`}
              </span>
            )}
          </p>
        </div>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
          {connected && (
            <div className="video-overlay-badge" style={{
              position: 'static',
              background: 'rgba(16,185,129,0.15)',
              border: '1px solid rgba(16,185,129,0.3)'
            }}>
              <span className="pulse-dot pulse-dot-green" />
              <span style={{ color: 'var(--accent-green)', fontSize: '0.78rem', fontWeight: 600 }}>LIVE</span>
            </div>
          )}
        </div>
      </div>

      {/* ── Detection Filter Panel ─────────────────────────── */}
      <div style={{
        marginBottom: 16, padding: '10px 14px', borderRadius: 10,
        background: 'var(--bg-secondary)', border: '1px solid var(--border)',
      }}>
        <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', gap:12 }}>
          <div style={{ display:'flex', alignItems:'center', gap:8 }}>
            <span style={{ fontSize:'0.78rem', fontWeight:700, color:'var(--text-muted)', letterSpacing:'0.05em' }}>
              DETECTION FILTERS
            </span>
            <span style={{ fontSize:'0.72rem', color:'var(--text-muted)', background:'var(--bg-card)',
              padding:'1px 7px', borderRadius:99, border:'1px solid var(--border)' }}>
              {activeFilters.length}/{PPE_FILTERS.length} active
            </span>
          </div>
          <div style={{ display:'flex', gap:6 }}>
            <button className="btn btn-ghost btn-sm" style={{ padding:'3px 10px', fontSize:'0.72rem' }}
              onClick={() => setActiveFilters(PPE_FILTERS.map(f => f.id))}>All</button>
            <button className="btn btn-ghost btn-sm" style={{ padding:'3px 10px', fontSize:'0.72rem' }}
              onClick={() => setActiveFilters([])}>None</button>
            <button className="btn btn-ghost btn-sm" style={{ padding:'3px 10px', fontSize:'0.72rem' }}
              onClick={() => setShowFilters(s => !s)}>
              {showFilters ? '▲ Hide' : '▼ Edit'}
            </button>
          </div>
        </div>

        {/* Expanded toggle grid */}
        {showFilters && (
          <div style={{ display:'flex', flexWrap:'wrap', gap:8, marginTop:12 }}>
            {PPE_FILTERS.map(f => {
              const on = activeFilters.includes(f.id)
              return (
                <button key={f.id}
                  onClick={() => setActiveFilters(prev =>
                    on ? prev.filter(x => x !== f.id) : [...prev, f.id]
                  )}
                  style={{
                    display:'flex', alignItems:'center', gap:6,
                    padding:'7px 14px', borderRadius:99, cursor:'pointer',
                    fontSize:'0.82rem', fontWeight:600,
                    border:`1px solid ${on ? 'rgba(249,115,22,0.5)' : 'var(--border)'}`,
                    background: on ? 'rgba(249,115,22,0.12)' : 'var(--bg-card)',
                    color: on ? '#fb923c' : 'var(--text-muted)',
                    transition:'all 0.15s',
                  }}>
                  <span style={{ fontSize:'1rem' }}>{f.icon}</span>
                  {f.label}
                  {on
                    ? <span style={{ width:7, height:7, borderRadius:'50%', background:'#fb923c', flexShrink:0 }} />
                    : <span style={{ width:7, height:7, borderRadius:'50%', background:'var(--border)', flexShrink:0 }} />}
                </button>
              )
            })}
          </div>
        )}

        {/* Collapsed chip strip */}
        {!showFilters && (
          <div style={{ display:'flex', flexWrap:'wrap', gap:5, marginTop:8 }}>
            {PPE_FILTERS.map(f => {
              const on = activeFilters.includes(f.id)
              return (
                <button key={f.id}
                  onClick={() => setActiveFilters(prev =>
                    on ? prev.filter(x => x !== f.id) : [...prev, f.id]
                  )}
                  title={on ? `Disable ${f.label} detection` : `Enable ${f.label} detection`}
                  style={{
                    fontSize:'0.7rem', padding:'3px 9px', borderRadius:99, cursor:'pointer',
                    border:`1px solid ${on ? 'rgba(249,115,22,0.35)' : 'var(--border)'}`,
                    background: on ? 'rgba(249,115,22,0.08)' : 'transparent',
                    color: on ? '#fb923c' : 'var(--text-muted)',
                    transition:'all 0.12s',
                  }}>
                  {f.icon} {f.label}
                </button>
              )
            })}
          </div>
        )}

        {/* ── Phone Zone Row ─────────────────────────────────── */}
        <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between',
          borderTop:'1px solid var(--border)', marginTop:10, paddingTop:10 }}>
          <div style={{ display:'flex', alignItems:'center', gap:10 }}>
            <span style={{ fontSize:'1rem' }}>📱</span>
            <div>
              <div style={{ fontSize:'0.82rem', fontWeight:700, color:'var(--text-primary)' }}>
                No Phone Zone
              </div>
              <div style={{ fontSize:'0.7rem', color:'var(--text-muted)' }}>
                Any phone detected triggers alert
              </div>
            </div>
          </div>
          <div style={{ display:'flex', alignItems:'center', gap:10 }}>
            {/* Phone status badge — always visible */}
            <span style={{
              fontSize:'0.72rem', fontWeight:600, padding:'3px 10px', borderRadius:99,
              background: phoneStatus === 'safe'
                ? 'rgba(16,185,129,0.12)'
                : phoneStatus === 'in_hand'
                ? 'rgba(234,179,8,0.12)'
                : 'rgba(239,68,68,0.12)',
              color: phoneStatus === 'safe'
                ? '#10b981'
                : phoneStatus === 'in_hand'
                ? '#eab308'
                : '#ef4444',
              border: `1px solid ${phoneStatus === 'safe'
                ? 'rgba(16,185,129,0.3)'
                : phoneStatus === 'in_hand'
                ? 'rgba(234,179,8,0.3)'
                : 'rgba(239,68,68,0.3)'}`,
              transition: 'all 0.3s',
            }}>
              {phoneStatus === 'safe'        ? '🟢 No Phone'
               : phoneStatus === 'in_hand'  ? '🟡 Phone in Hand'
               : phoneStatus === 'calling'  ? '🔴 Calling Alert'
               : '🔴 Zone Violation'}
            </span>
            {/* Toggle button */}
            <button
              onClick={async () => {
                const next = !noPhoneZone
                setNoPhoneZone(next)
                try {
                  const { usersApi } = await import('../api/api')
                  await usersApi.updateConfig({ no_phone_zone: next })
                } catch { /* best-effort save */ }
              }}
              style={{
                width:44, height:24, borderRadius:12, cursor:'pointer',
                border:'none', padding:0, transition:'background 0.2s',
                background: noPhoneZone ? '#ef4444' : 'rgba(255,255,255,0.12)',
                position:'relative', flexShrink:0,
              }}
            >
              <span style={{
                position:'absolute', top:3, width:18, height:18, borderRadius:'50%',
                background:'#fff', transition:'left 0.2s',
                left: noPhoneZone ? 23 : 3,
              }} />
            </button>
          </div>
        </div>
      </div>

      {/* ── Mode Tabs ──────────────────────────────────────── */}
      <div className="tab-bar" style={{ marginBottom: 20, maxWidth: 460 }}>
        {[
          { id: 'webcam', label: '📷 Webcam' },
          { id: 'rtsp',   label: '📡 CCTV/RTSP' },
          { id: 'upload', label: '🎬 Upload Video' },
        ].map(m => (
          <button key={m.id}
            className={`tab ${mode === m.id ? 'active' : ''}`}
            onClick={() => { if (streaming) stopStream(); setMode(m.id) }}>
            {m.label}
          </button>
        ))}
      </div>

      <div className="grid-2" style={{ alignItems: 'start' }}>

        {/* ── Left: Video feed ──────────────────────────────── */}
        <div>
          <div className="video-container" style={{ minHeight: 300 }}>
            <video ref={videoRef} style={{ display: 'none' }} muted />
            <canvas ref={canvasRef} className="video-canvas"
              style={{ display: streaming ? 'block' : 'none' }} />

            {!streaming && mode !== 'upload' && (
              <div className="no-feed">
                <VideoOff size={40} style={{ opacity: 0.3 }} />
                <div>No feed active</div>
                <div style={{ fontSize: '0.78rem' }}>
                  {mode === 'webcam' ? 'Click Start Detection to begin' : 'Enter RTSP URL and click Start'}
                </div>
              </div>
            )}

            {/* Alert overlay */}
            {streaming && currentAlert && (
              <div className="alert-overlay">
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#fff', fontWeight: 700 }}>
                  <AlertTriangle size={18} />
                  {currentAlert.message}
                </div>
              </div>
            )}

            {/* Compliant badge */}
            {streaming && !currentAlert && detectionInfo && (
              <div className="video-overlay-badge">
                <CheckCircle size={12} color="var(--accent-green)" />
                <span style={{ color: 'var(--accent-green)', fontSize: '0.72rem' }}>All Compliant</span>
              </div>
            )}

            {/* Violation count badge */}
            {streaming && detectionInfo && detectionInfo.violationsCount > 0 && (
              <div style={{
                position: 'absolute', top: 52, left: 12,
                background: 'rgba(220,38,38,0.85)', color: '#fff',
                padding: '3px 10px', borderRadius: 99,
                fontSize: '0.72rem', fontWeight: 700,
                boxShadow: '0 2px 8px rgba(220,38,38,0.4)'
              }}>
                🚨 {detectionInfo.violationsCount}/{detectionInfo.personsCount} violating
              </div>
            )}

            {/* Frame counter */}
            {streaming && (
              <div style={{
                position: 'absolute', top: 12, right: 12,
                background: 'rgba(0,0,0,0.6)', padding: '4px 10px',
                borderRadius: 99, fontSize: '0.72rem', color: 'var(--text-secondary)'
              }}>
                {frameCount} frames
              </div>
            )}
          </div>

          {/* RTSP input */}
          {mode === 'rtsp' && (
            <div className="form-group" style={{ marginTop: 12 }}>
              <label className="form-label">RTSP / Stream URL</label>
              <input className="form-input" placeholder="rtsp://192.168.1.1:554/stream"
                value={rtspUrl} onChange={e => setRtspUrl(e.target.value)} />
            </div>
          )}

          {/* Upload input */}
          {mode === 'upload' && (
            <div>
              <label className="btn btn-ghost" style={{ cursor: 'pointer', display: 'inline-flex', marginTop: 12 }}>
                <Upload size={15} /> Choose Video File
                <input type="file" accept="video/*" style={{ display: 'none' }} onChange={handleUpload} />
              </label>
              {uploadProgress > 0 && uploadProgress < 100 && (
                <div style={{ marginTop: 12 }}>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: 6 }}>
                    Uploading… {uploadProgress}%
                  </div>
                  <div className="progress-bar">
                    <div className="progress-fill" style={{ width: `${uploadProgress}%`, background: 'var(--accent-blue)' }} />
                  </div>
                </div>
              )}
              {jobStatus && <VideoJobPanel status={jobStatus} />}
            </div>
          )}

          {/* Controls */}
          {mode !== 'upload' && (
            <div className="video-controls">
              {!streaming ? (
                <button id="btn-start-detection" className="btn btn-success" onClick={startWebcam}>
                  <Zap size={15} /> Start Detection
                </button>
              ) : (
                <button id="btn-stop-detection" className="btn btn-danger" onClick={stopStream}>
                  <ZapOff size={15} /> Stop
                </button>
              )}
            </div>
          )}
        </div>

        {/* ── Right: Detection info panel ───────────────────── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

          {/* Status card */}
          <div className="card card-p">
            <h3 style={{ fontSize: '0.95rem', marginBottom: 14 }}>Detection Status</h3>

            {!detectionInfo ? (
              <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                Start monitoring to see live detections
              </div>
            ) : (
              <>
                {/* Overall compliance pill */}
                <div style={{
                  display: 'flex', alignItems: 'center', gap: 10,
                  padding: '10px 14px', borderRadius: 10, marginBottom: 14,
                  background: violationBg, border: `1px solid ${violationBorder}`
                }}>
                  {detectionInfo.isCompliant
                    ? <CheckCircle size={20} color="var(--accent-green)" />
                    : <AlertTriangle size={20} color="var(--accent-red)" />}
                  <strong style={{ color: detectionInfo.isCompliant ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                    {detectionInfo.isCompliant
                      ? `✓ All ${detectionInfo.personsCount} person(s) compliant`
                      : `⚠ ${detectionInfo.violationsCount}/${detectionInfo.personsCount} person(s) violating`}
                  </strong>
                </div>

                {/* Missing items badges */}
                {detectionInfo.missing.length > 0 && (
                  <div style={{ marginBottom: 14 }}>
                    <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginBottom: 6, letterSpacing: '0.05em' }}>
                      MISSING PPE
                    </div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                      {detectionInfo.missing.map(m => (
                        <span key={m} className="badge badge-critical">{m}</span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Per-person violation cards */}
                {detectionInfo.persons.length > 0 && (
                  <div>
                    <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginBottom: 8, letterSpacing: '0.05em' }}>
                      PER-PERSON STATUS
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                      {detectionInfo.persons.map((p, i) => (
                        <PersonCard key={i} person={p} index={i} />
                      ))}
                    </div>
                  </div>
                )}
              </>
            )}
          </div>

          {/* Role rules card */}
          <div className="card card-p">
            <h3 style={{ fontSize: '0.95rem', marginBottom: 12 }}>
              <HardHat size={15} style={{ marginRight: 6, verticalAlign: 'middle' }} />
              Role Rules — {user?.role || 'Not set'}
            </h3>
            {user?.role === 'None' && activeFilters.length > 0 ? (
              <div>
                <div style={{
                  padding: '8px 12px', borderRadius: 8, marginBottom: 10,
                  background: 'rgba(16,185,129,0.08)',
                  border: '1px solid rgba(16,185,129,0.25)',
                  fontSize: '0.78rem', color: 'var(--accent-green)',
                  display: 'flex', alignItems: 'center', gap: 6,
                }}>
                  <ShieldCheck size={13} /> Filters loaded from your Custom Safety Settings
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                  {activeFilters.map(f => {
                    const label = f.replace('NO-', '')
                    const icons = { Hardhat:'⛑️', Gloves:'🧤', Goggles:'🥽', Mask:'😷', 'Safety Vest':'🦺', 'Safety Shoes':'👟', 'ID Card':'🪪', Uniform:'👕' }
                    return (
                      <span key={f} style={{
                        padding: '3px 10px', borderRadius: 99, fontSize: '0.77rem',
                        background: 'rgba(16,185,129,0.12)', color: 'var(--accent-green)',
                        border: '1px solid rgba(16,185,129,0.25)', fontWeight: 600,
                      }}>
                        {icons[label] || '🛡️'} {label}
                      </span>
                    )
                  })}
                </div>
              </div>
            ) : user?.role === 'None' ? (
              <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                No custom PPE items selected. Go to{' '}
                <a href="/settings" style={{ color: 'var(--accent-green)' }}>
                  Settings → Safety Rules
                </a>{' '}
                to configure.
              </div>
            ) : (
              <RoleRules role={user?.role} />
            )}
          </div>

          {/* Model info card */}
          {modelMode && (
            <div className="card card-p" style={{ padding: '12px 16px' }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: 6 }}>MODEL</div>
              <div style={{ fontSize: '0.85rem', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{
                  width: 8, height: 8, borderRadius: '50%',
                  background: modelMode === 'ppe.pt' ? 'var(--accent-green)' :
                               modelMode === 'simulation' ? '#f59e0b' : 'var(--accent-blue)',
                  display: 'inline-block', flexShrink: 0
                }} />
                {modelMode === 'ppe.pt'
                  ? 'ppe.pt — Custom PPE model (10 classes)'
                  : modelMode === 'simulation'
                  ? 'Simulation mode — Place ppe.pt in backend/'
                  : `${modelMode} — COCO fallback`}
              </div>
              {modelMode === 'simulation' && (
                <div style={{ fontSize: '0.72rem', color: 'var(--accent-orange)', marginTop: 6 }}>
                  Download ppe.pt → biswadeep-roy/Safety-Detection-YOLOv8
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// ── Sub-components ───────────────────────────────────────────────

function PersonCard({ person, index }) {
  const isViolator = !person.is_compliant
  return (
    <div style={{
      padding: '8px 12px', borderRadius: 8,
      background: isViolator ? 'rgba(220,38,38,0.08)' : 'rgba(16,185,129,0.07)',
      border: `1px solid ${isViolator ? 'rgba(220,38,38,0.25)' : 'rgba(16,185,129,0.2)'}`,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: isViolator ? 6 : 0 }}>
        <User size={14} color={isViolator ? 'var(--accent-red)' : 'var(--accent-green)'} />
        <span style={{ fontSize: '0.82rem', fontWeight: 600,
          color: isViolator ? 'var(--accent-red)' : 'var(--accent-green)' }}>
          Person {index + 1}
        </span>
        <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginLeft: 'auto' }}>
          {(person.confidence * 100).toFixed(0)}% conf
        </span>
      </div>
      {isViolator && person.violation_labels?.length > 0 && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
          {person.violation_labels.map((v, j) => (
            <span key={j} style={{
              fontSize: '0.70rem', padding: '2px 8px', borderRadius: 99,
              background: 'rgba(220,38,38,0.2)', color: '#f87171', fontWeight: 600
            }}>{v}</span>
          ))}
        </div>
      )}
      {!isViolator && (
        <div style={{ fontSize: '0.75rem', color: 'var(--accent-green)', display: 'flex', gap: 4, flexWrap: 'wrap' }}>
          {person.ppe_found?.map((p, j) => (
            <span key={j} style={{
              padding: '1px 7px', borderRadius: 99,
              background: 'rgba(16,185,129,0.15)', color: '#34d399', fontSize: '0.70rem'
            }}>✓ {p}</span>
          ))}
        </div>
      )}
    </div>
  )
}

function RoleRules({ role }) {
  const items = ROLE_PPE_RULES[role] || ['No role configured — set your role in Profile']
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      {items.map((item, i) => (
        <div key={i} style={{
          fontSize: '0.83rem', color: 'var(--text-secondary)',
          padding: '6px 0', borderBottom: '1px solid var(--border)',
          display: 'flex', alignItems: 'center', gap: 8
        }}>
          <ShieldCheck size={13} color="var(--accent-blue)" style={{ flexShrink: 0 }} />
          {item}
        </div>
      ))}
    </div>
  )
}

function VideoJobPanel({ status }) {
  return (
    <div className="card card-p" style={{ marginTop: 14, padding: 16 }}>
      <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 10 }}>
        {status.status === 'processing' && <span className="spinner" style={{ width: 16, height: 16, borderWidth: 2 }} />}
        {status.status === 'complete'   && <CheckCircle size={16} color="var(--accent-green)" />}
        <strong style={{ fontSize: '0.85rem' }}>
          {status.status === 'processing'
            ? `Scanning violations… ${status.progress}%`
            : status.status === 'complete'
            ? `Done — ${status.total_alerts} alerts, ${status.total_violations ?? 0} violations`
            : 'Error during processing'}
        </strong>
      </div>

      {status.status === 'processing' && (
        <div className="progress-bar" style={{ marginBottom: 10 }}>
          <div className="progress-fill" style={{ width: `${status.progress}%`, background: 'var(--accent-cyan)' }} />
        </div>
      )}

      {status.alerts?.slice(0, 6).map((a, i) => (
        <div key={i} style={{
          fontSize: '0.78rem', padding: '6px 0',
          borderTop: '1px solid var(--border)',
          color: 'var(--text-secondary)', marginTop: 4,
          display: 'flex', justifyContent: 'space-between', gap: 8
        }}>
          <span style={{ color: 'var(--accent-orange)' }}>@{a.timestamp_sec}s</span>
          <span style={{ flex: 1 }}>{a.message}</span>
          {a.violations_count > 0 && (
            <span style={{
              background: 'rgba(220,38,38,0.15)', color: '#f87171',
              padding: '1px 7px', borderRadius: 99, fontSize: '0.70rem', fontWeight: 600, flexShrink: 0
            }}>
              {a.violations_count} violating
            </span>
          )}
        </div>
      ))}
    </div>
  )
}
