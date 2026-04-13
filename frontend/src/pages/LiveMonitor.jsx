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
const FRAME_INTERVAL = 80 // ms -> ~12.5 fps

// Role rule descriptions for the info panel
const ROLE_PPE_RULES = {
  'Doctor':              ['Mask required', 'Gloves required'],
  'Traffic Police':      ['Helmet required'],
  'Construction Worker': ['Hardhat required', 'Safety Vest required', 'Mask required', 'Gloves required', 'Goggles required', 'Safety Shoes required'],
  'College':             ['ID Card required', 'Uniform required'],
  'Home':                ['Face recognition - unknown persons trigger alert'],
}

// All filterable PPE classes
const PPE_FILTERS = [
  { id: 'NO-Hardhat',      label: 'Hardhat',      icon: '👷' },
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

  // Phone status latch — hold alert state for 2.5s to prevent flickering
  const phoneStatusLatchRef = useRef({ status: 'safe', until: 0 })

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

  // —————————————————————————————————————————————————————————————————————————————
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

      // Phone status with 5s hold — prevents rapid flickering
      // Timer resets on EVERY non-safe detection so alert stays while phone is visible.
      if (data.phone_status) {
        const now      = Date.now()
        const latch    = phoneStatusLatchRef.current
        const incoming = data.phone_status
        const HOLD_MS  = 5000
        const priority = { zone_violation: 3, calling: 3, in_hand: 2, safe: 1 }
        const inPri    = priority[incoming] || 1
        const latPri   = priority[latch.status] || 1

        if (incoming !== 'safe') {
          // Phone detected: always upgrade to higher/equal severity and RESET hold timer
          if (inPri >= latPri || now > latch.until) {
            phoneStatusLatchRef.current = { status: incoming, until: now + HOLD_MS }
            setPhoneStatus(incoming)
          } else {
            // Same or lower priority but still detected — just refresh the hold timer
            phoneStatusLatchRef.current = { ...latch, until: now + HOLD_MS }
          }
        } else {
          // Phone gone — only clear after hold period expires
          if (now > latch.until) {
            phoneStatusLatchRef.current = { status: 'safe', until: 0 }
            setPhoneStatus('safe')
          }
        }
      }

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

  // —————————————————————————————————————————————————————————————————————————————
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

  // —————————————————————————————————————————————————————————————————————————————
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

  // —————————————————————————————————————————————————————————————————————————————
  const violationBg = currentAlert
    ? 'rgba(220,38,38,0.12)'
    : 'rgba(16,185,129,0.08)'
  const violationBorder = currentAlert
    ? 'rgba(220,38,38,0.35)'
    : 'rgba(16,185,129,0.25)'

  return (
    <div className="page-container">
      {/* ————————————————————————————————————————————————————————————————————————————— */}
      <div className="page-header">
        <div>
          <h1 className="page-title">Occupational Safety Monitoring</h1>
          <p className="page-subtitle">
            Real-time violation detection · Role: <strong style={{ color: 'var(--accent-blue)' }}>{user?.role || 'Not set'}</strong>
            {modelMode && (
              <span style={{ marginLeft: 10, fontSize: '0.72rem', color: 'var(--text-muted)',
                background: 'rgba(255,255,255,0.06)', padding: '2px 8px', borderRadius: 99 }}>
                {modelMode === 'ppe.pt' ? '🎯 ppe.pt' : modelMode === 'simulation' ? '🔵 Simulation' : `⚙️ ${modelMode}`}
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

      {/* ————————————————————————————————————————————————————————————————————————————— */}
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

        {/* ————————————————————————————————————————————————————————————————————————————— */}
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

      {/* ————————————————————————————————————————————————————————————————————————————— */}
      <div className="tab-bar" style={{ marginBottom: 20, maxWidth: 460 }}>
        {[
          { id: 'webcam', label: 'Webcam' },
          { id: 'rtsp',   label: 'CCTV / RTSP' },
          { id: 'upload', label: 'Upload Video' },
        ].map(m => (
          <button key={m.id}
            className={`tab ${mode === m.id ? 'active' : ''}`}
            onClick={() => { if (streaming) stopStream(); setMode(m.id) }}>
            {m.label}
          </button>
        ))}
      </div>

      <div className="grid-2" style={{ alignItems: 'start' }}>

        {/* -- Left: Video feed -------------------------------- */}
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
                ðŸš¨ {detectionInfo.violationsCount}/{detectionInfo.personsCount} violating
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
                    Uploading... {uploadProgress}%
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

        {/* -- Right: Detection info panel --------------------- */}
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
                      ? `âœ“ All ${detectionInfo.personsCount} person(s) compliant`
                      : `âš  ${detectionInfo.violationsCount}/${detectionInfo.personsCount} person(s) violating`}
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
              Role Rules - {user?.role || 'Not set'}
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
                    const icons = {}
                    return (
                      <span key={f} style={{
                        padding: '3px 10px', borderRadius: 99, fontSize: '0.77rem',
                        background: 'rgba(16,185,129,0.12)', color: 'var(--accent-green)',
                        border: '1px solid rgba(16,185,129,0.25)', fontWeight: 600,
                      }}>
                        {label}
                      </span>
                    )
                  })}
                </div>
              </div>
            ) : user?.role === 'None' ? (
              <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                No custom PPE items selected. Go to{' '}
                <a href="/settings" style={{ color: 'var(--accent-green)' }}>
                  Settings / Safety Rules

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
                               modelMode === 'simulation' ? '#f59e0b' :
                               modelMode?.startsWith('helmet:') ? '#818cf8' : 'var(--accent-blue)',
                  display: 'inline-block', flexShrink: 0
                }} />
                {modelMode === 'ppe.pt'
                  ? 'ppe.pt â€” Custom PPE model (10 classes)'
                  : modelMode === 'simulation'
                  ? 'Simulation mode - Place ppe.pt in backend/'
                  : modelMode === 'helmet:keremberke'
                  ? 'Dedicated Helmet Model (keremberke/yolov8m)'
                  : modelMode === 'helmet:ppe.pt(strict)'
                  ? 'Helmet via ppe.pt - strict logic (downloading...)'
                  : `${modelMode} - COCO fallback`}
              </div>
              {modelMode === 'simulation' && (
                <div style={{ fontSize: '0.72rem', color: 'var(--accent-orange)', marginTop: 6 }}>
                  Download ppe.pt -> biswadeep-roy/Safety-Detection-YOLOv8
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// -- Sub-components -----------------------------------------------

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
            }}>âœ“ {p}</span>
          ))}
        </div>
      )}
    </div>
  )
}

function RoleRules({ role }) {
  const items = ROLE_PPE_RULES[role] || ['No role configured â€” set your role in Profile']
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
  const videoRef   = useRef(null)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration,    setDuration]    = useState(0)
  const [playing,     setPlaying]     = useState(false)
  const [tooltip,     setTooltip]     = useState(null) // {x, vt}
  const [activeVtIdx, setActiveVtIdx] = useState(-1)
  const listRef = useRef(null)

  const SEV_COLORS = {
    critical: { bg: 'rgba(220,38,38,0.15)',  color: '#f87171', border: 'rgba(220,38,38,0.3)',  dot: '#ef4444' },
    high:     { bg: 'rgba(234,88,12,0.12)',  color: '#fb923c', border: 'rgba(234,88,12,0.3)',  dot: '#f97316' },
    medium:   { bg: 'rgba(234,179,8,0.12)',  color: '#fbbf24', border: 'rgba(234,179,8,0.3)',  dot: '#eab308' },
    low:      { bg: 'rgba(16,185,129,0.10)', color: '#34d399', border: 'rgba(16,185,129,0.3)', dot: '#10b981' },
  }

  const isComplete   = status.status === 'complete'
  const isProcessing = status.status === 'processing'
  const isError      = status.status === 'error'
  const ppeSummary   = status.ppe_summary || {}
  const ppeSummaryEntries = Object.entries(ppeSummary)
  const maxViolCount = ppeSummaryEntries.length ? Math.max(...ppeSummaryEntries.map(([,v]) => v)) : 1
  const vts          = status.violation_timestamps || []
  const videoDur     = status.video_duration_sec   || duration || 1
  const videoUrl     = status.annotated_video_url  // e.g. /uploads/annotated_{id}.mp4

  // Sync currentTime while video plays
  useEffect(() => {
    const vid = videoRef.current
    if (!vid) return
    const onTime  = () => {
      const ct = vid.currentTime
      setCurrentTime(ct)
      // Find active violation
      const idx = vts.findIndex((v, i) => {
        const next = vts[i + 1]?.ts ?? Infinity
        return ct >= v.ts && ct < next
      })
      setActiveVtIdx(idx)
    }
    const onLoad  = () => setDuration(vid.duration || videoDur)
    const onPlay  = () => setPlaying(true)
    const onPause = () => setPlaying(false)
    vid.addEventListener('timeupdate', onTime)
    vid.addEventListener('loadedmetadata', onLoad)
    vid.addEventListener('play', onPlay)
    vid.addEventListener('pause', onPause)
    return () => {
      vid.removeEventListener('timeupdate', onTime)
      vid.removeEventListener('loadedmetadata', onLoad)
      vid.removeEventListener('play', onPlay)
      vid.removeEventListener('pause', onPause)
    }
  }, [vts, videoDur])

  // Auto-scroll violation list to active entry
  useEffect(() => {
    if (activeVtIdx < 0 || !listRef.current) return
    const child = listRef.current.children[activeVtIdx]
    if (child) child.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
  }, [activeVtIdx])

  const seekTo = (ts) => {
    const vid = videoRef.current
    if (!vid) return
    vid.currentTime = ts
    vid.play()
  }

  const togglePlay = () => {
    const vid = videoRef.current
    if (!vid) return
    playing ? vid.pause() : vid.play()
  }

  // -- Custom timeline click -> seek ------------------------------
  const handleTimelineClick = (e) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const ratio = (e.clientX - rect.left) / rect.width
    const ts = ratio * (duration || videoDur)
    seekTo(Math.max(0, ts))
  }

  const playheadPct = duration > 0 ? (currentTime / duration) * 100 : 0

  return (
    <div style={{ marginTop: 14 }}>

      {/* -- Header ----------------------------------- */}
      <div className="card card-p" style={{ padding: 16, marginBottom: 12 }}>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 12 }}>
          {isProcessing && <span className="spinner" style={{ width: 16, height: 16, borderWidth: 2 }} />}
          {isComplete   && <CheckCircle size={16} color="var(--accent-green)" />}
          {isError      && <AlertTriangle size={16} color="var(--accent-red)" />}
          <strong style={{ fontSize: '0.88rem' }}>
            {isProcessing
              ? `Scanning for violations... ${status.progress}%`
              : isComplete
              ? `Analysis complete - ${status.total_alerts} alert${status.total_alerts !== 1 ? 's' : ''} · ${status.total_violations ?? 0} violations · ${videoDur}s`
              : 'Processing error'}

          </strong>
        </div>

        {isProcessing && (
          <div className="progress-bar" style={{ marginBottom: 0 }}>
            <div className="progress-fill" style={{ width: `${status.progress}%`, background: 'var(--accent-cyan)' }} />
          </div>
        )}

        {/* Stats row */}
        {isComplete && (
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginTop: 4 }}>
            {[
              { label: 'Frames scanned',   val: status.frames_processed ?? 0 },
              { label: 'Alerts found',     val: status.total_alerts ?? 0 },
              { label: 'Total violations', val: status.total_violations ?? 0 },
              { label: 'Duration',         val: `${videoDur}s` },
            ].map(({ label, val }) => (
              <div key={label} style={{
                flex: 1, minWidth: 80, textAlign: 'center',
                padding: '8px 10px', borderRadius: 8,
                background: 'var(--bg-secondary)', border: '1px solid var(--border)',
              }}>
                <div style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--text-primary)' }}>{val}</div>
                <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', marginTop: 2 }}>{label}</div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* -- Annotated Video Player -------------------- */}
      {isComplete && videoUrl && (
        <div className="card card-p" style={{ padding: 0, overflow: 'hidden', marginBottom: 12 }}>

          {/* Video element */}
          <div style={{ position: 'relative', background: '#000', lineHeight: 0 }}>
            <video
              ref={videoRef}
              src={videoUrl}
              style={{ width: '100%', maxHeight: 340, display: 'block' }}
              preload="metadata"
              onClick={togglePlay}
            />

            {/* Big play icon overlay */}
            {!playing && (
              <div onClick={togglePlay} style={{
                position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center',
                cursor: 'pointer', background: 'rgba(0,0,0,0.3)',
              }}>
                <div style={{
                  width: 56, height: 56, borderRadius: '50%',
                  background: 'rgba(255,255,255,0.18)', backdropFilter: 'blur(4px)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  border: '2px solid rgba(255,255,255,0.4)',
                }}>
                  <span style={{ fontSize: 22, marginLeft: 4, color: '#fff' }}>></span>
                </div>
              </div>
            )}

            {/* Violation count overlay while playing */}
            {playing && activeVtIdx >= 0 && vts[activeVtIdx] && (
              <div style={{
                position: 'absolute', top: 12, left: 12, pointerEvents: 'none',
                background: 'rgba(220,38,38,0.85)', color: '#fff', borderRadius: 8,
                padding: '4px 12px', fontSize: '0.78rem', fontWeight: 700,
                boxShadow: '0 2px 12px rgba(220,38,38,0.5)',
              }}>
                ðŸš¨ {vts[activeVtIdx].items?.join(', ') || 'Violation'}
              </div>
            )}
          </div>

          {/* -- Custom Timeline ------------------------ */}
          <div style={{ padding: '12px 16px 4px', background: 'var(--bg-card)' }}>

            {/* Time scrubber with violation markers */}
            <div
              onClick={handleTimelineClick}
              style={{ position: 'relative', height: 28, cursor: 'pointer', userSelect: 'none' }}
            >
              {/* Track background */}
              <div style={{
                position: 'absolute', top: 10, left: 0, right: 0, height: 8,
                background: 'var(--bg-secondary)', borderRadius: 4, overflow: 'visible',
              }}>
                {/* Green progress fill */}
                <div style={{
                  position: 'absolute', left: 0, top: 0, height: '100%', borderRadius: 4,
                  width: `${playheadPct}%`,
                  background: 'linear-gradient(90deg,var(--accent-cyan),var(--accent-blue))',
                  transition: 'width 0.1s linear',
                }} />

                {/* Red violation marker segments */}
                {vts.map((vt, i) => {
                  const left = (vt.ts / videoDur) * 100
                  const sev  = SEV_COLORS[vt.severity] || SEV_COLORS.medium
                  return (
                    <div
                      key={i}
                      onMouseEnter={(e) => {
                        const rect = e.currentTarget.parentElement.parentElement.getBoundingClientRect()
                        setTooltip({ x: left, vt, idx: i })
                      }}
                      onMouseLeave={() => setTooltip(null)}
                      onClick={(e) => { e.stopPropagation(); seekTo(vt.ts) }}
                      style={{
                        position: 'absolute', top: -3, width: 14, height: 14,
                        left: `${left}%`, transform: 'translateX(-50%)',
                        borderRadius: '50%', border: '2px solid #0a0a0f',
                        background: sev.dot, cursor: 'pointer', zIndex: 3,
                        boxShadow: activeVtIdx === i ? `0 0 8px ${sev.dot}` : 'none',
                        transform: `translateX(-50%) scale(${activeVtIdx === i ? 1.5 : 1})`,
                        transition: 'transform 0.15s, box-shadow 0.15s',
                      }}
                    />
                  )
                })}
              </div>

              {/* Playhead */}
              <div style={{
                position: 'absolute', top: 0, height: 28, width: 3, borderRadius: 2,
                background: '#fff', left: `${playheadPct}%`,
                transform: 'translateX(-50%)', pointerEvents: 'none',
                boxShadow: '0 0 8px rgba(255,255,255,0.6)',
                transition: 'left 0.1s linear',
              }} />

              {/* Tooltip */}
              {tooltip && (
                <div style={{
                  position: 'absolute', bottom: 28, left: `${tooltip.x}%`,
                  transform: 'translateX(-50%)', background: '#1a1a2e',
                  border: `1px solid ${(SEV_COLORS[tooltip.vt.severity] || SEV_COLORS.medium).border}`,
                  borderRadius: 8, padding: '6px 10px', zIndex: 20,
                  minWidth: 120, pointerEvents: 'none',
                  boxShadow: '0 4px 20px rgba(0,0,0,0.6)',
                }}>
                  <div style={{ fontSize: '0.72rem', fontWeight: 700, color: (SEV_COLORS[tooltip.vt.severity]||SEV_COLORS.medium).color, marginBottom: 2 }}>
                    @ {tooltip.vt.ts}s
                  </div>
                  {tooltip.vt.items?.map((item, j) => (
                    <div key={j} style={{ fontSize: '0.70rem', color: 'var(--text-secondary)' }}>! {item}</div>
                  ))}
                  <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', marginTop: 2 }}>
                    Click to seek
                  </div>
                </div>
              )}
            </div>

            {/* Time labels + controls */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 4 }}>
              <span style={{ fontSize: '0.70rem', color: 'var(--text-muted)' }}>
                {currentTime.toFixed(1)}s
              </span>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                <button onClick={togglePlay} style={{
                  background: 'var(--accent-blue)', border: 'none', borderRadius: 6,
                  color: '#fff', cursor: 'pointer', padding: '3px 14px', fontSize: '0.78rem',
                  fontWeight: 600,
                }}>
                  {playing ? 'Pause' : 'Play'}
                </button>
                <a
                  href={videoUrl}
                  download
                  style={{
                    background: 'var(--bg-secondary)', border: '1px solid var(--border)',
                    borderRadius: 6, color: 'var(--text-secondary)', padding: '3px 12px',
                    fontSize: '0.78rem', textDecoration: 'none', fontWeight: 600,
                  }}
                >
                  Download
                </a>
              </div>
              <span style={{ fontSize: '0.70rem', color: 'var(--text-muted)' }}>
                {videoDur.toFixed(1)}s
              </span>
            </div>

            {/* Violation legend */}
            <div style={{ display: 'flex', gap: 14, marginTop: 8, marginBottom: 4, flexWrap: 'wrap' }}>
              {[
                { label: 'Critical', dot: '#ef4444' },
                { label: 'High',     dot: '#f97316' },
                { label: 'Medium',   dot: '#eab308' },
                { label: 'Safe',     color: 'var(--accent-cyan)', isLine: true },
              ].map(({ label, dot, color, isLine }) => (
                <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                  {isLine
                    ? <div style={{ width: 18, height: 4, borderRadius: 2, background: 'var(--accent-cyan)' }} />
                    : <div style={{ width: 10, height: 10, borderRadius: '50%', background: dot, border: '2px solid #0a0a0f' }} />}
                  <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>{label}</span>
                </div>
              ))}
              <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)', marginLeft: 'auto' }}>
                {vts.length} violation marker{vts.length !== 1 ? 's' : ''}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* -- PPE Miss Summary ------------------------- */}
      {isComplete && ppeSummaryEntries.length > 0 && (
        <div className="card card-p" style={{ padding: 16, marginBottom: 12 }}>
          <div style={{ fontSize: '0.72rem', fontWeight: 700, color: 'var(--text-muted)',
            letterSpacing: '0.05em', marginBottom: 10 }}>MOST-MISSED PPE ITEMS</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            {ppeSummaryEntries.map(([item, count]) => (
              <div key={item} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', minWidth: 110, flexShrink: 0 }}>{item}</span>
                <div style={{ flex: 1, height: 6, borderRadius: 3, background: 'var(--bg-secondary)', overflow: 'hidden' }}>
                  <div style={{
                    height: '100%', borderRadius: 3,
                    width: `${Math.round((count / maxViolCount) * 100)}%`,
                    background: 'var(--accent-red)', transition: 'width 0.4s',
                  }} />
                </div>
                <span style={{ fontSize: '0.72rem', fontWeight: 700, color: 'var(--accent-red)', minWidth: 28, textAlign: 'right' }}>{count}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* -- Violation Timeline List ------------------- */}
      {vts.length > 0 && (
        <div className="card card-p" style={{ padding: 16 }}>
          <div style={{ fontSize: '0.72rem', fontWeight: 700, color: 'var(--text-muted)',
            letterSpacing: '0.05em', marginBottom: 8 }}>VIOLATION TIMELINE</div>
          <div ref={listRef} style={{ display: 'flex', flexDirection: 'column', gap: 6,
            maxHeight: 340, overflowY: 'auto', paddingRight: 4 }}>
            {vts.map((vt, i) => {
              const sev    = SEV_COLORS[vt.severity] || SEV_COLORS.medium
              const active = activeVtIdx === i
              return (
                <div
                  key={i}
                  onClick={() => seekTo(vt.ts)}
                  style={{
                    display: 'flex', gap: 10, alignItems: 'flex-start',
                    padding: '8px 10px', borderRadius: 8, cursor: 'pointer',
                    background: active ? sev.bg : 'var(--bg-secondary)',
                    border: `1px solid ${active ? sev.border : 'var(--border)'}`,
                    transition: 'all 0.2s',
                    boxShadow: active ? `0 0 12px ${sev.dot}40` : 'none',
                  }}
                >
                  {vt.thumbnail_b64 && (
                    <img src={vt.thumbnail_b64} alt="violation frame"
                      style={{ width: 52, height: 36, objectFit: 'cover',
                        borderRadius: 4, flexShrink: 0, border: `1px solid ${sev.border}` }} />
                  )}
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 3 }}>
                      <span style={{ fontSize: '0.8rem', fontWeight: 700, color: sev.color }}>@{vt.ts}s</span>
                      {vt.violations > 0 && (
                        <span style={{ fontSize: '0.68rem', fontWeight: 700, color: sev.color,
                          padding: '1px 7px', borderRadius: 99, border: `1px solid ${sev.border}`,
                          background: sev.bg, flexShrink: 0 }}>
                          {vt.violations}/{vt.persons} person{vt.persons !== 1 ? 's' : ''}
                        </span>
                      )}
                      {active && (
                        <span style={{ fontSize: '0.65rem', background: sev.dot, color: '#fff',
                          padding: '1px 6px', borderRadius: 99, fontWeight: 700, flexShrink: 0 }}>
                          NOW
                        </span>
                      )}
                    </div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                      {vt.items?.map((m, j) => (
                        <span key={j} style={{
                          fontSize: '0.68rem', padding: '1px 7px', borderRadius: 99,
                          background: 'rgba(0,0,0,0.25)', color: sev.color, fontWeight: 600,
                        }}>âš  {m}</span>
                      ))}
                    </div>
                  </div>
                  {/* Seek arrow */}
                  <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', flexShrink: 0, alignSelf: 'center' }}>></span>
                </div>
              )
            })}
          </div>
          {isComplete && (
            <div style={{ fontSize: '0.70rem', color: 'var(--text-muted)', textAlign: 'center', marginTop: 8 }}>
              {vts.length} violation moment{vts.length !== 1 ? 's' : ''} â€” click any to jump to that point
            </div>
          )}
        </div>
      )}

      {isError && (
        <div className="card card-p" style={{ fontSize: '0.82rem', color: 'var(--accent-red)',
          padding: '12px 16px', background: 'rgba(220,38,38,0.1)', border: '1px solid rgba(220,38,38,0.3)' }}>
          Processing error â€” try uploading the video again
        </div>
      )}
    </div>
  )
}
