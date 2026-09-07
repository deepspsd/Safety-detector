import { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import { useAuth } from '../context/AuthContext'
import { useToast } from '../context/ToastContext'
import { videoApi } from '../api/api'
import {
  Video, VideoOff, Upload, Wifi,
  AlertTriangle, CheckCircle, ZapOff, Zap,
  User, ShieldCheck, ShieldAlert, HardHat
} from 'lucide-react'

// WebSocket URLs use the Vite proxy in development. In production, normalize
// an API base ending in /api back to its origin because WS routes live at /ws.
function getWsBaseURL() {
  const envURL = import.meta.env.VITE_API_BASE_URL
  if (envURL && !envURL.includes('localhost') && !envURL.includes('127.0.0.1')) {
    return envURL
      .replace(/^http/, 'ws')
      .replace(/\/api\/?$/, '')
      .replace(/\/$/, '')
  }
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
  return `${proto}://${window.location.host}`
}

const WS_BASE_URL = getWsBaseURL()
const WS_URL = `${WS_BASE_URL}/ws/detect`
const CCTV_WS_URL = `${WS_BASE_URL}/ws/detect-cctv`
const FRAME_INTERVAL = 80 // ms -> ~12.5 fps

// Role rule descriptions for the info panel
const ROLE_PPE_RULES = {
  'Bakery Worker': ['Head Cap required', 'Uniform required', 'Face Mask required', 'Gloves required', 'No Bangles (food safety)'],
  'Factory Worker': ['Head Cap required', 'Uniform required', 'No Bangles (food safety)'],
  'Doctor': ['Mask required', 'Gloves required'],
  'Traffic Police': ['Helmet required'],
  'Construction Worker': ['Hardhat required', 'Safety Vest required', 'Mask required', 'Gloves required', 'Goggles required', 'Safety Shoes required'],
  'College': ['ID Card required', 'Uniform required'],
  'Home': ['Face recognition - unknown persons trigger alert'],
}

// All filterable PPE classes (used for manual toggle UI)
const PPE_FILTERS = [
  { id: 'NO-Bakery-Head-Cap', label: 'Head Cap', icon: '🧢' },
  { id: 'NO-Uniform', label: 'Uniform', icon: '👕' },
  { id: 'Bangles', label: 'Bangles (ban)', icon: '🚨' },
  { id: 'Cash', label: 'Cash', icon: '💵' },
  { id: 'Cylinder', label: 'Cylinder', icon: '🛢️' },
  { id: 'NO-Hardhat', label: 'Hardhat', icon: '👷' },
  { id: 'NO-Safety Vest', label: 'Safety Vest', icon: '🦺' },
  { id: 'NO-Mask', label: 'Mask', icon: '😷' },
  { id: 'NO-Gloves', label: 'Gloves', icon: '🧤' },
  { id: 'NO-Goggles', label: 'Goggles', icon: '🥽' },
  { id: 'NO-Safety Shoes', label: 'Safety Shoes', icon: '👟' },
  { id: 'NO-ID Card', label: 'ID Card', icon: '🪪' },
]

// Role → default filter set sent at WebSocket handshake.
// MUST match the violation class names the backend model actually emits.
const ROLE_FILTERS = {
  'Bakery Worker': ['NO-Bakery-Head-Cap', 'NO-Uniform', 'NO-Gloves', 'Bangles', 'NO-Mask'],
  'Construction Worker': ['NO-Hardhat', 'NO-Safety Vest', 'NO-Mask', 'NO-Gloves', 'NO-Goggles', 'NO-Safety Shoes'],
  'Doctor': ['NO-Mask', 'NO-Gloves'],
  'Traffic Police': ['NO-Hardhat'],
  'College': ['NO-ID Card', 'NO-Uniform'],
  'Home': [],
  'None': [],   // seeded from saved custom PPE items
  'Factory Worker': ['NO-Bakery-Head-Cap', 'NO-Uniform', 'Bangles'],
}

// Severity badge colours
const SEV_CLASS = {
  critical: 'badge-critical',
  high: 'badge-high',
  medium: 'badge-medium',
  low: 'badge-low',
}

export default function LiveMonitor() {
  const { user, customPpeItems, noPhoneZone: savedNoPhoneZone } = useAuth()
  const { addToast } = useToast()

  const [mode, setMode] = useState('webcam')
  const [rtspUrl, setRtspUrl] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [connected, setConnected] = useState(false)
  const [currentAlert, setCurrentAlert] = useState(null)
  const [detectionInfo, setDetectionInfo] = useState(null)
  const [frameCount, setFrameCount] = useState(0)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [jobStatus, setJobStatus] = useState(null)
  const [modelMode, setModelMode] = useState('')
  // CCTV extras
  const [camFps, setCamFps] = useState(0)
  const [faceResult, setFaceResult] = useState(null)
  const [enableFace, setEnableFace] = useState(true)  // face recognition toggle
  // Phone detection state — default ON so detection works immediately
  const [noPhoneZone, setNoPhoneZone] = useState(savedNoPhoneZone !== undefined ? !!savedNoPhoneZone : true)
  const [phoneStatus, setPhoneStatus] = useState('safe')
  // Cash monitoring state
  const [cashAlert, setCashAlert] = useState(null)  // theft alert message | null
  const [payeeData, setPayeeData] = useState(null)  // { snapshot, id, time }
  // Active zone state — resolved from camera DB or sent by server
  const [activeZone, setActiveZone] = useState('default')
  // IMPORTANT: Do NOT fall back to PPE_FILTERS.map(f=>f.id) for known roles —
  // that was sending 8 construction filters to Bakery Worker users.
  const getFiltersForRole = (role) => {
    if (!role) return []           // wait for auth
    if (role === 'None') {
      if (customPpeItems?.length) return customPpeItems
      return []                                            // custom — user sets them
    }
    return ROLE_FILTERS[role] ?? []    // unknown role → empty (no false positives)
  }

  const [activeFilters, setActiveFilters] = useState(() => getFiltersForRole(user?.role))
  const [showFilters, setShowFilters] = useState(false)
  // keep enableFace in a ref so WS callbacks always read latest value
  const enableFaceRef = useRef(enableFace)
  useEffect(() => { enableFaceRef.current = enableFace }, [enableFace])

  // Keep noPhoneZone in a ref so WS callbacks always read latest value
  const noPhoneZoneRef = useRef(noPhoneZone)
  useEffect(() => { noPhoneZoneRef.current = noPhoneZone }, [noPhoneZone])

  // Phone status latch — hold alert state for 2.5s to prevent flickering
  const phoneStatusLatchRef = useRef({ status: 'safe', until: 0 })

  const videoRef = useRef(null)   // live webcam element (always plays)
  const canvasRef = useRef(null)   // display-only: shows annotated frames from WS
  const captureRef = useRef(null)   // hidden: captures raw frames to send to WS
  const cctvImgRef = useRef(null)   // <img> showing raw MJPEG CCTV stream
  const wsRef = useRef(null)
  const streamRef = useRef(null)
  const intervalRef = useRef(null)
  const pollRef = useRef(null)
  const hasAnnotatedRef = useRef(false)  // true once first annotated frame received
  // Ref always holds the LATEST activeFilters — avoids stale closure in ws callbacks
  const activeFiltersRef = useRef(activeFilters)

  const token = localStorage.getItem('token')

  // Keep the ref in sync with state on every render
  useEffect(() => { activeFiltersRef.current = activeFilters }, [activeFilters])

  // KEY FIX: Sync role → filters whenever auth context resolves.
  // useState initializer fires BEFORE user is loaded from localStorage/API,
  // so user?.role is undefined at first mount → wrong/empty filters.
  // This effect fires once user.role is available and sets the correct filters,
  // then immediately writes into activeFiltersRef so the next WS handshake is correct.
  useEffect(() => {
    if (!user?.role) return
    const correct = getFiltersForRole(user.role)
    setActiveFilters(correct)
    activeFiltersRef.current = correct   // update ref NOW, don't wait for render cycle
    // Remove unused setFiltersReady

    console.log(`[ROLE FILTERS] ✅ role=${user.role} →`, correct)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user?.role])

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
      if (data.status === 'connected') {
        setConnected(true)
        if (data.zone_type) setActiveZone(data.zone_type)
        return
      }
      if (data.status === 'filters_updated') {
        if (data.zone_type) setActiveZone(data.zone_type)
        return
      }
      if (data.error) { console.warn('[WS] error:', data.error); return }
      // Only process messages that actually have detection results
      if (data.annotated_frame === undefined && data.is_compliant === undefined) return
      // Update zone from every frame response
      if (data.zone_type && data.zone_type !== 'default') setActiveZone(data.zone_type)

      setFrameCount(f => f + 1)
      if (data.model_mode) setModelMode(data.model_mode)

      // Phone status with 5s hold — prevents rapid flickering
      // Timer resets on EVERY non-safe detection so alert stays while phone is visible.
      if (data.phone_status) {
        const now = Date.now()
        const latch = phoneStatusLatchRef.current
        const incoming = data.phone_status
        const HOLD_MS = 5000
        const priority = { zone_violation: 3, calling: 3, in_hand: 2, safe: 1 }
        const inPri = priority[incoming] || 1
        const latPri = priority[latch.status] || 1

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

      const _dets = data.detections || []
      // Worker fall / machine anomaly / object throwing / bangles → fire banner + toast
      const _fallDet = _dets.find(d => d.label?.toLowerCase().includes('fall'))
      const _anomalyDet = _dets.find(d => d.label?.toLowerCase().includes('anomaly'))
      const _throwDet = _dets.find(d => d.label?.toLowerCase().includes('throw'))
      const _banglesDet = _dets.find(d => d.label?.toLowerCase().includes('bangle'))
      if (_fallDet) {
        setCurrentAlert({ message: '🚨 Worker Fall Detected!', severity: 'critical' })
        addToast('🚨 Worker Fall!', 'A worker fall event was detected.', 'danger', 6000)
      }
      if (_anomalyDet) {
        setCurrentAlert({ message: '⚠️ Machine Anomaly Detected!', severity: 'high' })
      }
      if (_throwDet) {
        setCurrentAlert({ message: '🏃 Object Throwing Detected!', severity: 'high' })
        addToast('🏃 Object Throwing!', 'Aggressive throwing motion was detected.', 'danger', 5000)
      }
      if (_banglesDet) {
        setCurrentAlert({ message: '🚫 Bangles Detected (Food Safety)!', severity: 'high' })
        addToast('🚫 Bangles Violation', 'Bangles are not allowed in food production areas.', 'danger', 4000)
      }

      setDetectionInfo({
        isCompliant: data.is_compliant,
        missing: data.missing_items || [],
        detections: _dets,
        persons: data.persons || [],
        violationsCount: data.violations_count ?? 0,
        personsCount: data.persons_count ?? 0,
        phoneDetected: data.phone_detected ?? false,
        phoneStatus: data.phone_status || 'safe',
        cashDetected: data.cash_detected ?? false,
      })

      if (data.cash_alert) {
        setCashAlert(data.cash_alert)
        addToast('💰 Cash Theft Alert!', data.cash_alert, 'danger', 8000)
      } else if (data.cash_detected === false && cashAlert) {
        setCashAlert(null)
      }

      if (data.payee_snapshot_b64) {
        setPayeeData({
          snapshot: data.payee_snapshot_b64,
          id: data.payee_id || 'Vendor/Payee',
          time: new Date().toLocaleTimeString(),
        })
        addToast('📸 Payee Photo Logged', 'Vendor payee photo captured during cash exchange', 'info', 5000)
      }

      // Draw annotated frame onto the DISPLAY canvas.
      // captureRef handles sending; canvasRef shows bounding boxes.
      if (data.annotated_frame && canvasRef.current) {
        hasAnnotatedRef.current = true
        const img = new Image()
        img.onload = () => {
          const cv = canvasRef.current
          if (!cv) return
          // Resize canvas to match frame if needed
          if (img.width > 0 && cv.width !== img.width) {
            cv.width = img.width
            cv.height = img.height
          }
          const ctx = cv.getContext('2d')
          if (ctx) ctx.drawImage(img, 0, 0, cv.width, cv.height)
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
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: { ideal: 'environment' } }
      })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }
      hasAnnotatedRef.current = false
      connectWs()
      setStreaming(true)

      // Use a SEPARATE hidden canvas to capture raw frames for the backend.
      // The visible canvasRef is ONLY updated when annotated frames arrive from WS.
      // This prevents raw frames from overwriting bounding-box overlays every 80ms.
      intervalRef.current = setInterval(() => {
        if (!videoRef.current || !captureRef.current || !wsRef.current) return
        if (wsRef.current.readyState !== WebSocket.OPEN) return
        const vw = videoRef.current.videoWidth || 640
        const vh = videoRef.current.videoHeight || 480
        captureRef.current.width = vw
        captureRef.current.height = vh
        const ctx = captureRef.current.getContext('2d')
        ctx.drawImage(videoRef.current, 0, 0, vw, vh)
        const b64 = captureRef.current.toDataURL('image/jpeg', 0.65)
        wsRef.current.send(JSON.stringify({
          frame: b64,
          filters: activeFiltersRef.current,
          no_phone_zone: noPhoneZoneRef.current,
        }))
      }, FRAME_INTERVAL)
    } catch (err) {
      addToast('Camera error', err.message || 'Could not access webcam', 'danger')
    }
  }

  // ——————————————————————————————————————————————————————————————————————————————
  // CCTV / IP Camera mode — backend pulls frames directly from the camera URL
  const startCctv = useCallback(() => {
    const url = rtspUrl.trim()

    // ── Validate URL before connecting ───────────────────────────────────────
    if (!url) {
      addToast('Missing URL', 'Enter the IP camera stream URL first', 'danger')
      return
    }

    const isHttp = url.startsWith('http://') || url.startsWith('https://')
    const isRtsp = url.startsWith('rtsp://') || url.startsWith('rtsps://')
    if (!isHttp && !isRtsp) {
      addToast('Invalid URL', 'URL must start with http:// or rtsp://', 'danger')
      return
    }

    // Check for malformed IP addresses (e.g. "10165.131.102" missing a dot)
    if (isHttp) {
      try {
        const parsed = new URL(url)
        const host = parsed.hostname
        // If it looks like an IP (all digits and dots), validate each octet
        if (/^[\d.]+$/.test(host)) {
          const octets = host.split('.')
          if (octets.length !== 4) {
            addToast(
              'Invalid IP address',
              `"${host}" is not a valid IPv4 address — check for missing or extra dots.\nExample: 10.165.131.102`,
              'danger'
            )
            return
          }
          const bad = octets.find(o => o === '' || isNaN(Number(o)) || Number(o) > 255)
          if (bad !== undefined) {
            addToast(
              'Invalid IP address',
              `Each part of an IP must be 0–255. Check "${host}".`,
              'danger'
            )
            return
          }
        }
      } catch {
        addToast('Invalid URL', `Cannot parse URL: "${url}"`, 'danger')
        return
      }
    }

    // ── Close any lingering session before opening a new one ─────────────────
    // Guard against OPEN *and* CONNECTING/CLOSING states — all three can
    // leave a dangling backend session if we open a second socket on top.
    const prev = wsRef.current
    if (prev) {
      if (prev.readyState === WebSocket.OPEN || prev.readyState === WebSocket.CONNECTING) {
        prev.close()
      }
      wsRef.current = null
    }

    setFaceResult(null); setCamFps(0)
    const ws = new WebSocket(CCTV_WS_URL)
    wsRef.current = ws
    hasAnnotatedRef.current = false

    ws.onopen = () => {
      ws.send(JSON.stringify({
        token,
        camera_url: rtspUrl.trim(),
        filters: activeFiltersRef.current,
        no_phone_zone: noPhoneZoneRef.current,
        enable_face: enableFaceRef.current,
      }))
    }

    ws.onmessage = (e) => {
      const data = JSON.parse(e.data)
      if (data.status === 'connected') {
        setConnected(true); setStreaming(true)
        if (data.zone_type) setActiveZone(data.zone_type)
        return
      }
      if (data.error) {
        addToast('CCTV Error', data.error, 'danger')
        ws.close(); setStreaming(false); setConnected(false)
        return
      }
      if (data.annotated_frame === undefined && data.is_compliant === undefined) return

      setFrameCount(f => f + 1)
      if (data.model_mode) setModelMode(data.model_mode)
      if (data.cam_fps !== undefined) setCamFps(data.cam_fps)
      if (data.face_result) setFaceResult(data.face_result)

      // Phone status latch
      if (data.phone_status) {
        const now = Date.now(), latch = phoneStatusLatchRef.current
        const incoming = data.phone_status, HOLD_MS = 5000
        const priority = { zone_violation: 3, calling: 3, in_hand: 2, safe: 1 }
        const inPri = priority[incoming] || 1, latPri = priority[latch.status] || 1
        if (incoming !== 'safe') {
          if (inPri >= latPri || now > latch.until) {
            phoneStatusLatchRef.current = { status: incoming, until: now + HOLD_MS }
            setPhoneStatus(incoming)
          } else { phoneStatusLatchRef.current = { ...latch, until: now + HOLD_MS } }
        } else if (now > latch.until) {
          phoneStatusLatchRef.current = { status: 'safe', until: 0 }; setPhoneStatus('safe')
        }
      }

      if (data.zone_type && data.zone_type !== 'default') setActiveZone(data.zone_type)

      const _cctvDets = data.detections || []
      const _cctvFall = _cctvDets.find(d => d.label?.toLowerCase().includes('fall'))
      const _cctvAnom = _cctvDets.find(d => d.label?.toLowerCase().includes('anomaly'))
      const _cctvThrow = _cctvDets.find(d => d.label?.toLowerCase().includes('throw'))
      const _cctvBangles = _cctvDets.find(d => d.label?.toLowerCase().includes('bangle'))
      if (_cctvFall) {
        setCurrentAlert({ message: '🚨 Worker Fall Detected!', severity: 'critical' })
        addToast('🚨 Worker Fall!', 'A worker fall event was detected.', 'danger', 6000)
      }
      if (_cctvAnom) {
        setCurrentAlert({ message: '⚠️ Machine Anomaly Detected!', severity: 'high' })
      }
      if (_cctvThrow) {
        setCurrentAlert({ message: '🏃 Object Throwing Detected!', severity: 'high' })
        addToast('🏃 Object Throwing!', 'Aggressive throwing motion was detected.', 'danger', 5000)
      }
      if (_cctvBangles) {
        setCurrentAlert({ message: '🚫 Bangles Detected (Food Safety)!', severity: 'high' })
        addToast('🚫 Bangles Violation', 'Bangles are not allowed in food production areas.', 'danger', 4000)
      }

      setDetectionInfo({
        isCompliant: data.is_compliant,
        missing: data.missing_items || [],
        detections: _cctvDets,
        persons: data.persons || [],
        violationsCount: data.violations_count ?? 0,
        personsCount: data.persons_count ?? 0,
        phoneDetected: data.phone_detected ?? false,
        phoneStatus: data.phone_status || 'safe',
        cashDetected: data.cash_detected ?? false,
      })

      // Cash theft alert banner — show when server fires a theft event
      if (data.cash_alert) {
        setCashAlert(data.cash_alert)
        addToast('💰 Cash Theft Alert!', data.cash_alert, 'danger', 8000)
      } else if (!data.cash_alert && cashAlert) {
        // Clear only when the server explicitly sends no alert (not on empty frames)
        if (data.cash_detected === false) setCashAlert(null)
      }

      if (data.payee_snapshot_b64) {
        setPayeeData({
          snapshot: data.payee_snapshot_b64,
          id: data.payee_id || 'Vendor/Payee',
          time: new Date().toLocaleTimeString(),
        })
        addToast('📸 Payee Photo Logged', 'Vendor payee photo captured during cash exchange', 'info', 5000)
      }

      // Draw backend-annotated frame (PPE + face boxes already merged)
      if (data.annotated_frame && canvasRef.current) {
        hasAnnotatedRef.current = true
        const img = new Image()
        img.onload = () => {
          const cv = canvasRef.current; if (!cv) return
          if (img.width > 0 && cv.width !== img.width) { cv.width = img.width; cv.height = img.height }
          const ctx = cv.getContext('2d'); if (ctx) ctx.drawImage(img, 0, 0, cv.width, cv.height)
        }
        img.src = data.annotated_frame
      }

      if (!data.is_compliant && data.alert_message) {
        setCurrentAlert({ message: data.alert_message, severity: data.severity })
        if (data.alert_saved || data.face_alert_saved)
          addToast('⚠️ Alert Saved!', data.alert_message, 'danger', 5000)
      } else { setCurrentAlert(null) }
    }

    ws.onclose = () => { setConnected(false); setStreaming(false); setFaceResult(null); setCamFps(0) }
    ws.onerror = () => { setConnected(false); setStreaming(false) }

    // Sync all live state every 2 s (filters + phone zone + face toggle)
    intervalRef.current = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({
          filters: activeFiltersRef.current,
          no_phone_zone: noPhoneZoneRef.current,
          enable_face: enableFaceRef.current,
        }))
      }
    }, 2000)
  }, [token, addToast, rtspUrl])


  // ——————————————————————————————————————————————————————————————————————————————
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
    setFaceResult(null)
    setCamFps(0)
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
      const id = res.data.job_id
      // Remove unused setJobId
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
              <span style={{
                marginLeft: 10, fontSize: '0.72rem', color: 'var(--text-muted)',
                background: 'rgba(255,255,255,0.06)', padding: '2px 8px', borderRadius: 99
              }}>
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
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontSize: '0.78rem', fontWeight: 700, color: 'var(--text-muted)', letterSpacing: '0.05em' }}>
              DETECTION FILTERS
            </span>
            <span style={{
              fontSize: '0.72rem', color: 'var(--text-muted)', background: 'var(--bg-card)',
              padding: '1px 7px', borderRadius: 99, border: '1px solid var(--border)'
            }}>
              {activeFilters.length}/{PPE_FILTERS.length} active
            </span>
          </div>
          <div style={{ display: 'flex', gap: 6 }}>
            <button className="btn btn-ghost btn-sm" style={{ padding: '3px 10px', fontSize: '0.72rem' }}
              onClick={() => setActiveFilters(PPE_FILTERS.map(f => f.id))}>All</button>
            <button className="btn btn-ghost btn-sm" style={{ padding: '3px 10px', fontSize: '0.72rem' }}
              onClick={() => setActiveFilters([])}>None</button>
            <button className="btn btn-ghost btn-sm" style={{ padding: '3px 10px', fontSize: '0.72rem' }}
              onClick={() => setShowFilters(s => !s)}>
              {showFilters ? '▲ Hide' : '▼ Edit'}
            </button>
          </div>
        </div>

        {/* Expanded toggle grid */}
        {showFilters && (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 12 }}>
            {PPE_FILTERS.map(f => {
              const on = activeFilters.includes(f.id)
              return (
                <button key={f.id}
                  onClick={() => setActiveFilters(prev =>
                    on ? prev.filter(x => x !== f.id) : [...prev, f.id]
                  )}
                  style={{
                    display: 'flex', alignItems: 'center', gap: 6,
                    padding: '7px 14px', borderRadius: 99, cursor: 'pointer',
                    fontSize: '0.82rem', fontWeight: 600,
                    border: `1px solid ${on ? 'rgba(249,115,22,0.5)' : 'var(--border)'}`,
                    background: on ? 'rgba(249,115,22,0.12)' : 'var(--bg-card)',
                    color: on ? '#fb923c' : 'var(--text-muted)',
                    transition: 'all 0.15s',
                  }}>
                  <span style={{ fontSize: '1rem' }}>{f.icon}</span>
                  {f.label}
                  {on
                    ? <span style={{ width: 7, height: 7, borderRadius: '50%', background: '#fb923c', flexShrink: 0 }} />
                    : <span style={{ width: 7, height: 7, borderRadius: '50%', background: 'var(--border)', flexShrink: 0 }} />}
                </button>
              )
            })}
          </div>
        )}

        {/* Collapsed chip strip */}
        {!showFilters && (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5, marginTop: 8 }}>
            {PPE_FILTERS.map(f => {
              const on = activeFilters.includes(f.id)
              return (
                <button key={f.id}
                  onClick={() => setActiveFilters(prev =>
                    on ? prev.filter(x => x !== f.id) : [...prev, f.id]
                  )}
                  title={on ? `Disable ${f.label} detection` : `Enable ${f.label} detection`}
                  style={{
                    fontSize: '0.7rem', padding: '3px 9px', borderRadius: 99, cursor: 'pointer',
                    border: `1px solid ${on ? 'rgba(249,115,22,0.35)' : 'var(--border)'}`,
                    background: on ? 'rgba(249,115,22,0.08)' : 'transparent',
                    color: on ? '#fb923c' : 'var(--text-muted)',
                    transition: 'all 0.12s',
                  }}>
                  {f.icon} {f.label}
                </button>
              )
            })}
          </div>
        )}

        {/* ————————————————————————————————————————————————————————————————————————————— */}
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          borderTop: '1px solid var(--border)', marginTop: 10, paddingTop: 10
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <span style={{ fontSize: '1rem' }}>📱</span>
            <div>
              <div style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                No Phone Zone
              </div>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                Any phone detected triggers alert
              </div>
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            {/* Phone status badge — always visible */}
            <span style={{
              fontSize: '0.72rem', fontWeight: 600, padding: '3px 10px', borderRadius: 99,
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
              {phoneStatus === 'safe' ? '🟢 No Phone'
                : phoneStatus === 'in_hand' ? '🟡 Phone in Hand'
                  : phoneStatus === 'calling' ? '🔴 Calling Alert'
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
                width: 44, height: 24, borderRadius: 12, cursor: 'pointer',
                border: 'none', padding: 0, transition: 'background 0.2s',
                background: noPhoneZone ? '#ef4444' : 'rgba(255,255,255,0.12)',
                position: 'relative', flexShrink: 0,
              }}
            >
              <span style={{
                position: 'absolute', top: 3, width: 18, height: 18, borderRadius: '50%',
                background: '#fff', transition: 'left 0.2s',
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
          { id: 'rtsp', label: 'CCTV / RTSP' },
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
          <div className="video-container" style={{ minHeight: 300, position: 'relative' }}>
            {/* Hidden capture canvas — raw frames only, never shown to user */}
            <canvas ref={captureRef} style={{ display: 'none' }} />

            {/* Raw MJPEG stream for CCTV mode — shown as background */}
            {mode === 'rtsp' && streaming && (
              <img
                ref={cctvImgRef}
                src={rtspUrl.trim()}
                alt="CCTV feed"
                style={{
                  display: 'block',
                  position: 'absolute', top: 0, left: 0,
                  width: '100%', height: '100%', objectFit: 'cover',
                  opacity: hasAnnotatedRef.current ? 0 : 1,
                }}
                onError={() => { /* img will keep retrying for MJPEG */ }}
              />
            )}

            {/* Live webcam video: shown as background until first annotated frame */}
            <video ref={videoRef} muted playsInline
              style={{
                display: (streaming && mode === 'webcam') ? 'block' : 'none',
                position: 'absolute', top: 0, left: 0,
                width: '100%', height: '100%', objectFit: 'cover',
                opacity: hasAnnotatedRef.current ? 0 : 1,
              }}
            />

            {/* Annotated display canvas: overlays bounding boxes on top of live video */}
            <canvas ref={canvasRef} className="video-canvas"
              style={{
                display: streaming ? 'block' : 'none',
                position: 'absolute', top: 0, left: 0,
                width: '100%', height: '100%',
              }}
            />

            {!streaming && mode !== 'upload' && (
              <div className="no-feed">
                <VideoOff size={40} style={{ opacity: 0.3 }} />
                <div>No feed active</div>
                <div style={{ fontSize: '0.78rem' }}>
                  {mode === 'webcam'
                    ? 'Click Start Detection to begin'
                    : 'Confirm the IP camera URL below and click Start'}
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
                {detectionInfo.violationsCount}/{detectionInfo.personsCount} violating
              </div>
            )}

            {/* Frame counter + cam FPS + active zone badge */}
            {streaming && (
              <div style={{
                position: 'absolute', top: 12, right: 12, display: 'flex', gap: 6, alignItems: 'center'
              }}>
                {/* Active zone badge */}
                {activeZone && activeZone !== 'default' && (
                  <div style={{
                    background: 'rgba(16,185,129,0.80)', padding: '4px 10px',
                    borderRadius: 99, fontSize: '0.70rem', color: '#fff', fontWeight: 700,
                    textTransform: 'uppercase', letterSpacing: '0.05em',
                  }}>
                    🏭 {activeZone.replace(/_/g, ' ')}
                  </div>
                )}
                {mode === 'rtsp' && camFps > 0 && (
                  <div style={{
                    background: 'rgba(99,102,241,0.75)', padding: '4px 10px',
                    borderRadius: 99, fontSize: '0.72rem', color: '#fff', fontWeight: 700,
                  }}>
                    {camFps} fps
                  </div>
                )}
                <div style={{
                  background: 'rgba(0,0,0,0.6)', padding: '4px 10px',
                  borderRadius: 99, fontSize: '0.72rem', color: 'var(--text-secondary)'
                }}>
                  {frameCount} frames
                </div>
              </div>
            )}
          </div>

          {/* RTSP / CCTV URL input + face toggle */}
          {mode === 'rtsp' && (() => {
            // Inline validation — check IP while typing
            let urlWarning = null
            if (rtspUrl && (rtspUrl.startsWith('http://') || rtspUrl.startsWith('https://'))) {
              try {
                const h = new URL(rtspUrl).hostname
                if (/^[\d.]+$/.test(h)) {
                  const parts = h.split('.')
                  if (parts.length !== 4) {
                    urlWarning = `⚠️ "${h}" looks like an invalid IP — missing a dot? (e.g. 10.165.131.102)`
                  } else if (parts.some(p => p === '' || isNaN(Number(p)) || Number(p) > 255)) {
                    urlWarning = `⚠️ Each IP part must be 0–255. Check "${h}"`
                  }
                }
              } catch { urlWarning = '⚠️ URL format is invalid' }
            }

            return (
              <div className="form-group" style={{ marginTop: 12 }}>
                <label className="form-label">IP Camera / MJPEG Stream URL</label>

                {/* URL input with clear button */}
                <div style={{ position: 'relative' }}>
                  <input
                    className="form-input"
                    type="url"
                    spellCheck={false}
                    placeholder="http://10.62.212.243:8080/video"
                    value={rtspUrl}
                    onChange={e => setRtspUrl(e.target.value)}
                    disabled={streaming}
                    style={{
                      paddingRight: rtspUrl ? 32 : 12,
                      borderColor: urlWarning ? 'var(--accent-red, #ef4444)' : undefined,
                    }}
                  />
                  {rtspUrl && !streaming && (
                    <button
                      type="button"
                      onClick={() => setRtspUrl('')}
                      title="Clear"
                      style={{
                        position: 'absolute', right: 8, top: '50%',
                        transform: 'translateY(-50%)',
                        background: 'none', border: 'none',
                        color: 'var(--text-muted)', cursor: 'pointer',
                        fontSize: '1rem', lineHeight: 1, padding: 0,
                      }}
                    >✕</button>
                  )}
                </div>

                {/* Inline validation error */}
                {urlWarning && (
                  <div style={{
                    marginTop: 6, padding: '6px 10px', borderRadius: 7,
                    background: 'rgba(239,68,68,0.10)',
                    border: '1px solid rgba(239,68,68,0.35)',
                    fontSize: '0.75rem', color: '#ef4444', lineHeight: 1.5,
                  }}>
                    {urlWarning}
                    {rtspUrl.includes('http://') && (() => {
                      try {
                        const h = new URL(rtspUrl).hostname
                        const suggested = h.replace(/(\d{3,})(?=\d)/g, '$1.')
                        if (suggested !== h) return ` — Did you mean ${suggested}?`
                      } catch { /* ignore */ }
                      return null
                    })()}
                  </div>
                )}

                {/* Quick-fill preset buttons */}
                {!streaming && (
                  <div style={{ display: 'flex', gap: 6, marginTop: 6, flexWrap: 'wrap' }}>
                    <button
                      type="button"
                      className="btn btn-ghost"
                      style={{ fontSize: '0.68rem', padding: '3px 8px', height: 'auto' }}
                      onClick={() => setRtspUrl('http://10.62.212.243:8080/video')}
                    >📱 IP Webcam (MJPEG)</button>
                    <button
                      type="button"
                      className="btn btn-ghost"
                      style={{ fontSize: '0.68rem', padding: '3px 8px', height: 'auto' }}
                      onClick={() => setRtspUrl('rtsp://admin:password@192.168.1.1:554/stream')}
                    >📷 RTSP Example</button>
                  </div>
                )}

                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: 6 }}>
                  Supports: <code style={{ fontSize: '0.72rem' }}>http://IP:PORT/video</code> (MJPEG)
                  &nbsp;•&nbsp; <code style={{ fontSize: '0.72rem' }}>rtsp://user:pass@IP:554/stream</code>
                </div>

                {/* Face recognition toggle */}
                <div style={{
                  display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                  marginTop: 10, padding: '8px 12px', borderRadius: 8,
                  background: 'var(--bg-card)', border: '1px solid var(--border)',
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span style={{ fontSize: '1rem' }}>🫥</span>
                    <div>
                      <div style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-primary)' }}>Face Recognition</div>
                      <div style={{ fontSize: '0.70rem', color: 'var(--text-muted)' }}>Identify known/unknown persons in stream</div>
                    </div>
                  </div>
                  <button
                    onClick={() => setEnableFace(f => !f)}
                    style={{
                      width: 44, height: 24, borderRadius: 12, cursor: 'pointer',
                      border: 'none', padding: 0, transition: 'background 0.2s',
                      background: enableFace ? '#6366f1' : 'rgba(255,255,255,0.12)',
                      position: 'relative', flexShrink: 0,
                    }}
                  >
                    <span style={{
                      position: 'absolute', top: 3, width: 18, height: 18,
                      borderRadius: '50%', background: '#fff', transition: 'left 0.2s',
                      left: enableFace ? 23 : 3,
                    }} />
                  </button>
                </div>
              </div>
            )
          })()}

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
                <button
                  id="btn-start-detection"
                  className="btn btn-success"
                  onClick={mode === 'rtsp' ? startCctv : startWebcam}
                >
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
                {/* ── Cash Theft Alert Banner ─────────────────────────────── */}
                {cashAlert && (
                  <div style={{
                    display: 'flex', alignItems: 'flex-start', gap: 10,
                    padding: '10px 14px', borderRadius: 10, marginBottom: 14,
                    background: 'rgba(234,179,8,0.15)',
                    border: '1px solid rgba(234,179,8,0.55)',
                    animation: 'pulse 1.5s ease-in-out infinite',
                  }}>
                    <span style={{ fontSize: '1.3rem', flexShrink: 0 }}>💰</span>
                    <div>
                      <div style={{ color: '#eab308', fontWeight: 700, fontSize: '0.85rem', marginBottom: 2 }}>
                        CASH THEFT ALERT
                      </div>
                      <div style={{ color: '#fde047', fontSize: '0.78rem', lineHeight: 1.4 }}>
                        {cashAlert}
                      </div>
                    </div>
                    <button
                      onClick={() => setCashAlert(null)}
                      style={{
                        marginLeft: 'auto', background: 'none', border: 'none',
                        cursor: 'pointer', color: '#eab308', fontSize: '1rem', flexShrink: 0
                      }}
                      title="Dismiss"
                    >✕</button>
                  </div>
                )}

                {/* ── Vendor Payee Snapshot Card (REQ-SH-2) ────────────────── */}
                {payeeData && (
                  <div style={{
                    display: 'flex', flexDirection: 'column', gap: 8,
                    padding: '10px 14px', borderRadius: 10, marginBottom: 14,
                    background: 'rgba(99,102,241,0.12)',
                    border: '1px solid rgba(99,102,241,0.45)',
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: '#818cf8', fontWeight: 700, fontSize: '0.82rem' }}>
                        <span>📸</span> VENDOR / PAYEE CAPTURE
                      </div>
                      <button
                        onClick={() => setPayeeData(null)}
                        style={{ background: 'none', border: 'none', color: '#818cf8', cursor: 'pointer', fontSize: '0.9rem' }}
                        title="Dismiss"
                      >✕</button>
                    </div>
                    <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
                      <img
                        src={`data:image/jpeg;base64,${payeeData.snapshot}`}
                        alt="Payee Snapshot"
                        style={{
                          width: 80, height: 80, objectFit: 'cover', borderRadius: 8,
                          border: '2px solid #6366f1', flexShrink: 0,
                        }}
                      />
                      <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', display: 'flex', flexDirection: 'column', gap: 4 }}>
                        <div style={{ color: '#fff', fontWeight: 600 }}>ID: {payeeData.id}</div>
                        <div>Time: {payeeData.time}</div>
                        <div style={{ color: '#10b981', fontWeight: 600, display: 'flex', alignItems: 'center', gap: 4 }}>
                          <span>✓</span> Cash exchange recorded
                        </div>
                      </div>
                    </div>
                  </div>
                )}

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
                      ? `All ${detectionInfo.personsCount} person(s) compliant`
                      : ` ${detectionInfo.violationsCount}/${detectionInfo.personsCount} person(s) violating`}
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

                {/* Detected Objects & Items (Cash, Cylinder, Bangles, etc.) */}
                {detectionInfo.detections && detectionInfo.detections.filter(d => d.label !== 'Person').length > 0 && (
                  <div style={{ marginBottom: 14 }}>
                    <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginBottom: 6, letterSpacing: '0.05em' }}>
                      DETECTED OBJECTS & ITEMS
                    </div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                      {detectionInfo.detections
                        .filter(d => d.label !== 'Person')
                        .map((d, idx) => {
                          const lbl = d.label?.toLowerCase() || ''
                          const isCash = lbl === 'cash'
                          const isCyl = lbl.includes('cylinder')
                          const isBangles = lbl.includes('bangles')
                          const isFall = lbl.includes('fall')
                          const isAnom = lbl.includes('anomaly')
                          const isHairnet = lbl.includes('head-cap') || lbl.includes('hairnet')
                          const isThrow = lbl.includes('throwing')
                          const icon = isCash ? '💵' : isCyl ? '🛢️' : isBangles ? '🚨'
                            : isFall ? '🚨' : isAnom ? '⚠️' : isHairnet ? '🧢'
                              : isThrow ? '🤚' : '📦'
                          const bg = isCash ? 'rgba(234,179,8,0.15)' : isCyl ? 'rgba(6,182,212,0.15)'
                            : isBangles || isFall ? 'rgba(239,68,68,0.15)'
                              : isAnom ? 'rgba(245,158,11,0.15)'
                                : isHairnet ? 'rgba(99,102,241,0.15)'
                                  : isThrow ? 'rgba(249,115,22,0.15)' : 'rgba(255,255,255,0.08)'
                          const clr = isCash ? '#eab308' : isCyl ? '#06b6d4'
                            : isBangles || isFall ? '#ef4444'
                              : isAnom ? '#f59e0b'
                                : isHairnet ? '#6366f1'
                                  : isThrow ? '#f97316' : 'var(--text-primary)'
                          const bdr = `1px solid ${isCash ? 'rgba(234,179,8,0.4)' : isCyl ? 'rgba(6,182,212,0.4)'
                            : isBangles || isFall ? 'rgba(239,68,68,0.4)'
                              : isAnom ? 'rgba(245,158,11,0.4)'
                                : isHairnet ? 'rgba(99,102,241,0.4)'
                                  : isThrow ? 'rgba(249,115,22,0.4)' : 'var(--border)'}`
                          return (
                            <span
                              key={idx}
                              style={{
                                display: 'inline-flex', alignItems: 'center', gap: 4,
                                padding: '4px 10px', borderRadius: 99, fontSize: '0.75rem', fontWeight: 600,
                                background: bg, color: clr, border: bdr,
                              }}
                            >
                              <span>{icon}</span> {d.label} {Math.round((d.confidence || 0) * 100)}%
                            </span>
                          )
                        })}
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

          {/* Face Recognition results card — CCTV mode only */}
          {mode === 'rtsp' && streaming && (
            <div className="card card-p">
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
                <h3 style={{ fontSize: '0.95rem', margin: 0 }}>🫥 Face Recognition</h3>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  {faceResult && (
                    <span style={{
                      fontSize: '0.72rem', fontWeight: 700, padding: '2px 9px', borderRadius: 99,
                      background: faceResult.unknown_detected ? 'rgba(239,68,68,0.15)' : 'rgba(16,185,129,0.12)',
                      color: faceResult.unknown_detected ? '#ef4444' : '#10b981',
                      border: `1px solid ${faceResult.unknown_detected ? 'rgba(239,68,68,0.3)' : 'rgba(16,185,129,0.3)'}`,
                    }}>
                      {faceResult.unknown_detected ? '⚠️ UNKNOWN' : `✓ ${faceResult.face_count} face${faceResult.face_count !== 1 ? 's' : ''}`}
                    </span>
                  )}
                  <button
                    onClick={() => setEnableFace(f => !f)}
                    title={enableFace ? 'Disable face recognition' : 'Enable face recognition'}
                    style={{
                      width: 36, height: 20, borderRadius: 10, cursor: 'pointer',
                      border: 'none', padding: 0, transition: 'background 0.2s',
                      background: enableFace ? '#6366f1' : 'rgba(255,255,255,0.12)',
                      position: 'relative', flexShrink: 0,
                    }}
                  >
                    <span style={{
                      position: 'absolute', top: 2, width: 16, height: 16,
                      borderRadius: '50%', background: '#fff', transition: 'left 0.2s',
                      left: enableFace ? 18 : 2,
                    }} />
                  </button>
                </div>
              </div>

              {!enableFace ? (
                <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>Face recognition is disabled</div>
              ) : !faceResult || faceResult.face_count === 0 ? (
                <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>No faces detected in frame</div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {faceResult.faces?.map((face, i) => (
                    <div key={i} style={{
                      display: 'flex', alignItems: 'center', gap: 10,
                      padding: '8px 12px', borderRadius: 8,
                      background: face.is_unknown ? 'rgba(239,68,68,0.08)' : 'rgba(16,185,129,0.07)',
                      border: `1px solid ${face.is_unknown ? 'rgba(239,68,68,0.25)' : 'rgba(16,185,129,0.2)'}`,
                    }}>
                      <span style={{ fontSize: '1.3rem' }}>{face.is_unknown ? '❌' : '✅'}</span>
                      <div style={{ flex: 1 }}>
                        <div style={{
                          fontSize: '0.85rem', fontWeight: 700,
                          color: face.is_unknown ? 'var(--accent-red)' : 'var(--accent-green)',
                        }}>
                          {face.is_unknown ? 'UNKNOWN' : face.label}
                        </div>
                        <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                          Confidence: {Math.round(face.confidence * 100)}%
                        </div>
                      </div>
                      {face.is_unknown && (
                        <span style={{
                          fontSize: '0.68rem', fontWeight: 700, padding: '2px 8px',
                          borderRadius: 99, background: 'rgba(239,68,68,0.2)', color: '#f87171',
                          border: '1px solid rgba(239,68,68,0.3)',
                        }}>ALERT</span>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

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
                  Download ppe.pt - biswadeep-roy/Safety-Detection-YOLOv8
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
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: (isViolator || person.has_uniform !== undefined) ? 6 : 0 }}>
        <User size={14} color={isViolator ? 'var(--accent-red)' : 'var(--accent-green)'} />
        <span style={{
          fontSize: '0.82rem', fontWeight: 600,
          color: isViolator ? 'var(--accent-red)' : 'var(--accent-green)'
        }}>
          Person {index + 1}
        </span>
        <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginLeft: 'auto' }}>
          {(person.confidence * 100).toFixed(0)}% conf
        </span>
      </div>
      {person.has_uniform !== undefined && (
        <div style={{ marginBottom: 6, display: 'flex', gap: 6, alignItems: 'center' }}>
          <span style={{
            fontSize: '0.70rem', padding: '2px 8px', borderRadius: 99,
            background: person.has_uniform ? 'rgba(16,185,129,0.15)' : 'rgba(220,38,38,0.2)',
            color: person.has_uniform ? '#34d399' : '#f87171',
            fontWeight: 600, border: `1px solid ${person.has_uniform ? 'rgba(16,185,129,0.3)' : 'rgba(220,38,38,0.3)'}`
          }}>
            {person.has_uniform ? '👕 Uniform OK' : '👕 No Uniform'}
            {person.uniform_confidence ? ` (${Math.round(person.uniform_confidence * 100)}%)` : ''}
          </span>
        </div>
      )}
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
            }}>{p}</span>
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
  const videoRef = useRef(null)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [tooltip, setTooltip] = useState(null) // {x, vt}
  const [activeVtIdx, setActiveVtIdx] = useState(-1)
  const listRef = useRef(null)

  const SEV_COLORS = {
    critical: { bg: 'rgba(220,38,38,0.15)', color: '#f87171', border: 'rgba(220,38,38,0.3)', dot: '#ef4444' },
    high: { bg: 'rgba(234,88,12,0.12)', color: '#fb923c', border: 'rgba(234,88,12,0.3)', dot: '#f97316' },
    medium: { bg: 'rgba(234,179,8,0.12)', color: '#fbbf24', border: 'rgba(234,179,8,0.3)', dot: '#eab308' },
    low: { bg: 'rgba(16,185,129,0.10)', color: '#34d399', border: 'rgba(16,185,129,0.3)', dot: '#10b981' },
  }

  const isComplete = status.status === 'complete'
  const isProcessing = status.status === 'processing'
  const isError = status.status === 'error'
  const ppeSummary = status.ppe_summary || {}
  const ppeSummaryEntries = Object.entries(ppeSummary)
  const maxViolCount = ppeSummaryEntries.length ? Math.max(...ppeSummaryEntries.map(([, v]) => v)) : 1
  const vts = useMemo(() => status.violation_timestamps || [], [status.violation_timestamps])
  const videoDur = status.video_duration_sec || duration || 1
  const videoUrl = status.annotated_video_url  // e.g. /uploads/annotated_{id}.mp4

  // Sync currentTime while video plays
  useEffect(() => {
    const vid = videoRef.current
    if (!vid) return
    const onTime = () => {
      const ct = vid.currentTime
      setCurrentTime(ct)
      // Find active violation
      const idx = vts.findIndex((v, i) => {
        const next = vts[i + 1]?.ts ?? Infinity
        return ct >= v.ts && ct < next
      })
      setActiveVtIdx(idx)
    }
    const onLoad = () => setDuration(vid.duration || videoDur)
    const onPlay = () => setPlaying(true)
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
          {isComplete && <CheckCircle size={16} color="var(--accent-green)" />}
          {isError && <AlertTriangle size={16} color="var(--accent-red)" />}
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
              { label: 'Frames scanned', val: status.frames_processed ?? 0 },
              { label: 'Alerts found', val: status.total_alerts ?? 0 },
              { label: 'Total violations', val: status.total_violations ?? 0 },
              { label: 'Duration', val: `${videoDur}s` },
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
                  <span style={{ fontSize: 22, marginLeft: 4, color: '#fff' }}>&#9654;</span>
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
                {vts[activeVtIdx].items?.join(', ') || 'Violation'}
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
                  const sev = SEV_COLORS[vt.severity] || SEV_COLORS.medium
                  return (
                    <div
                      key={i}
                      onMouseEnter={() => {
                        setTooltip({ x: left, vt, idx: i })
                      }}
                      onMouseLeave={() => setTooltip(null)}
                      onClick={(e) => { e.stopPropagation(); seekTo(vt.ts) }}
                      style={{
                        position: 'absolute', top: -3, width: 14, height: 14,
                        left: `${left}%`,
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
                  <div style={{ fontSize: '0.72rem', fontWeight: 700, color: (SEV_COLORS[tooltip.vt.severity] || SEV_COLORS.medium).color, marginBottom: 2 }}>
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
                { label: 'High', dot: '#f97316' },
                { label: 'Medium', dot: '#eab308' },
                { label: 'Safe', color: 'var(--accent-cyan)', isLine: true },
              ].map(({ label, dot, isLine }) => (
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
          <div style={{
            fontSize: '0.72rem', fontWeight: 700, color: 'var(--text-muted)',
            letterSpacing: '0.05em', marginBottom: 10
          }}>MOST-MISSED PPE ITEMS</div>
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
          <div style={{
            fontSize: '0.72rem', fontWeight: 700, color: 'var(--text-muted)',
            letterSpacing: '0.05em', marginBottom: 8
          }}>VIOLATION TIMELINE</div>
          <div ref={listRef} style={{
            display: 'flex', flexDirection: 'column', gap: 6,
            maxHeight: 340, overflowY: 'auto', paddingRight: 4
          }}>
            {vts.map((vt, i) => {
              const sev = SEV_COLORS[vt.severity] || SEV_COLORS.medium
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
                      style={{
                        width: 52, height: 36, objectFit: 'cover',
                        borderRadius: 4, flexShrink: 0, border: `1px solid ${sev.border}`
                      }} />
                  )}
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 3 }}>
                      <span style={{ fontSize: '0.8rem', fontWeight: 700, color: sev.color }}>@{vt.ts}s</span>
                      {vt.violations > 0 && (
                        <span style={{
                          fontSize: '0.68rem', fontWeight: 700, color: sev.color,
                          padding: '1px 7px', borderRadius: 99, border: `1px solid ${sev.border}`,
                          background: sev.bg, flexShrink: 0
                        }}>
                          {vt.violations}/{vt.persons} person{vt.persons !== 1 ? 's' : ''}
                        </span>
                      )}
                      {active && (
                        <span style={{
                          fontSize: '0.65rem', background: sev.dot, color: '#fff',
                          padding: '1px 6px', borderRadius: 99, fontWeight: 700, flexShrink: 0
                        }}>
                          NOW
                        </span>
                      )}
                    </div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                      {vt.items?.map((m, j) => (
                        <span key={j} style={{
                          fontSize: '0.68rem', padding: '1px 7px', borderRadius: 99,
                          background: 'rgba(0,0,0,0.25)', color: sev.color, fontWeight: 600,
                        }}> {m}</span>
                      ))}
                    </div>
                  </div>
                  {/* Seek arrow */}
                  <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', flexShrink: 0, alignSelf: 'center' }}>&#9654;</span>
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
        <div className="card card-p" style={{
          fontSize: '0.82rem', color: 'var(--accent-red)',
          padding: '12px 16px', background: 'rgba(220,38,38,0.1)', border: '1px solid rgba(220,38,38,0.3)'
        }}>
          Processing error â€” try uploading the video again
        </div>
      )}
    </div>
  )
}
