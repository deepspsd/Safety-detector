import { useState, useEffect, useCallback, useRef } from 'react'
import QRCode from 'qrcode'
import api from '../api/api'
import { useToast } from '../context/ToastContext'
import {
  FileText, Upload, CheckCircle, XCircle, RefreshCw,
  Filter, Eye, EyeOff, Trash2, Camera, CameraOff,
  QrCode, Smartphone, Clock, AlertCircle, ChevronDown, ChevronUp, Copy, ZoomIn, X
} from 'lucide-react'

// --- API helpers ---
const documentsApi = {
  stats:           ()           => api.get('/documents/stats'),
  list:            (params)     => api.get('/documents/', { params }),
  approve:         (id, table)  => api.patch('/documents/' + id + '/approve?table=' + table),
  reject:          (id, table)  => api.patch('/documents/' + id + '/reject?table=' + table),
  delete:          (id, table)  => api.delete('/documents/' + id + '?table=' + table),
  qrToken:         (direction)  => api.get('/documents/qr-token', { params: { direction } }),
  pendingApproval: ()           => api.get('/documents/pending-approval'),
}

// --- Image Lightbox ---
function ImageLightbox({ src, label, onClose }) {
  useEffect(() => {
    const onKey = e => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])
  if (!src) return null
  return (
    <div onClick={onClose} style={{
      position: 'fixed', inset: 0, zIndex: 9999,
      background: 'rgba(0,0,0,0.88)', backdropFilter: 'blur(6px)',
      display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
      padding: 24, cursor: 'zoom-out',
      animation: 'lb-fade 0.2s ease',
    }}>
      <style>{`
        @keyframes lb-fade { from{opacity:0;transform:scale(0.94)} to{opacity:1;transform:scale(1)} }
        .lb-img { max-width:min(92vw,900px); max-height:82vh; object-fit:contain;
          border-radius:12px; box-shadow:0 25px 80px rgba(0,0,0,0.7);
          animation:lb-fade 0.22s cubic-bezier(.22,1,.36,1); }
      `}</style>
      <div onClick={e => e.stopPropagation()} style={{ position: 'relative', display: 'inline-block' }}>
        <img src={src} alt={label || 'Invoice'} className='lb-img' />
        <button onClick={onClose} style={{
          position: 'absolute', top: -14, right: -14, width: 32, height: 32,
          borderRadius: '50%', border: 'none', cursor: 'pointer',
          background: '#ef4444', color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center',
          boxShadow: '0 4px 12px rgba(0,0,0,0.4)', zIndex: 1,
        }}><X size={16} /></button>
        {label && (
          <div style={{ position: 'absolute', bottom: -36, left: 0, right: 0, textAlign: 'center',
            fontSize: '0.8rem', color: 'rgba(255,255,255,0.6)', letterSpacing: '0.04em' }}>{label}</div>
        )}
      </div>
    </div>
  )
}

// Reusable clickable thumbnail wrapper
function Thumb({ src, label, style }) {
  const [open, setOpen] = useState(false)
  if (!src) return null
  return (
    <>
      <div style={{ position: 'relative', display: 'inline-block', cursor: 'zoom-in' }} onClick={() => setOpen(true)}>
        <img src={src} alt={label || 'photo'} style={{ display: 'block', ...style }} />
        <div style={{
          position: 'absolute', inset: 0, borderRadius: style?.borderRadius || 6,
          background: 'rgba(0,0,0,0)', display: 'flex', alignItems: 'center', justifyContent: 'center',
          transition: 'background 0.2s',
          opacity: 0,
        }}
          onMouseEnter={e => { e.currentTarget.style.background='rgba(0,0,0,0.45)'; e.currentTarget.style.opacity=1 }}
          onMouseLeave={e => { e.currentTarget.style.background='rgba(0,0,0,0)'; e.currentTarget.style.opacity=0 }}>
          <ZoomIn size={22} color='#fff' />
        </div>
      </div>
      {open && <ImageLightbox src={src} label={label} onClose={() => setOpen(false)} />}
    </>
  )
}

function Stat({ label, value, color, highlight }) {
  return (
    <div style={{
      background: highlight ? 'rgba(245,158,11,0.08)' : 'var(--bg-card)',
      border: '1px solid ' + (highlight ? 'rgba(245,158,11,0.35)' : 'var(--border)'),
      borderRadius: 'var(--radius-lg)', padding: '14px 20px', flex: 1, minWidth: 130,
    }}>
      <div style={{ fontSize: '1.6rem', fontWeight: 800, color, letterSpacing: '-0.02em' }}>{value}</div>
      <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.07em', marginTop: 4 }}>{label}</div>
    </div>
  )
}

// --- Direction badge ---
function DirBadge({ direction }) {
  const isIn = direction === 'inward'
  return (
    <span style={{
      fontSize: '0.7rem', fontWeight: 700, padding: '2px 9px', borderRadius: 99,
      color:      isIn ? 'var(--accent-green)' : '#f97316',
      background: isIn ? 'rgba(16,185,129,0.12)' : 'rgba(249,115,22,0.12)',
      border:     '1px solid ' + (isIn ? 'rgba(16,185,129,0.3)' : 'rgba(249,115,22,0.3)'),
    }}>
      {isIn ? 'Inward' : 'Outward'}
    </span>
  )
}

// --- Status badge ---
function StatusBadge({ approved, ocr_available }) {
  if (!ocr_available) return (
    <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>OCR unavailable</span>
  )
  return approved
    ? <span style={{ fontSize: '0.7rem', color: 'var(--accent-green)', fontWeight: 700, display: 'inline-flex', alignItems: 'center', gap: 3 }}><CheckCircle size={11} />Approved</span>
    : <span style={{ fontSize: '0.7rem', color: '#ef4444', fontWeight: 700, display: 'inline-flex', alignItems: 'center', gap: 3 }}><XCircle size={11} />Rejected</span>
}

// --- Result banner ---
function ResultBanner({ result }) {
  if (!result) return null
  const isOutward = result.direction === 'outward'
  return (
    <div style={{
      padding: 16, borderRadius: 12, marginTop: 14,
      background: result.approved ? 'rgba(16,185,129,0.07)' : 'rgba(239,68,68,0.07)',
      border: '1px solid ' + (result.approved ? 'rgba(16,185,129,0.28)' : 'rgba(239,68,68,0.28)'),
    }}>
      <div style={{ fontWeight: 800, fontSize: '0.95rem', color: result.approved ? 'var(--accent-green)' : '#ef4444', marginBottom: 10 }}>
        {result.approved
          ? (isOutward ? 'Order Form Accepted - outward cleared' : 'Invoice Accepted - entry approved')
          : (isOutward ? 'Order Form Rejected - review required' : 'Invoice Rejected - review required')}
      </div>
      <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', alignItems: 'flex-start' }}>
        {result.snapshot_b64 && (
          <Thumb src={result.snapshot_b64} label='Invoice scan'
            style={{ width: 120, height: 90, objectFit: 'cover', borderRadius: 8,
              border: '2px solid ' + (result.approved ? 'rgba(16,185,129,0.5)' : 'rgba(239,68,68,0.5)') }} />
        )}
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginBottom: 4, textTransform: 'uppercase' }}>OCR Text Extracted</div>
          {result.ocr_available ? (
            <div style={{ fontSize: '0.76rem', color: 'var(--text-secondary)', fontFamily: 'monospace',
              maxHeight: 90, overflowY: 'auto', whiteSpace: 'pre-wrap', wordBreak: 'break-word',
              background: 'var(--bg-secondary)', borderRadius: 6, padding: '8px 10px', border: '1px solid var(--border)' }}>
              {result.raw_text || '(No text extracted)'}
            </div>
          ) : (
            <div style={{ fontSize: '0.76rem', color: '#f97316', fontStyle: 'italic' }}>OCR unavailable - install Tesseract</div>
          )}
          {result.raw_text && (
            <div style={{ marginTop: 6, fontSize: '0.68rem', color: result.approved ? 'var(--accent-green)' : '#ef4444' }}>
              {result.approved ? 'Document pattern matched' : 'Pattern not matched - admin review recommended'}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// --- QR Panel ---
// LAN IP for QR: use VITE_LAN_IP so the QR URL is reachable from phones on the same WiFi.
// Falls back to window.location.hostname (e.g. if opened from LAN directly).
const FRONTEND_BASE = (() => {
  const lanIp = import.meta.env.VITE_LAN_IP
  const port  = window.location.port ? ':' + window.location.port : ''
  if (lanIp) return window.location.protocol + '//' + lanIp + port
  return window.location.origin
})()

function QRPanel({ onScanned }) {
  const { addToast } = useToast()
  const canvasRef = useRef()
  const timerRef = useRef()
  const [direction, setDirection] = useState('inward')
  const [tokenData, setTokenData] = useState(null)
  const [generating, setGenerating] = useState(false)
  const [timeLeft, setTimeLeft] = useState(0)

  const generateQR = async () => {
    setGenerating(true)
    clearInterval(timerRef.current)
    try {
      const r = await documentsApi.qrToken(direction)
      const data = r.data
      setTokenData(data)
      setTimeLeft(data.expires_in_seconds)
      const uploadUrl = FRONTEND_BASE + '/upload-invoice?token=' + encodeURIComponent(data.token) + '&direction=' + direction
      await QRCode.toCanvas(canvasRef.current, uploadUrl, { width: 220, margin: 2, color: { dark: '#0f172a', light: '#ffffff' } })
      let remaining = data.expires_in_seconds
      timerRef.current = setInterval(() => {
        remaining -= 1
        setTimeLeft(remaining)
        if (remaining <= 0) {
          clearInterval(timerRef.current)
          setTokenData(null)
          addToast('QR Expired', 'Generate a new one', 'warning')
        }
      }, 1000)
    } catch { addToast('Failed to generate QR', '', 'danger') }
    finally { setGenerating(false) }
  }

  useEffect(() => () => clearInterval(timerRef.current), [])

  const copyUrl = () => {
    if (!tokenData) return
    const url = FRONTEND_BASE + '/upload-invoice?token=' + encodeURIComponent(tokenData.token) + '&direction=' + direction
    navigator.clipboard.writeText(url)
    addToast('URL copied', 'Send to worker phone', 'success')
  }

  const mins = Math.floor(timeLeft / 60)
  const secs = String(timeLeft % 60).padStart(2, '0')
  const urgentColor = timeLeft < 120 ? '#ef4444' : timeLeft < 300 ? '#f97316' : 'var(--accent-green)'

  return (
    <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap', alignItems: 'flex-start' }}>
      <div style={{ flex: 1, minWidth: 200 }}>
        <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.6 }}>
          Generate a QR code for the worker to scan with their phone. They photo the invoice and you approve here.
        </div>
        <div style={{ display: 'flex', gap: 6, marginBottom: 14 }}>
          {['inward', 'outward'].map(d => (
            <button key={d} onClick={() => { setDirection(d); setTokenData(null) }}
              className={'btn ' + (direction === d ? 'btn-primary' : 'btn-ghost')}
              style={{ fontSize: '0.76rem', padding: '4px 12px' }}>
              {d === 'inward' ? 'Inward' : 'Outward'}
            </button>
          ))}
        </div>
        <button className='btn btn-primary' style={{ gap: 7, width: '100%', justifyContent: 'center', padding: '10px 0' }}
          onClick={generateQR} disabled={generating}>
          {generating
            ? <><span className='spinner' /> Generating...</>
            : <><QrCode size={15} /> {tokenData ? 'Regenerate QR' : 'Generate QR Code'}</>}
        </button>
        {tokenData && (
          <div style={{ marginTop: 12, display: 'flex', flexDirection: 'column', gap: 8 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '8px 12px',
              borderRadius: 8, background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}>
              <Clock size={13} color={urgentColor} />
              <span style={{ fontSize: '0.8rem', color: urgentColor, fontWeight: 700, fontVariantNumeric: 'tabular-nums' }}>
                Expires in {mins}:{secs}
              </span>
            </div>
            <button className='btn btn-ghost' style={{ gap: 6, fontSize: '0.76rem' }} onClick={copyUrl}>
              <Copy size={12} /> Copy Upload URL
            </button>
            <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>
              Show QR to worker or send the URL to their phone on same WiFi.
            </div>
          </div>
        )}
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10 }}>
        <div style={{
          borderRadius: 16, overflow: 'hidden',
          border: '3px solid ' + (tokenData ? 'var(--accent-green)' : 'var(--border)'),
          background: '#fff',
          boxShadow: tokenData ? '0 0 20px rgba(16,185,129,0.2)' : 'none',
          transition: 'all 0.3s ease', position: 'relative',
        }}>
          <canvas ref={canvasRef} style={{ display: 'block' }} width={220} height={220} />
          {!tokenData && (
            <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column',
              alignItems: 'center', justifyContent: 'center', background: 'rgba(15,23,42,0.85)', gap: 8 }}>
              <QrCode size={32} color='var(--text-muted)' />
              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textAlign: 'center', maxWidth: 120 }}>
                Click Generate to create QR
              </span>
            </div>
          )}
        </div>
        {tokenData && (
          <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', textAlign: 'center', display: 'flex', alignItems: 'center', gap: 4 }}>
            <Smartphone size={11} /> Worker scans this on their phone
          </div>
        )}
      </div>
    </div>
  )
}

// --- Scan panel ---
function ScanPanel({ onScanned }) {
  const { addToast } = useToast()
  const [mode, setMode] = useState('qr')
  const [direction, setDirection] = useState('inward')
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const videoRef = useRef()
  const camCanvasRef = useRef()
  const streamRef = useRef(null)
  const [camActive, setCamActive] = useState(false)
  const [camError, setCamError] = useState('')
  const [captured, setCaptured] = useState(null)
  const fileRef = useRef()
  const [scanning, setScanning] = useState(false)
  const [result, setResult] = useState(null)

  const startCamera = async () => {
    setCamError(''); setCaptured(null); setResult(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: 'environment' } } })
      streamRef.current = stream
      if (videoRef.current) { videoRef.current.srcObject = stream; videoRef.current.play() }
      setCamActive(true)
    } catch (err) { setCamError('Camera access denied. ' + err.message) }
  }
  const stopCamera = () => {
    streamRef.current?.getTracks().forEach(t => t.stop())
    streamRef.current = null; setCamActive(false)
  }
  useEffect(() => { if (mode !== 'camera') stopCamera(); return () => stopCamera() }, [mode])

  const captureAndScan = async () => {
    const video = videoRef.current; const canvas = camCanvasRef.current
    if (!video || !canvas) return
    canvas.width = video.videoWidth; canvas.height = video.videoHeight
    canvas.getContext('2d').drawImage(video, 0, 0)
    const dataUrl = canvas.toDataURL('image/jpeg', 0.92)
    setCaptured(dataUrl); stopCamera()
    const res = await fetch(dataUrl); const blob = await res.blob()
    await runScan(new File([blob], 'capture.jpg', { type: 'image/jpeg' }))
  }

  const handleFile = e => {
    const f = e.target.files?.[0]; if (!f) return
    setFile(f); setResult(null)
    const reader = new FileReader()
    reader.onload = ev => setPreview(ev.target.result)
    reader.readAsDataURL(f)
  }

  const runScan = async (fileObj) => {
    setScanning(true); setResult(null)
    try {
      const form = new FormData()
      form.append('file', fileObj); form.append('direction', direction)
      const r = await api.post('/documents/scan', form, { headers: { 'Content-Type': 'multipart/form-data' } })
      setResult(r.data)
      if (r.data.approved) addToast('Invoice Accepted', '', 'success')
      else addToast('Invoice Rejected', 'OCR could not verify', 'warning')
      onScanned?.()
    } catch { addToast('Scan failed', 'Check backend and OCR setup', 'danger') }
    finally { setScanning(false) }
  }

  const modes = [
    { id: 'qr', label: 'QR Upload', icon: <QrCode size={12} /> },
    { id: 'upload', label: 'Upload', icon: <Upload size={12} /> },
    { id: 'camera', label: 'Camera', icon: <Camera size={12} /> },
  ]

  return (
    <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border)',
      borderRadius: 'var(--radius-lg)', padding: 22, marginBottom: 20 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16, flexWrap: 'wrap', gap: 8 }}>
        <div style={{ fontWeight: 700, fontSize: '0.88rem', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: 7 }}>
          <FileText size={15} /> Document Scan
        </div>
        <div style={{ display: 'flex', gap: 4, background: 'var(--bg-secondary)', borderRadius: 10, padding: 3 }}>
          {modes.map(m => (
            <button key={m.id} onClick={() => { setMode(m.id); setResult(null); setCaptured(null) }}
              style={{
                display: 'flex', alignItems: 'center', gap: 5,
                fontSize: '0.74rem', padding: '5px 12px', borderRadius: 7, border: 'none', cursor: 'pointer',
                background: mode === m.id ? 'var(--accent-blue)' : 'transparent',
                color: mode === m.id ? '#fff' : 'var(--text-muted)',
                fontWeight: mode === m.id ? 700 : 400, transition: 'all 0.18s',
              }}>
              {m.icon} {m.label}
            </button>
          ))}
        </div>
      </div>

      {mode === 'qr' && <QRPanel onScanned={onScanned} />}

      {mode === 'upload' && (
        <div>
          <div style={{ display: 'flex', gap: 6, marginBottom: 12 }}>
            {['inward', 'outward'].map(d => (
              <button key={d} onClick={() => setDirection(d)}
                className={'btn ' + (direction === d ? 'btn-primary' : 'btn-ghost')}
                style={{ fontSize: '0.76rem', padding: '4px 12px' }}>
                {d === 'inward' ? 'Inward' : 'Outward'}
              </button>
            ))}
          </div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
            <button className='btn btn-ghost' style={{ gap: 6, fontSize: '0.78rem' }} onClick={() => fileRef.current?.click()}>
              <Upload size={13} /> {file ? file.name : 'Choose Image'}
            </button>
            <input ref={fileRef} type='file' accept='image/*' style={{ display: 'none' }} onChange={handleFile} />
            <button className='btn btn-primary' style={{ gap: 6, fontSize: '0.78rem' }}
              onClick={() => { if (!file) { addToast('No file selected', '', 'warning'); return }; runScan(file) }}
              disabled={scanning || !file}>
              {scanning ? 'Scanning...' : <><FileText size={13} /> Run OCR</>}
            </button>
          </div>
          {preview && <img src={preview} alt='doc' style={{ marginTop: 12, maxWidth: 220, maxHeight: 150, objectFit: 'contain', borderRadius: 8, border: '1px solid var(--border)' }} />}
        </div>
      )}

      {mode === 'camera' && (
        <div>
          <div style={{ display: 'flex', gap: 6, marginBottom: 12 }}>
            {['inward', 'outward'].map(d => (
              <button key={d} onClick={() => setDirection(d)}
                className={'btn ' + (direction === d ? 'btn-primary' : 'btn-ghost')}
                style={{ fontSize: '0.76rem', padding: '4px 12px' }}>
                {d === 'inward' ? 'Inward' : 'Outward'}
              </button>
            ))}
          </div>
          {camError && <div style={{ color: '#ef4444', fontSize: '0.8rem', marginBottom: 8, display: 'flex', gap: 6, alignItems: 'center' }}><CameraOff size={14} /> {camError}</div>}
          {!captured && (
            <div style={{ position: 'relative', display: 'inline-block' }}>
              <video ref={videoRef} autoPlay playsInline muted
                style={{ display: camActive ? 'block' : 'none', maxWidth: '100%', width: 380, borderRadius: 10, border: '2px solid var(--accent-blue)', background: '#000' }} />
              <canvas ref={camCanvasRef} style={{ display: 'none' }} />
            </div>
          )}
          {captured && <img src={captured} alt='captured' style={{ maxWidth: 380, borderRadius: 10, border: '2px solid var(--accent-green)', marginBottom: 8 }} />}
          <div style={{ display: 'flex', gap: 8, marginTop: 10, flexWrap: 'wrap' }}>
            {!camActive && !captured && <button className='btn btn-primary' style={{ gap: 6, fontSize: '0.78rem' }} onClick={startCamera}><Camera size={13} /> Open Camera</button>}
            {camActive && (
              <>
                <button className='btn btn-primary' style={{ gap: 6, fontSize: '0.78rem', background: 'var(--accent-green)' }} onClick={captureAndScan} disabled={scanning}>
                  {scanning ? 'Scanning...' : <><Camera size={13} /> Capture and Scan</>}
                </button>
                <button className='btn btn-ghost' style={{ gap: 6, fontSize: '0.78rem', color: '#ef4444' }} onClick={stopCamera}><CameraOff size={13} /> Stop</button>
              </>
            )}
            {captured && !scanning && <button className='btn btn-ghost' style={{ fontSize: '0.78rem' }} onClick={() => { setCaptured(null); setResult(null); startCamera() }}>Retake</button>}
          </div>
        </div>
      )}
      {mode !== 'qr' && <ResultBanner result={result} />}
    </div>
  )
}

// --- Pending Approval Panel ---
function PendingPanel({ records, onAction }) {
  const { addToast } = useToast()
  const [expanded, setExpanded] = useState(null)

  const act = async (fn, msg) => {
    try { await fn(); addToast(msg, '', 'success'); onAction() }
    catch { addToast('Failed', '', 'danger') }
  }

  if (records.length === 0) return null

  return (
    <div style={{ background: 'rgba(245,158,11,0.05)', border: '1px solid rgba(245,158,11,0.3)',
      borderRadius: 'var(--radius-lg)', overflow: 'hidden', marginBottom: 20 }}>
      <div style={{ padding: '12px 18px', borderBottom: '1px solid rgba(245,158,11,0.2)', display: 'flex', alignItems: 'center', gap: 8 }}>
        <AlertCircle size={14} color='#f59e0b' />
        <span style={{ fontWeight: 700, fontSize: '0.85rem', color: '#f59e0b' }}>
          {records.length} Phone Upload{records.length > 1 ? 's' : ''} Awaiting Approval
        </span>
      </div>
      {records.map((r, i) => (
        <div key={r.table + '-' + r.id} style={{ padding: '12px 18px',
          borderBottom: i < records.length - 1 ? '1px solid rgba(245,158,11,0.12)' : 'none' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
            <Smartphone size={13} color='#f59e0b' />
            <DirBadge direction={r.direction} />
            <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
              {new Date(r.timestamp).toLocaleString('en-IN', { dateStyle: 'short', timeStyle: 'short' })}
            </span>
            <div style={{ marginLeft: 'auto', display: 'flex', gap: 6, alignItems: 'center' }}>
              <button className='btn btn-ghost' style={{ padding: '3px 8px' }}
                onClick={() => setExpanded(expanded === r.id ? null : r.id)}>
                {expanded === r.id ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
              </button>
              <button onClick={() => act(() => documentsApi.approve(r.id, r.table), 'Approved')}
                style={{ fontSize: '0.72rem', padding: '4px 12px', borderRadius: 7, border: 'none', cursor: 'pointer',
                  background: 'rgba(16,185,129,0.15)', color: 'var(--accent-green)', fontWeight: 700 }}>
                Approve
              </button>
              <button onClick={() => act(() => documentsApi.reject(r.id, r.table), 'Rejected')}
                style={{ fontSize: '0.72rem', padding: '4px 12px', borderRadius: 7, border: 'none', cursor: 'pointer',
                  background: 'rgba(239,68,68,0.12)', color: '#ef4444', fontWeight: 700 }}>
                Reject
              </button>
            </div>
          </div>
          {expanded === r.id && (
            <div style={{ marginTop: 10, display: 'flex', gap: 12, flexWrap: 'wrap' }}>
              {r.snapshot_b64 && <Thumb src={r.snapshot_b64} label='Invoice photo' style={{ maxWidth: 180, borderRadius: 8, border: '1px solid var(--border)', display: 'block' }} />}
              {r.person_snapshot_b64 && <Thumb src={r.person_snapshot_b64} label='Carrier snapshot' style={{ maxWidth: 180, borderRadius: 8, border: '1px solid var(--border)', display: 'block' }} />}
              <div style={{ flex: 1, fontSize: '0.76rem', fontFamily: 'monospace', whiteSpace: 'pre-wrap', wordBreak: 'break-word',
                background: 'var(--bg-secondary)', borderRadius: 8, padding: '8px 10px', border: '1px solid var(--border)',
                maxHeight: 120, overflowY: 'auto', color: 'var(--text-secondary)' }}>
                {r.raw_ocr_text || '(No OCR text extracted)'}
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  )
}

// --- Main page ---
export default function Documents() {
  const { addToast } = useToast()
  const [stats, setStats] = useState(null)
  const [records, setRecords] = useState([])
  const [pending, setPending] = useState([])
  const [loading, setLoading] = useState(true)
  const [direction, setDirection] = useState('')
  const [approved, setApproved] = useState('')
  const [expandedId, setExpandedId] = useState(null)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const params = {}
      if (direction) params.direction = direction
      if (approved !== '') params.approved = approved === 'true'
      const [sRes, rRes, pRes] = await Promise.all([
        documentsApi.stats(),
        documentsApi.list(params),
        documentsApi.pendingApproval(),
      ])
      setStats(sRes.data)
      setRecords(Array.isArray(rRes.data) ? rRes.data : [])
      setPending(Array.isArray(pRes.data?.records) ? pRes.data.records : [])
    } catch { addToast('Failed to load documents', '', 'danger') }
    finally { setLoading(false) }
  }, [direction, approved, addToast])

  useEffect(() => { load() }, [load])

  // ── Auto-poll: check pending every 10s so phone uploads appear automatically ──
  useEffect(() => {
    const poll = setInterval(async () => {
      try {
        const pRes = await documentsApi.pendingApproval()
        const fresh = Array.isArray(pRes.data?.records) ? pRes.data.records : []
        // Only trigger full reload when pending count changes (new upload arrived)
        setPending(prev => {
          if (fresh.length !== prev.length) {
            load()
            if (fresh.length > prev.length) addToast('New phone upload', 'A worker submitted an invoice', 'info')
          }
          return fresh
        })
      } catch { /* silent — don't spam toasts on poll failure */ }
    }, 10000)
    return () => clearInterval(poll)
  }, [load, addToast])

  const handleApprove = async (id, table) => {
    try { await documentsApi.approve(id, table); addToast('Approved', '', 'success'); load() }
    catch { addToast('Failed', '', 'danger') }
  }
  const handleReject = async (id, table) => {
    try { await documentsApi.reject(id, table); addToast('Rejected', '', 'success'); load() }
    catch { addToast('Failed', '', 'danger') }
  }
  const handleDelete = async (id, table) => {
    if (!window.confirm('Permanently delete this record?')) return
    try { await documentsApi.delete(id, table); addToast('Deleted', '', 'success'); load() }
    catch { addToast('Delete failed', '', 'danger') }
  }

  return (
    <div className='page-container'>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 24, flexWrap: 'wrap', gap: 12 }}>
        <div>
          <h1 style={{ fontSize: '1.4rem', fontWeight: 800, color: 'var(--text-primary)', margin: 0, display: 'flex', alignItems: 'center', gap: 9 }}>
            <FileText size={20} />
            Document Scan Log
            {pending.length > 0 && (
              <span style={{ fontSize: '0.7rem', fontWeight: 800, padding: '2px 9px', borderRadius: 99, background: '#f59e0b', color: '#000' }}>
                {pending.length} pending
              </span>
            )}
          </h1>
          <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: 3 }}>
            Invoice and order-form OCR gate - QR phone upload, manual scan, inward and outward
          </div>
        </div>
        <button className='btn btn-ghost' onClick={load} style={{ gap: 5 }}><RefreshCw size={14} /> Refresh</button>
      </div>

      {stats && (
        <div style={{ display: 'flex', gap: 10, marginBottom: 20, flexWrap: 'wrap' }}>
          <Stat label='Today Total' value={stats.today_total}    color='var(--text-primary)' />
          <Stat label='Inward'      value={stats.today_inward}   color='var(--accent-green)' />
          <Stat label='Outward'     value={stats.today_outward}  color='#f97316' />
          <Stat label='Approved'    value={stats.today_approved} color='var(--accent-green)' />
          <Stat label='Rejected'    value={stats.today_rejected} color='#ef4444' />
          {pending.length > 0 && <Stat label='Pending' value={pending.length} color='#f59e0b' highlight />}
        </div>
      )}

      <PendingPanel records={pending} onAction={load} />
      <ScanPanel onScanned={load} />

      <div style={{ display: 'flex', gap: 8, marginBottom: 14, flexWrap: 'wrap', alignItems: 'center' }}>
        <Filter size={14} color='var(--text-muted)' />
        {['', 'inward', 'outward'].map(d => (
          <button key={d || 'all'} onClick={() => setDirection(d)}
            className={'btn ' + (direction === d ? 'btn-primary' : 'btn-ghost')}
            style={{ fontSize: '0.76rem', padding: '4px 12px' }}>
            {d || 'All'}
          </button>
        ))}
        <div style={{ width: 1, height: 22, background: 'var(--border)', margin: '0 4px' }} />
        {[['', 'All Status'], ['true', 'Approved'], ['false', 'Rejected']].map(([v, label]) => (
          <button key={v || 'all-status'} onClick={() => setApproved(v)}
            className={'btn ' + (approved === v ? 'btn-primary' : 'btn-ghost')}
            style={{ fontSize: '0.76rem', padding: '4px 12px' }}>
            {label}
          </button>
        ))}
      </div>

      <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 'var(--radius-lg)', overflow: 'hidden' }}>
        {loading ? (
          <div style={{ padding: 32, textAlign: 'center', color: 'var(--text-muted)' }}><span className='spinner' style={{ marginRight: 8 }} />Loading...</div>
        ) : records.length === 0 ? (
          <div style={{ padding: 40, textAlign: 'center', color: 'var(--text-muted)' }}>No documents found for current filters.</div>
        ) : (
          <div>
            {records.map((r, i) => (
              <div key={r.table + '-' + r.id} style={{ padding: '12px 18px',
                borderBottom: i < records.length - 1 ? '1px solid var(--border)' : 'none',
                background: i % 2 === 0 ? 'transparent' : 'var(--bg-secondary)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 9, flexWrap: 'wrap' }}>
                  <DirBadge direction={r.direction} />
                  <StatusBadge approved={r.approved} ocr_available={r.ocr_available} />
                  {r.submitted_by_phone && (
                    <span style={{ fontSize: '0.68rem', fontWeight: 700, padding: '1px 7px', borderRadius: 99,
                      background: 'rgba(245,158,11,0.12)', color: '#f59e0b', border: '1px solid rgba(245,158,11,0.3)',
                      display: 'inline-flex', alignItems: 'center', gap: 3 }}>
                      <Smartphone size={10} /> Phone
                    </span>
                  )}
                  {r.goods_count != null && (
                    <span style={{ fontSize: '0.72rem', fontWeight: 600, padding: '2px 8px', borderRadius: 6,
                      background: 'rgba(59,130,246,0.15)', color: 'var(--accent-blue)', border: '1px solid rgba(59,130,246,0.3)' }}>
                      Qty: {r.goods_count}
                    </span>
                  )}
                  <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginLeft: 'auto' }}>
                    {new Date(r.timestamp).toLocaleString('en-IN', { dateStyle: 'short', timeStyle: 'short' })}
                  </span>
                  <button className='btn btn-ghost btn-icon' onClick={() => setExpandedId(expandedId === r.id ? null : r.id)} style={{ padding: 4 }}>
                    {expandedId === r.id ? <EyeOff size={14} /> : <Eye size={14} />}
                  </button>
                  {!r.approved && <button className='btn btn-ghost' style={{ fontSize: '0.7rem', padding: '2px 8px', color: 'var(--accent-green)' }} onClick={() => handleApprove(r.id, r.table)}>Approve</button>}
                  {r.approved && <button className='btn btn-ghost' style={{ fontSize: '0.7rem', padding: '2px 8px', color: '#ef4444' }} onClick={() => handleReject(r.id, r.table)}>Reject</button>}
                  <button className='btn btn-ghost btn-icon' title='Delete' onClick={() => handleDelete(r.id, r.table)} style={{ padding: 4, color: '#ef4444', marginLeft: 2 }}><Trash2 size={13} /></button>
                </div>
                {expandedId === r.id && (
                  <div style={{ marginTop: 10, padding: '10px 14px', background: 'var(--bg-secondary)', borderRadius: 8, fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                    <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 10 }}>
                      {r.snapshot_b64 && <div><div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: 4 }}>Document Scan</div>
                        <Thumb src={r.snapshot_b64} label='Document scan'
                          style={{ maxWidth: 180, maxHeight: 130, objectFit: 'cover', borderRadius: 6, border: '1px solid var(--border)', display: 'block' }} /></div>}
                      {r.person_snapshot_b64 && <div><div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: 4 }}>Carrier Snapshot</div>
                        <Thumb src={r.person_snapshot_b64} label='Carrier snapshot'
                          style={{ maxWidth: 180, maxHeight: 130, objectFit: 'cover', borderRadius: 6, border: '1px solid var(--border)', display: 'block' }} /></div>}
                    </div>
                    {r.goods_count != null && <div style={{ marginBottom: 6, fontWeight: 600, color: 'var(--accent-blue)' }}>Parsed Goods Quantity: {r.goods_count} units</div>}
                    <div style={{ fontFamily: 'monospace', whiteSpace: 'pre-wrap', wordBreak: 'break-word', maxHeight: 120, overflowY: 'auto' }}>{r.raw_ocr_text || '(No OCR text)'}</div>
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
