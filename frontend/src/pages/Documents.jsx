import { useState, useEffect, useCallback, useRef } from 'react'
import QRCode from 'qrcode'
import api from '../api/api'
import { useToast } from '../context/ToastContext'
import {
  FileText, Upload, CheckCircle2, XCircle, RefreshCw,
  Filter, Eye, EyeOff, Trash2, Camera, CameraOff,
  QrCode, Smartphone, Clock, AlertCircle, ChevronDown, ChevronUp,
  Copy, ZoomIn, X, Printer, Globe, Wifi, Settings2, ExternalLink,
  Truck, Package, Scale, Building, ShieldCheck, Search, Sparkles
} from 'lucide-react'

// --- API helpers ---
const documentsApi = {
  stats:              ()           => api.get('/documents/stats'),
  list:               (params)     => api.get('/documents/', { params }),
  approve:            (id, table)  => api.patch('/documents/' + id + '/approve?table=' + table),
  reject:             (id, table)  => api.patch('/documents/' + id + '/reject?table=' + table),
  delete:             (id, table)  => api.delete('/documents/' + id + '?table=' + table),
  qrToken:            (direction)  => api.get('/documents/qr-token', { params: { direction } }),
  pendingApproval:    ()           => api.get('/documents/pending-approval'),
  fixedQrConfig:      ()           => api.get('/documents/fixed-qr-config'),
  updateFixedQrConfig:(url)        => api.post('/documents/fixed-qr-config', { public_gate_url: url }),
}

// --- Unambiguous Indian Standard Time (IST) Date/Time Formatter ---
export function formatDocumentTimestamp(ts) {
  if (!ts) return '—'
  try {
    const raw = String(ts).trim()
    const iso = raw.endsWith('Z') || raw.includes('+') ? raw : raw + 'Z'
    const d = new Date(iso)
    if (isNaN(d.getTime())) return String(ts)
    return d.toLocaleString('en-IN', {
      timeZone: 'Asia/Kolkata',
      day: '2-digit',
      month: 'short',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: true,
      timeZoneName: 'short',
    })
  } catch {
    return String(ts)
  }
}

// --- Gentle Web Audio Chime for Gate Notifications ---
function playNotificationSound() {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)()
    const osc = ctx.createOscillator()
    const gain = ctx.createGain()
    osc.type = 'sine'
    osc.frequency.setValueAtTime(587.33, ctx.currentTime) // D5
    osc.frequency.exponentialRampToValueAtTime(880, ctx.currentTime + 0.15) // A5
    gain.gain.setValueAtTime(0.2, ctx.currentTime)
    gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.35)
    osc.connect(gain)
    gain.connect(ctx.destination)
    osc.start()
    osc.stop(ctx.currentTime + 0.36)
  } catch (e) {
    // AudioContext blocked by browser policy until user interacts
  }
}

// --- Image Lightbox Modal ---
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
      background: 'rgba(0,0,0,0.88)', backdropFilter: 'blur(8px)',
      display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
      padding: 24, cursor: 'zoom-out',
      animation: 'lb-fade 0.2s ease',
    }}>
      <style>{`
        @keyframes lb-fade { from{opacity:0;transform:scale(0.96)} to{opacity:1;transform:scale(1)} }
        .lb-img { max-width:min(92vw,950px); max-height:82vh; object-fit:contain;
          border-radius:14px; box-shadow:0 30px 90px rgba(0,0,0,0.8);
          animation:lb-fade 0.22s cubic-bezier(.22,1,.36,1); }
      `}</style>
      <div onClick={e => e.stopPropagation()} style={{ position: 'relative', display: 'inline-block' }}>
        <img src={src} alt={label || 'Document Photo'} className='lb-img' />
        <button onClick={onClose} style={{
          position: 'absolute', top: -14, right: -14, width: 34, height: 34,
          borderRadius: '50%', border: 'none', cursor: 'pointer',
          background: '#ef4444', color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center',
          boxShadow: '0 4px 14px rgba(0,0,0,0.4)', zIndex: 1,
        }}><X size={18} /></button>
        {label && (
          <div style={{ position: 'absolute', bottom: -36, left: 0, right: 0, textAlign: 'center',
            fontSize: '0.85rem', color: 'rgba(255,255,255,0.7)', letterSpacing: '0.04em' }}>{label}</div>
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
          position: 'absolute', inset: 0, borderRadius: style?.borderRadius || 8,
          background: 'rgba(0,0,0,0)', display: 'flex', alignItems: 'center', justifyContent: 'center',
          transition: 'background 0.2s', opacity: 0,
        }}
          onMouseEnter={e => { e.currentTarget.style.background='rgba(0,0,0,0.45)'; e.currentTarget.style.opacity=1 }}
          onMouseLeave={e => { e.currentTarget.style.background='rgba(0,0,0,0)'; e.currentTarget.style.opacity=0 }}>
          <ZoomIn size={20} color='#fff' />
        </div>
      </div>
      {open && <ImageLightbox src={src} label={label} onClose={() => setOpen(false)} />}
    </>
  )
}

function Stat({ label, value, color, highlight, icon: Icon }) {
  return (
    <div style={{
      background: highlight ? 'rgba(245,158,11,0.08)' : 'var(--bg-card)',
      border: '1px solid ' + (highlight ? 'rgba(245,158,11,0.35)' : 'var(--border)'),
      borderRadius: 'var(--radius-lg)', padding: '16px 20px', flex: 1, minWidth: 140,
      display: 'flex', flexDirection: 'column', justifyContent: 'space-between',
      position: 'relative', overflow: 'hidden',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.07em', fontWeight: 600 }}>
          {label}
        </div>
        {Icon && <Icon size={16} color={color || 'var(--text-muted)'} style={{ opacity: 0.7 }} />}
      </div>
      <div style={{ fontSize: '1.8rem', fontWeight: 800, color, letterSpacing: '-0.02em', marginTop: 6 }}>
        {value}
      </div>
    </div>
  )
}

// --- Direction badge ---
function DirBadge({ direction }) {
  const isIn = direction === 'inward'
  return (
    <span style={{
      fontSize: '0.72rem', fontWeight: 700, padding: '3px 10px', borderRadius: 99,
      color:      isIn ? 'var(--accent-green)' : '#f97316',
      background: isIn ? 'rgba(16,185,129,0.12)' : 'rgba(249,115,22,0.12)',
      border:     '1px solid ' + (isIn ? 'rgba(16,185,129,0.3)' : 'rgba(249,115,22,0.3)'),
      display:    'inline-flex', alignItems: 'center', gap: 4,
    }}>
      <span style={{ width: 6, height: 6, borderRadius: '50%', background: isIn ? 'var(--accent-green)' : '#f97316' }} />
      {isIn ? 'INWARD' : 'OUTWARD'}
    </span>
  )
}

// --- Status badge ---
function StatusBadge({ approved, status, reject_reason, ocr_available }) {
  if (status === 'auto_rejected') {
    return (
      <span
        title={reject_reason || 'Auto-rejected: Non-document photo'}
        style={{
          fontSize: '0.72rem', color: '#ef4444', fontWeight: 800,
          display: 'inline-flex', alignItems: 'center', gap: 4,
          background: 'rgba(239,68,68,0.15)', border: '1px solid rgba(239,68,68,0.3)',
          padding: '3px 8px', borderRadius: 6,
        }}
      >
        <XCircle size={12} /> Auto-Rejected
      </span>
    )
  }
  if (status === 'rejected') {
    return (
      <span
        title={reject_reason || 'Rejected by Admin'}
        style={{
          fontSize: '0.72rem', color: '#ef4444', fontWeight: 700,
          display: 'inline-flex', alignItems: 'center', gap: 4,
          background: 'rgba(239,68,68,0.12)', padding: '3px 8px', borderRadius: 6,
        }}
      >
        <XCircle size={12} /> Rejected
      </span>
    )
  }
  if (!ocr_available) return (
    <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontStyle: 'italic', padding: '2px 8px', borderRadius: 6, background: 'var(--bg-secondary)' }}>
      OCR unverified
    </span>
  )
  return approved || status === 'approved'
    ? <span style={{ fontSize: '0.72rem', color: 'var(--accent-green)', fontWeight: 700, display: 'inline-flex', alignItems: 'center', gap: 4, background: 'rgba(16,185,129,0.12)', padding: '3px 8px', borderRadius: 6 }}>
        <CheckCircle2 size={12} /> Approved
      </span>
    : <span style={{ fontSize: '0.72rem', color: '#f59e0b', fontWeight: 700, display: 'inline-flex', alignItems: 'center', gap: 4, background: 'rgba(245,158,11,0.12)', padding: '3px 8px', borderRadius: 6 }}>
        <AlertCircle size={12} /> Pending Approval
      </span>
}

// --- Fixed Printable QR Poster Modal ---
function PrintPosterModal({ qrUrl, onClose }) {
  const posterCanvasRef = useRef(null)

  useEffect(() => {
    if (posterCanvasRef.current && qrUrl) {
      QRCode.toCanvas(posterCanvasRef.current, qrUrl, {
        width: 320,
        margin: 1,
        color: { dark: '#0a0f1d', light: '#ffffff' },
        errorCorrectionLevel: 'H',
      })
    }
  }, [qrUrl])

  const triggerPrint = () => {
    window.print()
  }

  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 10000,
      background: 'rgba(0,0,0,0.85)', backdropFilter: 'blur(8px)',
      display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 20,
    }}>
      <style>{`
        @media print {
          body * { visibility: hidden; }
          #gate-printable-poster, #gate-printable-poster * { visibility: visible; }
          #gate-printable-poster {
            position: absolute; left: 0; top: 0; width: 100%;
            padding: 40px; margin: 0; box-shadow: none !important; border: 2px solid #000 !important;
            background: #fff !important; color: #000 !important;
          }
          .no-print { display: none !important; }
        }
      `}</style>

      <div style={{
        background: 'var(--bg-card)', borderRadius: 20, border: '1px solid var(--border)',
        maxWidth: 580, width: '100%', maxHeight: '92vh', overflowY: 'auto', padding: 24,
        boxShadow: '0 25px 70px rgba(0,0,0,0.6)',
      }}>
        <div className="no-print" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 18 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <Printer size={18} color="var(--accent-blue)" />
            <span style={{ fontWeight: 800, fontSize: '1.1rem' }}>Print Official Gate Placard</span>
          </div>
          <button onClick={onClose} className="btn btn-ghost btn-icon"><X size={16} /></button>
        </div>

        {/* The Actual Printable Poster Sheet */}
        <div id="gate-printable-poster" style={{
          background: '#ffffff', color: '#090d16', borderRadius: 16, padding: '36px 30px',
          textAlign: 'center', border: '3px solid #0f172a', boxShadow: '0 10px 30px rgba(0,0,0,0.15)',
        }}>
          {/* Header */}
          <div style={{ borderBottom: '2px solid #e2e8f0', paddingBottom: 16, marginBottom: 20 }}>
            <div style={{ fontSize: '0.8rem', fontWeight: 800, letterSpacing: '0.14em', color: '#475569', textTransform: 'uppercase', marginBottom: 4 }}>
              OccuSafe Security & Logistics Systems
            </div>
            <div style={{ fontSize: '1.75rem', fontWeight: 900, color: '#0f172a', letterSpacing: '-0.03em', textTransform: 'uppercase' }}>
              Vendor & Driver Gate Pass
            </div>
            <div style={{ fontSize: '0.95rem', color: '#334155', fontWeight: 600, marginTop: 4 }}>
              Scan with Smartphone Camera to Submit Inward / Outward Paperwork
            </div>
          </div>

          {/* QR Code with Frame */}
          <div style={{
            display: 'inline-block', padding: 14, background: '#ffffff',
            borderRadius: 16, border: '4px solid #0f172a', marginBottom: 20,
            boxShadow: '0 8px 24px rgba(0,0,0,0.08)',
          }}>
            <canvas ref={posterCanvasRef} style={{ display: 'block' }} width={320} height={320} />
          </div>

          {/* Step-by-Step Instructions */}
          <div style={{
            background: '#f8fafc', borderRadius: 12, padding: '16px 20px',
            border: '1px solid #cbd5e1', textAlign: 'left', marginBottom: 20,
          }}>
            <div style={{ fontSize: '0.85rem', fontWeight: 800, color: '#0f172a', textTransform: 'uppercase', marginBottom: 8, letterSpacing: '0.04em' }}>
              How to Register at Gate:
            </div>
            <ol style={{ margin: 0, paddingLeft: 20, fontSize: '0.85rem', color: '#334155', lineHeight: 1.6, fontWeight: 500 }}>
              <li><strong>Scan QR code</strong> with your phone camera (no app install needed).</li>
              <li>Select <strong>Inward (Material Delivery)</strong> or <strong>Outward (Finished Goods)</strong>.</li>
              <li>Snap a clear photo of the <strong>Invoice / Order Form</strong> and driver photo.</li>
              <li>Receive instant <strong>Digital Gate Pass</strong> on your phone to show the guard.</li>
            </ol>
          </div>

          <div style={{ fontSize: '0.75rem', color: '#64748b', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Works over 4G/5G Cellular or Local Wi-Fi • Secure On-Premise Audit
          </div>
        </div>

        {/* Action Buttons */}
        <div className="no-print" style={{ display: 'flex', gap: 10, marginTop: 20, justifyContent: 'flex-end' }}>
          <button onClick={onClose} className="btn btn-ghost">Cancel</button>
          <button onClick={triggerPrint} className="btn btn-primary" style={{ gap: 8 }}>
            <Printer size={16} />
            <span>Print A4 Poster Now</span>
          </button>
        </div>
      </div>
    </div>
  )
}

// --- Fixed QR Manager Panel ---
function FixedQRManager() {
  const { addToast } = useToast()
  const canvasRef = useRef(null)
  const [config, setConfig] = useState(null)
  const [networkMode, setNetworkMode] = useState('public') // 'public' | 'lan' | 'custom'
  const [customUrl, setCustomUrl] = useState('')
  const [savingUrl, setSavingUrl] = useState(false)
  const [showPoster, setShowPoster] = useState(false)

  const loadConfig = useCallback(async () => {
    try {
      const res = await documentsApi.fixedQrConfig()
      setConfig(res.data)
      if (res.data.public_gate_url) {
        setCustomUrl(res.data.public_gate_url)
      }
    } catch {
      // Fallback if endpoint fails
    }
  }, [])

  useEffect(() => { loadConfig() }, [loadConfig])

  // Resolve final QR Target URL based on mode
  const effectiveQrUrl = (() => {
    const fixedKey = config?.fixed_key || 'occusafe-gate-fixed'
    if (networkMode === 'public') {
      const publicBase = config?.public_gate_url || config?.ngrok_url
      if (publicBase) {
        return `${publicBase.replace(/\/$/, '')}/upload-invoice?gate_key=${fixedKey}`
      }
      if (window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
        return `${window.location.origin}/upload-invoice?gate_key=${fixedKey}`
      }
      if (config?.lan_url) {
        return `${config.lan_url.replace(/\/$/, '')}/upload-invoice?gate_key=${fixedKey}`
      }
      return `${window.location.origin}/upload-invoice?gate_key=${fixedKey}`
    }
    if (networkMode === 'lan') {
      const lanBase = config?.lan_url || (config?.lan_ip ? `http://${config.lan_ip}:5173` : '') || window.location.origin
      return `${lanBase.replace(/\/$/, '')}/upload-invoice?gate_key=${fixedKey}`
    }
    // custom
    const base = customUrl.trim().replace(/\/$/, '') || config?.ngrok_url || window.location.origin
    return `${base}/upload-invoice?gate_key=${fixedKey}`
  })()

  // Render QR Canvas whenever effective URL changes
  useEffect(() => {
    if (canvasRef.current && effectiveQrUrl) {
      QRCode.toCanvas(canvasRef.current, effectiveQrUrl, {
        width: 200,
        margin: 2,
        color: { dark: '#0b1120', light: '#ffffff' },
      })
    }
  }, [effectiveQrUrl])

  const copyUrl = () => {
    navigator.clipboard.writeText(effectiveQrUrl)
    addToast('URL Copied', 'Paste into vendor message or mobile browser', 'success')
  }

  const saveCustomUrl = async () => {
    if (!customUrl.trim()) return
    setSavingUrl(true)
    try {
      await documentsApi.updateFixedQrConfig(customUrl.trim())
      addToast('Public Gateway Saved', 'Fixed QR code updated for any-network access', 'success')
      await loadConfig()
    } catch {
      addToast('Save failed', 'Could not update gateway URL', 'danger')
    } finally {
      setSavingUrl(false)
    }
  }

  const isLocalhost = effectiveQrUrl.includes('localhost') || effectiveQrUrl.includes('127.0.0.1')

  return (
    <div style={{
      background: 'linear-gradient(135deg, rgba(15,23,42,0.95), rgba(30,41,59,0.7))',
      border: '1px solid rgba(59,130,246,0.3)', borderRadius: 'var(--radius-lg)',
      padding: '22px 24px', marginBottom: 24, boxShadow: '0 12px 35px rgba(0,0,0,0.3)',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16, flexWrap: 'wrap', gap: 10 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{
            width: 38, height: 38, borderRadius: 10, background: 'rgba(59,130,246,0.15)',
            border: '1px solid rgba(59,130,246,0.35)', display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <QrCode size={20} color="#60a5fa" />
          </div>
          <div>
            <h3 style={{ margin: 0, fontSize: '1.05rem', fontWeight: 800, color: '#f8fafc', display: 'flex', alignItems: 'center', gap: 8 }}>
              Fixed Dock QR Code (Permanent Printout)
              <span style={{ fontSize: '0.65rem', fontWeight: 700, padding: '2px 8px', borderRadius: 99, background: 'rgba(16,185,129,0.15)', color: '#10b981', border: '1px solid rgba(16,185,129,0.3)' }}>
                NO EXPIRATION
              </span>
            </h3>
            <div style={{ fontSize: '0.75rem', color: '#94a3b8', marginTop: 2 }}>
              Print and mount at entrance gate. Vendors scan over any 4G/5G mobile connection or Wi-Fi to submit paperwork.
            </div>
          </div>
        </div>

        <button
          onClick={() => setShowPoster(true)}
          className="btn btn-primary"
          style={{ gap: 8, padding: '9px 18px', fontSize: '0.85rem' }}
        >
          <Printer size={15} />
          <span>Print Gate Placard</span>
        </button>
      </div>

      <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap', alignItems: 'flex-start' }}>
        {/* Network Target Selector */}
        <div style={{ flex: 1, minWidth: 260 }}>
          <div style={{ fontSize: '0.75rem', fontWeight: 700, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 8 }}>
            Select Network Routing Target:
          </div>

          <div style={{ display: 'flex', gap: 8, marginBottom: 14 }}>
            <button
              onClick={() => setNetworkMode('public')}
              className={'btn ' + (networkMode === 'public' ? 'btn-primary' : 'btn-ghost')}
              style={{ fontSize: '0.76rem', padding: '6px 14px', gap: 6 }}
            >
              <Globe size={13} />
              <span>Public Internet (Any 4G/5G)</span>
            </button>
            <button
              onClick={() => setNetworkMode('lan')}
              className={'btn ' + (networkMode === 'lan' ? 'btn-primary' : 'btn-ghost')}
              style={{ fontSize: '0.76rem', padding: '6px 14px', gap: 6 }}
            >
              <Wifi size={13} />
              <span>Local Wi-Fi Only</span>
            </button>
            <button
              onClick={() => setNetworkMode('custom')}
              className={'btn ' + (networkMode === 'custom' ? 'btn-primary' : 'btn-ghost')}
              style={{ fontSize: '0.76rem', padding: '6px 14px', gap: 6 }}
            >
              <Settings2 size={13} />
              <span>Custom Gateway</span>
            </button>
          </div>

          {networkMode === 'public' && (
            <div style={{ background: 'rgba(15,23,42,0.6)', padding: '12px 14px', borderRadius: 10, border: '1px solid #1e293b', marginBottom: 12 }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6, flexWrap: 'wrap', gap: 6 }}>
                <div style={{ fontSize: '0.74rem', color: '#cbd5e1', fontWeight: 600 }}>
                  <strong>Active Public Gateway (Any 4G/5G):</strong>
                </div>
                {config?.ngrok_url ? (
                  <span style={{ fontSize: '0.65rem', fontWeight: 700, padding: '2px 8px', borderRadius: 99, background: 'rgba(16,185,129,0.15)', color: '#10b981', border: '1px solid rgba(16,185,129,0.3)' }}>
                    🟢 LIVE NGROK TUNNEL
                  </span>
                ) : (
                  <span style={{ fontSize: '0.65rem', fontWeight: 700, padding: '2px 8px', borderRadius: 99, background: 'rgba(59,130,246,0.15)', color: '#60a5fa', border: '1px solid rgba(59,130,246,0.3)' }}>
                    ENTERPRISE GATEWAY
                  </span>
                )}
              </div>
              <div style={{ fontFamily: 'monospace', fontSize: '0.82rem', color: '#60a5fa', wordBreak: 'break-all', fontWeight: 700 }}>
                {config?.public_gate_url || config?.ngrok_url || (window.location.hostname !== 'localhost' ? window.location.origin : config?.lan_url)}
              </div>
              <div style={{ fontSize: '0.7rem', color: '#94a3b8', marginTop: 4 }}>
                {config?.ngrok_url
                  ? 'Auto-detected active ngrok tunnel. Vendors on any cellular network (Jio, Airtel, Vodafone) can scan and upload.'
                  : 'Allows vendors and drivers over any mobile network to upload paperwork anywhere.'}
              </div>
            </div>
          )}

          {networkMode === 'lan' && (
            <div style={{ background: 'rgba(15,23,42,0.6)', padding: '12px 14px', borderRadius: 10, border: '1px solid #1e293b', marginBottom: 12 }}>
              <div style={{ fontSize: '0.74rem', color: '#cbd5e1', marginBottom: 6, fontWeight: 600 }}>
                <strong>Local Facility Wi-Fi Address:</strong>
              </div>
              <div style={{ fontFamily: 'monospace', fontSize: '0.82rem', color: '#38bdf8', wordBreak: 'break-all', fontWeight: 700 }}>
                {config?.lan_url || `http://${window.location.hostname}:5173`}
              </div>
              <div style={{ fontSize: '0.7rem', color: '#94a3b8', marginTop: 4 }}>
                Requires vendor/driver mobile phone to be connected to the facility local Wi-Fi router.
              </div>
            </div>
          )}

          {isLocalhost && (
            <div style={{ background: 'rgba(239,68,68,0.12)', border: '1px solid rgba(239,68,68,0.3)', padding: '8px 12px', borderRadius: 8, marginBottom: 10, fontSize: '0.72rem', color: '#fca5a5', display: 'flex', alignItems: 'center', gap: 6 }}>
              <AlertCircle size={14} color="#ef4444" />
              <span>Warning: Host browser is on localhost. Mobile cannot resolve localhost. Switch to Public Tunnel or Local Wi-Fi above.</span>
            </div>
          )}

          {networkMode === 'custom' && (
            <div style={{ marginBottom: 12 }}>
              <label style={{ display: 'block', fontSize: '0.72rem', color: '#94a3b8', marginBottom: 4 }}>
                Enter Public Tunnel / ngrok / Cloudflare URL:
              </label>
              <div style={{ display: 'flex', gap: 6 }}>
                <input
                  type="text"
                  placeholder="https://your-tunnel.ngrok-free.app"
                  value={customUrl}
                  onChange={e => setCustomUrl(e.target.value)}
                  style={{
                    flex: 1, padding: '8px 12px', borderRadius: 8, background: '#090d16',
                    border: '1px solid #23334d', color: '#f8fafc', fontSize: '0.8rem',
                  }}
                />
                <button
                  onClick={saveCustomUrl}
                  disabled={savingUrl || !customUrl}
                  className="btn btn-primary"
                  style={{ fontSize: '0.76rem', padding: '6px 12px' }}
                >
                  {savingUrl ? 'Saving...' : 'Save'}
                </button>
              </div>
            </div>
          )}

          <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginTop: 10 }}>
            <button onClick={copyUrl} className="btn btn-ghost" style={{ gap: 6, fontSize: '0.76rem' }}>
              <Copy size={13} /> Copy Portal URL
            </button>
            <a
              href={effectiveQrUrl}
              target="_blank"
              rel="noreferrer"
              className="btn btn-ghost"
              style={{ gap: 6, fontSize: '0.76rem', textDecoration: 'none' }}
            >
              <ExternalLink size={13} /> Test Portal
            </a>
          </div>
        </div>

        {/* QR Code Canvas */}
        <div style={{
          display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8,
          background: '#ffffff', borderRadius: 16, padding: 12, border: '3px solid #3b82f6',
          boxShadow: '0 10px 25px rgba(0,0,0,0.3)',
        }}>
          <canvas ref={canvasRef} style={{ display: 'block' }} width={200} height={200} />
          <div style={{ fontSize: '0.68rem', fontWeight: 800, color: '#090d16', textAlign: 'center', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            SCAN TO UPLOAD DOCUMENT
          </div>
        </div>
      </div>

      {showPoster && (
        <PrintPosterModal qrUrl={effectiveQrUrl} onClose={() => setShowPoster(false)} />
      )}
    </div>
  )
}

// --- Manual OCR Scan & Camera Drawer ---
function ManualScanPanel({ onScanned }) {
  const { addToast } = useToast()
  const [direction, setDirection] = useState('inward')
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [vendorName, setVendorName] = useState('')
  const [vehicleNo, setVehicleNo] = useState('')
  const [goodsCount, setGoodsCount] = useState('')
  const [weight, setWeight] = useState('')
  const [docNumber, setDocNumber] = useState('')
  const [scanning, setScanning] = useState(false)
  const [result, setResult] = useState(null)
  const fileRef = useRef(null)

  const handleFile = e => {
    const f = e.target.files?.[0]
    if (!f) return
    setFile(f); setResult(null)
    const reader = new FileReader()
    reader.onload = ev => setPreview(ev.target.result)
    reader.readAsDataURL(f)
  }

  const runScan = async () => {
    if (!file) {
      addToast('Document Photo Required', 'Please choose or photograph the document', 'warning')
      return
    }
    if (!vendorName.trim() || !vehicleNo.trim() || !goodsCount || !weight.trim() || !docNumber.trim()) {
      addToast('Mandatory Fields Required', 'Vendor Name, Vehicle No, Qty, Weight, and Invoice # are all compulsory', 'warning')
      return
    }
    setScanning(true); setResult(null)
    try {
      const form = new FormData()
      form.append('file', file)
      form.append('direction', direction)
      form.append('vendor_name', vendorName.trim())
      form.append('vehicle_no', vehicleNo.trim().toUpperCase())
      form.append('goods_count', goodsCount)
      form.append('weight', weight.trim())
      form.append('doc_number', docNumber.trim())

      const r = await api.post('/documents/scan', form, { headers: { 'Content-Type': 'multipart/form-data' } })
      setResult(r.data)
      if (r.data.approved) {
        addToast('Document Approved', 'OCR pattern matched successfully', 'success')
      } else if (r.data.auto_rejected) {
        addToast('Document Auto-Rejected', r.data.reject_reason || 'Non-document photo detected', 'danger')
      } else {
        addToast('Document Logged', 'Admin review recommended', 'warning')
      }
      onScanned?.()
    } catch (err) {
      const msg = err.response?.data?.detail || 'Check backend OCR service'
      addToast('Scan failed', msg, 'danger')
    } finally {
      setScanning(false)
    }
  }

  return (
    <div style={{
      background: 'var(--bg-card)', border: '1px solid var(--border)',
      borderRadius: 'var(--radius-lg)', padding: '20px 22px', marginBottom: 24,
    }}>
      <div style={{ fontWeight: 800, fontSize: '0.92rem', marginBottom: 14, display: 'flex', alignItems: 'center', gap: 8 }}>
        <FileText size={16} color="var(--accent-blue)" />
        <span>Manual Desk Scan & Upload</span>
      </div>

      <div style={{ display: 'flex', gap: 6, marginBottom: 14 }}>
        {['inward', 'outward'].map(d => (
          <button
            key={d}
            onClick={() => setDirection(d)}
            className={'btn ' + (direction === d ? 'btn-primary' : 'btn-ghost')}
            style={{ fontSize: '0.78rem', padding: '5px 14px' }}
          >
            {d === 'inward' ? 'Inward (Delivery)' : 'Outward (Dispatch)'}
          </button>
        ))}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))', gap: 10, marginBottom: 14 }}>
        <input
          type="text"
          placeholder="Vendor / Supplier * (Compulsory)"
          value={vendorName}
          onChange={e => setVendorName(e.target.value)}
          className="form-control"
          style={{ padding: '8px 12px', fontSize: '0.8rem' }}
        />
        <input
          type="text"
          placeholder="Vehicle Number * (Compulsory)"
          value={vehicleNo}
          onChange={e => setVehicleNo(e.target.value.toUpperCase())}
          className="form-control"
          style={{ padding: '8px 12px', fontSize: '0.8rem', textTransform: 'uppercase' }}
        />
        <input
          type="number"
          min="1"
          placeholder="Goods Count * (Compulsory)"
          value={goodsCount}
          onChange={e => setGoodsCount(e.target.value)}
          className="form-control"
          style={{ padding: '8px 12px', fontSize: '0.8rem' }}
        />
        <input
          type="text"
          placeholder="Weight (e.g. 50 kg) * (Compulsory)"
          value={weight}
          onChange={e => setWeight(e.target.value)}
          className="form-control"
          style={{ padding: '8px 12px', fontSize: '0.8rem' }}
        />
        <input
          type="text"
          placeholder="Invoice / Doc Ref # * (Compulsory)"
          value={docNumber}
          onChange={e => setDocNumber(e.target.value)}
          className="form-control"
          style={{ padding: '8px 12px', fontSize: '0.8rem' }}
        />
      </div>

      <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
        <button className="btn btn-ghost" onClick={() => fileRef.current?.click()} style={{ gap: 6, fontSize: '0.8rem' }}>
          <Upload size={14} /> {file ? file.name : 'Choose Document Image *'}
        </button>
        <input ref={fileRef} type="file" accept="image/*" style={{ display: 'none' }} onChange={handleFile} />

        <button
          onClick={runScan}
          disabled={scanning || !file}
          className="btn btn-primary"
          style={{ gap: 6, fontSize: '0.8rem' }}
        >
          {scanning ? 'Running OCR...' : <><Sparkles size={14} /> Process & Save</>}
        </button>
      </div>

      {preview && (
        <div style={{ marginTop: 14 }}>
          <img src={preview} alt="preview" style={{ maxWidth: 220, maxHeight: 140, objectFit: 'contain', borderRadius: 8, border: '1px solid var(--border)' }} />
        </div>
      )}

      {result && (
        <div style={{
          marginTop: 14, padding: 14, borderRadius: 10,
          background: result.approved ? 'rgba(16,185,129,0.08)' : 'rgba(245,158,11,0.08)',
          border: '1px solid ' + (result.approved ? 'rgba(16,185,129,0.3)' : 'rgba(245,158,11,0.3)'),
        }}>
          <div style={{ fontWeight: 800, fontSize: '0.88rem', color: result.approved ? 'var(--accent-green)' : '#f59e0b', marginBottom: 6 }}>
            {result.approved ? '✅ Document Verified & Logged' : '⚠️ Logged for Gate Inspection'}
          </div>
          <div style={{ fontFamily: 'monospace', fontSize: '0.75rem', color: 'var(--text-secondary)', maxHeight: 80, overflowY: 'auto' }}>
            {result.raw_text || '(No OCR text extracted)'}
          </div>
        </div>
      )}
    </div>
  )
}

// --- Pending Approval Queue Banner ---
function PendingApprovalQueue({ records, onAction }) {
  const { addToast } = useToast()
  const [expanded, setExpanded] = useState(null)

  const act = async (fn, msg) => {
    try {
      await fn()
      addToast(msg, '', 'success')
      onAction()
    } catch {
      addToast('Failed to update status', '', 'danger')
    }
  }

  if (records.length === 0) return null

  return (
    <div style={{
      background: 'rgba(245,158,11,0.06)', border: '1px solid rgba(245,158,11,0.35)',
      borderRadius: 'var(--radius-lg)', overflow: 'hidden', marginBottom: 24,
    }}>
      <div style={{
        padding: '14px 20px', borderBottom: '1px solid rgba(245,158,11,0.2)',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{
            width: 26, height: 26, borderRadius: '50%', background: '#f59e0b', color: '#090d16',
            display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 900, fontSize: '0.82rem',
            boxShadow: '0 0 10px rgba(245,158,11,0.5)',
          }}>
            {records.length}
          </div>
          <span style={{ fontWeight: 800, fontSize: '0.94rem', color: '#f59e0b' }}>
            🔔 {records.length === 1 ? '1 new invoice added for approval' : `${records.length} new invoices added for approval`}
          </span>
        </div>
        <span style={{ fontSize: '0.72rem', color: '#f59e0b', fontWeight: 700, padding: '3px 8px', borderRadius: 6, background: 'rgba(245,158,11,0.15)', border: '1px solid rgba(245,158,11,0.3)' }}>
          ACTION REQUIRED
        </span>
      </div>

      {records.map((r, i) => (
        <div key={r.table + '-' + r.id} style={{
          padding: '14px 20px',
          borderBottom: i < records.length - 1 ? '1px solid rgba(245,158,11,0.12)' : 'none',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
            <Smartphone size={15} color="#f59e0b" />
            <DirBadge direction={r.direction} />
            <strong style={{ fontSize: '0.82rem', color: 'var(--text-primary)' }}>
              {r.vendor_name || 'Vendor / Driver'}
            </strong>
            {r.vehicle_no && (
              <span style={{ fontSize: '0.72rem', background: '#0f172a', padding: '2px 8px', borderRadius: 6, border: '1px solid var(--border)' }}>
                {r.vehicle_no}
              </span>
            )}
            {r.goods_count != null && (
              <span style={{ fontSize: '0.72rem', color: '#60a5fa' }}>
                Qty: {r.goods_count} pcs
              </span>
            )}
            {r.weight && (
              <span style={{ fontSize: '0.72rem', color: '#10b981' }}>
                Wt: {r.weight}
              </span>
            )}
            <span style={{ fontSize: '0.74rem', color: 'var(--text-muted)', marginLeft: 'auto', whiteSpace: 'nowrap' }}>
              {formatDocumentTimestamp(r.timestamp)}
            </span>

            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              <button
                className="btn btn-ghost btn-icon"
                onClick={() => setExpanded(expanded === r.id ? null : r.id)}
                title="Toggle Photo Preview"
              >
                {expanded === r.id ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
              </button>
              <button
                onClick={() => act(() => documentsApi.approve(r.id, r.table), 'Gate Pass Approved')}
                style={{
                  fontSize: '0.74rem', padding: '5px 14px', borderRadius: 8, border: 'none',
                  cursor: 'pointer', background: 'rgba(16,185,129,0.18)', color: 'var(--accent-green)', fontWeight: 800,
                }}
              >
                Approve Entry
              </button>
              <button
                onClick={() => act(() => documentsApi.reject(r.id, r.table), 'Gate Pass Rejected')}
                style={{
                  fontSize: '0.74rem', padding: '5px 14px', borderRadius: 8, border: 'none',
                  cursor: 'pointer', background: 'rgba(239,68,68,0.15)', color: '#ef4444', fontWeight: 800,
                }}
              >
                Reject
              </button>
            </div>
          </div>

          {expanded === r.id && (
            <div style={{ marginTop: 12, display: 'flex', gap: 14, flexWrap: 'wrap', background: 'var(--bg-secondary)', padding: 12, borderRadius: 10 }}>
              {r.snapshot_b64 && (
                <div>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: 4 }}>Document Photo</div>
                  <Thumb src={r.snapshot_b64} label="Document photo" style={{ maxWidth: 200, maxHeight: 150, borderRadius: 8, border: '1px solid var(--border)' }} />
                </div>
              )}
              {r.person_snapshot_b64 && (
                <div>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: 4 }}>Driver / Carrier Photo</div>
                  <Thumb src={r.person_snapshot_b64} label="Driver photo" style={{ maxWidth: 200, maxHeight: 150, borderRadius: 8, border: '1px solid var(--border)' }} />
                </div>
              )}
              <div style={{ flex: 1, minWidth: 200 }}>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: 4 }}>OCR Extracted Content:</div>
                <div style={{
                  fontFamily: 'monospace', fontSize: '0.75rem', color: 'var(--text-secondary)',
                  maxHeight: 120, overflowY: 'auto', whiteSpace: 'pre-wrap', lineHeight: 1.5,
                  background: 'var(--bg-card)', padding: 8, borderRadius: 6, border: '1px solid var(--border)'
                }}>
                  {r.raw_ocr_text || '(No OCR text parsed)'}
                </div>
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  )
}

// --- Main Documents Page Component ---
export default function Documents() {
  const { addToast } = useToast()
  const [stats, setStats] = useState(null)
  const [records, setRecords] = useState([])
  const [pending, setPending] = useState([])
  const [loading, setLoading] = useState(true)
  const [direction, setDirection] = useState('')
  const [approved, setApproved] = useState('')
  const [searchQuery, setSearchQuery] = useState('')
  const [expandedId, setExpandedId] = useState(null)

  const loadData = useCallback(async () => {
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
    } catch {
      addToast('Failed to load documents', '', 'danger')
    } finally {
      setLoading(false)
    }
  }, [direction, approved, addToast])

  useEffect(() => { loadData() }, [loadData])

  // Real-time auto-polling: refresh every 5 seconds for immediate gate notifications
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const pRes = await documentsApi.pendingApproval()
        const fresh = Array.isArray(pRes.data?.records) ? pRes.data.records : []
        setPending(prev => {
          if (fresh.length !== prev.length) {
            loadData()
            if (fresh.length > prev.length) {
              playNotificationSound()
              const latest = fresh[0]
              addToast(
                fresh.length === 1 ? '📢 1 New Invoice Added for Approval' : `📢 ${fresh.length} New Invoices Added for Approval`,
                `${latest.direction === 'inward' ? 'Inward' : 'Outward'} from ${latest.vendor_name || 'Driver'} (${latest.vehicle_no || 'At Gate'})`,
                'warning'
              )
            }
          }
          return fresh
        })
      } catch {
        // silent fail on poll
      }
    }, 5000)
    return () => clearInterval(interval)
  }, [loadData, addToast])

  const handleApprove = async (id, table) => {
    try {
      await documentsApi.approve(id, table)
      addToast('Document Approved', '', 'success')
      loadData()
    } catch {
      addToast('Approval failed', '', 'danger')
    }
  }

  const handleReject = async (id, table) => {
    try {
      await documentsApi.reject(id, table)
      addToast('Document Rejected', '', 'success')
      loadData()
    } catch {
      addToast('Rejection failed', '', 'danger')
    }
  }

  const handleDelete = async (id, table) => {
    if (!window.confirm('Permanently delete this gate record?')) return
    try {
      await documentsApi.delete(id, table)
      addToast('Record Deleted', '', 'success')
      loadData()
    } catch {
      addToast('Delete failed', '', 'danger')
    }
  }

  // Filter records by search
  const filteredRecords = records.filter(r => {
    if (!searchQuery) return true
    const q = searchQuery.toLowerCase()
    return (
      (r.vendor_name && r.vendor_name.toLowerCase().includes(q)) ||
      (r.vehicle_no && r.vehicle_no.toLowerCase().includes(q)) ||
      (r.doc_number && r.doc_number.toLowerCase().includes(q)) ||
      (r.raw_ocr_text && r.raw_ocr_text.toLowerCase().includes(q))
    )
  })

  return (
    <div className="page-container">
      {/* Page Title & Actions */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 24, flexWrap: 'wrap', gap: 14 }}>
        <div>
          <h1 style={{ fontSize: '1.45rem', fontWeight: 900, color: 'var(--text-primary)', margin: 0, display: 'flex', alignItems: 'center', gap: 10 }}>
            <FileText size={22} color="var(--accent-blue)" />
            Document Clearance & OCR Gate
            {pending.length > 0 && (
              <span style={{
                fontSize: '0.74rem', fontWeight: 800, padding: '3px 10px', borderRadius: 99,
                background: '#f59e0b', color: '#000',
              }}>
                {pending.length} pending
              </span>
            )}
          </h1>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: 4 }}>
            Inward/Outward invoice & order form verification. Powered by permanent QR dock scan & on-premise OCR.
          </div>
        </div>

        <button className="btn btn-ghost" onClick={loadData} style={{ gap: 6, fontSize: '0.82rem' }}>
          <RefreshCw size={14} /> Refresh
        </button>
      </div>

      {/* KPI Stats Bar */}
      {stats && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 12, marginBottom: 24 }}>
          <Stat label="Today Total"  value={stats.today_total}    color="var(--text-primary)" icon={FileText} />
          <Stat label="Inward"       value={stats.today_inward}   color="var(--accent-green)" icon={Truck} />
          <Stat label="Outward"      value={stats.today_outward}  color="#f97316" icon={Package} />
          <Stat label="Approved"     value={stats.today_approved} color="var(--accent-green)" icon={CheckCircle2} />
          <Stat label="Rejected"     value={stats.today_rejected} color="#ef4444" icon={XCircle} />
          {pending.length > 0 && <Stat label="Pending Queue" value={pending.length} color="#f59e0b" highlight icon={AlertCircle} />}
        </div>
      )}

      {/* Pending Review Queue (Phone Uploads) */}
      <PendingApprovalQueue records={pending} onAction={loadData} />

      {/* Fixed QR Manager & Printable Placard Feature */}
      <FixedQRManager />

      {/* Manual OCR Scan Drawer */}
      <ManualScanPanel onScanned={loadData} />

      {/* Filter and Search Bar */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        marginBottom: 16, flexWrap: 'wrap', gap: 12,
      }}>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
          <Filter size={14} color="var(--text-muted)" />
          {['', 'inward', 'outward'].map(d => (
            <button
              key={d || 'all'}
              onClick={() => setDirection(d)}
              className={'btn ' + (direction === d ? 'btn-primary' : 'btn-ghost')}
              style={{ fontSize: '0.76rem', padding: '4px 12px' }}
            >
              {d ? (d === 'inward' ? 'Inward' : 'Outward') : 'All Types'}
            </button>
          ))}

          <div style={{ width: 1, height: 20, background: 'var(--border)', margin: '0 4px' }} />

          {[['', 'All Status'], ['true', 'Approved'], ['false', 'Needs Review']].map(([v, label]) => (
            <button
              key={v || 'all-status'}
              onClick={() => setApproved(v)}
              className={'btn ' + (approved === v ? 'btn-primary' : 'btn-ghost')}
              style={{ fontSize: '0.76rem', padding: '4px 12px' }}
            >
              {label}
            </button>
          ))}
        </div>

        {/* Search Input */}
        <div style={{ position: 'relative', width: '100%', maxWidth: 260 }}>
          <Search size={14} color="var(--text-muted)" style={{ position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)' }} />
          <input
            type="text"
            placeholder="Search vendor, vehicle, invoice..."
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            style={{
              width: '100%', padding: '7px 12px 7px 32px', borderRadius: 8,
              background: 'var(--bg-card)', border: '1px solid var(--border)',
              color: 'var(--text-primary)', fontSize: '0.78rem', boxSizing: 'border-box',
            }}
          />
        </div>
      </div>

      {/* Document Records Table */}
      <div style={{
        background: 'var(--bg-card)', border: '1px solid var(--border)',
        borderRadius: 'var(--radius-lg)', overflow: 'hidden',
      }}>
        {loading ? (
          <div style={{ padding: 40, textAlign: 'center', color: 'var(--text-muted)' }}>
            <span className="spinner" style={{ marginRight: 8 }} />
            Loading gate records...
          </div>
        ) : filteredRecords.length === 0 ? (
          <div style={{ padding: 48, textAlign: 'center', color: 'var(--text-muted)' }}>
            <FileText size={32} style={{ opacity: 0.3, marginBottom: 8 }} />
            <div>No gate document records match your current filters.</div>
          </div>
        ) : (
          <div>
            {filteredRecords.map((r, i) => (
              <div
                key={r.table + '-' + r.id}
                style={{
                  padding: '14px 20px',
                  borderBottom: i < filteredRecords.length - 1 ? '1px solid var(--border)' : 'none',
                  background: i % 2 === 0 ? 'transparent' : 'var(--bg-secondary)',
                  transition: 'background 0.2s',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
                  {/* Gate Ref Code */}
                  <span style={{
                    fontFamily: 'monospace', fontSize: '0.74rem', fontWeight: 800, color: '#60a5fa',
                    background: 'rgba(59,130,246,0.1)', padding: '3px 8px', borderRadius: 6,
                  }}>
                    {r.direction === 'inward' ? `GP-IN-${r.id}` : `GP-OUT-${r.id}`}
                  </span>

                  <DirBadge direction={r.direction} />
                  <StatusBadge approved={r.approved} status={r.status} reject_reason={r.reject_reason} ocr_available={r.ocr_available} />

                  {r.submitted_by_phone && (
                    <span style={{
                      fontSize: '0.68rem', fontWeight: 700, padding: '2px 8px', borderRadius: 99,
                      background: 'rgba(245,158,11,0.12)', color: '#f59e0b', border: '1px solid rgba(245,158,11,0.3)',
                      display: 'inline-flex', alignItems: 'center', gap: 4,
                    }}>
                      <Smartphone size={11} /> Phone Scan
                    </span>
                  )}

                  {/* Vendor Name */}
                  {r.vendor_name && (
                    <strong style={{ fontSize: '0.82rem', color: 'var(--text-primary)', display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                      <Building size={12} color="var(--text-muted)" />
                      {r.vendor_name}
                    </strong>
                  )}

                  {/* Vehicle Number Badge */}
                  {r.vehicle_no && (
                    <span style={{
                      fontSize: '0.72rem', fontWeight: 800, padding: '2px 8px', borderRadius: 6,
                      background: '#0a0f1d', color: '#f8fafc', border: '1px solid #23334d',
                      display: 'inline-flex', alignItems: 'center', gap: 4, letterSpacing: '0.04em',
                    }}>
                      <Truck size={11} color="#60a5fa" />
                      {r.vehicle_no}
                    </span>
                  )}

                  {/* Goods Count & Weight */}
                  {(r.goods_count != null || r.weight) && (
                    <span style={{
                      fontSize: '0.72rem', fontWeight: 600, padding: '2px 8px', borderRadius: 6,
                      background: 'rgba(59,130,246,0.12)', color: 'var(--accent-blue)',
                    }}>
                      {r.goods_count != null ? `${r.goods_count} pcs` : ''}
                      {r.goods_count != null && r.weight ? ' • ' : ''}
                      {r.weight || ''}
                    </span>
                  )}

                  {/* Timestamp */}
                  <span style={{ fontSize: '0.74rem', color: 'var(--text-muted)', marginLeft: 'auto', whiteSpace: 'nowrap' }}>
                    {formatDocumentTimestamp(r.timestamp)}
                  </span>

                  {/* Actions */}
                  <button
                    className="btn btn-ghost btn-icon"
                    onClick={() => setExpandedId(expandedId === r.id ? null : r.id)}
                    style={{ padding: 4 }}
                    title="View Document Details & Photos"
                  >
                    {expandedId === r.id ? <EyeOff size={15} /> : <Eye size={15} />}
                  </button>

                  {!r.approved ? (
                    <button
                      className="btn btn-ghost"
                      style={{ fontSize: '0.72rem', padding: '3px 10px', color: 'var(--accent-green)' }}
                      onClick={() => handleApprove(r.id, r.table)}
                    >
                      Approve
                    </button>
                  ) : (
                    <button
                      className="btn btn-ghost"
                      style={{ fontSize: '0.72rem', padding: '3px 10px', color: '#ef4444' }}
                      onClick={() => handleReject(r.id, r.table)}
                    >
                      Reject
                    </button>
                  )}

                  <button
                    className="btn btn-ghost btn-icon"
                    title="Delete Record"
                    onClick={() => handleDelete(r.id, r.table)}
                    style={{ padding: 4, color: '#ef4444' }}
                  >
                    <Trash2 size={14} />
                  </button>
                </div>

                {/* Expanded Details Drawer with Dual Snapshots */}
                {expandedId === r.id && (
                  <div style={{
                    marginTop: 14, padding: '14px 18px', background: 'var(--bg-secondary)',
                    borderRadius: 10, fontSize: '0.78rem', color: 'var(--text-secondary)',
                  }}>
                    <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 12 }}>
                      {r.snapshot_b64 && (
                        <div>
                          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: 4, fontWeight: 700 }}>
                            📄 Document Photo
                          </div>
                          <Thumb
                            src={r.snapshot_b64}
                            label="Document scan photo"
                            style={{ maxWidth: 220, maxHeight: 160, objectFit: 'cover', borderRadius: 8, border: '1px solid var(--border)' }}
                          />
                        </div>
                      )}

                      {r.person_snapshot_b64 && (
                        <div>
                          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: 4, fontWeight: 700 }}>
                            👤 Driver / Carrier Photo
                          </div>
                          <Thumb
                            src={r.person_snapshot_b64}
                            label="Carrier verification snapshot"
                            style={{ maxWidth: 220, maxHeight: 160, objectFit: 'cover', borderRadius: 8, border: '1px solid var(--border)' }}
                          />
                        </div>
                      )}

                      <div style={{ flex: 1, minWidth: 220 }}>
                        <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: 4, fontWeight: 700 }}>
                          Gate & Delivery Details
                        </div>
                        <div style={{ background: 'var(--bg-card)', padding: '10px 14px', borderRadius: 8, border: '1px solid var(--border)' }}>
                          <div><strong>Vendor / Party:</strong> {r.vendor_name || 'Not provided'}</div>
                          <div style={{ marginTop: 3 }}><strong>Vehicle Number:</strong> {r.vehicle_no || 'Not provided'}</div>
                          <div style={{ marginTop: 3 }}><strong>Document Ref:</strong> {r.doc_number || 'Auto-scanned'}</div>
                          <div style={{ marginTop: 3 }}><strong>Declared Goods:</strong> {r.goods_count ? `${r.goods_count} pcs` : 'Unspecified'}</div>
                          <div style={{ marginTop: 3 }}><strong>Weight:</strong> {r.weight || 'Unspecified'}</div>
                          <div style={{ marginTop: 3 }}><strong>Submitted At:</strong> {formatDocumentTimestamp(r.timestamp)}</div>
                          {r.reject_reason && (
                            <div style={{ marginTop: 6, padding: '6px 10px', borderRadius: 6, background: 'rgba(239,68,68,0.12)', border: '1px solid rgba(239,68,68,0.25)', color: '#ef4444', fontSize: '0.72rem' }}>
                              <strong>⛔ Rejection Reason:</strong> {r.reject_reason}
                            </div>
                          )}
                        </div>
                      </div>
                    </div>

                    <div>
                      <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: 4, fontWeight: 700 }}>
                        OCR Extracted Text
                      </div>
                      <div style={{
                        fontFamily: 'monospace', fontSize: '0.75rem', whiteSpace: 'pre-wrap',
                        wordBreak: 'break-word', maxHeight: 120, overflowY: 'auto',
                        background: 'var(--bg-card)', padding: '10px 12px', borderRadius: 8,
                        border: '1px solid var(--border)', color: '#cbd5e1', lineHeight: 1.5,
                      }}>
                        {r.raw_ocr_text || '(No OCR text parsed)'}
                      </div>
                    </div>
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
