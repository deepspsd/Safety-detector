import { BrowserRouter, Routes, Route, Navigate, NavLink } from 'react-router-dom'
import { AuthProvider, useAuth } from './context/AuthContext'
import { ToastProvider } from './context/ToastContext'
import { ThemeProvider } from './context/ThemeContext'
import Sidebar from './components/Sidebar'
import PWAInstallPrompt from './components/PWAInstallPrompt'
import Login from './pages/Login'
import Signup from './pages/Signup'
import Profile from './pages/Profile'
import Dashboard from './pages/Dashboard'
import LiveMonitor from './pages/LiveMonitor'
import Alerts from './pages/Alerts'
import AlertsReview from './pages/AlertsReview'
import Settings from './pages/Settings'
import SetupRules from './pages/SetupRules'
import FloorOverview from './pages/FloorOverview'
import Cameras from './pages/Cameras'
import Employees from './pages/Employees'
import Attendance from './pages/Attendance'
import Documents from './pages/Documents'
import WorkflowMonitor from './pages/WorkflowMonitor'
import Diagnostics from './pages/Diagnostics'
import { useState, useRef } from 'react'
import { Layers, Bell, LayoutDashboard, Camera, Upload, CheckCircle, XCircle } from 'lucide-react'


// -- Mobile Invoice Upload Page (public, accessed via QR code) --
function MobileUpload() {
  const params = new URLSearchParams(window.location.search)
  const token  = params.get('token') || ''
  const dir    = params.get('direction') || 'inward'

  const [file,    setFile]    = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result,  setResult]  = useState(null)
  const [err,     setErr]     = useState('')
  const fileRef = useRef()

  const handleFile = e => {
    const f = e.target.files && e.target.files[0]; if (!f) return
    setFile(f); setResult(null); setErr('')
    const reader = new FileReader()
    reader.onload = ev => setPreview(ev.target.result)
    reader.readAsDataURL(f)
  }

  const submit = async () => {
    if (!file) { setErr('Please select an image first'); return }
    setLoading(true); setErr('')
    try {
      const form = new FormData()
      form.append('token', token)
      form.append('direction', dir)
      form.append('file', file)
      const backendUrl = import.meta.env.VITE_API_BASE_URL || window.location.origin.replace(':5173', ':8000')
      const res = await fetch(backendUrl + '/documents/mobile-upload', { method: 'POST', body: form })

      if (res.status === 401) { setErr('QR session invalid - ask admin to generate a new QR'); return }
      if (res.status === 410) { setErr('QR code expired - ask admin to generate a new one'); return }
      if (!res.ok) { setErr('Upload failed (status ' + res.status + ')'); return }
      setResult(await res.json())
    } catch(e) { setErr('Network error - make sure you are on the same WiFi') }
    finally { setLoading(false) }
  }

  const isIn = dir === 'inward'

  return (
    <div style={{ minHeight: '100vh', background: '#0f172a', display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '32px 20px', fontFamily: 'system-ui, sans-serif' }}>
      <div style={{ width: '100%', maxWidth: 400, marginBottom: 24, textAlign: 'center' }}>
        <div style={{ fontSize: '0.7rem', letterSpacing: '0.12em', color: '#64748b', textTransform: 'uppercase', marginBottom: 4 }}>OccuSafe Factory Monitor</div>
        <h1 style={{ margin: 0, fontSize: '1.4rem', fontWeight: 800, color: '#f8fafc' }}>Invoice Upload</h1>
        <div style={{ marginTop: 8 }}>
          <span style={{ display: 'inline-block', fontSize: '0.75rem', fontWeight: 700, padding: '3px 12px', borderRadius: 99,
            background: isIn ? 'rgba(16,185,129,0.15)' : 'rgba(249,115,22,0.15)',
            color: isIn ? '#10b981' : '#f97316',
            border: '1px solid ' + (isIn ? 'rgba(16,185,129,0.35)' : 'rgba(249,115,22,0.35)') }}>
            {isIn ? 'INWARD ENTRY' : 'OUTWARD EXIT'}
          </span>
        </div>
      </div>
      <div style={{ width: '100%', maxWidth: 400, background: '#1e293b', borderRadius: 20, padding: 24, border: '1px solid #334155' }}>
        {!result ? (
          <>
            <div style={{ fontSize: '0.82rem', color: '#94a3b8', marginBottom: 18, lineHeight: 1.6 }}>
              Take a clear photo of the {isIn ? 'invoice' : 'order form'} and upload it below.
            </div>
            {preview ? (
              <div style={{ marginBottom: 16, position: 'relative' }}>
                <img src={preview} alt="preview" style={{ width: '100%', borderRadius: 12, border: '2px solid #3b82f6', maxHeight: 260, objectFit: 'contain', background: '#0f172a' }} />
                <button onClick={() => { setFile(null); setPreview(null) }}
                  style={{ position: 'absolute', top: 8, right: 8, background: 'rgba(239,68,68,0.85)', border: 'none', borderRadius: 99, color: '#fff', fontSize: '0.75rem', padding: '3px 10px', cursor: 'pointer' }}>
                  Remove
                </button>
              </div>
            ) : (
              <div onClick={() => fileRef.current && fileRef.current.click()}
                style={{ border: '2px dashed #334155', borderRadius: 14, padding: '36px 20px', textAlign: 'center', cursor: 'pointer', background: '#0f172a', marginBottom: 16 }}>
                <Camera size={32} color="#475569" style={{ marginBottom: 10 }} />
                <div style={{ fontSize: '0.9rem', color: '#94a3b8', fontWeight: 600, marginBottom: 4 }}>Tap to take photo</div>
                <div style={{ fontSize: '0.75rem', color: '#64748b' }}>or choose from gallery</div>
              </div>
            )}
            <input ref={fileRef} type="file" accept="image/*" capture="environment" style={{ display: 'none' }} onChange={handleFile} />
            {err && (
              <div style={{ background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)', borderRadius: 10, padding: '10px 14px', marginBottom: 14, fontSize: '0.82rem', color: '#ef4444', lineHeight: 1.5 }}>
                {err}
              </div>
            )}
            <button onClick={submit} disabled={loading || !file}
              style={{ width: '100%', padding: '14px 0', borderRadius: 12, border: 'none',
                cursor: (file && !loading) ? 'pointer' : 'not-allowed',
                background: (file && !loading) ? 'linear-gradient(135deg,#3b82f6,#2563eb)' : '#1e293b',
                color: (file && !loading) ? '#fff' : '#475569',
                fontWeight: 700, fontSize: '1rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}>
              {loading
                ? <span>Uploading...</span>
                : <><Upload size={16} /> Submit Invoice</>}
            </button>
          </>
        ) : (
          <div style={{ textAlign: 'center', position: 'relative', overflow: 'visible' }}>
            <style>{`
              @keyframes mu-popIn  { 0%{transform:scale(0) rotate(-15deg);opacity:0} 70%{transform:scale(1.18) rotate(3deg)} 100%{transform:scale(1) rotate(0deg);opacity:1} }
              @keyframes mu-fadeUp { from{opacity:0;transform:translateY(18px)} to{opacity:1;transform:translateY(0)} }
              @keyframes mu-burst  { 0%{transform:scale(0);opacity:1} 100%{transform:scale(2.8);opacity:0} }
              @keyframes mu-float  { 0%,100%{transform:translateY(0px)} 50%{transform:translateY(-7px)} }
              @keyframes mu-ring   { 0%{transform:translateX(-50%) scale(0.5);opacity:0.9} 100%{transform:translateX(-50%) scale(2.2);opacity:0} }
            `}</style>
            {result.approved ? (
              <div style={{ position: 'relative', display: 'inline-block', marginBottom: 24 }}>
                <div style={{ width: 84, height: 84, borderRadius: '50%',
                  background: 'radial-gradient(circle, rgba(16,185,129,0.2) 0%, rgba(16,185,129,0.05) 100%)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  animation: 'mu-popIn 0.55s cubic-bezier(.22,1,.36,1) both' }}>
                  <CheckCircle size={46} color="#10b981" strokeWidth={2.5} />
                </div>
                {/* burst ring */}
                <div style={{ position:'absolute', width: 84, height: 84, borderRadius:'50%',
                  border: '3px solid #10b981', left:'50%', top:0,
                  animation: 'mu-ring 0.7s ease-out 0.1s both', pointerEvents:'none' }} />
                {/* confetti dots */}
                {['#10b981','#3b82f6','#f59e0b','#ec4899','#8b5cf6','#f97316'].map((c,i) => (
                  <div key={i} style={{
                    position:'absolute', width:8, height:8, borderRadius:'50%', background:c,
                    top:'42px', left:'42px',
                    animation: `mu-burst 0.7s ease-out ${0.1+i*0.06}s both`,
                    transformOrigin: `${Math.cos(i*60*Math.PI/180)*34}px ${Math.sin(i*60*Math.PI/180)*34}px`
                  }} />
                ))}
              </div>
            ) : (
              <div style={{ width: 84, height: 84, borderRadius: '50%',
                background: 'rgba(239,68,68,0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center',
                margin: '0 auto 24px', animation: 'mu-popIn 0.5s cubic-bezier(.22,1,.36,1) both' }}>
                <XCircle size={46} color="#ef4444" strokeWidth={2.5} />
              </div>
            )}
            <div style={{ fontSize: '1.35rem', fontWeight: 800, color: result.approved ? '#10b981' : '#ef4444',
              marginBottom: 8, animation: 'mu-fadeUp 0.4s ease 0.35s both, mu-float 3s ease-in-out 1.2s infinite' }}>
              {result.approved ? '🎉 Submitted!' : 'Needs Review'}
            </div>
            <div style={{ fontSize: '0.83rem', color: '#94a3b8', marginBottom: 20, lineHeight: 1.7, animation: 'mu-fadeUp 0.4s ease 0.5s both' }}>
              {result.approved
                ? 'Invoice sent for admin review.\nYou\'re all set!'
                : 'Admin will review your document.\nPlease wait for approval.'}
            </div>
            {/* OCR Result — always shown */}
            <div style={{ animation: 'mu-fadeUp 0.4s ease 0.6s both', textAlign: 'left', background: '#0f172a', borderRadius: 12, border: '1px solid #334155', marginBottom: 18, overflow: 'hidden' }}>
              <div style={{ padding: '8px 14px', borderBottom: '1px solid #1e293b', display: 'flex', alignItems: 'center', gap: 7 }}>
                <span style={{ fontSize: '0.65rem', fontWeight: 700, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.08em', flex: 1 }}>📄 OCR Extracted Text</span>
                {result.ocr_available === false && (
                  <span style={{ fontSize: '0.6rem', background: 'rgba(239,68,68,0.12)', color: '#ef4444', padding: '2px 7px', borderRadius: 99, border: '1px solid rgba(239,68,68,0.25)' }}>Unavailable</span>
                )}
                {result.goods_count != null && (
                  <span style={{ fontSize: '0.65rem', background: 'rgba(16,185,129,0.12)', color: '#10b981', padding: '2px 8px', borderRadius: 99, border: '1px solid rgba(16,185,129,0.25)' }}>
                    {result.goods_count} items
                  </span>
                )}
              </div>
              <div style={{ padding: '12px 14px', fontFamily: 'monospace', fontSize: '0.78rem', color: result.raw_text ? '#e2e8f0' : '#475569',
                whiteSpace: 'pre-wrap', wordBreak: 'break-word', maxHeight: 160, overflowY: 'auto', lineHeight: 1.6 }}>
                {result.raw_text && !result.raw_text.startsWith('[')
                  ? result.raw_text
                  : result.raw_text?.startsWith('[OCR_UNAVAILABLE')
                    ? 'OCR engine not installed on server.\nText cannot be extracted from this image.'
                    : '⚠ No readable text detected.\nMake sure the invoice is in clear focus.'}
              </div>
            </div>

            <button onClick={() => { setResult(null); setFile(null); setPreview(null) }}
              style={{ animation: 'mu-fadeUp 0.4s ease 0.7s both', padding: '13px 36px', borderRadius: 12, border: '1px solid #334155', cursor: 'pointer',
                background: 'linear-gradient(135deg,#1e293b,#0f172a)', color: '#94a3b8', fontSize: '0.88rem', fontWeight: 600,
                transition: 'border-color 0.2s, color 0.2s' }}>
              Upload Another
            </button>
          </div>
        )}
      </div>
      <div style={{ marginTop: 20, fontSize: '0.68rem', color: '#334155', textAlign: 'center' }}>OccuSafe - Factory Safety Monitor</div>
    </div>
  )
}


function ProtectedRoute({ children }) {
  const { user, loading } = useAuth()
  if (loading) return (
    <div style={{ display:'flex', alignItems:'center', justifyContent:'center',
      minHeight:'100vh', gap:12, color:'var(--text-secondary)' }}>
      <span className="spinner" />
      <span>Loading OccuSafe…</span>
    </div>
  )
  if (!user) return <Navigate to="/login" replace />
  return children
}

/** Bottom navigation bar — mobile only */
function BottomNav() {
  return (
    <nav className="bottom-nav" aria-label="Bottom navigation">
      <NavLink to="/floors/ground" className={({ isActive }) => `bottom-nav-item ${isActive ? 'active' : ''}`}>
        <Layers size={20} />
        <span>Floors</span>
      </NavLink>
      <NavLink to="/alerts" className={({ isActive }) => `bottom-nav-item ${isActive ? 'active' : ''}`}>
        <Bell size={20} />
        <span>Alerts</span>
      </NavLink>
      <NavLink to="/dashboard" className={({ isActive }) => `bottom-nav-item ${isActive ? 'active' : ''}`}>
        <LayoutDashboard size={20} />
        <span>Dashboard</span>
      </NavLink>
    </nav>
  )
}

function AppLayout({ children }) {
  return (
    <div className="app-layout">
      <Sidebar />
      <main className="main-content">{children}</main>
      <BottomNav />
    </div>
  )
}

function AppRoutes() {
  const { user } = useAuth()
  return (
    <Routes>
      {/* ── Public ──────────────────────────────────────── */}
      <Route path="/login"  element={user ? <Navigate to="/floors/ground" /> : <Login  />} />
      <Route path="/signup" element={user ? <Navigate to="/floors/ground" /> : <Signup />} />
      <Route path="/upload-invoice" element={<MobileUpload />} />

      {/* ── Protected — default landing is /floors/ground ─ */}
      <Route path="/" element={<Navigate to={user ? '/floors/ground' : '/login'} replace />} />

      {/* Monitoring */}
      <Route path="/floors/:floorId"  element={<ProtectedRoute><AppLayout><FloorOverview /></AppLayout></ProtectedRoute>} />
      <Route path="/monitor"          element={<ProtectedRoute><AppLayout><LiveMonitor   /></AppLayout></ProtectedRoute>} />
      <Route path="/alerts"           element={<ProtectedRoute><AppLayout><Alerts        /></AppLayout></ProtectedRoute>} />
      <Route path="/alerts/review"    element={<ProtectedRoute><AppLayout><AlertsReview  /></AppLayout></ProtectedRoute>} />
      <Route path="/dashboard"        element={<ProtectedRoute><AppLayout><Dashboard     /></AppLayout></ProtectedRoute>} />
      <Route path="/attendance"       element={<ProtectedRoute><AppLayout><Attendance    /></AppLayout></ProtectedRoute>} />
      <Route path="/documents"        element={<ProtectedRoute><AppLayout><Documents     /></AppLayout></ProtectedRoute>} />
      <Route path="/workflow"         element={<ProtectedRoute><AppLayout><WorkflowMonitor /></AppLayout></ProtectedRoute>} />
      <Route path="/diagnostics"      element={<ProtectedRoute><AppLayout><Diagnostics /></AppLayout></ProtectedRoute>} />

      {/* Settings */}
      <Route path="/settings/cameras"   element={<ProtectedRoute><AppLayout><Cameras   /></AppLayout></ProtectedRoute>} />
      <Route path="/settings/employees" element={<ProtectedRoute><AppLayout><Employees /></AppLayout></ProtectedRoute>} />
      <Route path="/settings"           element={<ProtectedRoute><AppLayout><Settings  /></AppLayout></ProtectedRoute>} />

      {/* Account */}
      <Route path="/profile"     element={<ProtectedRoute><Profile /></ProtectedRoute>} />
      <Route path="/setup-rules" element={<ProtectedRoute><SetupRules /></ProtectedRoute>} />

      {/* Catch-all */}
      <Route path="*" element={<Navigate to={user ? '/floors/ground' : '/login'} replace />} />
    </Routes>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <ThemeProvider>
        <AuthProvider>
          <ToastProvider>
            <AppRoutes />
            <PWAInstallPrompt />
          </ToastProvider>
        </AuthProvider>
      </ThemeProvider>
    </BrowserRouter>
  )
}
