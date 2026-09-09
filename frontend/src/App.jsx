import { useState, useEffect } from 'react'
import { BrowserRouter, Routes, Route, Navigate, NavLink } from 'react-router-dom'
import { AuthProvider, useAuth } from './context/AuthContext'
import { ToastProvider } from './context/ToastContext'
import { ThemeProvider } from './context/ThemeContext'
import Sidebar from './components/Sidebar'
import PWAInstallPrompt from './components/PWAInstallPrompt'
import api from './api/api'
import Login from './pages/Login'
import Signup from './pages/Signup'
import Profile from './pages/Profile'
import Dashboard from './pages/Dashboard'
import LiveMonitor from './pages/LiveMonitor'
import Alerts from './pages/Alerts'
import Settings from './pages/Settings'
import SetupRules from './pages/SetupRules'
import FloorOverview from './pages/FloorOverview'
import Cameras from './pages/Cameras'
import Employees from './pages/Employees'
import Attendance from './pages/Attendance'
import Documents from './pages/Documents'
import WorkflowMonitor from './pages/WorkflowMonitor'
import Diagnostics from './pages/Diagnostics'
import MobileUpload from './pages/MobileUpload'
import { Layers, Bell, LayoutDashboard, FileText } from 'lucide-react'


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
  const { user } = useAuth()
  const [pendingDocs, setPendingDocs] = useState(0)

  useEffect(() => {
    if (!user) return
    const fetch = () => {
      api.get('/documents/pending-approval')
        .then(r => {
          const list = Array.isArray(r.data?.records) ? r.data.records : []
          setPendingDocs(list.length)
        })
        .catch(() => {})
    }
    fetch()
    const t = setInterval(fetch, 6000)
    return () => clearInterval(t)
  }, [user])

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
      <NavLink to="/documents" className={({ isActive }) => `bottom-nav-item ${isActive ? 'active' : ''}`} style={{ position: 'relative' }}>
        <FileText size={20} />
        <span>Docs</span>
        {pendingDocs > 0 && (
          <span style={{
            position: 'absolute', top: 4, right: '20%',
            background: '#f59e0b', color: '#090d16', fontWeight: 900,
            fontSize: '0.62rem', minWidth: 16, height: 16, borderRadius: 99,
            display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '0 4px',
            boxShadow: '0 0 8px rgba(245,158,11,0.6)',
          }}>
            {pendingDocs}
          </span>
        )}
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
      <Route path="/vendor-upload" element={<MobileUpload />} />
      <Route path="/gate-portal" element={<MobileUpload />} />

      {/* ── Protected — default landing is /floors/ground ─ */}
      <Route path="/" element={<Navigate to={user ? '/floors/ground' : '/login'} replace />} />

      {/* Monitoring */}
      <Route path="/floors/:floorId"  element={<ProtectedRoute><AppLayout><FloorOverview /></AppLayout></ProtectedRoute>} />
      <Route path="/monitor"          element={<ProtectedRoute><AppLayout><LiveMonitor   /></AppLayout></ProtectedRoute>} />
      <Route path="/alerts"           element={<ProtectedRoute><AppLayout><Alerts        /></AppLayout></ProtectedRoute>} />
      <Route path="/alerts/review"    element={<Navigate to="/alerts" replace />} />
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
