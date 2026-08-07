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
import TVDisplay from './pages/TVDisplay'
import { Layers, Bell, LayoutDashboard } from 'lucide-react'

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

      {/* TV display — public, no auth, no layout */}
      <Route path="/tv" element={<TVDisplay />} />

      {/* ── Protected — default landing is /floors/ground ─ */}
      <Route path="/" element={<Navigate to={user ? '/floors/ground' : '/login'} replace />} />

      {/* Monitoring */}
      <Route path="/floors/:floorId"  element={<ProtectedRoute><AppLayout><FloorOverview /></AppLayout></ProtectedRoute>} />
      <Route path="/monitor"          element={<ProtectedRoute><AppLayout><LiveMonitor   /></AppLayout></ProtectedRoute>} />
      <Route path="/alerts"           element={<ProtectedRoute><AppLayout><Alerts        /></AppLayout></ProtectedRoute>} />
      <Route path="/alerts/review"    element={<ProtectedRoute><AppLayout><AlertsReview  /></AppLayout></ProtectedRoute>} />
      <Route path="/dashboard"        element={<ProtectedRoute><AppLayout><Dashboard     /></AppLayout></ProtectedRoute>} />

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
