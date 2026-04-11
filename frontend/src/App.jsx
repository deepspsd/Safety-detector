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
import Settings from './pages/Settings'
import { Video, Bell, LayoutDashboard } from 'lucide-react'

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

/** Bottom navigation bar — mobile only (order: Monitor → Alerts → Dashboard) */
function BottomNav() {
  return (
    <nav className="bottom-nav" aria-label="Bottom navigation">
      <NavLink to="/monitor" className={({ isActive }) => `bottom-nav-item ${isActive ? 'active' : ''}`}>
        <Video size={20} />
        <span>Monitor</span>
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
      {/* Public */}
      <Route path="/login"  element={user ? <Navigate to="/monitor" /> : <Login  />} />
      <Route path="/signup" element={user ? <Navigate to="/monitor" /> : <Signup />} />

      {/* Protected — default landing is /monitor */}
      <Route path="/" element={<Navigate to={user ? "/monitor" : "/login"} replace />} />
      <Route path="/profile"   element={<ProtectedRoute><Profile /></ProtectedRoute>} />
      <Route path="/monitor"   element={<ProtectedRoute><AppLayout><LiveMonitor /></AppLayout></ProtectedRoute>} />
      <Route path="/alerts"    element={<ProtectedRoute><AppLayout><Alerts /></AppLayout></ProtectedRoute>} />
      <Route path="/dashboard" element={<ProtectedRoute><AppLayout><Dashboard /></AppLayout></ProtectedRoute>} />
      <Route path="/settings"  element={<ProtectedRoute><AppLayout><Settings /></AppLayout></ProtectedRoute>} />
      <Route path="*"          element={<Navigate to={user ? "/monitor" : "/login"} replace />} />
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
