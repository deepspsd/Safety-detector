import { useState, useEffect } from 'react'
import { NavLink, useNavigate, useLocation } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { useTheme } from '../context/ThemeContext'
import {
  Video, Bell, LayoutDashboard, Settings, Shield,
  LogOut, UserCircle, HardHat, Sun, Moon, Menu, X
} from 'lucide-react'

// Navigation order: Monitor → Alerts → Dashboard
const NAV = [
  { to: '/monitor',   icon: Video,           label: 'Live Monitor' },
  { to: '/alerts',    icon: Bell,            label: 'Alerts'       },
  { to: '/dashboard', icon: LayoutDashboard, label: 'Dashboard'    },
]
const BOTTOM_NAV = [
  { to: '/profile',  icon: UserCircle, label: 'Profile'  },
  { to: '/settings', icon: Settings,   label: 'Settings' },
]

const ROLE_ICON = {
  'Construction Worker': '🏗️',
  'Doctor':              '🩺',
  'Traffic Police':      '🚓',
  'College':             '🎓',
  'Home':                '🏠',
}

export default function Sidebar() {
  const { user, logout } = useAuth()
  const { theme, toggle } = useTheme()
  const navigate  = useNavigate()
  const location  = useLocation()
  const [open, setOpen] = useState(false)

  useEffect(() => { setOpen(false) }, [location.pathname])
  useEffect(() => {
    document.body.style.overflow = open ? 'hidden' : ''
    return () => { document.body.style.overflow = '' }
  }, [open])

  const initials = user?.name
    ? user.name.split(' ').map(w => w[0]).join('').toUpperCase().slice(0, 2)
    : user?.email?.[0]?.toUpperCase() || '?'

  const handleLogout  = () => { logout(); navigate('/login') }
  const roleIcon      = ROLE_ICON[user?.role] || '👷'
  const isConstruction = user?.role === 'Construction Worker' || !user?.role

  return (
    <>
      {/* ── Mobile top bar ─────────────────────────────── */}
      <header className="mobile-header">
        <button className="mobile-menu-btn" onClick={() => setOpen(true)} aria-label="Open menu">
          <Menu size={22} />
        </button>
        <div className="mobile-header-logo">
          <div className="sidebar-logo-icon" style={{ width: 28, height: 28 }}>
            <Shield size={14} color="#fff" />
          </div>
          <span>OccuSafe</span>
        </div>
        <button className="theme-toggle-btn" onClick={toggle}>
          {theme === 'dark' ? <Sun size={15} /> : <Moon size={15} />}
        </button>
      </header>

      {open && <div className="sidebar-overlay" onClick={() => setOpen(false)} />}

      <aside className={`sidebar ${open ? 'open' : ''}`}>
        {/* Logo */}
        <div className="sidebar-logo">
          <div className="sidebar-logo-icon">
            <Shield size={18} color="#fff" />
          </div>
          <span>OccuSafe</span>
          <button className="theme-toggle-btn" onClick={toggle} style={{ marginLeft: 'auto' }}>
            {theme === 'dark' ? <Sun size={15} /> : <Moon size={15} />}
          </button>
          <button className="sidebar-close-btn" onClick={() => setOpen(false)} aria-label="Close">
            <X size={18} />
          </button>
        </div>

        {/* Construction banner */}
        {isConstruction && (
          <div style={{
            margin: '0 10px 8px', padding: '8px 12px',
            borderRadius: 8,
            background: 'rgba(249,115,22,0.08)',
            border: '1px solid rgba(249,115,22,0.2)',
            display: 'flex', alignItems: 'center', gap: 6,
          }}>
            <HardHat size={13} color="#fb923c" />
            <span style={{ fontSize: '0.7rem', fontWeight: 700, color: '#fb923c', letterSpacing: '0.03em' }}>
              OCCUPATIONAL SAFETY
            </span>
          </div>
        )}

        {/* Main nav */}
        <div className="nav-section-label">Navigation</div>
        {NAV.map(({ to, icon: Icon, label }) => (
          <NavLink key={to} to={to}
            className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
            <Icon size={17} />
            {label}
          </NavLink>
        ))}

        <div className="nav-section-label" style={{ marginTop: 12 }}>Account</div>
        {BOTTOM_NAV.map(({ to, icon: Icon, label }) => (
          <NavLink key={to} to={to}
            className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
            <Icon size={17} />
            {label}
          </NavLink>
        ))}

        {/* User */}
        <div className="sidebar-bottom">
          <div className="sidebar-user">
            <div className="sidebar-avatar">{initials}</div>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div className="sidebar-user-name">{user?.name || 'User'}</div>
              {user?.role && (
                <div style={{ marginTop: 3 }}>
                  <span className="role-pill">{roleIcon} {user.role}</span>
                </div>
              )}
            </div>
          </div>
          <button
            className="nav-item btn-ghost"
            style={{ width: 'calc(100% - 20px)', margin: '0 10px', border: 'none' }}
            onClick={handleLogout}
          >
            <LogOut size={15} />
            Sign Out
          </button>
        </div>
      </aside>
    </>
  )
}
