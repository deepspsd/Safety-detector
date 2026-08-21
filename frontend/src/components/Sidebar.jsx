/* eslint-disable react-hooks/set-state-in-effect */
import { useState, useEffect } from 'react'
import { NavLink, useNavigate, useLocation } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { useTheme } from '../context/ThemeContext'
import { alertsApi } from '../api/api'
import {
  Video, Bell, LayoutDashboard, Settings, Shield,
  LogOut, UserCircle, HardHat, Sun, Moon, Menu, X,
  Camera, Users, Layers, ChevronDown, ChevronRight,
  ShieldAlert, ClipboardList, FileText, Briefcase
} from 'lucide-react'

const FLOORS = [
  { id: 'ground', label: 'Ground Floor', icon: '🏭' },
  { id: 'first',  label: 'First Floor',  icon: '🏗️' },
  { id: 'second', label: 'Second Floor', icon: '🏢' },
  { id: 'shop',   label: 'Shop Floor',   icon: '🛒' },
]

const ROLE_ICON = {
  'Bakery Worker':       '🧁',
  'Construction Worker': '🏗️',
  'Doctor':              '🩺',
  'Traffic Police':      '🚓',
  'College':             '🎓',
  'Home':                '🏠',
}

function NavSection({ label }) {
  return (
    <div className="nav-section-label" style={{ marginTop: 12 }}>{label}</div>
  )
}

export default function Sidebar() {
  const { user, logout }    = useAuth()
  const { theme, toggle }   = useTheme()
  const navigate            = useNavigate()
  const location            = useLocation()
  const [open,     setOpen]     = useState(false)
  const [floorsOpen, setFloorsOpen] = useState(false)
  const [pending,  setPending]  = useState(0)

  // Auto-expand floors section when on a /floors/* route
  useEffect(() => {
    if (location.pathname.startsWith('/floors')) setFloorsOpen(true)
  }, [location.pathname])

  // Close sidebar on navigation (mobile)
  useEffect(() => { setOpen(false) }, [location.pathname])
  useEffect(() => {
    document.body.style.overflow = open ? 'hidden' : ''
    return () => { document.body.style.overflow = '' }
  }, [open])

  // Poll pending-review count for badge
  useEffect(() => {
    if (!user) return
    const fetch = () =>
      alertsApi.pending()
        .then(r => {
          const list = Array.isArray(r.data) ? r.data : (r.data.alerts || [])
          setPending(list.length)
        })
        .catch(() => {})
    fetch()
    const t = setInterval(fetch, 30000)
    return () => clearInterval(t)
  }, [user])

  const initials = user?.name
    ? user.name.split(' ').map(w => w[0]).join('').toUpperCase().slice(0, 2)
    : user?.email?.[0]?.toUpperCase() || '?'

  const handleLogout   = () => { logout(); navigate('/login') }
  const roleIcon       = ROLE_ICON[user?.role] || '👷'
  const isBakery       = user?.role === 'Bakery Worker'
  const isConstruction = user?.role === 'Construction Worker' || !user?.role
  const isFloorActive  = location.pathname.startsWith('/floors')

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

        {/* Role banner */}
        {isBakery && (
          <div style={{
            margin: '0 10px 8px', padding: '8px 12px',
            borderRadius: 8,
            background: 'rgba(249,115,22,0.08)',
            border: '1px solid rgba(249,115,22,0.2)',
            display: 'flex', alignItems: 'center', gap: 6,
          }}>
            <span style={{ fontSize: '0.85rem' }}>🧁</span>
            <span style={{ fontSize: '0.7rem', fontWeight: 700, color: '#fb923c', letterSpacing: '0.03em' }}>
              BAKERY FOOD SAFETY
            </span>
          </div>
        )}
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

        {/* ── MONITORING ─────────────────────────── */}
        <NavSection label="Monitoring" />

        {/* Floors — collapsible sub-menu */}
        <button
          className={`nav-item ${isFloorActive ? 'active' : ''}`}
          style={{ width: '100%', background: 'none', border: 'none', cursor: 'pointer', justifyContent: 'space-between' }}
          onClick={() => setFloorsOpen(v => !v)}
          aria-expanded={floorsOpen}
        >
          <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <Layers size={17} />
            Floors
          </span>
          {floorsOpen ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
        </button>

        {floorsOpen && (
          <div style={{ paddingLeft: 12 }}>
            {FLOORS.map(f => (
              <NavLink key={f.id} to={`/floors/${f.id}`}
                className={({ isActive }) => `nav-item nav-item-sub ${isActive ? 'active' : ''}`}>
                <span style={{ fontSize: '0.9rem' }}>{f.icon}</span>
                <span style={{ fontSize: '0.82rem' }}>{f.label}</span>
              </NavLink>
            ))}
          </div>
        )}

        <NavLink to="/monitor" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
          <Video size={17} />
          Live Monitor
        </NavLink>

        <NavLink to="/alerts" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
          <Bell size={17} />
          Alerts
        </NavLink>

        {/* Review queue — with pending badge */}
        <NavLink to="/alerts/review" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
          <ShieldAlert size={17} />
          Review Queue
          {pending > 0 && (
            <span className="pending-badge">{pending > 99 ? '99+' : pending}</span>
          )}
        </NavLink>

        <NavLink to="/dashboard" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
          <LayoutDashboard size={17} />
          Dashboard
        </NavLink>

        <NavLink to="/attendance" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
          <ClipboardList size={17} />
          Attendance
        </NavLink>

        <NavLink to="/documents" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
          <FileText size={17} />
          Documents
        </NavLink>

        <NavLink to="/workflow" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
          <Briefcase size={17} />
          Workflow
        </NavLink>

        {/* ── SETTINGS ───────────────────────────── */}
        <NavSection label="Settings" />

        <NavLink to="/settings/cameras" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
          <Camera size={17} />
          Cameras
        </NavLink>

        <NavLink to="/settings/employees" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
          <Users size={17} />
          Employees
        </NavLink>

        <NavLink to="/settings" end className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
          <Settings size={17} />
          Settings
        </NavLink>

        {/* ── ACCOUNT ────────────────────────────── */}
        <NavSection label="Account" />

        <NavLink to="/profile" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
          <UserCircle size={17} />
          Profile
        </NavLink>


        {/* User card + logout */}
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
