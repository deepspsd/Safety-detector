import { useState, useEffect, useCallback } from 'react'
import { useAuth } from '../context/AuthContext'
import { useToast } from '../context/ToastContext'
import { useTheme } from '../context/ThemeContext'
import { usersApi, facesApi, camerasApi } from '../api/api'
import ZonePainter from '../components/ZonePainter'
import {
  Settings as SettingsIcon, Camera, Bell, User,
  Shield, Upload, Trash2, CheckCircle, Sun, Moon,
  Palette, HardHat
} from 'lucide-react'

const CAMERA_TYPES = [
  { id: 'webcam', label: 'Webcam',        desc: 'Built-in or USB camera' },
  { id: 'rtsp',   label: 'CCTV / RTSP',  desc: 'Network camera stream'  },
  { id: 'upload', label: 'Video Upload',  desc: 'Process recorded footage'},
]

const ROLES = [
  'Bakery Worker', 'Construction Worker', 'Doctor', 'Traffic Police', 'College', 'Home', 'None'
]

// PPE requirements by role (for display)
const ROLE_PPE = {
  'Bakery Worker':       ['Head Cap', 'Face Mask', 'Gloves', 'No Bangles'],
  'Construction Worker': ['Hardhat', 'Safety Vest', 'Gloves', 'Safety Goggles'],
  'Doctor':              ['Face Mask', 'Gloves'],
  'Traffic Police':      ['Hardhat'],
  'College':             ['ID Card'],
  'Home':                ['Face Recognition'],
}

export default function Settings() {
  const { user, updateUser, customPpeItems, setCustomPpeItems, noPhoneZone: savedNoPhoneZone, setNoPhoneZone } = useAuth()
  const { addToast }         = useToast()
  const { toggle, isDark } = useTheme()

  const [config,    setConfig]    = useState({ camera_type:'webcam', rtsp_url:'', notify_sound:true, notify_ui:true, detection_sensitivity:0.5 })
  const [profile,   setProfile]   = useState({ name: user?.name || '', role: user?.role || 'Construction Worker' })
  const [faces,     setFaces]     = useState([])
  const [faceLabel, setFaceLabel] = useState('Owner')
  const [faceImg,   setFaceImg]   = useState(null)
  const [saving,    setSaving]    = useState(false)
  const [activeTab, setActiveTab] = useState('appearance')
  const [locationLoading, setLocationLoading] = useState(false)
  const [locationStatus,  setLocationStatus]  = useState(null)
  // Custom PPE — local copy editable in the Safety Rules tab
  const [customPpe, setCustomPpe] = useState(customPpeItems || [])
  // Phone zone — local copy editable in the Phone tab
  const [noPhoneZoneLocal, setNoPhoneZoneLocal] = useState(!!savedNoPhoneZone)
  // editing state for inline label rename
  const [editingFaceId,    setEditingFaceId]    = useState(null)
  const [editingFaceLabel, setEditingFaceLabel] = useState('')

  useEffect(() => {
    usersApi.getConfig().then(r => {
      setConfig(r.data)
      if (r.data.custom_ppe_items?.length) setCustomPpe(r.data.custom_ppe_items)
      if (r.data.no_phone_zone !== undefined) setNoPhoneZoneLocal(!!r.data.no_phone_zone)
    }).catch(() => {})
    // Load faces for ALL roles, not just Home
    facesApi.list().then(r => setFaces(r.data)).catch(() => {})
  }, [user])

  const saveConfig = async () => {
    setSaving(true)
    try {
      await usersApi.updateConfig(config)
      addToast('Settings saved', 'Camera configuration updated', 'success')
    } catch { addToast('Save failed', '', 'danger') }
    finally { setSaving(false) }
  }

  const saveProfile = async () => {
    setSaving(true)
    try {
      await usersApi.updateProfile(profile)
      updateUser(profile)
      addToast('Profile updated', `Role set to ${profile.role}`, 'success')
    } catch { addToast('Update failed', '', 'danger') }
    finally { setSaving(false) }
  }

  const savePpe = async () => {
    setSaving(true)
    try {
      await usersApi.updateCustomPpe(customPpe)
      setCustomPpeItems(customPpe)
      addToast('Safety Rules saved', `${customPpe.length} item(s) selected`, 'success')
    } catch { addToast('Save failed', '', 'danger') }
    finally { setSaving(false) }
  }

  const savePhoneSettings = async () => {
    setSaving(true)
    try {
      await usersApi.updateConfig({ no_phone_zone: noPhoneZoneLocal })
      setNoPhoneZone(noPhoneZoneLocal)   // update context so LiveMonitor picks it up
      addToast('Phone settings saved', noPhoneZoneLocal ? 'No-Phone Zone is ON' : 'No-Phone Zone is OFF', 'success')
    } catch { addToast('Save failed', '', 'danger') }
    finally { setSaving(false) }
  }

  const handleFaceImage = (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = () => setFaceImg(reader.result)
    reader.readAsDataURL(file)
  }

  const registerFace = async () => {
    if (!faceImg) return addToast('No image selected', '', 'danger')
    setSaving(true)
    try {
      await facesApi.register(faceLabel, faceImg)
      addToast('Face registered!', `"${faceLabel}" added`, 'success')
      const res = await facesApi.list()
      setFaces(res.data)
      setFaceImg(null)
    } catch (e) { addToast('Registration failed', e.response?.data?.detail || '', 'danger') }
    finally { setSaving(false) }
  }

  const deleteFace = async (id) => {
    try {
      await facesApi.delete(id)
      setFaces(f => f.filter(x => x.id !== id))
      addToast('Face removed', '', 'success')
    } catch { addToast('Delete failed', '', 'danger') }
  }

  const renameFace = async (id, newLabel) => {
    if (!newLabel.trim() || newLabel === faces.find(f => f.id === id)?.label) {
      setEditingFaceId(null); return
    }
    try {
      await facesApi.rename(id, newLabel.trim())
      setFaces(prev => prev.map(f => f.id === id ? { ...f, label: newLabel.trim() } : f))
      addToast('Name updated', `Renamed to "${newLabel.trim()}"`, 'success')
    } catch { addToast('Rename failed', '', 'danger') }
    setEditingFaceId(null)
  }

  const setConf = (k, v) => setConfig(c => ({ ...c, [k]: v }))

  // ── Location-based role detection ───────────────────────────────
  const PLACE_ROLE_MAP = [
    // Hospitals / Medical
    { keywords: ['hospital', 'clinic', 'medical', 'health', 'pharmacy', 'dentist', 'doctor', 'nursing'], role: 'Doctor' },
    // Construction
    { keywords: ['construction', 'building site', 'site', 'industrial', 'factory', 'warehouse', 'plant', 'workshop'], role: 'Construction Worker' },
    // College / University
    { keywords: ['university', 'college', 'school', 'institute', 'campus', 'academy', 'polytechnic'], role: 'College' },
    // Traffic / Roads
    { keywords: ['traffic', 'police', 'highway', 'junction', 'signal', 'toll', 'road patrol'], role: 'Traffic Police' },
  ]

  const detectLocationRole = async () => {
    if (!navigator.geolocation) {
      setLocationStatus({ ok: false, message: 'Geolocation not supported by your browser.' })
      return
    }
    setLocationLoading(true)
    setLocationStatus(null)

    navigator.geolocation.getCurrentPosition(
      async (pos) => {
        try {
          const { latitude, longitude } = pos.coords
          // Use OpenStreetMap Nominatim (free, no API key)
          const res  = await fetch(
            `https://nominatim.openstreetmap.org/reverse?lat=${latitude}&lon=${longitude}&zoom=18&format=json`,
            { headers: { 'Accept-Language': 'en', 'User-Agent': 'OccuSafe-Monitor/1.0' } }
          )
          const data = await res.json()

          // Collect all text from the address object to search
          const addressText = [
            data.display_name || '',
            data.type || '',
            data.category || '',
            ...Object.values(data.address || {}),
          ].join(' ').toLowerCase()

          let matched = null
          for (const entry of PLACE_ROLE_MAP) {
            if (entry.keywords.some(kw => addressText.includes(kw))) {
              matched = entry.role
              break
            }
          }

          if (matched) {
            setProfile(p => ({ ...p, role: matched }))
            setLocationStatus({
              ok: true,
              message: `📍 Detected: ${data.display_name?.split(',')[0]} → Role set to "${matched}"`
            })
          } else {
            setLocationStatus({
              ok: false,
              message: `📍 Location found (${data.display_name?.split(',')[0]}) but no matching role. Please set manually.`
            })
          }
        } catch {
          setLocationStatus({ ok: false, message: 'Could not fetch location data. Check internet connection.' })
        } finally {
          setLocationLoading(false)
        }
      },
      (err) => {
        setLocationLoading(false)
        setLocationStatus({ ok: false, message: `Location access denied: ${err.message}` })
      },
      { timeout: 10000, maximumAge: 60000 }
    )
  }

  const tabs = [
    { id: 'appearance',    label: '🎨 Appearance'   },
    { id: 'safety_rules',  label: '🛡️ Safety Rules'  },
    { id: 'phone',         label: '📱 Phone'         },
    { id: 'camera',        label: '📷 Camera'        },
    { id: 'cameras_mgmt',  label: '🎥 Cameras'       },
    { id: 'notifications', label: '🔔 Alerts'        },
    { id: 'profile',       label: '👤 Profile'       },
    { id: 'faces',         label: '🔍 Faces'         },
  ]

  return (
    <div className="page-container">
      <div className="page-header">
        <div>
          <h1 className="page-title">Settings</h1>
          <p className="page-subtitle">Configure detection, appearance, and notification preferences</p>
        </div>
      </div>

      <div className="tab-bar" style={{ maxWidth: 600, marginBottom: 24 }}>
        {tabs.map(t => (
          <button key={t.id} className={`tab ${activeTab === t.id ? 'active' : ''}`}
            onClick={() => setActiveTab(t.id)}>
            {t.label}
          </button>
        ))}
      </div>

      {/* ── Appearance Tab ── */}
      {activeTab === 'appearance' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20, maxWidth: 600 }}>

          {/* Theme toggle card */}
          <div className="card card-p">
            <h3 style={{ fontSize: '1rem', marginBottom: 20, display: 'flex', alignItems: 'center', gap: 8 }}>
              <Palette size={18} /> Theme & Display
            </h3>

            <div style={{
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              padding: '16px 20px', borderRadius: 12,
              background: isDark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.04)',
              border: '1px solid var(--border)',
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
                <div style={{
                  width: 44, height: 44, borderRadius: 12, display: 'flex',
                  alignItems: 'center', justifyContent: 'center',
                  background: isDark ? 'rgba(139,92,246,0.15)' : 'rgba(245,158,11,0.12)',
                }}>
                  {isDark ? <Moon size={20} color="#a78bfa" /> : <Sun size={20} color="#f59e0b" />}
                </div>
                <div>
                  <div style={{ fontWeight: 600, fontSize: '0.95rem' }}>
                    {isDark ? 'Dark Mode' : 'Light Mode'}
                  </div>
                  <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: 2 }}>
                    {isDark
                      ? 'Ideal for low-light industrial environments'
                      : 'Bright and clear for daytime monitoring'}
                  </div>
                </div>
              </div>
              {/* Animated toggle */}
              <div
                onClick={toggle}
                className={`toggle ${isDark ? '' : 'on'}`}
                style={{ '--toggle-off': 'rgba(139,92,246,0.25)', cursor: 'pointer' }}
                title="Switch theme"
              />
            </div>

            {/* Preview swatches */}
            <div style={{ marginTop: 16, display: 'flex', gap: 8 }}>
              {/* Dark preview */}
              <div
                onClick={() => !isDark && toggle()}
                style={{
                  flex: 1, borderRadius: 10, overflow: 'hidden', cursor: 'pointer',
                  border: `2px solid ${isDark ? 'var(--accent-purple)' : 'transparent'}`,
                  transition: 'border 0.2s',
                }}>
                <div style={{ height: 56, background: '#080c14', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <div style={{ width: 60, height: 8, borderRadius: 4, background: 'rgba(59,130,246,0.5)' }} />
                </div>
                <div style={{ background: '#0d1421', padding: '6px 8px', fontSize: '0.7rem', fontWeight: 600, color: '#94a3b8', textAlign: 'center' }}>
                  Dark {isDark && '✓'}
                </div>
              </div>
              {/* Light preview */}
              <div
                onClick={() => isDark && toggle()}
                style={{
                  flex: 1, borderRadius: 10, overflow: 'hidden', cursor: 'pointer',
                  border: `2px solid ${!isDark ? 'var(--accent-orange)' : 'transparent'}`,
                  transition: 'border 0.2s',
                }}>
                <div style={{ height: 56, background: '#f0f4f8', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <div style={{ width: 60, height: 8, borderRadius: 4, background: 'rgba(59,130,246,0.4)' }} />
                </div>
                <div style={{ background: '#fff', padding: '6px 8px', fontSize: '0.7rem', fontWeight: 600, color: '#475569', textAlign: 'center' }}>
                  Light {!isDark && '✓'}
                </div>
              </div>
            </div>
          </div>

          {/* PPE Requirements info card */}
          <div className="card card-p">
            <h3 style={{ fontSize: '1rem', marginBottom: 16, display: 'flex', alignItems: 'center', gap: 8 }}>
              <HardHat size={18} color="var(--accent-construction)" />
              PPE Requirements — {user?.role || 'No role set'}
            </h3>
            {user?.role && ROLE_PPE[user.role] ? (
              <div className="ppe-grid">
                {ROLE_PPE[user.role].map(item => (
                  <div key={item} className="ppe-item ok">
                    <span style={{ fontSize: '1.2rem' }}>
                      {item === 'Hardhat' || item === 'Safety Goggles' ? '⛑️' :
                       item === 'Safety Vest' ? '🦺' :
                       item === 'Gloves' ? '🧤' :
                       item === 'Face Mask' ? '😷' :
                       item === 'ID Card' ? '🪪' : '🔍'}
                    </span>
                    {item}
                  </div>
                ))}
              </div>
            ) : (
              <p style={{ fontSize: '0.85rem' }}>Set your role in the Profile tab to see PPE requirements.</p>
            )}
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 12 }}>
              The system will alert you when any of these items are missing on a detected person.
            </p>
          </div>
        </div>
      )}

      {/* ── Safety Rules Tab ── */}
      {activeTab === 'safety_rules' && (() => {
        const ALL_PPE = [
          { id: 'NO-Hardhat',      label: 'Hardhat / Helmet', icon: '⛑️',  desc: 'Head protection' },
          { id: 'NO-Gloves',       label: 'Gloves',           icon: '🧤',  desc: 'Hand protection'  },
          { id: 'NO-Goggles',      label: 'Goggles',          icon: '🥽',  desc: 'Eye protection'   },
          { id: 'NO-Mask',         label: 'Mask',             icon: '😷',  desc: 'Face / respiratory protection' },
          { id: 'NO-Safety Vest',  label: 'Safety Vest',      icon: '🦺',  desc: 'High-visibility vest' },
          { id: 'NO-Safety Shoes', label: 'Safety Shoes',     icon: '👟',  desc: 'Foot protection'  },
          { id: 'NO-ID Card',      label: 'ID Card',          icon: '🪪',  desc: 'Identity verification' },
          { id: 'NO-Uniform',      label: 'Uniform',          icon: '👕',  desc: 'Standard uniform compliance' },
        ]
        const toggle = (id) => setCustomPpe(prev =>
          prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]
        )
        return (
          <div className="card card-p" style={{ maxWidth: 600 }}>
            <h3 style={{ fontSize: '1rem', marginBottom: 6, display: 'flex', alignItems: 'center', gap: 8 }}>
              <Shield size={18} color="var(--accent-green)" /> Custom Safety Rules
            </h3>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: 20 }}>
              Select which PPE items to monitor. These rules apply when you're in <strong>Live Monitor</strong>.
              {user?.role === 'None'
                ? ' Your role is set to Custom — these selections will be your active detection filters.'
                : ' These override the default rules for your role.'}
            </p>

            {user?.role === 'None' && (
              <div style={{
                padding: '10px 14px', borderRadius: 8, marginBottom: 18,
                background: 'rgba(16,185,129,0.08)',
                border: '1px solid rgba(16,185,129,0.25)',
                fontSize: '0.8rem', color: 'var(--accent-green)',
                display: 'flex', alignItems: 'center', gap: 8,
              }}>
                <Shield size={14} /> Custom role active — detection uses only your selected items below.
              </div>
            )}

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(240px,1fr))', gap: 10, marginBottom: 24 }}>
              {ALL_PPE.map(item => {
                const on = customPpe.includes(item.id)
                return (
                  <label key={item.id} onClick={() => toggle(item.id)} style={{
                    display: 'flex', alignItems: 'center', gap: 14,
                    padding: '12px 16px', borderRadius: 12, cursor: 'pointer',
                    background: on ? 'rgba(16,185,129,0.08)' : 'var(--bg-card)',
                    border: `1.5px solid ${on ? 'rgba(16,185,129,0.4)' : 'var(--border)'}`,
                    transition: 'all 0.15s', userSelect: 'none',
                  }}>
                    <span style={{ fontSize: '1.4rem', flexShrink: 0 }}>{item.icon}</span>
                    <div style={{ flex: 1 }}>
                      <div style={{ fontWeight: 600, fontSize: '0.875rem', color: on ? 'var(--accent-green)' : 'var(--text-primary)' }}>
                        {item.label}
                      </div>
                      <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: 2 }}>{item.desc}</div>
                    </div>
                    <div style={{
                      width: 20, height: 20, borderRadius: '50%', flexShrink: 0,
                      border: `2px solid ${on ? '#10b981' : 'var(--border)'}`,
                      background: on ? '#10b981' : 'transparent',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      transition: 'all 0.15s',
                    }}>
                      {on && <CheckCircle size={12} color="#fff" />}
                    </div>
                  </label>
                )
              })}
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              <button className="btn btn-primary" onClick={savePpe} disabled={saving}>
                {saving
                  ? <><span className="spinner" style={{ width:15, height:15, borderWidth:2 }} /> Saving…</>
                  : `Save (${customPpe.length} selected)`}
              </button>
              {customPpe.length > 0 && (
                <button className="btn btn-ghost" onClick={() => setCustomPpe([])} style={{ fontSize: '0.82rem' }}>
                  Clear All
                </button>
              )}
            </div>
          </div>
        )
      })()}

      {/* ── Camera Tab ── */}
      {activeTab === 'camera' && (
        <div className="card card-p" style={{ maxWidth: 600 }}>
          <h3 style={{ fontSize: '1rem', marginBottom: 20, display: 'flex', alignItems: 'center', gap: 8 }}>
            <Camera size={18} /> Camera Configuration
          </h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginBottom: 20 }}>
            {CAMERA_TYPES.map(t => (
              <label key={t.id} style={{
                display: 'flex', alignItems: 'center', gap: 14, padding: '13px 16px',
                borderRadius: 10, cursor: 'pointer',
                background: config.camera_type === t.id ? 'rgba(59,130,246,0.1)' : 'rgba(255,255,255,0.03)',
                border: `1.5px solid ${config.camera_type === t.id ? 'var(--accent-blue)' : 'var(--border)'}`
              }}>
                <input type="radio" style={{ display: 'none' }} checked={config.camera_type === t.id}
                  onChange={() => setConf('camera_type', t.id)} />
                <div style={{ flex: 1 }}>
                  <div style={{ fontWeight: 600, fontSize: '0.875rem' }}>{t.label}</div>
                  <div style={{ fontSize: '0.77rem', color: 'var(--text-muted)', marginTop: 2 }}>{t.desc}</div>
                </div>
                {config.camera_type === t.id && <CheckCircle size={16} color="var(--accent-blue)" />}
              </label>
            ))}
          </div>
          {config.camera_type === 'rtsp' && (
            <div className="form-group" style={{ marginBottom: 20 }}>
              <label className="form-label">RTSP URL</label>
              <input className="form-input" placeholder="rtsp://username:password@192.168.1.1:554/stream"
                value={config.rtsp_url || ''} onChange={e => setConf('rtsp_url', e.target.value)} />
            </div>
          )}
          <div className="form-group" style={{ marginBottom: 20 }}>
            <label className="form-label">Detection Sensitivity: {Math.round(config.detection_sensitivity * 100)}%</label>
            <input type="range" min="0.2" max="0.9" step="0.05"
              value={config.detection_sensitivity}
              onChange={e => setConf('detection_sensitivity', parseFloat(e.target.value))}
              style={{ width: '100%', accentColor: 'var(--accent-blue)' }} />
            <div style={{ display:'flex', justifyContent:'space-between', fontSize:'0.72rem', color:'var(--text-muted)' }}>
              <span>Low (more detections)</span><span>High (stricter)</span>
            </div>
          </div>
          <button className="btn btn-primary" onClick={saveConfig} disabled={saving}>
            {saving ? <span className="spinner" style={{ width:15, height:15, borderWidth:2 }} /> : 'Save Camera Settings'}
          </button>
        </div>
      )}

      {/* ── Notifications Tab ── */}
      {activeTab === 'notifications' && (
        <div className="card card-p" style={{ maxWidth: 600 }}>
          <h3 style={{ fontSize: '1rem', marginBottom: 20, display: 'flex', alignItems: 'center', gap: 8 }}>
            <Bell size={18} /> Alert Preferences
          </h3>
          <div style={{ borderRadius: 10, overflow: 'hidden', border: '1px solid var(--border)' }}>
            <div className="toggle-wrap" style={{ padding: '14px 18px', borderBottom: '1px solid var(--border)' }}>
              <div>
                <div className="toggle-label">UI Notifications</div>
                <div className="toggle-sub">Show alert toasts when violations are detected</div>
              </div>
              <div className={`toggle ${config.notify_ui ? 'on' : ''}`}
                onClick={() => setConf('notify_ui', !config.notify_ui)} />
            </div>
            <div className="toggle-wrap" style={{ padding: '14px 18px' }}>
              <div>
                <div className="toggle-label">Sound Alerts</div>
                <div className="toggle-sub">Play audio when critical violation detected</div>
              </div>
              <div className={`toggle ${config.notify_sound ? 'on' : ''}`}
                onClick={() => setConf('notify_sound', !config.notify_sound)} />
            </div>
          </div>
          <button className="btn btn-primary" style={{ marginTop: 20 }} onClick={saveConfig} disabled={saving}>
            Save Preferences
          </button>
        </div>
      )}

      {/* ── Profile Tab ── */}
      {activeTab === 'profile' && (
        <div className="card card-p" style={{ maxWidth: 600 }}>
          <h3 style={{ fontSize: '1rem', marginBottom: 20, display: 'flex', alignItems: 'center', gap: 8 }}>
            <User size={18} /> Profile Details
          </h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            <div className="form-group">
              <label className="form-label">Full Name</label>
              <input className="form-input" value={profile.name}
                onChange={e => setProfile(p => ({ ...p, name: e.target.value }))} />
            </div>
            <div className="form-group">
              <label className="form-label">Email</label>
              <input className="form-input" value={user?.email || ''} disabled style={{ opacity: 0.5 }} />
            </div>
            <div className="form-group">
              <label className="form-label" style={{ display:'flex', alignItems:'center', gap:5 }}>
                <HardHat size={12} color="var(--accent-construction)" /> Monitoring Role
              </label>
              <select className="form-select" value={profile.role}
                onChange={e => setProfile(p => ({ ...p, role: e.target.value }))}>
                {ROLES.map(r => <option key={r} value={r}>{r}</option>)}
              </select>
              {profile.role === 'Construction Worker' && (
                <div style={{ fontSize:'0.75rem', color:'var(--accent-construction)', marginTop:4, display:'flex', alignItems:'center', gap:4 }}>
                  <HardHat size={11} /> Manufacturing safety mode — detects Hardhat, Vest, Gloves, Goggles
                </div>
              )}
            </div>

            {/* Location-based role detection */}
            <div style={{
              padding: '14px 16px', borderRadius: 12,
              background: 'rgba(59,130,246,0.06)',
              border: '1px solid rgba(59,130,246,0.2)',
            }}>
              <div style={{ fontSize:'0.8rem', fontWeight:600, marginBottom:4, color:'var(--text-primary)' }}>
                📍 Set Role Automatically by Location
              </div>
              <div style={{ fontSize:'0.75rem', color:'var(--text-muted)', marginBottom:12 }}>
                Uses your GPS location to detect nearby workplaces (hospital, college, construction site, etc.) and set your role accordingly.
              </div>
              <button
                className="btn btn-ghost"
                style={{ display:'flex', alignItems:'center', gap:8, fontSize:'0.85rem', padding:'8px 16px' }}
                onClick={detectLocationRole}
                disabled={locationLoading}
              >
                {locationLoading
                  ? <><span className="spinner" style={{ width:14, height:14, borderWidth:2 }} /> Detecting location…</>
                  : <>🌍 Set Account Using Location</>}
              </button>
              {locationStatus && (
                <div style={{
                  marginTop: 10, padding: '8px 12px', borderRadius: 8, fontSize: '0.78rem',
                  background: locationStatus.ok ? 'rgba(16,185,129,0.1)' : 'rgba(220,38,38,0.1)',
                  border: `1px solid ${locationStatus.ok ? 'rgba(16,185,129,0.3)' : 'rgba(220,38,38,0.3)'}`,
                  color: locationStatus.ok ? 'var(--accent-green)' : 'var(--accent-red)',
                }}>
                  {locationStatus.message}
                </div>
              )}
            </div>
          </div>
          <button className="btn btn-primary" style={{ marginTop: 20 }} onClick={saveProfile} disabled={saving}>
            {saving ? <span className="spinner" style={{ width:15, height:15, borderWidth:2 }} /> : 'Update Profile'}
          </button>
        </div>
      )}

      {/* ── Phone Tab ── */}
      {activeTab === 'phone' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20, maxWidth: 600 }}>

          {/* No-Phone Zone Toggle Card */}
          <div className="card card-p">
            <h3 style={{ fontSize: '1rem', marginBottom: 6, display: 'flex', alignItems: 'center', gap: 8 }}>
              📱 Phone Detection Settings
            </h3>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: 24 }}>
              Configure how the system handles phone usage during live monitoring.
            </p>

            {/* No Phone Zone row */}
            <div style={{
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              padding: '16px 20px', borderRadius: 12,
              background: noPhoneZoneLocal ? 'rgba(239,68,68,0.06)' : 'rgba(255,255,255,0.04)',
              border: `1px solid ${noPhoneZoneLocal ? 'rgba(239,68,68,0.3)' : 'var(--border)'}`,
              transition: 'all 0.2s',
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
                <div style={{
                  width: 44, height: 44, borderRadius: 12, display: 'flex',
                  alignItems: 'center', justifyContent: 'center', fontSize: '1.4rem',
                  background: noPhoneZoneLocal ? 'rgba(239,68,68,0.15)' : 'rgba(255,255,255,0.06)',
                }}>🚫</div>
                <div>
                  <div style={{ fontWeight: 600, fontSize: '0.95rem' }}>No Phone Zone</div>
                  <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: 2 }}>
                    {noPhoneZoneLocal
                      ? '🔴 Any phone detected → immediate alert'
                      : '🟢 Only near-ear (calling) usage triggers alert'}
                  </div>
                </div>
              </div>
              {/* Animated toggle switch */}
              <button
                onClick={() => setNoPhoneZoneLocal(v => !v)}
                style={{
                  width: 52, height: 28, borderRadius: 14, cursor: 'pointer',
                  border: 'none', padding: 0, flexShrink: 0,
                  background: noPhoneZoneLocal ? '#ef4444' : 'rgba(255,255,255,0.14)',
                  position: 'relative', transition: 'background 0.2s',
                }}
              >
                <span style={{
                  position: 'absolute', top: 4, width: 20, height: 20, borderRadius: '50%',
                  background: '#fff', transition: 'left 0.2s',
                  left: noPhoneZoneLocal ? 28 : 4,
                  boxShadow: '0 1px 4px rgba(0,0,0,0.3)',
                }} />
              </button>
            </div>
          </div>

          {/* Alert Logic Info Cards */}
          <div className="card card-p">
            <h3 style={{ fontSize: '0.9rem', marginBottom: 16, color: 'var(--text-secondary)' }}>
              How Phone Detection Works
            </h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              {[
                { icon: '🟢', status: 'No Phone', desc: 'No phone detected in frame — no alert', color: '#10b981' },
                { icon: '🟡', status: 'Phone in Hand', desc: 'Phone visible but not near ear — silent label, no alert (unless No-Phone Zone is ON)', color: '#eab308' },
                { icon: '🔴', status: 'Phone Near Ear (Calling)', desc: 'Phone center in head region — triggers "Unsafe Phone Usage" alert', color: '#ef4444' },
                { icon: '🚫', status: 'No-Phone Zone Violation', desc: 'Any phone + No-Phone Zone ON — triggers "Phone Not Allowed Here" alert', color: '#ef4444' },
              ].map(item => (
                <div key={item.status} style={{
                  display: 'flex', alignItems: 'flex-start', gap: 12,
                  padding: '12px 14px', borderRadius: 10,
                  background: 'rgba(255,255,255,0.03)', border: '1px solid var(--border)',
                }}>
                  <span style={{ fontSize: '1.1rem', flexShrink: 0, marginTop: 1 }}>{item.icon}</span>
                  <div>
                    <div style={{ fontWeight: 600, fontSize: '0.85rem', color: item.color }}>{item.status}</div>
                    <div style={{ fontSize: '0.77rem', color: 'var(--text-muted)', marginTop: 3 }}>{item.desc}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <button className="btn btn-primary" onClick={savePhoneSettings} disabled={saving}>
            {saving
              ? <span className="spinner" style={{ width:15, height:15, borderWidth:2 }} />
              : '💾 Save Phone Settings'}
          </button>
        </div>
      )}

      {/* ── Faces Tab — Smart Access Control ── */}
      {activeTab === 'faces' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20, maxWidth: 700 }}>

          {/* Header */}
          <div className="card card-p" style={{ background: 'linear-gradient(135deg,rgba(99,102,241,0.12),rgba(139,92,246,0.08))', border: '1px solid rgba(99,102,241,0.25)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
              <div style={{ fontSize: '2.5rem' }}>🔍</div>
              <div>
                <div style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-primary)' }}>Smart Access Control</div>
                <div style={{ fontSize: '0.80rem', color: 'var(--text-muted)', marginTop: 3 }}>
                  Register known persons. The system alerts when an unregistered face is detected in any camera feed.
                </div>
              </div>
              {faces.length > 0 && (
                <div style={{ marginLeft: 'auto', textAlign: 'center', flexShrink: 0 }}>
                  <div style={{ fontSize: '1.6rem', fontWeight: 800, color: '#6366f1' }}>{faces.length}</div>
                  <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', letterSpacing: '.04em' }}>REGISTERED</div>
                </div>
              )}
            </div>
          </div>

          {/* Register new face */}
          <div className="card card-p">
            <h3 style={{ fontSize: '0.95rem', marginBottom: 16, display: 'flex', alignItems: 'center', gap: 8 }}>
              <Upload size={16} color="#6366f1" /> Register New Person
            </h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>

              {/* Left: form */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                <div className="form-group">
                  <label className="form-label">Person Name / Label</label>
                  <input
                    className="form-input"
                    placeholder="e.g. Owner, Rohith, Staff 1"
                    value={faceLabel}
                    onChange={e => setFaceLabel(e.target.value)}
                  />
                </div>

                {/* Drag-drop upload zone */}
                <label style={{
                  display: 'flex', flexDirection: 'column', alignItems: 'center',
                  justifyContent: 'center', gap: 8, padding: '20px 16px',
                  border: `2px dashed ${faceImg ? 'rgba(99,102,241,0.6)' : 'var(--border)'}`,
                  borderRadius: 12, cursor: 'pointer', textAlign: 'center',
                  background: faceImg ? 'rgba(99,102,241,0.06)' : 'rgba(255,255,255,0.02)',
                  transition: 'all 0.2s',
                }}>
                  <span style={{ fontSize: '1.8rem' }}>{faceImg ? '✅' : '📷'}</span>
                  <div style={{ fontSize: '0.82rem', fontWeight: 600, color: faceImg ? '#6366f1' : 'var(--text-secondary)' }}>
                    {faceImg ? 'Photo selected' : 'Click to upload photo'}
                  </div>
                  <div style={{ fontSize: '0.70rem', color: 'var(--text-muted)' }}>
                    JPG / PNG • Clear front-facing photo
                  </div>
                  <input type="file" accept="image/*" style={{ display: 'none' }} onChange={handleFaceImage} />
                </label>

                <button
                  className="btn btn-primary"
                  onClick={registerFace}
                  disabled={saving || !faceImg || !faceLabel.trim()}
                  style={{ background: 'linear-gradient(135deg,#6366f1,#8b5cf6)' }}
                >
                  {saving
                    ? <><span className="spinner" style={{ width:14, height:14, borderWidth:2 }} /> Registering…</>
                    : <><Shield size={14} /> Register Face</>}
                </button>
              </div>

              {/* Right: preview */}
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 10 }}>
                {faceImg ? (
                  <>
                    <img
                      src={faceImg}
                      alt="Preview"
                      style={{ width: 130, height: 130, objectFit: 'cover', borderRadius: '50%',
                               border: '3px solid #6366f1', boxShadow: '0 0 20px rgba(99,102,241,0.3)' }}
                    />
                    <div style={{ fontSize: '0.78rem', color: '#6366f1', fontWeight: 600 }}>{faceLabel || 'No name'}</div>
                    <button className="btn btn-ghost btn-sm" style={{ fontSize: '0.72rem' }}
                      onClick={() => setFaceImg(null)}>✕ Remove</button>
                  </>
                ) : (
                  <div style={{ width: 130, height: 130, borderRadius: '50%',
                                background: 'rgba(99,102,241,0.08)', border: '2px dashed rgba(99,102,241,0.3)',
                                display: 'flex', alignItems: 'center', justifyContent: 'center',
                                fontSize: '2.5rem' }}>👤</div>
                )}
                <div style={{ fontSize: '0.70rem', color: 'var(--text-muted)', textAlign: 'center', maxWidth: 130 }}>
                  Use a clear, well-lit front-facing photo for best accuracy
                </div>
              </div>
            </div>
          </div>

          {/* Registered faces grid */}
          {faces.length > 0 ? (
            <div className="card card-p">
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
                <h3 style={{ fontSize: '0.95rem', margin: 0, display: 'flex', alignItems: 'center', gap: 8 }}>
                  <CheckCircle size={16} color="var(--accent-green)" /> Authorized Persons
                  <span style={{ fontSize: '0.72rem', fontWeight: 700, padding: '2px 9px', borderRadius: 99,
                                 background: 'rgba(16,185,129,0.12)', color: '#10b981',
                                 border: '1px solid rgba(16,185,129,0.3)' }}>
                    {faces.length} registered
                  </span>
                </h3>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(160px,1fr))', gap: 12 }}>
                {faces.map(f => (
                  <div key={f.id} style={{
                    display: 'flex', flexDirection: 'column', alignItems: 'center',
                    padding: '16px 12px 12px', borderRadius: 14, gap: 10,
                    background: 'var(--bg-card)', border: '1px solid var(--border)',
                    position: 'relative', transition: 'box-shadow 0.2s',
                  }}
                    onMouseEnter={e => e.currentTarget.style.boxShadow='0 4px 20px rgba(99,102,241,0.18)'}
                    onMouseLeave={e => e.currentTarget.style.boxShadow='none'}
                  >
                    {/* Delete button */}
                    <button
                      className="btn btn-ghost btn-icon btn-sm"
                      onClick={() => deleteFace(f.id)}
                      title="Remove"
                      style={{ position: 'absolute', top: 8, right: 8, opacity: 0.6,
                               width: 24, height: 24, padding: 0, fontSize: '0.7rem' }}
                    >
                      <Trash2 size={12} />
                    </button>

                    {/* Avatar */}
                    {f.thumbnail_b64 ? (
                      <img
                        src={f.thumbnail_b64}
                        alt={f.label}
                        style={{ width: 80, height: 80, borderRadius: '50%', objectFit: 'cover',
                                 border: '2px solid rgba(99,102,241,0.5)',
                                 boxShadow: '0 2px 12px rgba(99,102,241,0.25)' }}
                      />
                    ) : (
                      <div style={{ width: 80, height: 80, borderRadius: '50%', fontSize: '2rem',
                                   background: 'linear-gradient(135deg,#6366f1,#8b5cf6)',
                                   display: 'flex', alignItems: 'center', justifyContent: 'center' }}>👤</div>
                    )}

                    {/* Authorized badge */}
                    <span style={{ fontSize: '0.62rem', fontWeight: 700, padding: '2px 8px',
                                   borderRadius: 99, background: 'rgba(16,185,129,0.12)',
                                   color: '#10b981', border: '1px solid rgba(16,185,129,0.3)',
                                   letterSpacing: '.04em' }}>✓ AUTHORIZED</span>

                    {/* Editable name */}
                    {editingFaceId === f.id ? (
                      <input
                        autoFocus
                        className="form-input"
                        style={{ textAlign: 'center', fontSize: '0.82rem', padding: '4px 8px',
                                 width: '100%', borderRadius: 8 }}
                        value={editingFaceLabel}
                        onChange={e => setEditingFaceLabel(e.target.value)}
                        onBlur={() => renameFace(f.id, editingFaceLabel)}
                        onKeyDown={e => {
                          if (e.key === 'Enter') renameFace(f.id, editingFaceLabel)
                          if (e.key === 'Escape') setEditingFaceId(null)
                        }}
                      />
                    ) : (
                      <div
                        title="Click to rename"
                        onClick={() => { setEditingFaceId(f.id); setEditingFaceLabel(f.label) }}
                        style={{ fontWeight: 700, fontSize: '0.88rem', cursor: 'text',
                                 color: 'var(--text-primary)', textAlign: 'center',
                                 padding: '2px 6px', borderRadius: 6, width: '100%',
                                 border: '1px solid transparent',
                                 transition: 'border 0.15s' }}
                        onMouseEnter={e => e.currentTarget.style.border='1px solid var(--border)'}
                        onMouseLeave={e => e.currentTarget.style.border='1px solid transparent'}
                      >
                        {f.label}
                      </div>
                    )}

                    <div style={{ fontSize: '0.67rem', color: 'var(--text-muted)' }}>
                      {new Date(f.created_at).toLocaleDateString()}
                    </div>
                  </div>
                ))}
              </div>

              <div style={{ marginTop: 14, fontSize: '0.72rem', color: 'var(--text-muted)',
                            display: 'flex', alignItems: 'center', gap: 6 }}>
                <span>💡</span> Click a name to rename • Hover for delete button
              </div>
            </div>
          ) : (
            <div className="card card-p" style={{ textAlign: 'center', padding: '40px 24px' }}>
              <div style={{ fontSize: '3rem', marginBottom: 12 }}>👥</div>
              <div style={{ fontWeight: 700, color: 'var(--text-primary)', marginBottom: 6 }}>No faces registered yet</div>
              <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                Add your first person above. Once registered, the system will identify them in all camera feeds.
              </div>
            </div>
          )}

          {/* Tips */}
          <div className="card card-p" style={{ background: 'rgba(59,130,246,0.05)', border: '1px solid rgba(59,130,246,0.2)' }}>
            <div style={{ fontSize: '0.85rem', fontWeight: 700, marginBottom: 10, color: 'var(--text-primary)' }}>📋 Tips for Best Accuracy</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {[
                { icon: '📸', tip: 'Use a clear, front-facing photo with good lighting' },
                { icon: '🔢', tip: 'Register the same person multiple times from different angles for better recognition' },
                { icon: '🚫', tip: 'Avoid photos with sunglasses, masks, or heavy blur' },
                { icon: '👥', tip: 'Works with multiple people in the same frame simultaneously' },
                { icon: '⚡', tip: 'Face recognition runs alongside PPE detection in real time' },
              ].map(({ icon, tip }) => (
                <div key={tip} style={{ display: 'flex', gap: 10, fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                  <span>{icon}</span><span>{tip}</span>
                </div>
              ))}
            </div>
          </div>

        </div>
      )}

      {/* ── Cameras Management Tab ── */}
      {activeTab === 'cameras_mgmt' && (
        <CamerasTab addToast={addToast} />
      )}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────
// CamerasTab — LAN discovery + per-camera zone calibration
// ─────────────────────────────────────────────────────────────────
function CamerasTab({ addToast }) {
  const [cameras,          setCameras]          = useState([])
  const [selectedCamera,   setSelectedCamera]   = useState(null)
  const [discovering,      setDiscovering]      = useState(false)
  const [discoveryResults, setDiscoveryResults] = useState(null)
  // Manual RTSP form
  const [addForm,          setAddForm]          = useState({ name: '', floor: 'ground', rtsp_url: '', zone_type: '', camera_code: '', department: '', purpose: '', camera_type: 'rtsp' })
  // HikVision quick-add form
  const [hikForm,          setHikForm]          = useState({ name: '', ip: '', username: 'admin', password: '', floor: 'ground', channel: 1, ai_stream: 'sub', display_stream: 'main' })
  const [addingCamera,     setAddingCamera]     = useState(false)
  const [showAddForm,      setShowAddForm]      = useState(false)
  const [addMode,          setAddMode]          = useState('manual')  // 'manual' | 'hikvision'

  const loadCameras = useCallback(async () => {
    try {
      const res = await camerasApi.list()
      setCameras(res.data)
    } catch {
      addToast('Failed to load cameras', '', 'danger')
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => { loadCameras() }, [loadCameras])

  const discover = async () => {
    setDiscovering(true)
    setDiscoveryResults(null)
    try {
      const res = await camerasApi.discover()
      setDiscoveryResults(res.data)
    } catch {
      addToast('Discovery failed', 'Check network access', 'danger')
    } finally {
      setDiscovering(false)
    }
  }

  const addCamera = async () => {
    if (!addForm.name.trim()) return addToast('Name required', '', 'warning')
    setAddingCamera(true)
    try {
      await camerasApi.create({ ...addForm, status: addForm.rtsp_url ? 'online' : 'offline' })
      addToast('Camera added', addForm.name, 'success')
      setShowAddForm(false)
      setAddForm({ name: '', floor: 'ground', rtsp_url: '', zone_type: '', camera_code: '', department: '', purpose: '', camera_type: 'rtsp' })
      await loadCameras()
    } catch (err) {
      addToast('Add failed', err.response?.data?.detail || '', 'danger')
    } finally {
      setAddingCamera(false)
    }
  }

  const addHikVisionCamera = async () => {
    if (!hikForm.name.trim()) return addToast('Name required', '', 'warning')
    if (!hikForm.ip.trim())   return addToast('IP address required', '', 'warning')
    setAddingCamera(true)
    try {
      const res = await camerasApi.hikVisionAdd({
        ...hikForm,
        channel: Number(hikForm.channel),
      })
      addToast('HikVision camera added', `${res.data.name} — ${res.data.rtsp_url.replace(/\/\/[^@]+@/, '//***@')}`, 'success')
      setShowAddForm(false)
      setHikForm({ name: '', ip: '', username: 'admin', password: '', floor: 'ground', channel: 1, ai_stream: 'sub', display_stream: 'main' })
      await loadCameras()
    } catch (err) {
      addToast('HikVision add failed', err.response?.data?.detail || err.message || '', 'danger')
    } finally {
      setAddingCamera(false)
    }
  }

  const deleteCamera = async (e, id) => {
    e.stopPropagation()
    if (!window.confirm('Remove this camera?')) return
    try {
      await camerasApi.delete(id)
      addToast('Camera removed', '', 'success')
      if (selectedCamera?.id === id) setSelectedCamera(null)
      await loadCameras()
    } catch {
      addToast('Failed to remove camera', '', 'danger')
    }
  }

  const prefillFromDiscovery = (host) => {
    const guess = host.rtsp_guesses?.[0] || ''
    setAddForm(f => ({ ...f, name: host.hostname || host.ip, rtsp_url: guess }))
    setShowAddForm(true)
    setDiscoveryResults(null)
  }

  const STATUS_COLOUR = { online: '#4caf50', offline: '#9e9e9e', error: '#f44336' }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24, maxWidth: 860 }}>

      {/* ── Header row ── */}
      <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
        <h2 style={{ fontSize: '1.1rem', fontWeight: 700, margin: 0 }}>🎥 Camera Management</h2>
        <button className="btn btn-ghost" style={{ marginLeft: 'auto' }} onClick={loadCameras}>↻ Refresh</button>
        <button
          className="btn btn-secondary"
          onClick={discover}
          disabled={discovering}
        >
          {discovering
            ? <><span className="spinner" style={{ width: 14, height: 14, marginRight: 6 }} />Scanning LAN…</>
            : '🔍 Discover on LAN'}
        </button>
        <button className="btn btn-primary" onClick={() => setShowAddForm(v => !v)}>
          {showAddForm ? '✕ Cancel' : '+ Add Camera'}
        </button>
      </div>

      {/* ── LAN discovery results ── */}
      {discoveryResults && (
        <div className="card card-p">
          <div style={{ fontWeight: 700, marginBottom: 10 }}>
            🌐 {discoveryResults.cameras_found.length} device(s) found on {discoveryResults.subnet}
          </div>
          {discoveryResults.cameras_found.length === 0
            ? <div style={{ color: 'var(--text-secondary)', fontSize: 13 }}>No cameras found. Check that cameras are powered and on the same subnet.</div>
            : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {discoveryResults.cameras_found.map(h => (
                  <div key={h.ip} style={{ display: 'flex', gap: 12, alignItems: 'center', padding: '10px 14px', borderRadius: 10, border: '1px solid var(--border)', background: 'var(--bg-card)' }}>
                    <div style={{ flex: 1 }}>
                      <div style={{ fontWeight: 600, fontSize: 14 }}>{h.ip}</div>
                      {h.hostname && <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>{h.hostname}</div>}
                      <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginTop: 2 }}>Port {h.open_port} open</div>
                    </div>
                    <div style={{ fontSize: 11, color: '#888', flex: 1 }}>
                      {h.rtsp_guesses?.[0] && <code style={{ fontSize: 10 }}>{h.rtsp_guesses[0].replace('<user>:<pass>@', '')}</code>}
                    </div>
                    <button className="btn btn-primary" style={{ fontSize: 12, padding: '5px 14px' }} onClick={() => prefillFromDiscovery(h)}>Add</button>
                  </div>
                ))}
              </div>
            )}
        </div>
      )}

      {/* ── Add camera form ── */}
      {showAddForm && (
        <div className="card card-p">
          {/* Mode toggle */}
          <div style={{ display: 'flex', gap: 6, marginBottom: 16 }}>
            {[['manual', '📡 Manual RTSP'], ['hikvision', '📷 HikVision Quick-Add']].map(([m, label]) => (
              <button key={m} className={`btn ${addMode === m ? 'btn-primary' : 'btn-ghost'}`}
                style={{ fontSize: '0.78rem', padding: '5px 14px' }}
                onClick={() => setAddMode(m)}>{label}</button>
            ))}
          </div>

          {addMode === 'manual' ? (
            <>
              <div style={{ fontWeight: 700, marginBottom: 14 }}>+ Add Camera (Manual RTSP)</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                <div>
                  <label style={{ fontSize: 12, color: 'var(--text-secondary)' }}>Display Name *</label>
                  <input className="input" value={addForm.name} onChange={e => setAddForm(f => ({ ...f, name: e.target.value }))} placeholder="e.g. Ground Floor Entrance" />
                </div>
                <div>
                  <label style={{ fontSize: 12, color: 'var(--text-secondary)' }}>Floor</label>
                  <input className="input" list="camera-floors" value={addForm.floor} onChange={e => setAddForm(f => ({ ...f, floor: e.target.value }))} placeholder="e.g. ground or third" />
                  <datalist id="camera-floors">
                    <option value="ground" /><option value="first" /><option value="second" /><option value="shop" /><option value="store" />
                  </datalist>
                </div>
                <div style={{ gridColumn: 'span 2' }}>
                  <label style={{ fontSize: 12, color: 'var(--text-secondary)' }}>RTSP / HTTP URL</label>
                  <input className="input" value={addForm.rtsp_url} onChange={e => setAddForm(f => ({ ...f, rtsp_url: e.target.value }))} placeholder="rtsp://user:pass@192.168.1.x:554/Streaming/Channels/101" />
                </div>
                <div>
                  <label style={{ fontSize: 12, color: 'var(--text-secondary)' }}>Zone Type (for idle limit)</label>
                  <input className="input" value={addForm.zone_type} onChange={e => setAddForm(f => ({ ...f, zone_type: e.target.value }))} placeholder="shop / default / cashbox…" />
                </div>
                <div>
                  <label style={{ fontSize: 12, color: 'var(--text-secondary)' }}>Camera ID</label>
                  <input className="input" value={addForm.camera_code} onChange={e => setAddForm(f => ({ ...f, camera_code: e.target.value }))} placeholder="e.g. GF-PACK-01" />
                </div>
                <div>
                  <label style={{ fontSize: 12, color: 'var(--text-secondary)' }}>Department</label>
                  <input className="input" value={addForm.department} onChange={e => setAddForm(f => ({ ...f, department: e.target.value }))} placeholder="e.g. Packing" />
                </div>
                <div>
                  <label style={{ fontSize: 12, color: 'var(--text-secondary)' }}>Purpose</label>
                  <input className="input" value={addForm.purpose} onChange={e => setAddForm(f => ({ ...f, purpose: e.target.value }))} placeholder="e.g. Workflow and PPE" />
                </div>
              </div>
              <button className="btn btn-primary" style={{ marginTop: 14 }} onClick={addCamera} disabled={addingCamera}>
                {addingCamera ? 'Adding…' : 'Add Camera'}
              </button>
            </>
          ) : (
            <>
              <div style={{ fontWeight: 700, marginBottom: 4 }}>📷 HikVision Quick-Add</div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: 14 }}>
                Enter IP and credentials — the RTSP URL is built automatically.<br />
                Works with all HikVision NVR/IP cameras using standard ONVIF/RTSP ports.
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                <div>
                  <label style={{ fontSize: 12, color: 'var(--text-secondary)' }}>Display Name *</label>
                  <input className="input" value={hikForm.name} onChange={e => setHikForm(f => ({ ...f, name: e.target.value }))} placeholder="e.g. Entrance Cam" />
                </div>
                <div>
                  <label style={{ fontSize: 12, color: 'var(--text-secondary)' }}>Camera IP Address *</label>
                  <input className="input" value={hikForm.ip} onChange={e => setHikForm(f => ({ ...f, ip: e.target.value }))} placeholder="192.168.1.64" />
                </div>
                <div>
                  <label style={{ fontSize: 12, color: 'var(--text-secondary)' }}>Username</label>
                  <input className="input" value={hikForm.username} onChange={e => setHikForm(f => ({ ...f, username: e.target.value }))} placeholder="admin" />
                </div>
                <div>
                  <label style={{ fontSize: 12, color: 'var(--text-secondary)' }}>Password</label>
                  <input className="input" type="password" value={hikForm.password} onChange={e => setHikForm(f => ({ ...f, password: e.target.value }))} placeholder="••••••••" />
                </div>
                <div>
                  <label style={{ fontSize: 12, color: 'var(--text-secondary)' }}>Floor</label>
                  <input className="input" list="hik-floors" value={hikForm.floor} onChange={e => setHikForm(f => ({ ...f, floor: e.target.value }))} placeholder="ground" />
                  <datalist id="hik-floors">
                    <option value="ground" /><option value="first" /><option value="second" /><option value="shop" />
                  </datalist>
                </div>
                <div>
                  <label style={{ fontSize: 12, color: 'var(--text-secondary)' }}>Channel (1–32)</label>
                  <input className="input" type="number" min={1} max={32} value={hikForm.channel} onChange={e => setHikForm(f => ({ ...f, channel: e.target.value }))} />
                </div>
                <div>
                  <label style={{ fontSize: 12, color: 'var(--text-secondary)' }}>AI Stream</label>
                  <select className="input" value={hikForm.ai_stream} onChange={e => setHikForm(f => ({ ...f, ai_stream: e.target.value }))}>
                    <option value="sub">Sub-stream (D1 / recommended for AI)</option>
                    <option value="main">Main-stream (HD)</option>
                  </select>
                </div>
                <div>
                  <label style={{ fontSize: 12, color: 'var(--text-secondary)' }}>Display Stream</label>
                  <select className="input" value={hikForm.display_stream} onChange={e => setHikForm(f => ({ ...f, display_stream: e.target.value }))}>
                    <option value="main">Main-stream (HD — recommended for display)</option>
                    <option value="sub">Sub-stream</option>
                  </select>
                </div>
              </div>
              <div style={{ marginTop: 12, padding: '8px 12px', borderRadius: 8, background: 'rgba(59,130,246,0.07)', border: '1px solid rgba(59,130,246,0.2)', fontSize: '0.73rem', color: 'var(--text-muted)' }}>
                💡 URL preview: <code style={{ color: 'var(--accent-blue)' }}>rtsp://{hikForm.username || 'admin'}:***@{hikForm.ip || '192.168.x.x'}:554/Streaming/Channels/{hikForm.channel || 1}{hikForm.ai_stream === 'main' ? '01' : '02'}</code>
              </div>
              <button className="btn btn-primary" style={{ marginTop: 14 }} onClick={addHikVisionCamera} disabled={addingCamera || !hikForm.ip}>
                {addingCamera ? 'Adding…' : '📡 Add HikVision Camera'}
              </button>
            </>
          )}
        </div>
      )}

      {/* ── Camera list ── */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        {cameras.length === 0 && <div style={{ color: 'var(--text-secondary)', textAlign: 'center', padding: 24 }}>No cameras configured yet.</div>}
        {cameras.map(cam => (
          <div key={cam.id} className="card card-p" style={{ cursor: 'pointer', border: selectedCamera?.id === cam.id ? '1px solid var(--accent)' : undefined }}>
            <div style={{ display: 'flex', gap: 12, alignItems: 'center' }} onClick={() => setSelectedCamera(selectedCamera?.id === cam.id ? null : cam)}>
              <div style={{ width: 12, height: 12, borderRadius: '50%', background: STATUS_COLOUR[cam.status] || '#9e9e9e', flexShrink: 0 }} />
              <div style={{ flex: 1 }}>
                <div style={{ fontWeight: 700 }}>{cam.name}</div>
                <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                  {cam.floor} floor • {cam.zone_type || 'general'} • {cam.status}
                  {cam.rtsp_url && <span> &bull; <code style={{ fontSize: 10 }}>{cam.rtsp_url.replace(/\/\/[^@]+@/, '//***@')}</code></span>}
                </div>
              </div>
              <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>{selectedCamera?.id === cam.id ? '▲ Calibrate' : '▼ Calibrate'}</div>
              <button 
                className="btn btn-ghost btn-sm btn-icon"
                onClick={(e) => deleteCamera(e, cam.id)}
                title="Remove Camera"
                style={{ color: '#ef4444', padding: 6 }}
              >
                <Trash2 size={16} />
              </button>
            </div>

            {/* Zone calibration panel (inline) */}
            {selectedCamera?.id === cam.id && (
              <div style={{ marginTop: 20, paddingTop: 20, borderTop: '1px solid var(--border)' }}>
                <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 14 }}>🗺 Zone Calibration — {cam.name}</div>
                <ZonePainter cameraId={cam.id} onSaved={loadCameras} />
              </div>
            )}
          </div>
        ))}
      </div>

    </div>
  )
}
