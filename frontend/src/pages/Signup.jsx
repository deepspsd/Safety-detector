import { useState, useEffect } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { Shield, Mail, Lock, User, MapPin, HardHat, Loader, CheckCircle } from 'lucide-react'

const ROLES = [
  { id: 'Bakery Worker',       icon: '🧁', label: 'Bakery Worker',       desc: 'Head cap, mask, gloves, no bangles — food safety' },
  { id: 'Construction Worker', icon: '🏗️', label: 'Construction Worker', desc: 'Hardhat, vest, gloves, mask, goggles, shoes' },
  { id: 'Doctor',              icon: '🩺', label: 'Doctor / Medical',     desc: 'Mask, gloves, lab coat detection'           },
  { id: 'Traffic Police',      icon: '🚓', label: 'Traffic Police',        desc: 'Helmet & safety compliance'                },
  { id: 'College',             icon: '🎓', label: 'College / Campus',      desc: 'ID card & uniform monitoring'              },
  { id: 'Home',                icon: '🏠', label: 'Home / Private',        desc: 'Face recognition — unknown person alert'   },
  { id: 'None',                icon: '🛠️', label: 'Custom / None',         desc: 'Define your own PPE rules in Settings'     },
]

// Map OSM place types → role suggestions
function detectRoleFromPlace(tags = {}) {
  const t = JSON.stringify(tags).toLowerCase()
  if (t.includes('hospital') || t.includes('clinic') || t.includes('doctor') || t.includes('medical'))
    return { role: 'Doctor', reason: 'Detected a healthcare facility nearby' }
  if (t.includes('construction') || t.includes('industrial') || t.includes('factory') || t.includes('building'))
    return { role: 'Construction Worker', reason: 'Detected a construction/industrial site nearby' }
  if (t.includes('university') || t.includes('college') || t.includes('school') || t.includes('campus') || t.includes('education'))
    return { role: 'College', reason: 'Detected an educational institution nearby' }
  if (t.includes('traffic') || t.includes('police') || t.includes('junction') || t.includes('highway'))
    return { role: 'Traffic Police', reason: 'Detected a traffic/road area nearby' }
  return null
}

export default function Signup() {
  const { signup } = useAuth()
  const navigate   = useNavigate()

  const [form,        setForm]        = useState({ name: '', email: '', password: '' })
  const [role,        setRole]        = useState('Construction Worker')
  const [error,       setError]       = useState('')
  const [loading,     setLoading]     = useState(false)
  const [locStatus,   setLocStatus]   = useState('idle') // idle | detecting | done | denied
  const [suggestion,  setSuggestion]  = useState(null)   // { role, reason }
  const [locLabel,    setLocLabel]    = useState('')

  // ── Geolocation on mount ────────────────────────────────────
  useEffect(() => {
    detectLocation()
  }, [])

  async function detectLocation() {
    if (!navigator.geolocation) return
    setLocStatus('detecting')
    navigator.geolocation.getCurrentPosition(
      async (pos) => {
        try {
          const { latitude, longitude } = pos.coords
          // Reverse geocode with Nominatim (free, no API key)
          const res = await fetch(
            `https://nominatim.openstreetmap.org/reverse?lat=${latitude}&lon=${longitude}&format=json&addressdetails=1`,
            { headers: { 'Accept-Language': 'en' } }
          )
          const data = await res.json()
          setLocLabel(data.display_name?.split(',').slice(0, 3).join(', ') || 'Current location')

          // Try to detect role from address tags
          const suggestion = detectRoleFromPlace(data)
          if (suggestion) {
            setSuggestion(suggestion)
            setRole(suggestion.role) // auto-select suggested role
          }
          setLocStatus('done')

          // Also try nearby amenities
          const q = `[out:json][timeout:10];
            (node(around:500,${latitude},${longitude})[amenity~"hospital|clinic|school|university|police|construction"];
             node(around:500,${latitude},${longitude})[building~"industrial|construction"];);
            out 5;`
          const ovRes = await fetch('https://overpass-api.de/api/interpreter', {
            method: 'POST', body: `data=${encodeURIComponent(q)}`
          })
          if (ovRes.ok) {
            const ovData = await ovRes.json()
            for (const el of ovData.elements || []) {
              const s = detectRoleFromPlace(el.tags || {})
              if (s) {
                setSuggestion(s)
                setRole(s.role)
                break
              }
            }
          }
        } catch {
          setLocStatus('done')
        }
      },
      () => setLocStatus('denied')
    )
  }

  const set = (k) => (e) => setForm(f => ({ ...f, [k]: e.target.value }))

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    if (!form.name.trim())                     return setError('Full name is required')
    if (!form.email.includes('@'))             return setError('Enter a valid email address')
    if (form.password.length < 6)             return setError('Password must be at least 6 characters')

    setLoading(true)
    try {
      await signup(form.email, form.password, form.name, role)
      if (role === 'None') {
        navigate('/setup-rules')
      } else {
        navigate('/monitor')
      }
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || 'Account creation failed'
      setError(typeof msg === 'string' ? msg : JSON.stringify(msg))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="auth-page">
      <div className="auth-card" style={{ maxWidth: 520 }}>
        {/* Header */}
        <div className="auth-header">
          <div className="auth-logo">
            <Shield size={22} color="#fff" />
          </div>
          <h1 className="auth-title">Occupational Safety Monitoring</h1>
          <p className="auth-subtitle">Create your safety monitoring account</p>
        </div>

        {/* Location detection banner */}
        <div style={{
          padding: '10px 14px', borderRadius: 10, marginBottom: 20,
          background: locStatus === 'done'
            ? 'rgba(16,185,129,0.08)' : locStatus === 'denied'
            ? 'rgba(239,68,68,0.08)'  : 'rgba(249,115,22,0.08)',
          border: `1px solid ${locStatus === 'done' ? 'rgba(16,185,129,0.3)' : locStatus === 'denied' ? 'rgba(239,68,68,0.3)' : 'rgba(249,115,22,0.3)'}`,
          display: 'flex', alignItems: 'flex-start', gap: 10,
        }}>
          {locStatus === 'detecting' && <Loader size={16} style={{ color:'#fb923c', flexShrink:0, marginTop:2, animation:'spin 1s linear infinite' }} />}
          {locStatus === 'done'      && <CheckCircle size={16} style={{ color:'#10b981', flexShrink:0, marginTop:2 }} />}
          {locStatus === 'denied'    && <MapPin size={16} style={{ color:'#ef4444', flexShrink:0, marginTop:2 }} />}
          {locStatus === 'idle'      && <MapPin size={16} style={{ color:'#fb923c', flexShrink:0, marginTop:2 }} />}
          <div style={{ flex:1 }}>
            {locStatus === 'detecting' && <div style={{ fontSize:'0.8rem', color:'var(--text-secondary)' }}>Detecting your location to suggest the right role…</div>}
            {locStatus === 'denied'    && <div style={{ fontSize:'0.8rem', color:'var(--text-muted)' }}>Location access denied — please select your role manually.</div>}
            {locStatus === 'done' && !suggestion && (
              <div style={{ fontSize:'0.8rem', color:'var(--text-secondary)' }}>
                📍 {locLabel || 'Location detected'} — Select your role below.
              </div>
            )}
            {locStatus === 'done' && suggestion && (
              <>
                <div style={{ fontSize:'0.8rem', fontWeight:600, color:'#10b981' }}>
                  📍 {suggestion.reason}
                </div>
                <div style={{ fontSize:'0.75rem', color:'var(--text-muted)', marginTop:2 }}>
                  {locLabel} · Auto-selected: <strong>{suggestion.role}</strong>
                  &nbsp;—&nbsp;
                  <button onClick={() => setSuggestion(null)}
                    style={{ background:'none', border:'none', color:'var(--text-secondary)', cursor:'pointer', fontSize:'0.75rem', textDecoration:'underline' }}>
                    change
                  </button>
                </div>
              </>
            )}
            {locStatus === 'idle' && (
              <div style={{ fontSize:'0.8rem', color:'var(--text-muted)' }}>
                Enable location for smart role suggestion
                &nbsp;·&nbsp;
                <button onClick={detectLocation} style={{ background:'none', border:'none', color:'#fb923c', cursor:'pointer', fontSize:'0.8rem' }}>
                  Detect
                </button>
              </div>
            )}
          </div>
        </div>

        <form onSubmit={handleSubmit}>
          {/* Fields */}
          <div style={{ display:'flex', flexDirection:'column', gap:14, marginBottom:20 }}>
            <div className="form-group">
              <label className="form-label">FULL NAME</label>
              <div className="input-wrapper">
                <User size={16} className="input-icon" />
                <input className="form-input" type="text" placeholder="Enter your name"
                  value={form.name} onChange={set('name')} required />
              </div>
            </div>
            <div className="form-group">
              <label className="form-label">EMAIL ADDRESS</label>
              <div className="input-wrapper">
                <Mail size={16} className="input-icon" />
                <input className="form-input" type="email" placeholder="you@example.com"
                  value={form.email} onChange={set('email')} required />
              </div>
            </div>
            <div className="form-group">
              <label className="form-label">PASSWORD</label>
              <div className="input-wrapper">
                <Lock size={16} className="input-icon" />
                <input className="form-input" type="password" placeholder="Min 6 characters"
                  value={form.password} onChange={set('password')} required />
              </div>
            </div>
          </div>

          {/* Role selector */}
          <div style={{ marginBottom: 20 }}>
            <label className="form-label" style={{ display:'flex', alignItems:'center', gap:6 }}>
              <HardHat size={13} /> SELECT YOUR ROLE
            </label>
            <div style={{ display:'flex', flexDirection:'column', gap:8, marginTop:6 }}>
              {ROLES.map(r => (
                <label key={r.id} style={{
                  display: 'flex', alignItems: 'center', gap: 12,
                  padding: '10px 14px', borderRadius: 10, cursor: 'pointer',
                  border: `1px solid ${role === r.id ? 'var(--accent-construction)' : 'var(--border)'}`,
                  background: role === r.id ? 'rgba(249,115,22,0.08)' : 'var(--bg-card)',
                  transition: 'all 0.15s',
                }}>
                  <input type="radio" name="role" value={r.id}
                    checked={role === r.id} onChange={() => setRole(r.id)}
                    style={{ accentColor: '#f97316' }} />
                  <span style={{ fontSize: '1.2rem' }}>{r.icon}</span>
                  <div>
                    <div style={{ fontSize:'0.85rem', fontWeight:600, color:'var(--text-primary)' }}>
                      {r.label}
                      {role === r.id && suggestion?.role === r.id && (
                        <span style={{ marginLeft:8, fontSize:'0.7rem', background:'rgba(16,185,129,0.15)',
                          color:'#10b981', padding:'2px 6px', borderRadius:4 }}>
                          📍 Suggested
                        </span>
                      )}
                    </div>
                    <div style={{ fontSize:'0.72rem', color:'var(--text-muted)', marginTop:2 }}>{r.desc}</div>
                  </div>
                </label>
              ))}
            </div>
          </div>

          {error && (
            <div className="form-error" style={{ marginBottom: 14 }}>{error}</div>
          )}

          <button type="submit" className="btn btn-primary btn-block" disabled={loading}>
            {loading ? <><span className="spinner" /> Creating account…</> : 'Create Account'}
          </button>
        </form>

        <p style={{ textAlign:'center', marginTop:18, fontSize:'0.85rem', color:'var(--text-muted)' }}>
          Already have an account?{' '}
          <Link to="/login" style={{ color:'var(--accent-construction)', fontWeight:600 }}>Sign in</Link>
        </p>
      </div>
    </div>
  )
}
