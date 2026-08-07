import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { useToast } from '../context/ToastContext'
import { usersApi } from '../api/api'
import { UserCircle, Stethoscope, Shield, HardHat, GraduationCap, Home, ChevronRight, CheckCircle, ChevronLeft, UtensilsCrossed } from 'lucide-react'

const ROLES = [
  { id: 'Bakery Worker',       label: 'Bakery Worker',       desc: 'Head cap, mask, gloves, no bangles (food safety)', icon: UtensilsCrossed, color: '#f97316' },
  { id: 'Doctor',              label: 'Doctor',              desc: 'Detects gloves & lab coat compliance',              icon: Stethoscope,      color: '#06b6d4' },
  { id: 'Traffic Police',      label: 'Traffic Police',      desc: 'Monitors helmet usage',                             icon: Shield,           color: '#8b5cf6' },
  { id: 'Construction Worker', label: 'Construction Worker', desc: 'Checks helmet & safety vest',                       icon: HardHat,          color: '#f59e0b' },
  { id: 'College',             label: 'College',             desc: 'Verifies ID card & uniform',                        icon: GraduationCap,    color: '#10b981' },
  { id: 'Home',                label: 'Home User',           desc: 'Face recognition security',                         icon: Home,             color: '#3b82f6' },
]

export default function Profile() {
  const { user, updateUser } = useAuth()
  const { addToast } = useToast()
  const navigate = useNavigate()
  const [name, setName] = useState(user?.name || '')
  const [role, setRole] = useState(user?.role || '')
  const [saving, setSaving] = useState(false)

  const handleSave = async () => {
    if (!role) return addToast('Select a role', 'Please choose your monitoring role', 'danger')
    setSaving(true)
    try {
      await usersApi.updateProfile({ name, role })
      updateUser({ name, role })
      addToast('Profile saved!', `Role set to ${role}`, 'success')
      navigate('/dashboard')
    } catch {
      addToast('Save failed', 'Please try again', 'danger')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="auth-bg" style={{ alignItems: 'flex-start', paddingTop: 40 }}>
      <div style={{ width: '100%', maxWidth: 560 }}>
        <div className="auth-card" style={{ maxWidth: '100%' }}>
          <div style={{ marginBottom: 16 }}>
            <button
              onClick={() => navigate(-1)}
              style={{
                background: 'transparent', border: 'none',
                color: 'var(--text-muted)', cursor: 'pointer',
                display: 'flex', alignItems: 'center', padding: 0,
                fontSize: '0.9rem', fontWeight: 500, transition: 'color 0.2s'
              }}
              onMouseOver={e => e.currentTarget.style.color = 'var(--text-primary)'}
              onMouseOut={e => e.currentTarget.style.color = 'var(--text-muted)'}
            >
              <ChevronLeft size={18} style={{ marginRight: 4 }} />
              Back
            </button>
          </div>

          <div className="auth-header">
            <div className="auth-logo">
              <UserCircle size={26} color="#fff" />
            </div>
            <h1 className="auth-title">Setup Your Profile</h1>
            <p className="auth-subtitle">
              Tell us about yourself so we can tailor AI detection to your role.
            </p>
          </div>

          <div className="form-group" style={{ marginBottom: 24 }}>
            <label className="form-label">Your Name</label>
            <input className="form-input" type="text" placeholder="Full name"
              value={name} onChange={e => setName(e.target.value)} />
          </div>

          <div className="form-group">
            <label className="form-label">Select Your Role</label>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginTop: 6 }}>
              {ROLES.map(r => {
                const Icon = r.icon
                const selected = role === r.id
                return (
                  <button key={r.id} type="button" onClick={() => setRole(r.id)}
                    style={{
                      display: 'flex', alignItems: 'center', gap: 14,
                      padding: '14px 16px', borderRadius: 12,
                      background: selected ? `rgba(${hexToRgb(r.color)},0.12)` : 'rgba(255,255,255,0.04)',
                      border: `1.5px solid ${selected ? r.color : 'rgba(255,255,255,0.08)'}`,
                      cursor: 'pointer', transition: 'all 0.2s', textAlign: 'left',
                      boxShadow: selected ? `0 0 16px rgba(${hexToRgb(r.color)},0.2)` : 'none'
                    }}>
                    <div style={{
                      width: 40, height: 40, borderRadius: 10,
                      background: `rgba(${hexToRgb(r.color)},0.15)`,
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      color: r.color, flexShrink: 0
                    }}>
                      <Icon size={20} />
                    </div>
                    <div style={{ flex: 1 }}>
                      <div style={{ fontWeight: 600, fontSize: '0.9rem', color: 'var(--text-primary)' }}>{r.label}</div>
                      <div style={{ fontSize: '0.76rem', color: 'var(--text-secondary)', marginTop: 2 }}>{r.desc}</div>
                    </div>
                    {selected && <CheckCircle size={18} color={r.color} />}
                  </button>
                )
              })}
            </div>
          </div>

          <button className="btn btn-primary btn-lg" onClick={handleSave} disabled={saving || !role}
            style={{ marginTop: 24, width: '100%', justifyContent: 'center' }}>
            {saving ? <span className="spinner" style={{ width:18, height:18, borderWidth:2 }} /> : <>Save & Continue <ChevronRight size={16} /></>}
          </button>
        </div>
      </div>
    </div>
  )
}

function hexToRgb(hex) {
  const r = parseInt(hex.slice(1,3),16)
  const g = parseInt(hex.slice(3,5),16)
  const b = parseInt(hex.slice(5,7),16)
  return `${r},${g},${b}`
}
