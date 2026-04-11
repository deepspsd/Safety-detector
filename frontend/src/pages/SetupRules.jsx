import { useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { useToast } from '../context/ToastContext'
import { Shield, CheckCircle } from 'lucide-react'
import { usersApi } from '../api/api'

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

export default function SetupRules() {
  const navigate      = useNavigate()
  const { addToast }  = useToast()
  const { setCustomPpeItems } = useAuth()

  const [selected, setSelected] = useState([])
  const [saving,   setSaving]   = useState(false)

  const toggle = (id) =>
    setSelected(prev => prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id])

  const handleSave = async () => {
    if (selected.length === 0) return addToast('Nothing selected', 'Pick at least one PPE item', 'danger')
    setSaving(true)
    try {
      await usersApi.updateCustomPpe(selected)
      setCustomPpeItems(selected)
      addToast('Safety rules saved!', `${selected.length} item(s) will be monitored`, 'success')
      navigate('/monitor')
    } catch {
      addToast('Save failed', 'Could not save rules — try Settings later', 'danger')
      navigate('/monitor')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="auth-page" style={{ alignItems: 'flex-start', paddingTop: 40 }}>
      <div className="auth-card" style={{ maxWidth: 560 }}>

        {/* Header */}
        <div className="auth-header">
          <div className="auth-logo">
            <Shield size={26} color="#fff" />
          </div>
          <h1 className="auth-title">Set Your Safety Rules</h1>
          <p className="auth-subtitle">
            Select which PPE items you want the system to monitor and alert on.
            You can change these anytime in <strong>Settings → Safety Rules</strong>.
          </p>
        </div>

        {/* PPE grid */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(230px,1fr))', gap: 10, marginBottom: 28 }}>
          {ALL_PPE.map(item => {
            const on = selected.includes(item.id)
            return (
              <label
                key={item.id}
                onClick={() => toggle(item.id)}
                style={{
                  display: 'flex', alignItems: 'center', gap: 14,
                  padding: '13px 16px', borderRadius: 12, cursor: 'pointer',
                  background: on ? 'rgba(16,185,129,0.08)' : 'rgba(255,255,255,0.04)',
                  border: `1.5px solid ${on ? 'rgba(16,185,129,0.45)' : 'rgba(255,255,255,0.08)'}`,
                  transition: 'all 0.15s', userSelect: 'none',
                }}
              >
                <span style={{ fontSize: '1.5rem', flexShrink: 0 }}>{item.icon}</span>
                <div style={{ flex: 1 }}>
                  <div style={{
                    fontWeight: 600, fontSize: '0.875rem',
                    color: on ? 'var(--accent-green)' : 'var(--text-primary)',
                  }}>
                    {item.label}
                  </div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: 2 }}>{item.desc}</div>
                </div>
                <div style={{
                  width: 22, height: 22, borderRadius: '50%', flexShrink: 0,
                  border: `2px solid ${on ? '#10b981' : 'rgba(255,255,255,0.15)'}`,
                  background: on ? '#10b981' : 'transparent',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  transition: 'all 0.15s',
                }}>
                  {on && <CheckCircle size={13} color="#fff" />}
                </div>
              </label>
            )
          })}
        </div>

        {/* Actions */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          <button
            className="btn btn-primary btn-block"
            onClick={handleSave}
            disabled={saving || selected.length === 0}
          >
            {saving
              ? <><span className="spinner" style={{ width:16,height:16,borderWidth:2 }} /> Saving…</>
              : `Start Monitoring (${selected.length} selected)`}
          </button>
          <button
            className="btn btn-ghost btn-block"
            onClick={() => navigate('/monitor')}
            style={{ fontSize: '0.82rem' }}
          >
            Skip — I'll configure later in Settings
          </button>
        </div>

        <p style={{ textAlign: 'center', marginTop: 18, fontSize: '0.78rem', color: 'var(--text-muted)' }}>
          These rules apply only to your account.
        </p>
      </div>
    </div>
  )
}
