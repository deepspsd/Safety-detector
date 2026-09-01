import React, { useEffect, useState } from 'react'
import { X, ExternalLink, Activity, Wifi, ShieldAlert, Cpu } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import api from '../api/api'

export default function CameraFullscreenModal({ camera, onClose }) {
  const navigate = useNavigate()
  const [diagnostics, setDiagnostics] = useState(null)
  const BASE = import.meta.env.VITE_API_BASE_URL?.replace(/\/api$/, '') || ''
  const streamUrl = `${BASE}/api/cameras/${camera?.id}/stream`

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [onClose])

  useEffect(() => {
    if (!camera?.id) return
    let active = true
    const fetchDiags = async () => {
      try {
        const res = await api.get(`/cameras/${camera.id}/diagnostics`)
        if (active) setDiagnostics(res.data)
      } catch (err) {
        // ignore telemetry poll errors
      }
    }
    fetchDiags()
    const iv = setInterval(fetchDiags, 3000)
    return () => {
      active = false
      clearInterval(iv)
    }
  }, [camera?.id])

  if (!camera) return null

  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100vw',
        height: '100vh',
        backgroundColor: 'rgba(5, 7, 15, 0.88)',
        backdropFilter: 'blur(10px)',
        zIndex: 9999,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '24px',
      }}
      onClick={onClose}
    >
      <div
        style={{
          position: 'relative',
          width: '100%',
          maxWidth: '1200px',
          background: 'var(--bg-card, #111827)',
          border: '1px solid var(--border, #1f2937)',
          borderRadius: '16px',
          overflow: 'hidden',
          boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.7)',
          display: 'flex',
          flexDirection: 'column',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '16px 20px',
            borderBottom: '1px solid var(--border, #1f2937)',
            background: 'rgba(17, 24, 39, 0.95)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <span
              style={{
                width: '10px',
                height: '10px',
                borderRadius: '50%',
                background: camera.status === 'online' ? '#10b981' : '#ef4444',
                boxShadow: camera.status === 'online' ? '0 0 8px #10b981' : 'none',
              }}
            />
            <h3 style={{ margin: 0, fontSize: '1.1rem', fontWeight: 600, color: '#f3f4f6' }}>
              {camera.name}
            </h3>
            <span
              style={{
                fontSize: '0.75rem',
                padding: '3px 8px',
                borderRadius: '6px',
                background: 'rgba(59, 130, 246, 0.15)',
                color: '#60a5fa',
                textTransform: 'uppercase',
                fontWeight: 600,
                letterSpacing: '0.04em',
              }}
            >
              Floor: {camera.floor || 'ground'}
            </span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <button
              onClick={() => {
                onClose()
                navigate(`/monitor?camera=${camera.id}`)
              }}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                padding: '6px 12px',
                borderRadius: '8px',
                background: 'var(--accent-blue, #3b82f6)',
                color: '#fff',
                border: 'none',
                cursor: 'pointer',
                fontSize: '0.8rem',
                fontWeight: 500,
              }}
            >
              <ExternalLink size={14} /> Open Live Monitor
            </button>
            <button
              onClick={onClose}
              style={{
                background: 'transparent',
                border: 'none',
                color: '#9ca3af',
                cursor: 'pointer',
                padding: '6px',
                borderRadius: '8px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <X size={20} />
            </button>
          </div>
        </div>

        {/* Video Canvas Container */}
        <div
          style={{
            position: 'relative',
            width: '100%',
            background: '#000',
            aspectRatio: '16/9',
            maxHeight: '68vh',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <img
            src={streamUrl}
            alt={camera.name}
            style={{
              width: '100%',
              height: '100%',
              objectFit: 'contain',
            }}
          />
        </div>

        {/* Telemetry Footer */}
        <div
          style={{
            padding: '12px 20px',
            background: 'rgba(15, 23, 42, 0.9)',
            borderTop: '1px solid var(--border, #1f2937)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            fontSize: '0.8rem',
            color: '#94a3b8',
            flexWrap: 'wrap',
            gap: '12px',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Activity size={14} color="#34d399" />
              FPS: <strong style={{ color: '#f3f4f6' }}>{diagnostics?.metrics?.fps?.toFixed(1) || 'Live'}</strong>
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Wifi size={14} color="#60a5fa" />
              Stream: <strong style={{ color: '#f3f4f6' }}>{diagnostics?.metrics?.stream_status || camera.status}</strong>
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Cpu size={14} color="#a78bfa" />
              Head-Cap Tracks: <strong style={{ color: '#f3f4f6' }}>{diagnostics?.headcap_diagnostics?.length || 0}</strong>
            </span>
          </div>

          <span style={{ fontSize: '0.72rem', color: '#64748b' }}>
            Press <kbd style={{ background: '#1e293b', padding: '2px 5px', borderRadius: '4px' }}>ESC</kbd> to close
          </span>
        </div>
      </div>
    </div>
  )
}
