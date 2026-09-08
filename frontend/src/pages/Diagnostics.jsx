import { useState, useEffect } from 'react'
import { Activity, ShieldCheck, AlertCircle, RefreshCw, Cpu, Layers, Camera, CheckCircle, XCircle, Info } from 'lucide-react'
import api from '../api/api'

export default function Diagnostics() {
  const [cameras, setCameras] = useState([])
  const [selectedCamId, setSelectedCamId] = useState(null)
  const [diagnostics, setDiagnostics] = useState(null)
  const [modelHealth, setModelHealth] = useState([])
  const [_loading, setLoading] = useState(true)  // Prefixed with _ to indicate intentionally unused
  const [autoRefresh, setAutoRefresh] = useState(true)

  const BASE = import.meta.env.VITE_API_BASE_URL?.replace(/\/api$/, '') || ''

  const loadCameras = async () => {
    try {
      const res = await api.get('/cameras')
      const list = Array.isArray(res.data) ? res.data : (res.data.cameras || [])
      setCameras(list)
      if (list.length > 0 && !selectedCamId) {
        setSelectedCamId(list[0].id)
      }
    } catch (err) {
      console.error('Failed to load cameras for diagnostics', err)
    } finally {
      setLoading(false)
    }
  }

  const loadModelHealth = async () => {
    try {
      const res = await api.get('/detection/models/health')
      if (res.data?.models) {
        setModelHealth(res.data.models)
      }
    } catch (err) {
      console.error('Failed to load model health', err)
    }
  }

  const toggleModelLoad = async (modelName, isLoaded) => {
    try {
      const action = isLoaded ? 'unload' : 'load'
      await api.post(`/detection/models/${modelName}/${action}`)
      await loadModelHealth()
    } catch (err) {
      console.error(`Failed to ${isLoaded ? 'unload' : 'load'} model ${modelName}`, err)
    }
  }

  const loadDiagnostics = async (camId) => {
    if (!camId) return
    try {
      const res = await api.get(`/cameras/${camId}/diagnostics`)
      setDiagnostics(res.data)
    } catch (err) {
      console.error('Failed to load diagnostics', err)
    }
  }

  useEffect(() => {
    loadCameras()
    loadModelHealth()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (selectedCamId) {
      loadDiagnostics(selectedCamId)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedCamId])

  useEffect(() => {
    if (!autoRefresh) return
    const interval = setInterval(() => {
      if (selectedCamId) loadDiagnostics(selectedCamId)
      loadModelHealth()
    }, 2000)
    return () => clearInterval(interval)
  }, [autoRefresh, selectedCamId])

  const selectedCam = cameras.find((c) => c.id === selectedCamId)

  return (
    <div className="page-container">
      {/* Header */}
      <div className="page-header" style={{ marginBottom: 24 }}>
        <div>
          <h1 className="page-title" style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <Activity size={24} color="#3b82f6" />
            Computer Vision Diagnostics
          </h1>
          <p className="page-subtitle">
            Real-time inference telemetry, crop resolution validation, and head-cap compliance state machine inspection.
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.82rem', color: 'var(--text-secondary)', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
            />
            Live Refresh (2s)
          </label>
          <button className="btn btn-ghost btn-sm" onClick={() => loadDiagnostics(selectedCamId)}>
            <RefreshCw size={14} /> Refresh Now
          </button>
        </div>
      </div>

      {/* Camera Selector Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 24, overflowX: 'auto', paddingBottom: 6 }}>
        {cameras.map((c) => (
          <button
            key={c.id}
            onClick={() => setSelectedCamId(c.id)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              padding: '8px 16px',
              borderRadius: '8px',
              border: selectedCamId === c.id ? '1px solid #3b82f6' : '1px solid var(--border)',
              background: selectedCamId === c.id ? 'rgba(59, 130, 246, 0.15)' : 'var(--bg-card)',
              color: selectedCamId === c.id ? '#60a5fa' : 'var(--text-secondary)',
              fontWeight: 600,
              fontSize: '0.85rem',
              cursor: 'pointer',
              whiteSpace: 'nowrap',
            }}
          >
            <Camera size={14} />
            {c.name}
            <span
              style={{
                width: 7,
                height: 7,
                borderRadius: '50%',
                background: c.status === 'online' ? '#10b981' : '#64748b',
              }}
            />
          </button>
        ))}
      </div>

      {/* Main Diagnostic Content */}
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(320px, 450px) 1fr', gap: 24 }}>
        {/* Left Column: Stream Health & Video Preview */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          {/* Video Preview Card */}
          <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 12, overflow: 'hidden' }}>
            <div style={{ padding: '12px 16px', borderBottom: '1px solid var(--border)', fontWeight: 600, fontSize: '0.9rem', color: '#f3f4f6', display: 'flex', justifyContent: 'space-between' }}>
              <span>Live Feed Stream</span>
              <span style={{ fontSize: '0.75rem', color: '#34d399' }}>MJPEG Direct</span>
            </div>
            <div style={{ aspectRatio: '16/9', background: '#000', position: 'relative' }}>
              {selectedCamId ? (
                <img
                  src={`${BASE}/api/cameras/${selectedCamId}/stream`}
                  alt="Diagnostic Feed"
                  style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                />
              ) : (
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#64748b' }}>
                  No camera selected
                </div>
              )}
            </div>
          </div>

          {/* Metrics Card */}
          <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 12, padding: 18 }}>
            <h3 style={{ fontSize: '0.95rem', fontWeight: 600, color: '#f3f4f6', marginBottom: 14, display: 'flex', alignItems: 'center', gap: 8 }}>
              <Cpu size={16} color="#60a5fa" />
              Stream Metrics & Degradation Health
            </h3>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
              <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '10px 12px', borderRadius: 8 }}>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>STREAM STATUS</div>
                <div style={{ fontSize: '1rem', fontWeight: 700, color: diagnostics?.metrics?.stream_status === 'ONLINE' ? '#34d399' : '#f87171', marginTop: 2 }}>
                  {diagnostics?.metrics?.stream_status || selectedCam?.status?.toUpperCase() || 'UNKNOWN'}
                </div>
              </div>

              <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '10px 12px', borderRadius: 8 }}>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>EFFECTIVE FPS</div>
                <div style={{ fontSize: '1rem', fontWeight: 700, color: '#60a5fa', marginTop: 2 }}>
                  {diagnostics?.metrics?.fps?.toFixed(1) || '0.0'} fps
                </div>
              </div>

              <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '10px 12px', borderRadius: 8 }}>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>RECONNECT COUNT</div>
                <div style={{ fontSize: '1rem', fontWeight: 700, color: '#cbd5e1', marginTop: 2 }}>
                  {diagnostics?.metrics?.reconnect_count || 0}
                </div>
              </div>

              <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '10px 12px', borderRadius: 8 }}>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>CORRUPTED FRAMES</div>
                <div style={{ fontSize: '1rem', fontWeight: 700, color: '#fbbf24', marginTop: 2 }}>
                  {diagnostics?.metrics?.corrupted_frame_count || 0}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Head-Cap Crop Pipeline & Temporal Window */}
        <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 12, padding: 20 }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
            <h3 style={{ fontSize: '1.05rem', fontWeight: 600, color: '#f3f4f6', margin: 0, display: 'flex', alignItems: 'center', gap: 8 }}>
              <ShieldCheck size={18} color="#34d399" />
              Bakery Head-Cap Crop Pipeline & State Machine
            </h3>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              Active Tracks: <strong>{diagnostics?.headcap_diagnostics?.length || 0}</strong>
            </span>
          </div>

          <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: 16, background: 'rgba(59, 130, 246, 0.08)', padding: '10px 14px', borderRadius: 8, border: '1px solid rgba(59, 130, 246, 0.2)' }}>
            <Info size={14} style={{ display: 'inline', marginRight: 6, verticalAlign: '-2px', color: '#60a5fa' }} />
            <strong>Architecture Fix:</strong> Crops head area directly (top 30% of bbox) with minimum 32×32px validation. Predictions smoothed over a 10-frame sliding window with 3-second persistence gating.
          </div>

          {diagnostics?.headcap_diagnostics?.length === 0 ? (
            <div style={{ padding: '40px 20px', textAlign: 'center', color: 'var(--text-muted)', border: '1px dashed var(--border)', borderRadius: 8 }}>
              <p style={{ margin: 0, fontSize: '0.9rem' }}>No persons currently tracked in this camera.</p>
              <span style={{ fontSize: '0.75rem', marginTop: 4, display: 'block' }}>Walk into camera view to observe real-time head-cap crop inferences.</span>
            </div>
          ) : (
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.82rem', textAlign: 'left' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--border)', color: 'var(--text-muted)' }}>
                    <th style={{ padding: '8px 10px' }}>Track</th>
                    <th style={{ padding: '8px 10px' }}>Crop Size</th>
                    <th style={{ padding: '8px 10px' }}>Flag</th>
                    <th style={{ padding: '8px 10px' }}>Confidence</th>
                    <th style={{ padding: '8px 10px' }}>Smoothed</th>
                    <th style={{ padding: '8px 10px' }}>State</th>
                    <th style={{ padding: '8px 10px' }}>Window</th>
                  </tr>
                </thead>
                <tbody>
                  {diagnostics?.headcap_diagnostics?.map((d) => (
                    <tr key={d.track_id} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                      <td style={{ padding: '10px', fontWeight: 600, color: '#f3f4f6' }}>
                        #{d.track_id}
                      </td>
                      <td style={{ padding: '10px', color: '#cbd5e1' }}>
                        {d.crop_width}×{d.crop_height} px
                      </td>
                      <td style={{ padding: '10px' }}>
                        <span
                          style={{
                            padding: '2px 6px',
                            borderRadius: 4,
                            fontSize: '0.7rem',
                            fontWeight: 600,
                            background: d.diagnostic_flag === 'OK' ? 'rgba(16,185,129,0.15)' : 'rgba(245,158,11,0.15)',
                            color: d.diagnostic_flag === 'OK' ? '#34d399' : '#fbbf24',
                          }}
                        >
                          {d.diagnostic_flag}
                        </span>
                      </td>
                      <td style={{ padding: '10px', fontWeight: 600, color: d.confidence >= 0.45 ? '#34d399' : '#f87171' }}>
                        {(d.confidence * 100).toFixed(1)}%
                      </td>
                      <td style={{ padding: '10px' }}>
                        <span
                          style={{
                            fontWeight: 700,
                            color: d.smoothed_prediction === 'HEAD_CAP' ? '#34d399' : '#f87171',
                          }}
                        >
                          {d.smoothed_prediction}
                        </span>
                      </td>
                      <td style={{ padding: '10px' }}>
                        <span
                          style={{
                            padding: '2px 8px',
                            borderRadius: 99,
                            fontSize: '0.72rem',
                            fontWeight: 700,
                            background: d.state === 'COMPLIANT' ? 'rgba(16,185,129,0.2)' : 'rgba(239,68,68,0.2)',
                            color: d.state === 'COMPLIANT' ? '#34d399' : '#f87171',
                          }}
                        >
                          {d.state}
                        </span>
                      </td>
                      <td style={{ padding: '10px' }}>
                        <div style={{ display: 'flex', gap: 3 }}>
                          {d.window?.map((w, idx) => (
                            <span
                              key={idx}
                              style={{
                                width: 7,
                                height: 14,
                                borderRadius: 2,
                                background: w === 'HEAD_CAP' ? '#10b981' : '#ef4444',
                              }}
                              title={`Sample ${idx + 1}: ${w}`}
                            />
                          ))}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>

      {/* Registered Multi-Model Telemetry & Health */}
      <div style={{ marginTop: 24, background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 12, padding: 20 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
          <h3 style={{ fontSize: '1.05rem', fontWeight: 600, color: '#f3f4f6', margin: 0, display: 'flex', alignItems: 'center', gap: 8 }}>
            <Layers size={18} color="#60a5fa" />
            Registered Multi-Model CV Engines & Telemetry
          </h3>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            Total Models: <strong>{modelHealth.length}</strong>
          </span>
        </div>

        {modelHealth.length === 0 ? (
          <div style={{ padding: '24px', textAlign: 'center', color: 'var(--text-muted)' }}>
            No model telemetry available.
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.82rem', textAlign: 'left' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border)', color: 'var(--text-muted)' }}>
                  <th style={{ padding: '8px 10px' }}>Model Name</th>
                  <th style={{ padding: '8px 10px' }}>Type</th>
                  <th style={{ padding: '8px 10px' }}>Device</th>
                  <th style={{ padding: '8px 10px' }}>Status</th>
                  <th style={{ padding: '8px 10px' }}>Capabilities</th>
                  <th style={{ padding: '8px 10px' }}>Inferences</th>
                  <th style={{ padding: '8px 10px' }}>Avg Latency</th>
                  <th style={{ padding: '8px 10px', textAlign: 'right' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {modelHealth.map((m) => (
                  <tr key={m.model_name} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                    <td style={{ padding: '10px', fontWeight: 600, color: '#f3f4f6' }}>
                      {m.model_name}
                    </td>
                    <td style={{ padding: '10px', color: '#94a3b8' }}>
                      {m.model_type}
                    </td>
                    <td style={{ padding: '10px', color: '#cbd5e1' }}>
                      <span style={{ fontFamily: 'monospace', fontSize: '0.75rem', background: 'rgba(255,255,255,0.06)', padding: '2px 6px', borderRadius: 4 }}>
                        {m.device || 'cpu'}
                      </span>
                    </td>
                    <td style={{ padding: '10px' }}>
                      <span
                        style={{
                          padding: '2px 8px',
                          borderRadius: 99,
                          fontSize: '0.72rem',
                          fontWeight: 700,
                          background: m.status === 'loaded' ? 'rgba(16,185,129,0.15)' : 'rgba(100,116,139,0.15)',
                          color: m.status === 'loaded' ? '#34d399' : '#94a3b8',
                        }}
                      >
                        {m.status?.toUpperCase()}
                      </span>
                    </td>
                    <td style={{ padding: '10px' }}>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                        {m.capabilities?.map((cap) => (
                          <span
                            key={cap}
                            style={{
                              padding: '1px 6px',
                              borderRadius: 4,
                              fontSize: '0.7rem',
                              background: 'rgba(59,130,246,0.12)',
                              color: '#60a5fa',
                            }}
                          >
                            {cap}
                          </span>
                        ))}
                      </div>
                    </td>
                    <td style={{ padding: '10px', color: '#cbd5e1' }}>
                      {m.inference_count || 0}
                    </td>
                    <td style={{ padding: '10px', fontWeight: 600, color: '#60a5fa' }}>
                      {m.avg_latency_ms ? `${m.avg_latency_ms.toFixed(1)} ms` : '—'}
                    </td>
                    <td style={{ padding: '10px', textAlign: 'right' }}>
                      <button
                        className="btn btn-ghost btn-sm"
                        onClick={() => toggleModelLoad(m.model_name, m.status === 'loaded')}
                        style={{
                          padding: '3px 8px',
                          fontSize: '0.75rem',
                          color: m.status === 'loaded' ? '#f87171' : '#34d399',
                        }}
                      >
                        {m.status === 'loaded' ? 'Unload' : 'Load'}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
