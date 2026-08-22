/**
 * ZonePainter.jsx — Canvas-based zone polygon calibration tool
 * =============================================================
 * Renders the camera's latest frame as a static image, lets the admin
 * click to place polygon vertices, names the zone, and saves it via
 * POST /cameras/{id}/zones.
 *
 * No extra libraries — uses HTML5 <canvas> over an <img> tag.
 *
 * Props
 * ─────
 *   cameraId  : int — cameras.id
 *   onSaved   : () => void — optional callback after a zone is saved
 */
import { useState, useEffect, useRef, useCallback } from 'react'
import { camerasApi } from '../api/api'
import { useToast } from '../context/ToastContext'

// Zone colour palette — consistent across painter and overlay
const ZONE_COLOURS = {
  // Ground floor
  entrance:           '#00e5ff',
  entrance_outward:   '#00b8d4',
  glass_door:         '#40c4ff',
  loading:            '#80deea',
  packing:            '#4caf50',
  oven:               '#ff5722',
  oven_stove:         '#ff5722',
  stove:              '#ff5722',
  dough_table:        '#7c4dff',
  dough_mixing:       '#9c27b0',
  cutting_machine:    '#e91e63',
  machine:            '#f06292',
  stock:              '#ffc107',
  stock_area:         '#ffa000',
  raw_material:       '#ff8f00',
  finished_goods:     '#558b2f',
  window:             '#f44336',
  window_throw:       '#d50000',
  // Shop floor
  cashbox:            '#ff9800',
  cash_counter:       '#fb8c00',
  shop_counter:       '#ef6c00',
  vendor_desk:        '#ffb300',
  payment_desk:       '#ffa000',
  // All floors
  lift:               '#26a69a',
  cylinder_area:      '#8d6e63',
  camera_standing:    '#e91e63',
  cleaning_area:      '#90a4ae',
  dispatch:           '#78909c',
  employee_area:      '#5c6bc0',
  supervisor_area:    '#3949ab',
  document_scan_area: '#1e88e5',
}
const FALLBACK_COLOUR = '#ffeb3b'

// What each zone activates — shown next to the selector for admin clarity
const ZONE_META = {
  entrance:           { icon: '📥', rule: 'OCR gate — inward invoice scan + face attendance' },
  entrance_outward:   { icon: '📤', rule: 'OCR gate — outward order form + person snapshot' },
  glass_door:         { icon: '🚪', rule: 'Face attendance (glassdoor) + lift item tracking' },
  loading:            { icon: '🚛', rule: 'Loading/unloading monitor + OCR gate + stock check' },
  packing:            { icon: '📦', rule: 'Packing monitor — hand-motion idle alert (>5 min)' },
  oven:               { icon: '🔥', rule: 'Gas/oven idle alert (boil >10 min / oil idle >10 min)' },
  oven_stove:         { icon: '🔥', rule: 'Gas/oven idle alert (boil >10 min / oil idle >10 min)' },
  stove:              { icon: '🔥', rule: 'Gas/oven idle alert — alias for oven zone' },
  dough_table:        { icon: '🫓', rule: 'Machinery zone — shift check + post-job idle (must move)' },
  dough_mixing:       { icon: '🫓', rule: 'Dough mixing zone — shift check (6 AM first floor)' },
  cutting_machine:    { icon: '✂️', rule: 'Machine-on → worker must start; after finish → go to packing' },
  machine:            { icon: '⚙️', rule: 'Generic machinery zone — idle post-job alert' },
  stock:              { icon: '📊', rule: 'Stock zone — item exposure + dirty floor + raw material' },
  stock_area:         { icon: '📊', rule: 'Stock area — item exposure alert' },
  raw_material:       { icon: '🧱', rule: 'Raw material stock monitor — exposure + movement' },
  finished_goods:     { icon: '✅', rule: 'Finished goods — dispatch check (sent to vehicle?)' },
  window:             { icon: '🪟', rule: 'Window zone — throwing/passing goods alert (immediate)' },
  window_throw:       { icon: '🪟', rule: 'Window throw — stealing/passing items out alert' },
  cashbox:            { icon: '💰', rule: 'Cash monitor — pocket vs cashbox alert' },
  cash_counter:       { icon: '💳', rule: 'Cash counter — unauthorized access + stock check' },
  shop_counter:       { icon: '🏪', rule: 'Shop absence — alert if no person > 1 min' },
  vendor_desk:        { icon: '🤝', rule: 'Vendor payment — captures payee snapshot' },
  payment_desk:       { icon: '💳', rule: 'Payment desk — captures payee snapshot' },
  lift:               { icon: '🛗', rule: 'Lift monitor — person + item tracking across all floors' },
  cylinder_area:      { icon: '🫙', rule: 'Cylinder tracking — count + usage days' },
  camera_standing:    { icon: '🚫', rule: 'Camera blocking — alert if person stands in front >1 min' },
  cleaning_area:      { icon: '🧹', rule: 'Dirty floor detection baseline zone' },
  dispatch:           { icon: '🚚', rule: 'Dispatch area — outward goods monitoring' },
  employee_area:      { icon: '👷', rule: 'Employee zone — dress code + idle check' },
  supervisor_area:    { icon: '👔', rule: 'Supervisor area — activity monitoring' },
  document_scan_area: { icon: '📄', rule: 'Document scan zone — OCR gate for paperwork' },
}

function zoneColour(name) {
  return ZONE_COLOURS[name] || FALLBACK_COLOUR
}

// Ordered preset zone list — grouped logically for the dropdown
const PRESET_ZONES = [
  // Entrances & movement
  'entrance', 'entrance_outward', 'glass_door', 'loading', 'dispatch',
  // Ground floor production
  'dough_table', 'dough_mixing', 'cutting_machine', 'machine', 'oven', 'oven_stove', 'stove', 'packing',
  // Stock & goods
  'stock', 'stock_area', 'raw_material', 'finished_goods', 'cylinder_area',
  // Windows
  'window', 'window_throw',
  // Lift
  'lift',
  // Shop floor
  'cashbox', 'cash_counter', 'shop_counter', 'vendor_desk', 'payment_desk',
  // General
  'camera_standing', 'cleaning_area', 'employee_area', 'supervisor_area', 'document_scan_area',
]

export default function ZonePainter({ cameraId, onSaved }) {
  const { addToast } = useToast()

  // ── State ───────────────────────────────────────────────────────────────
  const [frameSrc,    setFrameSrc]    = useState(null)      // base64 data URI
  const [frameLoaded, setFrameLoaded] = useState(false)
  const [savedZones,  setSavedZones]  = useState([])        // from GET /zones
  const [points,      setPoints]      = useState([])        // current WIP polygon
  const [redoStack,   setRedoStack]   = useState([])
  const [dragIndex,   setDragIndex]   = useState(null)
  const [history,     setHistory]     = useState([])
  const [zoneName,    setZoneName]    = useState('entrance')
  const [customName,  setCustomName]  = useState('')
  const [saving,      setSaving]      = useState(false)
  const [loading,     setLoading]     = useState(true)

  const canvasRef = useRef(null)
  const imgRef    = useRef(null)

  const effectiveName = zoneName === '__custom__' ? customName.trim() : zoneName

  // ── Load frame + saved zones ────────────────────────────────────────────
  const loadAll = useCallback(async () => {
    if (!cameraId) return
    setLoading(true)
    try {
      const [snapRes, zonesRes, historyRes] = await Promise.all([
        camerasApi.snapshot(cameraId),
        camerasApi.getZones(cameraId),
        camerasApi.calibrationHistory(cameraId),
      ])
      setFrameSrc(snapRes.data.frame_b64 || null)
      setSavedZones(zonesRes.data || [])
      setHistory(historyRes.data || [])
    } catch {
      addToast('Load failed', 'Could not load camera frame or zones', 'danger')
    } finally {
      setLoading(false)
    }
  }, [cameraId, addToast])

  useEffect(() => { loadAll() }, [loadAll])

  // ── Canvas drawing ──────────────────────────────────────────────────────
  const redraw = useCallback(() => {
    const canvas = canvasRef.current
    const img    = imgRef.current
    if (!canvas || !img || !frameLoaded) return

    const ctx = canvas.getContext('2d')
    canvas.width  = img.clientWidth
    canvas.height = img.clientHeight
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    const scaleX = img.clientWidth  / img.naturalWidth
    const scaleY = img.clientHeight / img.naturalHeight

    // Draw saved zones
    savedZones.forEach(z => {
      const poly = JSON.parse(z.polygon_json)
      if (!poly || poly.length < 3) return
      const colour = zoneColour(z.zone_name)

      ctx.beginPath()
      ctx.moveTo(poly[0][0] * scaleX, poly[0][1] * scaleY)
      poly.slice(1).forEach(([x, y]) => ctx.lineTo(x * scaleX, y * scaleY))
      ctx.closePath()

      ctx.fillStyle   = colour + '33'   // 20% opacity fill
      ctx.strokeStyle = colour
      ctx.lineWidth   = 2
      ctx.setLineDash([])
      ctx.fill()
      ctx.stroke()

      // Zone label
      const cx = poly.reduce((s, p) => s + p[0], 0) / poly.length
      const cy = poly.reduce((s, p) => s + p[1], 0) / poly.length
      ctx.font      = 'bold 13px Inter, sans-serif'
      ctx.fillStyle = colour
      ctx.shadowColor = 'rgba(0,0,0,0.7)'
      ctx.shadowBlur  = 4
      ctx.fillText(z.zone_name, cx * scaleX - 20, cy * scaleY + 5)
      ctx.shadowBlur = 0
    })

    // Draw WIP polygon
    if (points.length > 0) {
      const wipColour = zoneColour(effectiveName) || FALLBACK_COLOUR
      ctx.beginPath()
      ctx.moveTo(points[0][0], points[0][1])
      points.slice(1).forEach(([x, y]) => ctx.lineTo(x, y))
      if (points.length >= 3) {
        ctx.closePath()
        ctx.fillStyle = wipColour + '22'
        ctx.fill()
      }
      ctx.strokeStyle = wipColour
      ctx.lineWidth   = 2
      ctx.setLineDash([6, 3])
      ctx.stroke()

      // Points
      points.forEach(([x, y]) => {
        ctx.beginPath()
        ctx.arc(x, y, 5, 0, Math.PI * 2)
        ctx.fillStyle = wipColour
        ctx.fill()
      })
    }
  }, [savedZones, points, frameLoaded, effectiveName])

  useEffect(() => { redraw() }, [redraw])

  // ── Canvas click handler ────────────────────────────────────────────────
  const pointFromEvent = useCallback((e) => {
    const canvas = canvasRef.current
    const rect   = canvas.getBoundingClientRect()
    return [e.clientX - rect.left, e.clientY - rect.top]
  }, [])

  const handlePointerDown = useCallback((e) => {
    const [x, y] = pointFromEvent(e)
    const existing = points.findIndex(([px, py]) => Math.hypot(px - x, py - y) <= 10)
    if (existing >= 0) {
      setDragIndex(existing)
      e.currentTarget.setPointerCapture?.(e.pointerId)
      return
    }
    setPoints(prev => [...prev, [x, y]])
    setRedoStack([])
  }, [pointFromEvent, points])

  const handlePointerMove = useCallback((e) => {
    if (dragIndex === null) return
    const point = pointFromEvent(e)
    setPoints(prev => prev.map((item, index) => index === dragIndex ? point : item))
  }, [dragIndex, pointFromEvent])

  const handlePointerUp = useCallback(() => setDragIndex(null), [])

  // Right-click → undo last point
  const handleCanvasContextMenu = useCallback((e) => {
    e.preventDefault()
    setPoints(prev => {
      if (!prev.length) return prev
      setRedoStack(stack => [...stack, prev[prev.length - 1]])
      return prev.slice(0, -1)
    })
  }, [])

  const undo = () => setPoints(prev => {
    if (!prev.length) return prev
    setRedoStack(stack => [...stack, prev[prev.length - 1]])
    return prev.slice(0, -1)
  })

  const redo = () => setRedoStack(prev => {
    if (!prev.length) return prev
    const point = prev[prev.length - 1]
    setPoints(points => [...points, point])
    return prev.slice(0, -1)
  })

  // ── Save zone ───────────────────────────────────────────────────────────
  const saveZone = async () => {
    if (!effectiveName) return addToast('Name required', 'Enter a zone name first', 'warning')
    if (points.length < 3) return addToast('Need 3+ points', 'Click the canvas to place polygon vertices', 'warning')

    // Convert canvas px → natural image px
    const img    = imgRef.current
    const scaleX = img.naturalWidth  / img.clientWidth
    const scaleY = img.naturalHeight / img.clientHeight
    const naturalPts = points.map(([x, y]) => [Math.round(x * scaleX), Math.round(y * scaleY)])

    setSaving(true)
    try {
      await camerasApi.createZone(cameraId, effectiveName, JSON.stringify(naturalPts), {
        preset_type: PRESET_ZONES.includes(effectiveName) ? effectiveName : null,
        zone_type: PRESET_ZONES.includes(effectiveName) ? effectiveName : 'custom',
        display_name: effectiveName,
        color: zoneColour(effectiveName),
      })
      await camerasApi.versionCalibration(cameraId, `Updated ${effectiveName} polygon`)
      addToast('Zone saved', `"${effectiveName}" polygon saved`, 'success')
      setPoints([])
      setRedoStack([])
      await loadAll()
      onSaved?.()
    } catch (err) {
      addToast('Save failed', err.response?.data?.detail || '', 'danger')
    } finally {
      setSaving(false)
    }
  }

  // ── Delete saved zone ───────────────────────────────────────────────────
  const deleteZone = async (name) => {
    try {
      await camerasApi.deleteZone(cameraId, name)
      await camerasApi.versionCalibration(cameraId, `Deleted ${name} polygon`)
      addToast('Zone deleted', `"${name}" removed`, 'success')
      await loadAll()
    } catch {
      addToast('Delete failed', '', 'danger')
    }
  }

  const editZone = (zone) => {
    const img = imgRef.current
    if (!img) return
    try {
      const natural = JSON.parse(zone.polygon_json)
      setPoints(natural.map(([x, y]) => [x * img.clientWidth / img.naturalWidth, y * img.clientHeight / img.naturalHeight]))
      setZoneName(PRESET_ZONES.includes(zone.zone_name) ? zone.zone_name : '__custom__')
      setCustomName(PRESET_ZONES.includes(zone.zone_name) ? '' : zone.zone_name)
      setRedoStack([])
    } catch {
      addToast('Edit failed', 'Saved polygon data is invalid', 'danger')
    }
  }

  const copyZone = (zone) => {
    editZone(zone)
    setZoneName('__custom__')
    setCustomName(`${zone.zone_name}_copy`)
  }

  const restoreVersion = async (version) => {
    try {
      await camerasApi.restoreCalibration(cameraId, version, `Restored version ${version}`)
      addToast('Calibration restored', `Version ${version} is active`, 'success')
      await loadAll()
      onSaved?.()
    } catch (err) {
      addToast('Restore failed', err.response?.data?.detail || '', 'danger')
    }
  }

  // ── Render ──────────────────────────────────────────────────────────────
  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: 'var(--text-secondary)' }}>
      <span className="spinner" style={{ marginRight: 8 }} />Loading camera frame…
    </div>
  )

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

      {/* Zone name selector */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center' }}>
          <label style={{ fontWeight: 600, fontSize: 14 }}>Zone name:</label>
          <select
            value={zoneName}
            onChange={e => setZoneName(e.target.value)}
            style={{ padding: '6px 12px', borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-card)', color: 'var(--text-primary)', minWidth: 220 }}
          >
            <optgroup label="── Entrances &amp; Movement ──">
              {['entrance', 'entrance_outward', 'glass_door', 'loading', 'dispatch'].map(z => (
                <option key={z} value={z}>{ZONE_META[z]?.icon || '●'} {z}</option>
              ))}
            </optgroup>
            <optgroup label="── Ground Floor Production ──">
              {['dough_table', 'dough_mixing', 'cutting_machine', 'machine', 'oven', 'oven_stove', 'stove', 'packing'].map(z => (
                <option key={z} value={z}>{ZONE_META[z]?.icon || '●'} {z}</option>
              ))}
            </optgroup>
            <optgroup label="── Stock &amp; Goods ──">
              {['stock', 'stock_area', 'raw_material', 'finished_goods', 'cylinder_area'].map(z => (
                <option key={z} value={z}>{ZONE_META[z]?.icon || '●'} {z}</option>
              ))}
            </optgroup>
            <optgroup label="── Windows ──">
              {['window', 'window_throw'].map(z => (
                <option key={z} value={z}>{ZONE_META[z]?.icon || '●'} {z}</option>
              ))}
            </optgroup>
            <optgroup label="── Lift ──">
              <option value="lift">{ZONE_META['lift']?.icon} lift</option>
            </optgroup>
            <optgroup label="── Shop Floor ──">
              {['cashbox', 'cash_counter', 'shop_counter', 'vendor_desk', 'payment_desk'].map(z => (
                <option key={z} value={z}>{ZONE_META[z]?.icon || '●'} {z}</option>
              ))}
            </optgroup>
            <optgroup label="── General ──">
              {['camera_standing', 'cleaning_area', 'employee_area', 'supervisor_area', 'document_scan_area'].map(z => (
                <option key={z} value={z}>{ZONE_META[z]?.icon || '●'} {z}</option>
              ))}
            </optgroup>
            <option value="__custom__">✏️ Custom…</option>
          </select>
          {zoneName === '__custom__' && (
            <input
              value={customName}
              onChange={e => setCustomName(e.target.value)}
              placeholder="e.g. dough_section_2"
              style={{ padding: '6px 12px', borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-card)', color: 'var(--text-primary)', width: 200 }}
            />
          )}
        </div>

        {/* Zone description — tells admin what this zone activates */}
        {zoneName !== '__custom__' && ZONE_META[zoneName] && (
          <div style={{
            padding: '6px 12px', borderRadius: 8,
            background: zoneColour(zoneName) + '18',
            border: `1px solid ${zoneColour(zoneName)}55`,
            fontSize: '0.78rem', color: 'var(--text-secondary)',
            display: 'flex', alignItems: 'center', gap: 8,
          }}>
            <span style={{ fontSize: '1.1rem' }}>{ZONE_META[zoneName].icon}</span>
            <span><strong style={{ color: zoneColour(zoneName) }}>Activates:</strong> {ZONE_META[zoneName].rule}</span>
          </div>
        )}
      </div>

      {/* Canvas over image */}
      <div style={{ position: 'relative', borderRadius: 12, overflow: 'hidden', border: '1px solid var(--border)', background: '#111', cursor: 'crosshair', userSelect: 'none' }}>
        {frameSrc ? (
          <>
            <img
              ref={imgRef}
              src={frameSrc}
              alt="Camera frame"
              onLoad={() => setFrameLoaded(true)}
              style={{ width: '100%', display: 'block', maxHeight: 520, objectFit: 'contain' }}
            />
            <canvas
              ref={canvasRef}
              onPointerDown={handlePointerDown}
              onPointerMove={handlePointerMove}
              onPointerUp={handlePointerUp}
              onContextMenu={handleCanvasContextMenu}
              style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }}
            />
          </>
        ) : (
          <div style={{ padding: 48, textAlign: 'center', color: 'var(--text-secondary)' }}>
            <div style={{ fontSize: 40, marginBottom: 8 }}>📷</div>
            <div>Camera offline — no frame available</div>
            <div style={{ fontSize: 12, marginTop: 4 }}>Start the camera reader first, then reload</div>
          </div>
        )}
      </div>

      {/* Instructions */}
      <div style={{ fontSize: 12, color: 'var(--text-secondary)', display: 'flex', gap: 20, flexWrap: 'wrap' }}>
        <span>🖱 Left-click to place points</span>
        <span>🖱 Right-click to undo last point</span>
        <span>✅ Need ≥ 3 points to save</span>
        <span>📐 Polygon auto-closes on save</span>
      </div>

      {/* Actions */}
      <div style={{ display: 'flex', gap: 10 }}>
        <button
          className="btn btn-ghost"
          onClick={() => { setPoints([]); setRedoStack([]) }}
          disabled={points.length === 0}
        >Clear ({points.length} pts)</button>
        <button className="btn btn-ghost" onClick={undo} disabled={!points.length}>Undo</button>
        <button className="btn btn-ghost" onClick={redo} disabled={!redoStack.length}>Redo</button>
        <button
          className="btn btn-primary"
          onClick={saveZone}
          disabled={saving || points.length < 3}
        >{saving ? 'Saving…' : `Save "${effectiveName || '…'}"`}</button>
        <button className="btn btn-ghost" onClick={loadAll} style={{ marginLeft: 'auto' }}>↻ Refresh</button>
      </div>

      {/* Saved zones list */}
      {savedZones.length > 0 && (
        <div>
          <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 8, color: 'var(--text-secondary)' }}>
            Saved zones for this camera
          </div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            {savedZones.map(z => {
              const colour = zoneColour(z.zone_name)
              return (
                <div key={z.id} style={{
                  display: 'flex', alignItems: 'center', gap: 6,
                  background: colour + '22', border: `1px solid ${colour}`,
                  borderRadius: 8, padding: '4px 10px', fontSize: 13,
                }}>
                  <span style={{ width: 10, height: 10, borderRadius: '50%', background: colour, display: 'inline-block' }} />
                  <span style={{ color: colour, fontWeight: 600 }}>{z.zone_name}</span>
                  <button
                    onClick={() => editZone(z)}
                    style={{ background: 'none', border: 'none', color: colour, cursor: 'pointer', fontSize: 12, padding: 0 }}
                    title={`Edit ${z.zone_name}`}
                  >Edit</button>
                  <button
                    onClick={() => copyZone(z)}
                    style={{ background: 'none', border: 'none', color: colour, cursor: 'pointer', fontSize: 12, padding: 0 }}
                    title={`Copy ${z.zone_name}`}
                  >Copy</button>
                  <button
                    onClick={() => deleteZone(z.zone_name)}
                    style={{ background: 'none', border: 'none', color: colour, cursor: 'pointer', fontSize: 14, lineHeight: 1, padding: 0 }}
                    title={`Delete ${z.zone_name}`}
                  >✕</button>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {history.length > 0 && (
        <div>
          <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 8, color: 'var(--text-secondary)' }}>
            Calibration history
          </div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            {history.slice(0, 8).map(item => (
              <button key={item.id} className="btn btn-ghost" onClick={() => restoreVersion(item.version)}
                title={item.change_note || `Restore version ${item.version}`}>
                Restore v{item.version}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
