import { useState, useRef, useEffect } from 'react'
import {
  Camera, Upload, CheckCircle2, AlertTriangle, RefreshCw, FileText,
  Truck, Package, UserCheck, ShieldCheck, ArrowDownLeft, ArrowUpRight,
  Scale, Hash, Building, ChevronRight, X, XCircle
} from 'lucide-react'

export default function MobileUpload() {
  const query = new URLSearchParams(window.location.search)
  const initialToken = query.get('gate_key') || query.get('token') || 'occusafe-gate-fixed'
  const initialDir   = query.get('direction') || 'inward'

  const [direction, setDirection] = useState(initialDir === 'outward' ? 'outward' : 'inward')
  const [token] = useState(initialToken)

  // Form Fields
  const [vendorName, setVendorName]   = useState('')
  const [vehicleNo, setVehicleNo]     = useState('')
  const [goodsCount, setGoodsCount]   = useState('')
  const [weight, setWeight]           = useState('')
  const [docNumber, setDocNumber]     = useState('')
  const [notes, setNotes]             = useState('')

  // Images
  const [docFile, setDocFile]           = useState(null)
  const [docPreview, setDocPreview]     = useState(null)
  const [personFile, setPersonFile]     = useState(null)
  const [personPreview, setPersonPreview] = useState(null)

  // Submission state
  const [submitting, setSubmitting] = useState(false)
  const [result, setResult]         = useState(null)
  const [errorMsg, setErrorMsg]     = useState('')

  const docCameraInputRef = useRef(null)
  const docFileInputRef = useRef(null)
  const personInputRef = useRef(null)

  // Document photo / file handler with strict format and size constraints
  const handleDocChange = (e) => {
    const file = e.target.files?.[0]
    if (!file) return

    // Format constraints: PDF, JPEG, PNG, WEBP, DOC, DOCX
    const allowedExtensions = ['.pdf', '.jpg', '.jpeg', '.png', '.webp', '.doc', '.docx']
    const ext = '.' + file.name.split('.').pop().toLowerCase()
    if (!allowedExtensions.includes(ext)) {
      setErrorMsg(`Unsupported file type (${ext}). Allowed: PDF, JPG, PNG, WEBP, and Word (.doc, .docx).`)
      return
    }

    // Size constraint: Max 25MB
    if (file.size > 25 * 1024 * 1024) {
      setErrorMsg('File size exceeds 25MB limit. Please upload a smaller file.')
      return
    }

    setDocFile(file)
    setErrorMsg('')

    if (file.type.startsWith('image/')) {
      const reader = new FileReader()
      reader.onload = ev => setDocPreview(ev.target.result)
      reader.readAsDataURL(file)
    } else {
      // PDF or Word Document
      setDocPreview(null)
    }
  }

  // Person/Driver photo handler
  const handlePersonChange = (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    setPersonFile(file)
    const reader = new FileReader()
    reader.onload = ev => setPersonPreview(ev.target.result)
    reader.readAsDataURL(file)
  }

  // Network-independent upload submit
  const handleSubmit = async (e) => {
    e?.preventDefault()
    if (!docFile) {
      setErrorMsg('Please photograph or attach the document.')
      return
    }
    if (!vendorName.trim()) {
      setErrorMsg('Vendor / Supplier Name is compulsory.')
      return
    }
    if (!vehicleNo.trim()) {
      setErrorMsg('Vehicle Number is compulsory.')
      return
    }
    if (!goodsCount || Number(goodsCount) <= 0) {
      setErrorMsg('Goods Count / Quantity is compulsory (must be greater than 0).')
      return
    }
    if (!weight.trim()) {
      setErrorMsg('Total Weight (e.g. 50 kg) is compulsory.')
      return
    }
    if (!docNumber.trim()) {
      setErrorMsg('Invoice / Document Reference Number is compulsory.')
      return
    }

    setSubmitting(true)
    setErrorMsg('')

    try {
      const formData = new FormData()
      formData.append('token', token)
      formData.append('direction', direction)
      formData.append('file', docFile)
      if (personFile) {
        formData.append('person_file', personFile)
      }
      formData.append('goods_count', goodsCount.trim())
      formData.append('weight', weight.trim())
      formData.append('vendor_name', vendorName.trim())
      formData.append('vehicle_no', vehicleNo.trim().toUpperCase())
      formData.append('doc_number', docNumber.trim())
      if (notes)       formData.append('notes', notes)

      // Network resilient URL resolution:
      // 1. If VITE_API_BASE_URL set, use it.
      // 2. Try same-origin /api/documents/mobile-upload (works via Vite proxy, Nginx reverse proxy, cloud tunnel).
      // 3. Fallback to direct port 8000 on current hostname.
      const endpoints = []
      if (import.meta.env.VITE_API_BASE_URL) {
        endpoints.push(import.meta.env.VITE_API_BASE_URL.replace(/\/$/, '') + '/documents/mobile-upload')
      }
      endpoints.push('/api/documents/mobile-upload')
      endpoints.push(window.location.protocol + '//' + window.location.hostname + ':8000/documents/mobile-upload')
      endpoints.push('/documents/mobile-upload')

      let response = null
      let lastErr = null

      for (const ep of endpoints) {
        try {
          const res = await fetch(ep, {
            method: 'POST',
            body: formData,
          })
          if (res.ok || res.status === 400 || res.status === 401 || res.status === 410) {
            response = res
            break
          }
        } catch (netErr) {
          lastErr = netErr
        }
      }

      if (!response) {
        throw new Error(lastErr?.message || 'Cannot reach gate server. Check connection.')
      }

      if (response.status === 401) {
        setErrorMsg('Gate pass security key invalid. Scan the latest QR poster.')
        return
      }
      if (response.status === 410) {
        setErrorMsg('QR session expired. Please rescan the permanent gate QR code.')
        return
      }
      if (!response.ok) {
        const errJson = await response.json().catch(() => ({}))
        setErrorMsg(errJson.detail || 'Upload failed with server status ' + response.status)
        return
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      console.error('[MobileUpload error]', err)
      setErrorMsg('Network error: Could not reach factory gate. Please verify data connection and retry.')
    } finally {
      setSubmitting(false)
    }
  }

  const resetForm = () => {
    setDocFile(null)
    setDocPreview(null)
    setPersonFile(null)
    setPersonPreview(null)
    setGoodsCount('')
    setWeight('')
    setVendorName('')
    setVehicleNo('')
    setDocNumber('')
    setNotes('')
    setResult(null)
    setErrorMsg('')
  }

  const isInward = direction === 'inward'

  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(180deg, #090d16 0%, #0f172a 100%)',
      color: '#f8fafc',
      fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
      padding: '20px 16px 40px',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
    }}>
      <style>{`
        @keyframes mu-pop { 0%{transform:scale(0.85);opacity:0} 100%{transform:scale(1);opacity:1} }
        @keyframes mu-pulse { 0%,100%{opacity:1} 50%{opacity:0.6} }
        input, select, textarea {
          transition: border-color 0.2s, box-shadow 0.2s;
        }
        input:focus, select:focus, textarea:focus {
          border-color: #3b82f6 !important;
          outline: none;
          box-shadow: 0 0 0 3px rgba(59,130,246,0.2);
        }
      `}</style>

      {/* Header */}
      <div style={{ width: '100%', maxWidth: 440, marginBottom: 18, textAlign: 'center' }}>
        <div style={{
          display: 'inline-flex', alignItems: 'center', gap: 6,
          padding: '4px 12px', borderRadius: 99, background: 'rgba(59,130,246,0.12)',
          border: '1px solid rgba(59,130,246,0.3)', marginBottom: 8,
        }}>
          <ShieldCheck size={14} color="#60a5fa" />
          <span style={{ fontSize: '0.72rem', fontWeight: 700, color: '#93c5fd', letterSpacing: '0.06em', textTransform: 'uppercase' }}>
            OccuSafe Gate Clearance
          </span>
        </div>
        <h1 style={{ margin: 0, fontSize: '1.45rem', fontWeight: 800, color: '#ffffff', letterSpacing: '-0.02em' }}>
          Document Gate Check-In
        </h1>
        <p style={{ margin: '4px 0 0', fontSize: '0.8rem', color: '#94a3b8' }}>
          Upload paperwork for instant gate verification & entry pass
        </p>
      </div>

      <div style={{
        width: '100%', maxWidth: 440,
        background: '#131d31',
        border: '1px solid #23334d',
        borderRadius: 20,
        padding: '22px 18px',
        boxShadow: '0 20px 50px rgba(0,0,0,0.45)',
      }}>
        {!result ? (
          <form onSubmit={handleSubmit}>
            {/* Movement Direction Toggle */}
            <div style={{ marginBottom: 18 }}>
              <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 700, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 8 }}>
                Select Movement Type *
              </label>
              <div style={{
                display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10,
                background: '#090d16', padding: 5, borderRadius: 14, border: '1px solid #1e293b'
              }}>
                <button
                  type="button"
                  onClick={() => setDirection('inward')}
                  style={{
                    display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
                    padding: '12px 10px', borderRadius: 10, border: 'none', cursor: 'pointer',
                    fontWeight: 700, fontSize: '0.88rem',
                    background: isInward ? 'linear-gradient(135deg, #10b981, #059669)' : 'transparent',
                    color: isInward ? '#ffffff' : '#94a3b8',
                    boxShadow: isInward ? '0 4px 12px rgba(16,185,129,0.3)' : 'none',
                    transition: 'all 0.2s ease',
                  }}
                >
                  <ArrowDownLeft size={16} strokeWidth={2.5} />
                  <span>INWARD (Entry)</span>
                </button>
                <button
                  type="button"
                  onClick={() => setDirection('outward')}
                  style={{
                    display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
                    padding: '12px 10px', borderRadius: 10, border: 'none', cursor: 'pointer',
                    fontWeight: 700, fontSize: '0.88rem',
                    background: !isInward ? 'linear-gradient(135deg, #f59e0b, #d97706)' : 'transparent',
                    color: !isInward ? '#ffffff' : '#94a3b8',
                    boxShadow: !isInward ? '0 4px 12px rgba(245,158,11,0.3)' : 'none',
                    transition: 'all 0.2s ease',
                  }}
                >
                  <ArrowUpRight size={16} strokeWidth={2.5} />
                  <span>OUTWARD (Exit)</span>
                </button>
              </div>
              <div style={{ fontSize: '0.72rem', color: '#64748b', marginTop: 6, textAlign: 'center' }}>
                {isInward
                  ? 'Raw Materials, Ingredients, Vendor Invoices, Packaging'
                  : 'Finished Bakery Goods, Dispatched Orders, Delivery Challan'}
              </div>
            </div>

            {/* Quick Vendor & Delivery Details */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, marginBottom: 14 }}>
              <div>
                <label style={{ display: 'block', fontSize: '0.72rem', fontWeight: 600, color: '#e2e8f0', marginBottom: 5 }}>
                  <Building size={11} style={{ display: 'inline', marginRight: 4 }} />
                  {isInward ? 'Supplier / Vendor' : 'Client / Recipient'} <span style={{ color: '#ef4444' }}>*</span>
                </label>
                <input
                  type="text"
                  required
                  placeholder="e.g. Flour Mills Ltd"
                  value={vendorName}
                  onChange={e => setVendorName(e.target.value)}
                  style={{
                    width: '100%', padding: '9px 11px', borderRadius: 9,
                    background: '#0a101d', border: '1px solid #23334d', color: '#f8fafc', fontSize: '0.82rem',
                    boxSizing: 'border-box',
                  }}
                />
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '0.72rem', fontWeight: 600, color: '#e2e8f0', marginBottom: 5 }}>
                  <Truck size={11} style={{ display: 'inline', marginRight: 4 }} />
                  Vehicle Number <span style={{ color: '#ef4444' }}>*</span>
                </label>
                <input
                  type="text"
                  required
                  placeholder="e.g. MH-12-AB-1234"
                  value={vehicleNo}
                  onChange={e => setVehicleNo(e.target.value.toUpperCase())}
                  style={{
                    width: '100%', padding: '9px 11px', borderRadius: 9,
                    background: '#0a101d', border: '1px solid #23334d', color: '#f8fafc', fontSize: '0.82rem',
                    boxSizing: 'border-box', textTransform: 'uppercase',
                  }}
                />
              </div>
            </div>

            {/* Goods Count & Weight */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, marginBottom: 14 }}>
              <div>
                <label style={{ display: 'block', fontSize: '0.72rem', fontWeight: 600, color: '#e2e8f0', marginBottom: 5 }}>
                  <Package size={11} style={{ display: 'inline', marginRight: 4 }} />
                  Goods Count / Qty <span style={{ color: '#ef4444' }}>*</span>
                </label>
                <input
                  type="number"
                  required
                  min="1"
                  placeholder="e.g. 50 (boxes/bags)"
                  value={goodsCount}
                  onChange={e => setGoodsCount(e.target.value)}
                  style={{
                    width: '100%', padding: '9px 11px', borderRadius: 9,
                    background: '#0a101d', border: '1px solid #23334d', color: '#f8fafc', fontSize: '0.82rem',
                    boxSizing: 'border-box',
                  }}
                />
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '0.72rem', fontWeight: 600, color: '#e2e8f0', marginBottom: 5 }}>
                  <Scale size={11} style={{ display: 'inline', marginRight: 4 }} />
                  Total Weight <span style={{ color: '#ef4444' }}>*</span>
                </label>
                <input
                  type="text"
                  required
                  placeholder="e.g. 250 kg"
                  value={weight}
                  onChange={e => setWeight(e.target.value)}
                  style={{
                    width: '100%', padding: '9px 11px', borderRadius: 9,
                    background: '#0a101d', border: '1px solid #23334d', color: '#f8fafc', fontSize: '0.82rem',
                    boxSizing: 'border-box',
                  }}
                />
              </div>
            </div>

            {/* Invoice / Challan Number */}
            <div style={{ marginBottom: 16 }}>
              <label style={{ display: 'block', fontSize: '0.72rem', fontWeight: 600, color: '#e2e8f0', marginBottom: 5 }}>
                <Hash size={11} style={{ display: 'inline', marginRight: 4 }} />
                {isInward ? 'Invoice Number' : 'Order Form / Challan #'} <span style={{ color: '#ef4444' }}>*</span>
              </label>
              <input
                type="text"
                required
                placeholder="e.g. INV-2026-0892 or Challan Ref"
                value={docNumber}
                onChange={e => setDocNumber(e.target.value)}
                style={{
                  width: '100%', padding: '9px 11px', borderRadius: 9,
                  background: '#0a101d', border: '1px solid #23334d', color: '#f8fafc', fontSize: '0.82rem',
                  boxSizing: 'border-box',
                }}
              />
            </div>

            {/* Document Photo / File Upload Card */}
            <div style={{ marginBottom: 16 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                <label style={{ fontSize: '0.78rem', fontWeight: 700, color: '#e2e8f0' }}>
                  📄 Document (Invoice / Challan) <span style={{ color: '#ef4444' }}>*</span>
                </label>
                <span style={{ fontSize: '0.7rem', color: '#60a5fa', fontWeight: 600 }}>PDF, Word or Camera Photo</span>
              </div>

              {docFile ? (
                <div style={{
                  position: 'relative', borderRadius: 12, overflow: 'hidden',
                  border: '2px solid #3b82f6', background: '#0a101d', padding: docPreview ? 0 : 16
                }}>
                  {docPreview ? (
                    <img src={docPreview} alt="Document Preview" style={{ width: '100%', maxHeight: 240, objectFit: 'contain', display: 'block' }} />
                  ) : (
                    <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
                      <div style={{
                        width: 48, height: 48, borderRadius: 10,
                        background: docFile.name.toLowerCase().endsWith('.pdf') ? 'rgba(239, 68, 68, 0.15)' : 'rgba(59, 130, 246, 0.15)',
                        border: docFile.name.toLowerCase().endsWith('.pdf') ? '1px solid rgba(239, 68, 68, 0.3)' : '1px solid rgba(59, 130, 246, 0.3)',
                        display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0
                      }}>
                        <FileText size={26} color={docFile.name.toLowerCase().endsWith('.pdf') ? '#f87171' : '#60a5fa'} />
                      </div>
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ fontSize: '0.84rem', fontWeight: 700, color: '#f8fafc', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                          {docFile.name}
                        </div>
                        <div style={{ fontSize: '0.72rem', color: '#94a3b8', marginTop: 2 }}>
                          {docFile.name.toLowerCase().endsWith('.pdf') ? 'PDF Document' : 'Word Document'} • {(docFile.size / 1024).toFixed(1)} KB
                        </div>
                        <div style={{ fontSize: '0.68rem', color: '#10b981', fontWeight: 600, marginTop: 4 }}>
                          ✓ Ready for OCR Verification
                        </div>
                      </div>
                    </div>
                  )}
                  <button
                    type="button"
                    onClick={() => { setDocFile(null); setDocPreview(null) }}
                    style={{
                      position: 'absolute', top: 8, right: 8,
                      background: 'rgba(239,68,68,0.9)', color: '#fff', border: 'none',
                      borderRadius: 99, padding: '4px 10px', fontSize: '0.72rem', fontWeight: 700, cursor: 'pointer',
                      display: 'flex', alignItems: 'center', gap: 4
                    }}
                  >
                    <X size={12} /> Change
                  </button>
                </div>
              ) : (
                <div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
                    {/* Option 1: Take Photo */}
                    <div
                      onClick={() => docCameraInputRef.current?.click()}
                      style={{
                        border: '2px dashed #2e4465', borderRadius: 14, padding: '18px 10px',
                        textAlign: 'center', cursor: 'pointer', background: 'rgba(15,23,42,0.6)',
                        transition: 'all 0.2s', display: 'flex', flexDirection: 'column', alignItems: 'center'
                      }}
                    >
                      <div style={{
                        width: 42, height: 42, borderRadius: '50%', background: 'rgba(59,130,246,0.12)',
                        display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: 8
                      }}>
                        <Camera size={22} color="#60a5fa" />
                      </div>
                      <div style={{ fontSize: '0.84rem', fontWeight: 700, color: '#f8fafc', marginBottom: 2 }}>
                        Take Photo
                      </div>
                      <div style={{ fontSize: '0.68rem', color: '#94a3b8' }}>
                        Scan via camera
                      </div>
                    </div>

                    {/* Option 2: Upload File / Document */}
                    <div
                      onClick={() => docFileInputRef.current?.click()}
                      style={{
                        border: '2px dashed #2e4465', borderRadius: 14, padding: '18px 10px',
                        textAlign: 'center', cursor: 'pointer', background: 'rgba(15,23,42,0.6)',
                        transition: 'all 0.2s', display: 'flex', flexDirection: 'column', alignItems: 'center'
                      }}
                    >
                      <div style={{
                        width: 42, height: 42, borderRadius: '50%', background: 'rgba(16,185,129,0.12)',
                        display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: 8
                      }}>
                        <Upload size={22} color="#34d399" />
                      </div>
                      <div style={{ fontSize: '0.84rem', fontWeight: 700, color: '#f8fafc', marginBottom: 2 }}>
                        Upload Doc
                      </div>
                      <div style={{ fontSize: '0.68rem', color: '#94a3b8' }}>
                        PDF, Word, or File
                      </div>
                    </div>
                  </div>
                  <div style={{ fontSize: '0.69rem', color: '#64748b', textAlign: 'center', marginTop: 7 }}>
                    Allowed: PDF, JPG, PNG, WEBP, DOCX, DOC (Max 25MB)
                  </div>
                </div>
              )}

              {/* Hidden camera capture input */}
              <input
                ref={docCameraInputRef}
                type="file"
                accept="image/*"
                capture="environment"
                style={{ display: 'none' }}
                onChange={handleDocChange}
              />
              {/* Hidden file selector input */}
              <input
                ref={docFileInputRef}
                type="file"
                accept=".pdf,.jpg,.jpeg,.png,.webp,.doc,.docx,application/pdf,application/msword,application/vnd.openxmlformats-officedocument.wordprocessingml.document,image/*"
                style={{ display: 'none' }}
                onChange={handleDocChange}
              />
            </div>

            {/* Person / Driver Photo Card (Client req.md requirement) */}
            <div style={{ marginBottom: 18 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                <label style={{ fontSize: '0.78rem', fontWeight: 700, color: '#e2e8f0' }}>
                  👤 Person / Driver Photo
                </label>
                <span style={{ fontSize: '0.7rem', color: '#94a3b8' }}>
                  {!isInward ? 'Mandatory for Outward' : 'Recommended'}
                </span>
              </div>

              {personPreview ? (
                <div style={{ position: 'relative', borderRadius: 12, overflow: 'hidden', border: '2px solid #10b981', background: '#0a101d' }}>
                  <img src={personPreview} alt="Person Preview" style={{ width: '100%', maxHeight: 180, objectFit: 'contain', display: 'block' }} />
                  <button
                    type="button"
                    onClick={() => { setPersonFile(null); setPersonPreview(null) }}
                    style={{
                      position: 'absolute', top: 8, right: 8,
                      background: 'rgba(239,68,68,0.9)', color: '#fff', border: 'none',
                      borderRadius: 99, padding: '4px 10px', fontSize: '0.72rem', fontWeight: 700, cursor: 'pointer',
                      display: 'flex', alignItems: 'center', gap: 4
                    }}
                  >
                    <X size={12} /> Retake
                  </button>
                </div>
              ) : (
                <div
                  onClick={() => personInputRef.current?.click()}
                  style={{
                    border: '1px dashed #23334d', borderRadius: 12, padding: '14px 12px',
                    textAlign: 'center', cursor: 'pointer', background: 'rgba(10,16,29,0.5)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 10
                  }}
                >
                  <UserCheck size={20} color="#94a3b8" />
                  <span style={{ fontSize: '0.78rem', color: '#94a3b8', fontWeight: 600 }}>
                    Snap Quick Driver/Payee Selfie
                  </span>
                </div>
              )}
              <input
                ref={personInputRef}
                type="file"
                accept="image/*"
                capture="user"
                style={{ display: 'none' }}
                onChange={handlePersonChange}
              />
            </div>

            {/* Error Message */}
            {errorMsg && (
              <div style={{
                background: 'rgba(239,68,68,0.12)', border: '1px solid rgba(239,68,68,0.35)',
                borderRadius: 10, padding: '10px 14px', marginBottom: 14,
                fontSize: '0.8rem', color: '#ef4444', lineHeight: 1.5,
              }}>
                <AlertTriangle size={14} style={{ display: 'inline', marginRight: 6, verticalAlign: -2 }} />
                {errorMsg}
              </div>
            )}

            {/* Submit Button */}
            <button
              type="submit"
              disabled={submitting || !docFile}
              style={{
                width: '100%', padding: '14px 0', borderRadius: 12, border: 'none',
                cursor: (docFile && !submitting) ? 'pointer' : 'not-allowed',
                background: (docFile && !submitting)
                  ? (isInward ? 'linear-gradient(135deg, #10b981, #059669)' : 'linear-gradient(135deg, #f59e0b, #d97706)')
                  : '#1e293b',
                color: (docFile && !submitting) ? '#ffffff' : '#64748b',
                fontWeight: 700, fontSize: '1rem',
                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
                boxShadow: (docFile && !submitting) ? '0 10px 25px rgba(0,0,0,0.3)' : 'none',
                transition: 'all 0.2s ease',
              }}
            >
              {submitting ? (
                <>
                  <span style={{ animation: 'mu-pulse 1s infinite' }}>Processing OCR...</span>
                </>
              ) : (
                <>
                  <Upload size={18} />
                  <span>Submit Document to Gate</span>
                </>
              )}
            </button>
          </form>
        ) : result.auto_rejected ? (
          /* Auto-Rejected Non-Invoice / Object Screen */
          <div style={{ animation: 'mu-pop 0.3s ease-out' }}>
            <div style={{ textAlign: 'center', marginBottom: 18 }}>
              <div style={{
                width: 72, height: 72, borderRadius: '50%',
                background: 'rgba(239,68,68,0.15)',
                border: '2px solid #ef4444',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                margin: '0 auto 12px',
              }}>
                <XCircle size={42} color="#ef4444" strokeWidth={2.5} />
              </div>

              <div style={{ fontSize: '1.25rem', fontWeight: 900, color: '#ef4444', marginBottom: 6 }}>
                GATE ENTRY AUTO-REJECTED
              </div>
              <div style={{ fontSize: '0.84rem', color: '#fca5a5', lineHeight: 1.5, padding: '0 10px' }}>
                The uploaded photo was automatically rejected because it is not recognized as a valid invoice or delivery document.
              </div>
            </div>

            {/* Rejection Details Box */}
            <div style={{
              background: 'rgba(239,68,68,0.08)', borderRadius: 14, border: '1px solid rgba(239,68,68,0.3)',
              padding: '14px 16px', marginBottom: 16,
            }}>
              <div style={{ fontSize: '0.72rem', color: '#f87171', textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 800, marginBottom: 6 }}>
                Security Rejection Reason
              </div>
              <div style={{ fontSize: '0.82rem', color: '#fecaca', lineHeight: 1.4 }}>
                {result.reject_reason || 'Non-document photo detected. No commercial billing keywords or invoice structures found.'}
              </div>
            </div>

            {/* Detected Text preview if any */}
            {result.raw_text && (
              <div style={{
                background: '#0a101d', borderRadius: 12, border: '1px solid #1e293b',
                padding: '10px 14px', marginBottom: 18,
              }}>
                <div style={{ fontSize: '0.68rem', color: '#64748b', fontWeight: 700, textTransform: 'uppercase', marginBottom: 4 }}>
                  Detected OCR Text:
                </div>
                <div style={{
                  fontFamily: 'monospace', fontSize: '0.74rem', color: '#94a3b8',
                  maxHeight: 70, overflowY: 'auto', whiteSpace: 'pre-wrap', lineHeight: 1.4,
                }}>
                  {result.raw_text || '(No text detected in photo)'}
                </div>
              </div>
            )}

            <button
              onClick={resetForm}
              style={{
                width: '100%', padding: '14px 0', borderRadius: 12,
                border: 'none', background: 'linear-gradient(135deg, #ef4444, #dc2626)',
                color: '#ffffff', fontWeight: 800, fontSize: '0.94rem',
                cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
                boxShadow: '0 10px 25px rgba(239,68,68,0.3)',
              }}
            >
              <Camera size={18} />
              <span>Retake Photo of Valid Invoice</span>
            </button>
          </div>
        ) : (
          /* Result & Digital Gate Pass Receipt */
          <div style={{ animation: 'mu-pop 0.3s ease-out' }}>
            <div style={{ textAlign: 'center', marginBottom: 18 }}>
              <div style={{
                width: 68, height: 68, borderRadius: '50%',
                background: result.approved ? 'rgba(16,185,129,0.15)' : 'rgba(245,158,11,0.15)',
                border: '2px solid ' + (result.approved ? '#10b981' : '#f59e0b'),
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                margin: '0 auto 12px',
              }}>
                {result.approved
                  ? <CheckCircle2 size={38} color="#10b981" strokeWidth={2.5} />
                  : <AlertTriangle size={34} color="#f59e0b" strokeWidth={2.5} />}
              </div>

              <div style={{
                fontSize: '1.25rem', fontWeight: 800,
                color: result.approved ? '#10b981' : '#f59e0b',
                marginBottom: 4,
              }}>
                {result.approved ? 'Entry Cleared & Logged!' : 'Document Submitted — Guard Review'}
              </div>
              <div style={{ fontSize: '0.8rem', color: '#94a3b8' }}>
                Present this screen to Gate Security for verification.
              </div>
            </div>

            {/* Digital Gate Pass Box */}
            <div style={{
              background: '#090d16', borderRadius: 14, border: '1px solid #1e2d42',
              padding: '14px 16px', marginBottom: 16,
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10, borderBottom: '1px solid #172338', paddingBottom: 8 }}>
                <span style={{ fontSize: '0.72rem', color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 700 }}>
                  Gate Pass Ref
                </span>
                <span style={{
                  fontSize: '0.92rem', fontFamily: 'monospace', fontWeight: 800, color: '#60a5fa',
                  background: 'rgba(59,130,246,0.12)', padding: '2px 8px', borderRadius: 6,
                }}>
                  {result.gate_pass_code || `GP-${result.record?.id || '001'}`}
                </span>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, fontSize: '0.78rem' }}>
                <div>
                  <span style={{ color: '#64748b', display: 'block', fontSize: '0.68rem' }}>Type</span>
                  <strong style={{ color: result.direction === 'inward' ? '#10b981' : '#f59e0b', textTransform: 'uppercase' }}>
                    {result.direction === 'inward' ? 'Inward (Entry)' : 'Outward (Exit)'}
                  </strong>
                </div>

                <div>
                  <span style={{ color: '#64748b', display: 'block', fontSize: '0.68rem' }}>Vehicle No</span>
                  <strong>{result.vehicle_no || 'Not specified'}</strong>
                </div>

                <div>
                  <span style={{ color: '#64748b', display: 'block', fontSize: '0.68rem' }}>Vendor / Party</span>
                  <strong>{result.vendor_name || 'Direct'}</strong>
                </div>

                <div>
                  <span style={{ color: '#64748b', display: 'block', fontSize: '0.68rem' }}>Goods & Weight</span>
                  <strong>
                    {result.goods_count ? `${result.goods_count} pcs ` : ''}
                    {result.weight ? `(${result.weight})` : (result.goods_count ? '' : 'Logged')}
                  </strong>
                </div>
              </div>
            </div>

            {/* OCR Extracted Text Preview */}
            <div style={{
              background: '#0a101d', borderRadius: 12, border: '1px solid #1e293b',
              padding: '10px 14px', marginBottom: 18,
            }}>
              <div style={{ fontSize: '0.68rem', color: '#64748b', fontWeight: 700, textTransform: 'uppercase', marginBottom: 4 }}>
                📄 OCR Verification Text
              </div>
              <div style={{
                fontFamily: 'monospace', fontSize: '0.75rem', color: '#cbd5e1',
                maxHeight: 90, overflowY: 'auto', whiteSpace: 'pre-wrap', lineHeight: 1.5,
              }}>
                {result.raw_text && !result.raw_text.startsWith('[')
                  ? result.raw_text
                  : 'Document image captured successfully. Stored on-premise for security audit.'}
              </div>
            </div>

            <button
              onClick={resetForm}
              style={{
                width: '100%', padding: '12px 0', borderRadius: 10,
                border: '1px solid #334155', background: '#1e293b',
                color: '#f8fafc', fontWeight: 700, fontSize: '0.88rem',
                cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
              }}
            >
              <RefreshCw size={14} />
              <span>Submit Another Document</span>
            </button>
          </div>
        )}
      </div>

      <div style={{ marginTop: 18, fontSize: '0.7rem', color: '#475569', textAlign: 'center' }}>
        OccuSafe On-Premise Safety & Gate Logistics • Secure Local Storage
      </div>
    </div>
  )
}
