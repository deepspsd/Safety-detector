/**
 * PWAInstallPrompt — shows an install banner when the browser
 * fires the `beforeinstallprompt` event.
 * Dismissed state is persisted in localStorage for 7 days.
 */
import { useState, useEffect } from 'react'
import { Shield, Download, X } from 'lucide-react'

const DISMISSED_KEY = 'pwa-install-dismissed'
const DISMISS_TTL   = 7 * 24 * 60 * 60 * 1000   // 7 days

export default function PWAInstallPrompt() {
  const [deferredPrompt, setDeferredPrompt] = useState(null)
  const [visible, setVisible] = useState(false)

  useEffect(() => {
    // Check if already dismissed within TTL
    const dismissed = localStorage.getItem(DISMISSED_KEY)
    if (dismissed && Date.now() - Number(dismissed) < DISMISS_TTL) return

    const handler = (e) => {
      e.preventDefault()
      setDeferredPrompt(e)
      setVisible(true)
    }
    window.addEventListener('beforeinstallprompt', handler)
    return () => window.removeEventListener('beforeinstallprompt', handler)
  }, [])

  const handleInstall = async () => {
    if (!deferredPrompt) return
    deferredPrompt.prompt()
    const { outcome } = await deferredPrompt.userChoice
    if (outcome === 'accepted') console.log('✅ PWA installed')
    setDeferredPrompt(null)
    setVisible(false)
  }

  const handleDismiss = () => {
    localStorage.setItem(DISMISSED_KEY, String(Date.now()))
    setVisible(false)
  }

  if (!visible) return null

  return (
    <div className="pwa-install-banner" role="dialog" aria-label="Install app">
      <div className="pwa-banner-icon">
        <Shield size={22} color="#fff" />
      </div>
      <div className="pwa-banner-text">
        <div className="pwa-banner-title">Install Safety AI</div>
        <div className="pwa-banner-sub">Add to home screen for quick access</div>
      </div>
      <button
        onClick={handleInstall}
        className="btn btn-primary btn-sm"
        style={{
          background: 'linear-gradient(135deg,#f97316,#ea580c)',
          flexShrink: 0, gap: 6,
        }}
      >
        <Download size={13} /> Install
      </button>
      <button
        onClick={handleDismiss}
        style={{ background:'none', border:'none', color:'var(--text-muted)',
          cursor:'pointer', padding: 4, flexShrink: 0 }}
        aria-label="Dismiss"
      >
        <X size={16} />
      </button>
    </div>
  )
}
