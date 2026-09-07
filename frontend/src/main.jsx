import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import './mobile.css'
import App from './App.jsx'

// ── Manual PWA Service Worker registration ────────────
// (No vite-plugin-pwa needed — works with Vite 8)
if ('serviceWorker' in navigator) {
  if (import.meta.env.MODE === 'production') {
    window.addEventListener('load', () => {
      navigator.serviceWorker
        .register('/sw.js', { scope: '/' })
        .then(reg => {
          console.log('✅ SafeGuard AI SW registered:', reg.scope)

          // Check for updates every 60 seconds
          setInterval(() => reg.update(), 60_000)

          // Notify user when a new version is waiting
          reg.addEventListener('updatefound', () => {
            const newWorker = reg.installing
            newWorker?.addEventListener('statechange', () => {
              if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
                if (confirm('🔄 New version of Safety AI available! Reload to update?')) {
                  newWorker.postMessage({ type: 'SKIP_WAITING' })
                  window.location.reload()
                }
              }
            })
          })
        })
        .catch(err => console.warn('SW registration failed:', err))
    })
  } else {
    // In development, unregister sw.js (cache worker) so it doesn't cache JS bundles,
    // but KEEP firebase-messaging-sw.js active so push notifications work!
    navigator.serviceWorker.getRegistrations().then(regs => {
      for (const reg of regs) {
        const scriptURL = reg.active?.scriptURL || reg.installing?.scriptURL || reg.waiting?.scriptURL || ''
        if (!scriptURL.includes('firebase-messaging-sw')) {
          reg.unregister()
          console.log('🗑️ Unregistered DEV Cache Service Worker:', scriptURL)
        }
      }
    })
    // Clear old caches that may be holding stale 503 "Offline" pages
    if ('caches' in window) {
      caches.keys().then(keys => {
        for (const k of keys) {
          if (k.startsWith('safeguard-ai')) caches.delete(k)
        }
      })
    }
  }
}

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
