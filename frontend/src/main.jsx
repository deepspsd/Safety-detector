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
    // Disable SW in development so it doesn't cache our JS bundles
    // and break Vite's HMR or dev API routing.
    navigator.serviceWorker.getRegistrations().then(regs => {
      for (const reg of regs) {
        reg.unregister()
        console.log('🗑️ Unregistered DEV Service Worker to prevent caching bugs')
      }
    })
  }
}

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
