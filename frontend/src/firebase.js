/**
 * firebase.js — Firebase app init + FCM token helper
 *
 * Add these to frontend/.env (get from Firebase Console):
 *   VITE_FIREBASE_API_KEY=
 *   VITE_FIREBASE_AUTH_DOMAIN=
 *   VITE_FIREBASE_PROJECT_ID=
 *   VITE_FIREBASE_MESSAGING_SENDER_ID=
 *   VITE_FIREBASE_APP_ID=
 *   VITE_FIREBASE_VAPID_KEY=   ← Web Push Certificate (Firebase Console → Cloud Messaging)
 */

import { initializeApp, getApps } from 'firebase/app'
import { getMessaging, getToken, onMessage } from 'firebase/messaging'

const firebaseConfig = {
  apiKey:            import.meta.env.VITE_FIREBASE_API_KEY,
  authDomain:        import.meta.env.VITE_FIREBASE_AUTH_DOMAIN,
  projectId:         import.meta.env.VITE_FIREBASE_PROJECT_ID,
  messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID,
  appId:             import.meta.env.VITE_FIREBASE_APP_ID,
}

// Only init if config is present — avoids crash in dev before .env is filled
const isConfigured = Boolean(firebaseConfig.apiKey && firebaseConfig.projectId)

let app = null
let messaging = null

if (isConfigured) {
  app = getApps().length === 0 ? initializeApp(firebaseConfig) : getApps()[0]
  messaging = getMessaging(app)
}

/**
 * Request notification permission, register the FCM service worker,
 * obtain a device token, and POST it to the backend.
 *
 * @param {string} authToken - Bearer token for the /users/me/fcm-token endpoint
 * @param {string} backendBase - e.g. 'http://localhost:8000'
 * @returns {Promise<string|null>} FCM token or null on failure
 */
export async function initFCM(authToken, backendBase) {
  if (!isConfigured) {
    console.debug('[FCM] Firebase config not set — push disabled')
    return null
  }

  try {
    // Request permission
    const perm = await Notification.requestPermission()
    if (perm !== 'granted') {
      console.warn('[FCM] Notification permission denied')
      return null
    }

    // Register the FCM service worker
    const swReg = await navigator.serviceWorker.register('/firebase-messaging-sw.js')

    // Wait for SW to become active (may start as 'installing')
    await navigator.serviceWorker.ready

    // Send Firebase config to SW (it can't use import.meta.env or bundler)
    const sendConfig = (sw) => {
      if (sw) sw.postMessage({ type: 'FIREBASE_CONFIG', config: firebaseConfig })
    }
    sendConfig(swReg.installing || swReg.waiting || swReg.active)

    // Get device token
    const token = await getToken(messaging, {
      vapidKey: import.meta.env.VITE_FIREBASE_VAPID_KEY,
      serviceWorkerRegistration: swReg,
    })

    if (!token) {
      console.warn('[FCM] getToken returned null — check VAPID key')
      return null
    }

    // Register token with backend
    await fetch(`${backendBase}/users/me/fcm-token`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${authToken}`,
      },
      body: JSON.stringify({
        token,
        device_name: `${navigator.userAgent.slice(0, 80)}`,
      }),
    })

    console.log('[FCM] Token registered ✅')
    return token
  } catch (err) {
    console.warn('[FCM] initFCM error:', err)
    return null
  }
}

/**
 * Listen for foreground FCM messages (when app tab is open).
 * Shows a browser notification manually since Firebase suppresses auto-show in foreground.
 *
 * @param {function} onAlert - optional callback(payload) for in-app banner
 */
export function listenForegroundMessages(onAlert) {
  if (!messaging) return
  onMessage(messaging, (payload) => {
    console.log('[FCM] Foreground message:', payload)
    const { title, body } = payload.notification || {}
    // Show native notification even when app is open
    if (Notification.permission === 'granted') {
      new Notification(title || '🚨 Safety Alert', {
        body,
        icon: '/pwa-192x192.png',
        badge: '/pwa-192x192.png',
        tag: `safety-alert-${payload.data?.detected_issue || 'generic'}`,
      })
    }
    // Also call in-app handler if provided
    if (typeof onAlert === 'function') onAlert(payload)
  })
}

export { isConfigured }
