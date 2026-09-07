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
    console.warn('[FCM] Firebase config not set — push disabled')
    return null
  }

  if (typeof window === 'undefined' || !('Notification' in window) || !('serviceWorker' in navigator)) {
    console.warn('[FCM] Notifications or ServiceWorker not supported in this browser')
    return null
  }

  if (!window.isSecureContext) {
    console.warn('[FCM] Browser is NOT in a secure context. Push notifications require HTTPS or localhost. If on mobile, use ngrok HTTPS URL.')
  }

  try {
    // Request permission
    const perm = await Notification.requestPermission()
    if (perm !== 'granted') {
      console.warn('[FCM] Notification permission not granted:', perm)
      return null
    }

    // Register the FCM service worker (root scope)
    const swReg = await navigator.serviceWorker.register('/firebase-messaging-sw.js', {
      scope: '/'
    })
    console.log('[FCM] Service worker registered with scope:', swReg.scope)

    // Wait for SW to be ready
    await navigator.serviceWorker.ready

    // Get device token
    const vapidKey = import.meta.env.VITE_FIREBASE_VAPID_KEY
    console.log('[FCM] Requesting FCM device token with VAPID key...')
    const token = await getToken(messaging, {
      vapidKey,
      serviceWorkerRegistration: swReg,
    })

    if (!token) {
      console.warn('[FCM] getToken returned null — check VAPID key and notification permission')
      return null
    }

    console.log('[FCM] Received device token:', token.slice(0, 20) + '...')

    // Register token with backend
    const tokenUrl = backendBase
      ? `${backendBase}/users/me/fcm-token`
      : '/api/users/me/fcm-token'
    console.log('[FCM] POSTing token to:', tokenUrl)
    const resp = await fetch(tokenUrl, {
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
    if (!resp.ok) {
      console.warn('[FCM] Token registration failed, status:', resp.status, await resp.text())
      return null
    }

    const regResult = await resp.json()
    console.log('[FCM] Token registered in backend ✅', regResult)
    return token
  } catch (err) {
    console.error('[FCM] initFCM error:', err)
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
