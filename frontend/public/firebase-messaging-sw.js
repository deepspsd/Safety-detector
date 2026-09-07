// firebase-messaging-sw.js
// Handles FCM push notifications in the background (when app tab is closed/hidden).
// Must live at root scope (/firebase-messaging-sw.js) — placed in public/.

importScripts('https://www.gstatic.com/firebasejs/10.12.0/firebase-app-compat.js');
importScripts('https://www.gstatic.com/firebasejs/10.12.0/firebase-messaging-compat.js');

// Firebase config is sent from the main thread via postMessage after SW registration.
// This avoids bundler limitations in SW scope.
let _messaging = null;

function _attachHandlers() {
  // ── Background push received ────────────────────────────────────────────
  _messaging.onBackgroundMessage((payload) => {
    console.log('[SW] Background FCM received:', payload);
    const { title, body } = payload.notification || {};
    const severity = payload.data?.severity || 'medium';

    self.registration.showNotification(title || '🚨 Safety Alert', {
      body: body || 'Safety alert triggered.',
      icon: '/pwa-192x192.png',
      badge: '/pwa-192x192.png',
      vibrate: severity === 'critical' ? [200, 100, 200, 100, 200] : [200, 100, 200],
      data: payload.data || {},
      actions: [
        { action: 'view',    title: '👁️ View Alert' },
        { action: 'dismiss', title: 'Dismiss' },
      ],
      // Collapse repeated alerts of same type — no spam
      tag: `safety-alert-${payload.data?.detected_issue || 'generic'}`,
      renotify: true,
    });
  });
}

function _initIfNeeded(config) {
  if (_messaging || !config || !config.apiKey) return;
  try {
    if (!firebase.apps.length) {
      firebase.initializeApp(config);
    }
    _messaging = firebase.messaging();
    _attachHandlers();
    console.log('[SW] Firebase initialized ✅');
  } catch (e) {
    console.error('[SW] Firebase init failed:', e);
  }
}

// Receive config from main thread
self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'FIREBASE_CONFIG') {
    _initIfNeeded(event.data.config);
  }
});

// ── Notification click handler ──────────────────────────────────────────────
self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  if (event.action === 'dismiss') return;
  event.waitUntil(
    clients.matchAll({ type: 'window', includeUncontrolled: true }).then((clientList) => {
      for (const client of clientList) {
        if (client.url && 'focus' in client) return client.focus();
      }
      return clients.openWindow ? clients.openWindow('/alerts') : undefined;
    })
  );
});
