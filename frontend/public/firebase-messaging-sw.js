// firebase-messaging-sw.js
// Handles FCM push notifications in the background (when app tab is closed/hidden).
// Must live at root scope (/firebase-messaging-sw.js) — placed in public/.

importScripts('https://www.gstatic.com/firebasejs/10.12.0/firebase-app-compat.js');
importScripts('https://www.gstatic.com/firebasejs/10.12.0/firebase-messaging-compat.js');

// Statically initialize Firebase so SW is ready immediately on background push events
const firebaseConfig = {
  apiKey: "AIzaSyBNli_ad-hpmsl5I31pS6covB18Fs0nsQs",
  authDomain: "occusafe-4f0c8.firebaseapp.com",
  projectId: "occusafe-4f0c8",
  messagingSenderId: "983747723184",
  appId: "1:983747723184:web:3313be9d9d8b741e642075",
};

firebase.initializeApp(firebaseConfig);
const messaging = firebase.messaging();

// ── Background push received ────────────────────────────────────────────────
messaging.onBackgroundMessage((payload) => {
  console.log('[SW] Background FCM received:', payload);
  const title = payload.notification?.title || payload.data?.title || '🚨 Safety Alert';
  const body = payload.notification?.body || payload.data?.body || payload.data?.message || 'Safety alert triggered.';
  const severity = payload.data?.severity || 'medium';

  self.registration.showNotification(title, {
    body: body,
    icon: '/pwa-192x192.png',
    badge: '/pwa-192x192.png',
    vibrate: severity === 'critical' ? [200, 100, 200, 100, 200] : [200, 100, 200],
    data: payload.data || {},
    actions: [
      { action: 'view',    title: '👁️ View Alert' },
      { action: 'dismiss', title: 'Dismiss' },
    ],
    tag: `safety-alert-${payload.data?.detected_issue || 'generic'}`,
    renotify: true,
  });
});

// Optional message event listener for compatibility
self.addEventListener('message', (event) => {
  if (event.data?.type === 'SKIP_WAITING') {
    self.skipWaiting();
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

