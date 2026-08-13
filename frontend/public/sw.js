/**
 * SafeGuard AI — Manual Service Worker (Vite 8 compatible)
 * Caches only the application shell and static build assets.
 * Authenticated API data and camera evidence always stay network-only.
 */

const CACHE_NAME  = 'safeguard-ai-v2'
const STATIC_URLS = [
  '/',
  '/index.html',
  '/pwa-192x192.png',
  '/pwa-512x512.png',
  '/favicon.svg',
  '/manifest.webmanifest',
]

// ── Install: pre-cache static shell ───────────────────
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(STATIC_URLS))
      .then(() => self.skipWaiting())
  )
})

// ── Activate: remove old caches ───────────────────────
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(
        keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k))
      )
    ).then(() => self.clients.claim())
  )
})

// ── Fetch: network-only data, cached shell/static assets ──
self.addEventListener('fetch', (event) => {
  const { request } = event
  const url = new URL(request.url)

  if (request.method !== 'GET' || url.origin !== self.location.origin) return

  // SPA navigations use the network, with the cached shell only as an
  // offline fallback. Route responses themselves are never cached.
  if (request.mode === 'navigate') {
    event.respondWith(fetchShell(request))
    return
  }

  const isStaticAsset =
    STATIC_URLS.includes(url.pathname) ||
    url.pathname.startsWith('/assets/') ||
    url.pathname.startsWith('/fonts/')

  // API, auth, camera snapshots, and uploaded evidence are intentionally
  // not intercepted, so the browser can never persist them in this cache.
  if (!isStaticAsset) return

  event.respondWith(cacheFirstStatic(request))
})

async function fetchShell(request) {
  try {
    return await fetch(request)
  } catch {
    return (await caches.match('/index.html')) ||
      new Response('Offline', { status: 503 })
  }
}

async function cacheFirstStatic(request) {
  const cached = await caches.match(request)
  if (cached) return cached

  try {
    const networkResponse = await fetch(request)
    if (networkResponse.ok) {
      const cache = await caches.open(CACHE_NAME)
      await cache.put(request, networkResponse.clone())
    }
    return networkResponse
  } catch {
    return new Response('Offline', { status: 503 })
  }
}

// ── Push notification support (future) ────────────────
self.addEventListener('message', (event) => {
  if (event.data?.type === 'SKIP_WAITING') self.skipWaiting()
})
