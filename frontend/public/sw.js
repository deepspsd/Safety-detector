/**
 * SafeGuard AI — Manual Service Worker (Vite 8 compatible)
 * Implements a Network-First strategy for API calls and
 * Cache-First for static assets.
 */

const CACHE_NAME  = 'safeguard-ai-v1'
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

// ── Fetch: Network-First for API, Cache-First for assets ──
self.addEventListener('fetch', (event) => {
  const { request } = event
  const url = new URL(request.url)

  // Skip non-GET and cross-origin requests
  if (request.method !== 'GET') return
  if (url.origin !== self.location.origin && !url.hostname.includes('localhost')) return

  // API calls → Network-First (5s timeout, fallback to cache)
  if (url.pathname.startsWith('/alerts') || url.pathname.startsWith('/auth') || url.pathname.startsWith('/users')) {
    event.respondWith(networkFirst(request))
    return
  }

  // Static assets → Cache-First
  event.respondWith(cacheFirst(request))
})

async function networkFirst(request) {
  try {
    const networkResponse = await fetch(request)
    if (networkResponse.ok) {
      const cache = await caches.open(CACHE_NAME)
      cache.put(request, networkResponse.clone())
    }
    return networkResponse
  } catch {
    const cached = await caches.match(request)
    return cached || new Response(JSON.stringify({ error: 'Offline' }), {
      headers: { 'Content-Type': 'application/json' }, status: 503
    })
  }
}

async function cacheFirst(request) {
  const cached = await caches.match(request)
  if (cached) return cached
  try {
    const networkResponse = await fetch(request)
    if (networkResponse.ok) {
      const cache = await caches.open(CACHE_NAME)
      cache.put(request, networkResponse.clone())
    }
    return networkResponse
  } catch {
    // Return the app shell for navigation requests (SPA fallback)
    if (request.mode === 'navigate') {
      return caches.match('/') || caches.match('/index.html')
    }
    return new Response('Offline', { status: 503 })
  }
}

// ── Push notification support (future) ────────────────
self.addEventListener('message', (event) => {
  if (event.data?.type === 'SKIP_WAITING') self.skipWaiting()
})
