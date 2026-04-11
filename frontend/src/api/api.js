import axios from 'axios'

// ── Smart backend URL detection ────────────────────────────────
// If the app is opened via a network IP (e.g. http://10.x.x.x:5173 on mobile),
// the backend is assumed to be on the SAME IP on port 8000.
// If opened via localhost, use localhost:8000.
// This env override is still honoured if set explicitly.
function getBaseURL() {
  if (import.meta.env.MODE === 'development') {
    // In development, route through Vite's dev server proxy
    // This bypasses Windows Firewall for port 8000 and fixes API timeouts on local network devices
    return '/api'
  }

  const envURL = import.meta.env.VITE_API_BASE_URL
  if (envURL && envURL !== 'http://localhost:8000') return envURL

  // Auto-detect for production if env isn't strictly set
  const host = window.location.hostname
  if (host === 'localhost' || host === '127.0.0.1') {
    return 'http://localhost:8000'
  }
  console.log("BASE_URL:", host)
  return `http://${host}:8000`
}

const BASE_URL = getBaseURL()
console.log("BASE_URL:", BASE_URL)

const api = axios.create({
  baseURL: BASE_URL,
  headers: { 'Content-Type': 'application/json' }
})

// Attach JWT token to every request
api.interceptors.request.use(cfg => {
  const token = localStorage.getItem('token')
  if (token) cfg.headers.Authorization = `Bearer ${token}`
  return cfg
})

// Redirect to login on 401
api.interceptors.response.use(
  r => r,
  err => {
    if (err.response?.status === 401) {
      localStorage.removeItem('token')
      window.location.href = '/login'
    }
    return Promise.reject(err)
  }
)

export default api

// ── Auth ──────────────────────────────────────
export const authApi = {
  signup: (email, password, name, role = 'Construction Worker') =>
    api.post('/auth/signup', { email, password, name, role }),
  login: (email, password) => {
    console.log("BASE_URL:", BASE_URL)
    const form = new URLSearchParams()
    form.append('username', email)
    form.append('password', password)
    return api.post('/auth/login', form, {
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
    })
  },
  me: () => api.get('/auth/me')
}

// ── Users ─────────────────────────────────────
export const usersApi = {
  updateProfile: (data) => api.put('/users/me', data),
  getConfig:     ()     => api.get('/users/me/config'),
  updateConfig:  (data) => api.put('/users/me/config', data),
  // Custom PPE helpers (convenience wrappers)
  getCustomPpe:    ()       => api.get('/users/me/config').then(r => r.data.custom_ppe_items || []),
  updateCustomPpe: (items)  => api.put('/users/me/config', { custom_ppe_items: items }),
}

// ── Alerts ────────────────────────────────────
export const alertsApi = {
  list: (params) => api.get('/alerts/', { params }),
  stats: () => api.get('/alerts/stats'),
  get: (id) => api.get(`/alerts/${id}`),
  delete: (id) => api.delete(`/alerts/${id}`),
  clearAll: () => api.delete('/alerts/')
}

// ── Faces ─────────────────────────────────────
export const facesApi = {
  list: () => api.get('/faces/'),
  register: (label, image_b64) => api.post('/faces/register', { label, image_b64 }),
  delete: (id) => api.delete(`/faces/${id}`)
}

// ── Video ─────────────────────────────────────
export const videoApi = {
  upload: (file, onProgress) => {
    const form = new FormData()
    form.append('file', file)
    return api.post('/video/upload', form, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: e => onProgress && onProgress(Math.round((e.loaded * 100) / e.total))
    })
  },
  status: (jobId) => api.get(`/video/status/${jobId}`)
}
