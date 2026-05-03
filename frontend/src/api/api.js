import axios from 'axios'

// ── Smart backend URL detection ────────────────────────────────
// In development: ALWAYS use the Vite proxy (/api → localhost:8000).
// This works correctly whether the browser opens via localhost OR
// a LAN IP (e.g. http://10.x.x.x:5173) — the proxy rewrites it server-side.
// In production: use the explicit env var or auto-detect.
function getBaseURL() {
  const envURL = import.meta.env.VITE_API_BASE_URL
  if (envURL && !envURL.includes('localhost') && !envURL.includes('127.0.0.1')) {
    return envURL
  }
  return '/api'
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
  list:     ()               => api.get('/faces/'),
  register: (label, image_b64) => api.post('/faces/register', { label, image_b64 }),
  rename:   (id, label)     => api.put(`/faces/${id}/label`, { label }),
  delete:   (id)            => api.delete(`/faces/${id}`),
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
