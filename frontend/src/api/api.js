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
console.log('BASE_URL:', BASE_URL)

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
    console.log('BASE_URL:', BASE_URL)
    const form = new URLSearchParams()
    form.append('username', email)
    form.append('password', password)
    return api.post('/auth/login', form, {
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
    })
  },
  me: () => api.get('/auth/me'),
}

// ── Users ─────────────────────────────────────
export const usersApi = {
  updateProfile:   (data)  => api.put('/users/me', data),
  getConfig:       ()      => api.get('/users/me/config'),
  updateConfig:    (data)  => api.put('/users/me/config', data),
  // Custom PPE helpers (convenience wrappers)
  getCustomPpe:    ()      => api.get('/users/me/config').then(r => r.data.custom_ppe_items || []),
  updateCustomPpe: (items) => api.put('/users/me/config', { custom_ppe_items: items }),
}

// ── Alerts ────────────────────────────────────
export const alertsApi = {
  list:    (params) => api.get('/alerts/', { params }),
  stats:   ()       => api.get('/alerts/stats'),
  // Pending-review queue (admin review UI)
  pending: ()       => api.get('/alerts/pending'),
  get:     (id)     => api.get(`/alerts/${id}`),
  // Promote pending → confirmed + fire Telegram
  confirm: (id)     => api.patch(`/alerts/${id}/confirm`),
  // Mark as dismissed (false positive)
  dismiss: (id)     => api.patch(`/alerts/${id}/dismiss`),
  delete:  (id)     => api.delete(`/alerts/${id}`),
  clearAll:()       => api.delete('/alerts/'),
}

// ── Faces / Employee roster ────────────────────
export const facesApi = {
  list:     ()                => api.get('/faces/'),
  register: (label, image_b64) => api.post('/faces/register', { label, image_b64 }),
  rename:   (id, label)      => api.put(`/faces/${id}/label`, { label }),
  delete:   (id)             => api.delete(`/faces/${id}`),
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
  status: (jobId) => api.get(`/video/status/${jobId}`),
}

// ── Cameras ───────────────────────────────────────
export const camerasApi = {
  list:    ()         => api.get('/cameras/'),
  get:     (id)       => api.get(`/cameras/${id}`),
  create:  (data)     => api.post('/cameras/', data),
  update:  (id, data) => api.put(`/cameras/${id}`, data),
  delete:  (id)       => api.delete(`/cameras/${id}`),
  restart: (id)       => api.post(`/cameras/${id}/restart`),
  // Zone calibration
  getZones:           (id)                   => api.get(`/cameras/${id}/zones`),
  createZone:         (id, zone_name, polygon_json, metadata = {}) =>
                        api.post(`/cameras/${id}/zones`, { zone_name, polygon_json, ...metadata }),
  deleteZone:         (id, zone_name)        => api.delete(`/cameras/${id}/zones/${encodeURIComponent(zone_name)}`),
  calibrationHistory: (id)                   => api.get(`/cameras/${id}/calibrations`),
  versionCalibration: (id, note = '')        => api.post(`/cameras/${id}/calibrations/snapshot`, null, { params: { note } }),
  restoreCalibration: (id, version, note='') => api.post(`/cameras/${id}/calibrations/${version}/restore`, { note }),
  // Latest JPEG frame (used for floor grid tiles and zone-calibration canvas)
  snapshot: (id) => api.get(`/cameras/${id}/snapshot`),
  // LAN discovery (Hikvision/Dahua/generic RTSP scan)
  discover: ()   => api.post('/cameras/discover'),
}

// ── Platform / Analytics ─────────────────────────
export const platformApi = {
  models:          ()         => api.get('/platform/models'),
  health:          (params)   => api.get('/platform/health', { params }),
  analytics:       (days = 7) => api.get('/platform/analytics/summary', { params: { days } }),
  alertCases:      ()         => api.get('/platform/alert-cases'),
  alertCaseAction: (id, action) => api.post(`/platform/alert-cases/${id}/${action}`),
}
