/* eslint-disable react-refresh/only-export-components */
import { createContext, useContext, useState, useEffect, useCallback } from 'react'
import { authApi, usersApi } from '../api/api'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)
  const [customPpeItems, setCustomPpeItems] = useState([])
  const [noPhoneZone, setNoPhoneZone] = useState(false)

  const fetchMe = useCallback(async () => {
    try {
      const res = await authApi.me()
      setUser(res.data)
      try {
        const cfgRes = await usersApi.getConfig()
        setCustomPpeItems(cfgRes.data.custom_ppe_items || [])
        setNoPhoneZone(!!cfgRes.data.no_phone_zone)
      } catch {
        setCustomPpeItems([])
        setNoPhoneZone(false)
      }
    } catch {
      setUser(null)
      setCustomPpeItems([])
      setNoPhoneZone(false)
      localStorage.removeItem('token')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    const token = localStorage.getItem('token')
    if (token) {
      fetchMe()
      // Re-register FCM on every page reload so token is always fresh in DB
      _setupFCM(token)
    } else {
      setLoading(false)
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fetchMe])

  const _setupFCM = async (accessToken) => {
    try {
      const { initFCM, listenForegroundMessages } = await import('../firebase')
      // Always use /api prefix so requests go through Vite proxy (localhost:8000).
      // This works on desktop AND on phone via ngrok — ngrok tunnels port 5173
      // which includes the Vite proxy, so /api always reaches the backend correctly.
      // Never use a hard-coded LAN IP here: it breaks when phone is on mobile data.
      const backendUrl = import.meta.env.VITE_API_BASE_URL || ''
      console.log('[FCM] Registering token, backendUrl=', backendUrl || '(vite proxy /api)')
      const fcmToken = await initFCM(accessToken, backendUrl)
      if (fcmToken) {
        console.log('[FCM] Token saved to backend ✅, setting up foreground listener')
        listenForegroundMessages()
      } else {
        console.warn('[FCM] initFCM returned null — check console for details')
      }
    } catch (err) {
      console.warn('[FCM] Setup failed:', err)
    }
  }

  const login = async (email, password) => {
    const res = await authApi.login(email, password)
    const token = res.data.access_token
    localStorage.setItem('token', token)
    await fetchMe()
    _setupFCM(token)
  }

  const signup = async (email, password, name, role = 'Construction Worker') => {
    const res = await authApi.signup(email, password, name, role)
    const token = res.data.access_token
    localStorage.setItem('token', token)
    await fetchMe()
    _setupFCM(token)
  }
  const logout = () => {
    localStorage.removeItem('token')
    setUser(null)
    setCustomPpeItems([])
    setNoPhoneZone(false)
  }

  const updateUser = (data) => setUser(prev => ({ ...prev, ...data }))

  const triggerFCMSetup = async () => {
    const token = localStorage.getItem('token')
    if (!token) return null
    return await _setupFCM(token)
  }

  return (
    <AuthContext.Provider value={{
      user, loading, login, signup, logout,
      updateUser, fetchMe,
      customPpeItems, setCustomPpeItems,
      noPhoneZone, setNoPhoneZone,
      setupFCM: triggerFCMSetup,
    }}>
      {children}
    </AuthContext.Provider>
  )
}
export const useAuth = () => {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used inside AuthProvider')
  return ctx
}
