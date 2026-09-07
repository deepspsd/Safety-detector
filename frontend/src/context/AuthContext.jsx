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
    if (token) fetchMe()
    else setLoading(false)
  }, [fetchMe])

  const _setupFCM = async (accessToken) => {
    try {
      const { initFCM, listenForegroundMessages } = await import('../firebase')
      // Use explicit backend URL if set; otherwise derive from current origin.
      // When accessed via ngrok (HTTPS tunnel), origin will be the ngrok URL
      // which doesn't serve the backend — so fall back to LAN IP:8000.
      const lanIp = import.meta.env.VITE_LAN_IP || '127.0.0.1'
      const isNgrok = window.location.hostname.includes('ngrok')
      const backendUrl = import.meta.env.VITE_API_BASE_URL
        || (isNgrok ? `http://${lanIp}:8000` : window.location.origin.replace(':5173', ':8000'))
      const fcmToken = await initFCM(accessToken, backendUrl)
      if (fcmToken) {
        listenForegroundMessages()
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

  return (
    <AuthContext.Provider value={{
      user, loading, login, signup, logout,
      updateUser, fetchMe,
      customPpeItems, setCustomPpeItems,
      noPhoneZone, setNoPhoneZone,
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
