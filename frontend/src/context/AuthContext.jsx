/* eslint-disable react-refresh/only-export-components */
import { createContext, useContext, useState, useEffect, useCallback } from 'react'
import { authApi, usersApi } from '../api/api'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [user,           setUser]           = useState(null)
  const [loading,        setLoading]        = useState(true)
  const [customPpeItems, setCustomPpeItems] = useState([])
  const [noPhoneZone,    setNoPhoneZone]    = useState(false)

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

  const login = async (email, password) => {
    const res = await authApi.login(email, password)
    localStorage.setItem('token', res.data.access_token)
    await fetchMe()
  }

  const signup = async (email, password, name, role = 'Construction Worker') => {
    const res = await authApi.signup(email, password, name, role)
    localStorage.setItem('token', res.data.access_token)
    await fetchMe()
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
