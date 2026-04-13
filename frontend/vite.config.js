import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// vite-plugin-pwa is incompatible with Vite 8.
// PWA is implemented manually: public/manifest.webmanifest + public/sw.js
export default defineConfig({
  plugins: [react()],

  server: {
    port: 5173,
    strictPort: true,
    host: true,          // expose on network so phones can connect
    hmr: {
      clientPort: 5173,
    },
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
      },
      '/uploads': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },

  },

  build: {
    rollupOptions: {
      output: {
        // Removed manualChunks to fix Vite 8 build error
      },
    },
  },
})
