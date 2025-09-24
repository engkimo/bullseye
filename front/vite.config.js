import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/docja': {
        target: process.env.VITE_DOCJA_BASE || 'http://localhost:8001',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/docja/, ''),
      },
      '/ollama': {
        target: process.env.VITE_OLLAMA_BASE || 'http://localhost:11434',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/ollama/, ''),
      },
    },
  },
})

