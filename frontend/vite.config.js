import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000
  },
  build: {
    outDir: 'dist'   // Render busca esta carpeta al publicar
  },
  base: './'         // importante para rutas relativas en producción
})
