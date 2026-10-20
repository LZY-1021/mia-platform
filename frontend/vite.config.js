import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import fs from 'node:fs'
import path from 'node:path'

const demoHost = 'mia-platform'
const certificatePath = path.resolve(import.meta.dirname, '.cert', `${demoHost}.pfx`)
const certificatePassword = 'mia-platform-local-demo'
const hasLocalCertificate = fs.existsSync(certificatePath)

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // `npm run demo` uses a trusted certificate created by scripts/setup-local-https.ps1.
    // Keep the normal `npm run dev` workflow available when the certificate has not
    // been set up yet.
    https: hasLocalCertificate
      ? {
          pfx: fs.readFileSync(certificatePath),
          passphrase: certificatePassword,
        }
      : undefined,
    allowedHosts: [demoHost],
    proxy: {
      '/api': {
        target: 'http://localhost:5000',
        changeOrigin: true,
      },
    },
  },
  build: {
    // 构建产物输出到 backend/static
    outDir: '../backend/static',
    emptyOutDir: true,
  },
})
