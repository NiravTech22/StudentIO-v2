import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
    plugins: [react()],
    base: '/', // Changed to root for Docker/Localhost deployment
    build: {
        outDir: 'dist',
        sourcemap: true,
        rollupOptions: {
            output: {
                manualChunks: {
                    'react-vendor': ['react', 'react-dom'],
                    'ui-vendor': ['framer-motion', 'lucide-react'],
                    'markdown-vendor': ['react-markdown', 'react-syntax-highlighter']
                }
            }
        }
    },
    server: {
        port: 5173,
        strictPort: false,
        open: true
    }
})
