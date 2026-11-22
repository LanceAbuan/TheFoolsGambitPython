import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Dev server proxies the FastAPI backend so the React app talks to the same
// REST + WS contract as the Svelte client, with no CORS or mixed-origin work.
const proxy = {
  '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true },
  '/ws': { target: 'ws://127.0.0.1:8000', ws: true },
};

export default defineConfig({
  plugins: [react()],
  server: { port: 5174, strictPort: true, proxy },
  preview: { port: 5175, strictPort: true, proxy },
  build: {
    outDir: 'dist',
    sourcemap: false,
    // rolldown-vite minifies CSS with Lightning CSS, which downlevels
    // `light-dark()` into a var(--lightningcss-light/dark) toggle pattern that
    // nothing defines — that turned every board square black in production.
    // esbuild's CSS minifier keeps `light-dark()` (Chrome 123+, Firefox 120+,
    // Safari 17.5+ all support it natively) untouched.
    cssMinify: 'esbuild',
  },
});
