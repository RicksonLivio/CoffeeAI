import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Plain-JS (ESM) config, as in the NFH web app: avoids bundling the config through esbuild,
// which fails when a Windows Node reads the WSL filesystem through \\wsl.localhost.
// base './' makes the build work under any GitHub Pages path (https://<user>.github.io/<repo>/).
export default defineConfig({
  base: './',
  plugins: [react()],
  server: {
    port: Number(process.env.COFFEEAI_PORT || 5174),
    strictPort: true,
  },
  preview: {
    port: Number(process.env.COFFEEAI_PREVIEW_PORT || 4174),
    strictPort: true,
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
    target: 'es2020',
    chunkSizeWarningLimit: 800,
    rollupOptions: {
      output: {
        manualChunks: {
          charts: ['recharts'],
        },
      },
    },
  },
});
