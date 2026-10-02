import { fileURLToPath, URL } from 'node:url';
import { defineConfig, loadEnv } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const target = `http://127.0.0.1:${env.PORT || '3000'}`;
  return {
    plugins: [svelte(), tailwindcss()],
    resolve: { alias: { $lib: fileURLToPath(new URL('./src/lib', import.meta.url)) } },
    server: {
      port: 5173, strictPort: true,
      proxy: { '/api': target, '/auth': target, '/ws': { target, ws: true } },
    },
  };
});
