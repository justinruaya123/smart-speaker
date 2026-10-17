import { createRequire } from 'node:module';

// The CLI loads the ESM Vite preprocessor through an unreferenced loader bridge.
// Keep the process alive until its diagnostic callback exits; an unfinished
// initialization must never become a silent successful check.
const deadline = setTimeout(() => { console.error('Svelte diagnostics did not complete within two minutes.'); process.exit(1); }, 120000);
process.once('exit', () => clearTimeout(deadline));
const require = createRequire(import.meta.url);
process.argv = [process.execPath, require.resolve('svelte-check/bin/svelte-check'), ...process.argv.slice(2)];
require('svelte-check/bin/svelte-check');
