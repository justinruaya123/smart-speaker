import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';

const root = fileURLToPath(new URL('..', import.meta.url));
// One awaited compiler chain works with npm, pnpm and the packaged Node runtime.
for (const args of [['node_modules/vite/bin/vite.js', 'build'],
  ['scripts/compress.mjs'], ['node_modules/typescript/bin/tsc', '-p', 'tsconfig.server.json']]) {
  await new Promise((done, reject) => {
    const child = spawn(process.execPath, [resolve(root, args[0]), ...args.slice(1)], { cwd: root, stdio: 'inherit', windowsHide: true });
    child.once('error', reject);
    child.once('exit', code => code === 0 ? done() : reject(new Error(`Build stage failed: ${args[0]} (${code})`)));
  });
}
