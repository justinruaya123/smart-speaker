import { spawnSync } from 'node:child_process';
import { copyFileSync, constants, mkdirSync } from 'node:fs';

const pnpm = process.platform === 'win32' ? 'pnpm.cmd' : 'pnpm';
const install = spawnSync(pnpm, ['install', '--frozen-lockfile'], { stdio: 'inherit' });
if (install.error) throw install.error;
if (install.status !== 0) process.exit(install.status ?? 1);

try {
  copyFileSync('.env.example', '.env', constants.COPYFILE_EXCL);
  console.log('Created .env from .env.example');
} catch (error) {
  if (error.code !== 'EEXIST') throw error;
  console.log('Kept existing .env');
}

console.log('UI dependencies ready. Use python start.py in the parent smart-speaker folder for the Python ONNX runtime, models and launch.');
