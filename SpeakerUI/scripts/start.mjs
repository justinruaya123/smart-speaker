// Compatibility alias: smart-speaker start.py owns setup, updates and service lifetime.
import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const local = resolve(root, process.platform === 'win32' ? '.venv/Scripts/python.exe' : '.venv/bin/python');
const python = process.env.SMART_SPEAKER_PYTHON || (existsSync(local) ? local : process.platform === 'win32' ? 'python' : 'python3');
const child = spawn(python, [resolve(root, 'start.py'), '--mode', 'serve', ...process.argv.slice(2)], { cwd: root, stdio: 'inherit', windowsHide: true });
child.on('error', error => { console.error(error.message); process.exitCode = 1; });
child.on('exit', code => { process.exitCode = code ?? 1; });
for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => child.kill(signal));
