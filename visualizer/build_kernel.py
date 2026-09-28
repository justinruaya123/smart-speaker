"""Build the dependency-free inspection kernel; no pretrained model download."""
from pathlib import Path
import base64
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
(ROOT / 'assets').mkdir(parents=True, exist_ok=True)
compiler = shutil.which('clang')
if compiler is None:
    raise SystemExit('Install clang with the wasm32 target, or use included kernel.wasm.')
subprocess.run([compiler, '--target=wasm32', '-O3', '-ffp-contract=off',
    '-nostdlib', str(ROOT / 'kernel.c'), '-o', str(ROOT / 'assets/kernel.wasm'),
    '-Wl,--no-entry', '-Wl,--export=conv', '-Wl,--export=ctc',
    '-Wl,--export=__heap_base', '-Wl,--export-memory',
    '-Wl,--initial-memory=33554432', '-Wl,--max-memory=67108864',
    '-Wl,--allow-undefined'], check=True)
(ROOT / 'kernel_base64.mjs').write_text('export default "' + base64.b64encode(
    (ROOT / 'assets/kernel.wasm').read_bytes()).decode() + '";\n')
print('Built', ROOT / 'assets/kernel.wasm')
