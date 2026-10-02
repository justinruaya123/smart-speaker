"""Build the dependency-free inspection kernel; no pretrained model download."""
from pathlib import Path
import base64
import shutil
import subprocess
import argparse

ROOT = Path(__file__).resolve().parent

def build(compiler=None,output_dir=None,module=None):
    output_dir=Path(output_dir) if output_dir else ROOT/'assets'
    module=Path(module) if module else ROOT/'kernel_base64.mjs'
    output_dir.mkdir(parents=True,exist_ok=True)
    compiler=str(compiler or shutil.which('clang') or '')
    if not compiler:raise RuntimeError('Provide a Clang or Zig compiler with wasm32 support.')
    if Path(compiler).stem=='zig':
        command=[compiler,'build-exe','-target','wasm32-freestanding','-O','ReleaseFast','-fno-entry',
            '--import-symbols','--export=conv','--export=ctc','--export=__heap_base','--export-memory',
            '--initial-memory=33554432','--max-memory=67108864','-femit-bin='+str(output_dir/'kernel.wasm'),
            '-cflags','-O3','-ffp-contract=off','--',str(ROOT/'kernel.c')]
    else:
        command=[compiler,'--target=wasm32', '-O3', '-ffp-contract=off',
    '-nostdlib', str(ROOT / 'kernel.c'), '-o', str(output_dir/'kernel.wasm'),
    '-Wl,--no-entry', '-Wl,--export=conv', '-Wl,--export=ctc',
    '-Wl,--export=__heap_base', '-Wl,--export-memory',
    '-Wl,--initial-memory=33554432', '-Wl,--max-memory=67108864',
    '-Wl,--allow-undefined']
    subprocess.run(command,check=True)
    module.write_text('export default "' + base64.b64encode(
        (output_dir/'kernel.wasm').read_bytes()).decode() + '";\n',encoding='utf-8')
    return output_dir/'kernel.wasm'

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler',type=Path);parser.add_argument('--output-dir',type=Path);parser.add_argument('--module',type=Path)
    args=parser.parse_args();print(build(args.compiler,args.output_dir,args.module))
