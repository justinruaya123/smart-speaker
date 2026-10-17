"""Build the inference ZIP from repository sources, model graphs and fixed audio.

Only pinned Python/Node/package dependencies are downloaded during setup. No
Hugging Face model, dataset, response clip or prebuilt inference ZIP is fetched.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shutil
import sys

from scripts.submission.common import ROOT, read, sha


def verify_local_responses(root=ROOT):
    from scripts.submission.response_runtime import validate_manifest
    folder=Path(root)/'submission/responses';manifest=read(folder/'manifest.json')
    validate_manifest(manifest);files={}
    for row in manifest['responses']:
        path=folder/row['path']
        if not path.is_file():raise FileNotFoundError('Missing required source audio: '+str(path))
        if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:
            raise ValueError('Source audio integrity failed: '+str(path))
        files[str(path.relative_to(root))]=row['sha256']
    return files


def build(output):
    from scripts.runtime.speaker_tools import node_tools, run
    from scripts.submission.package import main as package, validate_entries
    from scripts.submission.common import runtime_entries
    verify_local_responses();validate_entries(runtime_entries())
    if sha(ROOT/'submission/wake/hey-iris/iris.onnx')!='4cc32fcb2cf20e7bf1310dcbe1d4f4ad0a9ad99d35575d0e31e3c5f5a4db95ac':
        raise ValueError('Hey Iris source graph identity mismatch')
    node,pnpm,env=node_tools(ROOT)
    # The UI's package scripts invoke pnpm recursively; expose the pinned local shim.
    env['PATH']=str(pnpm.parents[2]/'.bin')+os.pathsep+env['PATH']
    ui=ROOT/'SpeakerUI'
    run([node,pnpm,'install','--frozen-lockfile'],cwd=ui,env=env)
    for task in ('check','test','build'):
        run([node,pnpm,'run',task],cwd=ui,env=env)
    production=ROOT/'.cache/submission/ui-production-source-build'
    production.mkdir(parents=True,exist_ok=True)
    for name in ('package.json','pnpm-lock.yaml'):
        shutil.copyfile(ui/name,production/name)
    run([node,pnpm,'install','--frozen-lockfile','--prod','--ignore-scripts','--config.node-linker=hoisted'],cwd=production,env=env)
    run([sys.executable,'-m','unittest','scripts.test.test_submission_inference','scripts.test.test_submission_speech',
         'scripts.test.test_submission_wake','scripts.test.test_submission_package','scripts.test.test_submission_source_build'],cwd=ROOT,env=env)
    package(['--output',str(Path(output).resolve()),'--node-modules',str(production/'node_modules'),'--no-update-inventory'])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'dist/final-submission/smart-speaker-inference-2m.zip')
    args=parser.parse_args(argv)
    if args.output.exists():raise FileExistsError('Preserve existing build: '+str(args.output))
    build(args.output)


if __name__=='__main__':main()
