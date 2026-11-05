"""Invoke the frozen baseline-A training snapshot, never the application."""
from pathlib import Path
import os
import subprocess
import sys


def main():
    source=Path(__file__).resolve().parents[1]/'submission/shared-demo-a/source'
    environment=dict(os.environ)
    environment.pop('PYTHONPATH',None)
    return subprocess.call([sys.executable,'-B',str(source/'reproduce_a.py'),*sys.argv[1:]],
                           cwd=source,env=environment)


if __name__=='__main__': raise SystemExit(main())
