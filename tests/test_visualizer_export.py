import subprocess
import sys
from pathlib import Path


def test_inspector_export_help_has_no_historical_payload_dependency():
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,str(root/'visualizer/export_assets.py'),'--help'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert '--bundle' in result.stdout and '--checkpoint' in result.stdout
