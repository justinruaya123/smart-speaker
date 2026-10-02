"""Create an offline HTML build with the exact local weights, WAVs and evidence."""
from __future__ import annotations

import base64
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
VIS = ROOT / "visualizer"


def build(*,assets_dir=None,output=None,kernel_module=None) -> Path:
    html = (VIS / "index.html").read_text(encoding='utf-8')
    style = (VIS / "style.css").read_text(encoding='utf-8')
    html = re.sub(r'<link rel="stylesheet"[^>]*>', f"<style>{style}</style>", html)
    html = re.sub(r'<script type="module"[^>]*></script>', "", html)
    assets = {"/visualizer/assets/" + path.name:
              base64.b64encode(path.read_bytes()).decode()
              for path in (Path(assets_dir) if assets_dir else VIS / "assets").iterdir() if path.is_file()}
    historical=ROOT / "final_suite/data/evidence.json.gz"
    if historical.exists():assets["/final_suite/data/evidence.json.gz"]=base64.b64encode(historical.read_bytes()).decode()
    boot = "const LOCAL_ASSETS=" + json.dumps(assets) + ";\n"
    boot += """window.fetch = async path => {
      const key = new URL(path, 'https://fixture.invalid/visualizer/').pathname;
      if (!(key in LOCAL_ASSETS)) return new Response('missing local asset', {status:404});
      return new Response(Uint8Array.from(atob(LOCAL_ASSETS[key]),c=>c.charCodeAt(0)));
    };\n"""
    boot += "window.inlineCaptureWorklet=" + json.dumps(
        (VIS / "capture-worklet.mjs").read_text(encoding='utf-8')) + ";\n"
    boot += (Path(kernel_module) if kernel_module else VIS / "kernel_base64.mjs").read_text(encoding='utf-8').replace(
        "export default", "const embeddedKernel =") + "\n"
    for name in ["core.mjs", "scene.mjs", "evidence.mjs", "app.mjs"]:
        source = (VIS / name).read_text(encoding='utf-8')
        source = re.sub(r"^import .*?;\n", "", source, flags=re.MULTILINE)
        source = re.sub(r"\bexport (?=(?:async )?(?:function|class|const))", "", source)
        boot += source + "\n"
    html = html.replace("</body>", "<script>" + boot + "</script></body>")
    target = Path(output) if output else VIS / "standalone.html"
    target.write_text(html,encoding='utf-8')
    return target


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-dir',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--kernel-module',type=Path)
    args=parser.parse_args();print(build(assets_dir=args.assets_dir,output=args.output,kernel_module=args.kernel_module))
