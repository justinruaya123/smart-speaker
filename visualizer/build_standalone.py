"""Create an offline HTML build with the exact local weights, WAVs and evidence."""
from __future__ import annotations

import base64
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
VIS = ROOT / "visualizer"


def build() -> Path:
    html = (VIS / "index.html").read_text()
    style = (VIS / "style.css").read_text()
    html = re.sub(r'<link rel="stylesheet"[^>]*>', f"<style>{style}</style>", html)
    html = re.sub(r'<script type="module"[^>]*></script>', "", html)
    assets = {"/visualizer/assets/" + path.name:
              base64.b64encode(path.read_bytes()).decode()
              for path in (VIS / "assets").iterdir() if path.is_file()}
    assets["/final_suite/data/evidence.json.gz"] = base64.b64encode(
        (ROOT / "final_suite/data/evidence.json.gz").read_bytes()).decode()
    boot = "const LOCAL_ASSETS=" + json.dumps(assets) + ";\n"
    boot += """window.fetch = async path => {
      const key = new URL(path, 'https://fixture.invalid/visualizer/').pathname;
      if (!(key in LOCAL_ASSETS)) return new Response('missing local asset', {status:404});
      return new Response(Uint8Array.from(atob(LOCAL_ASSETS[key]),c=>c.charCodeAt(0)));
    };\n"""
    boot += "window.inlineCaptureWorklet=" + json.dumps(
        (VIS / "capture-worklet.mjs").read_text()) + ";\n"
    boot += (VIS / "kernel_base64.mjs").read_text().replace(
        "export default", "const embeddedKernel =") + "\n"
    for name in ["core.mjs", "scene.mjs", "app.mjs"]:
        source = (VIS / name).read_text()
        source = re.sub(r"^import .*?;\n", "", source, flags=re.MULTILINE)
        source = re.sub(r"\bexport (?=(?:async )?(?:function|class|const))", "", source)
        boot += source + "\n"
    html = html.replace("</body>", "<script>" + boot + "</script></body>")
    target = VIS / "standalone.html"
    target.write_text(html)
    return target


if __name__ == "__main__":
    print(build())
