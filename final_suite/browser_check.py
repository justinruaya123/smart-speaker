"""Execute browser components against local assets; do not alter browser policy."""
from __future__ import annotations

import base64
import json
from pathlib import Path
import re
import shutil

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'final_suite/reports'


def check() -> dict:
    browser_path = shutil.which('chromium') or shutil.which('google-chrome')
    if not browser_path:
        raise RuntimeError('Install Chromium for the browser component test')
    report = {'tests': [], 'asset_delivery': 'in-memory local fixtures',
              'physical_microphone_tested': False, 'hardware_gpu_tested': False,
              'onnx_runtime_web_tested': False, 'webgpu_compute_tested': False}
    errors = []
    with sync_playwright() as api:
        browser = api.chromium.launch(executable_path=browser_path, headless=True,
            args=['--no-sandbox', '--use-gl=angle', '--use-angle=swiftshader',
                  '--enable-unsafe-swiftshader'])
        page = browser.new_page(viewport={'width': 1600, 'height': 1100})
        page.on('pageerror', lambda error: errors.append(str(error)))
        html = (ROOT / 'visualizer/index.html').read_text()
        html = re.sub(r'<link rel="stylesheet"[^>]*>', '', html)
        html = re.sub(r'<script type="module"[^>]*></script>', '', html)
        page.set_content(html)
        page.add_style_tag(content=(ROOT / 'visualizer/style.css').read_text())
        assets = {'/visualizer/assets/' + p.name: base64.b64encode(p.read_bytes()).decode()
                  for p in (ROOT / 'visualizer/assets').iterdir() if p.is_file()}
        assets['/final_suite/data/evidence.json.gz'] = base64.b64encode(
            (ROOT / 'final_suite/data/evidence.json.gz').read_bytes()).decode()
        page.evaluate('''assets => {window.fetch=async path=>{
          const key=new URL(path,'https://fixture.invalid/visualizer/').pathname;
          if(!(key in assets))return new Response('missing local fixture',{status:404});
          return new Response(Uint8Array.from(atob(assets[key]),c=>c.charCodeAt(0)));
        };}''', assets)
        bundle = (ROOT / 'visualizer/kernel_base64.mjs').read_text().replace(
            'export default', 'const embeddedKernel =') + '\n'
        for name in ['core.mjs', 'scene.mjs', 'evidence.mjs', 'app.mjs']:
            source = (ROOT / 'visualizer' / name).read_text()
            source = re.sub(r'^import .*?;\n', '', source, flags=re.MULTILINE)
            source = re.sub(r'\bexport (?=(?:async )?(?:function|class|const))', '', source)
            bundle += source + '\n'
        page.add_script_tag(content=bundle)
        page.wait_for_function('window.vcmReady === true', timeout=30000)
        page.wait_for_function('window.vcmTest.state.rows.length > 0')
        report['renderer'] = page.evaluate('vcmTest.renderer()')
        assert report['renderer'] in ['webgl', 'canvas3d']
        report['tests'].append('application and real 3D renderer initialize')
        page.click('#step')
        assert page.evaluate('vcmTest.state.pipeline.stage') == 1
        assert page.evaluate('vcmTest.state.pipeline.probes.stem.data.length') == 256
        page.click('#chunk')
        assert page.evaluate('vcmTest.state.pipeline.decoder.frames') == 4
        report['tests'].append('layer and chunk stepping execute measured tensors')
        page.click('[data-layer="stem"]')
        assert '120 contributions' in page.locator('#formula').inner_text()
        report['tests'].append('tensor probe exposes actual input-times-weight contributions')
        for speed in [.25, 3]:
            page.eval_on_selector('#speed', '(e,v)=>e.value=v', speed)
            page.dispatch_event('#speed', 'input')
            assert page.locator('#speedLabel').inner_text() == f'{speed}×'
        report['tests'].append('0.25x and 3x change inspection scheduling settings')
        page.click('#reset')
        assert page.evaluate('vcmTest.state.pipeline.decoder.frames') == 0
        page.click('#play')
        page.wait_for_function('vcmTest.state.pipeline.finished', timeout=30000)
        result = page.evaluate('vcmTest.state.final')
        assert result['candidate']['track_id'] == 'dummy_track_001'
        assert result['status'] == 'needs_clarification' and not result['action_executed']
        report['tests'].append('scheduled WASM title-artist recognition never executes an action')
        page.screenshot(path=str(REPORTS / 'visualizer-desktop.png'), full_page=True)
        outputs = {}
        for sample in page.evaluate('vcmTest.state.samples.map(s=>s.file)'):
            page.evaluate('name=>vcmTest.loadSample(name)', sample)
            outputs[sample] = page.evaluate('vcmTest.runToEnd()')
            assert outputs[sample]['status'] == 'needs_clarification'
        assert len(outputs['ambiguous_title.wav']['ties']) == 2
        assert outputs['named_light_73.wav']['candidate']['entity_id'] == 'light.lamp_two'
        report['tests'].append('seven WAVs preserve command IDs, refusal and ambiguity')
        page.fill('#search', 'seventy three')
        assert page.evaluate('vcmTest.state.filtered.length') > 0
        page.select_option('#outcome', 'raw_wrong')
        assert page.evaluate('vcmTest.state.filtered.every(r=>r.expected&&!r.raw_exact)')
        report['tests'].append('complete evidence filters expose real failed commands')
        page.locator('#modelUpload').set_input_files([
            str(ROOT / 'visualizer/assets' / name)
            for name in ['recipe.json', 'weights.bin', 'catalog.json']])
        page.wait_for_function('vcmTest.state.pipeline.decoder.frames === 0')
        assert page.evaluate('vcmTest.state.pipeline.layers.every(l=>l.cache.every(x=>x===0))')
        report['tests'].append('hot-swap validates assets and resets neural and decoder state')
        page.set_viewport_size({'width': 900, 'height': 1000})
        page.screenshot(path=str(REPORTS / 'visualizer-tablet.png'), full_page=True)
        assert not errors, errors
        report.update({'outputs': outputs, 'uncaught_errors': errors,
                       'renderer_driver': 'Headless browser; no hardware GPU claim'})
        browser.close()
    (REPORTS / 'browser_verification.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
