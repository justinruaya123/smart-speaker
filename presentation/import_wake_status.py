"""Import the finished wake study's actual failed gate, never promote a model."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REVISION = 'd1c1392'
SOURCE = 'experiments/iris-wake-publication-v1/study-results.json'


def main():
    revision = subprocess.check_output(['git', 'rev-parse', REVISION + '^{commit}'], cwd=ROOT, text=True).strip()
    payload = subprocess.check_output(['git', 'show', revision + ':' + SOURCE], cwd=ROOT)
    source = json.loads(payload)
    result = {key: source[key] for key in ('experiment', 'baseline', 'companion_test', 'candidate_artifact_publication',
                                          'candidate_parameters', 'verdict', 'release_disposition', 'limitations')}
    result['source_revision'] = revision
    result['source_path'] = SOURCE
    result['source_sha256'] = hashlib.sha256(payload).hexdigest()
    target = ROOT / 'presentation/evidence/wake-smoke-status.json'
    target.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'output': str(target), 'sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'verdict': result['verdict']}, indent=2))


if __name__ == '__main__':
    main()
