"""Fetch exact viewer-license notices, retaining observed URLs and hashes."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'reveal-LICENSE.txt': 'https://raw.githubusercontent.com/hakimel/reveal.js/6.0.1/LICENSE',
    'source-sans-LICENSE.md': 'https://raw.githubusercontent.com/adobe-fonts/source-sans/release/LICENSE.md',
}


def main():
    destination = ROOT / 'docs/site/assets/notices'
    destination.mkdir(parents=True, exist_ok=True)
    receipt = {'format': 'iris-viewer-notices-v1', 'sources': {}}
    for name, url in SOURCES.items():
        with urllib.request.urlopen(url, timeout=30) as response:
            payload = response.read()
            address = response.url
        if len(payload) < 500 or b'license' not in payload.lower():
            raise ValueError('not a valid license notice')
        downloaded_sha256 = hashlib.sha256(payload).hexdigest()
        payload = ('\n'.join(line.rstrip() for line in payload.decode('utf-8').splitlines()).rstrip() + '\n').encode('utf-8')
        (destination / name).write_bytes(payload)
        receipt['sources'][name] = {'url': address, 'download_sha256': downloaded_sha256,
                                  'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload),
                                  'normalization': 'LF and trailing-whitespace only; license wording unchanged'}
    (destination / 'sources.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
