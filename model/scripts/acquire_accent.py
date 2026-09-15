"""Small public Philippine-English read-speech diagnostic; never training data."""
import hashlib
import html
import json
from pathlib import Path
import re
import sys
import time
import urllib.request
from urllib.parse import urljoin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tinyvcm.data import write_json, write_jsonl, normalize_text


def get(url):
    request = urllib.request.Request(url, headers={"User-Agent": "TinyVCM academic speech evaluation"})
    with urllib.request.urlopen(request, timeout=45) as response:
        return response.read()


def clean(text):
    return html.unescape(re.sub(r"<[^>]+>", " ", text))


records = []
folder = ROOT / "data/raw/speech_accent_archive"
folder.mkdir(parents=True, exist_ok=True)
for name in ["english140", "english348", "tagalog1", "tagalog2", "tagalog4", "tagalog5", "tagalog6", "tagalog7", "tagalog8", "tagalog9"]:
    url = f"https://accent.gmu.edu/samples/{name}/"
    page = get(url).decode("utf-8")
    (folder / f"{name}.html").write_text(page, encoding="utf-8")
    audio = re.search(r'(?:src|href)=[\"\']([^\"\']+\.(?:mp3|wav))[\"\']', page)
    if not audio:
        raise RuntimeError(f"No audio URL on {url}")
    paragraph = re.search(r"Elicitation Paragraph</h\d>\s*(.*?)</aside>", page, re.S)
    if not paragraph:
        raise RuntimeError(f"Missing elicitation prompt on {url}")
    audio_url = urljoin(url, audio.group(1))
    data = get(audio_url)
    path = folder / (name + Path(audio_url).suffix)
    path.write_bytes(data)
    # Check the publisher's birthplace field rather than inferring accent from names.
    birthplace = re.search(r"birth place</td>\s*<td>(.*?)</td>", page, re.S)
    birth = clean(birthplace.group(1)).strip() if birthplace else "unavailable"
    if "philippin" not in birth.lower():
        raise RuntimeError(f"Philippine provenance not confirmed for {name}: {birth}")
    records.append({"id": f"saa/{name}", "source": "Speech Accent Archive", "audio": path.relative_to(ROOT).as_posix(), "url": url, "audio_url": audio_url, "sha256": hashlib.sha256(data).hexdigest(), "speaker_id": f"saa/{name}", "birth_place_as_reported": birth, "reference": normalize_text(clean(paragraph.group(1))), "reference_type": "elicitation prompt, not a verified verbatim transcript", "license": "CC-BY-NC-SA-4.0", "attribution": "Steven H. Weinberger and Matthew C. Kelley, Speech Accent Archive, George Mason University", "split": "external_diagnostic", "training_use": False})
    print(f"ACCENT {name}: {len(data)} bytes", flush=True)
write_jsonl(ROOT / "data/manifests/philippine_english_diagnostic.jsonl", records)
write_json(ROOT / "reports/accent_data_audit.json", {"retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "diagnostic": {"source": "Speech Accent Archive", "speakers": len(records), "use": "evaluation only; all read the same elicitation paragraph, references are prompts, not verified verbatim transcripts", "primary_url": "https://accent.gmu.edu/language/tagalog/", "license": "CC-BY-NC-SA-4.0"}, "training_accent_labels": "SLURP marks native/other English, not Philippine accent. No Philippine-English-specific training claim.", "additional_sources": [{"url": "https://mozilladatacollective.com/datasets/cmmxhw46c00tqnw07xyr94zjk", "status": "45.63 GB multilingual archive; not acquired in this phase"}, {"url": "https://huggingface.co/datasets/Reord-AI/kenya-philippines-twospeaker-english-dialogue", "status": "gated; requires sharing account contact details and accepting conditions; not downloaded"}]})
