"""Render the concise Iris README, including ten truthful JSON-backed diagrams."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIX = r'''# Iris

**Say Hey Iris. Give a supported command. Iris recognizes it locally and shows what happened.**

A finite voice-command model: **258,912 parameters · original epoch 44 · FP32 · 31 commands + UNKNOWN**. Recognition is local; optional services supply actions and external data. No transcript, ASR or LLM runs in the product. **Wake smoke gate FAILED; no replacement was promoted. This is an unqualified research build, not a working-acoustic-release claim.** Pi and fresh human live-microphone reliability remain unmeasured.

[Two-minute slides and script](docs/PRESENTATION-optionb-tiny-2minute.md) · [Detailed animated presentation](docs/site/index.html) · [Getting started](docs/IRIS-GETTING-STARTED.md) · [Model card](docs/MODEL-CARD-iris-epoch44.md) · [Verified JSON](docs/site/assets/iris.json)

## Quickstart

No release ZIP or large dataset is required. Start with Git and Python 3.11+; the launchers prepare a local Python 3.12 runtime when needed and reuse it. First setup needs network access. The ONNX models remain in Git; recognition never downloads a training corpus or runs fitting.

```sh
git clone --depth 1 --branch main --single-branch https://github.com/justinruaya123/smart-speaker.git
cd smart-speaker
python inference.py --check
python benchmark.py --output .cache/benchmark-run-01
python inference.py --mode ui --no-audio
python inference.py --wav /absolute/path/to/hey-iris-and-command.wav --wake-wav
```

`inference.py` and `benchmark.py` use the separate minimal lock at `scripts/runtime/epoch44/`, never training requirements. Core inference installs ten packages, not NCNN/OpenCV/SciPy/Torch. The default benchmark adds PyArrow only when called; device/audio modes add SoundDevice. Node24.13.0/pnpm12.5.1 and UI dependencies are fetched only for `--mode ui`. Advanced `--no-setup` uses the compatible environment's Python explicitly (`.venv/Scripts/python.exe` on Windows, `.venv/bin/python` on POSIX) and forbids dependency/benchmark-asset downloads.

The default benchmark downloads only the fixed **196-recording holdout, 14.55 MB**, from its original immutable source, verifies SHA-256 and reuses its local cache. It does not fetch the full TEST, training, supplemental or owner corpora. Original human and synthetic provenance and ambiguous exclusions are retained. This already-exposed regression is not fresh blind qualification or a Pi performance result until run on that Pi. For your own permitted mono16kHz audio, the explicit-manifest path does not download benchmark recordings:

```sh
python benchmark.py --manifest MANIFEST.jsonl --data-root ORIGINAL_AUDIO_ROOT --output NEW_RESULTS
```

The output must be a new directory. Recordings remain in ignored caches, not Git. The model stays `submission/models/optionb-tiny`, ONNX CPU. Checks/WAVs/benchmarks stay silent and action-free; owner-protected actions retain their independent veto. No recognition or benchmark command launches fitting.

### Raspberry Pi 4 (64-bit)

Use 64-bit Raspberry Pi OS (Bookworm/Trixie, aarch64). From the clone:

```sh
python3 inference.py --check
python3 benchmark.py --output .cache/pi-benchmark-01
```

Microphone/speaker access additionally needs the OS PortAudio library (`sudo apt install libportaudio2`); it is not needed for the model check or recorded-audio benchmark. The pinned SoundFile ARM64 wheel includes libsndfile. Physical Pi/live-microphone qualification remains unmeasured; Linux ARM64 CI is software compatibility, not Pi 4 hardware performance. Optional UI/providers remain separate from these two commands.

For an **optional full-corpus replay**, prepare actual evaluation audio without fitting using the [pinned original source/environment](docs/IRIS-GETTING-STARTED.md#prepare-original-evaluation-inputs-without-fitting). This is not required for the default small benchmark:

```sh
python -B reproduce.py prepare-evaluation \
  --source-checkout PINNED_SOURCE --source-python PINNED_PYTHON \
  --output .cache/iris-publication-evaluation/run-01
python benchmark.py --model-choice epoch44 \
  --manifest PATH --data-root PINNED_SOURCE --output NEWDIR --repeats 3
```

Use the actual pinned source revision `3c05b3beb46f06c3108089e0d8c79474670c6cb7` and its Python for `PINNED_SOURCE`/`PINNED_PYTHON`. Set `PATH` to `PINNED_SOURCE/.cache/iris-publication-evaluation/run-01/portable/human-test-holdout.jsonl`; `NEWDIR` must be new. Metadata alone is not audio. [Full setup, exact original prepare command and action boundaries](docs/IRIS-GETTING-STARTED.md).

## Pipeline

```mermaid
flowchart LR
  MIC[Host microphone PCM] --> WAKE[Native Hey Iris]
  WAKE --> CAP[One command / VAD]
  CAP --> VCM[Epoch44 atomic VCM]
  WAKE -. wake reference .-> OWNER[Independent owner / overlap]
  CAP --> OWNER
  VCM --> GUARD[Final acceptance + typed slots]
  OWNER --> GUARD
  GUARD --> ACT[One persistent dispatcher]
  ACT --> RECEIPT[Source-tagged result / UI]
  VCM -. tentative / never actuates .-> RECEIPT
  WAV[WAV / benchmark] --> VCM
  WAV -. non-actuating / silent .-> RECEIPT
```

One wake owns one command. The acoustic decision, owner veto and action outcome are separate. One host-owned playback path provides wake acknowledgment, intent beep and completion cue only after the final result. Unknown provider outcomes remain unknown and are not retried automatically.

OpenWakeWord belongs to this **inference wake stage**, not the command classifier or a separate product. The [experimental adapter](scripts/runtime/iris_openwakeword.py) and [source/evaluation record](experiments/iris-openwakeword-publication-v1/findings.json) live on **main** with the inference code. That candidate failed its acoustic gates and was not installed, promoted or wired into the live worker; the existing native wake default remains unchanged. Publishing source is not acoustic qualification.

## Model

```mermaid
flowchart LR
  PCM[Mono16kHz] --> MEL[40-bin causal log-mel]
  MEL --> NORM[Human TRAIN-only normalization]
  NORM --> STRIDE[Feature stride2]
  STRIDE --> STEM[Causal stem / kernel5]
  STEM --> BLOCKS[Four width160 residual blocks]
  BLOCKS --> POOL[Valid-length masked mean + max]
  POOL --> HEAD[320 features / one32-way head]
  HEAD --> POLICY[Non-UNKNOWN / posterior at least .50 / margin at least .10]
  POLICY --> CMD[Complete intent + fixed slots]
```

Two depthwise/pointwise stages per block; kernel 7, dilation 1/2/4/8, LayerNorm and SiLU. Class 0 is UNKNOWN. **18 intent families and 93 wording definitions are not 93 transcript outputs.** Graph:1,082,940 bytes; five-file bundle:1,088,682 bytes. Native wake82,114 parameters, owner129,985 runtime parameters and separate 2.33 MB VAD graph are **additional** costs. [Exact hashes, architecture and limits](docs/MODEL-CARD-iris-epoch44.md).

## Ten use cases

All31 positive classes are assigned exactly once by the [generator/tests](scripts/iris_publication_data.py). The [deck](docs/site/index.html#uses) shows ten selectors in one pipeline/JSON/output frame. These are **supported wording definitions and illustrated outputs**, not observed recognition/provider results.

| ID | Category | Output/source boundary |
|---|---|---|
'''
SUFFIX = r'''
Voice values stay finite: timers10/30/60s; alarms06:00/08:00/21:00 Manila; temperatures18/22/26°C; brightness20/60/100%; red/blue/green; drink water/study/exercise. Broader touch/configuration controls do not create acoustic recognition.

## Data, training and testing

| Split | Human | Synthetic |
|---|---:|---:|
| TRAIN |1,889|21,426|
| DEV |256|995|
| Calibration |210|0|
| TEST |1,050|3,368|
| HOLDOUT |96|100|

TRAIN uses 1,889 human +5,982 synthetic primary rows and 15,444 supplemental synthetic rows; [exact revisions/terms](docs/IRIS-CITATIONS.md). Only human TRAIN fits normalization. Original recipe: seed 53, CPU/four Torch threads,6 human +42 mixed epochs,96 batches/epoch,32 samples/batch, AdamW,4,608 updates,2,987.91s fit. [Actual curve/summary](presentation/evidence/manifest.json).

| Reproducibility statement | Status |
|---|---|
| Exact original artifact verification/loading | PASS: `python -m scripts.optionb_epoch44 verify --load` |
| Original source/features | Hash-bound pinned-source reconstruction evidence |
| GPU-to-GPU repeatability | Two historical full fits matched; selected epoch 28 |
| Exact original CPU epoch 44 retraining | **Not verified; comparison failed. No success lock is invented.** |

The [original-source runbook](docs/REPRODUCIBILITY-optionb-epoch44.md) preserves failures; benchmarking is a separate lane. No model/policy is retuned for these exposed cohorts:

| Cohort | Correct accepts / positives | Wrong accepts, including negatives | Negative false accepts | Ambiguous, unscored |
|---|---:|---:|---:|---:|
| HOLDOUT |72/80|3|1/14|2|
| TEST |393/609|59|24/256|185|

Exact metadata-wording subsets:58/62 and 173/232, not full 93-wording human coverage;21 definitions lack exact human references. Fresh human wake/command, ownership and Pi qualification remain unmeasured. **New synthetic wake study failed:** existing native wake detected 1/32 positives with 0/68 false wakes; the separately trained same-size companion detected 21/32 with 2/68 false wakes. **Neither passed; no replacement was promoted.** [Actual wake receipt](presentation/evidence/wake-smoke-status.json). Historical17/32 calibration is a different cohort, not the new gate. Static-site/software checks cannot turn this into an acoustic release pass.

## Desktop benchmark

| Actual run | p50 / p95 recognition latency |
|---|---:|
| Historical original PC |4.82 /8.41 ms|
| Historical later repeat v2 |5.36 /8.69 ms|
| 2026-10-24 reference driver, **under publication load** |9.34 /15.34 ms|
| Final integrated adapter / idle-host interaction |Separate receipt required; not inferred here|
| Raspberry Pi |**Not measured — `pi:null`**|

The historical and original-reference runs use Intel Core i7-11700/Windows 11/Python 3.13.14, CPUExecutionProvider, one ONNX/frontend BLAS thread; all named runs use 1,146 original human recordings ×3 =3,438 calls. Repeats do not inflate accuracy denominators. Scope: PCM validation, frontend, normalization, ONNX and finite decision; **excludes** speech, decode, wake/VAD waiting, worker/gateway/UI, audio and providers. The new reference run is not quiesced, not the new adapter and not end-to-end latency. No unrelated256-DEV memory probe is called full-PC memory. [Receipts/hashes](docs/site/assets/iris.json).

## Limits and real versus synthetic

Only enabled final live microphone commands may actuate after guards. Tentative previews, WAVs, checks and benchmarks never do. Missing providers show **Synthetic — no device changed**, not a real acknowledgment. Unknown outcomes remain unknown. OpenWeather caches keep honest timestamps; thermostat targets are synthetic only, separate from observed sensors. Recognized lights bind to the Hue lamp; CALL opens the dining camera **video only**, not telephony.

No arbitrary names, songs, numbers, dictation or general search. Setup/providers are not an offline/privacy absolute. Data-specific research, non-commercial and consent restrictions remain; no blanket MIT license is claimed for weights/recordings.

## Development and rollback

[Render/source instructions](presentation/README.md) · [Citation audit](docs/IRIS-CITATIONS.md) · [Release log](docs/RELEASE-iris-publication-v1.md) · [Revert guide](docs/REVERT-iris-publication-v1.md). Historical [root model card](MODEL_CARD.md), [dataset directory](docs/DATASETS.md) and [experiment index](EXPERIMENTS.md) are preserved; their969k/2M results do not describe this VCM.

```sh
python -B scripts/iris_publication_data.py --check
python -B -m unittest scripts.test.test_iris_publication_data -v
python -B presentation/validate.py
```

## Citation

Cite the exact repository revision and frozen model hash. Primary dataset: **DOI[10.57967/hf/10723](https://doi.org/10.57967/hf/10723)**, revision `25111444af3adff7588d86ab27c304bb080895cf`. Supplemental:[`martinnavs/ai231-fil-supplemental-data`](https://huggingface.co/datasets/martinnavs/ai231-fil-supplemental-data/tree/230c8b8d58caf78e4bc90886d9c6fa77fe6024b8), revision `230c8b8d58caf78e4bc90886d9c6fa77fe6024b8`. Retain original-source attribution, consent and research-use restrictions; the [audit](docs/IRIS-CITATIONS.md) distinguishes actual membership from wider sources.

```bibtex
@software{iris_epoch44_2026,
  author = {{Iris project contributors}},
  title = {Iris: Original OptionB Epoch44 Finite-Command Smart Speaker},
  year = {2026},
  url = {https://github.com/justinruaya123/smart-speaker},
  version = {OptionBTinyVCM-1 epoch44},
  note = {258912 parameters; FP32; graph SHA256 c1cd773b92337fd2a453554b047fa555c692fb981b5975700a32260388d4b9ce; cite exact source revision used}
}
```
'''


def render(data):
    parts = [PREFIX]
    for item in data['use_cases']:
        parts.append(f"| {item['id']} | {item['title']} | {item['output']['badge']} / {item['output']['headline']} |\n")
    parts.append('\n<details>\n<summary>Ten mini diagrams, exact JSON and honest example outputs</summary>\n\n')
    for item in data['use_cases']:
        examples = ' / '.join('“' + e['wording'] + '”' for e in item['examples'])
        parts.append(f"**{item['id']} · {item['title']} — {examples}**\n\n```mermaid\nflowchart LR\n")
        a, b, c = item['diagram']
        parts.append(f'  A["{a}"] --> B["{b}"] --> C["{c}"]\n```\n\n')
        commands = '; '.join('`' + json.dumps(e['command'], separators=(',', ':')) + '`' for e in item['examples'])
        output = item['output']
        parts.append(f"{commands} → **{output['badge']}: {output['headline']}.** {output['detail']}\n\n")
    parts.append('</details>\n')
    suffix = SUFFIX
    desktop = data['benchmarks']['desktop']
    if desktop:
        latency = desktop['latency']
        current = f"| Current adapter, product lock (**not quiesced**) | {latency['p50_ms']:.2f} / {latency['p95_ms']:.2f} ms |"
        description = f"\n\nCurrent environment: NumPy {desktop['environment']['numpy']} / ONNX Runtime {desktop['environment']['onnxruntime']}; peak whole-process RSS **{desktop['peak_process_rss_mib']:.2f} MiB**, not model-only RAM. [Actual audited receipt](presentation/evidence/desktop-product.json): all 1,146 WAV/decoded-PCM identities and 3,438 complete typed outputs independently matched frozen gold. Direct VCM only; wake/owner/UI/providers and speaking time excluded. This does not improve the failed wake gate."
        suffix = suffix.replace('| Final integrated adapter / idle-host interaction |Separate receipt required; not inferred here|', current)
        pi_row = '| Raspberry Pi |**Not measured — `pi:null`**|'
        suffix = suffix.replace(pi_row, pi_row + description)
    parts.append(suffix)
    return ''.join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    result = render(json.loads((ROOT / 'docs/site/assets/iris.json').read_text(encoding='utf-8')))
    destination = ROOT / 'README.md'
    if args.check:
        if destination.read_text(encoding='utf-8') != result:
            raise SystemExit('README is stale; run presentation/build_readme.py')
        print('readme_verified: two architecture + ten use-case diagrams; Citation last')
    else:
        destination.write_text(result, encoding='utf-8', newline='\n')
        print('readme_written: README.md')


if __name__ == '__main__':
    main()
