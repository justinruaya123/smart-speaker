# Independent clean GPU rebuild and publication

**Verified outcome:** the same GPU comparison model rebuilt independently on the pinned Windows/CPython3.13.14/Torch2.10.0+cu128/CUDA12.8/RTX3080 stack. This is not exact reproduction of the original CPU epoch44 best model. Application source was unchanged at `ad4d6262e7f91b2cb4cb4f75a1b1b082d6b088f3`. No new application code or reproduction framework was added.

| Gate | Actual result | Evidence |
| --- | --- | --- |
| New checkout/environment, source downloads | Own new installation; eight required Parquet shards and184 auxiliary files; two network failures and61-file authenticated retry disclosed | `summary.json`, `auxiliary-download-recovery.json` |
| Source reconstruction |23,315TRAIN/1,251DEV/210calibration rows; exact WAV/PCM | `preparation.json` |
| Features/normalizer |24,566 original tensor hashes and1,889-human normalization rows exact | `input-comparison.json` |
| Full GPU training |48epochs/4,608updates, real `cuda:0`, from random initialization; selected epoch28 | `training-result.json.gz`, `training-curve.jsonl.gz` |
| Independent GPU rebuild equality | All37 tensors, selected/foundation checkpoint bytes, all five bundle files,48epoch losses/DEV results excluding timing and final draw counts exact | `gpu-repeat-comparison.json` |
| DEV/calibration and selection | All466 prediction records and full policy-selection grid byte-identical | `runtime-and-policy-comparison.json`, `policy-selection.json.gz` |
| GPU-to-original-CPU comparison | **FAIL**, preserved original exact comparison; stock fit CLI exit1 | `original-CPU-comparison-failure.json`, `fit-comparison.json.gz` |
| New environment software |463 CUDA Python cases, no failures/errors/skips;49 compatible packages,15+10 frozen identities verified | `software-verification.json`, `software-tests.xml.gz` |
| HF artifact delivery | All49 files anonymously re-downloaded and hash-verified; old branches and frozen payloads unchanged | `hf-publication.json` |
| Pulled model loading | Downloaded original passes frozen worker check; downloaded GPU comparison graph executes in existing general Runtime | `HF-pulled-runtime-proof.json` |

HF dataset: `justinruaya123/ai231-smart-speaker`, additive branch `optionb-tiny-reproduction-v1`, immutable commit `009b87cb6cb2297073797b8c9571925f70fe92dd`, prefix `reproduction/optionb-tiny-v1/`. Original epoch44 assets are under `frozen/`; non-promoted epoch28 comparison assets are under `gpu-rebuild/`. Source and metadata provenance are bound separately.

The comparison bundle deliberately retains frozen `.50 posterior/.10 margin`; the freshly selected `.55/.30` operating point is recorded separately and not applied. The epoch44-specific preview adapter correctly rejects the GPU comparison graph. No tolerances, frozen expectations or guards were weakened, no selection success lock was written, and the original frozen best model remains unchanged. GPU repeatability does not close original CPU-result equality, dependent original-audio/native/PC acceptance, fresh acoustic qualification or portability to another accelerator.

No held-out PCM inference, new native replay/timing, physical Pi, live microphone/provider operation, installed-default change, replacement or action/qualification enablement is claimed. The downloaded-model forward probe is a software fixture, not recognition-quality evidence. Historical first-run and failed baseline/review receipts remain unchanged.
