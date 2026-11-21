# BMO documentation on GitHub Pages

The primary site is the four-slide BMO deck: architecture, dataset, RTX 4080 training, and VCM evidence. It uses SpeakerUI Ambient colors/system typography and code-rendered Three.js geometry. There are no AI-generated images or runtime CDN dependencies. The previous six-chapter presentation is preserved as [reference.html](site/reference.html); its original model, assets and receipts remain historical evidence.

The Pages project URL is `https://justinruaya123.github.io/smart-speaker/`. That URL is a deployment target, not proof that a particular revision is live. Check the successful deployment and public page before claiming publication.

## Serve and present

From the repository root:

```sh
python -m http.server 8770 --bind 127.0.0.1 --directory docs/site
```

Open `http://127.0.0.1:8770/`. The viewer needs only a modern browser, JavaScript and WebGL for the 3D scene; the stage explanations remain available if WebGL cannot initialize. Serve over HTTP rather than opening `index.html` as a file. All imports, audio, fetches and evidence links resolve relative to the site/module, so deployment below `/smart-speaker/` works without a domain-specific base URL or backend.

Use the footer navigation or arrow keys for slides, F for fullscreen and Presenter notes for narration. On architecture, Play/Step follows the recorded forward pass; Inside a block expands the two depthwise/pointwise convolution units and residual addition. Drag rotates, wheel zooms, canvas arrow keys rotate and Home resets. The log-mel SVG displays the same synthetic recording: 16 × 48 measured display-bin means over the original 40 × 296 grid. Its cursor follows actual audio playback. The sample is an already-published synthetic recording, not microphone input. Playback speed is explanatory, not measured model latency.

The Dataset slide links directly to the Hugging Face repository. Its panels show the frozen release, all 102 wordings, generation recipes, and explicit TTS model/provider citations. Pinned Qwen assets remain distinct from the unverified CosyVoice2 acquisition revision. The distillation section cites BMO's supervised cross-entropy recipe; no teacher checkpoint is attributed to this fit. The training scrubber shows all 48 measured RTX 4080 SUPER epochs. The Dataset slide includes a separate Pi 5 recorded-replay strip and links the full historical and shared-catalog scores. The Evidence slide exposes correct/rejected/wrong counts, the comparable-size development control, historical CPU timing and the measured Pi 5 processing/RTF/process-RSS results. Pi 4, power and live-microphone/wake/owner acceptance remain unmeasured; the original study's null placeholders are preserved.

## Source-backed content

- [Dataset generation and citations](site/bmo/data/dataset-recipe.json): frozen Hugging Face revision, original sources, exact row accounting, Edge/System.Speech prompts and voices, CosyVoice2 provenance, DEV-only Qwen asset pins and missing settings.
- [Physical Pi 5 recorded replay](site/bmo/data/pi5-replay.json): two sequential actual-host repeats, each with 196 original recordings × three passes per configuration. Full frozen scoring (194 eligible references), a separately labeled shared-catalog diagnostic (167), raw/accepted error counts, first-pass decisions, all timing samples, audio/PCM hashes and source fingerprints. Baseline A and historical epoch44 have different inventories; 27 full-set references are outside A's catalog. This is not live speech or qualification. Source checkout changes are disclosed, not published as a clean-commit reproduction.
- [VCM benchmark](site/bmo/data/benchmark.json): ten cohort groups, nine finite slot cells, matched-seed development comparison and timing scope. Scores belong to the published epoch-38 graph; the RTX 4080 epoch-36 candidate has no test evaluation.
- [Recorded forward pass](site/bmo/data/trace.json) and [provenance](site/bmo/data/provenance.json): all 37 logits match the unmodified published model exactly. The original graph, runtime and installed bundle were not changed.
- [Measured training history](site/bmo/data/training-history.json): the complete 48-epoch run, source hash and metric definitions. Exact original-weight comparison failed; this remains visible.
- [Canonical wordings](site/bmo/wordings.csv): 34 typed commands, three wordings each.

Source-specific dataset terms apply. The upstream composite DOI `10.57967/hf/10723` is not a DOI for this derivative. Scholarly citations, model IDs, source notices and original links are available in the Dataset citations and Generation recipe panels. Missing historical seeds, generator revisions, independent-speaker counts and exact A audio hours remain unknown. Restricted third-party audio is not mirrored; the page includes one already-published authored synthetic WAV with its existing source notice.

## Validation and deployment policy

```sh
python -B presentation/bmo_validate.py
python -B presentation/validate.py --require-integration
python -B -m unittest scripts.test.test_bmo_pages scripts.test.test_iris_publication_data -v
```

After intentionally changing site assets, update both inventories in order:

```sh
python -B presentation/bmo_validate.py --write-manifest
python -B presentation/validate.py --write-manifest --require-integration
```

Browser CI uses the existing pinned presentation dependency lock. `node presentation/bmo_qa.mjs` serves a strict `/smart-speaker/` prefix, blocks external viewer requests and checks interactions, layouts, recipe/citation panels, benchmark and reduced-motion startup. `node presentation/qa.mjs` continues to exercise the preserved historical deck. No scientific dependencies are installed by these checks.

The existing `.github/workflows/iris-publication.yml` workflow now publishes BMO documentation on a validated push to main. Its deployment needs static/provenance/browser validation and the standard GitHub Pages environment. Pages must use GitHub Actions as its source. PR runs do not deploy. Official workflow permissions and artifact requirements are documented by [GitHub](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).

This implementation deliberately separates documentation publication from acoustic model qualification. The independent `acoustic-release` job, its receipt and nonzero failure behavior remain unchanged. Publishing an evidence page does not promote a model, enable device actions, establish fresh human accuracy or qualify Raspberry Pi hardware. The earlier epoch44 policy that coupled static Pages deployment to acoustic acceptance is explicitly superseded for documentation, as recorded in the preserved presentation guide and citation document.

Exact-byte Git attributes protect site assets, frozen presentation evidence and the existing rendered-scene source across Windows/Linux checkouts. The first integration commit changes only the exact public-sample `.gitignore` exception; corpus audio and private captures remain ignored.
