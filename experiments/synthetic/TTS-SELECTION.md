# TTS choice and evidence

The selected generator is the open Qwen3-TTS 1.7B family from Hugging Face: VoiceDesign creates original voice anchors, and Base reuses those anchors to synthesize complete English commands. Both model repositories declare Apache-2.0. The exact immutable revisions, downloaded file hashes and parameter metadata are in `tts-acquisition.json`.

| Candidate | Relevant capability | Decision for this experiment |
|---|---|---|
| [Qwen3-TTS 1.7B](https://github.com/QwenLM/Qwen3-TTS) | English, natural-language voice design, reusable voice-clone prompts, batched generation | Acquired and executed on the local RTX 4080 SUPER. The official API explicitly documents design-then-clone. |
| [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) | 2B parameters, English and Tagalog among 30 supported languages, voice design and controllable cloning | Relevant alternative for a future accent comparison. Not downloaded or benchmarked in this run. Tagalog support by itself does not prove a Filipino accent when speaking English. |
| [MegaTTS3](https://huggingface.co/ByteDance/MegaTTS3) | English synthesis and reference-accent control | Its card says the reference encoder weights are withheld and inference requires supplied reference latents. This adds an external dependency for creating new voices. Not run. |

The Qwen card documents ten languages, including English, but does not publish a Filipino-English accent benchmark. Six of the twelve voice descriptions request Philippine English, and all command text remains English. Accent metadata is deliberately `requested`, not `verified`, until a listener reviews it. TTS transcript labels describe the intended text; acoustic transcription fidelity has not been independently adjudicated.

Twelve voices are split before generation: eight training, two development, and two test anchors. Each split includes ordinary English and Filipino-English prompts. Ages, pitch, gender, speaking rate, and accent descriptions vary. These are designed synthetic voices, not clones of identifiable people. Original anchor generation took approximately 9-11 seconds per voice on this machine.

Generation uses BF16 and PyTorch SDPA in the existing root environment. Both models are loaded sequentially. The 32-item pilot batch used excessive allocator reservation on Windows WDDM, so the measured working recipe uses 16 items with cache release between batches. `tts-batches.jsonl` records active and reserved GPU memory separately; active memory is not the same as total application or system memory. No universal VRAM requirement is inferred from parameter bytes alone.

SoX and FlashAttention are not installed. The package emits optional-dependency notices; the selected SDPA/soundfile/resampling path ran successfully without them. The generator runs offline after acquisition. No remote generation API is used.

This explicit synthetic experiment does not change the original human-only brightness, parameter or phonetic studies. It does not install a new production reader or alter the selected VCM checkpoint.
