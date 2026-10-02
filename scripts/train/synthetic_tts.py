"""Generate original designed voices and whole utterances, never splice words."""
import gc
import importlib.metadata
import json
import time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from .audio import resample_audio
from .synthetic_data import ROOT, DATA, SYN, rows, dump, digest, write_rows

ANCHOR = "Hello, I am ready to help. Please speak clearly and tell me what you would like to do today."


def generate(batch_size=16, limit=0):
    from qwen_tts import Qwen3TTSModel
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    if not torch.cuda.is_available(): raise RuntimeError("This experiment requires the requested CUDA device")
    voice_list = json.loads((DATA / "voices.json").read_text())
    model_root = ROOT / ".cache/tts/models"
    anchors = DATA / "voices"; anchors.mkdir(parents=True, exist_ok=True)
    def load(variant):
        folder = model_root / ("Qwen3-TTS-12Hz-1.7B-" + variant)
        if not (folder / "model.safetensors").exists(): raise RuntimeError("Finish Hugging Face acquisition first: " + variant)
        start = time.perf_counter()
        model = Qwen3TTSModel.from_pretrained(str(folder), device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True)
        print("MODEL_LOADED", variant, round(time.perf_counter()-start, 2), flush=True)
        return model
    missing = [v for v in voice_list if not (anchors / (v["id"] + ".wav")).exists()]
    if missing:
        model = load("VoiceDesign")
        for voice in missing:
            torch.manual_seed(voice["seed"])
            start = time.perf_counter()
            waveforms, rate = model.generate_voice_design(text=ANCHOR, instruct=voice["prompt"], language="English", max_new_tokens=384, do_sample=True, temperature=.8)
            path = anchors / (voice["id"] + ".wav")
            sf.write(path, waveforms[0], rate, subtype="PCM_16")
            dump(anchors / (voice["id"] + ".json"), {**voice, "text": ANCHOR, "audio_sha256": digest(path), "rate": rate,
                "duration": len(waveforms[0])/rate, "generation_seconds": time.perf_counter()-start})
            print("VOICE", voice["id"], round(time.perf_counter()-start, 2), flush=True)
        del model; gc.collect(); torch.cuda.empty_cache()
    model = load("Base")
    prompts = {v["id"]: model.create_voice_clone_prompt(ref_audio=str(anchors / (v["id"]+".wav")), ref_text=ANCHOR)[0] for v in voice_list}
    planned = rows(DATA / "planned.jsonl")
    existing = {r["id"]: r for r in rows(DATA / "manifest.jsonl")} if (DATA / "manifest.jsonl").exists() else {}
    for record in existing.values():
        if digest(ROOT / record["audio"]) != record["audio_sha256"]: raise ValueError("Previously generated audio changed")
    log = SYN / "tts-batches.jsonl"
    count = 0; started = time.perf_counter()
    # Stable original batching: resume replays a partially complete batch with its original seed.
    for offset in range(0, len(planned), batch_size):
        if (ROOT / ".cache/synthetic-v1/pause-generation").exists():
            print("Paused at completed batch boundary", flush=True)
            break
        batch = planned[offset:offset+batch_size]
        if all(r["id"] in existing for r in batch): continue
        if limit and count >= limit: break
        seed = batch[0]["seed"]
        # Release shape-dependent codec workspace between batches on Windows WDDM.
        torch.cuda.empty_cache()
        torch.manual_seed(seed); torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        waveforms, rate = model.generate_voice_clone(text=[r["transcript"] for r in batch], language=["English"]*len(batch),
            voice_clone_prompt=[prompts[r["voice_id"]] for r in batch], non_streaming_mode=True,
            max_new_tokens=160, do_sample=True, temperature=.75, top_p=.95)
        elapsed = time.perf_counter()-start
        for r, waveform in zip(batch, waveforms):
            if r["id"] in existing: continue
            pcm = resample_audio(waveform, rate)
            if not np.isfinite(pcm).all() or not 3200 <= len(pcm) <= 160000:
                raise ValueError("Invalid generated waveform: " + r["id"])
            path = ROOT / r["audio"]; path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, pcm, 16000, subtype="PCM_16")
            item = {**r, "audio_sha256": digest(path), "duration": len(pcm)/16000, "sample_rate":16000, "channels":1,
                    "seed":seed, "generation_batch_ids":[q["id"] for q in batch], "batch_size":batch_size,
                    "tts_model":"Qwen/Qwen3-TTS-12Hz-1.7B-Base", "voice_anchor_sha256":digest(anchors/(r["voice_id"]+".wav"))}
            existing[item["id"]] = item
            with (DATA / "manifest.jsonl").open("a", encoding="utf8") as stream: stream.write(json.dumps(item)+"\n")
            count += 1
        metrics = {"offset":offset,"n":len(batch),"seconds":elapsed,"cuda_peak_allocated_bytes":torch.cuda.max_memory_allocated(),
                   "cuda_peak_reserved_bytes":torch.cuda.max_memory_reserved(),"audio_seconds":sum(len(w)/rate for w in waveforms)}
        with log.open("a",encoding="utf8") as stream: stream.write(json.dumps(metrics)+"\n")
        print("GENERATED", len(existing), "/",len(planned),"batch_seconds",round(elapsed,2),"peak_GiB",round(metrics["cuda_peak_reserved_bytes"]/1024**3,2),flush=True)
    receipt = {"complete":len(existing)==len(planned),"planned":len(planned),"generated":len(existing),"manifest_sha256":digest(DATA/"manifest.jsonl"),
        "this_run_seconds":time.perf_counter()-started,"device":torch.cuda.get_device_name(),"batch_size":batch_size,
        "tts_package":importlib.metadata.version("qwen-tts"),"torch":torch.__version__,"attention":"sdpa","dtype":"bfloat16",
        "accent_human_verified":False,"tts_text_acoustically_adjudicated":False,"no_splicing":True}
    dump(SYN / "tts-generation.json",receipt)
    print(json.dumps(receipt),flush=True)
