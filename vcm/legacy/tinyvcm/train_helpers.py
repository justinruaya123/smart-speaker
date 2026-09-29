from concurrent.futures import ThreadPoolExecutor
import math
import random
import time
import numpy as np
import torch
from einops import rearrange, reduce
from .audio import DEFAULT, cached_features
from .data import ROOT
_FEATURE_MEMORY = {}
def training_features(record):
    value = _FEATURE_MEMORY.get(record["id"])
    return value if value is not None else cached_features(record, ROOT)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def warm_cache(rows, workers=6, memory_mib=0):
    started = last = time.monotonic()
    memory_bytes = sum(array.nbytes for array in _FEATURE_MEMORY.values()) if memory_mib else 0
    def cache(record):
        array = training_features(record)
        return array if memory_mib else array.shape[1]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, value in enumerate(pool.map(cache, rows), 1):
            frames = value.shape[1] if memory_mib else value
            rows[i - 1]["frames"] = frames
            if memory_mib and rows[i - 1]["id"] not in _FEATURE_MEMORY:
                memory_bytes += value.nbytes
                if memory_bytes > memory_mib * 2**20:
                    raise RuntimeError(f"Feature cache exceeds the requested {memory_mib} MiB budget")
                value.flags.writeable = False
                _FEATURE_MEMORY[rows[i - 1]["id"]] = value
            if time.monotonic() - last > 20:
                print(f"FEATURES {i}/{len(rows)} elapsed={time.monotonic()-started:.0f}s", flush=True)
                last = time.monotonic()
    if memory_mib:
        print(f"RAM FEATURE CACHE {len(_FEATURE_MEMORY)} arrays, {memory_bytes/2**20:.1f} MiB", flush=True)


def normalization(rows):
    if not rows or any(row.get("split") != "train" for row in rows):
        raise ValueError("Normalization fitting requires nonempty training-only records")
    total = np.zeros(40, dtype=np.float64)
    squares = np.zeros(40, dtype=np.float64)
    count = 0
    for row in rows:
        x = training_features(row).astype(np.float64)
        total += reduce(x, "m t -> m", "sum")
        squares += reduce(x * x, "m t -> m", "sum")
        count += x.shape[1]
    mean = total / count
    std = np.sqrt(np.maximum(squares / count - mean * mean, 1e-4))
    return {"mean": mean.tolist(), "std": std.tolist(), "training_frames": count}


def batches(rows, batch_size, rng=None, frame_budget=16000, indices=None):
    indices = list(range(len(rows))) if indices is None else list(indices)
    if rng is not None:
        rng.shuffle(indices)
    output = []
    for offset in range(0, len(indices), batch_size * 32):
        bucket = sorted(indices[offset:offset + batch_size * 32], key=lambda i: rows[i].get("frames", rows[i]["duration"] * 100))
        current, maximum = [], 0
        for index in bucket:
            length = rows[index].get("frames", math.ceil(rows[index]["duration"] * 100))
            if current and (len(current) >= batch_size or max(maximum, length) * (len(current) + 1) > frame_budget):
                output.append(current)
                current, maximum = [], 0
            current.append(index)
            maximum = max(maximum, length)
        if current:
            output.append(current)
    if rng is not None:
        rng.shuffle(output)
    return output


def make_batch(rows, indices, stats, augment=False, rng=None, speed_augment=False, frequency_warp=False, gain_augment_db=0.):
    arrays = [training_features(rows[i]) for i in indices]
    if augment and speed_augment:
        from scipy.ndimage import zoom
        # Feature-domain temporal resampling is explicitly distinct from waveform augmentation.
        arrays = [zoom(x, (1, rng.uniform(.9, 1.1)), order=1, prefilter=False) for x in arrays]
    if augment and frequency_warp:
        from scipy.ndimage import affine_transform
        warped = []
        for x in arrays:
            scale = rng.uniform(.9, 1.1)
            warped.append(affine_transform(x, [[scale, 0], [0, 1]], offset=[(1-scale)*19.5, 0],
                                           order=1, mode="nearest", prefilter=False))
        arrays = warped
    lengths = torch.tensor([x.shape[1] for x in arrays], dtype=torch.long)
    maximum = int(math.ceil(int(lengths.max()) / 32) * 32)
    batch = np.zeros((len(arrays), 40, maximum), dtype=np.float32)
    mean = rearrange(np.array(stats["mean"], dtype=np.float32), "m -> m 1")
    std = rearrange(np.array(stats["std"], dtype=np.float32), "m -> m 1")
    for j, x in enumerate(arrays):
        if augment and gain_augment_db:
            # Power-log features: 10 dB is ln(10), with the same frontend floor.
            # Training-only augmentation; the deployment frontend is unchanged.
            x = np.maximum(x + rng.uniform(-gain_augment_db, gain_augment_db) * np.log(10) / 10,
                           np.log(DEFAULT.log_floor))
        x = (x - mean) / std
        if augment:
            x = x.copy()
            # Feature-space augmentation is explicitly reported; no synthetic room claim.
            x += rng.uniform(-0.15, 0.15)
            width = rng.randrange(0, 6)
            begin = rng.randrange(0, 40 - width + 1)
            x[begin:begin + width, :] = 0
            width = rng.randrange(0, min(25, x.shape[1] // 10) + 1)
            begin = rng.randrange(0, x.shape[1] - width + 1)
            x[:, begin:begin + width] = 0
        batch[j, :, :x.shape[1]] = x
    return torch.from_numpy(batch), lengths

from .text import normalize_text
from .validation import edit_distance

