"""Convert a recording once for the inspector; never resample separate chunks."""
from __future__ import annotations

import argparse
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf


def convert(source: Path, destination: Path, channel: int = 0) -> None:
    pcm, rate = sf.read(source, dtype="float32", always_2d=True)
    if not 0 <= channel < pcm.shape[1]:
        raise ValueError("Requested channel is absent")
    if source.resolve() == destination.resolve():
        raise ValueError("Do not overwrite the source recording")
    mono = pcm[:, channel]
    if not np.isfinite(mono).all():
        raise ValueError("Nonfinite PCM")
    if len(mono) / rate > 7.8:
        raise ValueError("Maximum input is 7.8 seconds; do not silently truncate")
    divisor = gcd(rate, 16000)
    if rate != 16000:
        mono = resample_poly(mono, 16000 // divisor, rate // divisor)
    if destination.exists():
        raise FileExistsError(destination)
    sf.write(destination, mono, 16000, subtype="FLOAT")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--channel", type=int, default=0)
    args = parser.parse_args()
    convert(args.source, args.destination, args.channel)
