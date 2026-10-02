"""Isolated synthetic/mixed fixed-command experiment CLI."""
import os
import sys
from pathlib import Path
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train
train.register_sources()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["prepare", "generate", "train", "baseline", "evaluate", "check", "report", "predict"])
    parser.add_argument("--source", choices=["synthetic", "human", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--mode", choices=["synthetic", "mixed"], default="mixed")
    args = parser.parse_args()
    if args.command == "prepare":
        from tinyvcm.synthetic_data import prepare
        prepare()
    elif args.command == "generate":
        from tinyvcm.synthetic_tts import generate
        generate(args.batch_size, args.limit)
    elif args.command == "train":
        from tinyvcm.synthetic_training import fit
        for source in (["synthetic", "human"] if args.source == "both" else [args.source]): fit(source, args.epochs)
    elif args.command == "predict":
        if args.audio is None: parser.error("predict requires --audio")
        from tinyvcm.synthetic_eval import predict_file
        predict_file(args.audio, args.mode)
    else:
        from tinyvcm.synthetic_eval import main
        main(args.command)
