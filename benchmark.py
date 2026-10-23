"""Action-free replay: frozen epoch44 by default; --model-choice legacy is explicit."""
import argparse
import os
import sys

for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_key, '1')

from scripts.iris_epoch44_benchmark import benchmark


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    selection = argparse.ArgumentParser(add_help=False)
    selection.add_argument('--model-choice', choices=['epoch44', 'legacy'], default='epoch44')
    selected, remaining = selection.parse_known_args(args)
    if selected.model_choice == 'legacy':
        from scripts.runtime.iris_pi_benchmark import main as replay
        return replay(remaining)
    from scripts.iris_epoch44_benchmark import main as replay
    return replay(remaining)


if __name__ == '__main__':
    main()
