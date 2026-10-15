"""Render actual completed grid histories, with no inference or metric fitting."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/me2-vcm-training-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--root', required=True, type=Path)
parser.add_argument('--training', required=True, type=Path)
parser.add_argument('--inputs', required=True, type=Path)
parser.add_argument('--output', required=True, type=Path)
args = parser.parse_args()
sys.path.insert(0, str(args.root.resolve()))
from scripts.test.vcm_mixed_finalize import GRID, verify_training

context = verify_training(args.training, args.inputs)
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=False)
figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
colors = {53: '#2878b5', 71: '#c75b12', 89: '#7754a5'}
histories = {}
for selected, (arm, seed) in zip(context['grid'], GRID):
    path = context['training'] / f'{arm}-{seed}' / 'history.json'
    history = json.loads(path.read_bytes())
    histories[f'{arm}-{seed}'] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    epochs = [row['epoch'] for row in history]
    fields = ([row['loss'] for row in history],
              [100 * row['human_development']['macro_person_recall'] for row in history],
              [100 * row['human_development']['all']['recall'] for row in history],
              [-row['selection_key'][1] for row in history])
    style = '--' if arm == 'uniform' else '-'
    for axis, values in zip(axes.flat, fields):
        axis.plot(epochs, values, style, color=colors[seed], linewidth=1.5, label=f'{arm} / {seed}')
        index = selected['epoch'] - 1
        axis.scatter([epochs[index]], [values[index]], color=colors[seed], s=26, zorder=3)

labels = [('Training loss', 'Observed joint loss'),
          ('Human development: macro declared-person recall', 'Accepted exact commands (%)'),
          ('Human development: micro recall', 'Accepted exact commands (%)'),
          ('Human development: accepted errors', 'Wrong commands + negative false accepts')]
for axis, (title, ylabel) in zip(axes.flat, labels):
    axis.set(title=title, xlabel='Epoch (96 optimizer updates each)', ylabel=ylabel, xlim=(1, 30))
    axis.grid(alpha=.2)
for axis in (axes[0, 1], axes[1, 0]):
    axis.set_ylim(0, 100)
axes[0, 0].legend(fontsize=8, ncol=2)
figure.suptitle('VCM mixed final v2: all six completed A100 candidates\nDots show each development-selected epoch; these curves are not final benchmark accuracy', fontsize=13)
for extension in ('png', 'svg', 'pdf'):
    figure.savefig(output / ('training-curves.' + extension), dpi=180)
plt.close(figure)
shutil.copyfile(Path(__file__), output / 'plot-source.py')
files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in output.iterdir() if path.is_file()}
receipt = {'format': 'vcm-actual-learning-curves-v1', 'selected': context['selected'],
           'grid': context['grid'], 'histories': histories, 'files': files,
           'inference_performed': False, 'calibration_fitted': False, 'test_results_used': False,
           'actual_updates': context['grid_updates'], 'fixed_human_development_only_selection': True}
with (output / 'curves-receipt.json').open('x') as stream:
    json.dump(receipt, stream, indent=2, allow_nan=False)
    stream.write('\n')
print(json.dumps({'output': str(output), 'files': files, 'inference_performed': False}))
