"""Regenerate submission figures from hash-bound, portable training evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .common import ROOT, read, sha


START = '<!-- final-submission:start -->'
END = '<!-- final-submission:end -->'
ENCODER = 'Complete-utterance contextual encoder after VAD endpoint'
PRIMARY = '2m'
ORDER = ('2m', '935k')


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8')


def _copy_bound(source, destination, expected):
    data = source.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('Original evidence identity changed: '+str(source))
    if destination.exists() and destination.read_bytes() != data:
        raise ValueError('Preserve changed public evidence: '+str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        with destination.open('xb') as stream:
            stream.write(data)


def import_local_evidence(root=ROOT):
    """Copy metric-only histories verbatim; retain both exact recovery states."""
    root = Path(root)
    audit = read(root/'docs/model-dataset-usage.json')
    records = []
    for model, evidence in audit['models'].items():
        for record in evidence['histories']:
            phase = Path(record['path']).parent.name
            path = f'submission/training/{model}-{phase}.json'
            _copy_bound(root/record['path'], root/path, record['sha256'])
            records.append(dict(model=model, phase=phase, path=path,
                                original_path=record['path'], sha256=record['sha256'], epochs=record['epochs']))
    checkpoints = {}
    for model, filename in [('935k','compact-selected.pt'), ('2m','full-selected.pt')]:
        selected = audit['models'][model]['checkpoint']
        checkpoint = 'submission/checkpoints/'+filename
        _copy_bound(root/selected['path'], root/checkpoint, selected['sha256'])
        checkpoints[model] = dict(path=checkpoint, original_path=selected['path'], sha256=selected['sha256'])
    manifest = dict(format='submission-training-evidence-v1', histories=records,
                    primary_model=PRIMARY, checkpoint=checkpoints[PRIMARY], checkpoints=checkpoints,
                    history_copy='Verbatim JSON with training/development metrics; no raw fitting logs or audio.',
                    selected_compact='Human epoch 8 -> mixed-control epoch 28; repair performs zero additional fits.',
                    selected_comparison='Human epoch 20 -> mixed-control epoch 32; completion arm is retained as comparison evidence.')
    destination = root/'submission/training/manifest.json'
    if destination.exists() and read(destination)['histories'] != records:
        raise ValueError('Preserve changed training history evidence')
    _json(destination, manifest)
    return manifest


def load_histories(root=ROOT):
    root = Path(root)
    manifest = read(root/'submission/training/manifest.json')
    histories = {}
    for record in manifest['histories']:
        path = root/record['path']
        if sha(path) != record['sha256']:
            raise ValueError('Public history identity mismatch: '+record['path'])
        rows = read(path)
        if len(rows) != record['epochs'] or [r['epoch'] for r in rows] != list(range(1, len(rows)+1)):
            raise ValueError('Incomplete public training history: '+record['path'])
        histories[(record['model'], record['phase'])] = rows
    return histories


def _metrics(group, raw_exact):
    return dict(positive_n=group['positive_n'], negative_n=group['negative_n'],
                raw_complete_correct=raw_exact, correct_accepts=group['exact'],
                wrong_accepts=group['accepted']-group['exact']-group['negative_errors'],
                unknown_false_accepts=group['negative_errors'], precision=group['precision'],
                complete_recall=group['recall'])


def build_summary(root=ROOT):
    root = Path(root)
    audit_path = root/'docs/model-dataset-usage.json'
    audit = read(audit_path)
    summary = dict(format='submission-report-v1', primary_model=PRIMARY, encoder=ENCODER,
                   input='16 kHz mono; 40 x T log-mel; 25 ms window / 10 ms hop',
                   outputs='40 acoustic outputs; 19 intent outputs; 2 target outputs; finite typed catalog decoding',
                   physical_pi_measured=False, live_microphone_measured=False,
                   pi_measurements=dict(wake_accuracy=None, intent_accuracy=None, unknown_false_accept_rate=None,
                                        latency_p95_ms=None, real_time_factor=None, memory_mib=None, thermal_behavior=None),
                   models={}, evidence={str(audit_path.relative_to(root)):sha(audit_path)})
    for name in ORDER:
        evidence = audit['models'][name]
        source = evidence['source_results']
        results = read(root/source)
        summary['evidence'][source] = sha(root/source)
        expected = audit['evidence_files'].get(source)
        if expected and expected != sha(root/source):
            raise ValueError('Result identity changed: '+source)
        item = {key:evidence[key] for key in ('experiment', 'parameters', 'architecture', 'selected_mixed_epoch',
                 'selected_human_foundation_epoch', 'fitting_seconds_excluding_input_loading', 'gpu',
                 'gpu_count_used', 'training', 'export', 'qualified')}
        item['dataset'] = audit['views'][evidence['fitting_view']]
        item['selected_loss_components'] = evidence['selected_mixed_epoch_loss_components']
        folder = root/'submission/models'/('full' if name == '2m' else 'compact')
        if folder.is_dir():
            files = [p for p in folder.rglob('*') if p.is_file()]
            item['deployment_assets'] = dict(
                graph_bytes=(folder/'model.onnx').stat().st_size,
                wake_vad_bytes=sum(p.stat().st_size for p in files if 'wake' in p.relative_to(folder).parts),
                total_bundle_bytes=sum(p.stat().st_size for p in files),
                scope='Selected graph, wake/VAD graphs and metadata; excludes Python/Node/UI dependencies.')
        item['demo_policy'] = 'frozen per-intent compact recall policy' if name == '935k' else 'fixed .95 intent / .90 slot'
        item['test'] = {}; item['holdout'] = {}
        for split in ('test', 'holdout'):
            for kind in ('human', 'synthetic'):
                group = results[split]['relaxed_demo'][kind]['overall'] if name == '935k' else results[split]['application'][kind]
                raw = results[split]['raw'][kind]['overall']['exact'] if name == '935k' else group['raw_exact']
                item[split][kind] = _metrics(group, raw)
        summary['models'][name] = item
    official_path = root/'experiments/speaker-official-vcm-benchmark-v1/results.json'
    if official_path.exists():
        official = read(official_path)
        summary['evidence'][str(official_path.relative_to(root))] = sha(official_path)
        summary['official_recorded_holdout'] = {}
        for model, key in [('2m','full'), ('935k','compact')]:
            data = official['models'][key]
            policy = data['policies']['demo']
            summary['official_recorded_holdout'][model] = dict(
                rows=policy['official_command']['n'], command_accuracy=policy['official_command']['accuracy'],
                intent_accuracy=policy['official_intent']['accuracy'], command_macro_f1=policy['official_command']['macro_f1'],
                unknown_false_accepts=policy['official_command']['false_accepts'],
                unknown_n=policy['official_command']['n_out_of_scope'])
    release_path=root/'submission/release/inference.json'
    if release_path.exists():
        release=read(release_path)
        summary['inference_release']={key:release[key] for key in ('path','sha256','bytes','entrypoint')}
        summary['evidence'][str(release_path.relative_to(root))]=sha(release_path)
        downloads_path=root/'submission/release/downloads.json'
        if downloads_path.exists():
            downloads=read(downloads_path)
            summary['evidence'][str(downloads_path.relative_to(root))]=sha(downloads_path)
            for asset in downloads.get('assets',[]):
                if asset['kind']=='inference' and asset['sha256']==release['sha256']:
                    summary['inference_release']['url']=asset['url']
    reproduction_path=root/'submission/reproduction-full/verification.json'
    if reproduction_path.exists():
        summary['reproduction']=read(reproduction_path)
        summary['evidence'][str(reproduction_path.relative_to(root))]=sha(reproduction_path)
    return summary


def _save_figure(fig, output, name):
    for suffix in ('svg', 'png'):
        fig.savefig(output/(name+'.'+suffix), dpi=180, facecolor=fig.get_facecolor(),bbox_inches='tight',
                    metadata={'Creator':'scripts.submission.report', 'Date':None} if suffix == 'svg'
                    else {'Software':'scripts.submission.report'})


def attach_benchmark(summary, path):
    """Keep recorded desktop timing separate; fill Pi cells only from named Pi evidence."""
    path = Path(path); result = read(path)
    hardware = result.get('hardware', {})
    source = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
    summary['benchmark_receipt'] = dict(path=source, sha256=sha(path), hardware=hardware,
                                        mode=result.get('mode'), timing_excludes=result.get('timing_excludes', []))
    summary['evidence'][source] = sha(path)
    if not hardware.get('physical_pi_measured'):
        summary['desktop_benchmark'] = result
        return summary
    if (hardware.get('machine','').lower() not in ('aarch64','arm64','armv7l')
            or 'raspberry pi' not in hardware.get('model','').lower()):
        raise ValueError('Pi benchmark hardware identity is inconsistent')
    binding = result.get('model_binding', {}).get('model_sha256')
    if binding != summary['models'][summary['primary_model']]['export']['sha256']:
        raise ValueError('Pi benchmark does not match the selected primary graph')
    summary['physical_pi_measured'] = True
    measurements = summary['pi_measurements']
    measurements.update(latency_p95_ms=result['latency_ms']['p95'],
                        real_time_factor=result['rtf']['p95'], memory_mib=result.get('peak_rss_mib'))
    measurements['recorded_policies'] = result.get('policies', {})
    # Silent recorded replay contains no wake input; it cannot fill wake accuracy.
    measurements['intent_metric'] = 'accepted-intent recall per policy/kind; denominator is positive_n'
    measurements['thermal_behavior'] = result.get('thermal_summary')
    summary['pi_benchmark'] = result
    return summary


def render(root=ROOT, output=None, benchmark=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    root = Path(root); output = Path(output or root/'submission/figures')
    output.mkdir(parents=True, exist_ok=True)
    histories = load_histories(root)
    summary = build_summary(root)
    if benchmark is not None:
        attach_benchmark(summary, benchmark)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'svg.fonttype':'none', 'svg.hashsalt':'smart-speaker-submission-v1', 'font.size':10,
                         'axes.spines.top':False, 'axes.spines.right':False})
    colors = {'935k':'#147D92', '2m':'#8057A2'}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    fig.suptitle('Training and development histories | selected saved weights, no new fitting', fontsize=16, weight='bold')
    for row, name in enumerate(ORDER):
        item = summary['models'][name]
        for column, phase in enumerate(('human-shared', 'mixed-control')):
            data = histories[(name, phase)]; ax = axes[row,column]
            epochs = [x['epoch'] for x in data]
            objective = [x['losses'][0]+.30*x['losses'][1]+.15*x['losses'][2] for x in data]
            ax.plot(epochs, objective, color=colors[name], label='CTC + .30 intent CE + .15 target CE', lw=2)
            ax.plot(epochs, [x['losses'][0] for x in data], color=colors[name], ls=':', label='CTC component', alpha=.65)
            selected = item['selected_human_foundation_epoch'] if column == 0 else item['selected_mixed_epoch']
            ax.axvline(selected, color='#D07D25', ls='--', label=f'Selected epoch {selected}')
            ax.set(title=f'{name} | {phase}', xlabel='Epoch within phase', ylabel='Mean training loss')
            ax.grid(alpha=.15); ax.legend(fontsize=8)
        ax = axes[row,2]
        data = [r for r in histories[(name,'mixed-control')] if 'development' in r]
        for kind, color in [('human','#147D92'), ('synthetic','#8057A2')]:
            for field, style, label in [('raw_recall','-','raw'), ('recall','--','fixed-gate accepts')]:
                ax.plot([r['epoch'] for r in data], [100*r['development'][kind][field] for r in data],
                        ls=style, color=color, marker='o', ms=3, label=f'{kind}: {label}')
        ax.axvline(item['selected_mixed_epoch'], color='#D07D25', ls='--')
        ax.set(title=f'{name} | mixed development complete commands', xlabel='Mixed epoch', ylabel='Correct / positive references (%)', ylim=(0,100))
        ax.grid(alpha=.15); ax.legend(fontsize=8, loc='lower right')
    _save_figure(fig, output, 'training-curves'); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.1))
    fig.subplots_adjust(left=.07,right=.98,top=.84,bottom=.24,wspace=.24)
    fig.suptitle('Recorded project test | complete-command correctness and false accepts', fontsize=14, weight='bold')
    for ax, kind in zip(axes, ('human','synthetic')):
        for i, name in enumerate(ORDER):
            m = summary['models'][name]['test'][kind]
            ax.bar(i-.18, 100*m['raw_complete_correct']/m['positive_n'], width=.35, color=colors[name], alpha=.4)
            ax.bar(i+.18, 100*m['correct_accepts']/m['positive_n'], width=.35, color=colors[name])
            ax.text(i+.18, 100*m['correct_accepts']/m['positive_n']+2, f"{m['correct_accepts']}/{m['positive_n']}", ha='center', fontsize=10)
        ax.set(title=kind.capitalize()+' references', xticks=[0,1], xticklabels=['2M: fixed demo policy','935k: fitted demo policy'],
               ylabel='Correct / positive references (%)', ylim=(0,109))
        ax.grid(axis='y',alpha=.15)
    fig.text(.5,.035,'Pale bars: raw candidate. Solid: accepted correct. Human UNKNOWN accepts: 2M 3/229; 935k 1/229.\n'
             'Policies, precision and six human fitting labels differ. Qualification disabled; physical Pi/live tests pending.',
             ha='center', fontsize=10)
    _save_figure(fig, output, 'recorded-comparison'); plt.close(fig)

    fig = plt.figure(figsize=(16,9), facecolor='#F7F9FC')
    ax = fig.add_axes([0,0,1,1]); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
    ax.text(.035,.95,'FINITE VOICE COMMAND MODEL', fontsize=11, color='#147D92', weight='bold')
    ax.text(.035,.90,'Model / dataset / training', fontsize=28, color='#172A3A', weight='bold')
    ax.text(.035,.855,'Primary: 2,066,831 parameters  |  selected control epoch 32  |  direct command JSON / rejection',fontsize=13,color='#405363')
    blocks = [('Capture','16 kHz mono\nHey Jarvis + Silero'), ('Frontend','40 × T log-mel\n25 ms / 10 ms'),
              ('Contextual encoder','6 blocks; width 192\n4 heads; FFN 384'), ('Finite outputs','40 acoustic; 19 intent\n2 target outputs'),
              ('Typed validation','Finite catalog scoring\nCommand / rejection')]
    for i,(title,body) in enumerate(blocks):
        x=.035+i*.192
        ax.add_patch(FancyBboxPatch((x,.67),.172,.13,boxstyle='round,pad=0.008',facecolor='white',edgecolor='#D3DDE7'))
        ax.text(x+.086,.76,title,ha='center',fontsize=12,weight='bold',color='#172A3A')
        ax.text(x+.086,.705,body,ha='center',va='center',fontsize=10,color='#405363',linespacing=1.7)
        if i<4: ax.annotate('',xy=(x+.184,.735),xytext=(x+.175,.735),arrowprops=dict(arrowstyle='->',color='#637789',lw=1.5))
    ax.text(.035,.625,'Complete-utterance contextual encoder after VAD endpoint; no attention/KV caches or unrestricted transcript.',fontsize=12,color='#405363')
    columns = [(.035,'MODEL',[
        '2M INT8 ONNX: 3.026 MiB', '6 blocks / width 192 / 4 heads / FFN 384',
        '935k comparison: 935,567 parameters', '935k FP32 ONNX: 3.686 MiB',
        '18 command intents + UNKNOWN', 'Actions off by default; qualification failed']),
        (.365,'DATASET',[
        'AI231 primary + Filipino supplement', '23,295 fitting utterances / 14.039 hours',
        '1,869 human + 21,426 synthetic', '179 declared human IDs + 70 voice IDs',
        '1,251 dev / 210 human calibration', 'Per-source terms; DOI 10.57967/hf/10723']),
        (.695,'TRAINING',[
        'One shared NVIDIA A100-SXM4-40GB', 'Seed 53; PyTorch 2.10.0+cu128',
        'AdamW; CTC + .30 intent CE + .15 target CE', '24 human + 32 control + 32 completion epochs',
        'Selected lineage: human 20 → control 32', '33.17 min study wall-clock; one A100',
        'Epoch 32 CTC / intent / target: .2313 / .02766 / .000811'])]
    for x,title,lines in columns:
        ax.text(x,.56,title,fontsize=12,weight='bold',color='#147D92')
        ax.text(x,.515,'\n'.join(lines),fontsize=10.5,color='#172A3A',va='top',linespacing=1.85)
    ax.add_patch(FancyBboxPatch((.035,.105),.93,.115,boxstyle='round,pad=0.01',facecolor='#E9EFF5',edgecolor='none'))
    ax.text(.05,.185,'RASPBERRY PI VALIDATION',fontsize=11,weight='bold',color='#172A3A')
    if summary['physical_pi_measured']:
        pi = summary['pi_measurements']; policies = pi['recorded_policies']
        human = policies.get('demo', {}).get('human', {})
        raw = policies.get('raw', {}).get('human', {})
        ax.text(.05,.145,f"Recorded human commands: raw {raw.get('complete_correct','?')}/{raw.get('positive_n','?')}; demo {human.get('complete_correct','?')}/{human.get('positive_n','?')}; "
                f"UNKNOWN accepts {human.get('unknown_false_accepts','?')}/{human.get('negative_n','?')}",fontsize=11,color='#405363')
        ax.text(.05,.115,f"Inference p95 {pi['latency_p95_ms']:.2f} ms; RTF p95 {pi['real_time_factor']:.4f}; peak RSS {pi['memory_mib']} MiB. Wake/keyword accuracy: NOT MEASURED.",fontsize=11,color='#405363')
    else:
        ax.text(.05,.14,'Wake / intent accuracy, false accepts, p95 latency, RTF, memory and thermal behavior: NOT MEASURED',fontsize=12,color='#405363')
    ax.text(.035,.055,'Evidence: docs/REPRODUCIBILITY-935k-2m.md; docs/model-dataset-usage.json; submission/training/manifest.json',fontsize=9,color='#637789')
    _save_figure(fig, output, 'model-dataset-training-card'); plt.close(fig)
    _json(output/'metrics.json', summary)
    inventory = {p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != 'manifest.json'}
    _json(output/'manifest.json', dict(generator='python -m scripts.submission.report', files=inventory,
                                     evidence=summary['evidence'], training_manifest_sha256=sha(root/'submission/training/manifest.json')))
    return summary


def attach_reproduction_benchmark(summary, root):
    """Attach measurements only when they identify this run's regenerated graph."""
    root = Path(root); path = root/'results/benchmark/metrics.json'
    if not path.exists():
        return summary
    result = read(path)
    if result.get('model_binding',{}).get('model_sha256') != summary['onnx_sha256']:
        raise ValueError('Reproduction benchmark does not match the regenerated graph')
    hardware = result.get('hardware',{})
    physical_pi = bool(hardware.get('physical_pi_measured'))
    if physical_pi and (hardware.get('machine','').lower() not in ('aarch64','arm64','armv7l')
                        or 'raspberry pi' not in hardware.get('model','').lower()):
        raise ValueError('Pi benchmark hardware identity is inconsistent')
    summary['benchmark_receipt'] = dict(path=path.relative_to(root).as_posix(),sha256=sha(path),hardware=hardware)
    summary['pi_benchmark' if physical_pi else 'desktop_benchmark'] = result
    summary['physical_pi_measured'] = physical_pi
    summary['keyword_accuracy'] = None
    summary['live_microphone_measured'] = False
    return summary


def render_reproduction(root, output=None):
    """Render an isolated new run without the original research checkout/cache."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    root = Path(root); output = Path(output or root/'results/figures')
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','svg.hashsalt':'smart-speaker-submission-v1'})
    output.mkdir(parents=True, exist_ok=True)
    comparison_path = root/'results/comparison.json'
    comparison = read(comparison_path)
    recipe_model = comparison.get('recipe_model','compact')
    recipe_folder = 'reproduction-full' if recipe_model == 'full' else 'reproduction'
    recipe_path = root/'submission'/recipe_folder/'recipe.json'
    recipe = read(recipe_path)
    histories = {phase:read(root/'.cache/reproduction/training'/phase/'history.json')
                 for phase in ('human-shared','mixed-control')}
    selection_path = root/'results/selection-before-calibration.json'
    selection = read(selection_path) if selection_path.exists() else {}
    summary = dict(comparison, encoder=ENCODER, keyword_accuracy=None,
                   comparison_sha256=sha(comparison_path), recipe_sha256=sha(recipe_path),
                   histories={phase:sha(root/'.cache/reproduction/training'/phase/'history.json') for phase in histories})
    summary['training'] = dict(gpu=selection.get('gpu'),gpu_count_used=1,torch=selection.get('torch'),
                              cuda=selection.get('cuda'),fit_development_seconds=selection.get('seconds'),
                              epochs={phase:len(rows) for phase,rows in histories.items()},
                              optimizer_updates=sum(len(rows) for rows in histories.values())*recipe['training']['steps_per_epoch'],
                              selected_human_epoch=selection.get('foundation',{}).get('epoch'),
                              selected_mixed_epoch=selection.get('recognition_epoch'))
    attach_reproduction_benchmark(summary,root)
    fig, axes = plt.subplots(1,3,figsize=(15,4.8),constrained_layout=True)
    model_name = '2M' if recipe['parameters'] > 1_000_000 else '935k'
    fig.suptitle(f"{model_name} reproduction | original seed {comparison['original_seed']} / new seed {comparison['seed']}",fontsize=16,weight='bold')
    selected = [selection.get('foundation',{}).get('epoch'), selection.get('recognition_epoch')]
    for i,(phase,rows) in enumerate(histories.items()):
        epochs = [r['epoch'] for r in rows]
        axes[i].plot(epochs,[r['losses'][0]+.30*r['losses'][1]+.15*r['losses'][2] for r in rows],color='#147D92',label='Weighted objective')
        axes[i].plot(epochs,[r['losses'][0] for r in rows],ls=':',color='#8057A2',label='CTC component')
        if selected[i]: axes[i].axvline(selected[i],ls='--',color='#D07D25',label=f'Selected epoch {selected[i]}')
        axes[i].set(title=phase,xlabel='Epoch within phase',ylabel='Mean training loss')
        axes[i].grid(alpha=.15); axes[i].legend(fontsize=8)
    original = comparison['original_human_test']; new = comparison['results']['test']['human']
    for i,key in enumerate(('raw','demo')):
        old = original['raw_exact' if key == 'raw' else 'demo_exact']
        current = new[key]['exact']
        axes[2].bar(i-.18,100*old/original['positive_n'],.35,color='#8057A2',label='Submitted original' if i==0 else None)
        axes[2].bar(i+.18,100*current/new[key]['positive_n'],.35,color='#147D92',label='Reproduction' if i==0 else None)
    axes[2].set(title='Human project test',xticks=[0,1],xticklabels=['Raw correct','Demo correct accepts'],ylabel='Correct / positives (%)',ylim=(0,100))
    axes[2].legend(fontsize=8); axes[2].grid(axis='y',alpha=.15)
    _save_figure(fig,output,'training-curves'); plt.close(fig)
    fig = plt.figure(figsize=(16,9),facecolor='#F7F9FC'); ax=fig.add_axes([0,0,1,1]);ax.axis('off')
    ax.text(.05,.91,model_name+' VCM REPRODUCTION',fontsize=25,weight='bold',color='#172A3A')
    ax.text(.05,.85,ENCODER,fontsize=17,color='#147D92')
    ax.text(.05,.79,'16 kHz PCM → 40 × T log-mel → 6 contextual blocks → finite catalog → command JSON / rejection',fontsize=14,color='#405363')
    columns = [(.05,'MODEL / DATA',[
        f"{recipe['parameters']:,} VCM parameters; width {recipe['architecture']['width']}",
        f"{len(recipe['views']['train']):,} pinned fitting utterances",
        f"{len(recipe['views']['dev']):,} dev / {len(recipe['views']['calibration']):,} human calibration",
        'Dataset PCM, source and frontend identities audited',
        f"Seed {comparison['seed']} (submitted: {comparison['original_seed']})",
        'GPU: '+str(selection.get('gpu','not recorded')),
        'AdamW; CTC + .30 intent CE + .15 target CE',
        f"{sum(len(rows) for rows in histories.values())} epochs / {summary['training']['optimizer_updates']:,} optimizer updates"]),
        (.53,'REPRODUCTION RESULT',[
        f"Selected human / mixed epochs: {selected[0]} / {selected[1]}",
        f"Weights bitwise equal: {comparison['weights_bitwise_equal']}",
        f"Maximum weight difference: {comparison['maximum_weight_absolute_difference']:.6g}",
        f"Raw human correct: {new['raw']['exact']}/{new['raw']['positive_n']}",
        f"Demo human correct accepts: {new['demo']['exact']}/{new['demo']['positive_n']}",
        f"Demo UNKNOWN accepts: {new['demo']['negative_errors']}/{new['demo']['negative_n']}"])]
    for x,title,lines in columns:
        ax.text(x,.69,title,fontsize=14,weight='bold',color='#147D92')
        ax.text(x,.63,'\n'.join(lines),fontsize=12.5,color='#172A3A',va='top',linespacing=1.8)
    ax.text(.05,.24,f"Strict qualification enabled: {comparison['strict_qualification_enabled']}. Equal seeds do not guarantee identical CUDA CTC weights.",fontsize=13,color='#405363')
    measured = summary.get('pi_benchmark',summary.get('desktop_benchmark'))
    if measured:
        host = 'Raspberry Pi' if summary['physical_pi_measured'] else 'PC'
        latency=measured['latency_ms'];rtf=measured['rtf']
        ax.text(.05,.18,f"{host} recorded inference p50 / p95: {latency['p50']:.2f} / {latency['p95']:.2f} ms; "
                f"RTF p95: {rtf['p95']:.4f}; peak worker RSS: {measured['peak_rss_mib']:.2f} MiB",fontsize=14,color='#405363')
        pending = 'Live microphone and wake/keyword accuracy: NOT MEASURED' if summary['physical_pi_measured'] else 'Raspberry Pi, live microphone and wake/keyword accuracy: NOT MEASURED'
        ax.text(.05,.13,pending,fontsize=14,color='#405363')
    else:
        ax.text(.05,.14,'Pi latency / memory / RTF, live intent and wake/keyword accuracy: NOT MEASURED',fontsize=15,color='#405363')
    ax.text(.05,.07,'Source: results/comparison.json + selected fitting receipt; this card describes this new run only.',fontsize=11,color='#637789')
    _save_figure(fig,output,'model-dataset-training-card'); plt.close(fig)
    _json(output/'metrics.json',summary)
    return summary


def readme_section(summary):
    reproduced=''
    if 'reproduction' in summary:
        result=summary['reproduction'];training=result['training'];pc=result['pc_benchmark']
        human=result['results']['test']['human']
        reproduced=(f"**Reproduction checked on this host:** seed {result['seed']}, "
                    f"{training['optimizer_updates_executed']:,} optimizer updates, "
                    f"{training['fit_development_seconds']/60:.2f} minutes fitting/development on one {training['gpu']}. "
                    f"Weight tensors bitwise equal: **{result['weights_bitwise_equal']}**; maximum difference "
                    f"{result['maximum_weight_absolute_difference']:g}; exported ONNX SHA-256 matches the submission. "
                    f"Human project test: raw {human['raw']['exact']}/{human['raw']['positive_n']}, "
                    f"correct demo accepts {human['demo']['exact']}/{human['demo']['positive_n']}. "
                    f"Regenerated-model PC p50/p95 {pc['latency_ms']['p50']:.2f}/{pc['latency_ms']['p95']:.2f} ms; "
                    f"peak worker RSS {pc['peak_rss_mib']:.2f} MiB. "
                    'The original cached fitting/development frontend and single-thread evaluation frontend are explicitly pinned; '
                    'the initial rounding-drift evaluation is preserved, with no weight or threshold changes. '
                    'The fit used a separate source/output folder and the root interpreter; fresh locked uv setup was verified separately. '
                    '[Machine comparison](submission/reproduction-full/verification.json), '
                    '[new training curves](submission/reproduction-full/figures/training-curves.png), '
                    '[reproduction slide card](submission/reproduction-full/figures/model-dataset-training-card.png).')
    rows = []
    for name in ORDER:
        item = summary['models'][name]; h = item['test']['human']; s = item['test']['synthetic']
        rows.append(f"| {name} | {item['parameters']:,} | {item['export']['precision']} / {item['export']['graph_bytes']/2**20:.3f} MiB | "
                    f"{h['raw_complete_correct']}/{h['positive_n']} | {h['correct_accepts']}/{h['positive_n']} ({h['complete_recall']:.2%}) | "
                    f"{h['wrong_accepts']} | {h['unknown_false_accepts']}/{h['negative_n']} | {s['correct_accepts']}/{s['positive_n']} |")
    official = []
    for name, item in summary.get('official_recorded_holdout',{}).items():
        official.append(f"| {name} | {item['command_accuracy']:.2%} | {item['intent_accuracy']:.2%} | {item['command_macro_f1']:.4f} | {item['unknown_false_accepts']}/{item['unknown_n']} |")
    recipe_rows = []
    asset_rows = []
    for name in ORDER:
        item = summary['models'][name]; dataset=item['dataset']; losses=item['selected_loss_components']
        recipe_rows.append(f"| {name} | {dataset['rows']:,} / {dataset['duration_hours']:.6f} h | "
                           f"{dataset['kinds']['human']['rows']:,} + {dataset['kinds']['synthetic']['rows']:,} | "
                           f"{item['selected_human_foundation_epoch']} → {item['selected_mixed_epoch']} | "
                           f"{item['fitting_seconds_excluding_input_loading']/60:.4f} min | "
                           f"{losses['ctc']:.7f} / {losses['intent_cross_entropy']:.8f} / {losses['target_cross_entropy']:.9f} |")
        assets=item.get('deployment_assets')
        if assets:
            asset_rows.append(f"| {name} | {assets['graph_bytes']:,} | {assets['wake_vad_bytes']:,} | {assets['total_bundle_bytes']:,} |")
    pi_status = ('**Pi/live wake accuracy, intent accuracy, false accepts, p95 latency, RTF, memory and thermal behavior remain unmeasured** '
                 'until the posted scripts run on the actual device. Desktop replay and successful packaging do not fill these fields.')
    if summary['physical_pi_measured']:
        pi=summary['pi_measurements']
        pi_status=(f"Physical Pi recorded replay is attached: inference p95 {pi['latency_p95_ms']:.2f} ms; "
                   f"RTF p95 {pi['real_time_factor']:.4f}; peak RSS {pi['memory_mib']} MiB. "
                   'The hash-bound metrics retain raw/demo decisions and accepted-intent recall separately. '
                   '**Live wake/keyword accuracy and microphone accuracy remain unmeasured**; recorded replay does not measure them.')
    desktop=''
    if 'desktop_benchmark' in summary:
        host=summary['desktop_benchmark']; latency=host.get('latency_ms',{})
        count=host.get('parameters'); parameter_label=f'{count:,}' if isinstance(count,int) else 'unknown'
        table=[]
        for policy in ('raw','demo'):
            for kind in ('human','synthetic'):
                m=host.get('policies',{}).get(policy,{}).get(kind)
                if m:
                    table.append(f"| {policy} / {kind} | {m['complete_correct']}/{m['positive_n']} | {m['unknown_false_accepts']}/{m['negative_n']} |")
        desktop=(f"Latest attached **desktop recorded replay** ({parameter_label} parameters, "
                 f"{host.get('published_rows','?')} recordings): inference p50/p95 "
                 f"{latency.get('p50',0):.2f}/{latency.get('p95',0):.2f} ms; peak worker RSS {host.get('peak_rss_mib')} MiB. "
                 'Includes frontend, ONNX CPU, finite decoding and policy; excludes disk, utterance duration, wake listening and endpoint wait. '
                 'Its hardware/model identity and receipt hash are retained in [figure metrics](submission/figures/metrics.json). '
                 'Typed scoring excludes two uncertain references, retaining their predictions.\n\n'
                 '| Portable desktop policy / group | Complete correct / positive references | UNKNOWN accepts / negatives |\n'
                 '| --- | ---: | ---: |\n'+'\n'.join(table))
    release=('[Download source-built 2M inference ZIP](https://github.com/justinruaya123/smart-speaker/releases/latest/download/smart-speaker-inference-2m.zip). '
             'GitHub Actions builds this ZIP from the checked-out source, selected model/wake graphs and fixed audio, and attaches a checksum and member-hash receipt. '
             'To build locally with the locked inference dependencies: `python -m scripts.submission.build_inference`. No Hugging Face ZIP, model or audio download is used by this build.')
    return '\n'.join([START, '# Final submission: 2M direct VCM', '',
        'The primary model is **2,066,831 parameters, control mixed epoch 32** (`speaker-final-refinement-v1`); the 935,567-parameter pipeline-repair model is retained as comparison evidence. '
        'It encodes a complete utterance after the VAD endpoint and returns finite typed commands or rejection. The final submission uses ONNX Runtime CPU, one thread, pretrained Hey Jarvis/Silero activation and no transcription or cloud inference.', '',
        'Use the [`codex/final-submission` branch](https://github.com/justinruaya123/smart-speaker/tree/codex/final-submission) for these entry points. '
        'For a fresh checkout: `git clone --branch codex/final-submission --single-branch https://github.com/justinruaya123/smart-speaker.git`.', '',
        '```sh', 'python reproduce.py --seed 53 --yes --no-benchmark  # reconstruct pinned data and fit in a new workspace',
        'python benchmark.py   # choose silent PC replay or SSH Pi replay',
        'python inference.py   # choose CLI or prebuilt UI; input/output devices are prompted', '```', '',
        'See the [reproduction guide](docs/REPRODUCIBILITY-935k-2m.md), [dataset membership and terms](docs/DATASETS.md), '
        '[citations](docs/DATASET-CITATIONS.md), and [benchmark handoff](docs/BENCHMARK-HANDOFF.md). '
        'The [selected recovery checkpoint](submission/checkpoints/full-selected.pt) and [verbatim metric histories](submission/training/manifest.json) are retained separately from inference distribution. '
        'Use `--model-choice compact` to reproduce the comparison model explicitly.', '',
        reproduced, '',
        'For the GUI, copy `SpeakerUI/.env.example` to private root `.env` and configure existing HA/Spotify credentials and flags. '
        'The default allowlist is `light.hue_color_lamp_1,light.dining_light,camera.dining_area_main_camera_tc70_mainstream`. '
        'Spotify voice scope is generic **play music** on its active/configured Connect device. '
        'Device controls require explicit `ACTIONS_ENABLED=true`; CLI, reproduction and benchmarks remain action-free. '
        'Map finite light labels under `[bindings.lights]` in private root `config.toml` for spoken light actions. '
        '[GUI configuration](SpeakerUI/README.md) describes setup; credentials are excluded from Git and ZIPs.', '',
        '![Model, dataset and training card](submission/figures/model-dataset-training-card.png)', '',
        '| Model | Fitting rows / hours | Human + synthetic | Selected human → mixed epoch | Study fit/dev wall-clock | Selected CTC / intent CE / target CE |',
        '| --- | ---: | ---: | ---: | ---: | --- |', *recipe_rows, '',
        'Both use one shared A100-SXM4-40GB, seed 53, AdamW (weight decay .001), batch 32 / microbatch 8, '
        '96 updates/epoch, cosine learning rates .0008 human / .00025 mixed (floor .00002), and CTC + .30 intent CE + .15 target CE. '
        'Study times exclude input loading and include the unselected 2M completion arm; loss components are training means. '
        'CUDA CTC is not guaranteed bitwise deterministic.', '',
        '| Model | VCM graph bytes | Wake/VAD assets + metadata bytes | Complete selected bundle bytes |',
        '| --- | ---: | ---: | ---: |', *asset_rows, '',
        'Bundle bytes include graph, wake/VAD and JSON metadata; Python/Node/UI dependencies add to the deployed footprint.', '',
        'Recorded **project typed test**, with 636 human positive references, 229 human UNKNOWN references and 3,368 synthetic positives:', '',
        '| Model | Parameters | ONNX export | Human raw correct | Human correct accepts | Human wrong positive accepts | UNKNOWN accepts | Synthetic correct accepts |',
        '| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |', *rows, '',
        'The compact row uses the frozen per-intent recall policy; 2M uses fixed .95 intent / .90 slot gates. Six human fitting labels and export precision also differ, so this is not a controlled size-only ablation. '
        'Both fail strict qualification; qualified mode accepts zero commands. Published data was previously examined. '
        'Per-intent gaps remain, including compact human brightness correct accepts 0/23. [Compact results](experiments/speaker-pipeline-repair-v1/results.json) and [2M results](experiments/speaker-final-refinement-v1/results.json) retain the complete breakdowns.', '',
        'The external class scorer uses **all 196 holdout references**, including its own UNKNOWN/reference conventions; it has a different denominator from the project test above. '
        'These existing x86 recorded replays use demo decisions:', '',
        '| Model | Official command accuracy | Official intent accuracy | Command macro F1 | UNKNOWN accepts |',
        '| --- | ---: | ---: | ---: | ---: |', *official, '',
        'Source: [locked official replay](experiments/speaker-official-vcm-benchmark-v1/results.json).',
        pi_status, '', desktop, '',
        '![Training and development curves](submission/figures/training-curves.png)', '',
        'The plots show all completed foundation/control epochs, selected epochs and development checks every four epochs; losses are training component means. '
        'The unselected 2M completion-arm history is preserved. The compact repair performed zero additional fits. '
        'Regenerate SVG/PNG figures and the evidence tables with `uv run --group training python -m scripts.submission.report --update-readme`.', '',
        'Extract the prebuilt inference-only ZIP and run `python inference.py`; no pnpm build is required in the ZIP. '
        'It contains the selected graph, wake/VAD assets, minimal runtime lock, prebuilt UI and fixed acknowledgment audio. '
        'Training datasets, trainers, tests, recovery checkpoints and histories are excluded. '
        'Source-checkout GUI startup installs pinned Node/pnpm and builds missing UI assets automatically. '
        'The source build command `python -m scripts.submission.build_inference` installs pinned Node/pnpm dependencies, checks/tests/builds SpeakerUI, verifies the repository model/audio assets and assembles the ZIP. '
        'Root code licensing does not grant rights to dataset recordings, trained weights or third-party wake assets; retain the source-specific notices.', '',
        release, '',
        'The earlier README is preserved below as historical context. Its model defaults, causal architecture and action integrations describe earlier workflows.', END, '', ''])


def update_readme(root, summary):
    path = Path(root)/'README.md'
    original = path.read_text(encoding='utf-8')
    if START in original:
        if not original.startswith(START) or END not in original:
            raise ValueError('Unexpected final-submission section placement')
        original = original.split(END,1)[1].lstrip('\n')
    path.write_text(readme_section(summary)+original, encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--import-local-evidence',action='store_true',help='Hash-check and copy retained histories/checkpoint from the research cache')
    parser.add_argument('--update-readme',action='store_true')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--benchmark',type=Path,help='Attach a completed benchmark metrics.json; desktop receipts never fill Pi cells')
    args = parser.parse_args(argv)
    if args.import_local_evidence:
        import_local_evidence()
    benchmark = args.benchmark
    if benchmark is None and (ROOT/'submission/benchmark/pc-recorded-metrics.json').exists():
        benchmark = ROOT/'submission/benchmark/pc-recorded-metrics.json'
    summary = render(output=args.output,benchmark=benchmark)
    if args.update_readme:
        update_readme(ROOT,summary)
    state='physical Pi recorded replay attached; live wake remains unmeasured' if summary['physical_pi_measured'] else 'physical Pi measurements remain pending'
    print('Submission figures generated from verified histories; '+state+'.')


if __name__ == '__main__':
    main()
