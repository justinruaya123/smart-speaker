"""Full-corpus additions consumed by the existing canonical model publisher."""
import json
import csv
import io
from pathlib import Path


def benchmark_files(manifest):
    from scripts.runtime.pcv_full_benchmark import load_trials
    manifest = Path(manifest).resolve()
    rows = load_trials(manifest)
    entries = {'benchmark/manifest.json':manifest.read_bytes(),
               'benchmark/definitions.json':(manifest.parent/'definitions.json').read_bytes()}
    for row in rows:
        entries['benchmark/'+row['audio']] = (manifest.parent/row['audio']).read_bytes()
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=['file','transcript','command','variation','slot_value',
                                               'speaker_id','is_synthetic','accent_group','out_of_scope'])
    writer.writeheader()
    for row in rows:
        if row['track'] != 'primary_holdout':continue
        writer.writerow({'file':'../'+row['audio'], 'transcript':row['publisher_transcript'],
            'command':row['publisher_intent'], 'variation':row['publisher_variation'], 'slot_value':row['publisher_slot'],
            'speaker_id':row['speaker'], 'is_synthetic':int(row['synthetic']), 'accent_group':'see source card',
            'out_of_scope':int(row['publisher_intent']=='OUT_OF_SCOPE')})
    entries['benchmark/holdout/manifest.csv'] = stream.getvalue().encode()
    study=Path(__file__).resolve().parents[1]/'experiments/pcv-full-benchmark-v1'
    for name in ('protocol.json','publisher-metadata.json','source-card.txt','source-audit.json','acquisition.json','supplemental-protocol.json','supplemental-audit.json','variations.csv','external-source.json'):
        entries['benchmark/evidence/'+name]=(study/name).read_bytes()
    return entries


def main():
    from scripts.pcv_final_delivery import package
    from scripts.pcv_release import validate_release
    root = Path(__file__).resolve().parents[1]
    record = root/'experiments/pcv-release-train-v1/runs/release-v1'
    release = validate_release(root, record/'release.json', record/'selection-before-calibration.json')
    study = root/'experiments/pcv-full-benchmark-v1'
    output = root/'dist/pcv-full-benchmark-v1'
    # Make report links resolve inside the extracted archive, not into the checkout.
    report = root/'.cache/pcv-full-benchmark-v1/BENCHMARK.md'
    report.write_text((root/'docs/EXPERIMENT-pcv-full-benchmark-v1.md').read_text(encoding='utf8').replace(
        '../experiments/pcv-full-benchmark-v1/','../evidence/'),encoding='utf8')
    extras = {'docs/BENCHMARK.md':report}
    for name in ('protocol.json','publisher-metadata.json','source-card.txt','source-audit.json','acquisition.json',
                 'supplemental-protocol.json','supplemental-audit.json','variations.csv'):
        # CSV supported below as a source receipt, not executable content.
        if name.endswith('.csv'):continue
        extras['evidence/'+name] = study/name
    for path in study.glob('*results.json'):extras['evidence/'+path.name] = path
    extras['evidence/external-source.json'] = study/'external-source.json'
    extras['evidence/pc-trials.jsonl'] = root/'.cache/pcv-full-benchmark-v1/pc-full-single-core/trials.jsonl'
    for target in ('standalone','pi'):
        receipt = package(root/release['bundle'], output/f'smart-speaker-{target}-benchmark.zip',
            diagnostic=not release['qualified'], sample_manifest=root/release['sample_manifest'],
            source_receipt=root/release['source_receipt'], source_card=root/release['source_card'],
            extra_files=extras, app_target=target, benchmarks='skip',
            benchmark_manifest=root/'.cache/pcv-full-benchmark-v1/corpus/full-manifest.json')
        print(json.dumps(receipt), flush=True)


if __name__ == '__main__':main()
