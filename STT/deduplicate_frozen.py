"""One-time cleanup of inherited manifest resampling; no new data selection."""
from collections import Counter, defaultdict
import hashlib
import json
from common import ROOT, read_jsonl, write_json, write_jsonl, extract_number


def main():
    archive = ROOT/'data/archive'
    archive.mkdir(exist_ok=True)
    digests = {}
    audit = {}
    for name in ['manifest', 'numeric_manifest']:
        path = ROOT/'data'/f'{name}.jsonl'
        dest = archive/f'{name}_resampled.jsonl'
        assert path.resolve().is_relative_to(ROOT) and dest.resolve().is_relative_to(ROOT)
        rows = read_jsonl(dest if dest.exists() else path)
        grouped = defaultdict(list)
        duplicates = []
        for r in rows:
            audio = ROOT/r['audio']
            assert audio.resolve().is_relative_to(ROOT)
            if r['audio'] not in digests:
                with audio.open('rb') as f:
                    digests[r['audio']] = hashlib.file_digest(f, 'sha256').hexdigest()
            key = digests[r['audio']]
            r['audio_sha256'] = key
            grouped[key].append(r)
        good = []
        conflicts = []
        cross_split = []
        for group in grouped.values():
            if len({r['split'] for r in group}) > 1:
                cross_split.extend(group)
                continue
            if len({r['text'] for r in group}) > 1:
                conflicts.extend(group)
                continue
            good.append(group[0])
            for r in group[1:]:
                duplicates.append({'removed': r['id'], 'kept': group[0]['id'], 'source': r['source'], 'split': r['split']})
        if not dest.exists():
            path.rename(dest)
        write_jsonl(path, good)
        audit[name] = {'before_rows': len(rows), 'after_unique_audio': len(good),
                       'removed_duplicate_rows': len(duplicates), 'quarantined_label_conflict_rows': len(conflicts),
                       'quarantined_cross_split_rows': len(cross_split),
                       'duplicates_by_source_split': dict(Counter(x['source']+'/'+x['split'] for x in duplicates))}
        write_jsonl(ROOT/'evidence'/f'{name}_duplicate_rows.jsonl', duplicates)
        write_jsonl(ROOT/'evidence'/f'{name}_quarantined.jsonl', conflicts+cross_split)
        if name == 'manifest':
            old_audit = archive/'data_audit_resampled.json'
            previous = json.loads((old_audit if old_audit.exists() else ROOT/'evidence/data_audit.json').read_text())
            if not old_audit.exists():
                write_json(old_audit, previous)
            previous.update(rows=len(good), unique_audio_sha256=len(good),
                            removed_resampling_duplicates=len(duplicates),
                            source_split=dict(Counter(r['source']+'/'+r['split'] for r in good)),
                            hours=sum(r['duration'] for r in good)/3600,
                            human_numeric_coverage={s:dict(Counter(str(extract_number(r['text'])) for r in good if r['split']==s and extract_number(r['text']) is not None)) for s in ['train','dev','test']})
            write_json(ROOT/'evidence/data_audit.json', previous)
    for name in ['scratch_ctc', 'numeric_ctc']:
        path = ROOT/'runs'/name
        dest = ROOT/'runs'/(name+'_resampled')
        assert path.resolve().is_relative_to(ROOT) and dest.resolve().is_relative_to(ROOT)
        if not dest.exists():
            path.rename(dest)
    write_json(ROOT/'evidence/deduplication.json', audit)
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
