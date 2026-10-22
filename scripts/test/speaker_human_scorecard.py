"""Read-only human training/data scorecards; never selects a model or source.

Counts describe admitted reference labels, not independently listened truth.
All model metrics are conditional development diagnostics, not qualification.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
from statistics import NormalDist

import numpy as np

from scripts.runtime.pcv_schema import INTENTS, canonical, validate_command
from scripts.train.speaker_human_data import LABELS, SPLITS, coverage_report, human_metrics


def wilson_interval(successes, total, *, confidence=.95):
    """Recording-level descriptive Wilson interval; not a speaker adjustment."""
    if (type(successes) is not int or type(total) is not int
            or not 0 <= successes <= total or not 0 < confidence < 1):
        raise ValueError('Invalid binomial counts/confidence')
    if not total:
        return None
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    p = successes / total
    scale = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / scale
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / scale
    return [max(0., centre - radius), min(1., centre + radius)]


def _distribution(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return dict(n=0, minimum=None, median=None, p95=None, maximum=None, total_seconds=None)
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite descriptive data')
    return dict(n=len(a), minimum=float(a.min()), median=float(np.median(a)),
                p95=float(np.percentile(a, 95)), maximum=float(a.max()), total_seconds=float(a.sum()))


def _stratum(rows, audio):
    speakers = Counter(r['speaker'] for r in rows)
    diagnostics = [audio[r['id']] for r in rows if r['id'] in audio]
    commands, slots = defaultdict(list), defaultdict(list)
    for row in rows:
        if row['expected'] is not None:
            commands[canonical(row['expected'])].append(row)
        for name, value in (row['expected'] or {}).get('slots', {}).items():
            slots[name + '=' + canonical(value)].append(row)
    cell = lambda rs: dict(rows=len(rs), speakers=len({r['speaker'] for r in rs}))
    return dict(rows=len(rows), speakers=len(speakers),
        speaker_effective_count=len(rows) ** 2 / sum(n * n for n in speakers.values()) if rows else None,
        largest_speaker_fraction=max(speakers.values()) / len(rows) if rows else None,
        sources=dict(Counter(r.get('publisher_source', 'unknown') for r in rows)),
        transcript_origins=dict(Counter(r.get('transcript_source', 'unknown') for r in rows)),
        independently_reviewed_rows=sum(r.get('listening_status') == 'reviewer_attested_independent_consensus' for r in rows),
        unique_texts=len({r['text'] for r in rows if r.get('text')}),
        text_known_rows=sum(bool(r.get('text')) for r in rows),
        full_ctc_rows=sum(r.get('loss_mask', {}).get('ctc', bool(r.get('phones'))) for r in rows),
        semantic_only_rows=sum(r.get('loss_mask', {}).get('intent', False)
                               and not r.get('loss_mask', {}).get('ctc', True) for r in rows),
        typed_commands=len(commands), rejection_rows=sum(r['expected'] is None for r in rows),
        commands={k: cell(v) for k, v in sorted(commands.items())},
        slot_values={k: cell(v) for k, v in sorted(slots.items())},
        audio_audit_rows=len(diagnostics),
        duration=_distribution([r['duration_seconds'] for r in diagnostics if 'duration_seconds' in r]),
        flags=dict(Counter(flag for r in diagnostics for flag in r.get('flags', []))),
        missing_group_metadata={key: sum(not r.get(key) for r in rows) for key in
            ('session_id', 'parent_recording_id', 'speaker_alias_group', 'source_speaker_key')},
        duplicate_pcm_rows=len(rows) - len({r['pcm_sha256'] for r in rows}) if rows else 0,
        duplicate_pair_rows=len(rows) - len({r.get('pair', r['id']) for r in rows}) if rows else 0)


def dataset_profile(rows, audio_records, baseline, catalog) -> dict:
    """Describe the strongest admitted pool; never declare a best source fit."""
    rows = list(rows)
    if (len({r['id'] for r in rows}) != len(rows)
            or any(r.get('synthetic') is not False or r.get('kind') != 'human'
                   or r.get('metric_eligible') is not True or r['split'] not in SPLITS
                   or (r['split'] == 'train' and r.get('training_eligible') is not True) for r in rows)):
        raise ValueError('Expected unique admitted original-human references')
    for row in rows:
        if row['expected'] is not None:
            validate_command(row['expected'])
    audio = {r['id']: r for r in audio_records}
    if len(audio) != len(audio_records):
        raise ValueError('Duplicate audio audit identity')
    coverage = coverage_report(rows, baseline, catalog=catalog)
    result = dict(per_intent={}, totals={s: _stratum([r for r in rows if r['split'] == s], audio) for s in SPLITS})
    for label in LABELS:
        examples = [r for r in rows if (r['expected'] or {}).get('intent', 'UNKNOWN') == label]
        train = [r for r in examples if r['split'] == 'train']
        sources = []
        for source in sorted({r.get('publisher_source', 'unknown') for r in train}):
            source_rows = [r for r in train if r.get('publisher_source', 'unknown') == source]
            sources.append(dict(source=source, **_stratum(source_rows, audio)))
        sources.sort(key=lambda r: (-r['speakers'], -r['typed_commands'], -r['rows'], r['source']))
        result['per_intent'][label] = dict(
            splits={s: _stratum([r for r in examples if r['split'] == s], audio) for s in SPLITS},
            coverage=coverage[label], training_sources=sources, qualified_best_source=None,
            source_order='Descriptive speaker/command depth, not recognition quality or an ablation winner')
    return result


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _label(command):
    return command['intent'] if command is not None else 'UNKNOWN'


def _head(row):
    index = (row.get('diagnostic_candidate') or {}).get('intent_index')
    if index is None:
        return None
    if type(index) is not int or not 0 <= index <= len(INTENTS):
        raise ValueError('Invalid predicted head index')
    return INTENTS[index - 1] if index else 'UNKNOWN'


def _overall(rows) -> dict:
    base: dict = human_metrics(rows)
    result: dict = {k: v for k, v in base.items() if k != 'per_intent'}
    positive = [r for r in rows if r['expected'] is not None]
    negative = [r for r in rows if r['expected'] is None]
    exact = lambda r, key: r.get(key) is not None and r[key] == r['expected']
    accepted = sum(r.get('application_command') is not None for r in rows)
    raw_accepted = sum(r.get('candidate') is not None for r in rows)
    head_rows = [r for r in rows if _head(r) is not None]
    head_correct = sum(_head(r) == _label(r['expected']) for r in head_rows)
    raw_correct = base['raw_correct']
    correct = base['accepted_correct']
    result.update(rows=len(rows), accepted_n=accepted, raw_candidate_n=raw_accepted,
        accepted_f1=_ratio(2 * correct, len(positive) + accepted),
        raw_complete_recall=_ratio(raw_correct, len(positive)),
        raw_complete_precision=_ratio(raw_correct, raw_accepted),
        raw_complete_f1=_ratio(2 * raw_correct, len(positive) + raw_accepted),
        known_rejections=sum(r.get('application_command') is None for r in positive),
        raw_correct_rejected=sum(exact(r, 'candidate') and r.get('application_command') is None for r in positive),
        raw_intent_correct=sum(_label(r.get('candidate')) == _label(r['expected']) for r in positive),
        unknown_false_accept_rate=_ratio(base['unknown_false_accepts'], len(negative)),
        unknown_false_accept_wilson95=wilson_interval(base['unknown_false_accepts'], len(negative)),
        accepted_precision_wilson95=wilson_interval(correct, accepted),
        accepted_recall_wilson95=wilson_interval(correct, len(positive)),
        intent_head_correct=head_correct, intent_head_scored_rows=len(head_rows),
        intent_head_accuracy=_ratio(head_correct, len(head_rows)),
        acceptance_rate=_ratio(accepted, len(rows)),
        rejection_reasons=dict(Counter(r.get('application_reason') or 'unspecified'
                                       for r in rows if r.get('application_command') is None)))
    return result


def _head_metrics(rows, label):
    present = [r for r in rows if _head(r) is not None]
    reference_n = sum(_label(r['expected']) == label for r in present)
    predicted_n = sum(_head(r) == label for r in present)
    correct = sum(_head(r) == label and _label(r['expected']) == label for r in present)
    return dict(intent_head_correct=correct, intent_head_reference_n=reference_n,
        intent_head_precision=_ratio(correct, predicted_n), intent_head_recall=_ratio(correct, reference_n),
        intent_head_f1=_ratio(2 * correct, reference_n + predicted_n))


def _intent_metrics(rows, label):
    examples = [r for r in rows if _label(r['expected']) == label]
    if label == 'UNKNOWN':
        false_accepts = sum(r.get('application_command') is not None for r in examples)
        return dict(reference_n=len(examples), speakers=len({r['speaker'] for r in examples}),
            false_accepts=false_accepts, correct_rejections=len(examples) - false_accepts,
            false_accept_rate=_ratio(false_accepts, len(examples)),
            false_accept_wilson95=wilson_interval(false_accepts, len(examples)), **_head_metrics(rows, label))
    predicted = [r for r in rows if _label(r.get('application_command')) == label]
    raw_predicted = [r for r in rows if _label(r.get('candidate')) == label]
    exact = lambda r, key: r.get(key) is not None and r[key] == r['expected']
    tp = sum(exact(r, 'application_command') for r in examples)
    fp, fn = len(predicted) - tp, len(examples) - tp
    raw_correct = sum(exact(r, 'candidate') for r in examples)
    slot_metrics = {}
    for name in sorted({key for r in examples for key in r['expected']['slots']}):
        relevant = [r for r in examples if name in r['expected']['slots']]
        conditional = [r for r in relevant if _label(r.get('candidate')) == label]
        correct = sum(r['candidate']['slots'].get(name) == r['expected']['slots'][name] for r in conditional)
        slot_metrics[name] = dict(reference_n=len(relevant), raw_conditional_n=len(conditional),
            raw_conditional_correct=correct, raw_conditional_accuracy=_ratio(correct, len(conditional)),
            raw_joint_intent_slot_accuracy=_ratio(correct, len(relevant)))

    speaker_recalls = [sum(exact(r, 'application_command') for r in examples if r['speaker'] == speaker)
                       / sum(r['speaker'] == speaker for r in examples)
                       for speaker in sorted({r['speaker'] for r in examples})]
    return dict(reference_n=len(examples), speakers=len(speaker_recalls), tp=tp, fp=fp, fn=fn,
        raw_correct=raw_correct, raw_complete_recall=_ratio(raw_correct, len(examples)),
        raw_complete_precision=_ratio(raw_correct, len(raw_predicted)),
        raw_intent_correct=sum(_label(r.get('candidate')) == label for r in examples),
        accepted_precision=_ratio(tp, len(predicted)), accepted_recall=_ratio(tp, len(examples)),
        accepted_f1=_ratio(2 * tp, 2 * tp + fp + fn), accepted_predictions=len(predicted),
        accepted_precision_wilson95=wilson_interval(tp, len(predicted)),
        accepted_recall_wilson95=wilson_interval(tp, len(examples)),
        raw_recall_wilson95=wilson_interval(raw_correct, len(examples)),
        accepted_macro_speaker_recall=float(np.mean(speaker_recalls)) if speaker_recalls else None,
        **_head_metrics(rows, label), slots=slot_metrics,
        rejected=sum(r.get('application_command') is None for r in examples),
        wrong_positive_accepts=sum(r.get('application_command') is not None and not exact(r, 'application_command') for r in examples),
        rejected_despite_raw_correct=sum(exact(r, 'candidate') and r.get('application_command') is None for r in examples),
        rejection_reasons=dict(Counter(r.get('application_reason') or 'unspecified'
                                       for r in examples if r.get('application_command') is None)))


def prediction_metrics(records) -> dict:
    """Exact typed commands, head-only intent, slots and rejection are distinct."""
    records = list(records)
    if (len({r['id'] for r in records}) != len(records)
            or any(r.get('synthetic') is not False or r.get('kind') != 'human' for r in records)):
        raise ValueError('Expected unique original-human prediction rows')
    rows = [r for r in records if r.get('metric_eligible') is True]
    for row in rows:
        for key in ('expected', 'candidate', 'application_command'):
            if row.get(key) is not None:
                validate_command(row[key])
    confusion = {key: {label: {prediction: 0 for prediction in LABELS} for label in LABELS}
                 for key in ('raw_candidate', 'application', 'intent_head')}
    for row in rows:
        truth = _label(row['expected'])
        confusion['raw_candidate'][truth][_label(row.get('candidate'))] += 1
        confusion['application'][truth][_label(row.get('application_command'))] += 1
        head = _head(row)
        if head is not None:
            confusion['intent_head'][truth][head] += 1
    overall = _overall(rows)
    overall['ambiguous_n'] = len(records) - len(rows)
    by_source = {}
    for source in sorted({r.get('publisher_source', 'unknown') for r in rows}):
        subset = [r for r in rows if r.get('publisher_source', 'unknown') == source]
        by_source[source] = dict(overall=_overall(subset),
            per_intent={label: _intent_metrics(subset, label) for label in LABELS})
    return dict(overall=overall, per_intent={label: _intent_metrics(rows, label) for label in LABELS},
        per_source=by_source, confusion=confusion,
        confusion_note='UNKNOWN prediction column means no-command/rejection for candidate/application; intent_head uses only actual diagnostic logits, never reference intent_index.',
        interpretation='Development diagnostics on an already-trained checkpoint; not a new data treatment effect or blind qualification.')


def speaker_bootstrap(records, *, label=None, resamples=2000, seed=53) -> dict:
    """Percentile cluster intervals conditional on the observed speaker pool.

    Resample whole declared speaker groups, never individual recordings. A
    constant interval from one speaker is explicitly marked insufficient.
    """
    if label not in (None, *INTENTS) or type(resamples) is not int or resamples < 1:
        raise ValueError('Invalid bootstrap label/resample count')
    records = list(records)
    if any(r.get('synthetic') is not False or r.get('kind') != 'human' for r in records):
        raise ValueError('Speaker intervals require original-human records')
    rows = [r for r in records if r.get('metric_eligible') is True]
    speakers = sorted({r['speaker'] for r in rows})
    index = {speaker: i for i, speaker in enumerate(speakers)}
    counts = np.zeros((len(speakers), 5), dtype=np.int64)
    reference_speakers, negative_speakers = set(), set()
    for row in rows:
        truth = row['expected'] is not None and (label is None or _label(row['expected']) == label)
        predicted = row.get('application_command') is not None and (label is None or _label(row['application_command']) == label)
        correct = truth and row.get('application_command') == row['expected']
        negative = row['expected'] is None
        counts[index[row['speaker']]] += [correct, truth, predicted, negative and predicted, negative]
        if truth:
            reference_speakers.add(row['speaker'])
        if negative:
            negative_speakers.add(row['speaker'])
    if speakers:
        weights = np.random.default_rng(seed).multinomial(len(speakers),
            np.full(len(speakers), 1 / len(speakers)), size=resamples)
        totals = weights @ counts
    else:
        totals = np.empty((0, 5))
    result: dict = dict(speaker_n=len(reference_speakers), negative_speaker_n=len(negative_speakers),
        sampled_speaker_n=len(speakers), resamples=resamples, seed=seed,
        insufficient_speaker_evidence=len(reference_speakers) < 3,
        interpretation='Conditional on observed declared speakers; aliases/session independence are not verified and sparse intervals can degenerate.')
    for name, numerator, denominator in (('accepted_recall', 0, 1), ('accepted_precision', 0, 2),
                                         ('unknown_false_accept_rate', 3, 4)):
        valid = totals[:, denominator] > 0
        values = totals[valid, numerator] / totals[valid, denominator]
        result[name + '95'] = np.percentile(values, [2.5, 97.5]).tolist() if len(values) else None
        result[name + '_valid_resamples'] = int(valid.sum())
    return result
