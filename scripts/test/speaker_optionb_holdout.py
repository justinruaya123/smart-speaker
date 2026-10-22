"""Read-only OptionB annotation/coverage and safe human-feature loading.

Text operations inspect supplied supervision only, never runtime transcripts.
Cue compatibility is not an acoustic decision or permission to admit a label.
"""
from collections import Counter,defaultdict
import numpy as np
from scripts.runtime.pcv_schema import canonical,validate_command
from scripts.train.pcv_updated_prepare import spoken,annotate
from scripts.train.speaker_human_data import SPLITS,human_supervision,file_sha256,split_leakage
from vcm.catalog import PHONES


COURTESY={'please','kindly','could','can','would','you','just','now','for','me','the','a','an','to'}


def _cue_tokens(text,intent):
    ignored=COURTESY-{'a'} if intent=='ALARM' else COURTESY
    return tuple(w for w in spoken(text).split() if w not in ignored)


def cue_match(text,definition):
    """Conservative ordered-cue diagnostic; unknown content remains visible.

    Removing a small declared function/courtesy vocabulary cannot erase a
    negation, extra number, target, action or AM/PM distinction. Equal cues
    still require independent semantic/listening review before label admission.
    """
    command=definition['command'];validate_command(command)
    return _cue_tokens(text,command['intent'])==_cue_tokens(
        definition.get('spoken_text',definition['text']),command['intent'])


def wording_audit(rows,definitions,predictions=()) -> list[dict]:
    """All verbatim cells, with padded matches kept out of exact coverage.

    Empty input strata are marked unprovided, not assumed missing-corpus data.
    Typed/cue counts can overlap across equivalent benchmark wordings and must
    not be summed as independent recordings. Publisher buckets are separate.
    """
    rows=list(rows);predictions=list(predictions)
    if len({r['id'] for r in rows})!=len(rows) or any(
            r.get('kind') not in ('human','synthetic')
            or r.get('synthetic') is not (r['kind']=='synthetic')
            or r.get('split') not in SPLITS for r in rows):
        raise ValueError('Expected unique original evidence with declared provenance')
    source={r['id']:r for r in rows};pred={r['id']:r for r in predictions}
    if len(pred)!=len(predictions) or any(k not in source or r['expected']!=source[k]['expected']
                                         for k,r in pred.items()):
        raise ValueError('Prediction/reference identity mismatch')
    groups=defaultdict(list)
    texts={r['id']:spoken(r.get('publisher_transcript',r.get('text',''))) for r in rows}
    cue_cache={}
    for row in rows:groups[row['kind']+'/'+row['split']].append(row)
    output=[]
    for definition in definitions:
        command=definition['command'];validate_command(command)
        intent=command['intent'];text=spoken(definition.get('spoken_text',definition['text']))
        cue=_cue_tokens(text,intent);cells={}
        for kind in ('human','synthetic'):
            for split in SPLITS:
                key=kind+'/'+split;members=groups[key]
                exact=[];padded=[];typed=[];conflicts=ambiguous=buckets=0
                for row in members:
                    rid=row['id'];is_exact=texts[rid]==text
                    if row.get('metric_eligible') is not True:
                        ambiguous+=is_exact;continue
                    same=row['expected']==command
                    if same:typed.append(row)
                    if is_exact:
                        if same:exact.append(row)
                        else:conflicts+=1
                    elif same:
                        cache_key=(rid,intent)
                        if cache_key not in cue_cache:cue_cache[cache_key]=_cue_tokens(texts[rid],intent)
                        if cue_cache[cache_key]==cue:padded.append(row)
                    if same and spoken(row.get('publisher_variation',''))==text:buckets+=1
                scored=[pred[r['id']] for r in exact if r['id'] in pred]
                good=sum(r.get('application_command')==command for r in scored)
                raw=sum(r.get('candidate')==command for r in scored)
                accepted=sum(r.get('application_command') is not None for r in scored)
                cells[key]=dict(stratum_input_rows=len(members),input_scope_provided=bool(members),
                    exact_n=len(exact),exact_speakers=len({r['speaker'] for r in exact}),
                    cue_nonexact_n=len(padded),cue_nonexact_speakers=len({r['speaker'] for r in padded}),
                    typed_reference_n=len(typed),typed_speakers=len({r['speaker'] for r in typed}),
                    reference_conflict_n=conflicts,ambiguous_matching_n=ambiguous,publisher_bucket_n=buckets,
                    scored_n=len(scored),raw_correct=raw,accepted_correct=good,wrong_accepted=accepted-good,
                    rejected=len(scored)-accepted,accepted_recall=good/len(scored) if scored else None)
        output.append(dict(id=definition['id'],verbatim=definition['text'],spoken=text,command=command,strata=cells))
    return output


def annotation_review(row,metadata,*,policy=None) -> dict:
    """Re-run the pinned offline label rules without changing any reference.

    Rule agreement is not independent listening. Rule disagreement is a review
    item, never proof that the stored label is wrong or permission to train it.
    """
    if type(metadata.get('is_synthetic')) not in (int,bool) or bool(metadata['is_synthetic']) is not row['synthetic']:
        raise ValueError('Publisher provenance disagreement')
    if policy is None:
        derived=annotate(metadata)
        version='pcv-updated-voice-v1-base-rules-only'
    else:
        from scripts.train.pcv_release_prepare import annotate as release_annotate
        derived=release_annotate(metadata,policy=policy)
        version=policy['version']
    flags=[]
    if derived['expected']!=row['expected']:flags.append('semantic_reference_disagreement')
    if derived['slot_metadata_conflict']:flags.append('publisher_slot_or_family_conflict')
    if derived['reference_status']=='ambiguous' and row.get('metric_eligible') is True:
        flags.append('unresolved_reference_was_scored')
    if metadata.get('transcript_source')=='whisper':flags.append('asr_derived_not_independently_reviewed')
    if row['split']=='train':
        rebuilt=human_supervision(row)
        if any(row.get(k)!=rebuilt[k] for k in ('intent_index','target_index')):
            flags.append('inconsistent_dependent_supervision')
    return dict(id=row['id'],split=row['split'],stored_expected=row['expected'],
        annotation_rule_version=version,
        metadata_derived=derived['expected'],derived_status=derived['reference_status'],
        rule_reason=derived['reason'],flags=flags,admit_to_training=False,label_changed=False,
        independent_listening=False,action='review_only' if flags else 'metadata_rule_agreement_only')


def load_admitted_features(rows,normalization,*,reserved=()) -> tuple[dict,dict]:
    """Fail closed on provenance, frozen membership and dependent supervision.

    This loads a new research training view only, not a deployed model's input
    transform. No basename fallback, refitting, source repair or test loading.
    """
    rows=list(rows);reserved=list(reserved)
    if not rows or len({r['id'] for r in rows})!=len(rows) or any(
            r.get('split')!='train' or r.get('kind')!='human' or r.get('synthetic') is not False
            or r.get('metric_eligible') is not True or r.get('training_eligible') is not True
            or r.get('reference_status') in ('ambiguous','auxiliary') for r in rows):
        raise ValueError('Expected unique admitted human train rows')
    if split_leakage(rows+reserved) or any(r.get('split')=='train' for r in reserved):
        raise ValueError('Training/reserved split overlap')
    if len({r['pcm_sha256'] for r in rows})!=len(rows):
        raise ValueError('Duplicate admitted training PCM')
    identities=normalization.get('source_ids',[])
    if (len(identities)!=len(rows) or len(set(identities))!=len(identities)
            or set(identities)!={r['id'] for r in rows}
            or normalization.get('human_training_utterances')!=len(rows)):
        raise ValueError('Exact train-only normalization membership mismatch')
    mean=np.asarray(normalization['mean'],dtype=np.float32)
    std=np.asarray(normalization['std'],dtype=np.float32)
    if mean.shape!=(40,) or std.shape!=(40,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std<=0):
        raise ValueError('Invalid frozen human normalization')
    masks={}
    for row in rows:
        rebuilt=human_supervision(row)
        if any(type(row.get(key)) is not int or row[key]!=rebuilt[key] for key in ('intent_index','target_index')):
            raise ValueError('Inconsistent dependent supervision: '+row['id'])
        mask=row.get('loss_mask') or dict(ctc=bool(row.get('phones')),intent=True,target=row['target_index']!=-100)
        if set(mask)!={'ctc','intent','target'} or any(type(v) is not bool for v in mask.values()) or not mask['intent']:
            raise ValueError('Invalid head-specific supervision mask')
        if mask['target'] and row['target_index'] not in (0,1):
            raise ValueError('Supervised target mask requires a labelled target')
        if mask['ctc'] and (not row.get('phones') or any(type(p) is not int or not 1<=p<=len(PHONES) for p in row['phones'])):
            raise ValueError('Missing/invalid nonblank CTC supervision')
        masks[row['id']]=mask
    arrays={}
    for row in rows:
        if file_sha256(row['features'])!=row['features_sha256']:
            raise ValueError('Prepared feature checksum mismatch: '+row['id'])
        raw=np.load(row['features'],allow_pickle=False)
        if raw.dtype!=np.float32 or raw.ndim!=2 or raw.shape[0]!=40 or not raw.shape[1] or not np.isfinite(raw).all():
            raise ValueError('Invalid finite float32 40-bin features')
        if masks[row['id']]['ctc']:
            labels=row['phones'];required=len(labels)+sum(a==b for a,b in zip(labels,labels[1:]))
            if (raw.shape[1]+1)//2<required:raise ValueError('Insufficient CTC frames')
        values=(raw-mean[:,None])/std[:,None]
        if not np.isfinite(values).all():raise ValueError('Nonfinite normalized features')
        arrays[row['id']]=values
    return arrays,dict(rows=len(rows),full_ctc_rows=sum(m['ctc'] for m in masks.values()),
        semantic_only_rows=sum(not m['ctc'] for m in masks.values()),target_rows=sum(m['target'] for m in masks.values()),
        normalization_membership_verified=True,reserved_rows_checked=len(reserved),
        normalized_dtype='float32',source_paths_used_exactly=True,training_performed=False)
