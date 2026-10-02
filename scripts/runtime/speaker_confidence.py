"""Application confidence diagnostics; frozen benchmark decisions stay separate."""
import math
from scripts.runtime.pcv_schema import INTENTS, canonical, validate_command


def slot_confidences(scores, entries, candidate, frames):
    """Best-alias likelihood per value, normalized within the selected intent.

    Scores are per-frame CTC log likelihoods. Restore sequence likelihood before
    normalization. These are finite-catalog conditional scores, not calibrated
    empirical probabilities. Max aggregation prevents duplicate-alias inflation.
    """
    if not candidate:return {}
    result={}
    for key,value in candidate['slots'].items():
        competing={}
        for score,entry in zip(scores,entries):
            command=entry['command']
            if command is None or command['intent']!=candidate['intent'] or not math.isfinite(float(score)):continue
            slot=canonical(command['slots'].get(key))
            competing[slot]=max(competing.get(slot,-math.inf),float(score)*frames)
        selected=canonical(value)
        if selected not in competing:result[key]=0.;continue
        peak=max(competing.values())
        result[key]=math.exp(competing[selected]-peak)/sum(math.exp(v-peak) for v in competing.values())
    return result


def application_decision(event, config):
    mode=config['mode'];record=event.get('diagnostic_candidate') or {}
    command=record.get('candidate') if mode=='demo' else event['output'].get('command')
    result={'mode':mode,'accepted':False,'command':None,'reason':None,
            'intent_confidence':record.get('intent_probability'), 'slot_confidence':record.get('slot_confidence',{}),
            'intent_threshold':config['intent_threshold'],'slot_threshold':config['slot_threshold']}
    def reject(reason):return {**result,'reason':reason}
    if event['output'].get('reason')=='capture_fault':return reject('capture_fault')
    if mode=='qualified' and not event['output'].get('accepted'):return reject(event['output'].get('reason'))
    try:validate_command(command)
    except (ValueError,TypeError,KeyError):return reject('invalid_command')
    confidence=record.get('intent_probability')
    if type(confidence) not in (float,int) or not math.isfinite(confidence) or not config['intent_threshold']<=confidence<=1:return reject('intent_confidence')
    if record.get('intent_index')!=INTENTS.index(command['intent'])+1:return reject('intent_disagreement')
    for name in command['slots']:
        probability=result['slot_confidence'].get(name)
        if type(probability) not in (float,int) or not math.isfinite(probability) or not config['slot_threshold']<=probability<=1:return reject('slot_confidence')
    if command['intent'] in {'LIGHT_ON','LIGHT_OFF','BRIGHTNESS','COLOR'}:
        probability=record.get('target_named_probability')
        if type(probability) not in (int,float) or not math.isfinite(probability) or not 0<=probability<=1:return reject('target_confidence')
        named=command['slots']['light_id']!='light.all'
        if (named and probability<config['slot_threshold']) or (not named and 1-probability<config['slot_threshold']):return reject('target_confidence')
    return {**result,'accepted':True,'command':command}
