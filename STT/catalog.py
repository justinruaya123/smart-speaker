"""In-memory dynamic catalog experiment. No Spotify, playlist, or file integration."""
from __future__ import annotations
from difflib import SequenceMatcher
import re
import time
import torch
from einops import repeat, rearrange
from torch.nn.utils.rnn import pad_sequence
from common import CHARS, normalize, digest

TYPES={'song_name','artist_name'}
TOKEN={c:i for i,c in enumerate(CHARS)}

def parse_music_request(text):
    """Return inferred text fields and their positions, using only decoded text."""
    text=normalize(text)
    match=re.search(r'\b(?:play|listen to|put on|hear)\s+',text)
    if not match:return {}
    start=match.end()
    tail=text[start:]
    tail=re.sub(r'\s+(?:please|on spotify)$','',tail)
    prefix=re.match(r'(?:(?:the|a|some|my) )?(?:song|track|music|songs)\s+',tail)
    if prefix:start+=prefix.end();tail=tail[prefix.end():]
    marker=re.search(r'\b(?:by|from)\s+',tail)
    if marker:
        left=tail[:marker.start()].strip()
        right=tail[marker.end():].strip()
        result={}
        if left and left not in {'some music','music','songs','something'}:
            result['song_name']={'text':left,'start':start,'end':start+len(left)}
        if right:
            p=start+marker.end()
            result['artist_name']={'text':right,'start':p,'end':p+len(right)}
        return result
    if tail.startswith('by '):
        return {'artist_name':{'text':tail[3:],'start':start+3,'end':start+len(tail)}}
    return {'song_name':{'text':tail.strip(),'start':start,'end':start+len(tail.strip())}} if tail.strip() else {}

def build_catalogs(rows):
    by_split={s:{(e['type'],e['text']) for r in rows if r['split']==s for e in r.get('entities',[]) if e['type'] in TYPES and e['text']} for s in ['train','dev','test']}
    fixed=by_split['train']
    # A frozen synthetic playlist fixture. Only entity names are revealed, not
    # complete reference sentences/audio. Half of unseen entries stay absent.
    additions={x for x in by_split['dev']|by_split['test'] if x not in fixed and int(digest(x[0]+'/'+x[1])[:8],16)%2==0}
    return sorted(fixed),sorted(fixed|additions),sorted(additions)

@torch.inference_mode()
def score_candidates(log_probs, texts):
    tokens=[torch.tensor([TOKEN[c] for c in text],dtype=torch.long) for text in texts]
    target=pad_sequence(tokens,batch_first=True).to(log_probs.device)
    lengths=torch.tensor([len(x) for x in tokens],dtype=torch.long)
    frames=log_probs.shape[0]
    scores=torch.nn.functional.ctc_loss(repeat(log_probs,'time chars -> time batch chars',batch=len(texts)),target,
        torch.full((len(texts),),frames,dtype=torch.long),lengths,blank=0,reduction='none',zero_infinity=False)
    return scores.detach().cpu().tolist()

def rank_fields(text, log_probs, catalog):
    fields=parse_music_request(text)
    out={}
    for kind,field in fields.items():
        query=field['text']
        # An unqualified "play X" can name an artist or a song. A by/from
        # clause, when actually decoded, supplies an explicit artist cue.
        candidates=[(k,name) for k,name in catalog if k==kind or kind=='song_name']
        ranked=sorted(((SequenceMatcher(None,query,name,autojunk=False).ratio(),k,name) for k,name in candidates),reverse=True)[:8]
        if not ranked:continue
        hyps=[text]+[text[:field['start']]+name+text[field['end']:] for _,_,name in ranked]
        acoustic=score_candidates(log_probs,hyps)
        scored=[]
        for (similarity,candidate_kind,name),loss in zip(ranked,acoustic[1:]):
            penalty=(loss-acoustic[0])/max(1,len(log_probs))
            # Fixed acoustic tie-breaker, threshold calibrated separately on dev.
            score=similarity-.3*max(0,penalty)
            scored.append({'value':name,'type':candidate_kind,'score':score,'similarity':similarity,'acoustic_penalty':penalty})
        scored.sort(key=lambda x:x['score'],reverse=True)
        best=scored[0]
        best['margin']=best['score']-(scored[1]['score'] if len(scored)>1 else 0)
        best['query']=query
        destination=best['type']
        if destination not in out or out[destination]['score']<best['score']:
            out[destination]=best
    return out

def accept_fields(ranked,threshold,margin):
    return {kind:r['value'] for kind,r in ranked.items() if r['score']>=threshold and r['margin']>=margin}
