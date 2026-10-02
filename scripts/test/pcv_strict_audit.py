"""Source-only audit utilities. No download, training, installation or actions.

Reference classes are loaded from the checked-out source using an explicit AST
allowlist so trainer/runtime optional dependencies are not required for algebra
checks. Sandbox reconstruction is accepted only after Git blob-hash verification.
"""
from __future__ import annotations
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
PRIMARY='airimonda/ai231-me2-voice-commands'
OPTIONAL='martinnavs/ai231-fil-supplemental-data'
BASE='7d8296c96e19e1e65fc3f0cd1b82d2391df33dbc'
REFERENCES={
 'decoder':('experiments/phrase-contract-v1/pcvcm/decoder.py','02641b4a6f3342a09b2f0eaa6f6940054dd7c47c'),
 'context':('scripts/models/pcv_contextual.py','a85bc5e3e9f70ba3a4ef513217adefc7bfa5eead'),
 'pool':('scripts/models/pcv_release_pooling.py','db42705b880f1fb81be267d3fa1ac361bb2ceedd'),
}


def expand_schema(path: Path) -> list[dict]:
    """Expand the supplied 19-row Option B schema, not a hand-authored substitute."""
    result=[]
    with Path(path).open(newline='',encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            if row['Type'] not in {'Fixed','Slotted'}:raise ValueError('Unknown schema row type')
            values=[''] if row['Type']=='Fixed' else [row[f'Value {n}'].strip() for n in (1,2,3)]
            for variation in (1,2,3):
                template=row[f'Variation {variation}'].strip()
                for value in values:
                    phrase=re.sub(r'\{[^{}]+\}',lambda _:value,template)
                    result.append(dict(intent=row['Label'],value=value,phrase=phrase,
                        semantic_key=row['Label']+'|'+value.lower(),variation=variation))
    if len({r['phrase'].casefold() for r in result})!=len(result):raise ValueError('Duplicate phrase')
    return result


def route_source(row: dict, *, allow_optional: bool=False) -> str:
    """Return the publisher's reservation; internal train/dev/calibration comes later."""
    repo=row.get('repo')
    if repo!=PRIMARY and not (allow_optional and repo==OPTIONAL):raise ValueError('Unapproved audio repository')
    config=row.get('config','default');split=row.get('published_split')
    if config=='supplemental_synth':
        split=row.get('voice_split')
        if split not in {'train','test','holdout'}:raise ValueError('Missing/invalid reserved voice_split')
    elif config not in {'default','synthetic_negatives'}:raise ValueError('Unreviewed dataset config')
    if config=='default' and split=='numerals':return 'auxiliary'
    if split not in {'train','test','holdout'}:raise ValueError('Missing/invalid publisher split')
    return split


def split_components(rows: list[dict]) -> list[list[str]]:
    """Fail on PCM, known speaker, or derived-parent components spanning splits.

    `parents` must already be resolved to manifest IDs from source_files metadata.
    An unresolved reference is not assumed safe. Unknown speakers must be marked
    unknown upstream; they cannot establish a speaker-disjoint claim.
    """
    items={r['id']:r for r in rows}
    if len(items)!=len(rows):raise ValueError('Duplicate ID')
    parent={k:k for k in items}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(a,b):parent[find(a)]=find(b)
    seen={}
    for r in rows:
        for name in ('pcm','speaker_key'):
            key=(name,r.get(name))
            if not key[1]:continue
            if key in seen:union(r['id'],seen[key])
            else:seen[key]=r['id']
        for p in r.get('parents',[]):
            if p not in items:raise ValueError('Unresolved derived parent: '+str(p))
            union(r['id'],p)
    groups={}
    for k in items:groups.setdefault(find(k),[]).append(k)
    for ids in groups.values():
        if len({items[k]['split'] for k in ids})>1:raise ValueError('Cross-split component leak: '+','.join(ids))
    return list(groups.values())


def cp_upper(k: int,n: int,alpha: float=.05) -> float | None:
    """One-sided exact binomial upper bound. Not a speaker-cluster interval."""
    if type(k) is not int or type(n) is not int or not 0<=k<=n:raise ValueError('Invalid binomial counts')
    if not 0<alpha<1:raise ValueError('Invalid alpha')
    if n==0:return None
    if k==n:return 1.
    if k==0:return -math.expm1(math.log(alpha)/n)
    from scipy.stats import beta
    return float(beta.ppf(1-alpha,k+1,n-k))


def gate(*,exact:int,positive_n:int,wrong_accepted:int,accepted:int,fa:int,
         negative_n:int,human_intents:int,pi_verified:bool) -> dict:
    """Minimal release gate: counts refer to human end-to-end complete commands.

    wrong_accepted includes *all* incorrect accepted outputs, including OOS fires.
    This does not replace per-intent, schema-coverage or provenance qualification.
    """
    counts=(exact,positive_n,wrong_accepted,accepted,fa,negative_n,human_intents)
    if any(type(v) is not int or v<0 for v in counts):raise ValueError('Invalid count')
    if exact>positive_n or accepted!=exact+wrong_accepted or fa>negative_n or fa>wrong_accepted:
        raise ValueError('Inconsistent evaluation counts')
    recall=exact/positive_n if positive_n else None
    precision=exact/accepted if accepted else None
    upper=cp_upper(fa,negative_n)
    precision_lower=None if not accepted else 1-cp_upper(wrong_accepted,accepted)
    checks={'recall':recall is not None and recall>=.90,
            'precision_lower95':precision_lower is not None and precision_lower>=.99,
            'far_upper95':upper is not None and upper<=.01,
            'human_intent_coverage':human_intents==19,'physical_pi':pi_verified is True}
    return dict(qualified=all(checks.values()),checks=checks,recall=recall,precision=precision,
                precision_lower95=precision_lower,far_upper95=upper,
                limitation='Per-clip binomial bounds; correlated voices need grouped intervals too.')


def ctc_min_frames(tokens:list[int]) -> int:
    return len(tokens)+sum(a==b for a,b in zip(tokens,tokens[1:]))


def source_bytes(key: str) -> bytes:
    relative,expected=REFERENCES[key];path=ROOT/relative
    if not path.exists():path=ROOT/'.cache/reference'/Path(relative).name
    b=path.read_bytes()
    actual=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
    if actual!=expected:raise ValueError('Reference differs from pinned base; create a new audit version: '+relative)
    return b


def load_reference(kind: str) -> dict:
    import torch
    from torch import nn
    from einops import rearrange,reduce,repeat
    ns=dict(np=np,torch=torch,nn=nn,math=math,rearrange=rearrange,reduce=reduce,repeat=repeat,
            INTENTS=list(range(18)),canonical=lambda v:json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False),
            __name__='strict_audit_reference')
    specs=[('decoder',{'PrefixCTC','rank_commands'})] if kind=='decoder' else [
           ('context',{'ContextBlock','ContextualVCM'}),('pool',{'PooledContextVCM'})]
    if kind not in {'decoder','model'}:raise ValueError('Unknown probe reference')
    for key,names in specs:
        parsed=ast.parse(source_bytes(key),filename=REFERENCES[key][0])
        body=[n for n in parsed.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
        if {n.name for n in body}!=names:raise ValueError('Missing reference definition')
        exec(compile(ast.Module(body=body,type_ignores=[]),REFERENCES[key][0],'exec'),ns)
    return ns
