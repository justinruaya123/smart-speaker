"""Verify every Parler checkpoint tensor; repair only the diagnosed codec load defect."""
import hashlib
import json
from pathlib import Path

KNOWN_DRIFT={f'audio_encoder.model.decoder.model.0.parametrizations.weight.original{i}' for i in (0,1)}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def mapped_key(key):
    for old,new in (('.weight_g','.parametrizations.weight.original0'),('.weight_v','.parametrizations.weight.original1')):
        if key.endswith(old):return key[:-len(old)]+new
    return key


def repair_and_verify(model,folder):
    """Keep the caller's device/dtype; reject other drift and unexplained tensors."""
    import torch
    from safetensors import safe_open
    folder=Path(folder).resolve();index=folder/'model.safetensors.index.json'
    mapping=json.loads(index.read_text())['weight_map'];state=model.state_dict()
    if not mapping:raise ValueError('Empty checkpoint index')
    names={};shards={}
    for key,shard in mapping.items():
        path=(folder/shard).resolve()
        if not path.is_relative_to(folder) or path.parent!=folder or path.suffix!='.safetensors':raise ValueError('Invalid checkpoint shard path')
        name=mapped_key(key)
        if name in names:raise ValueError('Checkpoint remapping collision')
        if name not in state:raise ValueError('Checkpoint tensor absent from runtime: '+name)
        names[name]=key;shards.setdefault(path,set()).add(key)
    def signature(x):return (x.untyped_storage().data_ptr(),x.storage_offset(),tuple(x.shape),tuple(x.stride()),x.dtype,x.device)
    known={signature(state[k]):k for k in names};aliases={}
    for key,x in state.items():
        if key not in names:
            alias=known.get(signature(x))
            if alias is None:raise ValueError('Runtime tensor absent from checkpoint: '+key)
            aliases[key]=alias
    def tensors():
        for path,keys in shards.items():
            with safe_open(path,framework='pt',device='cpu') as f:
                if set(f.keys())!=keys:raise ValueError('Checkpoint shard/index coverage differs')
                for key in sorted(keys):yield mapped_key(key),f.get_tensor(key)
    codec={};different=[];dtypes={}
    for key,expected in tensors():
        actual=state[key]
        if actual.shape!=expected.shape:raise ValueError('Checkpoint shape differs: '+key)
        if not torch.isfinite(expected).all():raise ValueError('Nonfinite checkpoint tensor: '+key)
        converted=expected.to(dtype=actual.dtype)
        if not torch.isfinite(converted).all():raise ValueError('Nonfinite converted checkpoint tensor: '+key)
        if not torch.equal(actual.detach().cpu(),converted):different.append(key)
        if key.startswith('audio_encoder.'):codec[key.removeprefix('audio_encoder.')]=expected
        dtype=str(actual.dtype);dtypes[dtype]=dtypes.get(dtype,0)+1
    if set(different)-KNOWN_DRIFT:raise ValueError('Unexplained checkpoint loading drift: '+str(sorted(set(different)-KNOWN_DRIFT)))
    if not codec:raise ValueError('No embedded codec checkpoint tensors')
    model.audio_encoder.load_state_dict(codec,strict=True)
    state=model.state_dict()
    for key,expected in tensors():
        if not torch.equal(state[key].detach().cpu(),expected.to(dtype=state[key].dtype)):
            raise ValueError('Checkpoint verification failed after reload: '+key)
    return dict(changed_before=sorted(different),verified_after=True,tensors_checked=len(names),
        verified_aliases=aliases,runtime_dtypes=dtypes,precision_changed=False,
        files={p.name:digest(p) for p in [index,*shards]},source_sha256=digest(__file__))
