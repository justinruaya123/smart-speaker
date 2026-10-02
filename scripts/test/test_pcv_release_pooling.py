import torch
from scripts.models.pcv_release_pooling import PooledContextVCM


def test_padding_does_not_change_complete_utterance_heads():
    torch.manual_seed(101)
    model=PooledContextVCM(width=32,layers=2,heads=4,feedforward=64,dropout=0).eval()
    short=torch.randn(1,40,39)
    padded=torch.zeros(2,40,62);padded[0,:,:39]=short[0];padded[1]=torch.randn(40,62)
    with torch.no_grad():
        alone=model(short);batch=model(padded,torch.tensor([39,62]))
    for a,b in zip(alone[:3],batch[:3]):
        torch.testing.assert_close(a[0],b[0,:,:20],atol=3e-6,rtol=3e-6)


def test_semantic_pool_is_learnable_and_utterance_constant():
    torch.manual_seed(102)
    model=PooledContextVCM(width=32,layers=1,heads=4,feedforward=64,dropout=0)
    ctc,intent,target,_=model(torch.randn(2,40,39))
    assert ctc.shape==(2,40,20)
    torch.testing.assert_close(intent[:,:,0],intent[:,:,19])
    (intent.square().mean()+target.square().mean()).backward()
    assert model.pool.weight.grad is not None
    assert torch.isfinite(model.pool.weight.grad).all()
    assert model.pool.weight.grad.abs().sum()>0
