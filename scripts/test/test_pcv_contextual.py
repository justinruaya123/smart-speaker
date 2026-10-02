"""Endpoint-only VCM must mask padding and retain a small deployable graph."""
import torch


def test_contextual_encoder_ignores_batch_padding():
    from scripts.models.pcv_contextual import ContextualVCM
    torch.manual_seed(902)
    model=ContextualVCM(width=16,layers=2,heads=4,feedforward=32).eval()
    x=torch.randn(1,40,31)
    with torch.no_grad():
        alone=model(x,torch.tensor([31]))
        padded=model(torch.nn.functional.pad(x,(0,20)),torch.tensor([31]))
    assert alone[0].shape==(1,40,16)
    for a,b in zip(alone[:3],padded[:3]):
        torch.testing.assert_close(a,b[:,:,:16],atol=2e-6,rtol=2e-6)


def test_contextual_encoder_has_finite_masked_gradients():
    from scripts.models.pcv_contextual import ContextualVCM
    model=ContextualVCM(width=16,layers=2,heads=4,feedforward=32)
    x=torch.randn(2,40,24,requires_grad=True)
    outputs=model(x,torch.tensor([24,15]))
    loss=outputs[0][0].square().mean()+outputs[0][1,:,:8].square().mean()
    loss.backward()
    assert torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad[1,:,15:])==0


def test_contextual_default_parameter_budget():
    from scripts.models.pcv_contextual import ContextualVCM
    model=ContextualVCM()
    assert sum(p.numel() for p in model.parameters())<450000


def test_contextual_export_keeps_dropout_disabled(tmp_path):
    from scripts.models.pcv_contextual import ContextualVCM,export_onnx
    torch.set_num_threads(2);torch.manual_seed(902)
    model=ContextualVCM(width=16,layers=2,heads=4,feedforward=32)
    result=export_onnx(model,tmp_path/'context.onnx')
    assert result['pass']
    assert not model.training
