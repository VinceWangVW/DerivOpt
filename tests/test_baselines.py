import io

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from derivopt.baselines import ArchMulti
from derivopt.models import make_model


@pytest.fixture(autouse=True)
def small_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("backbone", ("unet", "fno", "convlstm", "transformer"))
@pytest.mark.parametrize("shape", ((9,), (9, 7)))
def test_two_scale_two_steps_and_checkpoint(backbone, shape):
    torch.manual_seed(30)
    def factory(in_channels, out_channels):
        return make_model(backbone, len(shape), in_channels, out_channels,
                          width=4, depth=1, modes=3, patch_size=2, num_heads=2)
    model = ArchMulti(factory, len(shape), in_channels=2, out_channels=3, width=4)
    fine_ids = {id(p) for p in model.fine_branch.parameters()}
    assert not fine_ids & {id(p) for p in model.coarse_branch.parameters()}
    initial = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    inputs = torch.randn(2, 2, *shape, requires_grad=True)
    target = torch.randn(2, 3, *shape)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        inputs.grad = None
        output = model(inputs)
        assert output.shape == target.shape
        loss = F.mse_loss(output, target)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(inputs.grad).all() and inputs.grad.abs().sum() > 0
        for branch in (model.fine_branch, model.coarse_branch, model.fusion):
            gradients = [p.grad for p in branch.parameters()]
            assert all(g is not None and torch.isfinite(g).all() for g in gradients)
            assert any(g.abs().sum() > 0 for g in gradients)
        optimizer.step()
    for prefix in ("fine_branch", "coarse_branch", "fusion"):
        assert any(not torch.equal(initial[name], parameter) for name, parameter in model.named_parameters()
                   if name.startswith(prefix))
    model.eval()
    with torch.no_grad():
        expected = model(inputs)
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    restored = ArchMulti(factory, len(shape), 2, 3, width=4).eval()
    restored.load_state_dict(torch.load(stream, weights_only=True))
    with torch.no_grad():
        torch.testing.assert_close(restored(inputs), expected, rtol=0, atol=0)


@pytest.mark.parametrize("shape", ((7,), (7, 5)))
def test_convlstm_state_forwarding_and_sequence_gradient(shape):
    factory = lambda i, o: make_model("convlstm", len(shape), i, o, width=4, depth=1, stateful=True)
    model = ArchMulti(factory, len(shape), 2, 2, width=4)
    inputs = torch.randn(2, 2, *shape)
    first = model(inputs)
    second = model(inputs)
    assert not torch.allclose(first, second)
    (first.square().mean() + second.square().mean()).backward()
    model.detach_state()
    for branch in (model.fine_branch, model.coarse_branch):
        assert branch._state is not None
        assert all(h.grad_fn is None and c.grad_fn is None for h, c in branch._state)
    model.reset_state()
    assert model.fine_branch._state is None and model.coarse_branch._state is None
    torch.testing.assert_close(model(inputs), first.detach(), rtol=0, atol=0)


@pytest.mark.parametrize("shape", ((1,), (1, 1), (2, 3)))
def test_pooling_preserves_small_edge_cells(shape):
    class Recorder(nn.Module):
        def __init__(self, i, o):
            super().__init__()
            self.conv = (nn.Conv1d if len(shape) == 1 else nn.Conv2d)(i, o, 1)
            self.last_input = None

        def forward(self, x):
            self.last_input = x.detach().clone()
            return self.conv(x)
    model = ArchMulti(Recorder, len(shape), 1, 1, width=2)
    inputs = torch.arange(torch.tensor(shape).prod()).float().reshape(1, 1, *shape) + 1
    output = model(inputs)
    assert output.shape == inputs.shape
    assert torch.equal(model.fine_branch.last_input, inputs)
    pool = F.avg_pool1d if len(shape) == 1 else F.avg_pool2d
    torch.testing.assert_close(model.coarse_branch.last_input,
                               pool(inputs, 2, 2, ceil_mode=True, count_include_pad=False))
    assert model.coarse_branch.last_input.min() >= 1  # no zero-padding dilution


def test_shared_factory_parameters_rejected():
    shared = nn.Conv1d(1, 2, 1)
    with pytest.raises(ValueError, match="independent"):
        ArchMulti(lambda i, o: shared, 1, 1, 1, width=2)


def test_wrong_feature_shape_rejected():
    model = ArchMulti(lambda i, o: nn.Conv1d(i, o+1, 1), 1, 1, 1, width=2)
    with pytest.raises(ValueError, match="feature channels"):
        model(torch.ones(1, 1, 4))
