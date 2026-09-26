"""Backbone shape, gradient, recurrent-state and checkpoint round-trip tests."""

import io

import pytest
import torch

from derivopt.models import ConvLSTM, SpectralConv, make_model


@pytest.fixture(autouse=True)
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("name", ["unet", "fno", "convlstm", "transformer"])
@pytest.mark.parametrize("shape", [(9,), (9, 7)])
def test_train_and_checkpoint_roundtrip(name, shape):
    torch.manual_seed(123)
    options = dict(spatial_dim=len(shape), in_channels=2, out_channels=3, width=8, depth=2, modes=3, num_heads=2, patch_size=4)
    model = make_model(name, **options)
    initial = {key: value.detach().clone() for key, value in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    inputs = torch.randn(2, 2, *shape)
    target = torch.randn(2, 3, *shape)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(inputs)
        assert prediction.shape == target.shape
        loss = (prediction - target).square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
        assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
        assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients)
        optimizer.step()
    assert any(not torch.equal(initial[key], value) for key, value in model.named_parameters() if "weight" in key)
    if name == "fno":
        complex_parameters = [(key, value) for key, value in model.named_parameters() if value.is_complex()]
        assert complex_parameters
        assert any(not torch.equal(initial[key], value) for key, value in complex_parameters)
    assert model.architecture_metadata["configuration"] == "explicit_constructor_settings"
    model.eval()
    with torch.no_grad():
        expected = model(inputs)
    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = make_model(name, **options).eval()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    with torch.no_grad():
        torch.testing.assert_close(restored(inputs), expected, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(1,), (1, 1), (2, 3), (7, 8)])
def test_spectral_modes_are_clipped_without_overlap(shape):
    layer = SpectralConv(len(shape), 1, 2, modes=16)
    inputs = torch.randn(2, 1, *shape, requires_grad=True)
    prediction = layer(inputs)
    assert prediction.shape == (2, 2, *shape)
    prediction.square().mean().backward()
    assert torch.isfinite(inputs.grad).all()


def test_fno_float64_real_layers_preserve_complex_weights():
    model = make_model("fno", 1, 1, 1, width=4, depth=1, modes=3).double()
    result = model(torch.randn(2, 1, 7, dtype=torch.float64))
    assert result.dtype == torch.float64
    assert any(parameter.is_complex() for parameter in model.parameters())
    result.square().mean().backward()


@pytest.mark.parametrize("shape", [(7,), (8,), (7, 8), (8, 7)])
def test_full_mode_spectral_identity(shape):
    layer = SpectralConv(len(shape), 1, 1, modes=16)
    with torch.no_grad():
        layer.weight_positive.fill_(1.0)
        if layer.weight_negative is not None:
            layer.weight_negative.fill_(1.0)
    inputs = torch.randn(2, 1, *shape)
    torch.testing.assert_close(layer(inputs), inputs, atol=1e-6, rtol=1e-6)


def test_spectral_convolution_rejects_unrepresented_frequency():
    layer = SpectralConv(1, 1, 1, modes=2)
    with torch.no_grad():
        layer.weight_positive.fill_(1.0)
    grid = torch.arange(16, dtype=torch.float32)
    inputs = torch.sin(2 * torch.pi * 3 * grid / 16).reshape(1, 1, 16)
    torch.testing.assert_close(layer(inputs), torch.zeros_like(inputs), atol=1e-6, rtol=0)


@pytest.mark.parametrize("shape", [(7,), (7, 5)])
def test_convlstm_recurrence_reset_and_detach(shape):
    torch.manual_seed(14)
    model = ConvLSTM(len(shape), 2, 2, width=4, depth=2, stateful=True)
    inputs = torch.randn(2, 2, *shape)
    first = model(inputs)
    second = model(inputs)
    assert not torch.allclose(first, second)
    model.detach_state()
    assert all(hidden.grad_fn is None and memory.grad_fn is None for hidden, memory in model._state)
    model.reset_state()
    torch.testing.assert_close(model(inputs), first, rtol=0, atol=0)
    with pytest.raises(ValueError, match="reset_state"):
        model(inputs[:1])
    model.reset_state()
    assert model(inputs[:1]).shape[0] == 1
    assert not any("_state" in key for key in model.state_dict())


def test_convlstm_stateless_calls_do_not_share_trajectories():
    model = make_model("convlstm", 1, 1, 1, width=4, depth=1)
    inputs = torch.randn(1, 1, 7)
    torch.testing.assert_close(model(inputs), model(inputs), rtol=0, atol=0)
    assert model._state is None


def test_transformer_token_limit_is_explicit():
    model = make_model("transformer", 2, 1, 1, width=8, num_heads=2, patch_size=2, max_tokens=4)
    with pytest.raises(ValueError, match="needs 9 tokens"):
        model(torch.randn(1, 1, 5, 5))


@pytest.mark.parametrize("name", ["unet", "fno", "convlstm", "transformer"])
def test_bad_input_shape_is_reported(name):
    model = make_model(name, 2, 2, 1, width=8, num_heads=2)
    with pytest.raises(ValueError, match="Expected"):
        model(torch.randn(1, 1, 9, 7))


def test_bad_backbone_or_configuration_is_reported():
    with pytest.raises(ValueError, match="Unknown backbone"):
        make_model("placeholder", 1, 1, 1)
    with pytest.raises(ValueError, match="divisible"):
        make_model("transformer", 2, 1, 1, width=7, num_heads=2)
    with pytest.raises(ValueError, match="spatial_dim"):
        make_model("unet", 3, 1, 1)
