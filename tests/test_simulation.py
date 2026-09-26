"""Integration tests for physical states, actual budgets and predictor feedback."""

import io

import pytest
import torch

from derivopt.calibration import calibrate
from derivopt.fields import primary_candidates
from derivopt.geometry import Geometry
from derivopt.operators import SpectralBasis, SpectralResampler
from derivopt.simulation import BudgetedSimulator, canonical_method


@pytest.fixture(autouse=True)
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def setup(spatial_dim=1, *, lifting=False):
    torch.manual_seed(3)
    if spatial_dim == 1:
        family, fine_shape, coarse_shape, channels, boundary = "advection", (32,), (16,), 1, "periodic"
    else:
        family, fine_shape, coarse_shape, channels, boundary = "diffusion_reaction", (8, 8), (4, 4), 2, "neumann"
    basis = SpectralBasis(fine_shape, boundary=boundary)
    resampler = SpectralResampler(basis, coarse_shape)
    lift = torch.full((1, channels, *fine_shape), 1.25 if lifting else 0.0, dtype=torch.float64)
    geometry = Geometry(basis, resampler, lift, {"family": family})
    residuals = torch.randn(8, channels, *fine_shape)
    calibration = calibrate(residuals, basis, resampler, primary_candidates(family, basis), (2, 4))
    return calibration, geometry, geometry.to_physical(residuals)


def simulator(calibration, geometry, method="derivopt", backbone="fno", **kwargs):
    bits = [0] * len(calibration.candidates)
    canonical = canonical_method(method)
    if canonical in {"primitive", "archmulti", "rolloutmulti"}:
        bits[0] = 2
    elif canonical == "best_single":
        bits[1] = 2
    elif canonical != "latent_hyper":
        bits[0] = bits[1] = 2
    return BudgetedSimulator(
        calibration, bits, geometry, method=method, backbone=backbone,
        model_kwargs={"width": 4, "depth": 1, "modes": 2, "num_heads": 1, "patch_size": 2},
        latent_kwargs={"width": 4, "latent_channels": 2}, **kwargs,
    )


@pytest.mark.parametrize("spatial_dim", [1, 2])
@pytest.mark.parametrize("backbone", ["unet", "fno", "convlstm", "transformer"])
def test_explicit_two_optimizer_steps_and_real_ledgers(spatial_dim, backbone):
    calibration, geometry, states = setup(spatial_dim)
    model = simulator(calibration, geometry, backbone=backbone)
    before = {name: value.detach().clone() for name, value in model.predictor.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(2):
        model.reset_state()
        optimizer.zero_grad(set_to_none=True)
        predicted, info = model.step(states[:2], actual_codec=False)
        assert predicted.shape == states[:2].shape
        loss = (predicted - states[2:4]).square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.predictor.parameters())
        optimizer.step()
        for payload, ledger in zip(info["payloads"], info["ledgers"]):
            ledger.validate(payload)
            assert ledger.total_bits == 8 * model.cap_bytes
    assert any(not torch.equal(value, before[name]) for name, value in model.predictor.named_parameters())
    assert model.predictor.in_channels == geometry.channels


@pytest.mark.parametrize("method", ["primitive", "best_single", "derivbase", "derivopt", "archmulti", "rolloutmulti", "latent_hyper", "derivopt_archmulti"])
def test_all_method_input_paths_actual_vs_training(method):
    calibration, geometry, states = setup(2, lifting=True)
    model = simulator(calibration, geometry, method=method)
    expected, actual_info = model.input_reconstruction(states[:2], actual_codec=True)
    training_value, training_info = model.input_reconstruction(states[:2], actual_codec=False)
    torch.testing.assert_close(training_value, expected, atol=0, rtol=0)
    assert len(actual_info["payloads"]) == len(training_info["payloads"]) == 2
    assert actual_info["payloads"] == training_info["payloads"]
    assert model.protocol["predictor_input_channels"] == geometry.channels
    if canonical_method(method) == "latent_hyper":
        assert training_info["rate_loss"].requires_grad
        assert all(ledger.hyper_bits > 0 and ledger.primary_bits > 0 for ledger in actual_info["ledgers"])
    else:
        observations = calibration.encode_fields(geometry.to_residual(states[:2]), model.design_bits)
        decoder = (calibration.decode_primitive_fields if method in {"primitive", "archmulti", "rolloutmulti"}
                   else calibration.decode_fields)
        direct = decoder(observations, model.design_bits)
        torch.testing.assert_close(expected, geometry.to_physical(direct), rtol=0, atol=0)


@pytest.mark.parametrize("spatial_dim", [1, 2])
def test_latent_codec_and_predictor_train_under_real_caps(spatial_dim):
    calibration, geometry, states = setup(spatial_dim)
    model = simulator(calibration, geometry, method="latent_hyper")
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        predicted, info = model.step(states[:2], actual_codec=False)
        loss = (predicted - states[2:4]).square().mean() + info["codec_reconstruction_loss"] + 0.01 * info["rate_loss"]
        loss.backward()
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        optimizer.step()
        for payload, ledger in zip(info["payloads"], info["ledgers"]):
            ledger.validate(payload)
            assert len(payload) == model.cap_bytes
    for prefix in ("predictor", "latent_codec.encoder", "latent_codec.decoder", "latent_codec.hyperencoder", "latent_codec.hyperdecoder"):
        assert any(not torch.equal(value, before[name]) for name, value in model.named_parameters() if name.startswith(prefix))


def test_closed_loop_reencodes_model_predictions_not_ground_truth():
    calibration, geometry, states = setup()
    model = simulator(calibration, geometry).eval()
    current = states[:1]
    prior_prediction = None
    records = []
    for _ in range(3):
        prediction, info = model.step(current, actual_codec=True)
        direct_input, _ = model.input_reconstruction(current, actual_codec=True)
        torch.testing.assert_close(info["decoded_input_fine"], direct_input, rtol=0, atol=0)
        if prior_prediction is not None:
            torch.testing.assert_close(current, prior_prediction, rtol=0, atol=0)
        records.extend(info["payloads"])
        current = prior_prediction = prediction
    assert len(records) == 3
    assert torch.isfinite(current).all()


def test_training_codec_preserves_gradient_to_feedback_state():
    calibration, geometry, states = setup()
    model = simulator(calibration, geometry)
    query = states[:1].clone().requires_grad_(True)
    reconstructed, _ = model.input_reconstruction(query, actual_codec=False)
    reconstructed.square().mean().backward()
    assert query.grad is not None and torch.isfinite(query.grad).all() and query.grad.abs().sum() > 0
    detached, _ = model.input_reconstruction(query, actual_codec=True)
    assert not detached.requires_grad


@pytest.mark.parametrize("method", ["derivopt", "latent_hyper", "derivopt_archmulti"])
def test_full_simulator_checkpoint_roundtrip(method):
    calibration, geometry, states = setup(2, lifting=True)
    model = simulator(calibration, geometry, method=method, mean=[0.3, -0.4], std=[0.7, 1.4]).eval()
    with torch.no_grad():
        expected, expected_info = model.step(states[:1], actual_codec=True)
    saved = io.BytesIO()
    torch.save(model.checkpoint_state(), saved)
    saved.seek(0)
    restored = BudgetedSimulator.from_checkpoint_state(torch.load(saved, weights_only=True)).eval()
    with torch.no_grad():
        result, info = restored.step(states[:1], actual_codec=True)
    torch.testing.assert_close(result, expected, atol=0, rtol=0)
    assert info["payloads"] == expected_info["payloads"]
    assert restored.protocol == model.protocol
    torch.testing.assert_close(restored.primitive_std, model.primitive_std)


def test_residual_update_uses_codec_input_and_adds_lifting_once():
    calibration, geometry, states = setup(lifting=True)
    model = simulator(calibration, geometry, prediction_mode="residual", mean=[9.0], std=[2.0])
    with torch.no_grad():
        for parameter in model.predictor.parameters():
            parameter.zero_()
    expected, _ = model.input_reconstruction(states[:1])
    predicted, _ = model.step(states[:1], actual_codec=True)
    # A fine-grid posterior may have a Nyquist sine prior mean that is absent
    # from the coarse nodal predictor interface. Compare its sampled interpolant.
    interpolated = geometry.resampler.decode(geometry.resampler.sample_decoded(geometry.to_residual(expected)))
    torch.testing.assert_close(predicted, geometry.to_physical(interpolated), atol=1e-6, rtol=1e-6)


def test_normalization_is_fixed_and_never_fit_on_query():
    calibration, geometry, states = setup()
    model = simulator(calibration, geometry, mean=[0.3], std=[0.7])
    before_mean, before_std = model.primitive_mean.clone(), model.primitive_std.clone()
    seen = []
    hook = model.predictor.register_forward_pre_hook(lambda _, args: seen.append(args[0].detach().clone()))
    _, info = model.step(states[:1] * 30, actual_codec=True)
    hook.remove()
    torch.testing.assert_close(seen[0], (info["decoded_input_coarse"] - 0.3) / 0.7)
    torch.testing.assert_close(model.primitive_mean, before_mean)
    torch.testing.assert_close(model.primitive_std, before_std)


def test_budget_and_method_mismatch_are_not_silently_repaired():
    calibration, geometry, _ = setup()
    with pytest.raises(ValueError, match="exceeds payload budget"):
        BudgetedSimulator(calibration, [4, 4, 4], geometry, budget_ratio=0.125)
    with pytest.raises(ValueError, match="requires exactly one primitive"):
        BudgetedSimulator(calibration, [0, 2, 0], geometry, method="primitive")
    with pytest.raises(ValueError, match="non-primitive"):
        BudgetedSimulator(calibration, [2, 0, 0], geometry, method="best_single")
    with pytest.raises(ValueError, match="record minimum"):
        BudgetedSimulator(calibration, None, geometry, method="latent_hyper", budget_ratio=0.125)
    with pytest.raises(ValueError, match="both"):
        simulator(calibration, geometry, mean=[0.0])
    with pytest.raises(ValueError, match="positive std"):
        simulator(calibration, geometry, mean=[0.0], std=[0.0])


def test_recurrent_state_reset_is_forwarded_to_predictor():
    calibration, geometry, states = setup()
    model = BudgetedSimulator(
        calibration, [2, 0, 0], geometry, method="archmulti", backbone="convlstm",
        model_kwargs={"width": 4, "depth": 1, "stateful": True},
    )
    first, _ = model.step(states[:1])
    second, _ = model.step(states[:1])
    assert not torch.allclose(first, second)
    model.detach_state()
    assert model.predictor.fine_branch._state[0][0].grad_fn is None
    model.reset_state()
    reset, _ = model.step(states[:1])
    torch.testing.assert_close(first, reset, rtol=0, atol=0)


@pytest.mark.parametrize("method", ["fixedmix_optbits", "perturbedcal", "valsearch"])
def test_unimplemented_method_names_are_rejected(method):
    calibration, geometry, _ = setup()
    with pytest.raises(ValueError, match="Unsupported simulation method"):
        BudgetedSimulator(calibration, [2, 0, 0], geometry, method=method)


@pytest.mark.parametrize("method", ["derivopt", "derivbase", "derivopt_archmulti"])
def test_empty_design_is_prior_only_with_full_padding_and_checkpoint(method):
    calibration, geometry, states = setup(lifting=True)
    model = BudgetedSimulator(
        calibration, [0] * len(calibration.candidates), geometry,
        method=method, backbone="fno", model_kwargs={"width": 4, "depth": 1, "modes": 2},
    )
    assert model.explicit_codec is None
    query = states[:2].clone().requires_grad_(True)
    decoded, info = model.input_reconstruction(query, actual_codec=False)
    coefficients = calibration.prior_mean.transpose(0, 1).reshape(1, geometry.channels, *geometry.basis.shape)
    expected = geometry.to_physical(geometry.basis.synthesis(coefficients).float()).expand_as(query)
    torch.testing.assert_close(decoded, expected, rtol=0, atol=0)
    changed, _ = model.input_reconstruction(query * 100 + 42, actual_codec=True)
    torch.testing.assert_close(decoded, changed, rtol=0, atol=0)
    assert not decoded.requires_grad
    assert info["empty_design"] is True
    for payload, ledger in zip(info["payloads"], info["ledgers"]):
        assert payload == bytes(model.cap_bytes)
        assert ledger.field_bits == ledger.primary_bits == ledger.hyper_bits == ledger.header_bits == 0
        assert ledger.padding_bits == ledger.total_bits == ledger.cap_bits
        ledger.validate(payload)
    before = {name: value.detach().clone() for name, value in model.predictor.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    prediction, _ = model.step(query, actual_codec=False)
    prediction.square().mean().backward()
    assert query.grad is None
    optimizer.step()
    assert any(not torch.equal(before[name], value) for name, value in model.predictor.named_parameters())
    model.eval()
    checkpoint = io.BytesIO()
    torch.save(model.checkpoint_state(), checkpoint)
    checkpoint.seek(0)
    restored = BudgetedSimulator.from_checkpoint_state(torch.load(checkpoint, weights_only=True)).eval()
    assert restored.design_bits == model.design_bits and restored.active_indices == []
    expected_prediction, expected_info = model.step(states[:1], actual_codec=True)
    restored_prediction, restored_info = restored.step(states[:1], actual_codec=True)
    torch.testing.assert_close(restored_prediction, expected_prediction, rtol=0, atol=0)
    assert restored_info["payloads"] == expected_info["payloads"]


@pytest.mark.parametrize("method", ["primitive", "best_single", "archmulti", "rolloutmulti"])
def test_empty_design_cannot_change_nonempty_baseline_definition(method):
    calibration, geometry, _ = setup()
    with pytest.raises(ValueError, match="requires exactly one"):
        BudgetedSimulator(calibration, [0] * len(calibration.candidates), geometry, method=method)


def test_zero_training_covariance_empty_design_is_well_defined():
    _, geometry, states = setup()
    zeros = torch.zeros_like(states)
    calibration = calibrate(
        zeros, geometry.basis, geometry.resampler,
        primary_candidates("advection", geometry.basis), (2, 4),
    )
    model = BudgetedSimulator(calibration, [0] * len(calibration.candidates), geometry)
    decoded, info = model.input_reconstruction(states[:2], actual_codec=True)
    torch.testing.assert_close(decoded, torch.zeros_like(decoded), atol=0, rtol=0)
    assert info["rate_estimate_bits"].sum() == 0
    predicted, _ = model.step(states[:2])
    (predicted - 1).square().mean().backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.predictor.parameters())
