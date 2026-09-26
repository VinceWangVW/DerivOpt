import math

import pytest
import torch

from derivopt.metrics import (build_fine_mask, compute_metrics, detail_horizon,
                             impute_pilot_failures, input_pass, normalized_rmse)
from derivopt.operators import SpectralBasis, SpectralResampler
from derivopt.protocol import MetricProtocol
from derivopt.reporting import configuration_macro, paired_seed_summary


def setup_bands(boundary="periodic"):
    basis = SpectralBasis((16,), boundary=boundary)
    expressible = SpectralResampler(basis, (10,)).expressible_mask
    fine = build_fine_mask(basis, expressible)
    return basis, expressible, fine


def two_band_state(basis, expressible, fine):
    coefficients = torch.zeros((1, 1, *basis.shape), dtype=torch.float64)
    coefficients[..., expressible & ~fine] = 1
    coefficients[..., fine] = 1
    return basis.synthesis(coefficients.to(torch.complex128) if basis.is_periodic else coefficients)


@pytest.mark.parametrize("boundary", ["periodic", "neumann"])
def test_perfect_prediction_and_explicit_formula(boundary):
    basis, expressible, fine = setup_bands(boundary)
    target = two_band_state(basis, expressible, fine)
    result = compute_metrics(target, target, basis, expressible, fine)
    for name in ("nrmse", "expr_rel", "fine_rel"):
        assert result[name].item() == 0
    assert result["q_fine"].item() == pytest.approx(1)
    assert result["e_out"].item() < 1e-28
    assert result["input_pass"].item()
    assert result["finite_mask"].item()


def test_analytic_uniform_scaling_and_zero_prediction():
    basis, expressible, fine = setup_bands()
    target = two_band_state(basis, expressible, fine)
    result = compute_metrics(1.5 * target, target, basis, expressible, fine)
    for name in ("nrmse", "expr_rel", "fine_rel"):
        assert result[name].item() == pytest.approx(.5)
    assert result["q_fine"].item() == pytest.approx(1)
    zero = compute_metrics(torch.zeros_like(target), target, basis, expressible, fine)
    assert zero["fine_rel"].item() == pytest.approx(1)
    assert math.isnan(zero["q_fine"].item())
    assert not zero["input_pass"].item()


def test_outside_band_leak_and_fine_loss():
    basis, expressible, fine = setup_bands()
    target = two_band_state(basis, expressible, fine)
    coeff = basis.analysis(target)
    lost_detail = basis.synthesis(coeff * (~fine))
    loss = compute_metrics(lost_detail, target, basis, expressible, fine)
    assert loss["fine_rel"].item() == pytest.approx(1)
    assert loss["q_fine"].item() < 1e-28
    leak_coeff = torch.zeros_like(coeff)
    leak_coeff[..., ~expressible] = 2
    leaked = target + basis.synthesis(leak_coeff)
    result = compute_metrics(leaked, target, basis, expressible, fine)
    expected = leak_coeff.abs().square().sum() / coeff[..., expressible].abs().square().sum()
    assert result["e_out"].item() == pytest.approx(expected.item())
    assert result["expr_rel"].item() < 1e-14
    assert not result["input_pass"].item()


def test_zero_energy_is_explicit_and_nan_never_passes():
    basis, expressible, fine = setup_bands()
    zero = torch.zeros(2, 1, 16, dtype=torch.float64)
    result = compute_metrics(zero, zero, basis, expressible, fine)
    for name in ("nrmse", "expr_rel", "fine_rel", "e_out"):
        assert torch.equal(result[name], torch.zeros(2, dtype=torch.float64))
    assert torch.isnan(result["q_fine"]).all()
    assert not result["q_defined"].any()
    assert not result["input_pass"].any()
    assert not result["finite_mask"].any()
    bad = zero.clone()
    bad[0, 0, 0] = float("nan")
    result = compute_metrics(bad, zero, basis, expressible, fine)
    assert not result["input_finite_mask"][0]
    assert torch.isnan(result["nrmse"][0])
    assert not result["input_pass"][0]


def test_constant_neumann_field_has_undefined_fine_energy():
    basis, expressible, fine = setup_bands("neumann")
    constant = torch.ones((1, 1, 16), dtype=torch.float64)
    result = compute_metrics(constant, constant, basis, expressible, fine)
    assert not result["q_defined"].item()
    assert not result["input_pass"].item()


def test_channel_normalization_not_mixed_unit_concatenation():
    target = torch.tensor([[[100., 100.], [1., 1.]]])
    prediction = target.clone()
    prediction[:, 1] = 0
    assert normalized_rmse(prediction, target).item() == pytest.approx(.5)
    assert torch.equal(normalized_rmse(prediction, target, return_channels=True), torch.tensor([[0., 1.]]))


def test_threshold_changes_do_not_modify_fine_rel():
    basis, expressible, fine = setup_bands()
    target = two_band_state(basis, expressible, fine)
    prediction = basis.synthesis(basis.analysis(target) * torch.where(fine, .9, 1.))
    strict = compute_metrics(prediction, target, basis, expressible, fine, protocol=MetricProtocol(tau_q=1.01))
    loose = compute_metrics(prediction, target, basis, expressible, fine, protocol=MetricProtocol(tau_q=2.))
    for name in ("nrmse", "expr_rel", "fine_rel", "q_fine", "e_out"):
        assert torch.equal(strict[name], loose[name])
    assert not strict["input_pass"].item()
    assert loose["input_pass"].item()
    assert torch.equal(input_pass(strict, MetricProtocol(tau_q=2)), loose["input_pass"])


def test_fine_band_retained_relative_and_open_lower_endpoint():
    basis = SpectralBasis((32,))
    expressible = basis.eigenvalues.sqrt() <= 8 * 2 * math.pi
    fine = build_fine_mask(basis, expressible)
    # Exactly six-eighths is excluded; seven and eight are retained.
    assert not fine[6] and fine[7] and fine[8]
    assert not fine[9]
    dc = torch.zeros(32, dtype=torch.bool)
    dc[0] = True
    assert not build_fine_mask(basis, dc).any()
    with pytest.raises(ValueError, match="subset"):
        compute_metrics(torch.ones(1, 1, 32), torch.ones(1, 1, 32), basis, dc, fine)


def test_nonradial_expressible_mask_is_respected():
    basis = SpectralBasis((8, 12))
    mask = SpectralResampler(basis, (4, 8)).expressible_mask
    fine = build_fine_mask(basis, mask)
    assert not (fine & ~mask).any()
    frequency = basis.eigenvalues.sqrt()
    assert torch.equal(fine, mask & (frequency > .75 * frequency[mask].max()))


def test_horizon_is_future_prefix_and_normalized_by_prediction_count():
    passed = torch.tensor([[False, True, True, True], [True, False, True, True],
                           [True, True, False, True], [True, True, True, True]])
    assert torch.allclose(detail_horizon(passed), torch.tensor([1., 0., 1/3, 1.], dtype=torch.float64))
    assert detail_horizon(torch.tensor([[True]])).item() == 0
    with pytest.raises(ValueError):
        detail_horizon(torch.empty(1, 0, dtype=torch.bool))
    with pytest.raises(ValueError):
        detail_horizon(torch.tensor([[1., float("nan")]]))


def test_pilot_imputation_is_separate_and_preserves_raw_metrics():
    errors = torch.tensor([.1, float("nan"), .2])
    horizons = torch.tensor([.3, .4, .2])
    result = impute_pilot_failures(errors, horizons, finite_mask=torch.tensor([True, True, False]))
    assert torch.equal(result["imputed_nrmse"], torch.tensor([.1, 1., 1.]))
    assert torch.equal(result["imputed_horizon"], torch.tensor([.3, 0., 0.]))
    assert result["finite_fraction"].item() == pytest.approx(1/3)
    assert torch.isnan(errors[1]) and horizons[1] == .4


def row(family, method, seed, value):
    return dict(family=family, backbone="fno", budget_ratio=.25, retain_frac=.125,
                method=method, seed=seed, nrmse=value)


def test_configuration_macro_not_dataset_size_weighted():
    rows = [row("small", "derivopt", 0, 1)] + [row("large", "derivopt", 0, 3)] * 100
    # Another seed must not give the large family twice the configuration weight.
    rows += [row("large", "derivopt", 1, 5)]
    summary = configuration_macro(rows, "nrmse")["derivopt"]
    assert summary["mean"] == pytest.approx(2.5)  # mean(1, mean(3,5))
    assert summary["configuration_count"] == 2
    with pytest.raises(ValueError, match="Nonfinite"):
        configuration_macro([row("small", "a", 0, float("nan"))], "nrmse")


def test_paired_seed_sample_sd_not_standard_error_or_population_sd():
    rows = []
    for family in ("advection", "burgers"):
        for seed, difference in enumerate((1., 3., 5.)):
            rows += [row(family, "a", seed, difference), row(family, "b", seed, 0.)]
    result = paired_seed_summary(rows, "nrmse", "a", "b")
    assert result["mean_delta"] == 3
    assert result["sample_sd_delta"] == 2
    assert result["seed_count"] == 3
    with pytest.raises(ValueError, match="coverage"):
        paired_seed_summary(rows[:-1], "nrmse", "a", "b")
    single = paired_seed_summary([row("a", "a", 0, 1), row("a", "b", 0, 2)], "nrmse", "a", "b")
    assert single["sample_sd_delta"] is None
