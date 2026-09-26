"""Regressions against the paper's projector/sample/quantize/interpolate order."""
import math

import pytest
import torch

from derivopt.calibration import CalibratedState, calibrate
from derivopt.fields import primary_candidates, small_expert_candidates
from derivopt.geometry import Geometry
from derivopt.metrics import compute_metrics
from derivopt.metrics import build_fine_mask
from derivopt.operators import SpectralBasis, SpectralResampler
from derivopt.simulation import BudgetedSimulator


def test_canonical_radial_encoder_rejects_a_representable_corner():
    basis = SpectralBasis((64, 64), (2 * math.pi, 2 * math.pi))
    resampler = SpectralResampler(basis, (16, 16))
    fine_x, fine_y = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
    # (6,6) is present on a 16x16 coarse lattice, but is outside |k|<=8.
    corner = torch.cos(2 * math.pi * (6 * fine_x.double() + 6 * fine_y.double()) / 64)[None, None]
    assert resampler.retained_cutoff == 8
    assert not resampler.expressible_mask[6, 6]
    assert resampler.interpolation_mask[6, 6]
    torch.testing.assert_close(resampler.coarsen(corner), torch.zeros(1, 1, 16, 16, dtype=torch.float64), atol=3e-14, rtol=0)
    coarse_x, coarse_y = torch.meshgrid(torch.arange(16), torch.arange(16), indexing="ij")
    coarse_corner = torch.cos(2 * math.pi * (6 * coarse_x.double() + 6 * coarse_y.double()) / 16)[None, None]
    torch.testing.assert_close(resampler.decode(coarse_corner), corner, atol=2e-14, rtol=0)


def test_canonical_closed_ball_and_fine_band_match_paper_lattice_counts():
    basis = SpectralBasis((256, 256), (2 * math.pi, 2 * math.pi))
    resampler = SpectralResampler(basis, (32, 32))
    frequencies = torch.fft.fftfreq(256, d=1 / 256).double()
    kx, ky = torch.meshgrid(frequencies, frequencies, indexing="ij")
    radius2 = kx.square() + ky.square()
    paper_retained = radius2 <= 16 ** 2
    paper_fine = paper_retained & (radius2 > 12 ** 2)
    actual_fine = build_fine_mask(basis, resampler.expressible_mask)
    assert torch.equal(resampler.expressible_mask, paper_retained)
    assert torch.equal(actual_fine, paper_fine)
    assert int(resampler.expressible_mask.sum()) == 797
    assert int(actual_fine.sum()) == int((actual_fine & paper_fine).sum()) == 356


def test_nyquist_cosine_survives_sampling_but_sine_is_unidentifiable():
    basis = SpectralBasis((32,), (2 * math.pi,))
    resampler = SpectralResampler(basis, (8,))
    x = torch.arange(32, dtype=torch.float64) * 2 * math.pi / 32
    cosine, sine = torch.cos(4 * x)[None, None], torch.sin(4 * x)[None, None]
    expected_samples = ((-1.) ** torch.arange(8, dtype=torch.float64))[None, None]
    torch.testing.assert_close(resampler.coarsen(cosine), expected_samples, atol=3e-15, rtol=3e-15)
    torch.testing.assert_close(resampler.decode(resampler.coarsen(cosine)), cosine, atol=3e-15, rtol=3e-15)
    torch.testing.assert_close(resampler.coarsen(sine), torch.zeros_like(expected_samples), atol=3e-15, rtol=0)
    torch.testing.assert_close(resampler.project(sine), sine, atol=3e-15, rtol=3e-15)
    # Full 1D interpolation and retained support coincide, so quantization
    # leaves the decoded field within the retained band.
    assert torch.equal(resampler.interpolation_mask, resampler.expressible_mask)


@pytest.mark.parametrize("boundary", ["neumann", "robin"])
def test_nonperiodic_radial_cutoff_uses_matched_boundary_frequencies(boundary):
    kwargs = {"robin_coefficients": ((1.0, 2.0), (0.0, 1.0))} if boundary == "robin" else {}
    basis = SpectralBasis((16, 18), (2.0, 3.0), boundary, **kwargs)
    resampler = SpectralResampler(basis, (6, 8))
    expected = min(float(basis.axis_eigenvalues[0][5].sqrt()), float(basis.axis_eigenvalues[1][7].sqrt()))
    assert float(resampler.retained_cutoff) == pytest.approx(expected)
    if boundary == "neumann":
        assert expected == pytest.approx(7 * math.pi / 3)
    assert resampler.interpolation_mask[5, 7]
    assert not resampler.expressible_mask[5, 7]
    coefficients = torch.zeros(1, 1, 16, 18, dtype=torch.float64)
    coefficients[0, 0, 5, 7] = 1
    corner = basis.synthesis(coefficients)
    torch.testing.assert_close(resampler.coarsen(corner), torch.zeros(1, 1, 6, 8, dtype=torch.float64), atol=1e-14, rtol=0)
    coarse = resampler.sample_decoded(corner)
    torch.testing.assert_close(resampler.decode(coarse), corner, atol=1e-14, rtol=0)


@pytest.mark.parametrize("shape,coarse", [((32,), (8,)), ((32, 32), (8, 8)), ((31, 31), (9, 9))])
def test_full_trigonometric_decode_preserves_coarse_samples_including_nyquist(shape, coarse):
    basis = SpectralBasis(shape)
    resampler = SpectralResampler(basis, coarse)
    torch.manual_seed(16)
    samples = torch.randn(2, 3, *coarse, dtype=torch.float64)
    torch.testing.assert_close(resampler.sample_decoded(resampler.decode(samples)), samples, atol=3e-15, rtol=3e-15)
    if len(shape) == 1:
        nyquist = ((-1.0) ** torch.arange(coarse[0], dtype=torch.float64))[None, None]
        expected = torch.cos(2 * math.pi * (coarse[0] // 2) * torch.arange(shape[0], dtype=torch.float64) / shape[0])[None, None]
        torch.testing.assert_close(resampler.decode(nyquist), expected, atol=3e-15, rtol=3e-15)


@pytest.mark.parametrize("method", ["primitive", "archmulti", "rolloutmulti"])
def test_primitive_baselines_retain_real_quantization_leakage_and_skip_posterior(method):
    torch.manual_seed(12)
    basis = SpectralBasis((32, 32), (2 * math.pi, 2 * math.pi))
    resampler = SpectralResampler(basis, (8, 8))
    training = torch.randn(24, 2, 32, 32, dtype=torch.float64)
    calibration = calibrate(training, basis, resampler, primary_candidates("incompressible_ns", basis), (2,), divergence_free=True)
    geometry = Geometry(basis, resampler, torch.zeros(1, 2, 32, 32, dtype=torch.float64), {"family": "incompressible_ns"})
    simulator = BudgetedSimulator(calibration, (2, 0), geometry, method=method,
                                  model_kwargs={"width": 4, "depth": 1, "modes": 2})
    query = torch.randn(2, 2, 32, 32)
    decoded, info = simulator.input_reconstruction(query)
    # Read actual bytes, then carry out the paper's prescribed inverse scaling
    # and interpolation independently of the baseline decoder implementation.
    stored = torch.stack([simulator.explicit_codec.decode(payload)[0] for payload in info["payloads"]])
    primitive_coarse = stored * calibration.rms_scales[0].float()[None, :, None, None]
    expected = resampler.decode(primitive_coarse)
    torch.testing.assert_close(decoded, expected, atol=0, rtol=0)
    leakage = compute_metrics(decoded, query, basis, resampler.expressible_mask)["e_out"]
    assert torch.all(leakage > 1e-5)
    assert not torch.allclose(decoded, calibration.decode_fields([stored], (2, 0)))
    _, step_info = simulator.step(query, actual_codec=True)
    torch.testing.assert_close(step_info["decoded_input_coarse"], primitive_coarse, atol=3e-7, rtol=3e-6)


def test_field_inverse_extension_uses_stored_observations_and_no_synthetic_noise():
    torch.manual_seed(4)
    basis = SpectralBasis((16, 16), (2 * math.pi, 2 * math.pi))
    resampler = SpectralResampler(basis, (8, 8))
    training = torch.randn(8, 2, 16, 16, dtype=torch.float64)
    calibration = calibrate(training, basis, resampler, primary_candidates("incompressible_ns", basis), (2,), divergence_free=True)
    unquantized = [resampler.coarsen(candidate.evaluate(training[:2], basis)) for candidate in calibration.candidates]
    decoded = calibration.decode_fields(unquantized, (2, 2))
    assert torch.all(compute_metrics(decoded, training[:2], basis, resampler.expressible_mask)["e_out"] < 1e-20)
    assert calibration.metadata["out_of_band_decoder"] == "physical_field_least_squares"
    # A known corner mode originates solely in supplied stored observations.
    xy = torch.arange(8, dtype=torch.float64) * 2 * math.pi / 8
    x, y = torch.meshgrid(xy, xy, indexing="ij")
    omega = torch.cos(3 * x + 3 * y)[None, None]
    reconstructed = calibration.decode_fields([omega], (0, 2))
    actual = basis.analysis(reconstructed).flatten(2).transpose(1, 2)
    observation = basis.analysis(resampler.decode(omega)).flatten(2).transpose(1, 2)
    symbol = calibration.candidates[1].matrix
    inverse = torch.linalg.pinv(symbol, rtol=1e-10)
    expected = torch.einsum("mco,bmo->bmc", inverse, observation)
    outside = (resampler.interpolation_mask & ~resampler.expressible_mask).flatten()
    torch.testing.assert_close(actual[:, outside], expected[:, outside], atol=1e-12, rtol=1e-12)
    assert actual[:, outside].abs().max() > 0.1
    primitive = torch.cat((omega, 2 * omega), dim=1)
    reconstructed_primitive = calibration.decode_fields([primitive], (2, 0))
    expected_primitive = resampler.decode(primitive * calibration.rms_scales[0][None, :, None, None])
    actual = basis.analysis(reconstructed_primitive).flatten(2).transpose(1, 2)
    expected = basis.analysis(expected_primitive).flatten(2).transpose(1, 2)
    torch.testing.assert_close(actual[:, outside], expected[:, outside], atol=1e-12, rtol=1e-12)
    legacy = calibration.state_dict()
    assert legacy["format_version"] == 3
    for version in (1, 2):
        legacy["format_version"] = version
        with pytest.raises(ValueError, match="recalibrate"):
            CalibratedState.from_state_dict(legacy)


def test_nonlinear_effective_response_counts_model_error_and_keeps_real_encoder():
    torch.manual_seed(51)
    basis = SpectralBasis((32,))
    resampler = SpectralResampler(basis, (16,))
    positive = torch.randn(12, 1, 32, dtype=torch.float64)
    training = torch.cat((positive, -positive))
    flux = next(candidate for candidate in small_expert_candidates("burgers", basis) if candidate.name == "flux")
    calibration = calibrate(training, basis, resampler, [flux], (32,), shellwise=False)
    candidate = calibration.candidates[0]
    # Paired +/- inputs have identical u^2/2, so their best linear response is
    # zero, although the true nonlinear generated field and residual are not.
    assert candidate.matrix.abs().max() == 0
    assert calibration.noises[0][32][resampler.expressible_mask.flatten()].max() > .01
    expected = training[:2].square() / (2 * calibration.rms_scales[0].reshape(1, 1, 1))
    torch.testing.assert_close(candidate.evaluate(training[:2], basis), expected)
    restored = CalibratedState.from_state_dict(calibration.state_dict())
    torch.testing.assert_close(restored.candidates[0].evaluate(training[:2], basis), expected)
    original_codes = calibration.encode_fields(training[:2], (32,))
    restored_codes = restored.encode_fields(training[:2], (32,))
    torch.testing.assert_close(original_codes[0], restored_codes[0], atol=0, rtol=0)
    assert calibration.reconstruct(training[:2], (32,)).abs().max() < 1e-12
