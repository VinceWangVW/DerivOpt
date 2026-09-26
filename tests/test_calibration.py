import io
import math

import pytest
import torch

from derivopt.calibration import CalibratedState, calibrate
from derivopt.fields import primary_candidates, shared_candidates
from derivopt.operators import SpectralBasis, SpectralResampler
from derivopt.quantization import ScalarQuantizer


def scalar_setup():
    torch.manual_seed(10)
    basis = SpectralBasis((16,), (2.0,))
    resampler = SpectralResampler(basis, (8,))
    training = torch.randn(8, 1, 16, dtype=torch.float64)
    return training, basis, resampler, primary_candidates("advection", basis)


def test_training_only_and_full_covariance_psd():
    torch.manual_seed(2)
    basis = SpectralBasis((8, 8), boundary="neumann")
    resampler = SpectralResampler(basis, (4, 4))
    first = torch.randn(8, 1, 8, 8, dtype=torch.float64)
    training = torch.cat((first, 2 * first + 0.05 * torch.randn_like(first)), dim=1)
    candidates = primary_candidates("diffusion_reaction", basis)
    with pytest.raises(ValueError, match="training split"):
        calibrate(training, basis, resampler, candidates, (2,), split="test")
    model = calibrate(training, basis, resampler, candidates, (2, 4))
    assert torch.linalg.eigvalsh(model.prior).min() >= -1e-12
    mask = resampler.expressible_mask.flatten()
    assert model.prior[mask, 0, 1].abs().mean() > 0.1
    assert model.metadata["split"] == "train"
    assert not model.prior.requires_grad


def test_candidate_generation_precedes_coarsening_and_rms_is_frozen():
    training, basis, resampler, candidates = scalar_setup()
    model = calibrate(training, basis, resampler, candidates, (2, 4))
    bits = (2, 4, 0)
    observed = model.encode_fields(training[:2], bits)
    for output, index in zip(observed, (0, 1)):
        expected = model.quantizers[index](resampler.coarsen(model.candidates[index].evaluate(training[:2], basis)), bits[index])
        torch.testing.assert_close(output, expected)
        field = model.candidates[index].evaluate(training, basis)
        torch.testing.assert_close(field.square().mean().sqrt(), torch.tensor(1.0, dtype=torch.float64))
    # RMS is measured on the generated fine field, including subsequently
    # discarded high modes, not recomputed from test or coarse samples.
    fine = candidates[1].evaluate(training, basis)
    expected_rms = fine.square().mean((0, 2)).sqrt()
    torch.testing.assert_close(model.rms_scales[1], expected_rms)
    before = model.rms_scales[1].clone()
    model.encode_fields(training[:2] * 10, bits)
    torch.testing.assert_close(model.rms_scales[1], before)


def test_posterior_matches_analytic_scalar_formula():
    basis = SpectralBasis((2,), boundary="neumann")
    resampler = SpectralResampler(basis, (2,))
    candidate = shared_candidates(basis, ("u",), retained_cutoff=resampler.retained_cutoff)[0]
    prior = torch.full((2, 1, 1), 2.0, dtype=torch.complex128)
    quantizer = ScalarQuantizer(torch.zeros(1), torch.ones(1), torch.ones(1))
    model = CalibratedState(
        basis, resampler, [candidate], [quantizer], prior, torch.zeros(2, 1, dtype=torch.complex128),
        [{2: torch.full((2, 1, 1), 4.0, dtype=torch.complex128)}],
        [{2: torch.ones(2, dtype=torch.complex128)}], [{2: torch.full((2,), 0.25, dtype=torch.float64)}],
        [{2: torch.zeros(2, 1, dtype=torch.complex128)}], torch.ones(2), [torch.ones(1)], torch.arange(2), {},
    )
    observation = torch.full((3, 1, 2), 3.0, dtype=torch.float64, requires_grad=True)
    result = model.decode_fields([observation], (2,))
    torch.testing.assert_close(result, observation * (2.0 / 2.25))
    result.sum().backward()
    torch.testing.assert_close(observation.grad, torch.full_like(observation, 2.0 / 2.25))
    # Learned observation intercept is subtracted; it is not assumed A*mu.
    model.offsets[0][2][0, 0] = math.sqrt(2)
    torch.testing.assert_close(model.decode_fields([observation], (2,)), (observation - 1) * (2.0 / 2.25))


def test_end_to_end_ste_and_checkpoint_roundtrip():
    training, basis, resampler, candidates = scalar_setup()
    model = calibrate(training, basis, resampler, candidates, (2, 4))
    query = training[:2].float().clone().requires_grad_(True)
    reconstructed = model.reconstruct(query, (2, 4, 0), straight_through=True)
    assert reconstructed.dtype == torch.float32
    reconstructed.square().mean().backward()
    assert query.grad is not None and torch.isfinite(query.grad).all() and query.grad.abs().sum() > 0
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = CalibratedState.from_state_dict(torch.load(buffer, weights_only=True))
    torch.testing.assert_close(restored.reconstruct(query, (2, 4, 0)), reconstructed)
    assert restored.candidate_names == model.candidate_names
    assert restored.component_counts == [1, 1, 1]


def test_incompressible_prior_keeps_transverse_correlation():
    torch.manual_seed(4)
    basis = SpectralBasis((12, 12), (2 * math.pi, 2 * math.pi))
    resampler = SpectralResampler(basis, (8, 8))
    stream = torch.randn(12, 1, 12, 12, dtype=torch.float64)
    training = torch.cat((basis.derivative(stream, 1), -basis.derivative(stream, 0)), dim=1)
    model = calibrate(training, basis, resampler, primary_candidates("incompressible_ns", basis), (2, 4), divergence_free=True)
    k = torch.stack([axis.flatten() for axis in basis.kvectors], dim=-1).to(torch.complex128)
    longitudinal_covariance = torch.einsum("mc,mcd,md->m", k.conj(), model.prior, k).real
    torch.testing.assert_close(longitudinal_covariance, torch.zeros_like(longitudinal_covariance), atol=1e-10, rtol=0)
    assert model.prior[:, 0, 1].abs().max() > 0.01
    output = model.reconstruct(training[:2], (2, 4))
    in_band = resampler.project(output)
    divergence = basis.derivative(in_band[:, :1], 0) + basis.derivative(in_band[:, 1:], 1)
    torch.testing.assert_close(divergence, torch.zeros_like(divergence), atol=1e-6, rtol=0)


def test_shell_pooling_and_precision_match_actual_observation_model():
    training, basis, resampler, candidates = scalar_setup()
    model = calibrate(training, basis, resampler, candidates, (2,), shellwise=True)
    for candidate, transfer, variance, precision in zip(model.candidates, model.transfers, model.noises, model.precisions):
        matrix = candidate.matrix * transfer[2][:, None, None]
        torch.testing.assert_close(precision[2], matrix.mH @ matrix / variance[2][:, None, None])
        for shell in torch.unique(model.shell_ids):
            mask = model.shell_ids == shell
            torch.testing.assert_close(transfer[2][mask], transfer[2][mask][0].expand_as(transfer[2][mask]))
            torch.testing.assert_close(variance[2][mask], variance[2][mask][0].expand_as(variance[2][mask]))
    torch.testing.assert_close(model.design_problem().prior, model.prior)


def test_all_zero_generated_shared_channels_stay_finite():
    basis = SpectralBasis((8, 8), (1.0, 1.0))
    resampler = SpectralResampler(basis, (4, 4))
    training = torch.ones(3, 2, 8, 8, dtype=torch.float64)
    model = calibrate(training, basis, resampler, shared_candidates(
        basis, ("vx", "vy"), retained_cutoff=resampler.retained_cutoff, vector_groups=((0, 1),)), (2,), divergence_free=True)
    assert len(model.candidates) == 22
    assert all(torch.isfinite(p[2]).all() for p in model.precisions)
    bits = (2,) + (0,) * 21
    result = model.reconstruct(training[:1], bits)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, training[:1])


def test_uncalibrated_width_invalid_bits_and_wrong_observation_shape_rejected():
    training, basis, resampler, candidates = scalar_setup()
    model = calibrate(training, basis, resampler, candidates, (2,))
    with pytest.raises(ValueError, match="uncalibrated"):
        model.reconstruct(training[:1], (3, 0, 0))
    with pytest.raises(ValueError, match="integers"):
        model.reconstruct(training[:1], (2.5, 0, 0))
    with pytest.raises(ValueError, match="expected finite shape"):
        model.decode_fields([torch.zeros(1, 1, 3)], (2, 0, 0))


def test_transfer_is_bit_independent_and_high_bit_noise_plateau_is_calibrated():
    training, basis, resampler, candidates = scalar_setup()
    model = calibrate(training, basis, resampler, candidates, (2, 4, 16, 24, 32))
    for transfer, noise, precision in zip(model.transfers, model.noises, model.precisions):
        assert all(torch.equal(transfer[2], transfer[width]) for width in (4, 16, 24, 32))
        # Interior-mode precisions reach the fitted quantization-noise floor.
        # The closed-ball Nyquist boundary also has sampling ambiguity, so its
        # nonzero effective residual need not be bitwise identical at 24/32 bits.
        interior = (basis.eigenvalues.sqrt() < resampler.retained_cutoff).flatten()
        assert torch.equal(noise[24][interior], noise[32][interior])
        assert torch.equal(precision[24][interior], precision[32][interior])
        torch.testing.assert_close(noise[24], noise[32], atol=1e-8, rtol=1e-5)
        assert noise[2][interior].max() > noise[32][interior].max()
    assert model.metadata["transfer_bit_independent"] is True


def test_candidate_specific_budget_filters_only_infeasible_widths():
    torch.manual_seed(8)
    basis = SpectralBasis((6, 6), boundary="neumann")
    resampler = SpectralResampler(basis, (3, 3))
    training = torch.randn(4, 2, 6, 6, dtype=torch.float64)
    model = calibrate(training, basis, resampler, primary_candidates("diffusion_reaction", basis), range(1, 9), budget_per_site=8)
    assert set(model.precisions[0]) == {1, 2, 3, 4}
    assert set(model.precisions[-1]) == set(range(1, 9))
    with pytest.raises(ValueError, match="no feasible"):
        calibrate(training, basis, resampler, primary_candidates("diffusion_reaction", basis), (4,), budget_per_site=2)


def test_wide_bit_widths_are_actually_quantized_and_calibrated():
    training, basis, resampler, candidates = scalar_setup()
    widths = (16, 32, 64, 128)
    model = calibrate(training, basis, resampler, candidates[:1], widths, budget_per_site=128)
    assert set(model.precisions[0]) == set(widths)
    for width in widths:
        output = model.reconstruct(training[:2], (width,))
        assert torch.isfinite(output).all()
        assert torch.isfinite(model.noises[0][width]).all()
    assert all(torch.equal(model.transfers[0][16], model.transfers[0][width]) for width in widths)
