import math

import pytest
import torch

from derivopt.operators import SpectralBasis, SpectralResampler
from derivopt.fields import primary_candidates, shared_candidates


@pytest.mark.parametrize("boundary", ["periodic", "neumann", "robin"])
def test_roundtrip_and_parseval(boundary):
    basis = SpectralBasis((9, 8), (2.0, 3.0), boundary, robin_coefficients=((0, 1), (2, 0)) if boundary == "robin" else None)
    state = torch.randn(2, 3, 9, 8, dtype=torch.float64, requires_grad=True)
    coefficients = basis.analysis(state)
    decoded = basis.synthesis(coefficients)
    torch.testing.assert_close(decoded, state, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(coefficients.abs().square().sum(), state.square().sum())
    decoded.square().sum().backward()
    torch.testing.assert_close(state.grad, 2 * state.detach())


def test_periodic_derivative_respects_length_and_zero_mode():
    n, length = 24, 3.0
    basis = SpectralBasis((n,), (length,))
    x = torch.arange(n, dtype=torch.float64) * length / n
    wave = 2 * math.pi * 3 / length
    state = (2 + torch.sin(wave * x))[None, None]
    torch.testing.assert_close(basis.derivative(state, 0), (wave * torch.cos(wave * x))[None, None], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(basis.spectral_power(state, 1), (wave**2 * torch.sin(wave * x))[None, None], atol=1e-10, rtol=1e-11)


def test_neumann_cosine_and_not_periodic_derivative():
    n, length, mode = 17, 2.5, 3
    basis = SpectralBasis((n,), (length,), "neumann")
    x = (torch.arange(n, dtype=torch.float64) + 0.5) * length / n
    state = torch.cos(mode * math.pi * x / length)[None, None]
    torch.testing.assert_close(basis.spectral_power(state, 0.5), state * (mode * math.pi / length), atol=1e-12, rtol=1e-12)
    with pytest.raises(ValueError, match="not diagonal"):
        basis.derivative(state, 0)


def test_robin_positive_operator_and_explicit_coefficients():
    with pytest.raises(ValueError, match="explicit"):
        SpectralBasis((8,), boundary="robin")
    basis = SpectralBasis((8,), (1.0,), "robin", robin_coefficients=(2.0, 0.0))
    state = torch.randn(2, 1, 8, dtype=torch.float64)
    torch.testing.assert_close(basis.spectral_power(state, 1), state @ basis.axis_operators[0].T, atol=2e-12, rtol=2e-12)
    assert basis.eigenvalues.min() > 0
    neumann_limit = SpectralBasis((8,), (1.0,), "robin", robin_coefficients=(0.0, 0.0))
    torch.testing.assert_close(neumann_limit.spectral_power(torch.ones_like(state), 1), torch.zeros_like(state), atol=1e-12, rtol=0)
    torch.testing.assert_close(neumann_limit.spectral_power(torch.ones_like(state), 0.5), torch.zeros_like(state), atol=1e-12, rtol=0)


def test_reaction_imbalance_is_normalized():
    basis = SpectralBasis((8, 9), boundary="neumann")
    state = torch.randn(2, 2, 8, 9, dtype=torch.float64)
    imbalance = next(c for c in primary_candidates("diffusion_reaction", basis) if c.name == "imbalance")
    torch.testing.assert_close(imbalance.evaluate(state, basis), (state[:, :1] - state[:, 1:]) / math.sqrt(2), atol=2e-12, rtol=2e-12)


def test_curl_divergence_and_streamfunction_means():
    basis = SpectralBasis((16, 18), (2 * math.pi, 2 * math.pi))
    x, y = torch.meshgrid(torch.arange(16, dtype=torch.float64) * 2 * math.pi / 16, torch.arange(18, dtype=torch.float64) * 2 * math.pi / 18, indexing="ij")
    psi = torch.sin(x) * torch.sin(y)
    vx, vy = torch.sin(x) * torch.cos(y), -torch.cos(x) * torch.sin(y)
    state = torch.stack((vx + 2, vy - 3))[None]
    candidates = {c.name: c for c in primary_candidates("incompressible_ns", basis, include_optional=True)}
    torch.testing.assert_close(candidates["vorticity"].evaluate(state, basis), (2 * psi)[None, None], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(candidates["streamfunction"].evaluate(state, basis), psi[None, None], atol=1e-12, rtol=1e-12)
    comp = torch.cat([torch.ones(1, 1, 16, 18, dtype=torch.float64), state, torch.ones(1, 1, 16, 18, dtype=torch.float64)], dim=1)
    cns = {c.name: c for c in primary_candidates("compressible_ns", basis)}
    torch.testing.assert_close(cns["divergence"].evaluate(comp, basis), torch.zeros_like(comp[:, :1]), atol=1e-12, rtol=0)


@pytest.mark.parametrize("count,vector,expected,boundary", [(1, (), 9, "periodic"), (2, (), 20, "neumann"), (2, ((0, 1),), 22, "periodic"), (4, ((1, 2),), 42, "periodic")])
def test_shared_library_counts_and_component_dct(count, vector, expected, boundary):
    basis = SpectralBasis((8, 8), boundary=boundary)
    candidates = shared_candidates(basis, tuple(f"q{i}" for i in range(count)), vector_groups=vector,
                                   retained_cutoff=basis.eigenvalues.sqrt().max())
    assert len(candidates) == expected
    assert all(c.n_components == 1 for c in candidates)
    if count > 1:
        rotation = torch.cat([c.matrix[:1] for c in candidates if c.name.startswith("component_dct")], dim=1)[0]
        torch.testing.assert_close(rotation @ rotation.mH, torch.eye(count, dtype=rotation.dtype))


def test_shared_band_partition_and_rms_scaling():
    basis = SpectralBasis((10, 11), (1.0, 2.0))
    masks = torch.stack([basis.band_mask(i / 4, (i + 1) / 4) for i in range(4)])
    assert torch.all(masks.sum(0) == 1)
    candidate = shared_candidates(basis, ("u",), retained_cutoff=basis.eigenvalues.sqrt().max())[0]
    torch.testing.assert_close(candidate.scaled(torch.tensor([2.0])).matrix, candidate.matrix / 2)
    with pytest.raises(ValueError, match="positive"):
        candidate.scaled(torch.tensor([0.0]))


def test_float32_candidate_keeps_predictor_precision_and_gradients():
    basis = SpectralBasis((8, 9), boundary="neumann")
    candidate = primary_candidates("diffusion_reaction", basis)[-1]
    state = torch.randn(2, 2, 8, 9, dtype=torch.float32, requires_grad=True)
    output = candidate.evaluate(state, basis)
    assert output.dtype == state.dtype
    output.square().mean().backward()
    assert state.grad is not None and torch.isfinite(state.grad).all()


def test_coordinate_constructor_checks_layout_spacing():
    coordinates = (torch.arange(8, dtype=torch.float64) + 0.5) * 0.2
    basis = SpectralBasis.from_coordinates((coordinates,), "neumann")
    assert basis.lengths == pytest.approx((1.6,))
    with pytest.raises(ValueError, match="uniform"):
        SpectralBasis.from_coordinates((torch.tensor([0.0, 0.1, 0.4]),))


def test_nyquist_derivative_is_real_zero():
    basis = SpectralBasis((8,))
    state = ((-1.0) ** torch.arange(8, dtype=torch.float64))[None, None]
    torch.testing.assert_close(basis.derivative(state, 0), torch.zeros_like(state), atol=1e-12, rtol=0)


@pytest.mark.parametrize("boundary,coefficients", [("periodic", None), ("neumann", None), ("robin", (0.0, 0.0))])
def test_spectral_resampler_constant_physical_amplitude(boundary, coefficients):
    fine = SpectralBasis((24,), (2.0,), boundary, robin_coefficients=coefficients)
    resampler = SpectralResampler(fine, (9,))
    state = torch.full((2, 3, 24), 3.25, dtype=torch.float64)
    torch.testing.assert_close(resampler.coarsen(state), torch.full((2, 3, 9), 3.25, dtype=torch.float64), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(resampler.decode(resampler.coarsen(state)), state, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("boundary", ["periodic", "neumann", "robin"])
def test_spectral_resampler_low_modes_roundtrip_and_gradient(boundary):
    fine = SpectralBasis((16, 18), (2.0, 3.0), boundary, robin_coefficients=((1, 0), (2, 0)) if boundary == "robin" else None)
    resampler = SpectralResampler(fine, (8, 9))
    raw = torch.randn(2, 3, 16, 18, dtype=torch.float64, requires_grad=True)
    mask = resampler.expressible_mask
    if boundary == "periodic":
        # Strict interior modes avoid the sampling ambiguity of a sine at
        # exactly the coarse Nyquist boundary of the closed retained ball.
        mask = mask & (fine.eigenvalues.sqrt() < resampler.retained_cutoff)
    state = fine.synthesis(fine.analysis(raw) * mask)
    decoded = resampler.decode(resampler.coarsen(state))
    torch.testing.assert_close(decoded, state, atol=3e-12, rtol=3e-12)
    decoded.square().mean().backward()
    assert raw.grad is not None and torch.isfinite(raw.grad).all() and raw.grad.abs().sum() > 0


def test_spectral_coarsen_evaluates_retained_periodic_mode_and_rejects_alias():
    fine = SpectralBasis((32,), (3.0,))
    resampler = SpectralResampler(fine, (12,))
    x = torch.arange(32, dtype=torch.float64) * 3 / 32
    xc = torch.arange(12, dtype=torch.float64) * 3 / 12
    low = (1 + torch.cos(2 * math.pi * 3 * x / 3))[None, None]
    high = torch.cos(2 * math.pi * 9 * x / 3)[None, None]
    expected = (1 + torch.cos(2 * math.pi * 3 * xc / 3))[None, None]
    torch.testing.assert_close(resampler.coarsen(low + high), expected, atol=3e-14, rtol=3e-14)
    # Naive stride/interpolation would alias frequency 9 to frequency -3.
    torch.testing.assert_close(resampler.coarsen(high), torch.zeros_like(expected), atol=3e-14, rtol=0)


def test_spectral_coarsen_evaluates_neumann_cosine_at_coarse_centres():
    fine = SpectralBasis((24,), (2.0,), "neumann")
    resampler = SpectralResampler(fine, (10,))
    xf = (torch.arange(24, dtype=torch.float64) + 0.5) * 2 / 24
    xc = (torch.arange(10, dtype=torch.float64) + 0.5) * 2 / 10
    low = torch.cos(3 * math.pi * xf / 2)[None, None]
    high = torch.cos(14 * math.pi * xf / 2)[None, None]
    expected = torch.cos(3 * math.pi * xc / 2)[None, None]
    torch.testing.assert_close(resampler.coarsen(low + high), expected, atol=2e-14, rtol=2e-14)


def test_coarse_nyquist_is_in_the_closed_ball_and_unchanged_axes_are_lossless():
    fine = SpectralBasis((16,))
    resampler = SpectralResampler(fine, (8,))
    assert resampler.expressible_mask[4] and resampler.expressible_mask[-4]
    state = torch.randn(2, 1, 16, dtype=torch.float64)
    identity = SpectralResampler(fine, (16,))
    assert identity.expressible_mask.all()
    torch.testing.assert_close(identity.decode(identity.coarsen(state)), state)
