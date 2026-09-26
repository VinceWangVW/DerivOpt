import torch

from derivopt.metrics import build_fine_mask, detail_horizon, detail_horizon_steps
from derivopt.operators import SpectralBasis, SpectralResampler


def test_horizon_steps_and_normalized_horizon_are_distinct():
    passes = torch.ones(10, 101, dtype=torch.bool)
    passes[:9, 0] = False
    steps = detail_horizon_steps(passes)
    normalized = detail_horizon(passes)
    assert float(steps.double().mean()) == 10  # Can exceed the input pass fraction.
    torch.testing.assert_close(normalized, steps.double()/100)
    assert float(normalized.mean()) == .1
    assert float(passes[:, 0].double().mean()) == .1
    assert bool((normalized <= passes[:, 0]).all())


def test_paper_fine_cutoff_is_not_max_observed_mode():
    basis = SpectralBasis((256, 256), (2*torch.pi, 2*torch.pi))
    sampler = SpectralResampler(basis, (32, 32))
    fine = build_fine_mask(basis, sampler.expressible_mask, cutoff=sampler.retained_cutoff)
    radius = basis.eigenvalues.sqrt()
    expected = sampler.expressible_mask & (radius > 12) & (radius <= 16)
    assert torch.equal(fine, expected)
    assert not bool((fine & (radius > 16)).any())
