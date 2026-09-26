from itertools import product
from types import SimpleNamespace

import pytest
import torch

from derivopt.metrics import build_fine_mask, detail_horizon, detail_horizon_steps
from derivopt.operators import SpectralBasis, SpectralResampler
from derivopt.protocol import HORIZON_PROTOCOL, HORIZON_UNITS
from derivopt.runner import score_predictions


def test_horizon_steps_and_normalized_horizon_are_distinct():
    passes = torch.ones(10, 101, dtype=torch.bool)
    passes[:9, 0] = False
    steps = detail_horizon_steps(passes)
    normalized = detail_horizon(passes)
    assert float(steps.double().mean()) == 100
    torch.testing.assert_close(normalized, steps.double()/100)
    assert float(normalized.mean()) == 1
    assert float(passes[:, 0].double().mean()) == .1


@pytest.mark.parametrize("length", range(2, 9))
def test_future_horizon_first_prediction_bound_and_no_later_recovery(length):
    passes = torch.tensor(list(product((False, True), repeat=length)))
    expected = []
    for sequence in passes[:, 1:].tolist():
        expected.append(next((index for index, value in enumerate(sequence) if not value), length-1))
    steps = detail_horizon_steps(passes)
    torch.testing.assert_close(steps, torch.tensor(expected))
    normalized = detail_horizon(passes)
    assert bool((normalized <= passes[:, 1]).all())
    assert float(normalized.mean()) <= float(passes[:, 1].double().mean())
    changed_input = passes.clone()
    changed_input[:, 0] = ~changed_input[:, 0]
    torch.testing.assert_close(detail_horizon(changed_input), normalized, rtol=0, atol=0)


@pytest.mark.parametrize("initial_pass", [False, True])
def test_no_predicted_frames_have_zero_horizon(initial_pass):
    passes = torch.tensor([[initial_pass]])
    assert detail_horizon(passes).item() == 0
    assert detail_horizon_steps(passes).item() == 0


def test_scoring_keeps_all_trajectories_and_separates_input_pass_from_future_horizon():
    basis = SpectralBasis((32,))
    sampler = SpectralResampler(basis, (16,))
    geometry = SimpleNamespace(basis=basis, resampler=sampler)
    x = torch.arange(32, dtype=torch.float64) / 32
    field = (torch.sin(4*torch.pi*x) + torch.sin(14*torch.pi*x)).reshape(1, 32)
    truth = field.repeat(11, 1, 1)
    predictions = []
    for index in range(100):
        estimate = torch.zeros_like(truth)
        if index < 3:
            estimate[0] = field
        if index < 19:
            estimate[1:] = field
        predictions.append({"trajectory_id": str(index), "predicted": estimate, "target": truth})
    result = score_predictions(predictions, geometry)
    summary = result["summary"]
    assert summary["trajectory_count"] == 100
    assert summary["input_pass"] == .03
    assert summary["detail_horizon"] == .19
    assert summary["detail_horizon_normalized"] == .19
    assert summary["detail_horizon_steps"] == 1.9
    assert result["horizon_protocol"] == HORIZON_PROTOCOL
    assert result["horizon_units"]["detail_horizon_steps"] == "prediction_steps"
    assert all(row["horizon_protocol"] == HORIZON_PROTOCOL for row in result["rows"])
    assert all(row["horizon_units"] == HORIZON_UNITS for row in result["rows"])


def test_scoring_requires_at_least_one_predicted_frame():
    basis = SpectralBasis((32,))
    geometry = SimpleNamespace(basis=basis, resampler=SpectralResampler(basis, (16,)))
    field = torch.ones(1, 1, 32, dtype=torch.float64)
    with pytest.raises(ValueError, match="at least one predicted frame"):
        score_predictions([{"trajectory_id": "one", "predicted": field, "target": field}], geometry)


def test_paper_fine_cutoff_is_not_max_observed_mode():
    basis = SpectralBasis((256, 256), (2*torch.pi, 2*torch.pi))
    sampler = SpectralResampler(basis, (32, 32))
    fine = build_fine_mask(basis, sampler.expressible_mask, cutoff=sampler.retained_cutoff)
    radius = basis.eigenvalues.sqrt()
    expected = sampler.expressible_mask & (radius > 12) & (radius <= 16)
    assert torch.equal(fine, expected)
    assert not bool((fine & (radius > 16)).any())
