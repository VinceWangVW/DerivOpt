import itertools

import pytest
import torch

from derivopt.selectors import DesignProblem
from derivopt.selectors_relaxation import convex_reference


def test_fractional_reference_can_have_a_real_integrality_gap():
    identity = torch.eye(2, dtype=torch.float64)[None]
    zero = torch.zeros_like(identity)
    first = torch.diag(torch.tensor([9., 0.], dtype=torch.float64))[None]
    second = first.flip((-2, -1))
    choices = [[(0, 0, zero), (1, 1, first)], [(0, 0, zero), (1, 1, second)]]
    result = convex_reference(choices, 1, identity, torch.ones(1), first)
    assert result.risk == pytest.approx(2 / 5.5)
    assert result.tangent_lower_bound == pytest.approx(2 / 5.5)
    assert result.best_integer_risk == pytest.approx(1.1)
    assert sum(result.best_integer_bits) <= 1
    # A fractional reference is not an exact integer design.
    assert result.risk < result.best_integer_risk
    assert 1 <= len(result.references) <= 4
    assert all(torch.linalg.eigvalsh(reference).min() >= -1e-12 for reference in result.references)


@pytest.mark.parametrize("seed", range(4))
def test_convex_reference_tangent_bounds_enumerated_noncommuting_problem(seed):
    torch.manual_seed(seed)
    dimension, modes = 3, 2
    p = torch.randn(modes, dimension, dimension, dtype=torch.complex128)
    prior = p @ p.mH
    precisions = []
    for _ in range(3):
        candidate = {}
        for width in (1, 2):
            a = torch.randn(modes, dimension, dimension, dtype=torch.complex128)
            candidate[width] = a @ a.mH
        precisions.append(candidate)
    problem = DesignProblem(prior, precisions, [1, 2, 1], ["a", "b", "c"])
    budget = 3
    allocations = [bits for bits in itertools.product(range(3), repeat=3) if problem.cost(bits) <= budget]
    optimum = min(allocations, key=problem.risk)
    zero = torch.zeros_like(prior)
    options = [[(0, 0, zero)] + [(bits, count * bits, matrix) for bits, matrix in candidate.items()]
               for count, candidate in zip(problem.component_counts, problem.whitened)]
    result = convex_reference(options, budget, problem.gram, problem.weights,
                              problem.information(optimum), max_iterations=12)
    assert result.risk <= problem.risk(optimum) + 1e-10
    assert result.tangent_lower_bound <= problem.risk(optimum) + 1e-10
    assert problem.cost(result.best_integer_bits) <= budget
    assert result.best_integer_risk == pytest.approx(problem.risk(result.best_integer_bits))
    assert torch.linalg.eigvalsh(result.information).min() >= -1e-10


def test_reference_requires_zero_options_and_positive_iteration_budget():
    gram = torch.ones(1, 1, 1, dtype=torch.float64)
    with pytest.raises(ValueError, match="zero"):
        convex_reference([[(1, 1, gram)]], 1, gram, torch.ones(1), gram)
    with pytest.raises(ValueError, match="max_iterations"):
        convex_reference([[(0, 0, gram * 0), (1, 1, gram)]], 1, gram, torch.ones(1), gram,
                         max_iterations=0)


def test_numerical_optimizer_failure_keeps_a_valid_reference(monkeypatch):
    identity = torch.eye(2, dtype=torch.float64)[None]
    zero = torch.zeros_like(identity)
    first = torch.diag(torch.tensor([9., 0.], dtype=torch.float64))[None]
    second = first.flip((-2, -1))
    options = [[(0, 0, zero), (1, 1, first)], [(0, 0, zero), (1, 1, second)]]

    def fail(*args, **kwargs):
        raise FloatingPointError("synthetic optional optimizer failure")

    monkeypatch.setattr("derivopt.selectors_relaxation.minimize", fail)
    result = convex_reference(options, 1, identity, torch.ones(1), first)
    torch.testing.assert_close(result.information, first)
    assert result.risk == pytest.approx(1.1)
    assert not hasattr(result, "exact")
    assert result.tangent_lower_bound <= 1.1
