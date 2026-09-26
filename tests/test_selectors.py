import itertools

import pytest
import torch

from derivopt.selectors import DesignProblem, SearchLimitExceeded, best_single, equal_allocation, select_exact, psd_option_envelope


def scalar_problem():
    prior = torch.tensor([[[2.]], [[3.]]], dtype=torch.float64)
    precision = [{b: torch.tensor([[[4.**b]], [[4.**b]]], dtype=torch.float64) for b in range(1, 9)},
                 {b: torch.tensor([[[.1*4.**b]], [[10*4.**b]]], dtype=torch.float64) for b in range(1, 9)}]
    return DesignProblem(prior, precision, [1, 1], ["u", "du"])


def test_scalar_formula():
    problem = scalar_problem()
    expected = 1/(1/2+4**2+.1*4**3)+1/(1/3+4**2+10*4**3)
    assert problem.risk([2, 3]) == pytest.approx(expected)


def test_exact_matches_independent_enumeration():
    problem = scalar_problem()
    budget = 7
    all_allocations = [bits for bits in itertools.product(range(8), repeat=2) if sum(bits) <= budget]
    reference = min(all_allocations, key=lambda bits: (problem.risk(bits), sum(bits), bits))
    selected = select_exact(problem, budget)
    assert selected.bits == reference and selected.exact
    assert select_exact(problem, budget, branch_bound=False).bits == reference
    base = equal_allocation(problem, selected, budget)
    assert base.selected_names == selected.selected_names
    assert len(set(bit for bit in base.bits if bit)) == 1


def test_singular_correlated_prior_preserves_nullspace():
    prior = torch.tensor([[[1., -1.], [-1., 1.]]], dtype=torch.float64)
    problem = DesignProblem(prior, [{1: torch.eye(2).double()[None]}], [2], ["u"])
    covariance = problem.posterior([1])
    torch.testing.assert_close(covariance @ torch.ones(1, 2, 1).double(), torch.zeros(1, 2, 1).double())
    assert problem.risk([1]) == pytest.approx(2/3)


def test_no_uncertified_return_on_resource_limit():
    with pytest.raises(SearchLimitExceeded, match="not certified"):
        select_exact(scalar_problem(), 8, max_nodes=1)


def test_nonmonotone_calibration_remains_exact():
    problem = scalar_problem()
    problem.precisions[0][4] = problem.precisions[0][1]*.1
    problem = DesignProblem(problem.prior, problem.precisions, [1, 1], ["u", "du"])
    assert select_exact(problem, 6).bits == select_exact(problem, 6, branch_bound=False).bits


def test_roundoff_negative_information_is_actually_projected():
    prior = torch.tensor([[[1e16]]], dtype=torch.float64)
    tiny_negative = torch.tensor([[[-1e-8]]], dtype=torch.float64)
    problem = DesignProblem(prior, [{1: tiny_negative}], [1], ["uninformative"])
    assert problem.risk([1]) == pytest.approx(1e16)
    assert torch.equal(problem.precisions[0][1], torch.zeros_like(tiny_negative))


def test_high_bit_and_identical_option_dominance():
    prior = torch.ones(1, 1, 1).double()
    problem = DesignProblem(prior, [{1: prior, 64: prior*2, 128: prior*2}], [1], ["derived"])
    optimum = select_exact(problem, 128)
    assert optimum.bits == (64,)
    assert equal_allocation(problem, optimum, 128).bits == (128,)


def test_zero_information_empty_optimum_but_active_single_control():
    prior = torch.ones(1, 1, 1).double()
    problem = DesignProblem(prior, [{b: prior*0 for b in range(1, 9)}], [1], ["derived"])
    selected = select_exact(problem, 8)
    assert selected.bits == (0,) and equal_allocation(problem, selected, 8).bits == (0,)
    assert best_single(problem, 8, derived_indices=[0]).bits == (1,)


@pytest.mark.parametrize("seed", range(8))
def test_psd_envelope_and_exact_search_against_noncommuting_exhaustive(seed):
    torch.manual_seed(seed)
    dimension, modes = 3, 3
    transform = torch.randn(modes, dimension, dimension, dtype=torch.complex128)
    prior = transform @ transform.mH
    options = []
    for _ in range(3):
        candidate = {}
        for width in range(1, 5):
            # Independently sampled matrices deliberately break monotonicity
            # and simultaneous diagonalization across bit choices.
            matrix = torch.randn(modes, dimension, dimension, dtype=torch.complex128)
            candidate[width] = matrix @ matrix.mH
        envelope = psd_option_envelope(list(candidate.values()))
        assert all(torch.linalg.eigvalsh(envelope - matrix).min() >= -1e-12 for matrix in candidate.values())
        options.append(candidate)
    problem = DesignProblem(prior, options, [1, 2, 1], ["a", "b", "c"], weights=torch.tensor([1.0, 0.0, 0.5]))
    feasible = [bits for bits in itertools.product(range(5), repeat=3) if bits[0] + 2 * bits[1] + bits[2] <= 6]
    reference = min(feasible, key=lambda bits: (problem.risk(bits), problem.cost(bits), bits))
    selected = select_exact(problem, 6)
    exhaustive = select_exact(problem, 6, branch_bound=False)
    assert selected.bits == reference == exhaustive.bits
    assert selected.risk == pytest.approx(problem.risk(selected.bits), rel=1e-13)


def test_psd_envelope_is_tight_for_fixed_direction_precision_curves():
    vector = torch.tensor([[1., 2., -0.5], [-2., 0.3, 1.]], dtype=torch.float64)
    matrix = vector[:, :, None] * vector[:, None, :]
    first = matrix * torch.tensor([2., 5.])[:, None, None]
    second = matrix * torch.tensor([7., 3.])[:, None, None]
    envelope = psd_option_envelope([first, second])
    expected = matrix * torch.tensor([7., 5.])[:, None, None]
    torch.testing.assert_close(envelope, expected, atol=1e-11, rtol=1e-12)
    assert torch.linalg.eigvalsh(envelope - first).min() >= -1e-12
    assert torch.linalg.eigvalsh(envelope - second).min() >= -1e-12


def test_zero_weight_modes_do_not_prevent_exact_option_dominance():
    prior = torch.ones(2, 1, 1, dtype=torch.float64)
    options = {1: torch.tensor([1., 0.]).reshape(2, 1, 1), 2: torch.tensor([1., 100.]).reshape(2, 1, 1)}
    problem = DesignProblem(prior, [options], [1], ["u"], weights=torch.tensor([1., 0.]))
    selected = select_exact(problem, 2)
    assert selected.bits == (1,)
    assert selected.risk == pytest.approx(0.5)


def test_empty_design_remains_feasible_when_active_options_exceed_budget():
    prior = torch.ones(1, 1, 1, dtype=torch.float64)
    problem = DesignProblem(prior, [{4: prior}], [2], ["u"])
    result = select_exact(problem, 1)
    assert result.bits == (0,) and result.risk == 1.0 and result.exact
    assert select_exact(problem, 8, allowed=[]).bits == (0,)
    for invalid in ([0.5], [True], [-1], [1]):
        with pytest.raises(ValueError, match="indices"):
            select_exact(problem, 8, allowed=invalid)
