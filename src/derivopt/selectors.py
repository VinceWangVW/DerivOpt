"""Exact finite-library integer state design under the calibrated Gaussian risk.

The search is exponential in the worst case. Branch-and-bound uses
positive-semidefinite precision upper bounds to certify the calibrated-risk
optimum. Reaching a configured resource limit raises SearchLimitExceeded.
"""
from dataclasses import asdict, dataclass
import math
from typing import Sequence

import torch


def hermitian(matrix):
    return (matrix + matrix.mH) / 2


def psd_sqrt(matrix: torch.Tensor):
    values, vectors = torch.linalg.eigh(hermitian(matrix))
    tolerance = 1e-7 * matrix.abs().amax(dim=(-2, -1)).clamp_min(1)
    if torch.any(values.min(-1).values < -tolerance):
        raise ValueError("covariance is not positive semidefinite")
    return (vectors * values.clamp_min(0).sqrt().unsqueeze(-2)) @ vectors.mH


@dataclass(frozen=True)
class Design:
    bits: tuple[int, ...]
    names: tuple[str, ...]
    risk: float
    field_bits_per_site: int
    evaluated_leaves: int = 0
    visited_nodes: int = 0
    exact: bool = True

    @property
    def selected_names(self):
        return tuple(name for name, bits in zip(self.names, self.bits) if bits > 0)

    def to_dict(self):
        return asdict(self)


class SearchLimitExceeded(RuntimeError):
    """A search stopped without proving optimality; not a completed design."""


class DesignProblem:
    def __init__(self, prior: torch.Tensor, precisions: Sequence[dict[int, torch.Tensor]],
                 component_counts: Sequence[int], names: Sequence[str], weights: torch.Tensor | None = None,
                 target: torch.Tensor | None = None):
        if prior.ndim != 3 or prior.shape[-1] != prior.shape[-2] or not torch.isfinite(prior).all():
            raise ValueError("prior must be finite [mode,latent,latent]")
        if not (len(precisions) == len(component_counts) == len(names)) or not names or len(set(names)) != len(names):
            raise ValueError("candidate descriptors must be nonempty, unique and aligned")
        if any(not isinstance(c, int) or c < 1 for c in component_counts):
            raise ValueError("component counts must be positive integers")
        self.root = psd_sqrt(prior)
        self.prior = hermitian(self.root @ self.root)
        self.names, self.component_counts = tuple(names), tuple(component_counts)
        self.precisions = []
        self.mode_count, self.latent_dim = prior.shape[0], prior.shape[-1]
        self.identity = torch.eye(self.latent_dim, device=prior.device, dtype=prior.dtype).expand_as(prior)
        self.weights = (torch.ones(self.mode_count, dtype=prior.real.dtype, device=prior.device) if weights is None
                        else torch.as_tensor(weights, device=prior.device, dtype=prior.real.dtype).reshape(-1))
        if self.weights.shape != (self.mode_count,) or not torch.isfinite(self.weights).all() or torch.any(self.weights < 0):
            raise ValueError("mode weights must be finite and nonnegative")
        if not torch.any(self.weights > 0):
            raise ValueError("at least one objective mode must have positive weight")
        self.objective_modes = self.weights > 0
        self.target = self.identity if target is None else target.to(prior)
        if self.target.ndim != 3 or self.target.shape[0] != self.mode_count or self.target.shape[-1] != self.latent_dim:
            raise ValueError("target must have shape [mode,target_component,latent]")
        self.gram = self.root @ self.target.mH @ self.target @ self.root
        self.whitened = []
        for options in precisions:
            if not options:
                raise ValueError("each candidate needs calibrated active bit options")
            transformed, projected = {}, {}
            for bits, precision in options.items():
                if not isinstance(bits, int) or isinstance(bits, bool) or bits < 1:
                    raise ValueError("calibrated bit widths must be positive integers")
                if precision.shape != prior.shape or not torch.isfinite(precision).all():
                    raise ValueError("candidate precision shape or finiteness mismatch")
                precision = hermitian(precision.to(prior))
                # Calibration errors may not introduce negative information.
                precision_root = psd_sqrt(precision)
                precision = hermitian(precision_root @ precision_root)
                projected[bits] = precision
                transformed[bits] = hermitian(self.root @ precision @ self.root)
            self.precisions.append(projected)
            self.whitened.append(transformed)

    def _risk_whitened(self, information):
        matrix = (self.identity + hermitian(information))[self.objective_modes]
        gram = self.gram[self.objective_modes]
        if self.latent_dim == 1:
            per_mode = (gram[..., 0, 0] / matrix[..., 0, 0]).real
        else:
            per_mode = torch.linalg.solve(matrix, gram).diagonal(dim1=-2, dim2=-1).sum(-1).real
        result = (self.weights[self.objective_modes] * per_mode).sum()
        if not torch.isfinite(result):
            raise FloatingPointError("non-finite calibrated reconstruction risk")
        if result < -max(1e-12, float(self.weights.sum())*1e-12):
            raise FloatingPointError("negative posterior risk: conditioning failed")
        return float(result.clamp_min(0))

    def information(self, bits):
        if len(bits) != len(self.names):
            raise ValueError("bit allocation length mismatch")
        information = torch.zeros_like(self.prior)
        for index, width in enumerate(bits):
            if not isinstance(width, int) or isinstance(width, bool) or width < 0:
                raise ValueError("bit allocation must contain nonnegative integers")
            if width == 0:
                continue
            if width not in self.whitened[index]:
                raise ValueError(f"uncalibrated bit width {width} for {self.names[index]}")
            information = information + self.whitened[index][width]
        return information

    def risk(self, bits):
        return self._risk_whitened(self.information(bits))

    def posterior(self, bits):
        information = self.information(bits)
        return hermitian(self.root @ torch.linalg.solve(self.identity+information, self.root))

    def cost(self, bits):
        self.information(bits)  # Validate widths before counting their cost.
        return sum(m*b for m, b in zip(self.component_counts, bits))

    def design(self, bits, *, budget=None, exact=True):
        bits = tuple(bits)
        cost = self.cost(bits)
        if budget is not None and cost > budget:
            raise ValueError("allocation exceeds integer per-site budget")
        return Design(bits, self.names, self.risk(bits), cost, exact=exact)


def psd_option_envelope(matrices: Sequence[torch.Tensor]):
    """A Loewner upper bound for mutually exclusive PSD matrix choices.

    At each mode choose R with maximal trace. For every choice P, let
    alpha=tr(P)/tr(R) in [0,1] and E=P-alpha*R. Then
    P <= R + ||E||_F I. Taking the maximum residual norm bounds every choice,
    including noncommuting and nonmonotone precision curves. Proportional
    fixed-transfer curves reduce to their largest per-mode precision.
    """
    if not matrices:
        raise ValueError("An envelope needs at least one matrix")
    stack = torch.stack(tuple(matrices))
    dimension = stack.shape[-1]
    if dimension == 1:
        return stack.real.amax(0).clamp_min(0).to(stack.dtype)
    traces = stack.diagonal(dim1=-2, dim2=-1).sum(-1).real.clamp_min(0)
    maximum, indices = traces.max(0)
    reference = stack[indices, torch.arange(stack.shape[1], device=stack.device)]
    alpha = torch.where(maximum[None] > 0, traces / maximum.clamp_min(torch.finfo(traces.dtype).tiny)[None], 0).clamp(0, 1)
    residual = stack - alpha[..., None, None] * reference[None]
    delta = torch.linalg.matrix_norm(residual, ord="fro").amax(0)
    # Outward roundoff allowance only loosens the analytic upper envelope; it
    # never permits more aggressive pruning or approximates away a candidate.
    magnitude = stack.abs().amax(dim=(0, 2, 3))
    delta = delta + (8 * dimension * dimension + 4) * torch.finfo(traces.dtype).eps * magnitude
    identity = torch.eye(dimension, dtype=stack.dtype, device=stack.device)
    return hermitian(reference + delta[:, None, None] * identity)


def select_exact(problem: DesignProblem, budget: int, *, allowed: Sequence[int] | None = None,
                 max_nodes: int | None = None, branch_bound: bool = True) -> Design:
    """Exhaust the feasible library or prune only with valid PSD bounds.

    A zero allocation denotes an absent candidate. Ties choose lower cost then
    lexicographically smaller allocation, without consulting outcome metrics.
    """
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
        raise ValueError("budget must be a positive integer")
    count = len(problem.names)
    eligible = tuple(range(count) if allowed is None else allowed)
    if any(not isinstance(index, int) or isinstance(index, bool) or index < 0 or index >= count for index in eligible):
        raise ValueError("invalid allowed candidate indices")
    active = set(eligible)
    # Modes of zero objective weight are independent blocks and contribute
    # exactly zero risk. Excluding them preserves the objective; full statistics
    # remain on the problem.
    mask = problem.objective_modes
    zero = torch.zeros_like(problem.prior[mask])
    identity, gram, weights = problem.identity[mask], problem.gram[mask], problem.weights[mask]
    def risk(information):
        matrix = identity + hermitian(information)
        if problem.latent_dim == 1:
            per_mode = (gram[..., 0, 0] / matrix[..., 0, 0]).real
        else:
            per_mode = torch.linalg.solve(matrix, gram).diagonal(dim1=-2, dim2=-1).sum(-1).real
        value = (weights * per_mode).sum()
        if not torch.isfinite(value) or value < -max(1e-12, float(weights.sum()) * 1e-12):
            raise FloatingPointError("invalid posterior risk during exact selection")
        return float(value.clamp_min(0))
    options = []
    for index in range(count):
        feasible = [
            (width, width*problem.component_counts[index], matrix[mask])
            for width, matrix in sorted(problem.whitened[index].items())
            if index in active and width*problem.component_counts[index] <= budget]
        distinct = []
        for option in feasible:
            # Compare information matrices with exact equality. An identical
            # matrix at greater cost cannot win the score and cost tie-break.
            if torch.equal(option[2], zero) or any(torch.equal(option[2], previous[2]) for previous in distinct):
                continue
            distinct.append(option)
        options.append([(0, 0, zero)] + distinct)
    if all(len(candidate) == 1 for candidate in options):
        return Design((0,)*count, problem.names, risk(zero), 0, 1, 1)
    # Seed a valid incumbent from all one-candidate designs. This is only an
    # upper bound; it does not replace the subsequent exact search.
    best_bits, best_score, best_cost = (0,)*count, risk(zero), 0
    for index, candidate in enumerate(options):
        for width, cost, information in candidate[1:]:
            bits = [0]*count
            bits[index] = width
            score = risk(information)
            if (score, cost, tuple(bits)) < (best_score, best_cost, best_bits):
                best_bits, best_score, best_cost = tuple(bits), score, cost
    # A feasible greedy construction is only an incumbent upper bound. The
    # certified search below still visits or safely bounds every feasible
    # allocation; no greedy result is returned as an unproved optimum.
    seed_bits, seed_cost, seed_information = [0] * count, 0, zero.clone()
    seed_score = risk(seed_information)
    seed_matrices = [zero] * count
    while seed_cost < budget:
        improvement = None
        for index, candidate in enumerate(options):
            old_cost = seed_bits[index] * problem.component_counts[index]
            for width, cost, information in candidate[1:]:
                extra = cost - old_cost
                if extra <= 0 or seed_cost + extra > budget:
                    continue
                proposed_information = seed_information - seed_matrices[index] + information
                score = risk(proposed_information)
                if score < seed_score:
                    key = (-(seed_score - score) / extra, score, cost, index, width)
                    if improvement is None or key < improvement[0]:
                        improvement = (key, index, width, extra, information, proposed_information, score)
        if improvement is None:
            break
        _, index, width, extra, information, seed_information, seed_score = improvement
        seed_bits[index], seed_matrices[index] = width, information
        seed_cost += extra
        # Use the canonical field-order sum for incumbent scores as at leaves;
        # do not let subtract/add cancellation manufacture a lower objective.
        seed_information = sum(seed_matrices, zero.clone())
        seed_score = risk(seed_information)
        if (seed_score, seed_cost, tuple(seed_bits)) < (best_score, best_cost, best_bits):
            best_bits, best_score, best_cost = tuple(seed_bits), seed_score, seed_cost
    # Complement the gain-per-bit seed with uniformly reserved payload slots.
    # The latter can cover several primitive directions before cheap redundant
    # measurements exhaust the budget. Every seed is a real feasible integer
    # design; it supplies only an upper bound to the unchanged exact problem.
    for slots in range(1, min(budget, 2 * problem.latent_dim) + 1):
        slot_budget = budget // slots
        trial_bits, trial_matrices = [0] * count, [zero] * count
        trial_information, trial_score, trial_cost = zero.clone(), risk(zero), 0
        for _ in range(slots):
            best_addition = None
            for index, candidate in enumerate(options):
                if trial_bits[index]:
                    continue
                for width, cost, matrix in candidate[1:]:
                    if cost > slot_budget:
                        continue
                    score = risk(trial_information + matrix)
                    key = (score, cost, index, width)
                    if score < trial_score and (best_addition is None or key < best_addition[0]):
                        best_addition = (key, index, width, cost, matrix)
            if best_addition is None:
                break
            _, index, width, cost, matrix = best_addition
            trial_bits[index], trial_matrices[index] = width, matrix
            trial_cost += cost
            trial_information = sum(trial_matrices, zero.clone())
            trial_score = risk(trial_information)
            if (trial_score, trial_cost, tuple(trial_bits)) < (best_score, best_cost, best_bits):
                best_bits, best_score, best_cost = tuple(trial_bits), trial_score, trial_cost
    full_options = options
    # Branch on useful incumbent fields early. This only permutes enumeration;
    # externally visible names, allocations, and lexicographic ties keep their
    # original order. A field whose every active option is exactly zero in the
    # objective subspace has the already-proved lower-cost choice zero.
    order = [index for index, candidate in enumerate(options) if len(candidate) > 1]
    single_scores = {index: min(risk(matrix) for _, _, matrix in options[index][1:]) for index in order}
    order.sort(key=lambda index: (best_bits[index] == 0, single_scores[index], index))
    options = [options[index] for index in order]
    count = len(order)
    canonical_matrices = [{width: matrix for width, _, matrix in candidate} for candidate in full_options]
    # A remaining field may use only one bit width. Its PSD envelope is tighter
    # than adding every mutually exclusive precision and remains valid for
    # empirically nonmonotone curves. The suffix sum is an optimistic relaxation
    # of the shared remaining-budget constraint, hence a lower bound on risk.
    suffix_bounds = [[zero] * (budget + 1) for _ in range(count + 1)]
    # A second bound respects the shared integer budget. Diagonalize G in
    # tr[G(I+H)^-1], and independently maximize each future information diagonal
    # with a multiple-choice knapsack DP. For any positive-definite A,
    # (A^-1)_ii >= 1/A_ii; hence these maxima give a certified risk lower bound.
    gram_values, gram_vectors = torch.linalg.eigh(hermitian(gram))
    positive_gram = gram_values.clamp_min(0)
    negative_gram_allowance = float((-gram_values.clamp_max(0) * weights[:, None]).sum())
    diagonal_zero = torch.zeros_like(gram_values)
    option_diagonals = []
    for candidate in options:
        option_diagonals.append({width: (gram_vectors.mH @ matrix @ gram_vectors).diagonal(dim1=-2, dim2=-1).real
                                 for width, _, matrix in candidate})
    diagonal_bounds = [None] * (count + 1)
    diagonal_bounds[count] = torch.zeros((budget + 1, *diagonal_zero.shape), dtype=diagonal_zero.dtype, device=zero.device)
    if branch_bound:
        for depth in reversed(range(count)):
            envelope, previous_size = zero, 0
            for remaining in range(budget + 1):
                feasible = [precision for _, cost, precision in options[depth][1:] if cost <= remaining]
                if len(feasible) != previous_size:
                    envelope = psd_option_envelope(feasible) if feasible else zero
                    previous_size = len(feasible)
                suffix_bounds[depth][remaining] = envelope + suffix_bounds[depth + 1][remaining]
            future = diagonal_bounds[depth + 1]
            table = future.clone()
            for width, cost, _ in options[depth][1:]:
                # The maxima for different modes/coordinates need not share a
                # feasible design: relaxing this coupling can only lower risk.
                value = option_diagonals[depth][width].clamp_min(0)
                table[cost:] = torch.maximum(table[cost:], value[None] + future[:-cost])
            diagonal_bounds[depth] = table
    # Convex-risk supporting plane at a feasible incumbent. For W=-grad f(H0)
    # positive semidefinite, f(H) >= f(H0)+tr(W H0)-tr(W H). Maximizing the last
    # term over the remaining multiple-choice integer budget is a scalar exact
    # knapsack. This couples all modes/directions instead of independently
    # assigning their most favorable impossible future designs.
    tangent_weight = None
    tangent_constant = 0.0
    tangent_bounds = [None] * (count + 1)
    relaxation_diagnostics = {}
    if branch_bound:
        reference = sum((candidate[width] for candidate, width in zip(canonical_matrices, best_bits)), zero.clone())
        if count > 8:
            from .selectors_relaxation import convex_reference
            relaxation = convex_reference(options, budget, gram, weights, reference)
            reference = relaxation.information.to(reference)
            relaxation_diagnostics = {"reference_risk": relaxation.risk, "reference_lower_bound": relaxation.tangent_lower_bound,
                                      "reference_iterations": relaxation.iterations}
            if relaxation.best_integer_bits is not None:
                proposed = [0] * len(problem.names)
                for index, width in zip(order, relaxation.best_integer_bits):
                    proposed[index] = width
                proposed_cost = sum(width * components for width, components in zip(proposed, problem.component_counts))
                proposed_information = sum((candidate[width] for candidate, width in zip(canonical_matrices, proposed)), zero.clone())
                proposed_score = risk(proposed_information)
                if proposed_cost <= budget and (proposed_score, proposed_cost, tuple(proposed)) < (best_score, best_cost, best_bits):
                    best_bits, best_score, best_cost = tuple(proposed), proposed_score, proposed_cost
        inverse = torch.linalg.solve(identity + hermitian(reference), identity)
        positive_gram_matrix = (gram_vectors * positive_gram.unsqueeze(-2)) @ gram_vectors.mH
        tangent_weight = hermitian(inverse @ positive_gram_matrix @ inverse) * weights[:, None, None]
        reference_risk = float((weights * (positive_gram_matrix @ inverse).diagonal(dim1=-2, dim2=-1).sum(-1).real).sum())
        tangent_constant = reference_risk + float((tangent_weight.conj() * reference).sum().real) - negative_gram_allowance
        tangent_bounds[count] = torch.zeros(budget + 1, dtype=weights.dtype, device=weights.device)
        for depth in reversed(range(count)):
            future = tangent_bounds[depth + 1]
            table = future.clone()
            for _, cost, matrix in options[depth][1:]:
                score = float((tangent_weight.conj() * matrix).sum().real)
                table[cost:] = torch.maximum(table[cost:], future[:-cost] + max(0.0, score))
            tangent_bounds[depth] = table

    leaves, nodes = 0, 0
    root_bounds = {}
    allocation = [0]*len(problem.names)
    def visit(depth, remaining, information, diagonal_information):
        nonlocal leaves, nodes, best_bits, best_score, best_cost
        nodes += 1
        if max_nodes is not None and nodes > max_nodes:
            error = SearchLimitExceeded(f"exact selection exceeded {max_nodes} nodes; optimality not certified (incumbent risk={best_score:.12g}, bits={best_bits})")
            error.diagnostics = {"incumbent_risk": best_score, "incumbent_bits": best_bits,
                                 "prior_risk": risk(zero), "root_bounds": root_bounds, **relaxation_diagnostics}
            raise error
        if depth == count:
            leaves += 1
            canonical_information = sum((candidate[width] for candidate, width in zip(canonical_matrices, allocation)), zero.clone())
            score, cost, bits = risk(canonical_information), budget-remaining, tuple(allocation)
            if (score, cost, bits) < (best_score, best_cost, best_bits):
                best_bits, best_score, best_cost = bits, score, cost
            return
        if branch_bound:
            allowance = max(1e-12, abs(best_score)*1e-10)
            future_score = float(tangent_bounds[depth][remaining]) * (1 + (count - depth + 1) * 8 * torch.finfo(weights.dtype).eps)
            tangent_lower = tangent_constant - float((tangent_weight.conj() * information).sum().real) - future_score
            if depth == 0:
                root_bounds["tangent"] = tangent_lower
            if tangent_lower > best_score + allowance:
                return
            future_diagonal = diagonal_bounds[depth][remaining]
            # Outward allowance for positive DP sums, never an aggressive
            # pruning tolerance; ordinary field curves can be nonmonotone.
            future_diagonal = future_diagonal * (1 + (count - depth + 1) * 8 * torch.finfo(future_diagonal.dtype).eps)
            denominator = 1 + diagonal_information + future_diagonal
            diagonal_lower = float((weights[:, None] * positive_gram / denominator).sum()) - negative_gram_allowance
            if depth == 0:
                root_bounds["diagonal_knapsack"] = diagonal_lower
            if diagonal_lower > best_score + allowance:
                return
            lower = risk(information+suffix_bounds[depth][remaining])
            if depth == 0:
                root_bounds["psd_envelope"] = lower
            # Strict pruning with a conservative roundoff allowance keeps ties
            # available and does not erase tiny genuine risk improvements.
            if lower > best_score + allowance:
                return
        original_index = order[depth]
        ordered = sorted(options[depth], key=lambda item: (item[0] != best_bits[original_index], -item[0]))
        for width, cost, precision in ordered:
            if cost <= remaining:
                allocation[original_index] = width
                visit(depth+1, remaining-cost, information+precision, diagonal_information+option_diagonals[depth][width])
        allocation[original_index] = 0
    with torch.no_grad():
        visit(0, budget, zero, diagonal_zero)
    if not any(best_bits):
        # The empty design is optimal when all channels have zero information.
        pass
    return Design(best_bits, problem.names, best_score, best_cost, leaves, nodes)


def equal_allocation(problem: DesignProblem, selected: Design, budget: int) -> Design:
    """DerivBase: same subset, equal bits per carried scalar component.

    Remainder is padding, not an extra optimized assignment to one component.
    """
    included = [index for index, width in enumerate(selected.bits) if width > 0]
    if not included:
        return problem.design((0,)*len(problem.names), budget=budget)
    width = budget // sum(problem.component_counts[index] for index in included)
    if width < 1:
        raise ValueError("equal allocation cannot preserve the selected subset under this budget")
    bits = tuple(width if index in included else 0 for index in range(len(problem.names)))
    return problem.design(bits, budget=budget)


def best_single(problem: DesignProblem, budget: int, *, derived_indices: Sequence[int]) -> Design:
    if not derived_indices:
        raise ValueError("single-derived control needs a derived candidate")
    choices = []
    for index in derived_indices:
        for width in problem.precisions[index]:
            if width*problem.component_counts[index] <= budget:
                bits = [0]*len(problem.names)
                bits[index] = width
                choices.append(problem.design(bits, budget=budget))
    if not choices:
        raise ValueError("no active single-derived design fits the budget")
    return min(choices, key=lambda design: (design.risk, design.field_bits_per_site, design.bits))
