"""Convex reference points for certified discrete-search supporting planes.

This module supplies reference points; selectors.py performs exact integer
certification. A finite fully-corrective Frank--Wolfe iteration combines feasible
integer information matrices. A multiple-choice knapsack serves as the linear
oracle, and a small simplex SLSQP problem improves the convex weights. Any
resulting PSD point defines a valid tangent plane even when the optimizer
stops early.
"""
from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from scipy.optimize import minimize
import torch


@dataclass(frozen=True)
class RelaxationReference:
    information: torch.Tensor
    risk: float
    tangent_lower_bound: float
    iterations: int
    best_integer_bits: tuple[int, ...] | None
    best_integer_risk: float
    references: tuple[torch.Tensor, ...]


def convex_reference(options: Sequence[Sequence[tuple[int, int, torch.Tensor]]],
                     budget: int, gram: torch.Tensor, weights: torch.Tensor,
                     initial_information: torch.Tensor, *, max_iterations: int = 24) -> RelaxationReference:
    """Improve a feasible reference, without requiring convergence or exactness.

``options[field]`` contains ``(width, integer_cost, PSD_information)`` choices,
including a zero choice. Matrices are already restricted to objective modes.
The caller supplies a feasible initial information matrix. Returned integer
bits use the supplied field order and are merely feasible incumbent candidates.
The diagnostic tangent bound uses a PSD-projected Gram matrix; a caller whose
unprojected Gram has negative roundoff eigenvalues must subtract their weighted
negative-eigenvalue allowance, as the exact selector does.
    """
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
        raise ValueError("relaxation budget must be a positive integer")
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if not options or gram.ndim != 3 or gram.shape[-1] != gram.shape[-2]:
        raise ValueError("reference requires nonempty options and [mode,latent,latent] Gram")
    dtype = torch.complex128 if gram.is_complex() else torch.float64
    device = gram.device
    gram = gram.to(dtype)
    initial = initial_information.to(device=device, dtype=dtype)
    weights = torch.as_tensor(weights, device=device, dtype=torch.float64)
    if (initial.shape != gram.shape or weights.shape != gram.shape[:1]
            or not torch.isfinite(initial).all() or not torch.isfinite(gram).all()
            or not torch.isfinite(weights).all() or torch.any(weights < 0) or not torch.any(weights > 0)):
        raise ValueError("invalid finite reference objective")
    gram_values, gram_vectors = torch.linalg.eigh((gram + gram.mH) / 2)
    positive_gram = (gram_vectors * gram_values.clamp_min(0).unsqueeze(-2)) @ gram_vectors.mH
    identity = torch.eye(gram.shape[-1], dtype=dtype, device=device).expand_as(gram)
    zero = torch.zeros_like(initial)
    groups = []
    for candidate in options:
        group = []
        for width, cost, information in candidate:
            if (not isinstance(width, int) or isinstance(width, bool) or width < 0
                    or not isinstance(cost, int) or isinstance(cost, bool) or cost < 0
                    or (width == 0) != (cost == 0)):
                raise ValueError("options require consistent nonnegative integer widths/costs")
            if cost > budget:
                continue
            information = information.to(device=device, dtype=dtype)
            if information.shape != gram.shape or not torch.isfinite(information).all():
                raise ValueError("option information shape or finiteness mismatch")
            group.append((width, cost, (information + information.mH) / 2))
        if not group or not any(width == 0 and torch.equal(matrix, zero) for width, _, matrix in group):
            raise ValueError("each field requires its zero information choice")
        groups.append(group)

    def objective(information):
        inverse = torch.linalg.solve(identity + (information + information.mH) / 2, identity)
        value = float((weights * (positive_gram @ inverse).diagonal(dim1=-2, dim2=-1).sum(-1).real).sum())
        if not math.isfinite(value) or value < 0:
            raise FloatingPointError("invalid risk in convex reference optimization")
        gradient_weight = (inverse @ positive_gram @ inverse)
        gradient_weight = (gradient_weight + gradient_weight.mH) / 2 * weights[:, None, None]
        return value, gradient_weight

    def linear_oracle(weight):
        # At-most-budget DP; every field contributes at most one option.
        previous = np.zeros(budget + 1, dtype=np.float64)
        choices = []
        for group in groups:
            selected = np.zeros(budget + 1, dtype=np.int64)
            table = np.full(budget + 1, -np.inf, dtype=np.float64)
            for option, (_, cost, matrix) in enumerate(group):
                score = float((weight.conj() * matrix).sum().real)
                if not math.isfinite(score):
                    raise FloatingPointError("nonfinite relaxation linear score")
                proposed = previous[:budget + 1 - cost] + score
                better = proposed > table[cost:]
                table[cost:][better] = proposed[better]
                selected[cost:][better] = option
            previous = table
            choices.append(selected)
        remaining, bits, matrices = budget, [], []
        for group, selected in zip(reversed(groups), reversed(choices)):
            width, cost, matrix = group[int(selected[remaining])]
            bits.append(width)
            matrices.append(matrix)
            remaining -= cost
        information = sum(reversed(matrices), zero.clone())
        return float(previous[budget]), tuple(reversed(bits)), information

    atoms = [initial]
    coefficients = np.ones(1, dtype=np.float64)
    current = initial
    best_bits, best_risk = None, math.inf
    current_risk, weight = objective(current)
    scale = max(current_risk, np.finfo(np.float64).tiny)
    lower = -math.inf
    completed = 0
    history = [(-math.inf, current_risk, initial)]
    with torch.no_grad():
        for iteration in range(max_iterations):
            current_risk, weight = objective(current)
            try:
                maximum, integer_bits, integer_information = linear_oracle(weight)
            except FloatingPointError:
                break
            try:
                integer_risk, _ = objective(integer_information)
            except (FloatingPointError, torch.linalg.LinAlgError):
                integer_risk = math.inf
            if integer_risk < best_risk:
                best_bits, best_risk = integer_bits, integer_risk
            lower = current_risk + float((weight.conj() * current).sum().real) - maximum
            history.append((lower, current_risk, current))
            completed = iteration + 1
            # This only terminates reference improvement. It is never used as
            # an exactness certificate or permission to skip discrete search.
            if current_risk - lower <= max(1e-10, abs(current_risk) * 1e-7):
                break
            if any(torch.equal(integer_information, atom) for atom in atoms):
                break
            atoms.append(integer_information)
            coefficients = np.append(coefficients, 0.)
            stacked = torch.stack(atoms)

            def simplex_objective(values):
                tensor = torch.as_tensor(values, dtype=torch.float64, device=device)
                information = torch.einsum("a,amij->mij", tensor.to(dtype), stacked)
                value, tangent = objective(information)
                gradient = -(tangent.conj()[None] * stacked).sum(dim=(1, 2, 3)).real
                return value / scale, gradient.cpu().numpy() / scale

            try:
                optimized = minimize(simplex_objective, coefficients, method="SLSQP", jac=True,
                                     bounds=[(0., 1.)] * len(atoms),
                                     constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1.,
                                                   "jac": lambda x: np.ones_like(x)}],
                                     options={"maxiter": 80, "ftol": 1e-12, "disp": False})
            except (FloatingPointError, torch.linalg.LinAlgError):
                # Optional acceleration failing numerically does not replace
                # the caller's discrete search or invalidate its incumbent.
                break
            proposed = np.maximum(np.asarray(optimized.x, dtype=np.float64), 0.)
            if not np.isfinite(proposed).all() or proposed.sum() <= 0:
                break
            proposed /= proposed.sum()
            candidate = torch.einsum("a,amij->mij", torch.as_tensor(proposed, device=device).to(dtype), stacked)
            try:
                candidate_risk, _ = objective(candidate)
            except (FloatingPointError, torch.linalg.LinAlgError):
                break
            if candidate_risk >= current_risk:
                break
            current, coefficients = candidate, proposed
        current_risk, weight = objective(current)
        try:
            maximum, integer_bits, integer_information = linear_oracle(weight)
            lower = current_risk + float((weight.conj() * current).sum().real) - maximum
            history.append((lower, current_risk, current))
            integer_risk, _ = objective(integer_information)
            if integer_risk < best_risk:
                best_bits, best_risk = integer_bits, integer_risk
        except (FloatingPointError, torch.linalg.LinAlgError):
            pass
    # Keep a few strong global planes plus the initial local plane. Different
    # planes may dominate after some integer choices have been fixed.
    ranked = sorted(history, key=lambda item: item[0], reverse=True)
    references = []
    for _, _, reference in ranked:
        if not any(torch.equal(reference, previous) for previous in references):
            references.append(reference)
        if len(references) >= 3:
            break
    if not any(torch.equal(initial, previous) for previous in references):
        references.append(initial)
    lower, current_risk, current = ranked[0]
    return RelaxationReference(current, current_risk, lower, completed, best_bits, best_risk, tuple(references))
