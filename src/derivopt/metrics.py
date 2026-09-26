"""Per-sample physical-space and basis-adapted detail metrics.

Undefined values are retained in per-sample results. Thresholding, zero-energy
handling and pilot failure imputation are separate operations.
"""
from __future__ import annotations

import math
from typing import Mapping

import torch

from .operators import SpectralBasis
from .protocol import MetricProtocol


def _mask(mask: torch.Tensor, basis: SpectralBasis, name: str) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or tuple(mask.shape) != basis.shape:
        raise ValueError(f"{name} must be a bool tensor with spatial shape {basis.shape}")
    return mask


def build_fine_mask(basis: SpectralBasis, expressible_mask: torch.Tensor, *,
                    low: float = 0.75, high: float = 1.0, cutoff: float | None = None) -> torch.Tensor:
    """Intersect the expressible set with (low,high] of a retained cutoff.

    Frequency is sqrt(the boundary-adapted operator eigenvalue), i.e. radial in
    tensor-product frequency coordinates. The runner passes the resampler's
    physical cutoff; absent one, use the supplied mask's maximum frequency.
    An alternative band can be supplied explicitly to ``compute_metrics``.
    """
    mask = _mask(expressible_mask, basis, "expressible_mask")
    if not math.isfinite(low) or not math.isfinite(high) or not 0 <= low < high <= 1:
        raise ValueError("Fine-band edges must satisfy 0 <= low < high <= 1")
    frequency = basis.eigenvalues.sqrt().to(mask.device)
    if not bool(mask.any()):
        return torch.zeros_like(mask)
    maximum = frequency[mask].max() if cutoff is None else float(cutoff)
    if not math.isfinite(float(maximum)) or maximum < 0:
        raise ValueError("cutoff must be finite and nonnegative")
    if maximum == 0:
        return torch.zeros_like(mask)
    ratio = frequency / maximum
    return mask & (ratio > low) & (ratio <= high)


def _energy(coefficients: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is not None:
        # Indexing, instead of multiplying by zero, avoids NaN*0 pollution
        # outside a projector; the separate input-finite mask still records it.
        coefficients = coefficients[..., mask]
    return coefficients.abs().square().flatten(start_dim=1).sum(dim=1)


def _energy_ratio(numerator: torch.Tensor, denominator: torch.Tensor,
                  tolerance: torch.Tensor) -> torch.Tensor:
    """0/0 -> 0, positive/0 -> inf; existing nonfinite inputs stay nonfinite."""
    zero_den = (denominator <= tolerance) & torch.isfinite(numerator) & torch.isfinite(denominator)
    both_zero = zero_den & (numerator <= tolerance)
    ratio = numerator / denominator
    ratio = torch.where(zero_den & ~both_zero, torch.full_like(ratio, float("inf")), ratio)
    return torch.where(both_zero, torch.zeros_like(ratio), ratio)


def normalized_rmse(prediction: torch.Tensor, target: torch.Tensor, *,
                     return_channels: bool = False) -> torch.Tensor:
    """PDEBench-style spatial relative RMSE per channel, then equal channel mean.

    Inputs are [B,C,*space]. No batch, time, family or configuration averaging is
    performed. This differs from a concatenated multi-channel relative L2 norm.
    """
    if prediction.shape != target.shape or prediction.ndim < 3:
        raise ValueError("prediction and target must share [B,C,*space] shape")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError("Metrics require real floating physical fields")
    numerator = (prediction - target).square().flatten(start_dim=2).sum(dim=2)
    denominator = target.square().flatten(start_dim=2).sum(dim=2)
    channels = _energy_ratio(numerator, denominator, torch.zeros_like(denominator)).sqrt()
    return channels if return_channels else channels.mean(dim=1)


def input_pass(metrics: Mapping[str, torch.Tensor], protocol: MetricProtocol | None = None) -> torch.Tensor:
    """Apply global thresholds without changing any underlying error or energy."""
    protocol = protocol or MetricProtocol()
    needed = ("expr_rel", "fine_rel", "q_fine", "e_out")
    if any(name not in metrics for name in needed):
        raise ValueError(f"Thresholding requires {needed}")
    finite = torch.stack([torch.isfinite(metrics[name]) for name in needed]).all(dim=0)
    if "input_finite_mask" in metrics:
        finite = finite & metrics["input_finite_mask"]
    return (finite & (metrics["expr_rel"] < 1) & (metrics["fine_rel"] < 1)
            & (metrics["q_fine"] >= 1 / protocol.tau_q) & (metrics["q_fine"] <= protocol.tau_q)
            & (metrics["e_out"] < protocol.tau_out))


def compute_metrics(prediction: torch.Tensor, target: torch.Tensor, basis: SpectralBasis,
                    expressible_mask: torch.Tensor, fine_mask: torch.Tensor | None = None,
                    *, protocol: MetricProtocol | None = None,
                    zero_energy_rtol: float | None = None) -> dict[str, torch.Tensor]:
    """Return per-sample metrics; neither threshold failures nor NaNs are hidden.

    Detail norms concatenate the target channels as in the paper's ||g||_2.
    Fields are evaluated in basis.dtype to limit transform roundoff. A negligible
    energy threshold of 64*eps**2 times field energy distinguishes zero-energy
    bands from floating-point transform leakage; pass 0 for exact-zero semantics.
    """
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    basis._check_shape(prediction)
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError("Metrics require real floating physical fields")
    if prediction.device != target.device:
        raise ValueError("prediction and target must be on the same device")
    protocol = protocol or MetricProtocol()
    expressible_mask = _mask(expressible_mask, basis, "expressible_mask").to(prediction.device)
    fine_mask = build_fine_mask(basis, expressible_mask, low=protocol.fine_low,
                               high=protocol.fine_high) if fine_mask is None else _mask(fine_mask, basis, "fine_mask").to(prediction.device)
    if bool((fine_mask & ~expressible_mask).any()):
        raise ValueError("fine_mask must be a subset of expressible_mask")
    prediction, target = prediction.to(basis.dtype), target.to(basis.dtype)
    if zero_energy_rtol is None:
        zero_energy_rtol = 64 * torch.finfo(basis.dtype).eps ** 2
    if not math.isfinite(zero_energy_rtol) or zero_energy_rtol < 0:
        raise ValueError("zero_energy_rtol must be finite and nonnegative")
    pred_coeff, true_coeff = basis.analysis(prediction), basis.analysis(target)
    error_coeff = pred_coeff - true_coeff
    true_total, pred_total = _energy(true_coeff), _energy(pred_coeff)
    tolerance = true_total * zero_energy_rtol
    pred_tolerance = pred_total * zero_energy_rtol
    true_exp, true_fine = _energy(true_coeff, expressible_mask), _energy(true_coeff, fine_mask)
    pred_exp, pred_fine = _energy(pred_coeff, expressible_mask), _energy(pred_coeff, fine_mask)
    expr = _energy_ratio(_energy(error_coeff, expressible_mask), true_exp, tolerance).sqrt()
    fine = _energy_ratio(_energy(error_coeff, fine_mask), true_fine, tolerance).sqrt()
    outside = _energy_ratio(_energy(pred_coeff, ~expressible_mask), true_exp, tolerance)
    q_defined = ((true_exp > tolerance) & (true_fine > tolerance) & (pred_exp > pred_tolerance)
                 & torch.isfinite(true_exp) & torch.isfinite(true_fine) & torch.isfinite(pred_exp)
                 & torch.isfinite(pred_fine))
    ratio = (pred_fine / pred_exp) / (true_fine / true_exp)
    ratio = torch.where(q_defined, ratio, torch.full_like(ratio, float("nan")))
    raw_finite = (torch.isfinite(prediction).flatten(start_dim=1).all(dim=1)
                  & torch.isfinite(target).flatten(start_dim=1).all(dim=1))
    channel_nrmse = normalized_rmse(prediction, target, return_channels=True)
    result = {"nrmse": channel_nrmse.mean(dim=1), "nrmse_channels": channel_nrmse,
              "expr_rel": expr, "fine_rel": fine, "q_fine": ratio, "e_out": outside,
              "q_defined": q_defined, "input_finite_mask": raw_finite}
    result["finite_mask"] = raw_finite & torch.stack([torch.isfinite(result[key])
                            for key in ("nrmse", "expr_rel", "fine_rel", "q_fine", "e_out")]).all(dim=0)
    result["input_pass"] = input_pass(result, protocol)
    return result


def detail_horizon(passed: torch.Tensor) -> torch.Tensor:
    """Consecutive valid predictions from t=1, divided by prediction count.

    ``passed`` is bool [B,T+1] and includes the separately evaluated input t=0.
    Input validity does not gate this metric. A failed predicted frame ends the
    prefix; later recovery never extends it. With no predicted frames, return 0.
    """
    steps = detail_horizon_steps(passed)
    return steps.to(torch.float64) / max(passed.shape[1] - 1, 1)


def detail_horizon_steps(passed: torch.Tensor) -> torch.Tensor:
    """Number of consecutive valid predicted frames starting at t=1.

    The full bool [B,T+1] input includes t=0, which is ignored here and remains
    available for the separate input pass rate. The first failed prediction
    stops the prefix, including when the initial reconstruction failed.
    """
    if passed.ndim != 2 or passed.dtype != torch.bool or passed.shape[1] < 1:
        raise ValueError("passed must be a nonempty bool [B,T+1] tensor including t=0")
    return passed[:, 1:].to(torch.int64).cumprod(dim=1).sum(dim=1)


def impute_pilot_failures(nrmse: torch.Tensor, horizon: torch.Tensor, *,
                          finite_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Explicit pilot-only failure convention: invalid runs get nRMSE=1, horizon=0.

    Caller-supplied finite_mask can additionally flag failed optimization/rollout.
    Returns separate pilot diagnostics while preserving the original metrics.
    """
    if nrmse.shape != horizon.shape or nrmse.ndim != 1 or nrmse.numel() == 0:
        raise ValueError("Pilot diagnostics must be nonempty matching [runs] arrays")
    finite = torch.isfinite(nrmse) & torch.isfinite(horizon)
    if finite_mask is not None:
        if finite_mask.shape != nrmse.shape or finite_mask.dtype != torch.bool:
            raise ValueError("finite_mask must be a matching bool array")
        finite = finite & finite_mask
    return {"imputed_nrmse": torch.where(finite, nrmse, torch.ones_like(nrmse)),
            "imputed_horizon": torch.where(finite, horizon, torch.zeros_like(horizon)),
            "finite_mask": finite, "finite_fraction": finite.double().mean()}
