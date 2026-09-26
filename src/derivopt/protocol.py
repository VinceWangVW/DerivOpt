"""Main experiment constants, metric thresholds and canonical-regime selection."""
from dataclasses import dataclass
import math

BUDGET_RATIOS = (0.25, 0.5, 1.0)
RETAIN_FRACS = (0.125, 0.25, 0.5)
CANONICAL = (0.25, 0.125)
BACKBONES = ("unet", "fno", "convlstm", "transformer")
MAIN_METHODS = ("primitive", "best_single", "derivbase", "derivopt", "archmulti", "rolloutmulti", "latent_hyper")
HORIZON_PROTOCOL = "future_only_contiguous_prefix_v1; unconditional_trajectory_mean"
HORIZON_UNITS = "normalized; detail_horizon_steps in prediction_steps"


@dataclass(frozen=True)
class MetricProtocol:
    tau_q: float = 1.25
    tau_out: float = 0.10
    fine_low: float = 0.75
    fine_high: float = 1.0

    def __post_init__(self):
        if not all(math.isfinite(x) for x in (self.tau_q, self.tau_out, self.fine_low, self.fine_high)):
            raise ValueError("metric protocol must be finite")
        if self.tau_q < 1 or self.tau_out < 0 or not 0 <= self.fine_low < self.fine_high <= 1:
            raise ValueError("invalid detail protocol")


def select_canonical_from_pilot(records: list[dict], *, split: str):
    """Apply the NS pilot rule, without accepting DerivOpt or test metrics.

    Each record supplies R/f and already equally averaged run diagnostics.
    The helper applies the decision rule; callers supply the pilot diagnostics.
    """
    if split != "train":
        raise ValueError("canonical pilot must be training-only")
    eligible = []
    required = ("budget_ratio", "retain_frac", "primitive_finite_fraction", "primitive_imputed_nrmse",
                "primitive_imputed_horizon", "derivbase_imputed_nrmse", "archmulti_imputed_nrmse")
    for row in records:
        if any(key not in row for key in required):
            raise ValueError("pilot diagnostics are incomplete")
        if not all(math.isfinite(row[key]) for key in required):
            raise ValueError("pilot diagnostics must be finite after failure imputation")
        if (row["primitive_finite_fraction"] >= .95 and row["primitive_imputed_nrmse"] <= .15
                and row["primitive_imputed_horizon"] <= .10
                and min(row["derivbase_imputed_nrmse"], row["archmulti_imputed_nrmse"]) <= .15):
            eligible.append(row)
    if not eligible:
        raise ValueError("no eligible canonical regime; do not choose using test results")
    chosen = min(eligible, key=lambda r: (r["budget_ratio"]*r["retain_frac"]**2, r["budget_ratio"], r["retain_frac"]))
    return chosen["budget_ratio"], chosen["retain_frac"]
