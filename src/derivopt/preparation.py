"""Train-only state preparation, reusable across all four predictor backbones."""
from dataclasses import dataclass
import math

import torch

from .budget import PayloadBudget
from .calibration import CalibratedState, calibrate
from .config import ExperimentConfig
from .data import FAMILY_CHANNELS, PDEBenchDataset, partition_trajectories, split_trajectories
from .fields import primary_candidates, shared_candidates, small_expert_candidates
from .geometry import Geometry
from .io import dataset_identity
from .selectors import Design, best_single, equal_allocation, select_exact


@dataclass
class PreparedState:
    geometry: Geometry
    calibration: CalibratedState
    optimum: Design
    mean: torch.Tensor
    std: torch.Tensor
    partitions: dict
    config: dict
    provenance: dict

    def design_for(self, method: str):
        problem = self.calibration.design_problem()
        budget = PayloadBudget(math.prod(self.geometry.resampler.coarse_shape), self.mean.numel(), self.config["budget_ratio"])
        cap = budget.field_bits_per_site
        if method in ("derivopt", "derivopt_archmulti"):
            return self.optimum
        if method == "derivbase":
            return equal_allocation(problem, self.optimum, cap)
        if method in ("primitive", "archmulti", "rolloutmulti", "latent_hyper"):
            # Primary library groups the primitive vector; shared library keeps
            # its scalar components separate. Both allocate equally per scalar.
            indices = [i for i, field in enumerate(self.calibration.candidates) if field.primitive]
            if not indices:
                raise ValueError("candidate library must preserve primitive-only feasibility")
            width = cap // sum(problem.component_counts[i] for i in indices)
            return problem.design(tuple(width if i in indices else 0 for i in range(len(problem.names))), budget=cap)
        if method == "best_single":
            if self.config["library"] in ("shared", "small_expert"):
                return best_single(problem, cap, derived_indices=[i for i, f in enumerate(self.calibration.candidates) if not f.primitive])
            field_name = {"advection": "lambda", "burgers": "lambda", "diffusion_sorption": "lambda",
                          "diffusion_reaction": "imbalance", "radial_dam_break": "lambda",
                          "incompressible_ns": "vorticity", "compressible_ns": "divergence"}[self.config["family"]]
            index = problem.names.index(field_name)
            width = cap // problem.component_counts[index]
            return problem.design(tuple(width if i == index else 0 for i in range(len(problem.names))), budget=cap)
        raise ValueError(f"unknown method {method}")

    def state_dict(self):
        return {"format_version": 2, "geometry": self.geometry.state_dict(), "calibration": self.calibration.state_dict(),
                "optimum": self.optimum.to_dict(), "mean": self.mean.cpu(), "std": self.std.cpu(),
                "partitions": self.partitions, "config": self.config, "provenance": self.provenance}

    @classmethod
    def from_state_dict(cls, state):
        if state.get("format_version") != 2:
            raise ValueError("Prepared-state checkpoint requires fixed time-step metadata; regenerate preparation")
        if "time_step" not in state.get("geometry", {}).get("config", {}):
            raise ValueError("Prepared-state checkpoint lacks a fixed time step; regenerate preparation")
        return cls(Geometry.from_state_dict(state["geometry"]), CalibratedState.from_state_dict(state["calibration"]),
                   Design(**state["optimum"]), state["mean"], state["std"], state["partitions"], state["config"], state["provenance"])

    def reopen_splits(self, dataset):
        positions = {key: index for index, key in enumerate(dataset.trajectory_ids)}
        expected = [key for values in self.partitions.values() for key in values]
        if len(set(expected)) != len(expected) or set(expected) != set(positions):
            raise ValueError("dataset trajectory identities differ from the prepared train/val/test partitions")
        return {name: dataset.subset([positions[key] for key in keys], split=name) for name, keys in self.partitions.items()}


def prepare(dataset: PDEBenchDataset, config: ExperimentConfig, *, calibration_fold: int | None = None, folds: int = 5,
            partitions=None):
    config.validate()
    if dataset.family != config.family:
        raise ValueError("dataset/config family mismatch")
    splits = (split_trajectories(dataset, seed=config.split_seed) if partitions is None
              else partition_trajectories(dataset, partitions))
    training = splits["train"]
    if calibration_fold is not None:
        if not 0 <= calibration_fold < folds or folds < 2 or len(training) < folds:
            raise ValueError("invalid or undersized training-only calibration folds")
        selected = [i for i in range(len(training)) if i % folds != calibration_fold]
        training = training.subset(selected, split="train")
    representative = training.read_window(0, 0, 2)
    geometry = Geometry.from_sample(representative, config.retain_frac, boundary_override=config.boundary_override,
                                    boundary_parameters=config.boundary_parameters)
    samples, used_ids = [], []
    for trajectory_index in range(len(training)):
        count = min(training.trajectory_length(trajectory_index), config.calibration_states - len(samples))
        # Two frames establish the fixed physical time step even when only one
        # additional calibration state is needed. Never read a whole trajectory.
        sample = training.read_window(trajectory_index, 0, max(2, count))
        if tuple(sample.states.shape[1:]) != tuple(representative.states.shape[1:]):
            raise ValueError("a prepared experiment requires common field shapes; partition incompatible grids explicitly")
        if any(not torch.allclose(a, b) for a, b in zip(sample.coords, representative.coords)):
            raise ValueError("trajectories in one experiment must share physical coordinates")
        geometry.validate_sample(sample)
        samples.extend(sample.states[:count].unbind(0))
        used_ids.extend([sample.metadata["trajectory_id"]]*count)
        if len(samples) >= config.calibration_states:
            break
    if len(samples) < 2:
        raise ValueError("insufficient training states for calibration")
    physical = torch.stack(samples)
    residual = geometry.to_residual(physical)
    components = FAMILY_CHANNELS[config.family]
    if config.library == "shared":
        vectors = ((0, 1),) if config.family == "incompressible_ns" else (((1, 2),) if config.family == "compressible_ns" else ())
        candidates = shared_candidates(geometry.basis, components, retained_cutoff=geometry.resampler.retained_cutoff,
                                       vector_groups=vectors)
    elif config.library == "small_expert":
        candidates = small_expert_candidates(config.family, geometry.basis, components, known_lift=geometry.lifting)
    else:
        candidates = primary_candidates(config.family, geometry.basis, components)
    budget = PayloadBudget(math.prod(geometry.resampler.coarse_shape), len(components), config.budget_ratio)
    bits = budget.field_bits_per_site
    calibration = calibrate(residual, geometry.basis, geometry.resampler, candidates, range(1, bits+1),
                            budget_per_site=bits, split="train",
                            divergence_free=config.family == "incompressible_ns" and geometry.basis.is_periodic)
    optimum = select_exact(calibration.design_problem(), bits, max_nodes=config.selector_max_nodes)
    coarse = geometry.resampler.coarsen(residual)
    axes = (0, *range(2, coarse.ndim))
    mean, std = coarse.mean(axes), coarse.std(axes, correction=0).clamp_min(1e-6)
    return PreparedState(geometry, calibration, optimum, mean, std,
                         {key: list(view.trajectory_ids) for key, view in splits.items()}, config.to_dict(),
                         {"calibration_split": "train", "calibration_trajectory_ids": used_ids,
                          "partition_source": "random" if partitions is None else "explicit_manifest",
                          "time_step": geometry.config["time_step"],
                          "calibration_fold": calibration_fold, "calibration_folds": folds if calibration_fold is not None else None,
                          "dataset_identity": dataset_identity(dataset),
                          "candidate_library": config.library,
                          "candidate_scalar_channels": sum(candidate.n_components for candidate in candidates),
                          "candidate_fields": [candidate.name for candidate in candidates],
                          "candidate_channel_models": ["train_fitted_modal_response" if candidate.requires_empirical_response
                                                       else "analytic_modal_response" for candidate in candidates],
                          "retained_cutoff": float(geometry.resampler.retained_cutoff),
                          "data_metadata": representative.metadata, "calibration_state_count": len(samples)})
