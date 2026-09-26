"""Physical grid, boundary-adapted basis, and known boundary lifting."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import torch

from .data import FAMILY_CHANNELS, FAMILY_NDIM, TrajectorySample, fixed_time_step
from .operators import SpectralBasis, SpectralResampler


def _simulation_config(metadata: dict[str, Any]):
    config = metadata.get("simulation_config")
    if config is None:
        for location in ("trajectory_attributes", "file_attributes"):
            candidate = metadata.get(location, {}).get("config")
            if candidate is not None:
                if isinstance(candidate, str):
                    import yaml
                    candidate = yaml.safe_load(candidate)
                if not isinstance(candidate, dict):
                    raise ValueError("Simulation config metadata must be a mapping")
                config = candidate.get("sim", candidate)
                break
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError("simulation_config must be a mapping")
    return config


def _boundary_declarations(metadata, simulation):
    """Read only explicit boundary labels, never infer them from a filename."""
    keys = {"velocity_extrapolation", "boundary", "boundaries", "boundary_condition", "boundary_conditions", "bc", "bc_type", "periodic"}
    declarations = []

    def values(value):
        if isinstance(value, dict):
            return [item for inner in value.values() for item in values(inner)]
        if isinstance(value, (list, tuple)):
            return [item for inner in value for item in values(inner)]
        if isinstance(value, bool):
            return ["periodic" if value else "nonperiodic"]
        return [str(value).strip().lower()]

    for mapping in (simulation, metadata, metadata.get("file_attributes", {}), metadata.get("trajectory_attributes", {})):
        if isinstance(mapping, dict):
            for key, value in mapping.items():
                if str(key).lower() in keys:
                    declarations.extend(values(value))
    return tuple(dict.fromkeys(declarations))


def _retained_shape(shape, retain_frac, coarse_shape):
    fractions = (float(retain_frac),) * len(shape) if isinstance(retain_frac, (int, float)) else tuple(float(value) for value in retain_frac)
    if len(fractions) != len(shape) or any(not math.isfinite(value) or value <= 0 or value > 1 for value in fractions):
        raise ValueError("retain_frac must be in (0,1] for every spatial axis")
    if coarse_shape is None:
        selected = tuple(math.floor(n * fraction + 1e-12) for n, fraction in zip(shape, fractions))
    else:
        selected = tuple(int(value) for value in coarse_shape)
        if len(selected) != len(shape) or any(raw != value for raw, value in zip(coarse_shape, selected)):
            raise ValueError("coarse_shape must contain one integer per spatial axis")
    if any(nc < 2 or nc > nf for nf, nc in zip(shape, selected)):
        raise ValueError("Requested retained grid must have between 2 and the fine sample count per axis; choose a larger retain_frac")
    return selected, fractions


def _chosen_boundaries(family, declarations, override, ndim):
    if override is not None:
        chosen = override
    elif family == "incompressible_ns" and declarations and all(
        value in {"zero", "dirichlet", "no-slip", "no_slip", "noslip"} for value in declarations
    ):
        chosen = "robin"  # Infinite Robin coefficients give homogeneous Dirichlet faces.
    elif family == "compressible_ns" and declarations and all(
        value in {"trans", "transmissive", "outflow", "neumann", "zero_gradient", "zero-gradient"}
        for value in declarations
    ):
        chosen = "neumann"
    else:
        chosen = {"diffusion_sorption": "robin", "diffusion_reaction": "neumann",
                  "radial_dam_break": "neumann"}.get(family, "periodic")
    result = (chosen.lower(),) * ndim if isinstance(chosen, str) else tuple(str(value).lower() for value in chosen)
    if len(result) != ndim:
        raise ValueError("Boundary override requires one boundary label per spatial axis")
    return result


def _validate_declared_domain(sample, simulation):
    """Validate cell-centred coordinates against the declared physical domain."""
    for coordinates, names in zip(sample.coords, (("x_left", "x_right"), ("y_bottom", "y_top"))):
        if not all(name in simulation for name in names):
            continue
        left, right = (float(simulation[name]) for name in names)
        if not math.isfinite(left) or not math.isfinite(right) or right <= left:
            raise ValueError("Invalid declared physical domain")
        values = torch.as_tensor(coordinates, dtype=torch.float64)
        spacing = (right-left) / values.numel()
        expected = left + (torch.arange(values.numel(), dtype=torch.float64)+.5)*spacing
        if not torch.allclose(values, expected, rtol=0, atol=max(1e-10, spacing*1e-5)):
            raise ValueError("Saved coordinates are not cell centres of the declared physical domain; align the data values and coordinates before preparation")


@dataclass
class Geometry:
    basis: SpectralBasis
    resampler: SpectralResampler
    lifting: torch.Tensor
    config: dict[str, Any]
    coordinates: tuple[torch.Tensor, ...] | None = None

    @property
    def channels(self):
        return self.lifting.shape[1]

    @classmethod
    def from_sample(
        cls, sample: TrajectorySample, retain_frac: float | Sequence[float], *,
        boundary_override: str | Sequence[str] | None = None,
        boundary_parameters: dict[str, Any] | None = None,
        coarse_shape: Sequence[int] | None = None,
        device: str | torch.device = "cpu", dtype: torch.dtype = torch.float64,
    ):
        family = sample.family
        time_step, time_step_tolerance = fixed_time_step(
            sample.times, source_dtype=sample.metadata.get("time_coordinate_dtype"), return_tolerance=True)
        if family not in FAMILY_NDIM:
            raise ValueError(f"Unsupported family {family}")
        ndim = FAMILY_NDIM[family]
        shape = tuple(sample.states.shape[-ndim:])
        channels = sample.states.shape[-ndim - 1]
        if len(sample.coords) != ndim or channels != len(FAMILY_CHANNELS[family]):
            raise ValueError("Sample coordinate/component dimensions do not match its family")
        # Reuse the basis constructor's strict uniform-grid check. Physical
        # lengths are N*dx, not x_max and not the cell-centre coordinate span.
        coordinate_basis = SpectralBasis.from_coordinates(sample.coords, device=device, dtype=dtype)
        if coordinate_basis.shape != shape:
            raise ValueError("Coordinate sample counts do not match the state grid")
        lengths = coordinate_basis.lengths
        selected, fractions = _retained_shape(shape, retain_frac, coarse_shape)
        simulation = _simulation_config(sample.metadata)
        declared = _boundary_declarations(sample.metadata, simulation)
        boundaries = _chosen_boundaries(family, declared, boundary_override, ndim)
        if family in {"diffusion_reaction", "radial_dam_break"}:
            _validate_declared_domain(sample, simulation)
        if family in {"diffusion_reaction", "radial_dam_break"} and any(value != "neumann" for value in boundaries):
            raise ValueError(f"{family} requires the Neumann boundary-adapted candidate basis")
        if family == "diffusion_sorption" and boundaries != ("robin",):
            raise ValueError("Diffusion-Sorption uses a homogeneous Robin residual after known boundary lifting")
        parameters = dict(boundary_parameters or {})
        lifting = torch.zeros((1, channels, *shape), dtype=dtype, device=device)
        robin = None
        lifting_description = "zero"
        if family == "diffusion_sorption":
            required = ("D", "sol", "x_left", "x_right")
            missing = [key for key in required if key not in parameters and key not in simulation]
            if missing:
                raise ValueError("Diffusion-Sorption boundary parameters are required in simulation_config or boundary_parameters: " + ", ".join(missing))
            parameters = {key: float(parameters.get(key, simulation.get(key))) for key in required}
            diffusion, sol, left, right = (parameters[key] for key in required)
            if not all(math.isfinite(value) for value in parameters.values()) or diffusion <= 0 or right <= left:
                raise ValueError("Diffusion-Sorption requires finite sol/domain parameters, D>0 and x_right>x_left")
            length = right - left
            coordinates = torch.as_tensor(sample.coords[0], device=device, dtype=dtype)
            spacing = length / shape[0]
            expected = left + (torch.arange(shape[0], dtype=dtype, device=device) + 0.5) * spacing
            if not math.isclose(lengths[0], length, rel_tol=1e-5, abs_tol=1e-10) or not torch.allclose(coordinates, expected, rtol=1e-5, atol=max(1e-10, spacing * 1e-5)):
                raise ValueError("Diffusion-Sorption x_left/x_right must match the saved uniform cell-centred grid; endpoint or cropped grids need an explicit compatible conversion")
            robin = ((math.inf, 1 / diffusion),)
            lifting[0, 0] = sol * (length + diffusion - (coordinates - left)) / (length + diffusion)
            lifting_description = "sol*(L+D-(x-x_left))/(L+D)"
        elif family == "incompressible_ns" and boundaries == ("robin",)*ndim and boundary_override is None:
            robin = ((math.inf, math.inf),)*ndim
            lifting_description = "homogeneous_no_slip"
        elif any(value == "robin" for value in boundaries):
            if "robin_coefficients" not in parameters:
                raise ValueError("An explicit Robin override requires robin_coefficients")
            robin = parameters["robin_coefficients"]
        basis = SpectralBasis(shape, lengths, boundaries, robin_coefficients=robin, device=device, dtype=dtype)
        config = {
            "family": family, "fine_shape": shape, "coarse_shape": selected,
            "time_step": time_step, "time_step_tolerance": time_step_tolerance,
            "physical_lengths": lengths, "coordinate_layout": "uniform_cell_centres",
            "coordinate_first_centers": tuple(float(axis[0]) for axis in sample.coords),
            "coordinate_spacing": tuple(float(axis[1] - axis[0]) for axis in sample.coords),
            "retain_frac_requested": fractions,
            "retain_frac_realized": tuple(nc / nf for nf, nc in zip(shape, selected)),
            "coarse_shape_explicit": coarse_shape is not None,
            "boundaries": boundaries, "source_boundary_declarations": declared,
            "boundary_override": boundary_override, "boundary_parameters": parameters,
            "boundary_parameter_overrides": dict(boundary_parameters or {}),
            "lifting": lifting_description,
            "boundary_source": "explicit_override" if boundary_override is not None else ("source_and_family" if declared else "family_default"),
        }
        result = cls(basis, SpectralResampler(basis, selected), lifting, config,
                     tuple(torch.as_tensor(axis, dtype=torch.float64).detach().cpu().clone() for axis in sample.coords))
        result.validate_sample(sample)
        return result

    def validate_sample(self, sample: TrajectorySample) -> None:
        """Check a trajectory against this fixed geometry without rebuilding bases.

        Only small coordinate arrays and metadata are examined here; the data
        adapter already checks finite states/times. In particular, a grouped
        file's later trajectories cannot silently change its physical grid or
        boundary parameters after a train-only calibration sample limit.
        """
        if sample.family != self.config.get("family"):
            raise ValueError("Trajectory family differs from the fixed experiment geometry")
        if "time_step" not in self.config or "time_step_tolerance" not in self.config:
            raise ValueError("Geometry lacks fixed time-step metadata; regenerate preparation")
        time_step, uncertainty = fixed_time_step(
            sample.times, source_dtype=sample.metadata.get("time_coordinate_dtype"), return_tolerance=True)
        if abs(time_step - self.config["time_step"]) > uncertainty + self.config["time_step_tolerance"]:
            raise ValueError("Trajectory time step differs from the prepared fixed time step; partition incompatible sampling intervals explicitly")
        expected_shape = (self.channels, *self.basis.shape)
        if sample.states.ndim != self.basis.ndim + 2 or tuple(sample.states.shape[1:]) != expected_shape:
            raise ValueError("Trajectory field shape differs from the fixed experiment geometry")
        required = ("coordinate_first_centers", "coordinate_spacing", "boundary_parameter_overrides")
        if self.coordinates is None or any(key not in self.config for key in required):
            raise ValueError("Geometry checkpoint lacks coordinate/boundary validation metadata; regenerate its prepared state")
        if len(sample.coords) != self.basis.ndim:
            raise ValueError("Trajectory coordinate dimensions differ from the fixed geometry")
        for axis, expected, spacing, size in zip(
            sample.coords, self.coordinates, self.config["coordinate_spacing"], self.basis.shape
        ):
            coordinates = torch.as_tensor(axis, dtype=torch.float64, device="cpu")
            expected = expected.to(device="cpu", dtype=torch.float64)
            tolerance = max(1e-10, abs(spacing) * 1e-5)
            if (coordinates.shape != (size,) or not torch.isfinite(coordinates).all()
                    or not torch.allclose(coordinates, expected, rtol=0, atol=tolerance)):
                raise ValueError("Trajectory physical coordinates/domain differ from the saved uniform grid; partition incompatible grids explicitly")
        simulation = _simulation_config(sample.metadata)
        declarations = _boundary_declarations(sample.metadata, simulation)
        family = sample.family
        override = self.config["boundary_override"]
        boundaries = _chosen_boundaries(family, declarations, override, self.basis.ndim)
        if boundaries != self.basis.boundaries:
            raise ValueError("Trajectory boundary convention differs from the saved geometry")
        if override is None and declarations:
            aliases = {
                "periodic": {"periodic", "wrap", "circular"},
                "neumann": {"neumann", "zero_gradient", "zero-gradient", "zero_flux", "zero-flux", "no_flux", "no-flux", "noflux", "reflect", "reflective", "reflecting", "trans", "transmissive", "outflow"},
                "robin": {"robin", "cauchy", "dirichlet", "mixed", "dirichlet_robin", "dirichlet-robin", "zero", "no-slip", "no_slip", "noslip"},
            }
            allowed = set().union(*(aliases.get(boundary, {boundary}) for boundary in boundaries))
            if any(value not in allowed for value in declarations):
                raise ValueError(f"Trajectory source boundary declarations {declarations} conflict with saved geometry {boundaries}; use an explicit intentional boundary_override or compatible data")
        overrides = self.config["boundary_parameter_overrides"]
        if family in {"diffusion_reaction", "radial_dam_break"}:
            _validate_declared_domain(sample, simulation)
        if family == "diffusion_sorption":
            names = ("D", "sol", "x_left", "x_right")
            missing = [name for name in names if name not in overrides and name not in simulation]
            if missing:
                raise ValueError("Trajectory is missing required boundary parameters: " + ", ".join(missing))
            effective = {name: float(overrides.get(name, simulation.get(name))) for name in names}
            expected = self.config["boundary_parameters"]
            if any(not math.isfinite(value) or not math.isclose(value, expected[name], rel_tol=1e-9, abs_tol=1e-12)
                   for name, value in effective.items()):
                raise ValueError("Trajectory boundary parameters/lifting differ from the calibrated D, sol or physical domain; partition incompatible boundary conditions explicitly")

    def _lift_for(self, state):
        required = tuple(self.lifting.shape[1:])
        if state.ndim < len(required) or tuple(state.shape[-len(required):]) != required:
            raise ValueError(f"Expected state trailing dimensions {required}, got {tuple(state.shape)}")
        if not torch.is_floating_point(state):
            raise TypeError("Physical and residual states must be floating-point tensors")
        return self.lifting.squeeze(0).to(device=state.device, dtype=state.dtype)

    def to_residual(self, physical_state: torch.Tensor):
        return physical_state - self._lift_for(physical_state)

    def to_physical(self, residual_state: torch.Tensor):
        return residual_state + self._lift_for(residual_state)

    def state_dict(self):
        return {
            "format_version": 3,
            "basis": {"shape": self.basis.shape, "lengths": self.basis.lengths,
                      "boundary": self.basis.boundaries, "robin_coefficients": self.basis.robin_coefficients,
                      "dtype": str(self.basis.dtype).split(".")[-1]},
            "coarse_shape": self.resampler.coarse_shape,
            "lifting": self.lifting.detach().cpu(), "config": self.config,
            "coordinates": None if self.coordinates is None else tuple(axis.detach().cpu().clone() for axis in self.coordinates),
        }

    @classmethod
    def from_state_dict(cls, state, *, device="cpu"):
        if state.get("format_version") != 3:
            raise ValueError("Unsupported geometry checkpoint version; regenerate preparation with saved coordinate and fixed time-step metadata")
        if state.get("coordinates") is not None:
            step = state.get("config", {}).get("time_step")
            uncertainty = state.get("config", {}).get("time_step_tolerance")
            if not isinstance(step, (int, float)) or not math.isfinite(step) or step <= 0:
                raise ValueError("Geometry checkpoint lacks a valid fixed time step; regenerate preparation")
            if (not isinstance(uncertainty, (int, float)) or not math.isfinite(uncertainty)
                    or not 0 <= uncertainty < step * .02):
                raise ValueError("Geometry checkpoint lacks valid time-step precision metadata; regenerate preparation")
        specification = dict(state["basis"])
        specification["dtype"] = getattr(torch, specification["dtype"])
        basis = SpectralBasis(**specification, device=device)
        lifting = state["lifting"].to(device=device, dtype=basis.dtype)
        coordinates = state.get("coordinates")
        return cls(basis, SpectralResampler(basis, state["coarse_shape"]), lifting, state["config"],
                   None if coordinates is None else tuple(axis.detach().cpu().clone() for axis in coordinates))


def from_sample(sample, retain_frac, **kwargs):
    """Convenience alias for ``Geometry.from_sample``."""
    return Geometry.from_sample(sample, retain_frac, **kwargs)
