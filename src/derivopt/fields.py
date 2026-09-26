"""Physical candidate fields for the seven paper families and shared libraries.

Modal symbols act on primitive spectral coefficients; this is equivalent to
latent rotations when the prior and target are transformed consistently.
Empirical candidates generate real fine-grid fields and keep their fitted
modal response separate from the physical evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Sequence

import torch

from .operators import SpectralBasis


@dataclass(frozen=True)
class FieldCandidate:
    name: str
    matrix: torch.Tensor  # [number of modes, output scalar components, state components]
    component_names: tuple[str, ...]
    primitive: bool = False
    kind: str = "modal"
    parameters: dict = field(default_factory=dict)
    output_scale: torch.Tensor | None = None

    @property
    def requires_empirical_response(self):
        """A physical evaluator needs a train-fitted effective modal response."""
        return self.kind != "modal"

    @property
    def symbol(self):
        return self.matrix

    @property
    def n_components(self):
        return self.matrix.shape[1]

    @property
    def state_components(self):
        return self.matrix.shape[2]

    def apply_coefficients(self, coefficients: torch.Tensor) -> torch.Tensor:
        if self.requires_empirical_response:
            raise ValueError(f"{self.name} is not a diagonal spectral operator; evaluate its physical field")
        if coefficients.shape[1] != self.state_components or math.prod(coefficients.shape[2:]) != self.matrix.shape[0]:
            raise ValueError(f"Candidate {self.name} is incompatible with coefficient shape {tuple(coefficients.shape)}")
        flattened = coefficients.flatten(2).transpose(1, 2)
        # The input controls precision; a default float64 basis must not force
        # a float32 predictor's encoded state to float64 downstream.
        dtype = torch.complex128 if flattened.dtype in {torch.float64, torch.complex128} else torch.complex64
        result = torch.einsum("moc,bmc->bmo", self.matrix.to(device=flattened.device, dtype=dtype), flattened.to(dtype))
        return result.transpose(1, 2).reshape(coefficients.shape[0], self.n_components, *coefficients.shape[2:])

    def evaluate(self, state: torch.Tensor, basis: SpectralBasis):
        if not self.requires_empirical_response:
            return basis.synthesis(self.apply_coefficients(basis.analysis(state)))
        basis._check_shape(state)
        if state.shape[1] != self.state_components:
            raise ValueError(f"{self.name} requires {self.state_components} primitive components")
        physical = state
        if "known_lift" in self.parameters:
            physical = physical + self.parameters["known_lift"].to(device=state.device, dtype=state.dtype)
        component = self.parameters.get("component", 0)
        if self.kind in {"flux", "flux_derivative"}:
            result = physical[:, component:component + 1].square() / 2
            if self.kind == "flux_derivative":
                result = _physical_derivative(result, basis, self.parameters.get("axis", 0))
        elif self.kind == "raw_gradient":
            result = physical[:, component:component + 1]
            for _ in range(self.parameters.get("order", 1)):
                result = _physical_derivative(result, basis, self.parameters.get("axis", 0))
        elif self.kind == "radial_gradient":
            scalar = physical[:, component:component + 1]
            axes = [(torch.arange(size, device=state.device, dtype=state.dtype) + .5) * length / size - length / 2
                    for size, length in zip(basis.shape, basis.lengths)]
            centered = torch.meshgrid(*axes, indexing="ij")
            radius = sum(axis.square() for axis in centered).sqrt()
            denominator = radius.clamp_min(torch.finfo(state.dtype).tiny)
            result = sum(_physical_derivative(scalar, basis, axis) * coordinate / denominator
                         for axis, coordinate in enumerate(centered))
        elif self.kind in {"divergence", "curl", "streamfunction"}:
            vector = self.parameters["vector"]
            if self.kind == "divergence":
                result = sum(_physical_derivative(physical[:, channel:channel + 1], basis, axis)
                             for axis, channel in enumerate(vector))
            else:
                if basis.ndim == 2:
                    result = (_physical_derivative(physical[:, vector[1]:vector[1] + 1], basis, 0)
                              - _physical_derivative(physical[:, vector[0]:vector[0] + 1], basis, 1))
                else:
                    curls = [_physical_derivative(physical[:, vector[second]:vector[second] + 1], basis, first)
                             - _physical_derivative(physical[:, vector[first]:vector[first] + 1], basis, second)
                             for first, second in ((1, 2), (2, 0), (0, 1))]
                    result = torch.cat(curls, dim=1)
                if self.kind == "streamfunction":
                    if basis.ndim != 2:
                        raise ValueError("A scalar streamfunction requires two spatial dimensions")
                    result = basis.spectral_power(result, -1)
                elif "curl_component" in self.parameters:
                    index = self.parameters["curl_component"]
                    result = result[:, index:index + 1]
        else:
            raise ValueError(f"Unsupported physical candidate evaluator {self.kind}")
        if self.output_scale is not None:
            result = result / self.output_scale.to(device=state.device, dtype=state.dtype).reshape(1, -1, *([1] * basis.ndim))
        return result

    def scaled(self, scales: torch.Tensor):
        """Apply frozen train-RMS scales, one strictly positive scale per output."""
        scales = torch.as_tensor(scales, device=self.matrix.device, dtype=self.matrix.real.dtype)
        if scales.shape != (self.n_components,) or not torch.isfinite(scales).all() or not (scales > 0).all():
            raise ValueError("Every candidate output requires a finite positive train-RMS scale")
        output_scale = self.output_scale
        if self.requires_empirical_response:
            output_scale = scales if output_scale is None else output_scale.to(scales) * scales
        return replace(self, matrix=self.matrix / scales[None, :, None], output_scale=output_scale)

    def with_effective_matrix(self, matrix: torch.Tensor):
        if matrix.shape != self.matrix.shape or not torch.isfinite(matrix).all():
            raise ValueError("Effective response must be finite and match candidate output/state dimensions")
        return replace(self, matrix=matrix.to(self.matrix))

    def state_dict(self):
        def cpu(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu()
            if isinstance(value, dict):
                return {key: cpu(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return type(value)(cpu(item) for item in value)
            return value
        return {"name": self.name, "matrix": cpu(self.matrix), "component_names": self.component_names,
                "primitive": self.primitive, "kind": self.kind, "parameters": cpu(self.parameters),
                "output_scale": cpu(self.output_scale)}

    @classmethod
    def from_state_dict(cls, state, *, device="cpu"):
        def move(value):
            if isinstance(value, torch.Tensor):
                return value.to(device)
            if isinstance(value, dict):
                return {key: move(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return type(value)(move(item) for item in value)
            return value
        return cls(**move(state))


def _physical_derivative(state, basis, axis):
    """Periodic spectral derivative or nonwrapping spatial finite differences."""
    if basis.is_periodic:
        return basis.derivative(state, axis)
    return torch.gradient(state, spacing=(basis.lengths[axis] / basis.shape[axis],),
                          dim=(axis + 2,), edge_order=2 if basis.shape[axis] >= 3 else 1)[0]


def _physical_candidate(name, basis, channels, kind, *, outputs=1, **parameters):
    # This matrix declares dimensions until calibration fits the effective
    # response. The actual field is always generated by the physical evaluator.
    matrix = torch.zeros((basis.mode_count, outputs, channels), device=basis.device, dtype=basis.complex_dtype)
    names = (name,) if outputs == 1 else tuple(f"{name}_{i}" for i in range(outputs))
    return FieldCandidate(name, matrix, names, kind=kind, parameters=parameters)


def _identity(basis, components):
    return torch.eye(len(components), dtype=basis.complex_dtype, device=basis.device).expand(basis.mode_count, -1, -1).clone()


def _component(matrix, index):
    return matrix[:, index:index + 1, :]


def _differential_symbols(basis, channels, vector):
    if not basis.is_periodic:
        raise ValueError("Divergence/curl modal symbols currently require a periodic vector basis; nonperiodic vector bases are not silently approximated")
    if len(vector) != basis.ndim or basis.ndim not in {1, 2, 3}:
        raise ValueError("A vector needs one component per spatial dimension")
    divergence = torch.zeros((basis.mode_count, 1, channels), device=basis.device, dtype=basis.complex_dtype)
    for axis, component in enumerate(vector):
        divergence[:, 0, component] = 1j * basis.kvectors[axis].flatten()
    if basis.ndim == 1:
        return divergence, None
    curl = torch.zeros((basis.mode_count, 1 if basis.ndim == 2 else 3, channels), device=basis.device, dtype=basis.complex_dtype)
    if basis.ndim == 2:
        curl[:, 0, vector[1]] = 1j * basis.kvectors[0].flatten()
        curl[:, 0, vector[0]] = -1j * basis.kvectors[1].flatten()
    else:
        for output, (first, second) in enumerate(((1, 2), (2, 0), (0, 1))):
            curl[:, output, vector[second]] = 1j * basis.kvectors[first].flatten()
            curl[:, output, vector[first]] = -1j * basis.kvectors[second].flatten()
    return divergence, curl


_FAMILY_ALIASES = {
    "adv": "advection", "burgers": "burgers", "diff_sorp": "diffusion_sorption",
    "diffsorp": "diffusion_sorption", "diff-sorp": "diffusion_sorption", "ds": "diffusion_sorption",
    "diff_react": "diffusion_reaction", "diffreact": "diffusion_reaction", "diff-react": "diffusion_reaction", "dr": "diffusion_reaction",
    "rdb": "radial_dam_break", "radialdambreak": "radial_dam_break",
    "ins": "incompressible_ns", "ns": "incompressible_ns", "incompressiblens": "incompressible_ns",
    "cns": "compressible_ns", "compressiblens": "compressible_ns",
}


def canonical_family(family: str) -> str:
    key = family.lower().replace(" ", "_")
    return _FAMILY_ALIASES.get(key, key)


def default_components(family: str) -> tuple[str, ...]:
    return {
        "advection": ("u",), "burgers": ("u",), "diffusion_sorption": ("u",),
        "diffusion_reaction": ("u", "v"), "radial_dam_break": ("h",),
        "incompressible_ns": ("vx", "vy"), "compressible_ns": ("rho", "vx", "vy", "p"),
    }[canonical_family(family)]


def primary_candidates(family: str, basis: SpectralBasis, components: Sequence[str] | None = None, *, include_optional: bool = False):
    """Create the paper Table-6 linear candidates (nonlinear fluxes excluded)."""
    family = canonical_family(family)
    components = tuple(default_components(family) if components is None else components)
    expected_dims = 1 if family in {"advection", "burgers", "diffusion_sorption"} else 2
    if basis.ndim != expected_dims:
        raise ValueError(f"{family} requires {expected_dims} spatial dimensions")
    expected_channels = {"advection": 1, "burgers": 1, "diffusion_sorption": 1, "diffusion_reaction": 2,
                         "radial_dam_break": 1, "incompressible_ns": 2}
    if family in expected_channels and len(components) != expected_channels[family]:
        raise ValueError(f"Wrong number of primitive components for {family}")
    if family in {"diffusion_reaction", "radial_dam_break"} and any(b != "neumann" for b in basis.boundaries):
        raise ValueError(f"{family} requires a Neumann-adapted basis")
    if family == "diffusion_sorption" and any(b != "robin" for b in basis.boundaries):
        raise ValueError("diffusion_sorption requires an explicitly configured Robin/Cauchy-compatible basis")
    identity = _identity(basis, components)
    eigenvalues = basis.eigenvalues.flatten()
    primitive = FieldCandidate("primitive", identity, components, True)
    result = [primitive]
    if family in {"advection", "burgers", "diffusion_sorption", "diffusion_reaction", "radial_dam_break"}:
        result.extend([
            FieldCandidate("lambda", identity * eigenvalues.sqrt()[:, None, None], tuple("lambda_" + v for v in components)),
            FieldCandidate("laplacian", identity * eigenvalues[:, None, None], tuple("laplacian_" + v for v in components)),
        ])
        if family == "diffusion_reaction":
            result.append(FieldCandidate("imbalance", (_component(identity, 0) - _component(identity, 1)) / math.sqrt(2), ("d",)))
            if include_optional:
                result.append(FieldCandidate("common", (_component(identity, 0) + _component(identity, 1)) / math.sqrt(2), ("s",)))
    elif family in {"incompressible_ns", "compressible_ns"}:
        component_aliases = {"density": "rho", "pressure": "p", "v_x": "vx", "v_y": "vy"}
        normalized = {component_aliases.get(name.lower(), name.lower()): i for i, name in enumerate(components)}
        if family == "incompressible_ns":
            vector = (0, 1)
        else:
            if not {"vx", "vy"}.issubset(normalized):
                raise ValueError("Compressible components must explicitly identify vx and vy")
            vector = (normalized["vx"], normalized["vy"])
        if basis.is_periodic:
            divergence, curl = _differential_symbols(basis, len(components), vector)
            if family == "compressible_ns":
                result.append(FieldCandidate("divergence", divergence, ("chi",)))
            result.append(FieldCandidate("vorticity", curl, ("omega",)))
        else:
            if family == "compressible_ns":
                result.append(_physical_candidate("divergence", basis, len(components), "divergence", vector=vector))
            result.append(_physical_candidate("vorticity", basis, len(components), "curl", vector=vector))
        if family == "incompressible_ns" and include_optional:
            if basis.is_periodic:
                inverse = torch.where(eigenvalues > 0, eigenvalues.clamp_min(torch.finfo(eigenvalues.dtype).tiny).reciprocal(), 0)
                result.append(FieldCandidate("streamfunction", curl * inverse[:, None, None], ("psi",)))
            else:
                result.append(_physical_candidate("streamfunction", basis, len(components), "streamfunction", vector=vector))
        if family == "compressible_ns":
            thermodynamic = "p" if "p" in normalized else ("rho" if "rho" in normalized else None)
            if thermodynamic is None:
                raise ValueError("Compressible states require pressure p or density rho")
            result.append(FieldCandidate("lambda_" + thermodynamic, _component(identity, normalized[thermodynamic]) * eigenvalues.sqrt()[:, None, None], ("lambda_" + thermodynamic,)))
    else:
        raise ValueError(f"Unsupported main-experiment family {family}")
    return result


def small_expert_candidates(family: str, basis: SpectralBasis, components: Sequence[str] | None = None, *, known_lift=None):
    """Table-6 primary plus empirical additions, counted as scalar channels.

    Raw gradients use physical spacing and nonwrapping finite differences on
    nonperiodic grids; the radial direction is relative to the domain centre.
    Burgers flux channels are evaluated nonlinearly on the fine state. These
    empirical fields retain their physical evaluators and use training-fitted
    effective responses for calibration.
    """
    family = canonical_family(family)
    components = tuple(default_components(family) if components is None else components)
    grouped = primary_candidates(family, basis, components, include_optional=True)
    result = []
    for candidate in grouped:
        if candidate.n_components == 1 and not candidate.primitive:
            result.append(candidate)
        else:
            for index, component_name in enumerate(candidate.component_names):
                name = "primitive_" + component_name if candidate.primitive else component_name
                result.append(replace(candidate, name=name, matrix=candidate.matrix[:, index:index + 1],
                                      component_names=(component_name,)))
    identity = _identity(basis, components)
    eigenvalues = basis.eigenvalues.flatten()
    if family == "advection":
        if basis.is_periodic:
            result.extend([FieldCandidate("gradient", identity * (1j * basis.kvectors[0].flatten())[:, None, None], ("u_x",)),
                           FieldCandidate("second_derivative", -identity * eigenvalues[:, None, None], ("u_xx",))])
        else:
            result.extend(_physical_candidate(name, basis, len(components), "raw_gradient", order=order, axis=0)
                          for name, order in (("gradient", 1), ("second_derivative", 2)))
    elif family == "burgers":
        result.extend([_physical_candidate("flux", basis, 1, "flux"),
                       _physical_candidate("flux_derivative", basis, 1, "flux_derivative", axis=0)])
    elif family == "diffusion_sorption":
        parameters = {"axis": 0}
        if known_lift is not None:
            parameters["known_lift"] = torch.as_tensor(known_lift).detach().clone()
        result.append(_physical_candidate("raw_gradient", basis, 1, "raw_gradient", **parameters))
    elif family == "radial_dam_break":
        result.append(_physical_candidate("radial_gradient", basis, 1, "radial_gradient"))
    elif family == "incompressible_ns":
        result.extend(FieldCandidate("lambda_" + name, _component(identity, i) * eigenvalues.sqrt()[:, None, None],
                                     ("lambda_" + name,)) for i, name in enumerate(components))
    elif family == "compressible_ns":
        aliases = {"density": "rho", "pressure": "p"}
        normalized = {aliases.get(name.lower(), name.lower()): i for i, name in enumerate(components)}
        if not {"rho", "p"}.issubset(normalized):
            raise ValueError("Small Expert compressible state requires both density and pressure")
        result.extend([FieldCandidate("lambda_rho", _component(identity, normalized["rho"]) * eigenvalues.sqrt()[:, None, None], ("lambda_rho",)),
                       FieldCandidate("laplacian_p", _component(identity, normalized["p"]) * eigenvalues[:, None, None], ("laplacian_p",))])
    return result


def shared_candidates(basis: SpectralBasis, components: Sequence[str], *, retained_cutoff,
                      vector_groups: Sequence[Sequence[int]] = ()):
    """Frozen library using only component roles, dimension, and boundary.

    Nine candidates per primitive scalar: identity, four positive-Laplacian
    powers (1/2, 1, 3/2, 2), and four radial bands normalized by the configured
    retained cutoff, not the discarded fine-grid maximum. Multichannel states also
    receive a DCT-II rotation across physical components, not spatial axes.
    Train-RMS normalization is performed later from the calibration split.
    Eligible vectors use boundary-compatible differential symbols; in this
    nonperiodic case their real-space derivatives have fitted effective responses.
    """
    components = tuple(components)
    if not components:
        raise ValueError("A shared library requires at least one primitive component")
    identity = _identity(basis, components)
    eigenvalues = basis.eigenvalues.flatten()
    cutoff = torch.as_tensor(retained_cutoff, device=basis.device, dtype=basis.dtype)
    if cutoff.ndim != 0 or not torch.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("retained_cutoff must be a finite positive scalar in basis frequency units")
    radius = eigenvalues.sqrt() / cutoff
    result = []
    for index, name in enumerate(components):
        row = _component(identity, index)
        result.append(FieldCandidate("primitive_" + name, row, (name,), True))
        for order in (0.5, 1.0, 1.5, 2.0):
            label = f"power_{order:g}_{name}"
            result.append(FieldCandidate(label, row * eigenvalues.pow(order)[:, None, None], (label,)))
        for band in range(4):
            label = f"band_{band + 1}_{name}"
            mask = (radius >= band / 4) & ((radius <= 1) if band == 3 else (radius < (band + 1) / 4))
            result.append(FieldCandidate(label, row * mask[:, None, None], (label,)))
    for group_index, vector in enumerate(vector_groups):
        vector = tuple(int(index) for index in vector)
        if len(set(vector)) != len(vector) or any(v < 0 or v >= len(components) for v in vector):
            raise ValueError("Vector component indices must be distinct and in range")
        if len(vector) != basis.ndim or basis.ndim not in {1, 2, 3}:
            raise ValueError("A vector needs one component per spatial dimension")
        if basis.is_periodic:
            divergence, curl = _differential_symbols(basis, len(components), vector)
            result.append(FieldCandidate(f"divergence_{group_index}", divergence, (f"divergence_{group_index}",)))
            if curl is not None:
                for component in range(curl.shape[1]):
                    name = f"curl_{group_index}_{component}"
                    result.append(FieldCandidate(name, curl[:, component:component + 1, :], (name,)))
        else:
            result.append(_physical_candidate(f"divergence_{group_index}", basis, len(components), "divergence", vector=vector))
            for component in range(0 if basis.ndim == 1 else (1 if basis.ndim == 2 else 3)):
                name = f"curl_{group_index}_{component}"
                result.append(_physical_candidate(name, basis, len(components), "curl", vector=vector, curl_component=component))
    if len(components) > 1:
        count = len(components)
        for mode in range(count):
            positions = torch.arange(count, dtype=basis.dtype, device=basis.device) + 0.5
            weights = torch.cos(math.pi * mode * positions / count) * math.sqrt((1 if mode == 0 else 2) / count)
            matrix = weights[None, None, :].expand(basis.mode_count, 1, -1).to(basis.complex_dtype).clone()
            result.append(FieldCandidate(f"component_dct_{mode}", matrix, (f"component_dct_{mode}",)))
    return result
