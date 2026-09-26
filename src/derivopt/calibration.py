"""Training-only effective-channel calibration and Gaussian field fusion.

The model estimates input-state reconstruction risk with full component
covariance and PSD-safe conditioning. Rollout performance is evaluated separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .fields import FieldCandidate
from .operators import SpectralBasis, SpectralResampler
from .quantization import ScalarQuantizer


def _modal(coefficients):
    return coefficients.flatten(2).transpose(1, 2)


def _hermitian(matrix):
    return (matrix + matrix.mH) / 2


def _psd(matrix):
    values, vectors = torch.linalg.eigh(_hermitian(matrix))
    return (vectors * values.clamp_min(0).unsqueeze(-2)) @ vectors.mH


def _sqrt_psd(matrix):
    values, vectors = torch.linalg.eigh(_hermitian(matrix))
    return (vectors * values.clamp_min(0).sqrt().unsqueeze(-2)) @ vectors.mH


def _pool(values: torch.Tensor, shells: torch.Tensor):
    """Average first-axis modes within each shell, retaining matrix structure."""
    output = torch.zeros_like(values)
    for shell in torch.unique(shells):
        mask = shells == shell
        output[mask] = values[mask].mean(dim=0)
    return output


def _shell_ids(basis, shellwise, shell_count):
    if not shellwise:
        return torch.arange(basis.mode_count, device=basis.device)
    radius = basis.eigenvalues.sqrt().flatten()
    # Integer radial shells in units of the smallest positive mode frequency.
    positive = radius[radius > 0]
    if not len(positive):
        return torch.zeros_like(radius, dtype=torch.long)
    if shell_count is None:
        return torch.round(radius / positive.min()).long()
    if not isinstance(shell_count, int) or shell_count < 1:
        raise ValueError("shell_count must be a positive integer")
    return (radius / radius.max() * shell_count).floor().long().clamp_max(shell_count - 1)


@dataclass
class CalibratedState:
    basis: SpectralBasis
    resampler: SpectralResampler
    candidates: list[FieldCandidate]
    quantizers: list[ScalarQuantizer]
    prior: torch.Tensor
    prior_mean: torch.Tensor
    precisions: list[dict[int, torch.Tensor]]
    transfers: list[dict[int, torch.Tensor]]
    noises: list[dict[int, torch.Tensor]]
    offsets: list[dict[int, torch.Tensor]]
    weights: torch.Tensor
    rms_scales: list[torch.Tensor]
    shell_ids: torch.Tensor
    metadata: dict

    @property
    def candidate_names(self):
        return [candidate.name for candidate in self.candidates]

    @property
    def component_counts(self):
        return [candidate.n_components for candidate in self.candidates]

    @property
    def transfer(self):
        return self.transfers

    @property
    def noise(self):
        return self.noises

    def design_problem(self):
        from .selectors import DesignProblem
        return DesignProblem(self.prior, self.precisions, self.component_counts, self.candidate_names, weights=self.weights)

    def _validate_bits(self, design_bits: Sequence[int]):
        bits = tuple(int(value) for value in design_bits)
        if len(bits) != len(self.candidates):
            raise ValueError("design_bits must have one integer allocation per candidate")
        if any(value < 0 or (value and value not in precision) for value, precision in zip(bits, self.precisions)):
            raise ValueError("Allocation uses an uncalibrated or negative bit width")
        if any(isinstance(raw, bool) or raw != value for raw, value in zip(design_bits, bits)):
            raise ValueError("Bit allocations must be integers")
        return bits

    def encode_fields(self, state: torch.Tensor, design_bits: Sequence[int], *, straight_through: bool = False):
        """Fine candidate generation -> low-pass/sample -> component quantize.

        The returned list contains active fields only, in candidate order.
        Actual byte serialization can use these same fixed quantizers through
        ExplicitRecordCodec; this function preserves optional training gradients.
        """
        bits = self._validate_bits(design_bits)
        fields = []
        for candidate, quantizer, width in zip(self.candidates, self.quantizers, bits):
            if width:
                fine_field = candidate.evaluate(state, self.basis)
                fields.append(quantizer(self.resampler.coarsen(fine_field), width, straight_through=straight_through))
        return fields

    def decode_fields(self, decoded_coarse_list: Sequence[torch.Tensor | None], design_bits: Sequence[int]):
        """Fuse decoded coarse observations using only persisted statistics.

        Accepts either active fields only or a candidate-aligned list with None
        in omitted positions. Empty designs require at least one batch-shaped
        placeholder; normal benchmark designs always store an active field.
        """
        bits = self._validate_bits(design_bits)
        active = [index for index, width in enumerate(bits) if width]
        if len(decoded_coarse_list) == len(self.candidates):
            fields = [decoded_coarse_list[index] for index in active]
        elif len(decoded_coarse_list) == len(active):
            fields = list(decoded_coarse_list)
        else:
            raise ValueError("Decoded fields must be active-only or aligned with all candidates")
        if not fields or any(value is None for value in fields):
            raise ValueError("At least one active decoded field is required for batch shape")
        first = fields[0]
        device = first.device
        # Preserve the calibrated precision for singular-prior conditioning;
        # float32 eigensolver roundoff can turn a transverse nullspace into
        # spurious longitudinal information at high bit widths. The returned
        # physical state still uses the predictor's original float dtype.
        dtype = self.prior.dtype
        prior = self.prior.to(device=device, dtype=dtype)
        mean = self.prior_mean.to(device=device, dtype=dtype)
        batch = first.shape[0]
        precision = torch.zeros_like(prior)
        information = torch.zeros((batch, *mean.shape), device=device, dtype=dtype)
        outside = (self.resampler.interpolation_mask & ~self.resampler.expressible_mask).flatten().to(device)
        extension_matrices, extension_observations = [], []
        for index, coarse_field in zip(active, fields):
            width = bits[index]
            candidate = self.candidates[index]
            expected_shape = (batch, candidate.n_components, *self.resampler.coarse_shape)
            if tuple(coarse_field.shape) != expected_shape or not torch.isfinite(coarse_field).all():
                raise ValueError(f"Decoded field {candidate.name} expected finite shape {expected_shape}")
            observation = _modal(self.basis.analysis(self.resampler.decode(coarse_field))).to(dtype)
            transfer = self.transfers[index][width].to(device=device, dtype=dtype)
            matrix = candidate.matrix.to(device=device, dtype=dtype) * transfer[:, None, None]
            offset = self.offsets[index][width].to(device=device, dtype=dtype)
            noise = self.noises[index][width].to(device=device, dtype=self.prior.real.dtype)
            centered = observation - offset[None] - torch.einsum("moc,mc->mo", matrix, mean)[None]
            precision += self.precisions[index][width].to(device=device, dtype=dtype)
            information += torch.einsum("mco,bmo->bmc", matrix.mH, centered / noise[None, :, None])
            if bool(outside.any()):
                # Outside the retained band, use a noise-weighted minimum-norm
                # inverse of the stored field's physical response. This extension
                # uses the observed coefficients and frozen calibration, without
                # the low-pass transfer B or a prior contribution on these modes.
                scale = noise[outside].sqrt()
                extension_matrices.append(candidate.matrix.to(device=device, dtype=dtype)[outside] / scale[:, None, None])
                extension_observations.append(observation[:, outside] / scale[None, :, None])
        root = _sqrt_psd(prior)
        identity = torch.eye(prior.shape[-1], dtype=dtype, device=device).expand_as(prior)
        system = _hermitian(identity + root @ precision @ root)
        rhs = torch.einsum("mcd,bmd->mbc", root, information)
        correction = torch.linalg.solve(system, rhs.transpose(-1, -2)).transpose(-1, -2)
        correction = torch.einsum("mcd,mbd->bmc", root, correction)
        posterior_mean = mean[None] + correction
        if extension_matrices:
            physical_matrix = torch.cat(extension_matrices, dim=1)
            physical_observation = torch.cat(extension_observations, dim=2)
            inverse = torch.linalg.pinv(physical_matrix, rtol=1e-10)
            extended = torch.einsum("mco,bmo->bmc", inverse, physical_observation)
            posterior_mean = posterior_mean.clone()
            posterior_mean[:, outside] = extended
        coefficients = posterior_mean.transpose(1, 2).reshape(batch, mean.shape[-1], *self.basis.shape)
        result = self.basis.synthesis(coefficients).to(first.dtype)
        if not torch.isfinite(result).all():
            raise FloatingPointError("Non-finite calibrated posterior reconstruction")
        return result

    def decode_primitive_fields(self, decoded_coarse_list: Sequence[torch.Tensor], design_bits: Sequence[int]):
        """Paper's primitive baseline: inverse scaling and trig interpolation.

        The quantizer already reverses its componentwise standardization.
        Only candidate RMS scaling remains to be reversed here. No prior,
        fitted offset, Wiener shrinkage, or divergence projection is applied.
        Both grouped and separately stored primitive components are supported.
        """
        bits = self._validate_bits(design_bits)
        active = [index for index, width in enumerate(bits) if width]
        if not active or len(decoded_coarse_list) != len(active):
            raise ValueError("Primitive decoding requires all active primitive observations")
        channels = self.prior.shape[-1]
        first = decoded_coarse_list[0]
        primitive = first.new_zeros((first.shape[0], channels, *self.resampler.coarse_shape))
        seen = []
        for index, observation in zip(active, decoded_coarse_list):
            candidate = self.candidates[index]
            if not candidate.primitive:
                raise ValueError("Direct primitive decoding cannot contain derived fields")
            expected_shape = (first.shape[0], candidate.n_components, *self.resampler.coarse_shape)
            if tuple(observation.shape) != expected_shape or not torch.isfinite(observation).all():
                raise ValueError(f"Primitive observation expected finite shape {expected_shape}")
            scale = self.rms_scales[index]
            raw_matrix = candidate.matrix * scale.to(candidate.matrix.real.dtype)[None, :, None]
            component_indices = raw_matrix[0].abs().argmax(dim=-1)
            expected = torch.eye(channels, device=raw_matrix.device, dtype=raw_matrix.dtype)[component_indices]
            if not torch.allclose(raw_matrix, expected[None].expand_as(raw_matrix), rtol=1e-10, atol=1e-12):
                raise ValueError("Primitive candidates must be scaled coordinate selectors")
            seen.extend(component_indices.tolist())
            restored = observation * scale.to(observation).reshape(1, -1, *([1] * self.basis.ndim))
            primitive = primitive.index_copy(1, component_indices.to(primitive.device), restored)
        if sorted(seen) != list(range(channels)):
            raise ValueError("Primitive decoding must carry every primitive component exactly once")
        return self.resampler.decode(primitive)

    def reconstruct(self, state: torch.Tensor, design_bits: Sequence[int], *, straight_through: bool = False):
        return self.decode_fields(self.encode_fields(state, design_bits, straight_through=straight_through), design_bits)

    def state_dict(self):
        cpu = lambda tensor: tensor.detach().cpu()
        dictionaries = lambda items: [{int(key): cpu(value) for key, value in entry.items()} for entry in items]
        return {
            "format_version": 3,
            "basis": {"shape": self.basis.shape, "lengths": self.basis.lengths, "boundary": self.basis.boundaries,
                      "robin_coefficients": self.basis.robin_coefficients, "dtype": str(self.basis.dtype).split(".")[-1]},
            "coarse_shape": self.resampler.coarse_shape,
            "candidates": [candidate.state_dict() for candidate in self.candidates],
            "quantizers": [quantizer.state_dict() for quantizer in self.quantizers],
            "prior": cpu(self.prior), "prior_mean": cpu(self.prior_mean),
            "precisions": dictionaries(self.precisions), "transfers": dictionaries(self.transfers),
            "noises": dictionaries(self.noises), "offsets": dictionaries(self.offsets),
            "weights": cpu(self.weights), "rms_scales": [cpu(scale) for scale in self.rms_scales],
            "shell_ids": cpu(self.shell_ids), "metadata": self.metadata,
        }

    @classmethod
    def from_state_dict(cls, state, *, device="cpu"):
        if state.get("format_version") != 3:
            raise ValueError("Unsupported calibration checkpoint version; recalibrate with the closed radial retained-band / full interpolation protocol (version 3)")
        specification = dict(state["basis"])
        specification["dtype"] = getattr(torch, specification["dtype"])
        basis = SpectralBasis(**specification, device=device)
        tensors = lambda items: [{int(key): value.to(device) for key, value in entry.items()} for entry in items]
        return cls(
            basis, SpectralResampler(basis, state["coarse_shape"]),
            [FieldCandidate.from_state_dict(entry, device=device) for entry in state["candidates"]],
            [ScalarQuantizer(**{key: value.to(device) for key, value in entry.items()}) for entry in state["quantizers"]],
            state["prior"].to(device), state["prior_mean"].to(device), tensors(state["precisions"]),
            tensors(state["transfers"]), tensors(state["noises"]), tensors(state["offsets"]), state["weights"].to(device),
            [scale.to(device) for scale in state["rms_scales"]], state["shell_ids"].to(device), state["metadata"],
        )


@torch.no_grad()
def calibrate(
    training: torch.Tensor,
    basis: SpectralBasis,
    resampler: SpectralResampler,
    candidates: Sequence[FieldCandidate],
    bit_widths: Sequence[int] = (1, 2, 3, 4, 6, 8),
    *, split: str = "train", shellwise: bool = True, shell_count: int | None = None,
    weights: torch.Tensor | None = None, divergence_free: bool = False,
    rms_normalize: bool = True, noise_floor: float = 1e-8,
    budget_per_site: int | None = None,
) -> CalibratedState:
    """Fit spectra, shell transfer, and effective quantizer residual on training.

    ``training`` contains independent training states [N,C,*fine]. Statistical
    quality is not implied by passing the minimum N>=2 shape check. Supplied
    training provenance is enforced by the calling dataset split as well.
    """
    if split != "train":
        raise ValueError("Calibration must use the training split only")
    basis._check_shape(training)
    if training.shape[0] < 2 or not torch.isfinite(training).all():
        raise ValueError("Calibration requires at least two finite training states")
    if not candidates or any(candidate.state_components != training.shape[1] for candidate in candidates):
        raise ValueError("Candidate library must match the primitive state components")
    if tuple(resampler.fine_basis.shape) != basis.shape or resampler.fine_basis.lengths != basis.lengths:
        raise ValueError("Resampler and calibration basis must agree")
    if not noise_floor > 0:
        raise ValueError("noise_floor must be positive")
    widths = tuple(sorted(set(int(value) for value in bit_widths)))
    if not widths or any(value < 1 for value in widths) or any(isinstance(raw, bool) or raw != int(raw) for raw in bit_widths):
        raise ValueError("Calibration bit widths must be positive integers")
    if budget_per_site is not None and (isinstance(budget_per_site, bool) or not isinstance(budget_per_site, int) or budget_per_site < 1):
        raise ValueError("budget_per_site must be a positive integer when supplied")
    # Double precision keeps PSD/singular-covariance algebra stable; inference
    # casts the final physical result back to the predictor precision.
    training = training.to(device=basis.device, dtype=torch.float64)
    coefficients = _modal(basis.analysis(training)).to(torch.complex128)
    expressible = resampler.expressible_mask.flatten().to(training.device)
    shells = _shell_ids(basis, shellwise, shell_count)
    # Separate unobservable modes from observed modes in shell pooling.
    shells = shells * 2 + (~expressible).long()
    projector = None
    if divergence_free:
        if not basis.is_periodic or training.shape[1] != basis.ndim or basis.ndim not in {2, 3}:
            raise ValueError("Divergence-free calibration requires periodic velocity with ndim components")
        k = torch.stack([axis.flatten() for axis in basis.kvectors], dim=-1).to(torch.float64)
        norm2 = k.square().sum(-1)
        identity = torch.eye(basis.ndim, device=training.device, dtype=torch.float64).expand(basis.mode_count, -1, -1)
        longitudinal = k[:, :, None] * k[:, None, :] / norm2.clamp_min(torch.finfo(torch.float64).tiny)[:, None, None]
        projector = (identity - longitudinal).to(torch.complex128)
        coefficients = torch.einsum("mcd,bmd->bmc", projector, coefficients)
        # Candidate observations and the prior must describe the same state;
        # do not estimate observations from unprojected velocities after
        # explicitly requesting the incompressible latent model.
        training = basis.synthesis(coefficients.transpose(1, 2).reshape_as(training))
    mean = coefficients.mean(0)
    centered = coefficients - mean[None]
    covariance = torch.einsum("bmc,bmd->mcd", centered, centered.conj()) / training.shape[0]
    empirical_covariance = covariance
    response_inverse = response_scale = None
    if any(candidate.requires_empirical_response for candidate in candidates):
        response_scale = training.square().mean((0, *range(2, training.ndim))).sqrt().clamp_min(1e-12)
        normalized_covariance = empirical_covariance / (response_scale[None, :, None] * response_scale[None, None, :])
        # A per-mode relative pseudoinverse alone treats roundoff-only modes as
        # a full-strength regression source. Use an absolute numerical rank
        # tolerance referenced to training signal, after removing channel units.
        covariance_scale = torch.linalg.matrix_norm(normalized_covariance, ord="fro").max()
        rank_tolerance = 64 * torch.finfo(torch.float64).eps * covariance_scale
        response_inverse = torch.linalg.pinv(_hermitian(normalized_covariance), hermitian=True,
                                            atol=rank_tolerance, rtol=1e-10)
    if divergence_free and shellwise:
        rank = projector.diagonal(dim1=-2, dim2=-1).real.sum(-1).clamp_min(1)
        energy = covariance.diagonal(dim1=-2, dim2=-1).real.sum(-1) / rank
        covariance = _pool(energy, shells)[:, None, None] * projector
    elif shellwise:
        covariance = _pool(covariance, shells)
    covariance = _psd(covariance) * expressible[:, None, None]
    mean = mean * expressible[:, None]
    if weights is None:
        weights = expressible.to(torch.float64)
    else:
        weights = torch.as_tensor(weights, device=training.device, dtype=torch.float64).flatten()
        if weights.shape != (basis.mode_count,) or not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("weights must contain one finite nonnegative value per fine-grid mode")
        weights = weights * expressible
    result_candidates, quantizers, rms_scales = [], [], []
    precisions, transfers, noises, offsets = [], [], [], []
    for candidate in candidates:
        feasible_widths = tuple(width for width in widths if budget_per_site is None or candidate.n_components * width <= budget_per_site)
        if not feasible_widths:
            raise ValueError(f"Candidate {candidate.name} has no feasible supplied bit width under budget_per_site={budget_per_site}; omit it or supply a feasible width")
        fine_field = candidate.evaluate(training, basis)
        dimensions = (0, *range(2, fine_field.ndim))
        rms = fine_field.square().mean(dimensions).sqrt().clamp_min(1e-8) if rms_normalize else torch.ones(candidate.n_components, device=training.device, dtype=torch.float64)
        normalized = candidate.scaled(rms)
        fine_field = normalized.evaluate(training, basis)
        if normalized.requires_empirical_response:
            generated = _modal(basis.analysis(fine_field)).to(torch.complex128)
            generated_centered = generated - generated.mean(0)[None]
            cross_covariance = torch.einsum("bmo,bmc->moc", generated_centered, centered.conj()) / training.shape[0]
            cross_roundoff = (64 * torch.finfo(torch.float64).eps
                              * generated_centered.abs().square().mean(0).sqrt()[:, :, None]
                              * empirical_covariance.diagonal(dim1=-2, dim2=-1).real.clamp_min(0).sqrt()[:, None, :])
            cross_covariance = torch.where(cross_covariance.abs() > cross_roundoff, cross_covariance, 0)
            response = ((cross_covariance / response_scale[None, None, :]) @ response_inverse
                        / response_scale[None, None, :])
            normalized = normalized.with_effective_matrix(response)
        coarse = resampler.coarsen(fine_field)
        quantizer = ScalarQuantizer.fit(coarse, split="train")
        ideal = _modal(basis.analysis(resampler.project(fine_field))).to(torch.complex128)
        ideal_mean = ideal.mean(0)
        ideal_centered = ideal - ideal_mean[None]
        # B_c describes the spatial coarse channel, independent of b_c. Fit it
        # once before quantization; bit width changes only effective noise and
        # the quantizer bias. This matches B_c(ell) versus Sigma_c(ell;b_c).
        unquantized = _modal(basis.analysis(resampler.decode(coarse))).to(torch.complex128)
        unquantized_centered = unquantized - unquantized.mean(0)[None]
        energy = ideal_centered.abs().square().mean(dim=(0, 2))
        cross = (unquantized_centered * ideal_centered.conj()).mean(dim=(0, 2))
        if shellwise:
            energy, cross = _pool(energy, shells), _pool(cross, shells)
        transfer = torch.where(energy > noise_floor, cross / energy.clamp_min(noise_floor), 0) * expressible
        matrix = normalized.matrix.to(torch.complex128) * transfer[:, None, None]
        # Empirical nonlinear/non-diagonal candidates retain their true field
        # evaluator, while this fitted linear modal response is only the risk
        # model. Its approximation residual must be counted with quantization.
        modeled = torch.einsum("moc,bmc->bmo", normalized.matrix.to(torch.complex128), coefficients)
        modeled_mean = modeled.mean(0)
        modeled_centered = modeled - modeled_mean[None]
        unscaled_precision = _hermitian(matrix.mH @ matrix)
        candidate_precisions, candidate_transfers, candidate_noises, candidate_offsets = {}, {}, {}, {}
        for width in feasible_widths:
            decoded = quantizer(coarse, width)
            observed = _modal(basis.analysis(resampler.decode(decoded))).to(torch.complex128)
            observed_mean = observed.mean(0)
            observed_centered = observed - observed_mean[None]
            residual = observed_centered - transfer[None, :, None] * modeled_centered
            variance = residual.abs().square().mean(dim=(0, 2))
            if shellwise:
                variance = _pool(variance, shells)
            variance = variance.clamp_min(noise_floor)
            offset = observed_mean - transfer[:, None] * modeled_mean
            precision = unscaled_precision / variance[:, None, None]
            if not all(torch.isfinite(value).all() for value in (transfer, variance, offset, precision)):
                raise FloatingPointError(f"Non-finite effective calibration for {candidate.name}, {width} bits")
            candidate_precisions[width] = precision
            candidate_transfers[width] = transfer.clone()
            candidate_noises[width] = variance
            candidate_offsets[width] = offset
        result_candidates.append(normalized)
        quantizers.append(quantizer)
        rms_scales.append(rms)
        precisions.append(candidate_precisions)
        transfers.append(candidate_transfers)
        noises.append(candidate_noises)
        offsets.append(candidate_offsets)
    return CalibratedState(
        basis, resampler, result_candidates, quantizers, covariance, mean,
        precisions, transfers, noises, offsets, weights, rms_scales, shells,
        {"split": "train", "training_states": training.shape[0], "shellwise": shellwise,
         "shell_count": shell_count, "divergence_free": divergence_free, "rms_normalize": rms_normalize,
         "noise_floor": noise_floor, "bit_widths": widths, "covariance_ddof": 0,
         "affine_transfer_bias": True, "outside_expressible_prior": "zero",
         "out_of_band_decoder": "physical_field_least_squares",
         "out_of_band_response": "analytic_symbol_or_train_fitted_effective_response",
         "transfer_calibration": "unquantized_coarse_channel", "transfer_bit_independent": True,
         "retained_support": "boundary_adapted_radial",
         "interpolation_support": "full_coarse_trigonometric_or_boundary_adapted",
         "budget_per_site": budget_per_site},
    )
