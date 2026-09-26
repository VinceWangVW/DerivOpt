"""Boundary-adapted, differentiable spectral analysis on rectangular grids.

All transforms use an orthonormal convention. Non-periodic coordinates are
cell centres; Robin coefficients specify ``outward_derivative(u) + beta*u=0``.
The Robin basis diagonalizes the symmetric cell-centred finite-volume operator.
Inhomogeneous boundary data must first be lifted.
"""

from __future__ import annotations

import math
from functools import reduce
from operator import mul
from typing import Sequence

import torch


class SpectralBasis:
    """Tensor-product Fourier, Neumann cosine, or homogeneous Robin basis.

    ``lengths`` are physical domain lengths, not coordinate maxima. For a
    uniform cell-centred coordinate vector, length = spacing * sample count.
    A boundary can be specified per axis. Robin coefficients can be a pair
    for a 1D domain or one (left, right) pair per axis. ``float('inf')`` denotes
    homogeneous Dirichlet on that side. Negative Robin coefficients are not
    accepted because fractional powers require a nonnegative operator.
    """

    def __init__(
        self,
        shape: Sequence[int],
        lengths: Sequence[float] | None = None,
        boundary: str | Sequence[str] = "periodic",
        *,
        robin_coefficients: Sequence | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        self.shape = tuple(int(n) for n in shape)
        if not self.shape or any(n < 2 for n in self.shape):
            raise ValueError("Every spatial dimension must contain at least two samples")
        self.ndim = len(self.shape)
        self.lengths = tuple(float(v) for v in (lengths or (1.0,) * self.ndim))
        if len(self.lengths) != self.ndim or any(not math.isfinite(v) or v <= 0 for v in self.lengths):
            raise ValueError("lengths must have one positive physical length per spatial axis")
        self.boundaries = (boundary,) * self.ndim if isinstance(boundary, str) else tuple(boundary)
        if len(self.boundaries) != self.ndim or any(b not in {"periodic", "neumann", "robin"} for b in self.boundaries):
            raise ValueError("boundary must be periodic, neumann, or robin for each spatial axis")
        if dtype not in {torch.float32, torch.float64}:
            raise TypeError("Basis dtype must be float32 or float64")
        self.device = torch.device(device)
        self.dtype = dtype
        self.complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        self.mode_count = reduce(mul, self.shape, 1)
        self.modal_shape = self.shape
        if robin_coefficients is not None and self.ndim == 1 and len(robin_coefficients) == 2 and all(isinstance(v, (int, float)) for v in robin_coefficients):
            robin_coefficients = (robin_coefficients,)
        if "robin" in self.boundaries and (robin_coefficients is None or len(robin_coefficients) != self.ndim):
            raise ValueError("Robin boundaries require explicit (left, right) coefficients per axis")
        self.robin_coefficients = robin_coefficients
        self._matrices: list[torch.Tensor | None] = []
        axis_eigenvalues = []
        axis_frequencies = []
        derivative_frequencies = []
        self.axis_operators: list[torch.Tensor | None] = []
        for axis, (n, length, kind) in enumerate(zip(self.shape, self.lengths, self.boundaries)):
            if kind == "periodic":
                frequencies = 2 * math.pi * torch.fft.fftfreq(n, d=length / n, device=self.device, dtype=dtype)
                derivative = frequencies.clone()
                # An even-grid Nyquist sine has no real nodal representation.
                if n % 2 == 0:
                    derivative[n // 2] = 0
                eigenvalues = frequencies.square()
                matrix = operator = None
            elif kind == "neumann":
                modes = torch.arange(n, dtype=dtype, device=self.device)
                locations = (torch.arange(n, dtype=dtype, device=self.device) + 0.5) / n
                matrix = torch.cos(math.pi * locations[:, None] * modes[None, :]) * math.sqrt(2 / n)
                matrix[:, 0] /= math.sqrt(2)
                frequencies = math.pi * modes / length
                eigenvalues = frequencies.square()
                derivative = None
                operator = matrix @ torch.diag(eigenvalues) @ matrix.T
            else:
                coefficients = tuple(float(v) for v in robin_coefficients[axis])
                if len(coefficients) != 2 or any(math.isnan(v) or v < 0 for v in coefficients):
                    raise ValueError("Robin coefficients must be a nonnegative (left, right) pair")
                h = length / n
                # Face fluxes yield a symmetric positive-semidefinite stiffness.
                operator = torch.diag(torch.full((n,), 2 / h**2, dtype=dtype, device=self.device))
                offdiag = torch.full((n - 1,), -1 / h**2, dtype=dtype, device=self.device)
                operator += torch.diag(offdiag, 1) + torch.diag(offdiag, -1)
                operator[0, 0] = operator[-1, -1] = 1 / h**2
                for index, beta in ((0, coefficients[0]), (-1, coefficients[1])):
                    # Boundary face is half a grid spacing from the cell centre.
                    correction = 2 / h**2 if math.isinf(beta) else beta / (h * (1 + beta * h / 2))
                    operator[index, index] += correction
                eigenvalues, matrix = torch.linalg.eigh(operator)
                eigenvalues = eigenvalues.clamp_min(0)
                if coefficients == (0.0, 0.0):
                    # Preserve the exact Neumann nullspace rather than turn
                    # eigensolver roundoff into artificial DC detail energy.
                    eigenvalues[0] = 0
                # Fix eigenvector sign for stable serialization and diagnostics.
                pivots = matrix.abs().argmax(dim=0)
                signs = torch.sign(matrix[pivots, torch.arange(n, device=self.device)])
                matrix *= signs[None, :]
                frequencies = eigenvalues.sqrt()
                derivative = None
            self._matrices.append(matrix)
            self.axis_operators.append(operator)
            axis_eigenvalues.append(eigenvalues)
            axis_frequencies.append(frequencies)
            derivative_frequencies.append(derivative)
        self.axis_eigenvalues = tuple(axis_eigenvalues)
        self.eigenvalues = sum(torch.meshgrid(*axis_eigenvalues, indexing="ij"))
        self.frequencies = tuple(torch.meshgrid(*axis_frequencies, indexing="ij"))
        self.kvectors = tuple(torch.meshgrid(*derivative_frequencies, indexing="ij")) if self.is_periodic else None

    @property
    def is_periodic(self) -> bool:
        return all(b == "periodic" for b in self.boundaries)

    @classmethod
    def from_coordinates(cls, coordinates: Sequence[torch.Tensor], boundary="periodic", **kwargs):
        """Construct from explicitly cell-centred, uniformly spaced coordinates."""
        coords = [torch.as_tensor(v, dtype=torch.float64) for v in coordinates]
        for c in coords:
            if c.ndim != 1 or c.numel() < 2 or not torch.isfinite(c).all():
                raise ValueError("Coordinate arrays must be finite, 1D, and have at least two points")
            diffs = c.diff()
            if not (diffs > 0).all() or not torch.allclose(diffs, diffs[:1].expand_as(diffs), rtol=1e-5, atol=1e-10):
                raise ValueError("This rectangular-grid implementation requires uniform increasing coordinates")
        return cls([len(c) for c in coords], [float((c[1] - c[0]) * len(c)) for c in coords], boundary, **kwargs)

    def _check_shape(self, x: torch.Tensor):
        if x.ndim != self.ndim + 2 or tuple(x.shape[2:]) != self.shape:
            raise ValueError(f"Expected [batch, channels, {self.shape}], got {tuple(x.shape)}")

    def analysis(self, state: torch.Tensor) -> torch.Tensor:
        self._check_shape(state)
        result = state
        for axis, (kind, matrix) in enumerate(zip(self.boundaries, self._matrices), start=2):
            if kind == "periodic":
                result = torch.fft.fft(result, dim=axis, norm="ortho")
            else:
                transform = matrix.to(device=result.device, dtype=result.dtype)
                result = (result.movedim(axis, -1) @ transform).movedim(-1, axis)
        return result

    def synthesis(self, coefficients: torch.Tensor, *, real: bool = True) -> torch.Tensor:
        self._check_shape(coefficients)
        result = coefficients
        for axis in reversed(range(self.ndim)):
            kind, matrix = self.boundaries[axis], self._matrices[axis]
            dimension = axis + 2
            if kind == "periodic":
                result = torch.fft.ifft(result, dim=dimension, norm="ortho")
            else:
                transform = matrix.to(device=result.device, dtype=result.dtype)
                result = (result.movedim(dimension, -1) @ transform.T).movedim(-1, dimension)
        return result.real if real and result.is_complex() else result

    def spectral_power(self, state: torch.Tensor, order: float) -> torch.Tensor:
        """Apply the positive spatial operator (-Delta) ** order.

        The paper's Lambda is order=1/2 and its positive Delta symbol is order=1.
        Negative orders use the Moore-Penrose value zero on the nullspace.
        """
        if not math.isfinite(order):
            raise ValueError("order must be finite")
        values = self.eigenvalues.to(device=state.device, dtype=state.dtype)
        multiplier = values.pow(order) if order >= 0 else torch.where(values > 0, values.clamp_min(torch.finfo(values.dtype).tiny).pow(order), 0)
        return self.synthesis(self.analysis(state) * multiplier)

    def derivative(self, state: torch.Tensor, axis: int) -> torch.Tensor:
        """Periodic spatial derivative; nonperiodic derivatives change basis.

        A cosine derivative belongs to a sine basis, so the derivative is not
        diagonal in the Neumann basis. Nonperiodic detail candidates use the
        self-adjoint boundary-adapted powers above.
        """
        if not self.is_periodic:
            raise ValueError("A derivative is not diagonal in this nonperiodic scalar basis")
        if axis < 0 or axis >= self.ndim:
            raise ValueError("Derivative axis is outside the spatial dimensions")
        return self.synthesis(self.analysis(state) * (1j * self.kvectors[axis].to(state.device)))

    def band_mask(self, lower: float, upper: float) -> torch.Tensor:
        """Radial normalized spectral bands; final upper endpoint is inclusive."""
        if not 0 <= lower < upper <= 1:
            raise ValueError("Band edges must satisfy 0 <= lower < upper <= 1")
        radius = self.eigenvalues.sqrt()
        radius = radius / radius.max().clamp_min(torch.finfo(radius.dtype).tiny)
        return (radius >= lower) & ((radius <= upper) if upper == 1 else (radius < upper))


class SpectralResampler:
    """Anti-aliased coarsening and boundary-compatible modal interpolation.

    Fine-grid fields are transformed first, unrepresentable modes removed, and
    retained modes synthesized on the coarse physical grid. Encoding uses a
    boundary-adapted radial cutoff, separately
    from the full tensor-product support of coarse spectral interpolation.
    Thus quantization can create out-of-band corner energy on decode.
    The closed retained ball includes periodic Nyquist boundary modes. Coarse
    sampling sums their two fine-grid signs; a boundary cosine survives, while
    its sine is unidentifiable on the coarse nodes. Decode uses the standard
    cosine convention, splitting a Nyquist coefficient equally between signs.

    Neumann and Robin axes match the ordered modes of their self-adjoint
    operator. Robin's finite-volume eigenfunctions depend on resolution, so
    interpolation matches discrete mode indices between the two grids rather
    than samples of a single continuum eigenfunction.
    """

    def __init__(self, fine_basis: SpectralBasis, coarse_shape: Sequence[int]):
        coarse_shape = tuple(int(n) for n in coarse_shape)
        if len(coarse_shape) != fine_basis.ndim or any(n < 2 or n > m for n, m in zip(coarse_shape, fine_basis.shape)):
            raise ValueError("coarse_shape must have one size between 2 and the fine size per axis")
        self.fine_basis = fine_basis
        self.coarse_shape = coarse_shape
        self.coarse_basis = SpectralBasis(
            coarse_shape, fine_basis.lengths, fine_basis.boundaries,
            robin_coefficients=fine_basis.robin_coefficients,
            device=fine_basis.device, dtype=fine_basis.dtype,
        )
        self._fine_indices = []
        self._coarse_indices = []
        self._interpolation_fine_indices = []
        self._interpolation_coarse_indices = []
        self._interpolation_weights = []
        axis_masks = []
        interpolation_masks = []
        for nf, nc, boundary in zip(fine_basis.shape, coarse_shape, fine_basis.boundaries):
            if boundary == "periodic" and nc != nf:
                # The closed ball includes both signs at an even coarse
                # Nyquist. Sampling folds their coefficients onto one mode.
                maximum = nc // 2
                signed_modes = torch.arange(-maximum, maximum + 1, device=fine_basis.device)
                fine_indices = signed_modes.remainder(nf).long()
                coarse_indices = signed_modes.remainder(nc).long()
            else:
                fine_indices = coarse_indices = torch.arange(nc, device=fine_basis.device)
            self._fine_indices.append(fine_indices)
            self._coarse_indices.append(coarse_indices)
            mask = torch.zeros(nf, dtype=torch.bool, device=fine_basis.device)
            mask[fine_indices] = True
            axis_masks.append(mask)
            if boundary == "periodic" and nc != nf:
                signed = torch.arange(-(nc // 2), nc // 2 + 1, device=fine_basis.device)
                interpolation_fine = signed.remainder(nf).long()
                interpolation_coarse = signed.remainder(nc).long()
                weights = torch.ones(len(signed), dtype=fine_basis.dtype, device=fine_basis.device)
                if nc % 2 == 0:
                    weights[[0, -1]] = 0.5
            else:
                interpolation_fine, interpolation_coarse = fine_indices, coarse_indices
                weights = torch.ones(len(fine_indices), dtype=fine_basis.dtype, device=fine_basis.device)
            self._interpolation_fine_indices.append(interpolation_fine)
            self._interpolation_coarse_indices.append(interpolation_coarse)
            self._interpolation_weights.append(weights)
            interpolation_mask = torch.zeros(nf, dtype=torch.bool, device=fine_basis.device)
            interpolation_mask[interpolation_fine] = True
            interpolation_masks.append(interpolation_mask)
        mesh = torch.meshgrid(*axis_masks, indexing="ij")
        self.expressible_mask = torch.stack(mesh).all(dim=0)
        frequency = fine_basis.eigenvalues.sqrt()
        if coarse_shape != fine_basis.shape:
            axis_cutoffs = [
                math.pi * n / length if boundary == "periodic"
                else float(eigenvalues[n - 1].sqrt())
                for n, length, boundary, eigenvalues in zip(
                    coarse_shape, fine_basis.lengths, fine_basis.boundaries, fine_basis.axis_eigenvalues)
            ]
            self.retained_cutoff = frequency.new_tensor(min(axis_cutoffs))
            # A relative tolerance only protects exact boundary modes from
            # floating-point representation of pi; it does not widen a shell.
            tolerance = 16 * torch.finfo(frequency.dtype).eps
            self.expressible_mask &= frequency <= self.retained_cutoff * (1 + tolerance)
        else:
            self.retained_cutoff = frequency[self.expressible_mask].max()
        self.interpolation_mask = torch.stack(torch.meshgrid(*interpolation_masks, indexing="ij")).all(dim=0)
        self._coarsen_scale = math.sqrt(math.prod(coarse_shape) / math.prod(fine_basis.shape))

    @staticmethod
    def _transfer(coefficients, source_indices, target_indices, target_shape):
        output = coefficients
        for spatial_axis, (source, target, size) in enumerate(zip(source_indices, target_indices, target_shape), start=2):
            source = source.to(output.device)
            target = target.to(output.device)
            selected = output.index_select(spatial_axis, source)
            shape = list(output.shape)
            shape[spatial_axis] = size
            output = torch.zeros(shape, dtype=output.dtype, device=output.device).index_add(spatial_axis, target, selected)
        return output

    def coarsen_coefficients(self, fine_coefficients: torch.Tensor):
        self.fine_basis._check_shape(fine_coefficients)
        retained = fine_coefficients * self.expressible_mask.to(fine_coefficients.device)
        return self._transfer(retained, self._fine_indices, self._coarse_indices, self.coarse_shape) * self._coarsen_scale

    def decode_coefficients(self, coarse_coefficients: torch.Tensor):
        self.coarse_basis._check_shape(coarse_coefficients)
        output = coarse_coefficients
        for axis, (source, target, weights, size) in enumerate(zip(
            self._interpolation_coarse_indices, self._interpolation_fine_indices,
            self._interpolation_weights, self.fine_basis.shape), start=2
        ):
            selected = output.index_select(axis, source.to(output.device))
            weight_shape = [1] * output.ndim
            weight_shape[axis] = len(weights)
            selected = selected * weights.to(device=output.device, dtype=output.real.dtype).reshape(weight_shape)
            shape = list(output.shape)
            shape[axis] = size
            output = torch.zeros(shape, dtype=output.dtype, device=output.device).index_copy(axis, target.to(output.device), selected)
        return output / self._coarsen_scale

    def sample_decoded(self, fine_state: torch.Tensor):
        """Recover coarse nodal values from a decoded interpolant, without LP.

        This is the predictor interface after decoding, not the state encoder.
        Reapplying ``coarsen`` here would erase quantization-generated corner
        frequencies and change the primitive-state construction a second time.
        """
        output = self.fine_basis.analysis(fine_state)
        for axis, (source, target, size) in enumerate(zip(
            self._interpolation_fine_indices, self._interpolation_coarse_indices, self.coarse_shape
        ), start=2):
            selected = output.index_select(axis, source.to(output.device))
            shape = list(output.shape)
            shape[axis] = size
            # The two fine-grid Nyquist signs sum to the single coarse mode.
            output = torch.zeros(shape, dtype=output.dtype, device=output.device).index_add(axis, target.to(output.device), selected)
        return self.coarse_basis.synthesis(output * self._coarsen_scale)

    def coarsen(self, fine_state: torch.Tensor):
        return self.coarse_basis.synthesis(self.coarsen_coefficients(self.fine_basis.analysis(fine_state)))

    def decode(self, coarse_state: torch.Tensor):
        return self.fine_basis.synthesis(self.decode_coefficients(self.coarse_basis.analysis(coarse_state)))

    def project(self, fine_state: torch.Tensor):
        """Fine-grid expressible-band projection (before any quantization)."""
        coefficients = self.fine_basis.analysis(fine_state)
        return self.fine_basis.synthesis(coefficients * self.expressible_mask.to(coefficients.device))
