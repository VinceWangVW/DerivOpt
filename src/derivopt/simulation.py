"""Budgeted physical-state encoding, one-step prediction and closed-loop feedback.

Every step uses the same external codec path as predictor-free input evaluation.
Training gradients use straight-through quantization, but the forward values
and per-sample hard-cap ledgers come from actual serialized state records.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .baselines import ArchMulti
from .budget import PayloadBudget, PayloadLedger
from .calibration import CalibratedState
from .geometry import Geometry
from .latent import HyperpriorCodec
from .models import make_model
from .quantization import ExplicitRecordCodec


_METHODS = {
    "primitive": "primitive",
    "bestsingle": "best_single",
    "bestsinglederived": "best_single",
    "derivbase": "derivbase",
    "derivopt": "derivopt",
    "archmulti": "archmulti",
    "rolloutmulti": "rolloutmulti",
    "latent": "latent_hyper",
    "latenthyper": "latent_hyper",
    "latentaehyper": "latent_hyper",
    "derivoptarchmulti": "derivopt_archmulti",
}


def canonical_method(name: str) -> str:
    key = name.lower().replace("-", "").replace("_", "")
    if key not in _METHODS:
        raise ValueError(f"Unsupported simulation method {name!r}")
    return _METHODS[key]


class BudgetedSimulator(nn.Module):
    """One-step simulator with a shared decoded-primitive predictor interface.

    ``mean`` and ``std`` are fixed, training-only per-channel statistics of the
    **coarse residual primitive state**. None selects identity normalization;
    this class never fits statistics from the supplied evaluation states.

    The caller supplies the selected design. Architecture-side methods change
    the predictor while retaining the same state codec.
    ``rolloutmulti`` uses the same primitive-state simulator; its multi-step
    training objective belongs to the training loop.
    """

    def __init__(
        self,
        calibration: CalibratedState,
        design_bits: Sequence[int] | None,
        geometry: Geometry,
        *,
        backbone: str = "fno",
        method: str = "derivopt",
        model_kwargs: dict[str, Any] | None = None,
        mean: Tensor | Sequence[float] | None = None,
        std: Tensor | Sequence[float] | None = None,
        latent_kwargs: dict[str, Any] | None = None,
        budget_ratio: float = 0.25,
        prediction_mode: str = "direct",
        arch_width: int | None = None,
        arch_recurrence: str = "fine_branch_only",
    ) -> None:
        super().__init__()
        self.method = canonical_method(method)
        self.backbone_name = backbone
        self.calibration, self.geometry = calibration, geometry
        self.channels = calibration.prior.shape[-1]
        self.spatial_dim = geometry.basis.ndim
        self.fine_shape = geometry.basis.shape
        self.coarse_shape = geometry.resampler.coarse_shape
        if calibration.metadata.get("split") != "train":
            raise ValueError("Simulator requires an explicitly training-only calibration")
        for label, left, right in (
            ("fine shape", calibration.basis.shape, geometry.basis.shape),
            ("physical lengths", calibration.basis.lengths, geometry.basis.lengths),
            ("boundary types", calibration.basis.boundaries, geometry.basis.boundaries),
            ("Robin coefficients", calibration.basis.robin_coefficients, geometry.basis.robin_coefficients),
            ("coarse shape", calibration.resampler.coarse_shape, geometry.resampler.coarse_shape),
        ):
            if left != right:
                raise ValueError(f"Calibration and geometry disagree about {label}")
        if geometry.channels != self.channels:
            raise ValueError("Geometry lifting and calibrated primitive channel counts differ")
        if geometry.basis.device.type != "cpu" or calibration.basis.device.type != "cpu":
            raise ValueError("BudgetedSimulator currently supports CPU codec execution only")
        if prediction_mode not in {"direct", "residual"}:
            raise ValueError("prediction_mode must be 'direct' or 'residual'")
        if arch_recurrence != "fine_branch_only":
            raise ValueError("ArchMulti supports only fine_branch_only persistent recurrence")
        self.prediction_mode = prediction_mode
        self.budget = PayloadBudget(math.prod(self.coarse_shape), self.channels, budget_ratio)
        if (mean is None) != (std is None):
            raise ValueError("mean and std must either both be supplied or both be None")
        identity_normalization = mean is None
        mean_tensor = torch.zeros(self.channels) if mean is None else torch.as_tensor(mean, dtype=torch.float32).detach().cpu().clone()
        std_tensor = torch.ones(self.channels) if std is None else torch.as_tensor(std, dtype=torch.float32).detach().cpu().clone()
        if mean_tensor.shape != (self.channels,) or std_tensor.shape != (self.channels,):
            raise ValueError("mean and std require exactly one value per primitive channel")
        if not torch.isfinite(mean_tensor).all() or not torch.isfinite(std_tensor).all() or not (std_tensor > 0).all():
            raise ValueError("Normalization needs finite mean and finite strictly positive std")
        self.register_buffer("primitive_mean", mean_tensor)
        self.register_buffer("primitive_std", std_tensor)
        self.model_kwargs = dict(model_kwargs or {})
        if backbone.lower().replace("-", "").replace("_", "") == "convlstm":
            self.model_kwargs.setdefault("stateful", True)
        self.latent_kwargs = dict(latent_kwargs or {})
        if {"name", "spatial_dim", "in_channels", "out_channels"} & self.model_kwargs.keys():
            raise ValueError("Predictor family/dimensions/channels are controlled by the simulator")
        if {"spatial_dim", "channels", "shape"} & self.latent_kwargs.keys():
            raise ValueError("Latent dimensions/channels/shape are controlled by the simulator")
        self.arch_width = self.model_kwargs.get("width", 16) if arch_width is None else arch_width

        def factory(inputs: int, outputs: int) -> nn.Module:
            return make_model(backbone, self.spatial_dim, inputs, outputs, **self.model_kwargs)

        if self.method in {"archmulti", "derivopt_archmulti"}:
            self.predictor = ArchMulti(factory, self.spatial_dim, self.channels, self.channels, self.arch_width)
        else:
            self.predictor = factory(self.channels, self.channels)
        if self.method == "latent_hyper":
            if design_bits is not None and any(design_bits):
                raise ValueError("latent_hyper stores learned latents, not a simultaneous explicit-field design")
            self.design_bits = tuple(0 for _ in calibration.candidates)
            self.active_indices: list[int] = []
            self.explicit_codec = None
            self.latent_codec = HyperpriorCodec(
                self.spatial_dim, self.channels, shape=self.coarse_shape, **self.latent_kwargs
            )
            if self.budget.cap_bytes < self.latent_codec.minimum_record_bytes:
                raise ValueError(
                    f"Latent hard cap is {self.budget.cap_bytes} bytes, below the "
                    f"{self.latent_codec.minimum_record_bytes}-byte record minimum; choose a viable grid/budget."
                )
        else:
            if design_bits is None:
                raise ValueError("Explicit methods require a caller-selected design_bits allocation")
            self.design_bits = calibration._validate_bits(design_bits)
            self.active_indices = [index for index, bits in enumerate(self.design_bits) if bits]
            active_candidates = [calibration.candidates[index] for index in self.active_indices]
            if self.method in {"primitive", "archmulti", "rolloutmulti"}:
                primitive_indices = {index for index, candidate in enumerate(calibration.candidates) if candidate.primitive}
                if (set(self.active_indices) != primitive_indices
                        or sum(candidate.n_components for candidate in active_candidates) != self.channels):
                    raise ValueError(f"{self.method} requires exactly one primitive representation, comprising all primitive candidates and no derived fields")
            if self.method == "best_single" and (len(active_candidates) != 1 or active_candidates[0].primitive):
                raise ValueError("best_single requires exactly one non-primitive candidate in the supplied design")
            self.explicit_codec = (
                ExplicitRecordCodec(
                    [calibration.quantizers[index] for index in self.active_indices],
                    [self.design_bits[index] for index in self.active_indices],
                    [(calibration.candidates[index].n_components, *self.coarse_shape) for index in self.active_indices],
                    self.budget.cap_bytes,
                )
                if self.active_indices else None
            )
            self.latent_codec = None
        self.config = {
            "backbone": backbone, "method": self.method,
            "model_kwargs": copy.deepcopy(self.model_kwargs),
            "latent_kwargs": copy.deepcopy(self.latent_kwargs),
            "budget_ratio": float(budget_ratio), "prediction_mode": prediction_mode,
            "arch_width": self.arch_width, "arch_recurrence": arch_recurrence,
        }
        self.protocol = {
            "predictor_input": "decoded_coarse_residual_primitive",
            "predictor_input_channels": self.channels,
            "normalization": "identity" if identity_normalization else "caller_supplied_train_statistics",
            "quantization": "real_record_forward_with_straight_through_training",
            "payload_cap_bytes": self.budget.cap_bytes,
            "candidate_construction": "fine_residual_then_antialiased_coarsen",
            "prediction_mode": prediction_mode,
            "device_support": "cpu_float32",
            "explicit_decoder": (None if self.method == "latent_hyper" else
                                 "inverse_scaling_trigonometric_interpolation"
                                 if self.method in {"primitive", "archmulti", "rolloutmulti"}
                                 else "calibrated_posterior"),
            "retained_support": "boundary_adapted_radial",
        }

    @property
    def cap_bytes(self) -> int:
        return self.budget.cap_bytes

    def _validate_input(self, fine: Tensor) -> Tensor:
        expected = (self.channels, *self.fine_shape)
        if fine.ndim != self.spatial_dim + 2 or tuple(fine.shape[1:]) != expected or fine.shape[0] < 1:
            raise ValueError(f"Expected a nonempty [B,{','.join(map(str, expected))}] fine physical state")
        if fine.device.type != "cpu" or any(parameter.device.type != "cpu" for parameter in self.parameters()):
            raise ValueError("BudgetedSimulator actual state-record path currently supports CPU only, not CUDA or MPS")
        if any(parameter.dtype not in {torch.float32, torch.complex64} for parameter in self.parameters()):
            raise ValueError("BudgetedSimulator supports float32 predictors and complex64 FNO parameters")
        if not torch.is_floating_point(fine) or not torch.isfinite(fine).all():
            raise ValueError("Fine physical states must be finite real floating-point tensors")
        return fine.to(dtype=torch.float32)

    def reset_state(self) -> None:
        reset = getattr(self.predictor, "reset_state", None)
        if callable(reset):
            reset()

    def detach_state(self) -> None:
        detach = getattr(self.predictor, "detach_state", None)
        if callable(detach):
            detach()

    def _explicit_input(self, residual: Tensor, actual_codec: bool) -> tuple[Tensor, dict[str, Any]]:
        if not self.active_indices:
            # The empty design reconstructs the frozen training prior mean.
            # Its external record contains padding only and carries no input
            # observations.
            coefficients = self.calibration.prior_mean.transpose(0, 1).reshape(1, self.channels, *self.fine_shape)
            prior_state = self.geometry.basis.synthesis(coefficients).to(device=residual.device, dtype=residual.dtype)
            reconstructed = prior_state.expand(residual.shape[0], -1, *([-1] * self.spatial_dim))
            payloads = [bytes(self.cap_bytes) for _ in range(residual.shape[0])]
            ledgers = [PayloadLedger(cap_bits=self.cap_bytes * 8, padding_bits=self.cap_bytes * 8) for _ in payloads]
            for payload, ledger in zip(payloads, ledgers):
                ledger.validate(payload)
            return reconstructed, {
                "payloads": payloads, "ledgers": ledgers,
                "rate_loss": residual.new_zeros(()),
                # Distortion is a diagnostic/loss only; it is not supplied to
                # the decoder or predictor and does not change the prior state.
                "codec_reconstruction_loss": F.mse_loss(reconstructed, residual),
                "rate_estimate_bits": residual.new_zeros(residual.shape[0]),
                "empty_design": True,
            }
        fields = [
            self.geometry.resampler.coarsen(self.calibration.candidates[index].evaluate(residual, self.geometry.basis))
            for index in self.active_indices
        ]
        payloads, ledgers, sample_fields = [], [], []
        # Every training state is serialized as well: the differentiable path
        # changes gradients only, never forward values or hard-cap accounting.
        with torch.no_grad():
            for batch_index in range(residual.shape[0]):
                payload, ledger = self.explicit_codec.encode([field[batch_index] for field in fields])
                decoded = self.explicit_codec.decode(payload, dtype=residual.dtype)
                payloads.append(payload)
                ledgers.append(ledger)
                sample_fields.append(decoded)
        observations = []
        for field_index, field in enumerate(fields):
            decoded = torch.stack([sample[field_index] for sample in sample_fields])
            observations.append(decoded if actual_codec else decoded + (field - field.detach()))
        if self.method in {"primitive", "archmulti", "rolloutmulti"}:
            reconstructed = self.calibration.decode_primitive_fields(observations, self.design_bits)
        else:
            reconstructed = self.calibration.decode_fields(observations, self.design_bits)
        zero = residual.new_zeros(())
        return reconstructed, {
            "payloads": payloads, "ledgers": ledgers,
            "rate_loss": zero,
            "codec_reconstruction_loss": F.mse_loss(reconstructed, residual),
            "rate_estimate_bits": residual.new_tensor([ledger.field_bits for ledger in ledgers]),
        }

    def _latent_input(self, residual: Tensor, actual_codec: bool) -> tuple[Tensor, dict[str, Any]]:
        coarse = self.geometry.resampler.coarsen(residual)
        payloads, ledgers, outputs, rates, distortions, indices = [], [], [], [], [], []
        for batch_index in range(coarse.shape[0]):
            sample = coarse[batch_index:batch_index + 1]
            # Actual realized byte size chooses the quantizer even in training.
            payload, ledger = self.latent_codec.encode(sample, self.cap_bytes)
            index = self.latent_codec.record_info(payload)["step_index"]
            if actual_codec:
                output = self.latent_codec.decode(payload)
                with torch.no_grad():
                    _, aux = self.latent_codec(sample, step_index=index)
            else:
                output, aux = self.latent_codec(sample, step_index=index)
            payloads.append(payload)
            ledgers.append(ledger)
            outputs.append(output)
            rates.append(aux["rate_bits"].reshape(()))
            distortions.append(aux["reconstruction"])
            indices.append(index)
        decoded_coarse = torch.cat(outputs)
        rate_bits = torch.stack(rates)
        return self.geometry.resampler.decode(decoded_coarse), {
            "payloads": payloads, "ledgers": ledgers,
            "rate_loss": rate_bits.mean() / coarse[0].numel(),
            "codec_reconstruction_loss": torch.stack(distortions).mean(),
            "rate_estimate_bits": rate_bits,
            "quantization_step_indices": indices,
        }

    def input_reconstruction(self, fine: Tensor, actual_codec: bool = True) -> tuple[Tensor, dict[str, Any]]:
        """One predictor-free fine→state-record→fine reconstruction, including lift."""
        fine = self._validate_input(fine)
        residual = self.geometry.to_residual(fine)
        if self.latent_codec is None:
            reconstructed, info = self._explicit_input(residual, actual_codec)
        else:
            reconstructed, info = self._latent_input(residual, actual_codec)
        physical = self.geometry.to_physical(reconstructed)
        if not torch.isfinite(physical).all():
            raise FloatingPointError("Non-finite decoded physical input state")
        return physical, {
            **info, "method": self.method, "cap_bytes": self.cap_bytes,
            "actual_codec": actual_codec,
            "codec_forward": "serialized_record" if actual_codec else "serialized_record_with_training_gradient",
        }

    def step(self, fine: Tensor, actual_codec: bool = False) -> tuple[Tensor, dict[str, Any]]:
        """Encode this supplied state, predict one step and return fine physical output.

        For a closed loop, pass the returned state into the next call; this
        method never reads a ground-truth target or a future trajectory frame.
        """
        decoded_fine, info = self.input_reconstruction(fine, actual_codec=actual_codec)
        decoded_residual = self.geometry.to_residual(decoded_fine)
        decoded_coarse = self.geometry.resampler.sample_decoded(decoded_residual)
        broadcast = (1, self.channels, *([1] * self.spatial_dim))
        mean, std = self.primitive_mean.reshape(broadcast), self.primitive_std.reshape(broadcast)
        prediction = self.predictor((decoded_coarse - mean) / std)
        if prediction.shape != decoded_coarse.shape:
            raise ValueError("Predictor must preserve the common decoded primitive channel/grid interface")
        if self.prediction_mode == "residual":
            next_coarse = decoded_coarse + prediction * std
        else:
            next_coarse = prediction * std + mean
        next_fine = self.geometry.to_physical(self.geometry.resampler.decode(next_coarse))
        if not torch.isfinite(next_fine).all():
            raise FloatingPointError("Non-finite one-step simulator prediction")
        return next_fine, {**info, "decoded_input_fine": decoded_fine, "decoded_input_coarse": decoded_coarse}

    def forward(self, fine: Tensor, actual_codec: bool = False) -> Tensor:
        return self.step(fine, actual_codec=actual_codec)[0]

    def checkpoint_state(self) -> dict[str, Any]:
        """Snapshot weights, fixed normalization, codec, calibration and geometry."""
        return {
            "format_version": 1,
            "config": copy.deepcopy(self.config),
            "design_bits": self.design_bits,
            "normalization_mean": self.primitive_mean.detach().cpu().clone(),
            "normalization_std": self.primitive_std.detach().cpu().clone(),
            "protocol": copy.deepcopy(self.protocol),
            "module_state": copy.deepcopy(self.state_dict()),
            "calibration": self.calibration.state_dict(),
            "geometry": self.geometry.state_dict(),
        }

    @classmethod
    def from_checkpoint_state(cls, state: dict[str, Any]) -> "BudgetedSimulator":
        if state.get("format_version") != 1:
            raise ValueError("Unsupported simulator checkpoint version")
        config = copy.deepcopy(state["config"])
        if config["backbone"].lower().replace("-", "").replace("_", "") == "convlstm":
            if (canonical_method(config["method"]) in {"archmulti", "derivopt_archmulti"}
                    and config.get("model_kwargs", {}).get("stateful", False)
                    and "arch_recurrence" not in config):
                raise ValueError(
                    "Legacy stateful ConvLSTM ArchMulti checkpoint used two persistent branches; "
                    "it cannot be loaded as the fine-branch-only model. Train a new checkpoint."
                )
            # Older checkpoints omitted this setting and used the backbone's
            # stateless default. Preserve that recorded model's behavior.
            config.setdefault("model_kwargs", {}).setdefault("stateful", False)
        geometry = Geometry.from_state_dict(state["geometry"])
        calibration = CalibratedState.from_state_dict(state["calibration"])
        simulator = cls(
            calibration, state["design_bits"], geometry,
            mean=state["normalization_mean"], std=state["normalization_std"], **config,
        )
        simulator.load_state_dict(state["module_state"])
        simulator.protocol = copy.deepcopy(state["protocol"])
        simulator.reset_state()
        return simulator


__all__ = ["BudgetedSimulator", "canonical_method"]
