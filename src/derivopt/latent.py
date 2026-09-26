"""Learned scale-hyperprior codec with primary and side-information bitstreams.

Each state record includes framing and both streams in its byte ledger.
Decoding uses the record and fixed codec parameters without the source state.
"""

from __future__ import annotations

import math
import struct
import zlib
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .budget import PayloadLedger
from .entropy import decode_symbols, encode_symbols


_TOTAL = 1 << 16
_MAGIC = b"H1"


def _varint(value: int) -> bytes:
    if not 0 <= value < 1 << 35:
        raise ValueError("Stream length exceeds supported record format")
    output = bytearray()
    while value >= 128:
        output.append((value & 127) | 128)
        value >>= 7
    output.append(value)
    return bytes(output)


def _read_varint(payload: bytes, offset: int) -> tuple[int, int]:
    value, start = 0, offset
    for shift in range(0, 35, 7):
        if offset >= len(payload):
            raise ValueError("Truncated latent stream-length header")
        byte = payload[offset]
        offset += 1
        value |= (byte & 127) << shift
        if not byte & 128:
            if payload[start:offset] != _varint(value):
                raise ValueError("Noncanonical latent stream-length header")
            return value, offset
    raise ValueError("Invalid latent stream-length header")


def _gaussian_frequencies(scale: float, limit: int) -> list[int]:
    """Clipped Gaussian alphabet, positive integer masses summing to 2**16."""
    edges = [0.0]
    edges.extend(0.5 * (1 + math.erf((value + 0.5) / (scale * math.sqrt(2)))) for value in range(-limit, limit))
    edges.append(1.0)
    probabilities = [max(0.0, right - left) for left, right in zip(edges, edges[1:])]
    available = _TOTAL - len(probabilities)
    raw = [probability * available for probability in probabilities]
    frequencies = [1 + math.floor(value) for value in raw]
    remainder = _TOTAL - sum(frequencies)
    # Stable symbol order resolves equal fractional remainders deterministically.
    order = sorted(range(len(raw)), key=lambda index: (-(raw[index] - math.floor(raw[index])), index))
    for index in order[:remainder]:
        frequencies[index] += 1
    return frequencies


class HyperpriorCodec(nn.Module):
    """Trainable g_a/g_s and h_a/h_s with finite-alphabet Gaussian coding.

    Input shape and architecture are fixed in the decoder checkpoint. One
    record transmits the quantization-step index, stream lengths, checksum,
    hyperlatent stream, primary stream and all remaining-capacity padding.
    Encoding chooses the first feasible step from the declared ordered table;
    no validation or test-set fitting occurs inside encode().
    """

    minimum_record_bytes = 11  # 9-byte small header plus two >=1-byte streams

    def __init__(
        self,
        spatial_dim: int,
        channels: int,
        latent_channels: int = 4,
        width: int = 16,
        shape: Sequence[int] | None = None,
        *,
        hyper_channels: int | None = None,
        quant_steps: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0),
        symbol_limit: int = 31,
    ) -> None:
        super().__init__()
        if spatial_dim not in (1, 2) or min(channels, latent_channels, width) < 1:
            raise ValueError("HyperpriorCodec requires 1D/2D and positive channel/width counts")
        if shape is None or len(shape) != spatial_dim or any(not isinstance(size, int) or size < 1 for size in shape):
            raise ValueError("A fixed positive spatial shape must be supplied")
        if not isinstance(symbol_limit, int) or not 1 <= symbol_limit <= 127:
            raise ValueError("symbol_limit must be an integer in [1,127]")
        steps = tuple(float(step) for step in quant_steps)
        if not steps or len(steps) > 256 or any(not math.isfinite(step) or step <= 0 for step in steps):
            raise ValueError("quant_steps requires 1..256 finite positive entries")
        if any(left >= right for left, right in zip(steps, steps[1:])):
            raise ValueError("quant_steps must be strictly increasing")
        hyper_channels = max(1, latent_channels // 2) if hyper_channels is None else hyper_channels
        if hyper_channels < 1:
            raise ValueError("hyper_channels must be positive")
        self.spatial_dim, self.channels = spatial_dim, channels
        self.latent_channels, self.hyper_channels = latent_channels, hyper_channels
        self.width, self.shape, self.symbol_limit = width, tuple(shape), symbol_limit
        self.latent_shape = tuple((size + 3) // 4 for size in self.shape)
        self.hyper_shape = tuple((size + 1) // 2 for size in self.latent_shape)
        self.default_step_index = min(range(len(steps)), key=lambda index: abs(math.log(steps[index])))
        self.config: dict[str, Any] = {
            "spatial_dim": spatial_dim, "channels": channels, "latent_channels": latent_channels,
            "hyper_channels": hyper_channels, "width": width, "shape": self.shape,
            "quant_steps": steps, "symbol_limit": symbol_limit,
        }
        self.architecture_metadata = {
            **self.config,
            "configuration": "explicit_scale_hyperprior_settings",
            "reference": "https://arxiv.org/abs/1802.01436",
            "entropy_coding": "32bit_arithmetic_16bit_integer_gaussian_cdfs",
            "training_quantization": "clipped_round_forward_straight_through_backward",
        }
        conv = nn.Conv1d if spatial_dim == 1 else nn.Conv2d
        self.encoder = nn.Sequential(
            conv(channels, width, 3, stride=2, padding=1), nn.GELU(),
            conv(width, latent_channels, 3, stride=2, padding=1),
        )
        self.decoder = nn.Sequential(conv(latent_channels, width, 3, padding=1), nn.GELU(), conv(width, channels, 3, padding=1))
        self.hyperencoder = nn.Sequential(
            conv(latent_channels, width, 3, padding=1), nn.GELU(),
            conv(width, hyper_channels, 3, stride=2, padding=1),
        )
        self.hyperdecoder = nn.Sequential(conv(hyper_channels, width, 3, padding=1), nn.GELU(), conv(width, latent_channels, 3, padding=1))
        self.hyper_log_scale = nn.Parameter(torch.zeros(hyper_channels))
        step_tensor = torch.tensor(steps, dtype=torch.float32)
        if not torch.isfinite(step_tensor).all() or not (step_tensor > 0).all():
            raise ValueError("quant_steps must be representable as positive float32 values")
        self.register_buffer("quant_steps", step_tensor)
        scale_values = torch.logspace(math.log10(0.05), math.log10(128.0), 128, dtype=torch.float64)
        frequencies = torch.tensor([_gaussian_frequencies(float(scale), symbol_limit) for scale in scale_values], dtype=torch.int64)
        self.register_buffer("scale_table", scale_values.float())
        self.register_buffer("frequency_table", frequencies)
        self.register_buffer("cdf_table", F.pad(frequencies.cumsum(dim=1), (1, 0)))

    def get_extra_state(self) -> dict[str, Any]:
        return {"format": "H1", "config": self.config}

    def set_extra_state(self, state: dict[str, Any]) -> None:
        if state != self.get_extra_state():
            raise ValueError("Codec checkpoint configuration differs from this decoder specification")

    def _validate(self, x: Tensor, *, single: bool = False) -> None:
        if x.ndim != self.spatial_dim + 2 or x.shape[1] != self.channels or tuple(x.shape[2:]) != self.shape:
            raise ValueError(f"Expected [B,{self.channels},{','.join(map(str, self.shape))}], got {tuple(x.shape)}")
        if x.shape[0] < 1 or single and x.shape[0] != 1:
            raise ValueError("encode requires exactly one nonempty state at a time" if single else "Empty input batch")
        if not torch.is_floating_point(x) or not torch.isfinite(x).all():
            raise ValueError("Latent codec requires finite real floating-point states")

    def _step(self, index: int | None) -> tuple[int, Tensor]:
        index = self.default_step_index if index is None else index
        if not isinstance(index, int) or not 0 <= index < len(self.quant_steps):
            raise ValueError("Invalid quantization-step index")
        return index, self.quant_steps[index]

    def _resize(self, tensor: Tensor, shape: tuple[int, ...]) -> Tensor:
        return F.interpolate(tensor, size=shape, mode="linear" if self.spatial_dim == 1 else "bilinear", align_corners=False)

    def _quantize(self, value: Tensor, step: Tensor, straight_through: bool) -> Tensor:
        clipped = (value / step).clamp(-self.symbol_limit, self.symbol_limit)
        rounded = torch.round(clipped)
        return clipped + (rounded - clipped).detach() if straight_through else rounded

    def _scales(self, raw: Tensor) -> tuple[Tensor, Tensor]:
        raw = raw.clamp(float(self.scale_table[0]), float(self.scale_table[-1]))
        log_position = (raw.log() - self.scale_table[0].log()) / (self.scale_table[-1].log() - self.scale_table[0].log())
        indices = torch.round(log_position * (self.scale_table.numel() - 1)).long().clamp(0, self.scale_table.numel() - 1)
        table_scale = self.scale_table[indices]
        return raw + (table_scale - raw).detach(), indices

    def _conditional_scales(self, hyperlatent: Tensor, step: Tensor) -> tuple[Tensor, Tensor]:
        decoded = self.hyperdecoder(self._resize(hyperlatent, self.latent_shape))
        return self._scales((F.softplus(decoded) + 0.05) / step)

    def _hyper_scales(self, step: Tensor, batch_size: int) -> tuple[Tensor, Tensor]:
        scale = (F.softplus(self.hyper_log_scale) + 0.05) / step
        scale = scale.reshape(1, self.hyper_channels, *([1] * self.spatial_dim)).expand(batch_size, self.hyper_channels, *self.hyper_shape)
        return self._scales(scale)

    def _rate(self, symbols: Tensor, scale: Tensor, indices: Tensor) -> Tensor:
        upper = 0.5 * (1 + torch.erf((symbols + 0.5) / (scale * math.sqrt(2))))
        lower = 0.5 * (1 + torch.erf((symbols - 0.5) / (scale * math.sqrt(2))))
        upper = torch.where(symbols >= self.symbol_limit, torch.ones_like(upper), upper)
        lower = torch.where(symbols <= -self.symbol_limit, torch.zeros_like(lower), lower)
        continuous_mass = (upper - lower).clamp_min(1e-9)
        symbol_indices = symbols.detach().round().long() + self.symbol_limit
        coding_mass = self.frequency_table[indices, symbol_indices].to(scale.dtype) / _TOTAL
        # Forward uses the very CDF masses used by arithmetic coding. Backward
        # uses the differentiable Gaussian-CDF relaxation instead of integer steps.
        probability = continuous_mass + (coding_mass - continuous_mass).detach()
        return -torch.log2(probability).flatten(1).sum(1)

    def _analysis(self, x: Tensor, index: int | None, straight_through: bool) -> tuple[Tensor, dict[str, Any]]:
        index, step = self._step(index)
        latent = self.encoder(x)
        hyperlatent = self.hyperencoder(latent.abs())
        hyper_symbols = self._quantize(hyperlatent, step, straight_through)
        hyper_reconstruction = hyper_symbols * step
        scales, scale_indices = self._conditional_scales(hyper_reconstruction, step)
        latent_symbols = self._quantize(latent, step, straight_through)
        latent_reconstruction = latent_symbols * step
        xhat = self.decoder(self._resize(latent_reconstruction, self.shape))
        hyper_scales, hyper_indices = self._hyper_scales(step, x.shape[0])
        primary_rate = self._rate(latent_symbols, scales, scale_indices)
        hyper_rate = self._rate(hyper_symbols, hyper_scales, hyper_indices)
        return xhat, {
            "rate": (primary_rate + hyper_rate).mean(),
            "rate_bits": primary_rate + hyper_rate,
            "primary_rate_bits": primary_rate,
            "hyper_rate_bits": hyper_rate,
            "reconstruction": F.mse_loss(xhat, x),
            "latent": latent_reconstruction,
            "hyperlatent": hyper_reconstruction,
            "latent_symbols": latent_symbols,
            "hyperlatent_symbols": hyper_symbols,
            "primary_scale_indices": scale_indices,
            "hyper_scale_indices": hyper_indices,
            "step_index": index,
            "quant_step": float(step.detach()),
        }

    def forward(self, x: Tensor, step_index: int | None = None) -> tuple[Tensor, dict[str, Any]]:
        self._validate(x)
        return self._analysis(x, step_index, self.training)

    def _coding_ready(self) -> None:
        if any(parameter.device.type != "cpu" or parameter.dtype != torch.float32 for parameter in self.parameters()):
            raise ValueError("Actual entropy coding requires the checkpoint on CPU float32; forward training may use another device")

    def _cdf_rows(self, indices: Tensor) -> list[tuple[int, ...]]:
        table = [tuple(row) for row in self.cdf_table.detach().cpu().tolist()]
        return [table[index] for index in indices.detach().cpu().reshape(-1).tolist()]

    def _encode_stream(self, symbols: Tensor, indices: Tensor) -> bytes:
        values = (symbols.detach().round().long() + self.symbol_limit).reshape(-1).cpu().tolist()
        return encode_symbols(values, self._cdf_rows(indices))

    @staticmethod
    def _header(index: int, hyper_stream: bytes, primary_stream: bytes) -> bytes:
        # zlib is used ONLY for CRC32, never for entropy compression.
        prefix = _MAGIC + bytes([index]) + _varint(len(hyper_stream)) + _varint(len(primary_stream))
        checksum = zlib.crc32(prefix + hyper_stream + primary_stream)
        return prefix + struct.pack(">I", checksum)

    @torch.no_grad()
    def encode(self, x: Tensor, cap_bytes: int) -> tuple[bytes, PayloadLedger]:
        self._coding_ready()
        self._validate(x, single=True)
        if not isinstance(cap_bytes, int) or isinstance(cap_bytes, bool) or cap_bytes < self.minimum_record_bytes:
            raise ValueError(f"Hyperprior record requires at least {self.minimum_record_bytes} bytes; the requested state may need more")
        x = x.detach().cpu().float()
        minimum_seen = None
        for index in range(len(self.quant_steps)):
            _, aux = self._analysis(x, index, False)
            hyper_stream = self._encode_stream(aux["hyperlatent_symbols"], aux["hyper_scale_indices"])
            primary_stream = self._encode_stream(aux["latent_symbols"], aux["primary_scale_indices"])
            header = self._header(index, hyper_stream, primary_stream)
            used = len(header) + len(hyper_stream) + len(primary_stream)
            minimum_seen = used if minimum_seen is None else min(minimum_seen, used)
            if used <= cap_bytes:
                padding = cap_bytes - used
                payload = header + hyper_stream + primary_stream + bytes(padding)
                ledger = PayloadLedger(
                    cap_bits=cap_bytes * 8, primary_bits=len(primary_stream) * 8,
                    hyper_bits=len(hyper_stream) * 8, header_bits=len(header) * 8,
                    padding_bits=padding * 8,
                )
                ledger.validate(payload)
                return payload, ledger
        raise ValueError(
            f"No configured quantization step fits cap_bytes={cap_bytes}; "
            f"smallest realized record is {minimum_seen} bytes. Increase the cap or explicitly revise the fixed codec configuration."
        )

    @staticmethod
    def record_info(payload: bytes) -> dict[str, int]:
        """Inspect record framing and charged byte counts without latent decoding."""
        if not isinstance(payload, bytes) or len(payload) < HyperpriorCodec.minimum_record_bytes or payload[:2] != _MAGIC:
            raise ValueError("Invalid or truncated H1 hyperprior record")
        index = payload[2]
        hyper_length, offset = _read_varint(payload, 3)
        primary_length, offset = _read_varint(payload, offset)
        header_length = offset + 4
        end = header_length + hyper_length + primary_length
        if min(hyper_length, primary_length) < 1 or end > len(payload):
            raise ValueError("Truncated or empty hyperprior arithmetic stream")
        expected_crc = struct.unpack(">I", payload[offset:header_length])[0]
        if zlib.crc32(payload[:offset] + payload[header_length:end]) != expected_crc:
            raise ValueError("Hyperprior record checksum mismatch")
        if any(payload[end:]):
            raise ValueError("Hyperprior capacity padding must be zero")
        return {
            "step_index": index, "header_bytes": header_length,
            "hyper_bytes": hyper_length, "primary_bytes": primary_length,
            "padding_bytes": len(payload) - end,
        }

    @torch.no_grad()
    def decode(self, payload: bytes) -> Tensor:
        self._coding_ready()
        info = self.record_info(payload)
        _, step = self._step(info["step_index"])
        start = info["header_bytes"]
        hyper_stream = payload[start:start + info["hyper_bytes"]]
        start += info["hyper_bytes"]
        primary_stream = payload[start:start + info["primary_bytes"]]
        _, hyper_indices = self._hyper_scales(step, 1)
        hyper_values = decode_symbols(hyper_stream, self._cdf_rows(hyper_indices))
        hyper_symbols = torch.tensor(hyper_values, dtype=torch.float32).reshape(1, self.hyper_channels, *self.hyper_shape) - self.symbol_limit
        hyperlatent = hyper_symbols * step
        _, primary_indices = self._conditional_scales(hyperlatent, step)
        primary_values = decode_symbols(primary_stream, self._cdf_rows(primary_indices))
        primary_symbols = torch.tensor(primary_values, dtype=torch.float32).reshape(1, self.latent_channels, *self.latent_shape) - self.symbol_limit
        return self.decoder(self._resize(primary_symbols * step, self.shape))


__all__ = ["HyperpriorCodec"]
