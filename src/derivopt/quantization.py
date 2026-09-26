"""Checkpoint-fixed scalar quantization and real bit-packed external records.

No per-record scales, identities or allocations are hidden: these belong to a
fixed decoder specification saved with the checkpoint. Any dynamic metadata in
other codecs must be serialized and charged separately.
"""
from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import torch

from .budget import PayloadLedger
from .widecodes import WideCodes, quantize_unit_values


def _check_bits(bits: int) -> None:
    if not isinstance(bits, int) or isinstance(bits, bool) or bits < 1:
        raise ValueError("an active scalar channel requires a positive integer bit width")


def pack_unsigned(values: Sequence[torch.Tensor | WideCodes], bits: Sequence[int], cap_bytes: int | None = None) -> bytes:
    if len(values) != len(bits):
        raise ValueError("values and bit widths must have equal lengths")
    for width in bits:
        _check_bits(width)
    if cap_bytes is not None and (not isinstance(cap_bytes, int) or isinstance(cap_bytes, bool) or cap_bytes < 0):
        raise ValueError("cap_bytes must be a nonnegative integer")
    if any(width > 32 or isinstance(value, WideCodes) for value, width in zip(values, bits)):
        # A bounded bit buffer packs arbitrary-width integers without ever
        # casting a code to uint64, int64 or float. Adjacent fields are contiguous.
        payload = bytearray()
        buffer, occupied = 0, 0
        for value, width in zip(values, bits):
            if isinstance(value, WideCodes):
                codes = value.values
            else:
                array = value.detach().cpu().reshape(-1)
                if not torch.isfinite(array).all() or (array.is_floating_point() and not torch.equal(array, array.floor())):
                    raise ValueError("only finite unsigned integer codes can be packed")
                codes = tuple(int(code) for code in array.tolist())
            maximum = (1 << width) - 1
            for code in codes:
                if code < 0 or code > maximum:
                    raise ValueError("code outside configured quantization alphabet")
                buffer |= code << occupied
                occupied += width
                while occupied >= 8:
                    payload.append(buffer & 255)
                    buffer >>= 8
                    occupied -= 8
        if occupied:
            payload.append(buffer & 255)
        if cap_bytes is not None:
            if len(payload) > cap_bytes:
                raise ValueError(f"payload {len(payload)} bytes exceeds cap {cap_bytes}")
            payload.extend(bytes(cap_bytes - len(payload)))
        return bytes(payload)
    chunks = []
    for value, width in zip(values, bits):
        _check_bits(width)
        array = value.detach().cpu().reshape(-1).numpy()
        if not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all():
            raise ValueError("only finite unsigned integer codes can be packed")
        if (array < 0).any() or (array > (1 << width)-1).any():
            raise ValueError("code outside configured quantization alphabet")
        array = array.astype(np.uint64)
        # LSB-first stream, identical for packing and unpacking.
        chunks.append(((array[:, None] >> np.arange(width, dtype=np.uint64)) & 1).astype(np.uint8).reshape(-1))
    stream = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.uint8)
    raw = np.packbits(stream, bitorder="little").tobytes()
    if cap_bytes is not None:
        if cap_bytes < 0 or len(raw) > cap_bytes:
            raise ValueError(f"payload {len(raw)} bytes exceeds cap {cap_bytes}")
        raw += bytes(cap_bytes-len(raw))
    return raw


def unpack_unsigned(payload: bytes, shapes: Sequence[tuple[int, ...]], bits: Sequence[int]) -> list[torch.Tensor | WideCodes]:
    if len(shapes) != len(bits):
        raise ValueError("shapes and bit widths must have equal lengths")
    if any(any(not isinstance(n, int) or n < 1 for n in shape) for shape in shapes):
        raise ValueError("empty channel shape")
    sizes = [math.prod(shape) for shape in shapes]
    for width in bits:
        _check_bits(width)
    used = sum(size*width for size, width in zip(sizes, bits))
    if used > len(payload)*8:
        raise ValueError("truncated external-state record")
    if any(width > 32 for width in bits):
        offset, buffer, occupied = 0, 0, 0
        decoded = []
        for shape, size, width in zip(shapes, sizes, bits):
            maximum, values = (1 << width) - 1, []
            for _ in range(size):
                while occupied < width:
                    buffer |= payload[offset] << occupied
                    occupied += 8
                    offset += 1
                values.append(buffer & maximum)
                buffer >>= width
                occupied -= width
            decoded.append(WideCodes(tuple(values), tuple(shape)) if width > 32 else torch.tensor(values, dtype=torch.int64).reshape(shape))
        if buffer or any(payload[offset:]):
            raise ValueError("nonzero record padding: wrong specification or corrupted payload")
        return decoded
    stream = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="little")
    if np.any(stream[used:]):
        raise ValueError("nonzero record padding: wrong specification or corrupted payload")
    offset, decoded = 0, []
    for shape, size, width in zip(shapes, sizes, bits):
        chunk = stream[offset:offset+size*width].reshape(size, width).astype(np.uint64)
        codes = np.sum(chunk * (np.uint64(1) << np.arange(width, dtype=np.uint64)), axis=1)
        decoded.append(torch.from_numpy(codes.astype(np.int64)).reshape(shape))
        offset += size*width
    return decoded


@dataclass
class ScalarQuantizer:
    """Affine train-only standardization, fixed clipping, and uniform quantization.

    mean/std/radius each have one value per channel. Clipping radii are fitted
    exclusively from training states and are fixed at inference time.
    """
    mean: torch.Tensor
    std: torch.Tensor
    radius: torch.Tensor

    @classmethod
    def fit(cls, training: torch.Tensor, *, split: str, epsilon: float = 1e-8):
        if split != "train":
            raise ValueError("quantizer calibration must use the training split")
        if training.ndim < 3 or not torch.isfinite(training).all():
            raise ValueError("expected finite [N,C,*space] calibration states")
        axes = (0, *range(2, training.ndim))
        mean = training.mean(axes)
        shape = (1, -1) + (1,)*(training.ndim-2)
        std = training.std(axes, correction=0).clamp_min(epsilon)
        standardized = (training-mean.reshape(shape))/std.reshape(shape)
        radius = standardized.abs().amax(axes).clamp_min(1.0)
        return cls(mean.detach(), std.detach(), radius.detach())

    def _parameters(self, tensor: torch.Tensor):
        if tensor.ndim < 3 or tensor.shape[1] != self.mean.numel():
            raise ValueError("quantizer channel/shape mismatch")
        shape = (1, -1) + (1,)*(tensor.ndim-2)
        return tuple(value.to(tensor.device, torch.float64).reshape(shape) for value in (self.mean, self.std, self.radius))

    def codes(self, tensor: torch.Tensor, bits: int) -> torch.Tensor | WideCodes:
        _check_bits(bits)
        if not torch.isfinite(tensor).all():
            raise ValueError("cannot encode non-finite state")
        mean, std, radius = self._parameters(tensor)
        unit = (((tensor.double()-mean)/std).clamp(-radius, radius)/radius+1)/2
        if bits > 32:
            codes = quantize_unit_values(unit.detach().cpu().reshape(-1).tolist(), bits)
            return WideCodes(codes, tuple(tensor.shape), tensor.device)
        return torch.round(unit*((1 << bits)-1)).to(torch.int64)

    def from_codes(self, codes: torch.Tensor | WideCodes, bits: int, *, dtype=torch.float32) -> torch.Tensor:
        _check_bits(bits)
        if isinstance(codes, WideCodes) or bits > 32:
            if isinstance(codes, WideCodes):
                values = codes.values
            else:
                raw = codes.detach().cpu().reshape(-1)
                if not torch.isfinite(raw).all() or (raw.is_floating_point() and not torch.equal(raw, raw.floor())):
                    raise ValueError("invalid scalar quantizer code")
                values = tuple(int(value) for value in raw.tolist())
            maximum = (1 << bits) - 1
            if any(value < 0 or value > maximum for value in values):
                raise ValueError("invalid scalar quantizer code")
            mean, std, radius = self._parameters(codes)
            # Python's integer true division obtains the floating ratio without
            # first converting huge numerator/denominator integers to floats.
            unit = torch.tensor([value / maximum for value in values], dtype=torch.float64,
                                device=codes.device).reshape(codes.shape)
            return ((unit*2-1)*radius*std+mean).to(dtype)
        if not torch.isfinite(codes).all() or torch.any(codes < 0) or torch.any(codes > (1 << bits)-1):
            raise ValueError("invalid scalar quantizer code")
        mean, std, radius = self._parameters(codes)
        return (((codes.double()/((1 << bits)-1))*2-1)*radius*std+mean).to(dtype)

    def __call__(self, tensor: torch.Tensor, bits: int, *, straight_through: bool = False):
        decoded = self.from_codes(self.codes(tensor, bits), bits, dtype=tensor.dtype)
        return tensor+(decoded-tensor).detach() if straight_through else decoded

    def state_dict(self):
        return {key: value.detach().cpu() for key, value in vars(self).items()}


class ExplicitRecordCodec:
    """A fixed-layout record codec; each supplied tensor is one selected field."""
    def __init__(self, quantizers: Sequence[ScalarQuantizer], bits: Sequence[int], shapes: Sequence[tuple[int, ...]], cap_bytes: int):
        self.quantizers, self.bits, self.shapes = list(quantizers), list(bits), list(shapes)
        if not (len(self.quantizers) == len(self.bits) == len(self.shapes)) or not self.bits:
            raise ValueError("a nonempty fixed decoder specification is required")
        self.cap_bytes = cap_bytes
        for width in self.bits:
            _check_bits(width)
        self.field_bits = sum(math.prod(shape)*width for shape, width in zip(self.shapes, self.bits))
        if cap_bytes < 0 or self.field_bits > cap_bytes*8:
            raise ValueError("fixed field layout exceeds payload budget")

    def encode(self, fields: Sequence[torch.Tensor]) -> tuple[bytes, PayloadLedger]:
        if len(fields) != len(self.shapes):
            raise ValueError("selected field count mismatch")
        codes = []
        for tensor, quantizer, width, shape in zip(fields, self.quantizers, self.bits, self.shapes):
            if tuple(tensor.shape) != shape:
                raise ValueError("record field shape mismatch; encode one state at a time")
            codes.append(quantizer.codes(tensor.unsqueeze(0), width).squeeze(0))
        payload = pack_unsigned(codes, self.bits, self.cap_bytes)
        ledger = PayloadLedger(self.cap_bytes*8, field_bits=self.field_bits, padding_bits=len(payload)*8-self.field_bits)
        ledger.validate(payload)
        return payload, ledger

    def decode(self, payload: bytes, *, device="cpu", dtype=torch.float32):
        if len(payload) != self.cap_bytes:
            raise ValueError("record length differs from the fixed external interface")
        codes = unpack_unsigned(payload, self.shapes, self.bits)
        return [q.from_codes(c.unsqueeze(0).to(device), b, dtype=dtype).squeeze(0)
                for q, c, b in zip(self.quantizers, codes, self.bits)]
