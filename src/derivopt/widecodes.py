"""Shape-aware arbitrary-precision integer codes for wide scalar quantization."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch


@dataclass(frozen=True)
class WideCodes:
    """CPU integer storage with explicit shape and intended decoded Tensor device.

    Python integers preserve every code bit, including widths larger than uint64.
    Shape/device operations do not truncate or float-cast values. Codes themselves
    are discrete and have no autograd graph; the quantizer supplies optional STE.
    """
    values: tuple[int, ...]
    shape: tuple[int, ...]
    device: torch.device = torch.device("cpu")

    def __post_init__(self):
        shape = tuple(self.shape)
        values = tuple(self.values)
        if any(not isinstance(n, int) or n < 0 for n in shape) or math.prod(shape) != len(values):
            raise ValueError("WideCodes shape must match its integer value count")
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values):
            raise ValueError("WideCodes values must be unsigned Python integers")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "device", torch.device(self.device))

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def numel(self) -> int:
        return len(self.values)

    def unsqueeze(self, dim: int) -> "WideCodes":
        if not -self.ndim - 1 <= dim <= self.ndim:
            raise IndexError("WideCodes unsqueeze dimension out of range")
        dim = dim % (self.ndim + 1)
        return WideCodes(self.values, self.shape[:dim] + (1,) + self.shape[dim:], self.device)

    def squeeze(self, dim: int | None = None) -> "WideCodes":
        if dim is None:
            shape = tuple(size for size in self.shape if size != 1)
        else:
            if not -self.ndim <= dim < self.ndim:
                raise IndexError("WideCodes squeeze dimension out of range")
            dim %= self.ndim
            shape = self.shape[:dim] + self.shape[dim + 1:] if self.shape[dim] == 1 else self.shape
        return WideCodes(self.values, shape, self.device)

    def to(self, device: str | torch.device) -> "WideCodes":
        return WideCodes(self.values, self.shape, torch.device(device))


def round_ratio_even(numerator: int, denominator: int) -> int:
    """Exact nearest-integer rounding of a nonnegative ratio, ties to even."""
    if numerator < 0 or denominator <= 0:
        raise ValueError("nearest-even ratio requires numerator>=0 and denominator>0")
    quotient, remainder = divmod(numerator, denominator)
    twice = remainder * 2
    return quotient + int(twice > denominator or (twice == denominator and quotient % 2 == 1))


def quantize_unit_values(values: Sequence[float], bits: int) -> tuple[int, ...]:
    """Quantize represented binary floating inputs exactly to 2**bits levels."""
    if not isinstance(bits, int) or isinstance(bits, bool) or bits < 1:
        raise ValueError("bit width must be a positive integer")
    maximum = (1 << bits) - 1
    result = []
    for value in values:
        value = float(value)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("unit quantizer inputs must be finite and in [0,1]")
        numerator, denominator = value.as_integer_ratio()
        result.append(round_ratio_even(numerator * maximum, denominator))
    return tuple(result)
