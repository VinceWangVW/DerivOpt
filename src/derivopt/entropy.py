"""Small integer-CDF arithmetic coder used by the learned latent codec.

Streams include termination and byte-alignment bits. The checkpoint-fixed codec
shape determines the symbol count, and integer CDFs determine the coding intervals.
"""

from __future__ import annotations

from bisect import bisect_right
from typing import Sequence


_PRECISION = 32
_FULL = 1 << _PRECISION
_HALF = _FULL // 2
_QUARTER = _HALF // 2
_THREE_QUARTERS = 3 * _QUARTER


def _validate_cdf(cdf: Sequence[int]) -> None:
    if len(cdf) < 2 or cdf[0] != 0 or cdf[-1] >= _QUARTER:
        raise ValueError("Arithmetic CDF requires 0 < total < 2**30")
    if any(not isinstance(value, int) for value in cdf):
        raise ValueError("Arithmetic CDF entries must be integers")
    if any(left >= right for left, right in zip(cdf, cdf[1:])):
        raise ValueError("Arithmetic CDF must be strictly increasing")


def encode_symbols(symbols: Sequence[int], cdfs: Sequence[Sequence[int]]) -> bytes:
    """Encode known-count nonnegative symbol indices using per-symbol CDFs."""
    if len(symbols) != len(cdfs) or not symbols:
        raise ValueError("A nonempty symbol sequence and one CDF per symbol are required")
    checked = set()
    low, high, pending = 0, _FULL - 1, 0
    bits: list[int] = []

    def emit(bit: int) -> None:
        nonlocal pending
        bits.append(bit)
        bits.extend([1 - bit] * pending)
        pending = 0

    for symbol, cdf in zip(symbols, cdfs):
        identity = id(cdf)
        if identity not in checked:
            _validate_cdf(cdf)
            checked.add(identity)
        if not isinstance(symbol, int) or not 0 <= symbol < len(cdf) - 1:
            raise ValueError("Symbol is outside the entropy-coder alphabet")
        interval = high - low + 1
        high = low + interval * cdf[symbol + 1] // cdf[-1] - 1
        low = low + interval * cdf[symbol] // cdf[-1]
        while True:
            if high < _HALF:
                emit(0)
            elif low >= _HALF:
                emit(1)
                low -= _HALF
                high -= _HALF
            elif low >= _QUARTER and high < _THREE_QUARTERS:
                pending += 1
                low -= _QUARTER
                high -= _QUARTER
            else:
                break
            low *= 2
            high = 2 * high + 1
    pending += 1
    emit(0 if low < _QUARTER else 1)
    bits.extend([0] * ((-len(bits)) % 8))
    return bytes(sum(bits[start + bit] << (7 - bit) for bit in range(8)) for start in range(0, len(bits), 8))


def decode_symbols(payload: bytes, cdfs: Sequence[Sequence[int]]) -> list[int]:
    """Decode using the exact same sequence of checkpoint-derived CDFs."""
    if not payload or not cdfs:
        raise ValueError("A nonempty arithmetic stream and CDF sequence are required")
    position = 0

    def read() -> int:
        nonlocal position
        bit = (payload[position // 8] >> (7 - position % 8)) & 1 if position < 8 * len(payload) else 0
        position += 1
        return bit

    value = 0
    for _ in range(_PRECISION):
        value = 2 * value + read()
    low, high = 0, _FULL - 1
    result = []
    checked = set()
    for cdf in cdfs:
        identity = id(cdf)
        if identity not in checked:
            _validate_cdf(cdf)
            checked.add(identity)
        interval = high - low + 1
        scaled = ((value - low + 1) * cdf[-1] - 1) // interval
        symbol = bisect_right(cdf, scaled) - 1
        if not 0 <= symbol < len(cdf) - 1:
            raise ValueError("Invalid arithmetic stream")
        result.append(symbol)
        high = low + interval * cdf[symbol + 1] // cdf[-1] - 1
        low = low + interval * cdf[symbol] // cdf[-1]
        while True:
            if high < _HALF:
                pass
            elif low >= _HALF:
                low -= _HALF
                high -= _HALF
                value -= _HALF
            elif low >= _QUARTER and high < _THREE_QUARTERS:
                low -= _QUARTER
                high -= _QUARTER
                value -= _QUARTER
            else:
                break
            low *= 2
            high = 2 * high + 1
            value = 2 * value + read()
    # Reject noncanonical/truncated streams despite arithmetic zero extension.
    if encode_symbols(result, cdfs) != payload:
        raise ValueError("Noncanonical or truncated arithmetic stream")
    return result


__all__ = ["encode_symbols", "decode_symbols"]
