"""External-state budgets, deliberately separate from predictor memory."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class PayloadBudget:
    retained_sites: int
    primitive_components: int
    budget_ratio: float

    def __post_init__(self):
        if self.retained_sites < 1 or self.primitive_components < 1:
            raise ValueError("retained_sites and primitive_components must be positive")
        if not math.isfinite(self.budget_ratio) or not 0 < self.budget_ratio <= 1:
            raise ValueError("budget_ratio must be in (0, 1]")

    @property
    def nominal_bits(self) -> int:
        return math.floor(32 * self.primitive_components * self.retained_sites * self.budget_ratio)

    @property
    def cap_bytes(self) -> int:
        # A physical byte cannot be partially transmitted for free.
        return self.nominal_bits // 8

    @property
    def cap_bits(self) -> int:
        return self.cap_bytes * 8

    @property
    def field_bits_per_site(self) -> int:
        return self.cap_bits // self.retained_sites


@dataclass(frozen=True)
class PayloadLedger:
    cap_bits: int
    field_bits: int = 0
    primary_bits: int = 0
    hyper_bits: int = 0
    header_bits: int = 0
    padding_bits: int = 0

    @property
    def total_bits(self) -> int:
        return self.field_bits + self.primary_bits + self.hyper_bits + self.header_bits + self.padding_bits

    def validate(self, payload: bytes) -> None:
        if any(value < 0 for value in asdict(self).values()):
            raise ValueError("negative payload ledger entry")
        if self.total_bits != len(payload) * 8:
            raise ValueError("ledger does not match realized serialized bytes")
        if self.total_bits > self.cap_bits:
            raise ValueError("realized payload exceeds hard cap")

    def to_dict(self):
        return {**asdict(self), "total_bits": self.total_bits}

