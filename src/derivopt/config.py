"""Small explicit experiment configurations; no implicit test-set tuning."""
from dataclasses import asdict, dataclass, field, replace
from itertools import product
import json
import math
from numbers import Real
from pathlib import Path

from .data import FAMILY_CHANNELS
from .protocol import BACKBONES, BUDGET_RATIOS, MAIN_METHODS, RETAIN_FRACS


@dataclass
class ExperimentConfig:
    family: str = "incompressible_ns"
    backbone: str = "fno"
    method: str = "derivopt"
    budget_ratio: float = .25
    retain_frac: float = .125
    library: str = "primary"
    seed: int = 0
    split_seed: int = 0
    calibration_states: int = 32
    train_steps: int = 1000
    batch_size: int = 4
    learning_rate: float = .001
    rollout_steps: int = 20
    rollout_supervision: int = 3
    recurrent_training_steps: int = 3
    latent_rate_weight: float = .001
    latent_reconstruction_weight: float = 1.0
    validation_every: int = 50
    boundary_override: str | None = None
    boundary_parameters: dict = field(default_factory=dict)
    model_kwargs: dict | None = None
    latent_kwargs: dict = field(default_factory=dict)
    selector_max_nodes: int | None = None
    smoke: bool = False

    def __post_init__(self):
        if self.model_kwargs is None:
            # Two width-32 recurrent layers retain 512 KiB of FP32 hidden/cell
            # state per trajectory on the canonical 32 x 32 simulator grid.
            self.model_kwargs = {"width": 32 if self.backbone == "convlstm" else 16,
                                 "depth": 2, "modes": 8, "patch_size": 4, "num_heads": 4}

    def validate(self):
        if self.family not in FAMILY_CHANNELS or self.backbone not in BACKBONES:
            raise ValueError("unknown PDE family or backbone")
        if self.method not in (*MAIN_METHODS, "derivopt_archmulti"):
            raise ValueError("unknown method")
        if self.library not in ("primary", "small_expert", "shared"):
            raise ValueError("library must be primary, small_expert or shared")
        for name in ("budget_ratio", "retain_frac", "learning_rate", "latent_rate_weight", "latent_reconstruction_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if not 0 < self.budget_ratio <= 1 or not 0 < self.retain_frac <= 1:
            raise ValueError("budget and retain ratios must be in (0,1]")
        for name in ("calibration_states", "train_steps", "batch_size", "rollout_steps", "rollout_supervision",
                     "recurrent_training_steps", "validation_every"):
            if isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("seed", "split_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.selector_max_nodes is not None and (isinstance(self.selector_max_nodes, bool)
                or not isinstance(self.selector_max_nodes, int) or self.selector_max_nodes < 1):
            raise ValueError("selector_max_nodes must be a positive integer or None")
        def finite_options(value, name):
            if isinstance(value, dict):
                for key, item in value.items():
                    finite_options(item, f"{name}.{key}")
            elif isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    finite_options(item, f"{name}[{index}]")
            elif isinstance(value, Real) and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        finite_options(self.model_kwargs, "model_kwargs")
        finite_options(self.latent_kwargs, "latent_kwargs")
        if self.calibration_states < 2 or self.learning_rate <= 0:
            raise ValueError("need >=2 calibration states and a positive learning rate")
        if self.latent_rate_weight < 0 or self.latent_reconstruction_weight < 0:
            raise ValueError("loss weights cannot be negative")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_file(cls, path):
        path = Path(path)
        if path.suffix in (".yaml", ".yml"):
            import yaml
            values = yaml.safe_load(path.read_text())
        else:
            values = json.loads(path.read_text())
        return cls(**values).validate()

    @classmethod
    def small(cls, **kwargs):
        defaults = dict(calibration_states=6, train_steps=2, batch_size=1, rollout_steps=3,
                        validation_every=1, model_kwargs={"width": 4, "depth": 1, "modes": 2, "patch_size": 2, "num_heads": 1},
                        latent_kwargs={"latent_channels": 2, "width": 4}, smoke=True)
        defaults.update(kwargs)
        return cls(**defaults).validate()


def main_matrix(*, smoke=False):
    constructor = ExperimentConfig.small if smoke else ExperimentConfig
    for family, backbone, ratio, retain, method in product(FAMILY_CHANNELS, BACKBONES, BUDGET_RATIOS, RETAIN_FRACS, MAIN_METHODS):
        yield constructor(family=family, backbone=backbone, budget_ratio=ratio, retain_frac=retain, method=method).validate()


def paired_seed_configs(base: ExperimentConfig, seeds, methods=("derivopt", "archmulti")):
    for seed, method in product(seeds, methods):
        yield replace(base, seed=int(seed), method=method).validate()
