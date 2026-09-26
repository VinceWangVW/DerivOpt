"""Command-line entry points for DerivOpt data, experiments and evaluation."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
from typing import Any, Sequence

import torch

from .config import ExperimentConfig, main_matrix
from .data import FAMILY_CHANNELS, FAMILY_NDIM, PDEBenchDataset, write_fixture
from .protocol import BACKBONES, BUDGET_RATIOS, MAIN_METHODS, RETAIN_FRACS, MetricProtocol


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        # Keep undefined diagnostics explicit while emitting standards-valid JSON.
        return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return value


def _emit(value: Any, output: Path | None = None) -> None:
    serialized = json.dumps(_json_safe(value), indent=2, allow_nan=False) + "\n"
    if output is None:
        print(serialized, end="")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents both accidental replacement and a
        # check-then-write race against another manifest writer.
        with output.open("x", encoding="utf-8") as handle:
            handle.write(serialized)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="derivopt", description="Budget-aware carried-state PDE experiments")
    commands = parser.add_subparsers(dest="command", required=True)

    fixture = commands.add_parser("fixture", help="write a tiny manufactured native-schema HDF5 fixture")
    fixture.add_argument("--family", required=True, choices=tuple(FAMILY_CHANNELS))
    fixture.add_argument("--output", required=True, type=_path)
    fixture.add_argument("--trajectories", type=int, default=4)
    fixture.add_argument("--steps", type=int, default=5)
    fixture.add_argument("--size", type=int, nargs="+", help="one size or one size per axis; default 128 in 1D, 32x32 in 2D")
    fixture.add_argument("--seed", type=int, default=0)

    inspect = commands.add_parser("inspect-data", help="inspect schema and a bounded frame window of one trajectory")
    inspect.add_argument("--family", required=True, choices=tuple(FAMILY_CHANNELS))
    inspect.add_argument("--data", required=True, type=_path, nargs="+")
    inspect.add_argument("--index", type=int, default=0)
    inspect.add_argument("--frames", type=int, default=2,
                         help="read this many leading frames (default 2; at least 2 and no more than the trajectory length)")

    matrix = commands.add_parser("matrix", help="enumerate the 1764 main experiment configurations")
    matrix.add_argument("--output", type=_path, help="write the full JSON manifest")
    matrix.add_argument("--list", action="store_true", help="print the full manifest instead of only its dimensions")
    matrix.add_argument("--smoke", action="store_true", help="use small training/model settings in every configuration")
    matrix.add_argument("--seeds", type=int, nargs="+", help="repeat the matrix for each unique training seed; keep the split seed fixed")

    calibrate = commands.add_parser("calibrate", help="fit train-only channels and select a reusable carried-state design")
    calibrate.add_argument("--config", required=True, type=_path)
    calibrate.add_argument("--data", required=True, type=_path, nargs="+")
    calibrate.add_argument("--output", required=True, type=_path, help="prepared-state .pt or .pt.gz file")
    calibrate.add_argument("--partitions", type=_path, help="JSON manifest preserving explicit train/val/test file-local indices")
    calibrate.add_argument("--fold", type=int, help="leave out this zero-based fold from training-only calibration")
    calibrate.add_argument("--folds", type=int, default=5, help="number of training-trajectory calibration folds (default 5)")

    inputs = commands.add_parser("inputs", help="evaluate carried input states without a learned predictor")
    inputs.add_argument("--prepared", required=True, type=_path)
    inputs.add_argument("--data", required=True, type=_path, nargs="+")
    inputs.add_argument("--output", required=True, type=_path)
    inputs.add_argument("--split", choices=("train", "val", "test"), default="test")
    inputs.add_argument("--max-states", type=int, default=16)
    inputs.add_argument("--methods", nargs="+", choices=("primitive", "best_single", "derivbase", "derivopt"),
                        default=("primitive", "best_single", "derivbase", "derivopt"))

    train = commands.add_parser("train", help="train one configured method and save its checkpoint")
    train.add_argument("--config", required=True, type=_path)
    train.add_argument("--data", required=True, type=_path, nargs="+")
    train.add_argument("--output", required=True, type=_path, help="experiment output directory")
    train.add_argument("--prepared", type=_path, help="reuse a saved train-only prepared state")
    train.add_argument("--resume", type=_path, help="resume a compatible training checkpoint")
    train.add_argument("--partitions", type=_path, help="explicit partition JSON; must match saved partitions when reusing preparation or resuming")
    train.add_argument("--data-map", type=_path, help="JSON mapping old absolute paths to copied data files when reusing preparation or resuming")

    evaluate = commands.add_parser("evaluate", help="evaluate a checkpoint in a fully closed loop")
    evaluate.add_argument("--checkpoint", required=True, type=_path)
    evaluate.add_argument("--data", required=True, type=_path, nargs="+")
    evaluate.add_argument("--output", required=True, type=_path)
    evaluate.add_argument("--split", choices=("train", "val", "test"), default="test")
    evaluate.add_argument("--horizon", type=int)
    evaluate.add_argument("--max-trajectories", type=int)
    evaluate.add_argument("--data-map", type=_path, help="JSON mapping old absolute paths to copied data files")

    reevaluate = commands.add_parser("reevaluate", help="recompute detail metrics from saved predictions without retraining")
    reevaluate.add_argument("--predictions", required=True, type=_path)
    reevaluate.add_argument("--output", required=True, type=_path)
    reevaluate.add_argument("--tau-q", type=float, default=1.25)
    reevaluate.add_argument("--tau-out", type=float, default=.1)
    reevaluate.add_argument("--fine-low", type=float, default=.75)
    reevaluate.add_argument("--fine-high", type=float, default=1.)

    smoke = commands.add_parser("smoke", help="run short training, codec and rollout tests")
    smoke.add_argument("--output", required=True, type=_path)
    smoke.add_argument("--suite", choices=("quick", "main", "regimes", "shared", "all"), default="quick")
    return parser


def _dispatch(args: argparse.Namespace) -> None:
    if args.command == "fixture":
        ndim = FAMILY_NDIM[args.family]
        if args.size is None:
            size = 128 if ndim == 1 else 32
        else:
            size = args.size[0] if len(args.size) == 1 else tuple(args.size)
            if len(args.size) not in (1, ndim):
                raise ValueError("--size needs one integer or one integer per spatial axis")
        path = write_fixture(args.output, args.family, trajectories=args.trajectories,
                             steps=args.steps, spatial_size=size, seed=args.seed)
        sample = PDEBenchDataset(path, args.family)[0]
        _emit({"path": path, "family": args.family, "trajectories": args.trajectories,
               "trajectory_shape": list(sample.states.shape), "manufactured_fixture": True})
    elif args.command == "inspect-data":
        dataset = PDEBenchDataset(args.data, args.family)
        if not 0 <= args.index < len(dataset):
            raise ValueError("--index must identify an existing trajectory")
        trajectory_length = dataset.trajectory_length(args.index)
        if not 2 <= args.frames <= trajectory_length:
            raise ValueError(f"--frames must be between 2 and the trajectory length ({trajectory_length})")
        sample = dataset.read_window(args.index, 0, args.frames)
        _emit({"family": args.family, "files": dataset.paths, "trajectory_count": len(dataset),
               "inspected_index": args.index, "trajectory_length": trajectory_length,
               "sampled_frame_count": sample.states.shape[0], "statistics_scope": "sampled_frames",
               "shape": list(sample.states.shape),
               "dtype": str(sample.states.dtype), "channels": sample.metadata["channel_names"],
               "times": {"count": sample.times.numel(), "first": sample.times[0], "last": sample.times[-1]},
               "coordinates": [{"count": axis.numel(), "first": axis[0], "last": axis[-1],
                                "min_spacing": axis.diff().min(), "max_spacing": axis.diff().max()}
                               for axis in sample.coords],
               "value_range": [sample.states.min(), sample.states.max()], "metadata": sample.metadata})
    elif args.command == "matrix":
        seeds = args.seeds
        if seeds is not None and (any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds)):
            raise ValueError("--seeds must contain distinct nonnegative integers")
        configurations = [replace(config, seed=seed).to_dict()
                          for config in main_matrix(smoke=args.smoke)
                          for seed in (seeds if seeds is not None else (config.seed,))]
        summary = {"count": len(configurations), "family_count": len(FAMILY_CHANNELS),
                   "backbone_count": len(BACKBONES), "budget_count": len(BUDGET_RATIOS),
                   "retain_count": len(RETAIN_FRACS), "method_count": len(MAIN_METHODS),
                   "smoke_settings": args.smoke, "seeds": seeds if seeds is not None else [0],
                   "seed_count": len(seeds) if seeds is not None else 1}
        manifest = {**summary, "configurations": configurations}
        if args.output is not None:
            _emit(manifest, args.output)
        _emit(manifest if args.list else {**summary, **({"output": args.output} if args.output else {})})
    elif args.command == "calibrate":
        from .io import save_torch
        from .preparation import prepare
        config = ExperimentConfig.from_file(args.config)
        if args.folds < 2:
            raise ValueError("--folds must be at least 2")
        prepared = prepare(PDEBenchDataset(args.data, config.family), config,
                           calibration_fold=args.fold, folds=args.folds, partitions=args.partitions)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_torch(prepared.state_dict(), args.output)
        _emit({"prepared": args.output, "family": config.family, "library": config.library,
               "coarse_shape": prepared.geometry.resampler.coarse_shape,
               "design": prepared.optimum.to_dict(), "provenance": prepared.provenance})
    elif args.command == "inputs":
        from .input_analysis import evaluate_inputs
        if args.max_states < 1:
            raise ValueError("--max-states must be positive")
        if len(set(args.methods)) != len(args.methods):
            raise ValueError("--methods must not repeat a method")
        result = evaluate_inputs(args.prepared, args.data, args.output, methods=tuple(args.methods),
                                 split=args.split, max_states=args.max_states)
        _emit({"output": args.output, **result})
    elif args.command == "train":
        from .runner import train
        config = ExperimentConfig.from_file(args.config)
        result = train(args.data, config, args.output, prepared_path=args.prepared, resume_path=args.resume,
                       partitions=args.partitions, data_map=args.data_map)
        artifacts = {name: (args.output / name if (args.output / name).exists() else None)
                     for name in ("last.pt.gz", "best.pt.gz", "predictions.pt.gz", "results.json")}
        _emit({"artifacts": artifacts, **result})
    elif args.command == "evaluate":
        from .runner import evaluate_checkpoint
        if args.horizon is not None and args.horizon < 1:
            raise ValueError("--horizon must be positive")
        if args.max_trajectories is not None and args.max_trajectories < 1:
            raise ValueError("--max-trajectories must be positive")
        result = evaluate_checkpoint(args.checkpoint, args.data, args.output, split=args.split,
                                     max_trajectories=args.max_trajectories, horizon=args.horizon, data_map=args.data_map)
        artifacts = {name: (args.output / name if (args.output / name).exists() else None)
                     for name in ("predictions.pt.gz", "evaluation.json")}
        _emit({"artifacts": artifacts, **result})
    elif args.command == "reevaluate":
        from .runner import reevaluate
        protocol = MetricProtocol(tau_q=args.tau_q, tau_out=args.tau_out,
                                  fine_low=args.fine_low, fine_high=args.fine_high)
        result = reevaluate(args.predictions, args.output, protocol)
        _emit({"output": args.output, **result})
    elif args.command == "smoke":
        from .acceptance import run_suite
        _emit(run_suite(args.output, suite=args.suite))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _dispatch(args)
    except (ValueError, FileNotFoundError, FileExistsError, KeyError, TypeError) as error:
        parser.error(str(error))
    return 0
