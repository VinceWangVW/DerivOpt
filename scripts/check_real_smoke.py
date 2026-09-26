"""Run short CPU training/evaluation on downloaded native PDEBench subsets."""
import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
import time

import torch

from derivopt.config import ExperimentConfig
from derivopt.data import FAMILY_CHANNELS, PDEBenchDataset
from derivopt.io import save_json
from derivopt.preparation import prepare
from derivopt.runner import evaluate_checkpoint, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/real-smoke"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--families", nargs="+", choices=tuple(FAMILY_CHANNELS), default=list(FAMILY_CHANNELS))
    parser.add_argument("--methods", nargs="+", default=["derivopt", "primitive"])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new output directory")
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    report = {"device": "cpu", "data": str(args.data.resolve()), "cases": [],
              "scope": "real-data smoke test"}
    for family in args.families:
        started = time.monotonic()
        source = (args.data / f"{family}.h5").resolve()
        dataset = PDEBenchDataset(source, family)
        config = ExperimentConfig.small(family=family, calibration_states=2)
        partitions = {"version": 1, **{split: [{"file": str(source), "indices": [index]}]
                      for index, split in enumerate(("train", "val", "test"))}}
        print(f"{family}: prepare", flush=True)
        try:
            prepared = prepare(dataset, config, partitions=partitions)
            for method in args.methods:
                output = args.output / family / method
                chosen = replace(config, method=method)
                print(f"{family}/{method}: train and test", flush=True)
                result = train(source, chosen, output / "train", prepared=prepared)
                evaluated = evaluate_checkpoint(output / "train" / "best.pt.gz", source,
                                                output / "eval", horizon=3)
                if result["completed_steps"] != chosen.train_steps:
                    raise AssertionError("The requested optimizer updates did not complete")
                if not result["predictor_parameter_updated"] or not result["nonzero_finite_gradient"]:
                    raise AssertionError("Training must update predictor weights with finite nonzero gradients")
                summary = evaluated["summary"]
                if summary["trajectory_count"] != 1 or summary["finite_trajectory_count"] != 1:
                    raise AssertionError("Expected one held-out trajectory with finite metrics")
                if any(row["rollout_steps"] != 3 for row in evaluated["rows"]):
                    raise AssertionError("The requested test rollout did not complete")
                row = {"family": family, "method": method, "status": "passed",
                       "shape": prepared.geometry.basis.shape, "boundaries": prepared.geometry.basis.boundaries,
                       "completed_steps": result["completed_steps"], "test_trajectories": evaluated["summary"]["trajectory_count"],
                       "finite_test_trajectories": evaluated["summary"]["finite_trajectory_count"],
                       "time_step": prepared.geometry.config["time_step"],
                       "elapsed_seconds": time.monotonic()-started}
                report["cases"].append(row)
                print(json.dumps(row), flush=True)
                del result, evaluated
            del prepared
        except Exception as error:
            report["cases"].append({"family": family, "status": "failed",
                                     "error": f"{type(error).__name__}: {error}"})
            print(json.dumps(report["cases"][-1]), flush=True)
        save_json(report, args.output / "report.json", overwrite=True)
        gc.collect()
    if any(row["status"] != "passed" for row in report["cases"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
