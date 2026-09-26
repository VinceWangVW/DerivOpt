"""Short integration checks for training, serialization and closed-loop evaluation."""
from dataclasses import replace
from itertools import product
import math
from pathlib import Path
import time
import traceback

import torch

from .budget import PayloadBudget
from .config import ExperimentConfig, main_matrix
from .data import FAMILY_CHANNELS, FAMILY_NDIM, PDEBenchDataset, write_fixture
from .io import save_json
from .preparation import prepare
from .protocol import BACKBONES, BUDGET_RATIOS, MAIN_METHODS, RETAIN_FRACS
from .runner import build_simulator, environment, train


class AcceptanceFailure(RuntimeError):
    """At least one requested execution failed; inspect the saved report."""


def _training_check(paths, config, output, prepared):
    record = train(paths, config, output, prepared=prepared)
    if not record["predictor_parameter_updated"] or not record["nonzero_finite_gradient"]:
        raise AssertionError("training did not update the predictor with finite nonzero gradients")
    for row in record["evaluation"]["rows"]:
        # Q is undefined for zero-energy bands. Check physical error and horizon
        # here while preserving undefined detail metrics in the run record.
        if not math.isfinite(row["nrmse"]):
            raise AssertionError("nonfinite closed-loop physical error")
        if not 0 <= row["detail_horizon"] <= 1:
            raise AssertionError("detail horizon outside [0,1]")
    cap = PayloadBudget(math.prod(prepared.geometry.resampler.coarse_shape),
                        len(FAMILY_CHANNELS[config.family]), config.budget_ratio).cap_bits
    payloads = [bits for row in record["history"] for bits in row["payload_bits"]]
    payloads += [bits for row in record["evaluation"]["rows"] for bits in row["payload_bits"]]
    if not payloads or any(bits != cap for bits in payloads):
        raise AssertionError("actual padded training/evaluation records violate the payload cap")
    for filename in ("last.pt.gz", "best.pt.gz", "predictions.pt.gz", "results.json"):
        if not (output / filename).is_file():
            raise AssertionError(f"missing execution artifact {filename}")
    return {"completed_steps": record["completed_steps"], "predictor_parameter_updated": True,
            "nonzero_finite_gradient": True, "record_cap_bits": cap,
            "fine_shape": list(prepared.geometry.basis.shape),
            "coarse_shape": list(prepared.geometry.resampler.coarse_shape),
            "calibration_states": prepared.provenance["calibration_state_count"],
            "checked_records": len(payloads), "results": str((output / "results.json").resolve())}


def _regime_check(dataset, config):
    prepared = prepare(dataset, config)
    budget = PayloadBudget(math.prod(prepared.geometry.resampler.coarse_shape),
                           len(FAMILY_CHANNELS[config.family]), config.budget_ratio)
    inspected = []
    for method in MAIN_METHODS:
        simulator = build_simulator(prepared, replace(config, method=method))
        with torch.no_grad():
            decoded, info = simulator.input_reconstruction(dataset[0].states[:1], actual_codec=True)
        if not torch.isfinite(decoded).all():
            raise AssertionError(f"nonfinite {method} input reconstruction")
        for payload, ledger in zip(info["payloads"], info["ledgers"]):
            ledger.validate(payload)
            if ledger.total_bits != budget.cap_bits or len(payload)*8 != budget.cap_bits:
                raise AssertionError(f"incorrect {method} record length")
        inspected.append(method)
    return {"exact": prepared.optimum.exact, "visited_nodes": prepared.optimum.visited_nodes,
            "coarse_shape": list(prepared.geometry.resampler.coarse_shape),
            "cap_bits": budget.cap_bits, "methods_encoded_decoded": inspected}


def run_suite(output_dir, suite="quick"):
    """Run selected checks in a new directory and preserve failures in report.json.

    quick: seven family DerivOpt paths and all seven INS/FNO methods.
    main: all 196 canonical family/backbone/method short training paths.
    regimes: all 63 family/budget/grid calibrations plus all method codecs.
    shared: four explicit controls on all seven shared libraries and the
            DerivOpt+ArchMulti combination on four INS backbones.
    all: main + regimes + shared (quick is a subset of main).
    """
    if suite not in ("quick", "main", "regimes", "shared", "all"):
        raise ValueError("unknown acceptance suite")
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("acceptance output directory must be empty; choose a new directory")
    output.mkdir(parents=True, exist_ok=True)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    start = time.monotonic()
    report = {"suite": suite, "status": "running", "data_kind": "manufactured_native_schema_fixtures",
              "run_type": "smoke", "environment": environment(),
              "main_configurations_validated": len(list(main_matrix())), "checks": []}
    paths, datasets, shared_paths, prepared_cache = {}, {}, {}, {}

    def case(name, operation):
        tick = time.monotonic()
        try:
            detail = operation()
            entry = {"name": name, "status": "passed", "detail": detail}
        except Exception as error:
            entry = {"name": name, "status": "failed", "error": f"{type(error).__name__}: {error}",
                     "traceback": traceback.format_exc()}
        entry["elapsed_seconds"] = time.monotonic()-tick
        report["checks"].append(entry)
        save_json(report, output/"report.json", overwrite=True)

    def prepared_for(config):
        key = (config.family, config.library)
        if key not in prepared_cache:
            dataset = (PDEBenchDataset(shared_paths[config.family], config.family)
                       if config.library == "shared" and config.family in shared_paths else datasets[config.family])
            prepared_cache[key] = prepare(dataset, config)
        return prepared_cache[key]

    def training_case(config, group):
        name = f"{group}/{config.family}/{config.backbone}/{config.method}"
        path = shared_paths.get(config.family, paths[config.family]) if config.library == "shared" else paths[config.family]
        case(name, lambda: _training_check(path, config, output/name, prepared_for(config)))

    try:
        for family in FAMILY_CHANNELS:
            path = output/"fixtures"/f"{family}.h5"
            paths[family] = write_fixture(path, family, trajectories=4, steps=5,
                                         spatial_size=128 if FAMILY_NDIM[family] == 1 else 32)
            datasets[family] = PDEBenchDataset(path, family)
        if suite in ("main", "all"):
            for family, backbone, method in product(FAMILY_CHANNELS, BACKBONES, MAIN_METHODS):
                training_case(ExperimentConfig.small(family=family, backbone=backbone, method=method), "main")
        if suite == "quick":
            combinations = {(family, "fno", "derivopt") for family in FAMILY_CHANNELS}
            combinations.update(("incompressible_ns", "fno", method) for method in MAIN_METHODS)
            for family, backbone, method in sorted(combinations):
                training_case(ExperimentConfig.small(family=family, backbone=backbone, method=method), "quick")
        if suite in ("regimes", "all"):
            for family, ratio, retain in product(FAMILY_CHANNELS, BUDGET_RATIOS, RETAIN_FRACS):
                config = ExperimentConfig.small(family=family, budget_ratio=ratio, retain_frac=retain)
                name = f"regimes/{family}/R{ratio}_f{retain}"
                case(name, lambda c=config: _regime_check(datasets[c.family], c))
        if suite in ("shared", "all"):
            # A rectangular CNS fixture retains non-DC modes and derived-field
            # information while keeping this exact combinatorial search small.
            # Budget, retain fraction, all 42 candidates, calibration count and
            # the full feasible integer domain are unchanged. Search time here
            # is specific to this rectangular test grid.
            shared_paths["compressible_ns"] = write_fixture(
                output/"fixtures"/"compressible_ns_shared.h5", "compressible_ns",
                trajectories=4, steps=5, spatial_size=(32, 16))
            report["shared_cns_grid"] = {"fine": [32, 16], "coarse": [4, 2], "candidate_count": 42}
            for family, method in product(FAMILY_CHANNELS, ("primitive", "best_single", "derivbase", "derivopt")):
                training_case(ExperimentConfig.small(family=family, method=method, library="shared"), "shared")
            for backbone in BACKBONES:
                training_case(ExperimentConfig.small(backbone=backbone, method="derivopt_archmulti"), "combined")
        failures = sum(entry["status"] == "failed" for entry in report["checks"])
        report.update(status="failed" if failures else "passed", failed=failures,
                      passed=len(report["checks"])-failures, elapsed_seconds=time.monotonic()-start)
        save_json(report, output/"report.json", overwrite=True)
        if failures:
            raise AcceptanceFailure(f"{failures} acceptance checks failed; see {output / 'report.json'}")
        return {key: value for key, value in report.items() if key != "checks"} | {"report": str(output/"report.json")}
    finally:
        torch.set_num_threads(previous_threads)
