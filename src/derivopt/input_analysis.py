"""Predictor-free input fidelity and calibrated single-channel risk curves."""
from dataclasses import replace
from pathlib import Path

import torch

from .config import ExperimentConfig
from .data import PDEBenchDataset
from .io import dataset_identity, load_torch, save_json, save_torch
from .metrics import build_fine_mask, compute_metrics
from .preparation import PreparedState
from .runner import build_simulator


@torch.no_grad()
def evaluate_inputs(prepared_path, data_paths, output_dir, *,
                    methods=("primitive", "best_single", "derivbase", "derivopt"),
                    split="test", max_states=16):
    """Decode each held-out trajectory's t=0 without using any predictor.

    Learned-latent input diagnostics require a trained checkpoint and are
    available as the t=0 rows of ``evaluate_checkpoint`` instead.
    """
    methods = tuple(methods)
    allowed = {"primitive", "best_single", "derivbase", "derivopt"}
    if not methods or set(methods)-allowed or len(set(methods)) != len(methods):
        raise ValueError("input methods must be distinct explicit-state controls")
    if split not in ("train", "val", "test") or not isinstance(max_states, int) or max_states < 1:
        raise ValueError("need a valid split and positive max_states")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("input analysis output must be a new or empty directory")
    prepared = PreparedState.from_state_dict(load_torch(prepared_path))
    config = ExperimentConfig(**prepared.config).validate()
    dataset = PDEBenchDataset(data_paths, config.family)
    if prepared.provenance.get("dataset_identity") != dataset_identity(dataset):
        raise ValueError("source dataset identity differs from prepared input analysis")
    selected = prepared.reopen_splits(dataset)[split]
    fields, identities = [], []
    for index in range(min(len(selected), max_states)):
        sample = selected.read_window(index, 0, 2)
        prepared.geometry.validate_sample(sample)
        fields.append(sample.states[0])
        identities.append({"trajectory_id": sample.metadata["trajectory_id"], "frame": 0})
        if len(fields) >= max_states:
            break
    if not fields:
        raise ValueError("input analysis split has no states")
    target = torch.stack(fields)
    decoded_states, result = {}, {"family": config.family, "library": config.library, "split": split,
                                  "predictor_used": False, "frame_protocol": "trajectory_t0", "state_count": len(fields),
                                  "state_identities": identities, "methods": {}}
    for method in methods:
        simulator = build_simulator(prepared, replace(config, method=method))
        decoded, info = simulator.input_reconstruction(target, actual_codec=True)
        for payload, ledger in zip(info["payloads"], info["ledgers"]):
            ledger.validate(payload)
        mask = prepared.geometry.resampler.expressible_mask
        fine = build_fine_mask(prepared.geometry.basis, mask,
                              cutoff=prepared.geometry.resampler.retained_cutoff)
        metrics = compute_metrics(decoded, target, prepared.geometry.basis, mask, fine)
        means = {key: float(value.to(torch.float64).mean()) for key, value in metrics.items() if value.ndim == 1}
        result["methods"][method] = {"design": prepared.design_for(method).to_dict(), "mean": means,
                                      "per_state": metrics, "ledgers": [ledger.to_dict() for ledger in info["ledgers"]]}
        decoded_states[method] = decoded

    problem = prepared.calibration.design_problem()
    curves = []
    for index, options in enumerate(problem.precisions):
        for bits in sorted(options):
            allocation = tuple(bits if j == index else 0 for j in range(len(problem.names)))
            curves.append({"field": problem.names[index], "bits_per_component": bits,
                           "field_bits_per_site": problem.component_counts[index]*bits,
                           "calibrated_risk": problem.risk(allocation)})
    result["calibration_split"] = "train"
    save_json(result, output/"input_metrics.json")
    save_json({"source": "train-only calibrated channel model", "curves": curves}, output/"calibration_curves.json")
    save_torch({"geometry": prepared.geometry.state_dict(), "target": target, "decoded": decoded_states,
                "state_identities": identities}, output/"input_fields.pt.gz")
    return result
