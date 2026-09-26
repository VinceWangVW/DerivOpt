"""Training, closed-loop evaluation and prediction-only threshold re-evaluation."""
from dataclasses import asdict, replace
import copy
import math
from pathlib import Path
import platform
import time

import torch

from .config import ExperimentConfig
from .data import PDEBenchDataset, TrajectoryWindows, partition_trajectories
from .geometry import Geometry
from .io import dataset_identity, load_torch, remap_trajectory_ids, save_json, save_torch, validate_dataset_identity
from .metrics import build_fine_mask, compute_metrics, detail_horizon, detail_horizon_steps
from .preparation import PreparedState, prepare
from .protocol import MetricProtocol
from .simulation import BudgetedSimulator


def environment():
    return {"python": platform.python_version(), "platform": platform.platform(), "torch": str(torch.__version__),
            "device": "cpu", "threads": torch.get_num_threads()}


def build_simulator(prepared, config):
    for key in ("family", "budget_ratio", "retain_frac", "library", "split_seed", "calibration_states",
                "boundary_override", "boundary_parameters"):
        if prepared.config[key] != config.to_dict()[key]:
            raise ValueError(f"prepared state and experiment differ on {key}")
    design_bits = None if config.method == "latent_hyper" else prepared.design_for(config.method).bits
    torch.manual_seed(config.seed)
    return BudgetedSimulator(prepared.calibration, design_bits, prepared.geometry, backbone=config.backbone,
                             method=config.method, model_kwargs=config.model_kwargs, mean=prepared.mean, std=prepared.std,
                             latent_kwargs=config.latent_kwargs, budget_ratio=config.budget_ratio)


class GeometryCheckedDataset:
    """Validate each already-read trajectory once without extra field I/O.

    TrajectoryWindows delegates loading to this view, so shape/coordinate and
    boundary checks cover all sampled training trajectories, not only the small
    calibration subset. No spectral basis is constructed during these checks.
    """

    def __init__(self, dataset, geometry):
        self.dataset, self.geometry = dataset, geometry
        self.split = dataset.split
        self._validated = set()

    def __len__(self):
        return len(self.dataset)

    def trajectory_length(self, index):
        return self.dataset.trajectory_length(index)

    def __getitem__(self, index):
        sample = self.dataset[index]
        identity = sample.metadata["trajectory_id"]
        if identity not in self._validated:
            self.geometry.validate_sample(sample)
            self._validated.add(identity)
        return sample

    def read_window(self, index, start=0, stop=None):
        sample = self.dataset.read_window(index, start, stop)
        # Validate dt for each window, including rounded time coordinates at
        # later time origins. This touches coordinates, not extra field data.
        self.geometry.validate_sample(sample)
        return sample


def _batch_for_step(windows, step, batch_size, seed):
    """Deterministic shuffles indexed by global step, including resume."""
    permutations, selected = {}, []
    for position in range(step*batch_size, (step+1)*batch_size):
        epoch, offset = divmod(position, len(windows))
        if epoch not in permutations:
            permutations[epoch] = torch.randperm(len(windows), generator=torch.Generator().manual_seed(seed+epoch)).tolist()
        selected.append(windows[permutations[epoch][offset]])
    return torch.stack([row["inputs"][-1] for row in selected]), torch.stack([row["targets"] for row in selected])


def _loss(simulator, initial, targets, config):
    simulator.reset_state()
    current, losses, ledger_bits = initial, [], []
    shape = (1, simulator.channels, *([1]*simulator.spatial_dim))
    scale = simulator.primitive_std.reshape(shape)
    for target in targets.unbind(1):
        current, info = simulator.step(current, actual_codec=False)
        physical_loss = ((current-target)/scale).square().mean()
        loss = physical_loss
        if config.method == "latent_hyper":
            loss = loss + config.latent_rate_weight*info["rate_loss"] + config.latent_reconstruction_weight*info["codec_reconstruction_loss"]
        losses.append(loss)
        ledger_bits.extend(ledger.total_bits for ledger in info["ledgers"])
    return torch.stack(losses).mean(), ledger_bits


def _summary(rows):
    keys = ("nrmse", "fine_rel", "detail_horizon", "detail_horizon_normalized", "detail_horizon_steps",
            "input_fine_rel", "input_q_fine", "input_e_out", "input_pass")
    summary = {"trajectory_count": len(rows)}
    for key in keys:
        values = [float(row[key]) for row in rows]
        # Preserve undefined/nonfinite status, don't silently select a subset.
        summary[key] = sum(values)/len(values) if values else float("nan")
    summary["finite_trajectory_count"] = sum(row["finite"] for row in rows)
    return summary


def score_predictions(predictions, geometry, protocol=MetricProtocol()):
    rows = []
    mask = geometry.resampler.expressible_mask
    fine = build_fine_mask(geometry.basis, mask, low=protocol.fine_low, high=protocol.fine_high,
                           cutoff=geometry.resampler.retained_cutoff)
    for record in predictions:
        estimated, truth = record["predicted"], record["target"]
        if estimated.shape != truth.shape or estimated.ndim != geometry.basis.ndim+2:
            raise ValueError("prediction record must be matching [time,channel,*fine] fields")
        # Frames act as independent batch entries for the metric calculation.
        measured = compute_metrics(estimated, truth, geometry.basis, mask, fine, protocol=protocol)
        passed = measured["input_pass"].unsqueeze(0)
        horizon = float(detail_horizon(passed)[0])
        row = {"trajectory_id": record["trajectory_id"], "nrmse": float(measured["nrmse"][1:].mean()),
               "final_nrmse": float(measured["nrmse"][-1]), "fine_rel": float(measured["fine_rel"][-1]),
               "detail_horizon": horizon, "detail_horizon_normalized": horizon,
               "detail_horizon_steps": int(detail_horizon_steps(passed)[0]),
               "rollout_steps": estimated.shape[0]-1, "input_fine_rel": float(measured["fine_rel"][0]),
               "input_q_fine": float(measured["q_fine"][0]), "input_e_out": float(measured["e_out"][0]),
               "input_pass": bool(measured["input_pass"][0]), "finite": bool(measured["finite_mask"].all()),
               "timesteps": measured, "payload_bits": record.get("payload_bits", [])}
        rows.append(row)
    return {"rows": rows, "summary": _summary(rows), "metric_protocol": asdict(protocol),
            "horizon_units": {"detail_horizon": "normalized", "detail_horizon_normalized": "normalized",
                              "detail_horizon_steps": "transitions"},
            "horizon_protocol": "longest_contiguous_prefix_including_t0; unconditional_trajectory_mean"}


@torch.no_grad()
def evaluate_simulator(simulator, dataset, *, horizon, max_trajectories=None, protocol=MetricProtocol()):
    if horizon < 1:
        raise ValueError("evaluation horizon must be positive")
    simulator.eval()
    predictions = []
    count = len(dataset) if max_trajectories is None else min(len(dataset), max_trajectories)
    if count < 1:
        raise ValueError("evaluation needs at least one trajectory")
    for index in range(count):
        sample = (dataset.read_window(index, 0, min(horizon+1, dataset.trajectory_length(index)))
                  if hasattr(dataset, "read_window") else dataset[index])
        simulator.geometry.validate_sample(sample)
        simulator.reset_state()
        steps = min(horizon, sample.states.shape[0]-1)
        if steps < 1:
            raise ValueError("trajectory has no next time step")
        current = sample.states[0:1]
        decoded, input_info = simulator.input_reconstruction(current, actual_codec=True)
        estimated = [decoded[0]]
        payloads = [input_info["ledgers"][0].total_bits]
        for _ in range(steps):
            # Only the t=0 input is ever read here; future ground truth is used
            # exclusively below for scoring after all predictions are formed.
            current, info = simulator.step(current, actual_codec=True)
            estimated.append(current[0])
            payloads.append(info["ledgers"][0].total_bits)
        predictions.append({"trajectory_id": sample.metadata["trajectory_id"], "predicted": torch.stack(estimated),
                            "target": sample.states[:steps+1].clone(), "times": sample.times[:steps+1].clone(),
                            "payload_bits": payloads})
    return score_predictions(predictions, simulator.geometry, protocol), predictions


def train(data_paths, config: ExperimentConfig, output_dir, *, prepared_path=None, resume_path=None, prepared=None,
          partitions=None, data_map=None):
    """Train for the configured number of updates and evaluate the selected model."""
    config.validate()
    output = Path(output_dir)
    _check_output_available(output)
    dataset = PDEBenchDataset(data_paths, config.family)
    identity = dataset_identity(dataset)
    if prepared is not None and prepared_path is not None:
        raise ValueError("supply a prepared object or a prepared path, not both")
    if data_map is not None and prepared is None and prepared_path is None and resume_path is None:
        raise ValueError("data_map requires an existing prepared state or resume checkpoint")
    if resume_path:
        if prepared is not None or prepared_path is not None:
            raise ValueError("resume includes its fixed prepared state; do not supply a replacement prepared state")
        resumed = load_torch(resume_path)
        if resumed.get("format_version") != 2:
            raise ValueError("Resume requires a version-2 checkpoint with fixed preparation, source identity and RNG state")
        _validate_identity(resumed.get("dataset_identity"), identity, data_map=data_map)
        old_config = resumed["experiment_config"]
        # train_steps means ADDITIONAL updates on resume. All other training,
        # calibration, model, split and evaluation settings stay explicit/fixed.
        for key, value in config.to_dict().items():
            if key != "train_steps" and old_config.get(key) != value:
                raise ValueError(f"resume configuration changed {key}")
        prepared = PreparedState.from_state_dict(resumed["prepared"])
        if prepared.partitions != resumed["partitions"]:
            raise ValueError("Checkpoint preparation and training partitions disagree")
        simulator = BudgetedSimulator.from_checkpoint_state(resumed["simulator"])
        start_step = int(resumed["completed_steps"])
        history = list(resumed["history"])
        best = float(resumed["best_validation_nrmse"])
        best_training_state = resumed["best_training_state"]
    else:
        prepared = prepared or (PreparedState.from_state_dict(load_torch(prepared_path)) if prepared_path
                                else prepare(dataset, config, partitions=partitions))
        simulator, start_step = build_simulator(prepared, config), 0
        history, best, best_training_state = [], math.inf, None
    mapping = _validate_identity(prepared.provenance.get("dataset_identity"), identity, data_map=data_map)
    relocation = _relocation_record(mapping)
    if mapping:
        _validate_relocated_geometry(dataset, prepared.geometry)
        prepared = _relocate_prepared(prepared, identity, mapping, relocation)
    if partitions is not None:
        requested = {name: list(view.trajectory_ids) for name, view in partition_trajectories(dataset, partitions).items()}
        if requested != prepared.partitions:
            raise ValueError("Partition manifest differs from fixed prepared/checkpoint partitions")
    splits = prepared.reopen_splits(dataset)
    optimizer = torch.optim.Adam(simulator.parameters(), lr=config.learning_rate)
    if resume_path:
        optimizer.load_state_dict(resumed["optimizer"])
        torch.set_rng_state(resumed["torch_rng_state"])
    supervision = config.rollout_supervision if config.method == "rolloutmulti" else 1
    windows = TrajectoryWindows(GeometryCheckedDataset(splits["train"], simulator.geometry), history=1, horizon=supervision)
    _check_output_available(output)
    output.mkdir(parents=True, exist_ok=True)
    start_time = time.monotonic()
    before = {name: value.detach().clone() for name, value in simulator.predictor.named_parameters()}
    gradient_seen = False

    def snapshot(completed_steps):
        return {"simulator": simulator.checkpoint_state(), "optimizer": copy.deepcopy(optimizer.state_dict()),
                "torch_rng_state": torch.get_rng_state().clone(), "completed_steps": completed_steps,
                "history": copy.deepcopy(history)}

    def checkpoint(training_state, role):
        return {"format_version": 2, **training_state, "experiment_config": config.to_dict(),
                "partitions": prepared.partitions, "prepared": prepared.state_dict(),
                "provenance": prepared.provenance, "dataset_identity": identity, "environment": environment(),
                "data_relocation": relocation,
                "checkpoint_role": role, "best_validation_nrmse": best,
                "best_completed_steps": None if best_training_state is None else best_training_state["completed_steps"],
                "best_training_state": best_training_state,
                "selection_criterion": "validation_mean_closed_loop_nrmse"}

    for step in range(start_step, start_step+config.train_steps):
        simulator.train()
        initial, targets = _batch_for_step(windows, step, config.batch_size, config.seed)
        optimizer.zero_grad(set_to_none=True)
        loss, payload_bits = _loss(simulator, initial, targets, config)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite training loss")
        loss.backward()
        gradients = [parameter.grad for parameter in simulator.parameters() if parameter.grad is not None]
        if not gradients or not all(torch.isfinite(value).all() for value in gradients):
            raise FloatingPointError("missing or non-finite training gradients")
        gradient_seen = gradient_seen or any(bool(torch.any(value != 0)) for value in gradients)
        optimizer.step()
        simulator.detach_state()
        row = {"step": step+1, "loss": float(loss.detach()), "payload_bits": payload_bits,
               "supervision_steps": supervision}
        improved = False
        if (step+1) % config.validation_every == 0 or step == start_step+config.train_steps-1:
            validation, _ = evaluate_simulator(simulator, splits["val"], horizon=config.rollout_steps,
                                               max_trajectories=1 if config.smoke else None)
            criterion = validation["summary"]["nrmse"]
            row["validation_nrmse"] = criterion
            if math.isfinite(criterion) and criterion < best:
                best = criterion
                improved = True
        history.append(row)
        if improved:
            best_training_state = snapshot(step+1)
            save_torch(checkpoint(best_training_state, "best"), output/"best.pt.gz", overwrite=True)
    # A resumed run can retain an earlier best model without another improvement.
    if best_training_state is not None and not (output/"best.pt.gz").exists():
        save_torch(checkpoint(best_training_state, "best"), output/"best.pt.gz")
    save_torch(checkpoint(snapshot(start_step+config.train_steps), "last"), output/"last.pt.gz")
    changed = any(not torch.equal(before[name], parameter.detach()) for name, parameter in simulator.predictor.named_parameters())
    # Test data never selects the epoch: use the validation-selected checkpoint.
    evaluation_role = "best" if best_training_state is not None else "last_no_finite_validation"
    evaluation_steps = best_training_state["completed_steps"] if best_training_state is not None else start_step+config.train_steps
    if best_training_state is not None:
        evaluated_simulator = BudgetedSimulator.from_checkpoint_state(best_training_state["simulator"])
    else:
        evaluated_simulator = simulator
    evaluation, predictions = evaluate_simulator(evaluated_simulator, splits["test"], horizon=config.rollout_steps,
                                                 max_trajectories=1 if config.smoke else None)
    selected_design = (prepared.design_for(config.method).to_dict() if config.method != "latent_hyper" else
                       {"kind": "learned_hyperprior", "explicit_fields": False, "cap_bytes": simulator.cap_bytes})
    record = {"config": config.to_dict(), "status": "completed", "profile": "smoke" if config.smoke else "experiment",
              "environment": environment(), "completed_steps": start_step+config.train_steps,
              "executed_steps": config.train_steps, "resumed_from": None if resume_path is None else str(Path(resume_path).resolve()),
              "predictor_parameter_updated": changed, "nonzero_finite_gradient": gradient_seen,
              "elapsed_seconds": time.monotonic()-start_time, "selected_design": selected_design,
              "evaluation_checkpoint": evaluation_role, "evaluation_completed_steps": evaluation_steps,
              "best_validation_nrmse": best, "selection_criterion": "validation_mean_closed_loop_nrmse",
              "history": history, "evaluation": evaluation, "provenance": prepared.provenance,
              "dataset_identity": identity, "data_relocation": relocation}
    save_json(record, output/"results.json")
    save_torch({"geometry": evaluated_simulator.geometry.state_dict(), "predictions": predictions,
                "config": config.to_dict(), "evaluation_checkpoint": evaluation_role,
                "evaluation_completed_steps": evaluation_steps, "dataset_identity": identity,
                "data_relocation": relocation}, output/"predictions.pt.gz")
    return record


def _check_output_available(output):
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output must be a new or empty directory; refusing to overwrite existing artifacts: {output}. Use a new output directory, including for resume.")


def _validate_identity(expected, current, *, data_map=None):
    return validate_dataset_identity(expected, current, data_map=data_map)


def _relocation_record(mapping):
    return ({"files": dict(mapping), "content_verified": False,
             "policy": "user_declared_copy; equal_size_reader_trajectory_ids_and_geometry_shape; mapped_mtime_ignored"}
            if mapping else None)


def _validate_relocated_geometry(dataset, geometry):
    # The reader has already validated each native layout and component schema.
    # Check all indexed trajectory shapes without reading any extra state frames.
    for record in dataset._records:
        shape = tuple(len(range(0, size, dataset.spatial_stride)) for size in record.spatial_shape)
        if shape != geometry.basis.shape or len(range(0, record.steps, dataset.time_stride)) < 2:
            raise ValueError("Mapped dataset trajectory schema/shape is incompatible with checkpoint geometry")


def _relocate_prepared(prepared, identity, mapping, relocation):
    # Copy only bookkeeping: never mutate the caller's preparation or recalibrate.
    partitions = {name: remap_trajectory_ids(ids, mapping) for name, ids in prepared.partitions.items()}
    provenance = copy.deepcopy(prepared.provenance)
    provenance.setdefault("data_relocations", []).append({**relocation, "source_identity": provenance["dataset_identity"]})
    provenance["dataset_identity"] = copy.deepcopy(identity)
    provenance["calibration_trajectory_ids"] = remap_trajectory_ids(provenance["calibration_trajectory_ids"], mapping)
    metadata = provenance.get("data_metadata", {})
    if metadata.get("path") in mapping:
        metadata["path"] = mapping[metadata["path"]]
    if "trajectory_id" in metadata:
        metadata["trajectory_id"] = remap_trajectory_ids([metadata["trajectory_id"]], mapping)[0]
    return replace(prepared, partitions=partitions, provenance=provenance)


def evaluate_checkpoint(checkpoint_path, data_paths, output_dir, *, split="test", max_trajectories=None, horizon=None,
                        data_map=None):
    _check_output_available(output_dir)
    state = load_torch(checkpoint_path)
    simulator = BudgetedSimulator.from_checkpoint_state(state["simulator"])
    config = ExperimentConfig(**state["experiment_config"]).validate()
    dataset = PDEBenchDataset(data_paths, config.family)
    identity = dataset_identity(dataset)
    mapping = _validate_identity(state.get("dataset_identity"), identity, data_map=data_map)
    relocation = _relocation_record(mapping)
    if mapping:
        _validate_relocated_geometry(dataset, simulator.geometry)
    partitions = {name: remap_trajectory_ids(ids, mapping) for name, ids in state["partitions"].items()}
    if split not in ("train", "val", "test"):
        raise ValueError("unknown evaluation split")
    identities = {key: index for index, key in enumerate(dataset.trajectory_ids)}
    partition_ids = [key for values in partitions.values() for key in values]
    if len(set(partition_ids)) != len(partition_ids) or set(partition_ids) != set(identities):
        raise ValueError("checkpoint train/val/test partition identities do not match the supplied data")
    try:
        selected = dataset.subset([identities[key] for key in partitions[split]], split=split)
    except KeyError as error:
        raise ValueError("checkpoint partition not present in supplied data") from error
    evaluated, predictions = evaluate_simulator(simulator, selected, horizon=config.rollout_steps if horizon is None else horizon, max_trajectories=max_trajectories)
    evaluated["checkpoint"] = {"path": str(Path(checkpoint_path).resolve()),
                               "role": state.get("checkpoint_role", "unspecified"), "completed_steps": state["completed_steps"]}
    evaluated["dataset_identity"], evaluated["data_relocation"] = identity, relocation
    save_json(evaluated, Path(output_dir)/"evaluation.json")
    save_torch({"geometry": simulator.geometry.state_dict(), "predictions": predictions, "config": config.to_dict(),
                "dataset_identity": identity, "data_relocation": relocation}, Path(output_dir)/"predictions.pt.gz")
    return evaluated


def reevaluate(predictions_path, output_dir, protocol: MetricProtocol):
    _check_output_available(output_dir)
    saved = load_torch(predictions_path)
    geometry = Geometry.from_state_dict(saved["geometry"])
    result = score_predictions(saved["predictions"], geometry, protocol)
    result["source_predictions"] = str(Path(predictions_path).resolve())
    save_json(result, Path(output_dir)/"evaluation.json")
    return result
