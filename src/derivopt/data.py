"""Lazy, schema-checked PDEBench trajectory I/O and training-only statistics.

Supports tensor, grouped-trajectory and vector layouts with bounded time-window
reads and validation of coordinates, time axes and simulation metadata.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


FAMILY_CHANNELS = {
    "advection": ("u",),
    "burgers": ("u",),
    "diffusion_sorption": ("u",),
    "diffusion_reaction": ("u", "v"),
    "radial_dam_break": ("h",),
    "incompressible_ns": ("Vx", "Vy"),
    "compressible_ns": ("density", "Vx", "Vy", "pressure"),
}
FAMILY_NDIM = {key: (1 if key in ("advection", "burgers", "diffusion_sorption") else 2)
               for key in FAMILY_CHANNELS}


class DataFormatError(ValueError):
    """An input file does not satisfy a supported, declared data schema."""


@dataclass(frozen=True)
class TrajectorySample:
    states: torch.Tensor  # [T, C, X] or [T, C, X, Y], physical units
    coords: tuple[torch.Tensor, ...]
    times: torch.Tensor
    family: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _Record:
    path: Path
    key: str | int
    layout: str
    steps: int
    spatial_shape: tuple[int, ...]


def _require(container: h5py.Group, key: str) -> h5py.Dataset:
    if key not in container or not isinstance(container[key], h5py.Dataset):
        raise DataFormatError(f"{container.file.filename}:{container.name}: missing dataset {key!r}")
    return container[key]


def _axis(value: np.ndarray, length: int, label: str) -> torch.Tensor:
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (length,) or not np.isfinite(value).all():
        raise DataFormatError(f"{label} must contain {length} finite coordinates, got {value.shape}")
    if length > 1 and not np.all(np.diff(value) > 0):
        raise DataFormatError(f"{label} coordinates must be strictly increasing")
    return torch.from_numpy(value.copy())


def fixed_time_step(times, *, source_dtype=None, return_tolerance=False):
    """Infer a positive fixed step, allowing only timestamp storage roundoff.

    Check the full affine time grid rather than differences of rounded float32
    timestamps. ``source_dtype`` preserves that precision after the reader has
    converted coordinates to float64. A time origin does not change the step.
    ``return_tolerance`` also returns the endpoint-roundoff uncertainty in the
    inferred step, for comparisons between separately stored time axes.
    """
    values = torch.as_tensor(times).detach().cpu()
    dtype = np.dtype(source_dtype) if source_dtype is not None else values.numpy().dtype
    values = values.to(torch.float64)
    if values.ndim != 1 or values.numel() < 2 or not torch.isfinite(values).all():
        raise DataFormatError("A fixed time step requires at least two finite time coordinates")
    if not bool((values.diff() > 0).all()):
        raise DataFormatError("Time coordinates must be strictly increasing")
    step = float((values[-1] - values[0]) / (values.numel() - 1))
    # Each stored timestamp is within half an ULP of its unrounded value.
    # Interpolated endpoint error is a convex combination, whereas uncertainty
    # in dt divides the endpoint difference error by the number of transitions.
    # Do not reuse an all-grid tolerance as a per-step uncertainty.
    if np.issubdtype(dtype, np.floating):
        half_ulp = torch.from_numpy(np.abs(np.spacing(values.numpy().astype(dtype))).astype(np.float64)) * .5
    else:
        half_ulp = torch.zeros_like(values)
    weights = torch.arange(values.numel(), dtype=torch.float64) / (values.numel() - 1)
    endpoint_error = (1 - weights) * half_ulp[0] + weights * half_ulp[-1]
    tolerance = (half_ulp + endpoint_error).clamp_min(step * 1e-6)
    if float(tolerance.max()) >= step * 0.01:
        raise DataFormatError("Time coordinate precision is insufficient to establish a fixed time step")
    expected = values[0] + torch.arange(values.numel(), dtype=torch.float64) * step
    if not bool(((values - expected).abs() <= tolerance).all()):
        raise DataFormatError("Nonuniform time coordinates are unsupported; a fixed time step is required")
    uncertainty = max(step * 1e-6, float(half_ulp[0] + half_ulp[-1]) / (values.numel() - 1))
    return (float(step), float(uncertainty)) if return_tolerance else float(step)


def _attrs(group: h5py.Group) -> dict[str, Any]:
    result = {}
    for key, value in group.attrs.items():
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        elif isinstance(value, np.ndarray):
            value = value.tolist()
        elif isinstance(value, np.generic):
            value = value.item()
        result[key] = value
    return result


def _align_reaction_coordinates(states, coords, metadata):
    """Align a small published-grid phase offset to its declared Neumann domain.

    PDEBench's released DR coordinates are a stride-four slice of a finer mesh,
    offset by one eighth of a stored cell. Interpolate the *values*, independently
    on each axis, onto the declared-domain cell centres. Even reflection at the
    physical faces supplies the Neumann extension; the raw HDF5 is unchanged.
    """
    simulation = metadata.get("simulation_config", {})
    names = (("x_left", "x_right"), ("y_bottom", "y_top"))
    required = [name for pair in names for name in pair]
    if not all(name in simulation for name in required):
        if metadata.get("file_attributes", {}).get("derivopt_fixture") is True:
            return states, coords
        raise DataFormatError("Diffusion-Reaction requires declared x/y physical bounds to establish cell-centre alignment")
    original_axes = {name: axis.tolist() for name, axis in zip("xy", coords)}
    target_axes, shifts, domains = [], [], []
    result = states
    for dimension, (axis, (left_name, right_name)) in enumerate(zip(coords, names), start=2):
        values = axis.numpy()
        left, right = float(simulation[left_name]), float(simulation[right_name])
        if not math.isfinite(left) or not math.isfinite(right) or right <= left:
            raise DataFormatError("Diffusion-Reaction physical domain bounds must be finite and increasing")
        size, spacing = len(values), (right-left)/len(values)
        tolerance = max(1e-10, spacing*1e-5)
        if not np.allclose(np.diff(values), spacing, rtol=0, atol=tolerance):
            raise DataFormatError("Diffusion-Reaction requires a complete uniform grid with N*dx equal to its declared domain length; cropped or unknown grids are unsupported")
        expected = left + (np.arange(size, dtype=np.float64)+.5)*spacing
        shift = float(np.mean(values-expected))
        if (not np.allclose(values-expected, shift, rtol=0, atol=tolerance)
                or abs(shift) > .25*spacing+tolerance
                or values[0] <= left or values[-1] >= right):
            raise DataFormatError("Diffusion-Reaction grid has an unsupported coordinate phase or lies outside its declared physical domain")
        if abs(shift) > tolerance:
            extended = np.concatenate(([2*left-values[0]], values, [2*right-values[-1]]))
            upper = np.searchsorted(extended, expected, side="right").clip(1, size+1)
            lower = upper-1
            weight = (expected-extended[lower])/(extended[upper]-extended[lower])
            weight_shape = [1]*result.ndim
            weight_shape[dimension] = size
            lower_values = np.take(result, (lower-1).clip(0, size-1), axis=dimension)
            upper_values = np.take(result, (upper-1).clip(0, size-1), axis=dimension)
            result = (lower_values*(1-weight.reshape(weight_shape))
                      + upper_values*weight.reshape(weight_shape)).astype(states.dtype, copy=False)
        target_axes.append(torch.from_numpy(expected))
        shifts.append(shift/spacing)
        domains.append([left, right])
    metadata["source_axes"] = original_axes
    metadata["coordinate_preprocessing"] = {
        "method": "separable_linear_interpolation_with_neumann_even_reflection",
        "applied": any(abs(value) > 1e-5 for value in shifts),
        "source_phase_in_stored_cells": shifts, "declared_domains": domains,
        "target_layout": "declared_domain_uniform_cell_centres",
        "source_spatial_shape": list(states.shape[2:]), "target_spatial_shape": list(result.shape[2:]),
        "maximum_supported_phase_cells": .25,
    }
    return result, tuple(target_axes)


def _cfd_boundary_from_source(path, attributes):
    """Recognize only upstream documented CNS filename variants, never keywords."""
    name = path.name
    if "derivopt_source_subset" in attributes:
        try:
            subset = json.loads(attributes["derivopt_source_subset"])
            name = subset.get("source_filename", name)
        except (TypeError, ValueError, AttributeError):
            raise DataFormatError("Invalid source-subset filename provenance") from None
    if not isinstance(name, str) or Path(name).name != name:
        raise DataFormatError("Source filename provenance must be a single filename")
    match = re.fullmatch(r"2D_CFD_(?:Rand|Turb)_M[\deE.+-]+_Eta[\deE.+-]+_Zeta[\deE.+-]+_(periodic|trans)(?:_\d+)?_Train\.hdf5", name)
    if match:
        return match.group(1), name
    if name == "2D_shock.hdf5":
        return "trans", name
    return None, name


class PDEBenchDataset(Dataset):
    """Load one trajectory at a time; constructor only indexes HDF5 metadata.

    A dataset is intentionally not called "train" until ``subset(..., split="train")``
    or ``split_trajectories`` is used. Splits operate before time-window expansion.
    ``spatial_stride`` is an explicit I/O subset, NOT the paper's anti-alias/coarse
    operator; use 1 for scientific runs, and let the codec do coarsening.

    ``reaction_axis_order="yx"`` follows the actual upstream diffusion-reaction
    writer. Use "xy" only for an explicitly converted file, and record that choice.
    """

    def __init__(
        self,
        paths: str | Path | Sequence[str | Path],
        family: str,
        *,
        indices: Sequence[int] | None = None,
        spatial_stride: int = 1,
        time_stride: int = 1,
        reaction_axis_order: str = "yx",
    ):
        if family not in FAMILY_CHANNELS:
            raise ValueError(f"Unknown family {family!r}; choose {tuple(FAMILY_CHANNELS)}")
        if not isinstance(spatial_stride, int) or spatial_stride < 1:
            raise ValueError("spatial_stride must be a positive integer")
        if not isinstance(time_stride, int) or time_stride < 1:
            raise ValueError("time_stride must be a positive integer")
        if reaction_axis_order not in ("xy", "yx"):
            raise ValueError("reaction_axis_order must be 'xy' or 'yx'")
        if isinstance(paths, (str, Path)):
            paths = [paths]
        self.paths = tuple(Path(p).expanduser().resolve() for p in paths)
        if not self.paths or len(set(self.paths)) != len(self.paths):
            raise ValueError("Provide at least one distinct HDF5 file (no duplicates)")
        self.family = family
        self.spatial_stride = spatial_stride
        self.time_stride = time_stride
        self.reaction_axis_order = reaction_axis_order
        self.split = "unspecified"
        records: list[_Record] = []
        self._file_indices = {}
        for path in self.paths:
            if not path.is_file():
                raise FileNotFoundError(f"PDEBench file does not exist: {path}")
            with h5py.File(path, "r") as handle:
                file_records = self._index(handle, path)
                self._file_indices.update({(record.path, record.key): index for index, record in enumerate(file_records)})
                records.extend(file_records)
        if not records:
            raise DataFormatError("No trajectories found")
        if indices is not None:
            ids = [int(i) for i in indices]
            if len(set(ids)) != len(ids) or any(i < 0 or i >= len(records) for i in ids):
                raise ValueError("indices must be distinct valid trajectory indices")
            records = [records[i] for i in ids]
        self._records = tuple(records)

    def _index(self, handle: h5py.File, path: Path) -> list[_Record]:
        ndim = FAMILY_NDIM[self.family]
        if self.family in ("advection", "burgers"):
            values = _require(handle, "tensor")
            if values.ndim != 3:
                raise DataFormatError("1D scalar tensor must have shape [N,T,X]")
            _require(handle, "x-coordinate")
            _require(handle, "t-coordinate")
            return [_Record(path, i, "tensor", values.shape[1], values.shape[2:])
                    for i in range(values.shape[0])]
        if self.family == "compressible_ns":
            arrays = [_require(handle, key) for key in FAMILY_CHANNELS[self.family]]
            shape = arrays[0].shape
            if len(shape) not in (3, 4) or any(a.shape != shape for a in arrays):
                raise DataFormatError("2D CFD fields must share shape [N,T,X,Y] or a single [T,X,Y] trajectory")
            for key in ("x-coordinate", "y-coordinate", "t-coordinate"):
                _require(handle, key)
            if len(shape) == 3:
                return [_Record(path, 0, "cfd_single", shape[0], shape[1:])]
            return [_Record(path, i, "cfd", shape[1], shape[2:]) for i in range(shape[0])]
        if self.family == "incompressible_ns":
            if "velocity" not in handle:
                raise DataFormatError("INS requires primitive velocity[N,T,X,Y,2]; a vorticity-only "
                                      "file is not interchangeable with the paper's velocity state")
            values = _require(handle, "velocity")
            if values.ndim != 5 or values.shape[-1] != 2:
                raise DataFormatError("INS velocity must have shape [N,T,X,Y,2]")
            t = _require(handle, "t")
            if t.shape != values.shape[:2]:
                raise DataFormatError("INS t must have shape [N,T] matching velocity")
            latest = handle.attrs.get("latestIndex")
            if latest is not None and int(latest) != values.shape[1] - 1:
                # Extracted time windows retain the source latestIndex. Validate
                # it against the recorded original frame count and subset bounds.
                try:
                    subset = json.loads(handle.attrs["derivopt_source_subset"])
                    source_steps = subset["source_frame_count"]
                    start, stop = subset["source_frame_start"], subset["source_frame_stop"]
                    ids = subset["source_trajectory_indices"]
                    valid = (subset.get("format_version") == 1
                             and all(type(value) is int for value in (source_steps, start, stop))
                             and subset.get("source_frame_stride") == 1
                             and subset.get("source_latestIndex") == int(latest) == source_steps - 1
                             and 0 <= start < stop <= source_steps
                             and stop - start == values.shape[1]
                             and len(ids) == values.shape[0]
                             and all(type(value) is int and value >= 0 for value in ids)
                             and len(set(ids)) == len(ids))
                except (KeyError, TypeError, ValueError):
                    valid = False
                if not valid:
                    raise DataFormatError("INS file contains an incomplete simulation (latestIndex), or its source-subset provenance is invalid")
            if "config" not in handle.attrs and not all(k in handle for k in ("x-coordinate", "y-coordinate")):
                raise DataFormatError("INS file needs upstream config metadata or explicit coordinates; "
                                      "physical domain cannot be guessed")
            return [_Record(path, i, "velocity", values.shape[1], values.shape[2:-1])
                    for i in range(values.shape[0])]
        groups = [key for key in sorted(handle) if isinstance(handle[key], h5py.Group) and "data" in handle[key]]
        if not groups:
            raise DataFormatError(f"{self.family} requires per-trajectory groups with data and grid")
        result = []
        for key in groups:
            group = handle[key]
            values = _require(group, "data")
            if values.ndim != ndim + 2 or values.shape[-1] != len(FAMILY_CHANNELS[self.family]):
                raise DataFormatError(f"{key}/data must be [T,*space,C] with {len(FAMILY_CHANNELS[self.family])} channels")
            for axis_name in "xy"[:ndim]:
                _require(group, f"grid/{axis_name}")
            _require(group, "grid/t")
            shape = values.shape[1:-1]
            if self.family == "diffusion_reaction" and self.reaction_axis_order == "yx":
                shape = shape[::-1]
            result.append(_Record(path, key, "grouped", values.shape[0], shape))
        return result

    def __len__(self) -> int:
        return len(self._records)

    @property
    def trajectory_ids(self) -> tuple[str, ...]:
        return tuple(f"{r.path}::{r.key}" for r in self._records)

    def trajectory_length(self, index: int) -> int:
        return len(range(0, self._records[index].steps, self.time_stride))

    def subset(self, indices: Sequence[int], *, split: str) -> "PDEBenchDataset":
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train, val, or test")
        ids = [int(i) for i in indices]
        if len(set(ids)) != len(ids) or any(i < 0 or i >= len(self) for i in ids):
            raise ValueError("subset indices must be distinct and in range")
        # Do not reopen/reindex the full collection for each view.
        view = object.__new__(PDEBenchDataset)
        view.__dict__ = self.__dict__.copy()
        view._records = tuple(self._records[i] for i in ids)
        view.split = split
        return view

    def __getitem__(self, index: int) -> TrajectorySample:
        return self.read_window(index)

    def read_window(self, index: int, start: int = 0, stop: int | None = None) -> TrajectorySample:
        """Read only ``[start:stop]`` states from a trajectory's sampled time axis.

        ``start`` and ``stop`` index the view after ``time_stride``, matching
        ``dataset[index].states[start:stop]``. At least two frames are required to
        retain a verifiable time step. Coordinates and the complete small native
        time axis are validated; unrequested state frames are never materialized.
        """
        record = self._records[index]
        length = self.trajectory_length(index)
        stop = length if stop is None else stop
        if (type(start) is not int or type(stop) is not int
                or not 0 <= start < stop <= length or stop - start < 2):
            raise ValueError("Time window must contain at least two frames within the sampled trajectory")
        ts = slice(start*self.time_stride, min(stop*self.time_stride, record.steps), self.time_stride)
        ss = slice(None, None, self.spatial_stride)
        spatial = (ss,) * FAMILY_NDIM[self.family]
        metadata: dict[str, Any] = {"path": str(record.path), "trajectory": record.key,
                                   "trajectory_id": self.trajectory_ids[index],
                                   "layout": record.layout, "split": self.split,
                                   "channel_names": FAMILY_CHANNELS[self.family],
                                   "spatial_stride": self.spatial_stride,
                                   "time_stride": self.time_stride,
                                   "time_window": {"start": start, "stop": stop,
                                                   "stored_frame_start": ts.start,
                                                   "stored_frame_stop": ts.stop,
                                                   "stored_frame_step": ts.step}}
        with h5py.File(record.path, "r") as handle:
            metadata["file_attributes"] = _attrs(handle)
            if record.layout == "grouped":
                group = handle[str(record.key)]
                raw = np.asarray(group["data"][(ts, *spatial, slice(None))], dtype=np.float32)
                if self.family == "diffusion_reaction" and self.reaction_axis_order == "yx":
                    raw = raw.swapaxes(1, 2)
                states = np.moveaxis(raw, -1, 1)
                axes = [np.asarray(group[f"grid/{name}"])[ss] for name in "xy"[:len(spatial)]]
                times = np.asarray(group["grid/t"])
                metadata["trajectory_attributes"] = _attrs(group)
                if "config" in group.attrs:
                    import yaml
                    config = yaml.safe_load(group.attrs["config"])
                    if not isinstance(config, dict):
                        raise DataFormatError("Grouped trajectory config must be a YAML mapping")
                    metadata["simulation_config"] = config.get("sim", config)
                metadata["source_axis_order"] = self.reaction_axis_order if self.family == "diffusion_reaction" else "xy"[:len(spatial)]
            elif record.layout == "tensor":
                states = np.asarray(handle["tensor"][(record.key, ts, *spatial)], dtype=np.float32)[:, None]
                axes = [np.asarray(handle["x-coordinate"])[ss]]
                times = np.asarray(handle["t-coordinate"])
            elif record.layout in ("cfd", "cfd_single"):
                selection = (ts, *spatial) if record.layout == "cfd_single" else (record.key, ts, *spatial)
                states = np.stack([np.asarray(handle[key][selection], dtype=np.float32)
                                   for key in FAMILY_CHANNELS[self.family]], axis=1)
                axes = [np.asarray(handle[f"{name}-coordinate"])[ss] for name in "xy"]
                times = np.asarray(handle["t-coordinate"])
                boundary, source_name = _cfd_boundary_from_source(record.path, metadata["file_attributes"])
                metadata["source_filename"] = source_name
                if boundary is not None:
                    metadata["boundary"] = boundary
                    metadata["boundary_provenance"] = "official_pdebench_source_filename_convention"
            else:
                raw = np.asarray(handle["velocity"][(record.key, ts, *spatial, slice(None))], dtype=np.float32)
                states = np.moveaxis(raw, -1, 1)
                times = np.asarray(handle["t"][record.key])
                if all(f"{name}-coordinate" in handle for name in "xy"):
                    axes = [np.asarray(handle[f"{name}-coordinate"])[ss] for name in "xy"]
                    metadata["coordinate_source"] = "explicit"
                else:
                    # The upstream PhiFlow writer uses Box[0:nx,0:ny] and samples
                    # the staggered velocity onto a centered grid before saving.
                    import yaml
                    config = yaml.safe_load(handle.attrs["config"])
                    if not isinstance(config, dict) or tuple(config.get("grid_size", ())) != record.spatial_shape:
                        raise DataFormatError("INS config.grid_size must match velocity spatial shape")
                    axes = [(np.arange(n, dtype=np.float64) + 0.5)[ss] for n in record.spatial_shape]
                    metadata["coordinate_source"] = "upstream_phiflow_box_cell_centers"
                    metadata["simulation_config"] = config
            # Some NLE files store one additional unused output-time coordinate.
            # Only the first T values have corresponding states, never interpolate.
            if record.layout in ("tensor", "cfd", "cfd_single") and times.shape == (record.steps + 1,):
                times = times[:record.steps]
                metadata["unused_trailing_time_coordinate"] = True
            if times.shape != (record.steps,):
                raise DataFormatError(f"Time coordinate shape {times.shape} does not match {record.steps} states")
        if states.shape[0] < 2 or any(size < 2 for size in states.shape[2:]):
            raise DataFormatError("A rollout trajectory needs at least two times and two points per spatial axis")
        if not np.isfinite(states).all():
            raise DataFormatError(f"Non-finite states in {record.path} trajectory {record.key}")
        coords = tuple(_axis(axis, states.shape[2 + i], f"axis {i}") for i, axis in enumerate(axes))
        if self.family == "diffusion_reaction":
            states, coords = _align_reaction_coordinates(states, coords, metadata)
        # Validate the native grid before optional temporal subsampling as well.
        fixed_time_step(times)
        time_tensor = _axis(times[ts], states.shape[0], "time")
        metadata["time_coordinate_dtype"] = str(times.dtype)
        metadata["time_step"] = fixed_time_step(time_tensor, source_dtype=times.dtype)
        return TrajectorySample(torch.from_numpy(np.ascontiguousarray(states)), coords,
                                time_tensor, self.family, metadata)


def split_trajectories(dataset: PDEBenchDataset, *, train_fraction: float = 0.7,
                       val_fraction: float = 0.15, seed: int = 0) -> dict[str, PDEBenchDataset]:
    """Deterministic trajectory-level shuffle with three nonempty, disjoint splits.

    Tiny sets reserve at least one validation and one test trajectory. These
    defaults can be overridden or replaced by explicit pre-existing partitions.
    """
    if dataset.split != "unspecified":
        raise ValueError("Cannot randomly repartition an already labeled split; provide the complete unlabeled dataset")
    if not 0 < train_fraction < 1 or not 0 < val_fraction < 1 or train_fraction + val_fraction >= 1:
        raise ValueError("Split fractions must be positive and leave a nonzero test fraction")
    if len(dataset) < 3:
        raise ValueError("At least three trajectories are needed for train/val/test splits")
    order = np.random.default_rng(seed).permutation(len(dataset)).tolist()
    n_train = min(max(1, int(len(dataset) * train_fraction)), len(dataset) - 2)
    n_val = min(max(1, int(len(dataset) * val_fraction)), len(dataset) - n_train - 1)
    return {"train": dataset.subset(order[:n_train], split="train"),
            "val": dataset.subset(order[n_train:n_train + n_val], split="val"),
            "test": dataset.subset(order[n_train + n_val:], split="test")}


def partition_trajectories(dataset: PDEBenchDataset, manifest) -> dict[str, PDEBenchDataset]:
    """Apply an exhaustive, disjoint file-local trajectory partition manifest.

    JSON schema: ``{"version": 1, "train": [{"file": "train.h5",
    "indices": [0, 1]}], "val": [...], "test": [...]}``. Indices refer to
    zero-based trajectory order within each complete source file, not to the
    concatenated dataset. Relative files resolve beside the JSON manifest, or
    against the current directory for an in-memory dict. No file is opened here.
    """
    if dataset.split != "unspecified":
        raise ValueError("Cannot repartition an already labeled split; provide the complete unlabeled dataset")
    if isinstance(manifest, (str, Path)):
        path = Path(manifest).expanduser().resolve()
        base, manifest = path.parent, json.loads(path.read_text())
    else:
        base = Path.cwd()
    names = ("train", "val", "test")
    if (not isinstance(manifest, dict) or set(manifest) != {"version", *names}
            or type(manifest["version"]) is not int or manifest["version"] != 1):
        raise ValueError("Partition manifest requires version 1 and exactly train, val, test lists")
    positions = {(record.path, dataset._file_indices[(record.path, record.key)]): index
                 for index, record in enumerate(dataset._records)}
    seen, selected = set(), {}
    for name in names:
        entries = manifest[name]
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"Partition {name} must be a nonempty list")
        indices = []
        for entry in entries:
            if (not isinstance(entry, dict) or set(entry) != {"file", "indices"}
                    or not isinstance(entry["file"], str) or not entry["file"].strip()
                    or not isinstance(entry["indices"], list) or not entry["indices"]):
                raise ValueError("Each partition entry needs a file string and nonempty indices list")
            source = (base / Path(entry["file"]).expanduser()).resolve()
            for index in entry["indices"]:
                if type(index) is not int or index < 0:
                    raise ValueError("Partition indices must be nonnegative integers")
                key = (source, index)
                if key not in positions:
                    raise ValueError(f"Unknown partition file/trajectory index: {source}::{index}")
                if key in seen:
                    raise ValueError(f"Overlapping or duplicate partition trajectory: {source}::{index}")
                seen.add(key)
                indices.append(positions[key])
        selected[name] = dataset.subset(indices, split=name)
    if seen != set(positions):
        raise ValueError(f"Partition manifest omits {len(set(positions)-seen)} dataset trajectories")
    return selected


class TrajectoryWindows(Dataset):
    """Contiguous windows built after splitting, never across trajectory boundaries."""

    def __init__(self, dataset: PDEBenchDataset, *, history: int = 1, horizon: int = 1, stride: int = 1):
        if any(not isinstance(v, int) or v < 1 for v in (history, horizon, stride)):
            raise ValueError("history, horizon and stride must be positive integers")
        self.dataset, self.history, self.horizon = dataset, history, horizon
        self.split = dataset.split
        self._windows = [(i, t) for i in range(len(dataset))
                         for t in range(0, dataset.trajectory_length(i) - history - horizon + 1, stride)]
        if not self._windows:
            raise ValueError("No trajectory is long enough for the requested history+horizon")

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        trajectory, start = self._windows[index]
        stop = start + self.history + self.horizon
        # Geometry-checking views expose the same lazy API. Keep a fallback for
        # external custom Dataset wrappers without changing their public contract.
        if hasattr(self.dataset, "read_window"):
            sample = self.dataset.read_window(trajectory, start, stop)
            offset = 0
        else:
            sample = self.dataset[trajectory]
            offset = start
        return {"inputs": sample.states[offset:offset + self.history],
                "targets": sample.states[offset + self.history:offset + self.history + self.horizon],
                "times": sample.times[offset:offset + self.history + self.horizon], "coords": sample.coords,
                "trajectory_id": sample.metadata["trajectory_id"], "start": start}


class ChannelStandardizer:
    """Population channel statistics fitted exclusively to a train trajectory view."""

    def __init__(self, mean: torch.Tensor, scale: torch.Tensor, *, training_ids: Sequence[str] = (), count: int = 0):
        self.mean = torch.as_tensor(mean, dtype=torch.float64).detach().cpu().flatten()
        self.scale = torch.as_tensor(scale, dtype=torch.float64).detach().cpu().flatten()
        if self.mean.shape != self.scale.shape or not self.mean.numel():
            raise ValueError("mean and scale must have the same nonempty channel shape")
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.scale).all() or (self.scale <= 0).any():
            raise ValueError("Statistics must be finite, with positive scales")
        self.training_ids, self.count = tuple(training_ids), int(count)

    @classmethod
    def fit(cls, dataset: PDEBenchDataset, *, minimum_scale: float = 1e-8) -> "ChannelStandardizer":
        if dataset.split != "train":
            raise ValueError("Standardizer.fit accepts only a labeled train split; validation/test leakage is forbidden")
        if not len(dataset) or minimum_scale <= 0:
            raise ValueError("Need a nonempty train split and positive minimum_scale")
        total, mean, m2 = 0, None, None
        # Parallel-variance combination avoids sum-of-squares cancellation.
        for sample in dataset:
            values = sample.states.movedim(1, 0).reshape(sample.states.shape[1], -1).double()
            n, current = values.shape[1], values.mean(dim=1)
            current_m2 = ((values - current[:, None]) ** 2).sum(dim=1)
            if mean is None:
                total, mean, m2 = n, current, current_m2
            else:
                delta = current - mean
                m2 = m2 + current_m2 + delta.square() * total * n / (total + n)
                mean = mean + delta * n / (total + n)
                total += n
        assert mean is not None and m2 is not None
        return cls(mean, (m2 / total).sqrt().clamp_min(minimum_scale),
                   training_ids=dataset.trajectory_ids, count=total)

    def _stats(self, values: torch.Tensor, channel_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        channel_dim %= values.ndim
        if values.shape[channel_dim] != self.mean.numel():
            raise ValueError("Input channel count does not match fitted standardizer")
        shape = [1] * values.ndim
        shape[channel_dim] = self.mean.numel()
        return self.mean.to(values).reshape(shape), self.scale.to(values).reshape(shape)

    def transform(self, values: torch.Tensor, *, channel_dim: int = 1) -> torch.Tensor:
        mean, scale = self._stats(values, channel_dim)
        return (values - mean) / scale

    def inverse(self, values: torch.Tensor, *, channel_dim: int = 1) -> torch.Tensor:
        mean, scale = self._stats(values, channel_dim)
        return values * scale + mean

    def state_dict(self) -> dict[str, Any]:
        return {"version": 1, "mean": self.mean.tolist(), "scale": self.scale.tolist(),
                "training_ids": list(self.training_ids), "count": self.count}

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "ChannelStandardizer":
        if state.get("version") != 1:
            raise ValueError("Unsupported standardizer schema version")
        return cls(state["mean"], state["scale"], training_ids=state["training_ids"], count=state["count"])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.state_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ChannelStandardizer":
        return cls.from_state_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _sorption_fixture_field(x: np.ndarray, time: float, phase: float, amplitude: float,
                            *, diffusion: float = 5e-4, boundary_value: float = 1.0) -> np.ndarray:
    """Manufactured field with u(0)=sol and u(1)+D*u_x(1)=0.

    This satisfies the continuous Dirichlet/Robin boundary convention used for
    lifting, not the nonlinear diffusion-sorption evolution equation. The
    residual and its first derivative vanish at the right boundary; the residual
    itself vanishes at the left boundary. It contains several spatial frequencies.
    """
    x = np.asarray(x, dtype=np.float64)
    lift = boundary_value * (1 + diffusion - x) / (1 + diffusion)
    residual = (amplitude * np.exp(-time) * x * (1 - x)**2
                * (np.sin(np.pi * x + phase) + .3 * np.cos(3 * np.pi * x - .2 * time)))
    return lift + residual


def write_fixture(path: str | Path, family: str, *, trajectories: int = 4, steps: int = 5,
                  spatial_size: int | Sequence[int] = 12, seed: int = 0) -> Path:
    """Write manufactured test fields in a PDEBench-compatible HDF5 layout.

    Existing files are preserved. The fields are analytic test functions rather
    than numerical PDE solutions; the loader preserves their fixture metadata.
    """
    if family not in FAMILY_CHANNELS:
        raise ValueError(f"Unknown family {family!r}")
    ndim = FAMILY_NDIM[family]
    shape = (spatial_size,) * ndim if isinstance(spatial_size, int) else tuple(spatial_size)
    if len(shape) != ndim or min(shape) < 4 or trajectories < 1 or steps < 2:
        raise ValueError("Fixture needs correct dimensionality, >=4 spatial cells, >=1 trajectory and >=2 times")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    axes = tuple((np.arange(n) + 0.5) / n for n in shape)
    mesh = np.meshgrid(*axes, indexing="ij")
    times = np.linspace(0, 0.4, steps, dtype=np.float32)
    rng = np.random.default_rng(seed)
    data = []
    for index in range(trajectories):
        phase = float(rng.uniform(-0.2, 0.2))
        frames = []
        for time in times:
            x = 2 * np.pi * mesh[0]
            if ndim == 1:
                if family == "diffusion_sorption":
                    fields = [_sorption_fixture_field(mesh[0], float(time), phase, .2 + .02 * index)]
                else:
                    fields = [np.sin(x - time + phase) + 0.2 * np.cos(3 * x + 0.3 * time) + 0.05 * index]
            else:
                y = 2 * np.pi * mesh[1]
                a, b = x + phase - time, y + 0.2 * time
                if family == "incompressible_ns":
                    # Both components derive from a streamfunction; no vorticity substitution.
                    aspect = shape[1] / shape[0]
                    fields = [np.sin(a) * np.cos(b) + 0.15 * np.sin(3 * a) * np.cos(2 * b),
                              aspect * (-np.cos(a) * np.sin(b) - 0.225 * np.cos(3 * a) * np.sin(2 * b))]
                elif family == "compressible_ns":
                    fields = [1.5 + 0.2 * np.sin(a) * np.cos(b), np.sin(a) + 0.2 * np.cos(2 * b),
                              np.cos(b) + 0.15 * np.sin(3 * a), 2.0 + 0.3 * np.cos(a + b)]
                elif family == "diffusion_reaction":
                    fields = [np.cos(x) * np.cos(y) * np.exp(-time) + 0.15 * np.cos(3*x),
                              0.4 * np.cos(2*x) * np.cos(y) * np.exp(-0.5*time) + 0.1 * index]
                else:
                    fields = [1.5 + 0.3 * np.cos(x) * np.cos(y) * np.exp(-time) + 0.08 * np.cos(3*x)]
            frames.append(np.stack(fields, axis=-1))
        data.append(np.stack(frames).astype(np.float32))
    values = np.stack(data)  # N,T,*space,C
    with h5py.File(path, "x") as handle:
        handle.attrs["derivopt_fixture"] = True
        handle.attrs["provenance"] = "manufactured schema-compatible fixture; not PDEBench simulation results"
        if family in ("advection", "burgers"):
            handle.create_dataset("tensor", data=values[..., 0])
            handle.create_dataset("x-coordinate", data=axes[0])
            handle.create_dataset("t-coordinate", data=times)
        elif family == "compressible_ns":
            for i, key in enumerate(FAMILY_CHANNELS[family]):
                handle.create_dataset(key, data=values[..., i])
            for axis, name in zip(axes, "xy"):
                handle.create_dataset(f"{name}-coordinate", data=axis)
            handle.create_dataset("t-coordinate", data=times)
        elif family == "incompressible_ns":
            handle.create_dataset("velocity", data=values)
            handle.create_dataset("t", data=np.broadcast_to(times, (trajectories, steps)))
            handle.create_dataset("particles", data=np.ones((*values.shape[:-1], 1), dtype=np.float32))
            handle.create_dataset("force", data=np.zeros((trajectories, *shape, 2), dtype=np.float32))
            handle.attrs["latestIndex"] = steps - 1
            handle.attrs["config"] = json.dumps({"grid_size": list(shape), "n_batch": trajectories,
                                                "n_steps": steps, "velocity_extrapolation": "PERIODIC"})
        else:
            for i, trajectory in enumerate(values):
                group = handle.create_group(f"{i:04d}")
                # Match the upstream RD writer's [T,Y,X,C] reshape exactly.
                source = trajectory.swapaxes(1, 2) if family == "diffusion_reaction" else trajectory
                group.create_dataset("data", data=source)
                for axis, name in zip(axes, "xy"):
                    group.create_dataset(f"grid/{name}", data=axis)
                group.create_dataset("grid/t", data=times)
                if family == "diffusion_sorption":
                    group.attrs["config"] = json.dumps({"sim": {"D": 5e-4, "sol": 1.0,
                                                                "x_left": 0.0, "x_right": 1.0,
                                                                "xdim": shape[0]}})
                    group.attrs["manufactured_boundary"] = "u(0)=sol; u(1)+D*u_x(1)=0; not a PDE solution"
    return path
