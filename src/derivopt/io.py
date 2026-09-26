"""Portable results and tensor-only checkpoints."""
import gzip
import copy
import json
import math
import os
from pathlib import Path
import tempfile

import torch


def dataset_identity(dataset):
    """Metadata fingerprint for detecting accidental dataset changes on reuse.

    Records absolute paths, byte sizes, nanosecond modification times, sorted
    trajectory IDs and reader settings without reading the full file contents.
    """
    files = []
    for source in sorted(dataset.paths):
        stat = source.stat()
        files.append({"path": str(source.resolve()), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return {
        "policy": "absolute_paths_sizes_mtime_ns_sorted_trajectory_ids_reader_settings",
        "files": files, "trajectory_ids": sorted(dataset.trajectory_ids),
        "family": dataset.family, "spatial_stride": dataset.spatial_stride,
        "time_stride": dataset.time_stride, "reaction_axis_order": dataset.reaction_axis_order,
    }


def remap_trajectory_ids(identities, data_map):
    """Replace only explicit file prefixes; preserve every file-local key/order."""
    sources = sorted(data_map, key=len, reverse=True)
    result = []
    for identity in identities:
        source = next((path for path in sources if identity.startswith(path + "::")), None)
        result.append(identity if source is None else data_map[source] + identity[len(source):])
    return result


def validate_dataset_identity(expected, current, *, data_map=None):
    """Validate strict identity, or a user-declared relocation of equal-size files.

    Explicit mappings waive mtime only for their named files. This is not a
    content-hash check: the caller declares that each destination is a copy of
    its source. Reader settings, sizes and complete trajectory IDs still match.
    The runner additionally checks indexed shapes against checkpoint geometry.
    """
    if expected is None:
        raise ValueError("Prepared/checkpoint data identity is missing; prepare again with source identity recording")
    mapping = {}
    if data_map is not None:
        if isinstance(data_map, (str, Path)):
            with Path(data_map).open(encoding="utf-8") as stream:
                data_map = json.load(stream)
        if not isinstance(data_map, dict) or not data_map:
            raise ValueError("data_map must be a nonempty JSON object mapping old absolute paths to new absolute paths")
        expected_paths = {row["path"] for row in expected["files"]}
        current_paths = {row["path"] for row in current["files"]}
        for source, destination in data_map.items():
            if (not isinstance(source, str) or not isinstance(destination, str)
                    or not Path(source).is_absolute() or not Path(destination).is_absolute()):
                raise ValueError("data_map keys and values must be absolute file paths")
            destination = str(Path(destination).resolve())
            if source not in expected_paths or destination not in current_paths:
                raise ValueError("data_map must name checkpoint source files and supplied destination data files exactly")
            if source == destination:
                raise ValueError("data_map requires a changed path; it cannot waive mtime for a file at its original path")
            mapping[source] = destination
        relocated_paths = [mapping.get(path, path) for path in expected_paths]
        if len(set(relocated_paths)) != len(relocated_paths) or set(relocated_paths) != current_paths:
            raise ValueError("data_map must give a one-to-one mapping onto the supplied files, including unchanged files")
    relocated = copy.deepcopy(expected)
    current_files = {row["path"]: row for row in current["files"]}
    for row in relocated["files"]:
        if row["path"] in mapping:
            row["path"] = mapping[row["path"]]
            row["mtime_ns"] = current_files[row["path"]]["mtime_ns"]
    relocated["files"] = sorted(relocated["files"], key=lambda row: row["path"])
    relocated["trajectory_ids"] = sorted(remap_trajectory_ids(relocated["trajectory_ids"], mapping))
    if relocated != current:
        raise ValueError("Source dataset identity changed (paths, sizes, modification times, trajectories or reader settings)")
    return mapping


def _atomic_write(path, writer, *, overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            # Atomic no-replace publication, including a race after exists().
            os.link(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_torch(value, path, *, overwrite=False):
    path = Path(path)

    def write(stream):
        if path.suffix == ".gz":
            with gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as compressed:
                torch.save(value, compressed)
        else:
            torch.save(value, stream)

    _atomic_write(path, write, overwrite=overwrite)


def load_torch(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        return torch.load(stream, map_location="cpu", weights_only=True)


def json_ready(value):
    if isinstance(value, torch.Tensor):
        return json_ready(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        # JSON has no numeric representation for NaN or infinity.
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    return value


def save_json(value, path, *, overwrite=False):
    encoded = (json.dumps(json_ready(value), indent=2, allow_nan=False)+"\n").encode("utf-8")
    _atomic_write(path, lambda stream: stream.write(encoded), overwrite=overwrite)
