"""Fetch bounded, provenance-preserving slices of official PDEBench HDF5 files.

Source array values and spatial grids are preserved. The server must honor HTTP
byte ranges, and every family has a hard transfer budget. The downloaded subsets
are intended for short integration checks and use separate diagnostic partitions.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import time
import urllib.request

import h5py
import numpy as np


SOURCES = {
    "advection": (255675, "1D_Advection_Sols_beta1.0.hdf5"),
    "burgers": (281363, "1D_Burgers_Sols_Nu0.01.hdf5"),
    "diffusion_sorption": (133020, "1D_diff-sorp_NA_NA.h5"),
    "diffusion_reaction": (133017, "2D_diff-react_NA_NA.h5"),
    "radial_dam_break": (133021, "2D_rdb_NA_NA.h5"),
    "incompressible_ns": (133280, "ns_incom_inhom_2d_512-0.h5"),
    "compressible_ns": (164687, "2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5"),
}
CATALOG = "https://raw.githubusercontent.com/pdebench/PDEBench/main/pdebench/data_download/pdebench_data_urls.csv"
SUBSET_ATTRIBUTE = "derivopt_source_subset"


class RangeFile(io.RawIOBase):
    """Seekable HTTP byte-range reader with block cache and a hard transfer cap."""

    def __init__(self, url, byte_limit=128 * 1024**2, block_size=256 * 1024):
        self.url, self.byte_limit, self.block_size = url, byte_limit, block_size
        self.position, self.bytes_read, self.requests = 0, 0, 0
        self.size, self.cache = None, {}
        self._fetch(0, block_size)

    def _fetch(self, start, stop):
        if self.size is not None:
            stop = min(stop, self.size)
        count = stop - start
        if count <= 0:
            return
        if self.bytes_read + count > self.byte_limit:
            raise RuntimeError(f"Range download would exceed {self.byte_limit} bytes for {self.url}")
        for attempt in range(3):
            request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{stop-1}"})
            try:
                with urllib.request.urlopen(request, timeout=40) as response:
                    if response.status != 206:
                        raise RuntimeError(f"Server did not honor byte range: HTTP {response.status}")
                    actual_range = response.headers.get("Content-Range", "")
                    interval, total = actual_range.removeprefix("bytes ").split("/")
                    first, last = map(int, interval.split("-"))
                    size = int(total)
                    expected_last = min(stop, size) - 1
                    if first != start or last != expected_last or (self.size is not None and size != self.size):
                        raise RuntimeError(f"Unexpected Content-Range {actual_range!r}")
                    data = response.read(count + 1)
                    self.bytes_read += len(data)
                    self.requests += 1
                    if len(data) != last - first + 1:
                        raise RuntimeError("Incomplete or oversized HTTP range response")
                    self.size = size
                break
            except (OSError, TimeoutError):
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
        for offset in range(0, len(data), self.block_size):
            self.cache[(start + offset) // self.block_size] = data[offset:offset+self.block_size]

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else (self.position + offset if whence == 1 else self.size + offset)
        if self.position < 0:
            raise ValueError("Negative file position")
        return self.position

    def read(self, size=-1):
        if size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)
        result = bytearray()
        stop = self.position + size
        while self.position < stop:
            block = self.position // self.block_size
            if block not in self.cache:
                # Combine contiguous missing blocks into at most a 4 MiB request.
                last_block = min((stop-1)//self.block_size, block + 15)
                while last_block > block and any(i in self.cache for i in range(block, last_block+1)):
                    last_block -= 1
                self._fetch(block*self.block_size, (last_block+1)*self.block_size)
            cached = self.cache[block]
            offset = self.position % self.block_size
            piece = cached[offset:offset + stop-self.position]
            if not piece:
                break
            result.extend(piece)
            self.position += len(piece)
        return bytes(result)

    def readinto(self, buffer):
        value = self.read(len(buffer))
        buffer[:len(value)] = value
        return len(value)


def native(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(family, destination, trajectories=3, frames=5, byte_limit=128*1024**2):
    if not 3 <= trajectories or not 5 <= frames:
        raise ValueError("Real smoke subsets require >=3 independent trajectories and >=5 consecutive frames")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    output, partial = destination/f"{family}.h5", destination/f"{family}.partial.h5"
    manifest = destination/f"{family}.json"
    if any(path.exists() for path in (output, partial, manifest)):
        raise FileExistsError(f"Refusing to overwrite existing data or provenance for {family}")
    identifier, name = SOURCES[family]
    url = f"https://darus.uni-stuttgart.de/api/access/datafile/{identifier}"
    report = {"format_version": 1, "family": family, "source_url": url, "source_filename": name,
              "source_catalog": CATALOG, "source_trajectory_indices": list(range(trajectories)),
              "source_frame_start": 0, "source_frame_stop": frames, "source_frame_stride": 1,
              "spatial_selection": "all published spatial points; no resampling or coordinate conversion",
              "diagnostic_only": True, "original_file_checksum_verified": False, "copied_arrays": {}}
    remote = RangeFile(url, byte_limit)
    try:
        with h5py.File(remote, "r") as source, h5py.File(partial, "x") as target:
            report["source_bytes"] = remote.size
            report["source_root_attributes"] = {key: native(value) for key, value in source.attrs.items()}
            for key, value in source.attrs.items():
                target.attrs[key] = value

            def copy_array(source_path, target_path, selection):
                original = source[source_path]
                array = original[selection]
                if not np.isfinite(array).all():
                    raise ValueError(f"Nonfinite actual source array {source_path}")
                saved = target.create_dataset(target_path, data=array)
                for key, value in original.attrs.items():
                    saved.attrs[key] = value
                report["copied_arrays"][target_path] = {
                    "source_path": source_path, "source_shape": list(original.shape),
                    "saved_shape": list(array.shape), "dtype": str(array.dtype),
                    "array_sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()}

            if family in ("diffusion_sorption", "diffusion_reaction", "radial_dam_break"):
                groups = [f"{i:04d}" for i in range(trajectories)]
                report["source_groups"] = groups
                report["source_group_attributes"] = {}
                for key in groups:
                    original = source[key]
                    if original["data"].shape[0] < frames:
                        raise ValueError(f"Insufficient original frames in {key}")
                    group = target.create_group(key)
                    report["source_group_attributes"][key] = {k: native(v) for k, v in original.attrs.items()}
                    for attr, value in original.attrs.items():
                        group.attrs[attr] = value
                    copy_array(f"{key}/data", f"{key}/data", np.s_[:frames])
                    for axis in ("x",) if family == "diffusion_sorption" else ("x", "y"):
                        copy_array(f"{key}/grid/{axis}", f"{key}/grid/{axis}", np.s_[:])
                    copy_array(f"{key}/grid/t", f"{key}/grid/t", np.s_[:frames])
                    print(f"{family}: copied original group {key}, frames 0:{frames}", flush=True)
            else:
                fields = ("tensor",) if family in ("advection", "burgers") else (("velocity", "particles") if family == "incompressible_ns" else ("density", "Vx", "Vy", "pressure"))
                shape = source[fields[0]].shape
                if shape[0] < trajectories or shape[1] < frames:
                    raise ValueError("Insufficient original trajectories or frames")
                report["source_trajectory_count"], report["source_frame_count"] = shape[:2]
                for field in fields:
                    original = source[field]
                    # Separate independent trajectories keeps remote reads bounded.
                    values = [original[index, :frames] for index in range(trajectories)]
                    array = np.stack(values)
                    if not np.isfinite(array).all():
                        raise ValueError(f"Nonfinite actual source array {field}")
                    saved = target.create_dataset(field, data=array)
                    for key, value in original.attrs.items():
                        saved.attrs[key] = value
                    report["copied_arrays"][field] = {"source_path": field, "source_shape": list(original.shape),
                        "saved_shape": list(array.shape), "dtype": str(array.dtype),
                        "array_sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()}
                    print(f"{family}: copied {field}{list(array.shape)}", flush=True)
                if family == "incompressible_ns":
                    copy_array("t", "t", np.s_[:trajectories, :frames])
                    copy_array("force", "force", np.s_[:trajectories])
                    for axis in ("x-coordinate", "y-coordinate"):
                        if axis in source:
                            copy_array(axis, axis, np.s_[:])
                else:
                    copy_array("t-coordinate", "t-coordinate", np.s_[:frames])
                    for axis in ("x-coordinate",) if family in ("advection", "burgers") else ("x-coordinate", "y-coordinate"):
                        copy_array(axis, axis, np.s_[:])
            # Original attrs, including INS latestIndex/config, are preserved.
            target.attrs[SUBSET_ATTRIBUTE] = json.dumps({key: report[key] for key in (
                "format_version", "source_url", "source_filename", "source_trajectory_indices",
                "source_frame_start", "source_frame_stop", "source_frame_stride", "diagnostic_only")}
                | {"source_frame_count": report.get("source_frame_count"), "source_latestIndex": native(source.attrs.get("latestIndex"))})
        partial.rename(output)
        report.update({"status": "complete", "output": str(output), "output_bytes": output.stat().st_size,
                       "output_sha256": sha256(output), "downloaded_bytes": remote.bytes_read, "range_requests": remote.requests})
        manifest.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
        print(json.dumps({key: report[key] for key in ("family", "status", "output_bytes", "downloaded_bytes", "range_requests", "output")}), flush=True)
    finally:
        remote.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["all", *SOURCES], default="all")
    parser.add_argument("--output", type=Path, default=Path("data/real-smoke"))
    parser.add_argument("--trajectories", type=int, default=3)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--max-mib", type=int, default=128)
    args = parser.parse_args()
    if not 1 <= args.max_mib <= 512:
        parser.error("The per-family transfer cap must be between 1 and 512 MiB")
    for family in (SOURCES if args.family == "all" else [args.family]):
        download(family, args.output, args.trajectories, args.frames, args.max_mib*1024**2)
