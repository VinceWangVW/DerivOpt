# Data setup

Obtain the datasets from the [PDEBench repository](https://github.com/pdebench/PDEBench)
using its download instructions. Keep the original HDF5 arrays, coordinates,
time axes and simulation metadata. Raw data are not distributed with this code.

Suggested layout (filenames remain those supplied by the dataset):

```text
data/
  README.md
  advection/*.h5
  burgers/*.h5
  diffusion_sorption/*.h5
  diffusion_reaction/*.h5
  radial_dam_break/*.h5
  incompressible_ns/*.h5
  compressible_ns/*.h5
```

Only this README is exempted from the `data/*` ignore rule. The program does not
discover files by folder name: supply individual paths through `--data`, which
also accepts multiple files. Data can stay outside the repository.

## Required contents

`N` is the number of trajectories, `T` the number of stored times, and `X,Y` the
spatial dimensions. The family argument uses the names in the first column.

| Family | Arrays in each HDF5 file | State channels |
| --- | --- | --- |
| `advection`, `burgers` | `tensor[N,T,X]`, `x-coordinate`, `t-coordinate` | u |
| `diffusion_sorption` | Trajectory groups with `data[T,X,1]`, `grid/x`, `grid/t`, simulation config | u |
| `diffusion_reaction` | Trajectory groups with `data[T,Y,X,2]`, `grid/x`, `grid/y`, `grid/t`, config with physical domain bounds | u, v |
| `radial_dam_break` | Trajectory groups with `data[T,X,Y,1]`, `grid/x`, `grid/y`, `grid/t` | h |
| `incompressible_ns` | `velocity[N,T,X,Y,2]`, `t[N,T]`, simulation config | Vx, Vy |
| `compressible_ns` | `density`, `Vx`, `Vy`, `pressure`, each `[N,T,X,Y]` or single-trajectory `[T,X,Y]`; x/y/t coordinates | density, Vx, Vy, pressure |

Diffusion–Sorption needs diffusivity (`D`), boundary value (`sol`) and physical
bounds (`x_left`, `x_right`) in `config.sim`, or explicit `boundary_parameters`
in the experiment config. Incompressible NS needs both velocity components,
not a vorticity-only file. The reader returns `[T,C,X]` or `[T,C,X,Y]` states;
Diffusion–Reaction is transposed from the source writer's Y,X order.

## Check a file

```sh
python -m derivopt inspect-data --family advection --data data/advection/FILE.h5
```

Replace `FILE.h5` with the actual filename. This reads a two-frame window of one
trajectory (index 0 by default; use `--frames` for a longer excerpt). For a full
training-path check, use the commands below. Training, calibration and evaluation use
windowed reads, avoiding a full trajectory allocation for a short window.

For a bounded check using actual source arrays from all seven families:

```sh
python scripts/download_real_smoke.py --output data/real-smoke
python scripts/check_real_smoke.py --output runs/real-data-check
```

The download contains three independent trajectories per family, each with five
consecutive original states and the full published spatial grid. The check uses
separate train/validation/test trajectories, two training updates and a three-step
test rollout for DerivOpt and Primitive, including checkpoint reload. This is a
short execution check, not a benchmark run. Use new output directories on reruns.

## Dataset partitions and time spacing

By default, preparation makes a seeded 70/15/15 trajectory split. To preserve a
published split, use `--partitions data/splits.json` with `calibrate` or `train`.
For example, for a four-trajectory file:

```json
{
  "version": 1,
  "train": [{"file": "advection/FILE.h5", "indices": [0, 1]}],
  "val": [{"file": "advection/FILE.h5", "indices": [2]}],
  "test": [{"file": "advection/FILE.h5", "indices": [3]}]
}
```

Paths are relative to the manifest, and indices refer to trajectory order within
each source file (starting at zero), not the combined `--data` order. Use different
files in the entries when train/test are supplied separately. All three splits
must be nonempty, disjoint and cover every supplied trajectory exactly once.
The manifest cannot change the partitions in a reused prepared state/checkpoint.

Time coordinates must have a uniform spacing. Preparation records that spacing;
training and evaluation check each encountered trajectory against it. Different
sampling intervals require separate experiments. Ordinary float32 timestamp
roundoff is allowed; trajectories are not resampled automatically.

## Boundary handling and compatibility

- The native PDEBench file `ns_incom_inhom_2d_512-0.h5` declares
  `velocity_extrapolation: ZERO`. It uses a homogeneous Dirichlet velocity basis
  and nonwrapping physical curl/divergence. Its original writer uses grid-unit
  coordinates (`0.5, 1.5, ..., n-0.5`); these are preserved. Explicit coordinate
  arrays take precedence. Periodic NS uses a separate boundary configuration.
- The released `2D_diff-react_NA_NA.h5` has subsampled coordinates offset from
  the declared domain's cell centres. The reader aligns both field values and
  coordinates using linear interpolation with Neumann reflection; the source
  coordinates and transformation are recorded. This requires a complete uniform
  grid matching the declared domain, with an offset of at most one-quarter cell.
  Cropped grids and larger shifts are rejected; raw files are unchanged.
- Compressible NS accepts batched periodic data and single-trajectory shock
  layouts. Transmissive variants use zero-gradient boundary bases, not periodic
  ones. The seven-family real-data check uses the periodic CNS training file;
  shock layouts are covered by reader and geometry tests.
- Combine only trajectories with the same grid, physical domain, boundary
  conditions and time spacing. A successful schema check does not establish this.
- Scalar/CFD files may have one extra unused final timestamp, which is removed
  when matching it to the state array. Other length mismatches are rejected.
- Keep `spatial_stride=1` for standard experiments: spatial subsampling in the
  reader is separate from the codec's low-pass/coarsen operator.
- Prepared states and checkpoints record data locations and file metadata. To
  reuse them after copying data, follow [moving data files](../docs/commands.md#moving-data-files).
- Earlier prepared states/checkpoints without time-step metadata or the current
  radial calibration support need fresh preparation and training; the loader
  reports an explicit version error.

Unsupported layouts fail explicitly. Format references are the PDEBench
[scalar/CFD loader](https://github.com/pdebench/PDEBench/blob/main/pdebench/models/fno/utils.py),
[reaction simulator](https://github.com/pdebench/PDEBench/blob/main/pdebench/data_gen/src/sim_diff_react.py)
and [original INS writer](https://github.com/pdebench/PDEBench/blob/2f71cb84a0c0a7abdc69ccc050dccec0b586716f/pdebench/data_gen/src/sim_ns_incomp_2d.py).
