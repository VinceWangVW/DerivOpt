# Command guide

Run these commands from the project directory unless using absolute paths. The
package command `derivopt` and `python -m derivopt` are equivalent. Configuration
files may use JSON or YAML.

## Installation and tests

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest -q
.venv/bin/python -m derivopt smoke --suite quick --output runs/smoke-quick
```

The experiment runner currently supports CPU only. The smoke suite checks
training, gradients, storage budgets and rollouts. `main`, `regimes`, `shared`,
and `all` select broader smoke tests; these are short execution checks, not
benchmark training runs.

## Synthetic data and inspection

```sh
.venv/bin/python -m derivopt fixture --family advection --output data/advection-smoke.h5 --trajectories 4 --steps 5 --size 128
.venv/bin/python -m derivopt inspect-data --family advection --data data/advection-smoke.h5
```

Fixtures are synthetic test fields stored in the supported HDF5 layouts.
Existing files are not overwritten. Without
`--size`, defaults are 128 points in 1D and 32×32 in 2D. For a rectangular 2D
fixture, pass e.g. `--size 32 48`. `--data` accepts several paths. Inspection reads
the first two frames of one trajectory (change `--index` or `--frames` if needed),
and reports the full trajectory length separately.

For benchmark inputs, see [data setup](../data/README.md). Preserve physical
coordinates and simulation metadata. Diffusion–Sorption requires its source
boundary parameters or explicit `boundary_parameters` in the config.

## Prepare once, then train

```sh
.venv/bin/python -m derivopt calibrate --config configs/smoke_advection_fno.json --data data/advection-smoke.h5 --output runs/prepared/advection.pt.gz
.venv/bin/python -m derivopt train --config configs/smoke_advection_fno.json --data data/advection-smoke.h5 --prepared runs/prepared/advection.pt.gz --output runs/advection
```

`calibrate` uses the training partition only and saves the channel statistics,
field library, exact integer design, split identities and geometric operators.
Reuse compatible preparation across backbones/methods; do not recalibrate on test
data. `train` can perform preparation itself when `--prepared` is omitted.
Pass `--partitions data/splits.json` to `calibrate` or `train` to preserve explicit
file-local train/validation/test assignments. See [the manifest schema](../data/README.md#dataset-partitions-and-time-spacing).
Without it, preparation uses a seeded trajectory-level random split. A manifest
provided with an existing prepared state or resume checkpoint must match its saved
partitions; it cannot replace them. A fixed physical time step is saved and checked
for each encountered trajectory.
Training and evaluation require new or empty output directories; keep prepared
files outside the training output directory. Data files are matched by their
paths, sizes, modification times, trajectory identities and reader settings.

`last.pt.gz` contains the final weights, optimizer, preparation and RNG state for
resume. `best.pt.gz` is saved when a finite validation nRMSE improves and supplies
the final test evaluation in `results.json` and `predictions.pt.gz`. The summary
records the evaluated checkpoint role and step. If validation never becomes
finite, evaluation uses the final checkpoint and records that choice.

To continue training, keep all config fields except `train_steps` unchanged:

```sh
.venv/bin/python -m derivopt train --config configs/smoke_advection_fno.json --data data/advection-smoke.h5 --resume runs/advection/last.pt.gz --output runs/advection-resumed
```

On resume, `train_steps` is the number of **additional** updates. Reusing the
two-step smoke config above therefore adds two updates. Resume writes a new
directory, preserving the original run artifacts.

### Moving data files

When copying data to another location or machine, save a JSON mapping of the old
absolute paths (recorded in the checkpoint) to the new absolute paths:

```json
{"/old/data/advection.h5": "/new/data/advection.h5"}
```

Pass this file as `--data-map data-map.json` along with the new `--data` paths:

```sh
.venv/bin/python -m derivopt evaluate --checkpoint runs/advection/best.pt.gz --data /new/data/advection.h5 --data-map data-map.json --output runs/advection-moved
```

The same option works with `train --prepared` and `train --resume`. Python
callers can pass `data_map=path_or_dict` to `train` or `evaluate_checkpoint`.
Only mapped files may change path and modification time. File sizes, reader
settings, trajectory IDs, fixed partitions and grid shapes must still match;
sample coordinates, boundary conditions and time spacing are checked when read.
The mapping declares that these are copies of the same data, so do not use it
for modified datasets. The original files need not remain accessible.

Existing checkpoints are unchanged. A resumed run saves the new locations in its
checkpoints, so later runs at those locations need no mapping. Unmapped files
retain strict identity checks. This option is not used by `calibrate` or `inputs`.

### Configuration examples

- `canonical_ns_fno.json`: canonical NS budget/retention settings with an FNO
  predictor. Boundary conditions follow the supplied data.
- `smoke_advection_fno.json`: two-update FNO check with DerivOpt.
- `smoke_rolloutmulti.json`: the same small case with three-step closed-loop training supervision.
- `smoke_shared.json`: a small shared-candidate-library run.

The `smoke` field labels short functional runs, using the same selection and
codec paths. Adjust the training/model settings for longer experiments. Exact
selection raises a resource-limit error if the configured search limit is
exceeded; it does not silently switch to an approximate selector.

## Predictor-free input mechanism

```sh
.venv/bin/python -m derivopt inputs --prepared runs/prepared/advection.pt.gz --data data/advection-smoke.h5 --output runs/advection-inputs --split test --max-states 16
```

This applies coarsen–quantize–decode and evaluates the carried input state
at t=0 of each trajectory, without calling or training a predictor. `--max-states`
limits how many such initial states are inspected, not consecutive time frames.
The default controls are Primitive,
BestSingleDerived, DerivBase and DerivOpt. Use e.g. `--methods primitive derivopt`
to select a subset. The output directory contains `input_metrics.json`,
`input_fields.pt.gz`, and `calibration_curves.json`; calibration curves use only
training-calibrated channel statistics. Learned-latent t=0 metrics instead come
from evaluating a trained latent checkpoint.

## Checkpoint evaluation and re-evaluation

Use the checkpoint and predictions paths printed in each command's JSON result.
For the example above:

```sh
.venv/bin/python -m derivopt evaluate --checkpoint runs/advection/best.pt.gz --data data/advection-smoke.h5 --output runs/advection-eval --split test --horizon 3 --max-trajectories 1
.venv/bin/python -m derivopt reevaluate --predictions runs/advection-eval/predictions.pt.gz --output runs/advection-strict --tau-q 1.1 --tau-out 0.05 --fine-low 0.75 --fine-high 1.0
```

Evaluation uses closed-loop feedback. Re-evaluation reads saved predictions; it
does not train or regenerate trajectories. Changing only thresholds must leave
raw fineRel unchanged. Changing the fine-band definition is a separate metric
change and may change fineRel.

### Metric outputs

- `nrmse`: mean of per-channel relative RMSEs, averaged over predicted steps
  (excluding the initial state).
- `input_pass`: fraction of initial states passing the detail criteria. The
  defaults require expressible/fine relative errors < 1, `q_fine` in
  `[1/1.25, 1.25]`, and `e_out < 0.10`, with fine band `(0.75, 1]` of the retained
  cutoff.
- `detail_horizon_steps`: number of consecutive valid predicted frames starting
  at t=1, stopping at the first failed prediction. Later recovery does not extend
  the horizon. The initial state is evaluated separately and does not gate it.
- `detail_horizon_normalized`: valid predicted steps divided by rollout length.
  `detail_horizon` is an alias for this normalized value. Means include all
  evaluated trajectories, including those whose initial state fails.

Evaluation records `horizon_protocol` and `horizon_units`. Results that include
t=0 in the horizon are a different metric and cannot be pooled with the current
future-only horizon. `reevaluate` can score existing saved trajectories with the
current definition without retraining; it writes to a new output directory and
leaves the original metrics unchanged. The reporting helpers reject mixed
protocols and require explicit protocol metadata when summarizing horizons.

Undefined fine-energy ratios fail the detail test. Nonfinite diagnostics remain
in the records as JSON strings: `"NaN"`, `"Infinity"`, `"-Infinity"` in saved
results, and `"nan"`, `"inf"`, `"-inf"` in CLI output.

## Main configuration matrix

```sh
.venv/bin/python -m derivopt matrix
.venv/bin/python -m derivopt matrix --output runs/main-matrix.json
.venv/bin/python -m derivopt matrix --smoke --list
.venv/bin/python -m derivopt matrix --seeds 0 1 2 --output runs/main-three-seed-matrix.json
```

The main matrix is 7 PDE families × 4 backbones × 3 budgets × 3 retain fractions ×
7 methods = **1764 configurations**. This command generates a manifest; it does
not launch training. `--smoke` uses smaller models and fewer updates while
retaining the same experiment axes.

`--seeds` repeats each configuration for the supplied distinct nonnegative
training seeds while keeping the split seed fixed. Three seeds produce 5292
manifest entries. Each manifest configuration is an `ExperimentConfig`; dispatch
selected entries through `runner.train` with a separate output directory per run.

## Training-only calibration folds

At least five training trajectories are needed for five folds. A separate small
fixture makes the requirement explicit:

```sh
.venv/bin/python -m derivopt fixture --family advection --output data/advection-folds.h5 --trajectories 16 --steps 5 --size 128
.venv/bin/python -m derivopt calibrate --config configs/smoke_advection_fno.json --data data/advection-folds.h5 --output runs/prepared/advection-fold0.pt.gz --fold 0 --folds 5
```

Repeat `--fold 0` through `--fold 4` with distinct output filenames. Each leaves
one training-trajectory fold out of calibration. The original validation/test
partitions stay fixed and cannot leak into calibration. `calibration_states`
still limits how many eligible training states are used. Saved provenance records
the actual states/trajectory identities, fold and total fold count.

For matched model-seed comparisons, `config.paired_seed_configs(base, seeds)`
generates DerivOpt/ArchMulti pairs by default. Report measured per-trajectory rows
with `reporting.configuration_macro` and `reporting.paired_seed_summary`; the
latter rejects incomplete pairs and reports sample SD, not standard error.

ConvLSTM retains its hidden/cell state throughout each rollout and resets it
between trajectories. Training uses short teacher-forced sequences (default
`recurrent_training_steps=3`) with one-step loss at each frame, so the recurrent
weights receive temporal gradients. RolloutMulti instead feeds predictions back
for its configured multi-step loss. ArchMulti retains only its fine-branch recurrent state;
the coarse branch supplies frame-local features. With the default ConvLSTM
width 32 and depth 2, a 32-by-32 simulator grid retains 512 KiB per trajectory.
Evaluation records this separately as `internal_recurrent_state_bytes`; it is
not part of the external-state payload. Explicit custom widths/depths and grid
sizes change this internal memory cost.
Example for one common library/regime definition:

```python
import json
from pathlib import Path
from derivopt.reporting import configuration_macro, paired_seed_summary

paths = [Path("runs/derivopt-seed0/results.json"), Path("runs/archmulti-seed0/results.json")]
rows = []
for path in paths:
    run = json.loads(path.read_text())
    rows.extend({**run["config"], **row, "dataset_identity": run["dataset_identity"]}
                for row in run["evaluation"]["rows"])
macro = configuration_macro(rows, "nrmse")
paired = paired_seed_summary(rows, "nrmse", "derivopt", "archmulti")
```

Use matching libraries, data splits, calibration folds and training profiles.
The result includes configuration coverage and seed counts. Nonfinite scalar
metrics raise an error. With one seed, sample SD is undefined (`None`).
