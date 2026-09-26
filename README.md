# DerivOpt

Code for **Derived Fields Preserve Fine-Scale Detail in Budgeted Neural Simulators**  
Wenshuo Wang and Fan Zhang · [Paper (arXiv)](https://arxiv.org/abs/2603.29224)

DerivOpt selects physical fields and allocates their precision under a fixed
state-storage budget. The package provides field calibration and selection,
quantization, neural predictors, and input-state and closed-loop evaluation.

## Installation

Requires Python 3.11 or later. Run from this directory:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
```

The experiment runner currently supports CPU execution. Published benchmark
scores and pretrained benchmark weights are not included in this release.

## Data

Download the required HDF5 files from [PDEBench](https://github.com/pdebench/PDEBench)
and place them under `data/`, or pass their existing locations with `--data`.
See [data setup](data/README.md) for directory layout, required fields, boundary
handling, and a small real-data check covering all seven families.

```sh
python -m derivopt inspect-data --family advection --data data/advection/FILE.h5
```

Replace `FILE.h5` with the downloaded filename. Inspection reads the file schema
and a two-frame excerpt.

## Quick start

This small synthetic example checks training and evaluation without downloading
benchmark data. Use fresh file and output-directory names when running it again.

```sh
python -m derivopt fixture --family advection --output data/advection-example.h5 --trajectories 4 --steps 5 --size 128
python -m derivopt train --config configs/smoke_advection_fno.json --data data/advection-example.h5 --output runs/advection-example
python -m derivopt evaluate --checkpoint runs/advection-example/best.pt.gz --data data/advection-example.h5 --output runs/advection-example-eval --split test --horizon 3
```

Training writes checkpoints and `results.json`; evaluation also saves predictions.
`best.pt.gz` is selected by validation nRMSE, and `last.pt.gz` supports resuming.
The example uses two training updates and is not a benchmark experiment.

Input pass rate measures reconstruction at t=0. Detail horizon measures the
consecutive valid predicted steps from t=1, normalized by rollout length and
averaged over all test trajectories; initial failure does not gate this horizon.

The code includes seven time-dependent PDEBench families, U-Net/FNO/ConvLSTM/
Transformer predictors, and the Primitive, BestSingleDerived, DerivBase,
DerivOpt, ArchMulti, RolloutMulti and LatentAE-Hyper methods. Small Expert and
Shared field libraries and the DerivOpt + ArchMulti combination are also available.

See the [command guide](docs/commands.md) for configurations, calibration,
input-state analysis, evaluation and resume. Run tests with `python -m pytest -q`.

## License

This code is released under the [MIT License](LICENSE). PDEBench data retain
their original licenses and are downloaded separately.

## Citation

```bibtex
@misc{wang2026derivedfields,
  title={Derived Fields Preserve Fine-Scale Detail in Budgeted Neural Simulators},
  author={Wenshuo Wang and Fan Zhang},
  year={2026},
  eprint={2603.29224},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2603.29224}
}
```
