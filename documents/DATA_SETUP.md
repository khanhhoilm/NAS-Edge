# Data and Model Setup

This document mirrors the required assets described in the main README and groups them by benchmark.

## NAS-Bench-Suite-Zero (SuiteZero)

Place datasets under:

```
nasbench/nas_bench_suite_zero/naslib/data/
```

Required files:

- NAS-Bench-101: `nasbench_only108.pkl`
- NAS-Bench-201:
  - `nb201_cifar10_full_training.pickle`
  - `nb201_cifar100_full_training.pickle`
  - `nb201_ImageNet16_full_training.pickle`
- TransNAS-Bench-101:
  - `transnas-bench_v10141024.pth`

SuiteZero will auto-download NAS-Bench-301 surrogate models on first use unless downloads are blocked. If blocked, download the v1.0 bundle and place it in:

```
nasbench/nas_bench_suite_zero/naslib/data/nb_models_1.0/xgb_v1.0/
nasbench/nas_bench_suite_zero/naslib/data/nb_models_1.0/lgb_runtime_v1.0/
```

## NAS-Bench-301 (repo command)

The `nasbench301` subcommand auto-downloads surrogate ensembles into:

```
nasbench/nasbench301/nb_models_<version>/
```

Example run:

```
python3 main.py nasbench301 --version 1.0 --num_samples 10 --k 3
```

## HW-NAS-Bench

You need:

- HW-NAS-Bench pickle (e.g. `HW-NAS-Bench-v1_0.pickle`) at the repo root or provided via CLI.
- NAS-Bench-201 weights file `NAS-Bench-201-v1_1-096897.pth` and pass it via `--nas201-path`.

## AccelNASBench

Models expected under:

```
nasbench/accel_nasbench/anb_models_0_9/
```

Download helper:

```
python3 -c "from nasbench.accel_nasbench.model_downloader import download_models; download_models('0.9', delete_zip=True, download_dir='nasbench/accel_nasbench')"
```

## NAS-Bench-X11

Surrogate files are expected under:

```
nasbench/nas_bench_x11/models/
```

They are already included in this repository.
