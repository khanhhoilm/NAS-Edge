# Usage Guide

This guide summarizes how to run the CLI and where to place required assets.

## 1) Environment setup

Use the provided conda environment file:

- Create: `conda env create -f environment.yml`
- Activate: `conda activate nasbench-framework`

## 2) Entry point

All commands run through `main.py` and dispatch to `src.get_top_arch(args)`.

Examples live under the [examples/](../examples) directory.

## 3) Common CLI patterns

- Get help:
  - `python3 main.py -h`
  - `python3 main.py nasbench101 -h`
  - `python3 main.py nasbench201 -h`
  - `python3 main.py nasbench301 -h`
  - `python3 main.py hwnasbench -h`
  - `python3 main.py accelnasbench -h`
  - `python3 main.py suitezero -h`
  - `python3 main.py nasbench_x11 -h`

- Run an example script:
  - `bash examples/nasbench101.sh`

## 4) Benchmark quick starts

### NAS-Bench-101

- `python3 main.py nasbench101 --k 10`

### NAS-Bench-201

- `python3 main.py nasbench201 --dataset cifar10 --hp 12 --setname x-valid --k 10`

### NAS-Bench-301 (surrogate ensemble)

- `python3 main.py nasbench301 --num_samples 100 --version 1.0 --k 10`

### HW-NAS-Bench

- `python3 main.py hwnasbench --mode per_device --device raspi4 --metric latency --dataset cifar10 --search_space nasbench201 --split x-valid --k 10`

### AccelNASBench

- `python3 main.py accelnasbench --num-candidates 200 --top-k 10 --sort-by accuracy --seed 3`

### NAS-Bench-Suite-Zero

- `python3 main.py suitezero --search_space nasbench201 --dataset cifar100 --metric val_acc --k 10`

### NAS-Bench-X11

- `python3 main.py nasbench_x11 --search_space nb201 --epoch 199 --k 20`

## 5) Output notes

- Use `--json` (HW-NAS-Bench) or `--jsonl` (SuiteZero) for machine-readable output.
- Some commands need large model/data files. See the data setup guide.
