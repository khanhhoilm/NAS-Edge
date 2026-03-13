# NAS-EDGE: Unified NAS Benchmark and Deployment Toolkit

This repository provides a unified CLI to query multiple NAS benchmarks, export top architectures, and convert selected models for deployment (PyTorch -> ONNX -> RKNN).
It is prepared as a companion codebase for a research paper release.

## Why this repo

- One CLI entry point (`main.py`) for multiple NAS benchmarks.
- Consistent top-k querying workflow across search spaces.
- Result persistence to JSON for downstream processing.
- Optional deployment pipeline to RKNN-compatible models.

## Supported benchmarks

| Benchmark | CLI command | Main purpose |
|---|---|---|
| NAS-Bench-101 | `nasbench101` | Top-k architectures by validation accuracy |
| NAS-Bench-201 | `nasbench201` | Top-k architectures by dataset/split/hp |
| NAS-Bench-301 | `nasbench301` | Surrogate-based random search |
| HW-NAS-Bench | `hwnasbench` | Hardware-aware ranking (latency/energy/power/accuracy) |
| AccelNASBench | `accelnasbench` | Surrogate ranking by accuracy/throughput/latency |
| NAS-Bench-Suite-Zero | `suitezero` | NASLib-based unified query over multiple spaces |
| NAS-Bench-X11 | `nasbench_x11` | Cross-benchmark surrogate models |
| NAS-Bench-Graph | `nasbench_graph` | GNN architecture benchmark querying |

## Repository structure

```text
.
├── main.py
├── environment.yml
├── examples/                         # Ready-to-run command examples
├── scripts/
│   └── download_suitezero_data.py
├── src/
│   ├── nasbench/                     # Benchmarks + query modules
│   └── nas2model/                    # NAS architecture -> model conversion helpers
├── convert_nasbench2pytoch.py        # Train/export top architectures to TorchScript
├── convert_pytorch2onnx-rknn.py      # Convert .pth/.pt to ONNX and RKNN
├── measure_energy_results/           # Example measurement CSV outputs
└── documents/                        # Additional usage/setup notes
```

## 1) Environment setup

### Prerequisites

- Linux recommended.
- Conda (Miniconda/Anaconda).
- Python 3.8 (managed by `environment.yml`).

### Create environment

```bash
conda env create -f environment.yml
conda activate nasbench-framework
```

### Compatibility install (recommended)

```bash
pip install "pandas<2.0.0" "ConfigSpace==0.4.21" absl-py
```

### Quick sanity check

```bash
python3 -c "import torch, tensorflow as tf; print('torch', torch.__version__, 'tf', tf.__version__)"
python3 main.py -h
```

## 2) Data and model assets

Some benchmarks require large external files (TFRecord, `.pth`, surrogate bundles).
Use these paths for this repository layout:

- `src/nasbench/nasbench101/`
- `src/nasbench/nasbench201/`
- `src/nasbench/nasbench301/`
- `src/nasbench/hw_nas_bench/`
- `src/nasbench/accel_nasbench/`
- `src/nasbench/nas_bench_x11/models/`
- `src/nasbench/nas_bench_suite_zero/naslib/data/`

### Important path note

A few legacy modules/scripts still reference `nasbench/...` (without `src/`).
If you encounter missing-file errors due to this, create a compatibility symlink at repo root:

```bash
ln -s src/nasbench nasbench
```

### SuiteZero data helper

```bash
python3 scripts/download_suitezero_data.py
```

This downloads required NAS-Bench-Suite-Zero files (via Google Drive) into the expected data folder.

## 3) Quickstart commands

All commands run via `main.py`.

### Help

```bash
python3 main.py -h
python3 main.py nasbench101 -h
python3 main.py nasbench201 -h
python3 main.py nasbench301 -h
python3 main.py hwnasbench -h
python3 main.py accelnasbench -h
python3 main.py suitezero -h
python3 main.py nasbench_x11 -h
python3 main.py nasbench_graph -h
```

### Typical runs

```bash
# NAS-Bench-101
python3 main.py nasbench101 --epochs 36 --k 10

# NAS-Bench-201
python3 main.py nasbench201 --dataset cifar10 --hp 12 --setname x-valid --k 10

# NAS-Bench-301
python3 main.py nasbench301 --num_samples 100 --version 1.0 --k 10

# HW-NAS-Bench (per-device)
python3 main.py hwnasbench --mode per_device --device raspi4 --metric latency --dataset cifar10 --search_space nasbench201 --split x-valid --k 10

# AccelNASBench
python3 main.py accelnasbench --num-candidates 200 --top-k 10 --sort-by accuracy --seed 3

# SuiteZero
python3 main.py suitezero --search_space nasbench201 --dataset cifar100 --metric val_acc --k 10

# NAS-Bench-X11
python3 main.py nasbench_x11 --search_space nb201 --epoch 199 --k 20

# NAS-Bench-Graph
python3 main.py nasbench_graph --dataset cora --k 10
```

You can also run prepared scripts in `examples/*.sh`.

## 4) Output artifacts

By default, result files are saved under:

```text
topk_nasbench_outputs/<benchmark>/
```

You can override this with:

```bash
python3 main.py --results-dir <your_output_dir> <subcommand> ...
```

## 5) Deployment pipeline (optional)

This repository includes a practical path from benchmark architecture to deployment format:

1. Query top architectures (JSON outputs).
2. Convert architecture JSON -> trained/exported TorchScript (`convert_nasbench2pytoch.py`).
3. Convert TorchScript/PyTorch models -> ONNX + RKNN (`convert_pytorch2onnx-rknn.py`).

Example:

```bash
python3 convert_nasbench2pytoch.py \
  --nasbench_type nasbench101 \
  --epochs 1 \
  --json_folder topk_nasbench_outputs/nasbench101 \
  --folder_output topk_nasbench_models/pytorch/nasbench101

python3 convert_pytorch2onnx-rknn.py \
  --pth-folder topk_nasbench_models/pytorch/nasbench101 \
  --onnx-folder topk_nasbench_models/onnx/nasbench101 \
  --rknn-folder topk_nasbench_models/rknn/nasbench101 \
  --target-platform rk3588
```

## 6) Energy measurement on edge (`scripts/EnMeEdge`)

This repo includes an end-to-end energy measurement pipeline for edge inference:

- **Host side**: `scripts/EnMeEdge/server.py` reads power from FNB USB meter (FNB48/FNB58/FNB48S/C1), exposes `/start`, `/stop`, `/health`, and computes energy in the inference time window.
- **Edge side**: `scripts/EnMeEdge/inference_on_edge.py` runs inference on `.pth` / `.onnx` / `.rknn` models over CIFAR-10, sends start/stop timestamps to the host server, and saves latency/accuracy/memory/energy metrics to JSON.

Quick start:

```bash
# 1) On host connected to FNB meter
python3 scripts/EnMeEdge/server.py --host 0.0.0.0 --port 8000 --time-interval 0.01 

# 2) On edge device (set SERVER_IP in inference_on_edge.py)
python3 scripts/EnMeEdge/inference_on_edge.py \
  --folder_path topk_nasbench_models/rknn/nasbench201 \
  --output_json measure_energy_results/nasbench201_edge.json
```

Notes:

- `inference_on_edge.py` expects CIFAR-10 at `./data/cifar-10-batches-py/test_batch`.
- For `.pth`, input must be TorchScript (`torch.jit.load`); for `.rknn`, script prefers `rknnlite` and falls back to `rknn`.
- Example aggregated CSV results are provided in `measure_energy_results/*.csv`.
- Detailed usage is documented in `scripts/EnMeEdge/readme.md`.

## 7) Reproducibility tips

- Use global seed: `python3 main.py --seed 42 <subcommand> ...`
- Log your environment (`conda env export > env.lock.yml`) for paper artifact release.
- Keep generated outputs in versioned folders (e.g., `topk_nasbench_outputs/<exp_name>`).

## 8) Known limitations

- External benchmark assets are large and may require manual download.
- NAS-Bench-X11 and SuiteZero rely on surrogate/model files that may vary by release.
- TensorFlow warnings can appear due to TF1 compatibility layers in NAS-Bench-101 paths.

## 9) Additional documentation

- `documents/USAGE_GUIDE.md`
- `documents/DATA_SETUP.md`
- `documents/EXAMPLES_INDEX.md`

## 10) Citation (for paper release)

Please replace with your final BibTeX before publishing:

```bibtex
@article{nas_hoi_2026,
  title   = {NAS-HOI: Unified NAS Benchmark and Deployment Toolkit},
  author  = {Your Name and Co-authors},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

## 11) Pre-publication checklist

- [ ] Replace citation block with final paper metadata.
- [ ] Add `LICENSE` file.
- [ ] Verify all external download links still work.
- [ ] Confirm `examples/*.sh` run on a clean machine.
- [ ] Remove local/temporary artifacts before tagging release.

---
