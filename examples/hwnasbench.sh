#!/usr/bin/env bash
set -euo pipefail

# HW-NAS-Bench: per-device latency on raspi4
python3 main.py hwnasbench --mode per_device --device raspi4 --metric latency --dataset cifar10 --search_space nasbench201 --split x-valid --k 10

# HW-NAS-Bench: aggregate latency (mean)
python3 main.py hwnasbench --mode aggregate --agg mean --metric latency --json --k 10
