#!/usr/bin/env bash
set -euo pipefail

# NAS-Bench-301: sample 100 architectures from surrogate v1.0
python3 main.py nasbench301 --num_samples 100 --version 1.0 --k 10

# NAS-Bench-301: sample 500 architectures from surrogate v2.0
python3 main.py nasbench301 --num_samples 500 --version 2.0 --k 50
