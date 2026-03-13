#!/usr/bin/env bash
set -euo pipefail

# NAS-Bench-101: top-10 by validation accuracy
python3 main.py nasbench101 --k 10

# NAS-Bench-101: top-20 by validation accuracy
python3 main.py nasbench101 --k 20
