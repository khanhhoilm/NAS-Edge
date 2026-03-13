#!/usr/bin/env bash
set -euo pipefail

# NAS-Bench-X11: NB101 surrogate, top-10
python3 main.py nasbench_x11 --search_space nb101 --k 10

# NAS-Bench-X11: NB201 surrogate, epoch 199, top-20
python3 main.py nasbench_x11 --search_space nb201 --epoch 199 --k 20

# NAS-Bench-X11: NB301 surrogate, sample 10000, epoch 97, top-10
python3 main.py nasbench_x11 --search_space nb301 --num_samples 10000 --epoch 97 --k 10

# NAS-Bench-X11: NBNLP surrogate, sample 5000, epoch 50, top-15
python3 main.py nasbench_x11 --search_space nbnlp --num_samples 5000 --epoch 50 --k 15
