#!/usr/bin/env bash
set -euo pipefail

# AccelNASBench: rank by accuracy
python3 main.py accelnasbench --num-candidates 200 --top-k 10 --sort-by accuracy --seed 3

# AccelNASBench: rank by throughput on TPUv2
python3 main.py accelnasbench --sort-by throughput --throughput-device tpuv2 --model xgb

# AccelNASBench: rank by latency on zcu102
python3 main.py accelnasbench --sort-by latency --latency-device zcu102 --model xgb
