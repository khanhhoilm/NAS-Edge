#!/usr/bin/env bash
set -euo pipefail

# SuiteZero: TransBench101 Macro (Taskonomy), sample 100 candidates and take top-5
python3 main.py suitezero --search_space transbench101_macro --dataset class_scene --k 5 --num_samples 100

# SuiteZero: NAS-Bench-201 on CIFAR-100, rank by validation accuracy
python3 main.py suitezero --search_space nasbench201 --dataset cifar100 --metric val_acc --k 10

# SuiteZero: JSON Lines output
python3 main.py suitezero --search_space transbench101_micro --dataset class_scene --k 5 --num_samples 100 --jsonl
