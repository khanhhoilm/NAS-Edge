#!/usr/bin/env bash
set -euo pipefail

# NAS-Bench-201: CIFAR-10, hp=12, validation split
python3 main.py nasbench201 --dataset cifar10 --hp 12 --setname x-valid --k 10

# NAS-Bench-201: CIFAR-100, hp=200, test split, random architectures
python3 main.py nasbench201 --dataset cifar100 --hp 200 --setname x-test --is-random --k 10
