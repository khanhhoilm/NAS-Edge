import argparse
import os
from pathlib import Path
import heapq
import json
import hashlib
from datetime import datetime

from nasbench.nasbench201 import NASBench201API as API

api_file="src/nasbench/nasbench201/NAS-Bench-201-v1_1-096897.pth"


def get_top_arch(args):  
    api = API(api_file, verbose=False)

    heap: list[tuple[float, int, str]] = []
    missing_dataset = 0
    missing_metric = 0

    def _get_metrics(arch_index):
        
        try:
            info = api.query_meta_info_by_index(arch_index, hp=args.hp)
        except Exception:
            return False

        if args.dataset not in info.get_dataset_names():
            missing_dataset += 1
            return False

        try:
            metrics = info.get_compute_costs(args.dataset)
        except Exception:
            return False
            # missing_metric += 1
        return metrics
            
    for arch_index in api.evaluated_indexes:
        try:
            info = api.query_meta_info_by_index(arch_index, hp=args.hp)
        except Exception:
            continue

        if args.dataset not in info.get_dataset_names():
            missing_dataset += 1
            continue

        try:
            metrics = info.get_metrics(args.dataset, args.setname, iepoch=None, is_random=args.is_random)
        except Exception:
            missing_metric += 1
            continue

        acc = metrics.get('accuracy', None)
        if acc is None:
            missing_metric += 1
            continue

        acc_f = float(acc)
        arch_str = api[arch_index]

        if len(heap) < args.k:
            heapq.heappush(heap, (acc_f, arch_index, arch_str))
        else:
            if acc_f > heap[0][0]:
                heapq.heapreplace(heap, (acc_f, arch_index, arch_str))

    topk = sorted(heap, key=lambda x: x[0], reverse=True)
    if missing_dataset > 0:
        print(f'Note: skipped {missing_dataset} architectures missing dataset={args.dataset!r} for hp={args.hp!r}.')
    if missing_metric > 0:
        print(f'Note: skipped {missing_metric} architectures missing metric set={args.setname!r}.')
    print(f'Dataset: {args.dataset} | set: {args.setname} | hp: {args.hp} | top-k: {args.k}')

    # Write results to disk (mirrors nasbench101: results/<benchmark>/<hash>.json)
    out_dir = Path(getattr(args, "results_dir", "results")) / "nasbench201"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "benchmark": "nasbench201",
        "dataset": args.dataset,
        "setname": args.setname,
        "hp": args.hp,
        "is_random": bool(getattr(args, "is_random", False)),
        "k": int(args.k),
        "seed": int(getattr(args, "seed", 0)),
        "api_file": os.path.abspath(api_file),
        "written_at": datetime.utcnow().isoformat() + "Z",
        "items": [],
    }

    written_paths: list[str] = []
    for rank, (acc, arch_index, arch_str) in enumerate(topk, start=1):
        print(f'{rank:2d}. acc={acc:.4f}% | index={arch_index:5d} | arch={arch_str}')

        # Stable file name based on index + arch string
        out_path = out_dir / f"arch_index-{arch_index}.json"
        metric_path = out_dir / f"metric-{arch_index}.json"

        payload = {
            "benchmark": "nasbench201",
            "rank": int(rank),
            "accuracy": float(acc),
            "accuracy_unit": "percent",
            "arch_index": int(arch_index),
            "arch": arch_str,
            "dataset": args.dataset,
            "setname": args.setname,
            "hp": args.hp,
            "is_random": bool(getattr(args, "is_random", False)),
            "seed": int(getattr(args, "seed", 0)),
            "api_file": os.path.abspath(api_file),
            "written_at": datetime.utcnow().isoformat() + "Z",
        }

        metrics = _get_metrics(int(arch_index))

        payload.update(metrics)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        written_paths.append(str(out_path))
        summary["items"].append({
            "rank": int(rank),
            "arch_index": int(arch_index),
            "accuracy": float(acc),
            "arch": arch_str,
        })

    if written_paths:
        print("\nSaved results to:")
        for p in written_paths[:3]:
            print(f"- {p}")
        if len(written_paths) > 3:
            print(f"- ... ({len(written_paths)} files total)")
        print(f"Base folder: {out_dir}")

