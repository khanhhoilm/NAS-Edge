import os
import json
from datetime import datetime

import numpy as np




def get_top_arch(args):
    if not os.path.exists(args.tfrecord):
        raise FileNotFoundError(
            f"NAS-Bench-101 TFRecord not found at '{args.tfrecord}'. "
            "Download the dataset file and pass --tfrecord /path/to/nasbench_full.tfrecord."
        )

    from nasbench.nasbench101 import api

    nasbench = api.NASBench(args.tfrecord)
    records = []  # save (accuracy, arch_hash)

    for i, unique_hash in enumerate(nasbench.hash_iterator()):
        # Get metrics with specified epochs
        fixed_stats, computed_stats = nasbench.get_metrics_from_hash(unique_hash)

        # Get average accuracy from all runs
        for run_stats in computed_stats[args.epochs]:
            acc = run_stats['final_validation_accuracy']
            records.append((acc, unique_hash))

    print(f"Collected {len(records)} results from {i + 1} architectures")
  
    # Sort by accuracy in descending order
    records.sort(key=lambda x: x[0], reverse=True)

    # Get top-k
    # topk = records[:args.k]
    topk = []
    for acc, arch_hash in records:
        if len(topk) < args.k:
            if not topk or arch_hash not in [h for _, h in topk]:
                topk.append((acc, arch_hash))
        else:
            break


    print("\n" + "=" * 80)
    print(f"TOP {args.k} ARCHITECTURES WITH HIGHEST ACCURACY")
    print("=" * 80)

    written_paths = []

    for idx, (acc, arch_hash) in enumerate(topk):
        print(f"\nRank {idx+1}: Validation Accuracy = {acc:.6f}")
        print(f"Hash: {arch_hash}")
        
        # Get detailed information of the architecture
        fixed_stats, computed_stats = nasbench.get_metrics_from_hash(arch_hash)
        model_spec = fixed_stats['module_adjacency']
        operations = fixed_stats['module_operations']
        
        print(f"Operations: {operations}")
        print(f"Adjacency Matrix:\n{model_spec}")
        print("-" * 80)

        # Persist result to: results/<benchmark>/<hash>/result.json
        out_dir = os.path.join(getattr(args, "results_dir", "results"), "nasbench101")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"arch_hash-{arch_hash}.json")
        metric_path = os.path.join(out_dir, f"metric-{arch_hash}.json")

        payload = {
            "benchmark": "nasbench101",
            "hash": arch_hash,
            "rank": idx + 1,
            "validation_accuracy": float(acc),
            "epochs": int(args.epochs),
            "operations": list(operations),
            "adjacency_matrix": np.asarray(model_spec).tolist(),
            "seed": int(getattr(args, "seed", 0)),
            "tfrecord": os.path.abspath(args.tfrecord),
            "written_at": datetime.utcnow().isoformat() + "Z"
        }
        # computed_stats.update({"validation_accuracy": float(acc), "epochs": int(args.epochs)})
        for item in computed_stats[args.epochs]:
            if item.get("final_validation_accuracy", None) == acc:
                payload.update(item)
                break
        
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        
        # with open(metric_path, "w", encoding="utf-8") as f:
        #     json.dump(computed_stats, f, indent=2, ensure_ascii=False)
        
        written_paths.append(out_path)

    if written_paths:
        print("\nSaved results to:")
        # Keep output short (top-3 paths), but still indicate the base folder.
        for p in written_paths[:3]:
            print(f"- {p}")
        if len(written_paths) > 3:
            print(f"- ... ({len(written_paths)} files total)")
        print(f"Base folder: {os.path.join(getattr(args, 'results_dir', 'results'), 'nasbench101')}")

