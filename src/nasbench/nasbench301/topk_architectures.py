import argparse
import heapq
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
from ConfigSpace.read_and_write import json as cs_json

import nasbench301 as nb
from nasbench301.representations import Genotype


def _load_configspace(configspace_path: str, seed: int) -> Any:
    with open(configspace_path, "r", encoding="utf-8") as f:
        json_string = f.read()
    configspace = cs_json.read(json_string)
    try:
        configspace.seed(seed)
    except Exception:
        pass
    return configspace


def _ensure_models(version: str, download_dir: str) -> str:
    models_dir = os.path.join(download_dir, f"nb_models_{version}")
    if not os.path.exists(models_dir):
        nb.download_models(version=version, delete_zip=True, download_dir=download_dir)
    return models_dir


def _to_float(x: Any) -> float:
    if isinstance(x, (float, int)):
        return float(x)
    try:
        return float(np.asarray(x).reshape(-1)[0])
    except Exception:
        return float(x)


def _config_dict_to_genotype(config_dict: Dict[str, Any]) -> Genotype:
    base = "NetworkSelectorDatasetInfo:darts:"

    def parse_inputs(v: Any) -> List[int]:
        if isinstance(v, str):
            return [int(x) for x in v.split("_") if x != ""]
        if isinstance(v, (list, tuple)):
            return [int(x) for x in v]
        raise ValueError(f"Unsupported inputs_node format: {type(v)}")

    def build_cell(cell_type: str) -> List[Tuple[str, int]]:
        cell: List[Tuple[str, int]] = []
        start = 0
        n = 2
        for node_idx in range(4):
            if node_idx == 0:
                selected_inputs = [0, 1]
            else:
                key = f"{base}inputs_node_{cell_type}_{node_idx + 2}"
                if key not in config_dict:
                    raise KeyError(f"Missing required key {key} in config")
                selected_inputs = sorted(parse_inputs(config_dict[key]))
                if len(selected_inputs) != 2:
                    raise ValueError(f"Expected 2 inputs for {key}, got {selected_inputs}")

            for input_node in selected_inputs:
                edge_idx = start + input_node
                edge_key = f"{base}edge_{cell_type}_{edge_idx}"
                if edge_key not in config_dict:
                    raise KeyError(f"Missing required key {edge_key} in config")
                op = config_dict[edge_key]
                cell.append((op, int(input_node)))

            start += n
            n += 1

        return cell

    concat = [2, 3, 4, 5]
    normal = build_cell("normal")
    reduce = build_cell("reduce")
    return Genotype(normal=normal, normal_concat=concat, reduce=reduce, reduce_concat=concat)


def _genotype_to_jsonable(g: Genotype) -> Dict[str, Any]:
    return {
        "normal": list(map(list, g.normal)),
        "normal_concat": list(g.normal_concat),
        "reduce": list(map(list, g.reduce)),
        "reduce_concat": list(g.reduce_concat),
    }


def topk_architectures(
    *,
    version: str,
    k: int,
    num_samples: int,
    seed: int,
    with_noise: bool,
) -> List[Dict[str, Any]]:
    if k < 1:
        raise ValueError("k must be >= 1")
    if num_samples < 1:
        raise ValueError("num_samples must be >= 1")
    if k > num_samples:
        raise ValueError(f"k ({k}) cannot be greater than num_samples ({num_samples})")

    current_dir = os.path.dirname(os.path.abspath(__file__))

    models_dir = _ensure_models(version=version, download_dir=current_dir)
    performance_dir = os.path.join(models_dir, f"xgb_v{version}")

    performance_model = nb.load_ensemble(performance_dir)

    configspace_path = os.path.join(current_dir, "configspace.json")
    configspace = _load_configspace(configspace_path, seed=seed)

    # Keep a min-heap of size k with entries: (score, sample_index, genotype_json)
    heap: List[Tuple[float, int, Dict[str, Any]]] = []

    for i in range(num_samples):
        cfg = configspace.sample_configuration()
        pred = performance_model.predict(
            config=cfg,
            representation="configspace",
            with_noise=with_noise,
        )
        score = _to_float(pred)

        cfg_dict = cfg.get_dictionary() if hasattr(cfg, "get_dictionary") else dict(cfg)
        genotype = _config_dict_to_genotype(cfg_dict)
        genotype_json = _genotype_to_jsonable(genotype)

        entry = (score, i, genotype_json)
        if len(heap) < k:
            heapq.heappush(heap, entry)
        else:
            if score > heap[0][0]:
                heapq.heapreplace(heap, entry)

    # Sort descending by score
    heap.sort(key=lambda t: t[0], reverse=True)
    results: List[Dict[str, Any]] = []
    for rank, (score, sample_idx, genotype_json) in enumerate(heap, start=1):
        results.append(
            {
                "rank": rank,
                "predicted_accuracy": score,
                "sample_index": sample_idx,
                "genotype": genotype_json,
            }
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sample architectures and return top-k by predicted accuracy (NAS-Bench-301 surrogate)."
    )
    parser.add_argument("--k", type=int, default=10, help="Number of top architectures to return")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5000,
        help="How many architectures to sample from ConfigSpace",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling")
    parser.add_argument(
        "--version",
        type=str,
        default="0.9",
        choices=["0.9", "1.0"],
        help="Which NB301 surrogate version to use",
    )
    parser.add_argument(
        "--with-noise",
        action="store_true",
        help="Use noisy predictions (default: mean/denoised prediction)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional path to write JSON results",
    )

    args = parser.parse_args()

    if args.k > args.num_samples:
        parser.error(f"--k ({args.k}) cannot be greater than --num-samples ({args.num_samples})")

    results = topk_architectures(
        version=args.version,
        k=args.k,
        num_samples=args.num_samples,
        seed=args.seed,
        with_noise=args.with_noise,
    )

    for row in results:
        g = row["genotype"]
        g_str = (
            "Genotype("
            f"normal={[(op, int(node)) for op, node in g['normal']]}, "
            f"normal_concat={g['normal_concat']}, "
            f"reduce={[(op, int(node)) for op, node in g['reduce']]}, "
            f"reduce_concat={g['reduce_concat']}"
            ")"
        )
        print(f"acc_pred={row['predicted_accuracy']:.6f}\n{g_str}\n")

    if args.out:
        out_path = args.out
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, sort_keys=True)
        print(f"Wrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
