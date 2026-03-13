import argparse
import heapq
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from ConfigSpace.read_and_write import json as cs_json

import nasbench.nasbench301 as nb
from nasbench.nasbench301.representations import Genotype


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

    # Some Figshare bundles have historically reused v1.0 folder names even when
    # requested with a different version. In that case, keep a compat symlink so
    # the rest of the code can load from the expected path.
    expected_perf_dir = os.path.join(models_dir, f"xgb_v{version}")
    if not os.path.exists(expected_perf_dir):
        fallback_dir = os.path.join(models_dir, "xgb_v1.0")
        if os.path.isdir(fallback_dir):
            try:
                os.symlink("xgb_v1.0", expected_perf_dir)
            except FileExistsError:
                pass
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


def get_top_arch(args) -> List[Dict[str, Any]]:
    if args.k < 1:
        raise ValueError("k must be >= 1")
    if args.num_samples < 1:
        raise ValueError("num_samples must be >= 1")
    if args.k > args.num_samples:
        raise ValueError(f"k ({args.k}) cannot be greater than num_samples ({args.num_samples})")

    nabench301_path = "src/nasbench/nasbench301"

    models_dir = _ensure_models(version=args.version, download_dir=nabench301_path)
    performance_dir = os.path.join(models_dir, f"xgb_v{args.version}")

    runtime_dir = os.path.join(models_dir, f"lgb_runtime_v1.0")
    runtime_model = nb.load_ensemble(runtime_dir)

    performance_model = nb.load_ensemble(performance_dir)

    configspace_path = os.path.join(nabench301_path, "configspace.json")
    configspace = _load_configspace(configspace_path, seed=args.seed)

    # Keep a min-heap of size k with entries: (score, sample_index, genotype_json)
    heap: List[Tuple[float, int, Dict[str, Any]]] = []

    for i in range(args.num_samples):
        cfg = configspace.sample_configuration()
        pred = performance_model.predict(
            config=cfg,
            representation="configspace",
            with_noise=args.with_noise,
        )

        runtime = runtime_model.predict(
            config=cfg,
            representation="configspace"
        )

        score = _to_float(pred)

        cfg_dict = cfg.get_dictionary() if hasattr(cfg, "get_dictionary") else dict(cfg)

        genotype = _config_dict_to_genotype(cfg_dict)
        genotype_json = _genotype_to_jsonable(genotype)

        entry = (score, i, genotype_json, runtime)
        if len(heap) < args.k:
            heapq.heappush(heap, entry)
        else:
            if score > heap[0][0]:
                heapq.heapreplace(heap, entry)

    # Write results to disk (mirrors nasbench101: results/<benchmark>/<hash>.json)
    out_dir = Path(getattr(args, "results_dir", "results")) / "nasbench301"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Sort descending by score
    heap.sort(key=lambda t: t[0], reverse=True)
    results: List[Dict[str, Any]] = []
    for rank, (score, sample_idx, genotype_json, runtime) in enumerate(heap, start=1):
        payload = {
                "rank": rank,
                "predicted_accuracy": score,
                "sample_index": sample_idx,
                "genotype": genotype_json,
                "predicted_runtime": runtime,
            }
        results.append(payload)
        out_path = out_dir / f"sample_index-{sample_idx}.json"
        metric_path = out_dir / f"metric-{sample_idx}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

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
    return results

