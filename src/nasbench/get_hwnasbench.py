from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
from pathlib import Path

try:
    from typing import Literal
except ImportError:  # Python 3.7
    from typing_extensions import Literal

import numpy as np


MetricKind = Literal["latency", "energy", "peak_power", "avg_power", "inference_time", "accuracy"]
AggKind = Literal["mean", "max", "product"]
ModeKind = Literal["per_device", "aggregate"]
AccSplit = Literal["train", "valid", "test"]


def _unit_for(kind: str) -> str:
    if kind == "latency":
        return "ms"
    if kind == "energy":
        return "mJ"
    if kind in ("peak_power", "avg_power"):
        return "W"
    if kind == "inference_time":
        return "ms"
    if kind == "accuracy":
        return "%"
    return ""


def _load_hw_api(hwbench_path: str, search_space: str):
    from nasbench.hw_nas_bench import HWNASBenchAPI as HWAPI

    if not os.path.exists(hwbench_path):
        raise FileNotFoundError(
            f"HW-NAS-Bench dataset not found: {hwbench_path}. "
            "Download HW-NAS-Bench-v1_0.pickle and pass its path via --hwbench-path "
            "(or place it at the default path shown above)."
        )
    return HWAPI(hwbench_path, search_space=search_space)


def _load_nas201_api(nas201_path: str, verbose: bool = False):
    from nasbench.nasbench201.api_201 import NASBench201API

    if not os.path.exists(nas201_path):
        raise FileNotFoundError(
            f"NAS-Bench-201 file not found: {nas201_path}. "
            "Download NAS-Bench-201-v1_1-096897.pth and pass its path via --nas201-path."
        )
    return NASBench201API(nas201_path, verbose=verbose)


_HW_SUFFIXES = (
    "_latency",
    "_energy",
    "_peak_power",
    "_avg_power",
    "_inference_time",
)


def _available_hw_metric_keys(hw_api, dataset: str) -> List[str]:
    space = hw_api.search_space
    if space == "nasbench201":
        data = hw_api.HW_metrics[space][dataset]
    elif space == "fbnet":
        # FBNet stores lookup tables per metric at the top-level (no dataset nesting).
        data = hw_api.HW_metrics[space]
    else:
        raise ValueError(f"Unsupported search_space: {space}")
    keys: List[str] = []
    for key in data.keys():
        if any(key.endswith(s) for s in _HW_SUFFIXES):
            keys.append(key)
    return sorted(keys)


def _available_devices(hw_api, dataset: str, metric: str) -> List[str]:
    suffix = f"_{metric}"
    keys = _available_hw_metric_keys(hw_api, dataset)
    devices = []
    for key in keys:
        if key.endswith(suffix):
            devices.append(key[: -len(suffix)])
    return sorted(devices)


def _get_hw_metric_array(hw_api, dataset: str, device: str, metric: str) -> np.ndarray:
    space = hw_api.search_space
    if space != "nasbench201":
        raise ValueError(
            "_get_hw_metric_array is only valid for search_space='nasbench201'. "
            "For FBNet, metrics are stored as per-op lookup tables, so you must use sampling/candidates."
        )
    key = f"{device}_{metric}"
    data = hw_api.HW_metrics[space][dataset]
    # print(data)
    # exit()
    if key not in data:
        available = _available_devices(hw_api, dataset, metric)
        raise KeyError(
            f"Metric '{key}' not available for dataset='{dataset}'. "
            f"Available {metric} devices: {available}"
        )
    return np.asarray(data[key], dtype=np.float64)

def _try_get_arch_str(hw_api, arch: Union[int, Sequence[int]], dataset: str) -> Optional[str]:
    """Best-effort: return architecture string.

    - nasbench201: arch is an int index
    - fbnet: arch is a list of 22 op indices
    """
    try:
        config = hw_api.get_net_config(arch, dataset)
        if isinstance(config, dict) and "arch_str" in config:
            return str(config["arch_str"])
    except Exception:
        return None
    return None


def _sample_fbnet_architectures(num_samples: int, seed: Optional[int] = None) -> List[List[int]]:
    if num_samples <= 0:
        raise ValueError("fbnet num_samples must be > 0")
    rng = np.random.default_rng(seed)
    # FBNet: 22 blocks, each op index in [0..8]
    samples = rng.integers(low=0, high=9, size=(num_samples, 22), dtype=np.int64)
    return samples.tolist()


def _topk_hw_fbnet(
    hw_api,
    dataset: str,
    kind: str,
    topk: int,
    device_list: Sequence[str],
    mode: ModeKind,
    agg: AggKind,
    candidates: Sequence[Sequence[int]],
) -> Dict[str, List[RankedArch]]:
    if hw_api.search_space != "fbnet":
        raise ValueError("_topk_hw_fbnet only supports search_space='fbnet'")
    if not candidates:
        raise ValueError("FBNet candidate set is empty")

    unit = _unit_for(kind)

    def score_for(device: str) -> np.ndarray:
        metric_key = f"{device}_{kind}"
        lookup_table = hw_api.HW_metrics["fbnet"].get(metric_key)
        if lookup_table is None:
            raise KeyError(f"FBNet lookup table not found for metric '{metric_key}'")
        from nasbench.hw_nas_bench.hw_nas_bench_api import fbnet_get_metrics

        scores = np.empty(len(candidates), dtype=np.float64)
        for i, arch in enumerate(candidates):
            scores[i] = float(fbnet_get_metrics(list(arch), dataset, lookup_table))
        return scores

    if mode == "per_device":
        results: Dict[str, List[RankedArch]] = {}
        for device in device_list:
            scores = score_for(device)
            top_idx, top_vals = _topk_from_scores(scores, topk, smallest=True)
            items: List[RankedArch] = []
            for rank_i, val in zip(top_idx, top_vals):
                arch = list(map(int, candidates[int(rank_i)]))
                arch_str = _try_get_arch_str(hw_api, arch, dataset)
                extra: Dict[str, Union[str, float, int]] = {
                    "device": device,
                    "metric": f"{device}_{kind}",
                    "unit": unit,
                    "op_idx_list": arch,
                }
                if arch_str is not None:
                    extra["arch_str"] = arch_str
                items.append(RankedArch(int(rank_i), float(val), extra))
            results[device] = items
        return results

    if mode != "aggregate":
        raise ValueError(f"Unknown mode: {mode}")

    stacked = np.stack([score_for(d) for d in device_list], axis=1)  # [N, D]
    if agg == "mean":
        score = stacked.mean(axis=1)
    elif agg == "max":
        score = stacked.max(axis=1)
    elif agg == "product":
        score = stacked.prod(axis=1)
    else:
        raise ValueError(f"Unknown agg: {agg}")

    top_idx, top_vals = _topk_from_scores(score, topk, smallest=True)
    label = f"aggregate({agg})"
    items2: List[RankedArch] = []
    for rank_i, val in zip(top_idx, top_vals):
        arch = list(map(int, candidates[int(rank_i)]))
        arch_str = _try_get_arch_str(hw_api, arch, dataset)
        extra2: Dict[str, Union[str, float, int]] = {
            "devices": ",".join(device_list),
            "agg": agg,
            "kind": kind,
            "metric": f"aggregate_{agg}_{kind}",
            "unit": unit,
            "op_idx_list": arch,
        }
        if arch_str is not None:
            extra2["arch_str"] = arch_str
        items2.append(RankedArch(int(rank_i), float(val), extra2))
    return {label: items2}


def _try_get_accuracy(
    nas201_api,
    arch_index: int,
    dataset: str,
    hp: str,
    split: AccSplit,
    is_random: bool,
) -> Optional[float]:
    """Best-effort: return accuracy value (%) for an architecture."""
    try:
        # main.py uses NAS-Bench-201-style split names, while NAS-Bench-201 API
        # typically exposes {train, valid, test}-accuracy.
        split_name = str(split)
        if split_name in ("x-valid", "xvalid"):
            split_name = "valid"
        elif split_name in ("x-test", "xtest", "ori-test", "oritest"):
            split_name = "test"
        key = f"{split_name}-accuracy"
        info = nas201_api.get_more_info(int(arch_index), dataset, hp=hp, is_random=is_random)
        return float(info[key])
    except Exception:
        return None


def _topk_from_scores(scores: np.ndarray, k: int, smallest: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    if k <= 0:
        raise ValueError("k must be > 0")
    if scores.ndim != 1:
        raise ValueError("scores must be a 1D array")
    k = min(k, scores.shape[0])
    order = np.argsort(scores) if smallest else np.argsort(-scores)
    idx = order[:k]
    return idx, scores[idx]


@dataclass(frozen=True)
class RankedArch:
    arch_index: int
    score: float
    extra: Dict[str, Union[str, float, int]]


def topk_hw(
    hw_api,
    dataset: str,
    kind: str,
    topk: int = 10,
    devices: Union[str, Sequence[str]] = "all",
    mode: ModeKind = "per_device",
    agg: AggKind = "mean",
) -> Dict[str, List[RankedArch]]:
    """Top-k architectures by HW metric.

    Note: This function is for NAS-Bench-201 indices (search_space='nasbench201').
    FBNet top-k is handled via sampling in the CLI.

    - mode='per_device': returns {device -> topk list}
    - mode='aggregate': returns {aggregate(label) -> topk list}
    """
    if hw_api.search_space != "nasbench201":
        raise ValueError("topk_hw only supports search_space='nasbench201'.")

    if isinstance(devices, str):
        if devices == "all":
            device_list = _available_devices(hw_api, dataset, kind)
        else:
            device_list = [d.strip() for d in devices.split(",") if d.strip()]
    else:
        device_list = list(devices)

    if not device_list:
        available_keys = _available_hw_metric_keys(hw_api, dataset)
        raise KeyError(
            f"No devices found for metric='{kind}' on dataset='{dataset}'. "
            f"Available metric keys: {available_keys}"
        )

    if mode == "per_device":
        results: Dict[str, List[RankedArch]] = {}
        unit = _unit_for(kind)
        for device in device_list:
            arr = _get_hw_metric_array(hw_api, dataset, device, kind)
            idx, vals = _topk_from_scores(arr, topk, smallest=True)
            ranked: List[RankedArch] = []
            for i, v in zip(idx, vals):
                arch_str = _try_get_arch_str(hw_api, int(i), dataset)
                extra: Dict[str, Union[str, float, int]] = {
                    "device": device,
                    "metric": f"{device}_{kind}",
                    "unit": unit,
                }
                if arch_str is not None:
                    extra["arch_str"] = arch_str
                ranked.append(RankedArch(int(i), float(v), extra))
            results[device] = ranked
        return results

    if mode != "aggregate":
        raise ValueError(f"Unknown mode: {mode}")

    arrays = [_get_hw_metric_array(hw_api, dataset, device, kind) for device in device_list]
    stacked = np.stack(arrays, axis=1)  # [N, D]
    if agg == "mean":
        score = stacked.mean(axis=1)
    elif agg == "max":
        score = stacked.max(axis=1)
    elif agg == "product":
        score = stacked.prod(axis=1)
    else:
        raise ValueError(f"Unknown agg: {agg}")

    idx, vals = _topk_from_scores(score, topk, smallest=True)
    label = f"aggregate({agg})"
    ranked2: List[RankedArch] = []
    unit2 = _unit_for(kind)
    for i, v in zip(idx, vals):
        arch_str = _try_get_arch_str(hw_api, int(i), dataset)
        extra2: Dict[str, Union[str, float, int]] = {
            "devices": ",".join(device_list),
            "agg": agg,
            "kind": kind,
            "metric": f"aggregate_{agg}_{kind}",
            "unit": unit2,
        }
        if arch_str is not None:
            extra2["arch_str"] = arch_str
        ranked2.append(RankedArch(int(i), float(v), extra2))
    return {label: ranked2}


def topk_accuracy(
    nas201_api,
    dataset: str,
    topk: int = 10,
    hp: str = "200",
    split: AccSplit = "test",
    arch_indices: Optional[Iterable[int]] = None,
    is_random: bool = False,
) -> List[RankedArch]:
    """Top-k architectures by accuracy (NAS-Bench-201).

    split: train/valid/test -> uses keys train-accuracy / valid-accuracy / test-accuracy
    is_random=False averages across trials (recommended)
    """
    split_name = str(split)
    if split_name in ("x-valid", "xvalid"):
        split_name = "valid"
    elif split_name in ("x-test", "xtest", "ori-test", "oritest"):
        split_name = "test"
    preferred_key = f"{split_name}-accuracy"
    if arch_indices is None:
        arch_indices = range(len(nas201_api))

    indices = list(arch_indices)
    scores = np.empty(len(indices), dtype=np.float64)
    key = preferred_key
    for j, idx in enumerate(indices):
        info = nas201_api.get_more_info(idx, dataset, hp=hp, is_random=is_random)
        if preferred_key not in info:
            if split_name == "valid" and "test-accuracy" in info:
                key = "test-accuracy"
            elif "test-accuracy" in info:
                key = "test-accuracy"
            elif "valid-accuracy" in info:
                key = "valid-accuracy"
            elif "train-accuracy" in info:
                key = "train-accuracy"
            else:
                raise KeyError(
                    f"No accuracy key found for split='{split_name}'. Available keys: {sorted(info.keys())}"
                )
        else:
            key = preferred_key
        scores[j] = float(info[key])

    top_local, vals = _topk_from_scores(scores, topk, smallest=False)
    return [
        RankedArch(
            arch_index=int(indices[int(i)]),
            score=float(v),
            extra={"dataset": dataset, 
                   "hp": hp, 
                   "arch_index": int(indices[int(i)]),
                   "split": split_name, 
                   "metric": key, 
                   "value": float(v),
                   "unit": _unit_for("accuracy"),
                   "is_random": is_random,
                   "arch_str": _try_get_arch_str(nas201_api, int(indices[int(i)]), dataset)
                   },
        )
        for i, v in zip(top_local, vals)
    ]


def _as_jsonable(obj):
    if isinstance(obj, RankedArch):
        return {"arch_index": obj.arch_index, "score": obj.score, **obj.extra}
    if isinstance(obj, dict):
        return {k: _as_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_as_jsonable(x) for x in obj]
    return obj

HWNasBenchData = _load_hw_api("src/nasbench/hw_nas_bench/HW-NAS-Bench-v1_0.pickle", search_space="nasbench201").HW_metrics["nasbench201"]['cifar10']
HW_KEY_METRICS = HWNasBenchData.keys()

def _get_hw_metric_cifar10(arch_index: int, arch_str: str):
    _arch_str = HWNasBenchData['config'][arch_index]['arch_str']
    
    if _arch_str != arch_str:
        raise ValueError(f"arch_index {arch_index} does not match arch_str {arch_str} (index {_arch_str})")
    metrics = {}
    for HW_KEY in HW_KEY_METRICS:
        metric_value = HWNasBenchData[HW_KEY][arch_index]
        # print(f"{HW_KEY}: {metric_value}")
        metrics[HW_KEY] = metric_value
    metrics['arch_index'] = arch_index
    return metrics

def get_top_arch(args) -> int:

    selected_devices: Union[str, Sequence[str]]
    devices_arg = getattr(args, "devices", "all")
    selected_devices = args.device if getattr(args, "device", None) is not None else devices_arg

    out_dir = Path(getattr(args, "results_dir", "results")) / "hwnasbench"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Any non-accuracy metric is treated as a HW metric.
    if args.metric != "accuracy":
        hwbench_path = getattr(args, "hwbench_path", "src/nasbench/hw_nas_bench/HW-NAS-Bench-v1_0.pickle")
        hw_api = _load_hw_api(hwbench_path, args.search_space)
        if args.show_accuracy and hw_api.search_space != "nasbench201":
            raise ValueError("--show-accuracy is only supported for search_space='nasbench201'.")

        if hw_api.search_space == "fbnet":
            # Resolve devices
            if isinstance(selected_devices, str):
                if selected_devices == "all":
                    device_list = _available_devices(hw_api, args.dataset, args.metric)
                else:
                    device_list = [d.strip() for d in selected_devices.split(",") if d.strip()]
            else:
                device_list = list(selected_devices)

            candidates = _sample_fbnet_architectures(args.fbnet_samples, seed=args.fbnet_seed)
            out = _topk_hw_fbnet(
                hw_api,
                dataset=args.dataset,
                kind=args.metric,
                topk=args.k,
                device_list=device_list,
                mode=args.mode,  # type: ignore[arg-type]
                agg=args.agg,  # type: ignore[arg-type]
                candidates=candidates,
            )
        else:
            out = topk_hw(
                hw_api,
                dataset=args.dataset,
                kind=args.metric,
                topk=args.k,
                devices=selected_devices,
                mode=args.mode,  # type: ignore[arg-type]
                agg=args.agg,  # type: ignore[arg-type]
            )

        if args.show_accuracy:
            nas201_path = getattr(args, "nas201_path", "src/nasbench/nasbench201/NAS-Bench-201-v1_1-096897.pth")
            nas201_api = _load_nas201_api(nas201_path, verbose=False)
            enriched: Dict[str, List[RankedArch]] = {}
            for group, items in out.items():
                new_items: List[RankedArch] = []
                for item in items:
                    acc = _try_get_accuracy(
                        nas201_api,
                        item.arch_index,
                        dataset=args.dataset,
                        hp=args.hp,
                        split=args.split,
                        is_random=args.is_random,
                    )
                    if acc is None:
                        new_items.append(item)
                    else:
                        extra = dict(item.extra)
                        extra["accuracy_metric"] = f"{args.split}-accuracy"
                        extra["accuracy"] = float(acc)
                        extra["accuracy_unit"] = _unit_for("accuracy")
                        extra["accuracy_hp"] = args.hp
                        new_items.append(RankedArch(item.arch_index, item.score, extra))
                enriched[group] = new_items
            out = enriched
        

        if args.json:
            print(json.dumps(_as_jsonable(out), indent=2))
        else:
            for group, items in out.items():
                print(f"\n==> {args.metric} top-{args.k} @ {args.dataset} :: {group}")
                for rank, item in enumerate(items, start=1):
                    arch_str = item.extra.get("arch_str")
                    metric_key = item.extra.get("metric")
                    acc_val = item.extra.get("accuracy")
                    acc_key = item.extra.get("accuracy_metric")
                    unit = item.extra.get("unit")
                    acc_unit = item.extra.get("accuracy_unit")
                    op_idx_list = item.extra.get("op_idx_list")
                    parts = [
                        (
                            f"{rank:2d}. arch_id={item.arch_index:5d}"
                            if isinstance(op_idx_list, list)
                            else f"{rank:2d}. arch={item.arch_index:5d}"
                        ),
                        f"metric={metric_key}" if isinstance(metric_key, str) else None,
                        f"value={item.score:.6g}{unit}" if isinstance(unit, str) else f"value={item.score:.6g}",
                        (
                            f"{acc_key}={float(acc_val):.4f}{acc_unit}"
                            if isinstance(acc_key, str) and isinstance(acc_val, (int, float)) and isinstance(acc_unit, str)
                            else None
                        ),
                        f"op_idx_list={op_idx_list}" if isinstance(op_idx_list, list) else None,
                        f"arch_str={arch_str}" if isinstance(arch_str, str) else None,
                    ]
                    print("  ".join([p for p in parts if p is not None]))
        
        

        return 0

    # accuracy
    nas201_path = getattr(args, "nas201_path", "src/nasbench/nasbench201/NAS-Bench-201-v1_1-096897.pth")
    nas201_api = _load_nas201_api(nas201_path, verbose=False)
    ranked = topk_accuracy(
        nas201_api,
        dataset=args.dataset,
        topk=getattr(args, "topk", args.k),
        hp=args.hp,
        split=args.split,  # type: ignore[arg-type]
        is_random=getattr(args, "random_trial", args.is_random),
    )

    if args.json:
        print(json.dumps(_as_jsonable(ranked), indent=2))
    else:
        metric_key = ranked[0].extra.get("metric") if ranked else f"{args.split}-accuracy"
        print(f"\n==> accuracy({args.split}) top-{args.k} @ {args.dataset} (hp={args.hp}, metric={metric_key})")
        for rank, item in enumerate(ranked, start=1):
            arch_str = item.extra.get("arch_str")
            metric_k = item.extra.get("metric")
            unit = item.extra.get("unit")
            parts2 = [
                f"{rank:2d}. arch={item.arch_index:5d}",
                f"metric={metric_k}" if isinstance(metric_k, str) else None,
                f"value={item.score:.4f}{unit}" if isinstance(unit, str) else f"value={item.score:.4f}%",
                f"arch_str={arch_str}" if isinstance(arch_str, str) else None,
            ]
            print("  ".join([p for p in parts2 if p is not None]))
    

            out_path = out_dir / f"arch_index-{item.arch_index:05d}.json"
            metrics = _get_hw_metric_cifar10(item.arch_index, arch_str)
            metrics.update(item.extra)

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
                            
    return 0

