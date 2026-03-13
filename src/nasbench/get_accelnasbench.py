import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


try:
    from typing import Literal
except ImportError:  # Python 3.7
    from typing_extensions import Literal

import numpy as np

import nasbench.accel_nasbench.accelnb as anb
from nasbench.accel_nasbench.configurationspaces.searchspaces import EfficientNetSS


def _as_list(samples):
    return samples if isinstance(samples, list) else [samples]


@dataclass(frozen=True)
class TopResult:
    config: Any
    accuracy: float
    throughput: float
    latency: float


SortBy = Literal["accuracy", "throughput", "latency"]


def get_top_architectures(
    *,
    seed: int = 3,
    num_candidates: int = 200,
    top_k: int = 10,
    sort_by: SortBy = "accuracy",
    throughput_device: str = "tpuv2",
    latency_device: str = "zcu102",
    model: str = "xgb",
    search_space: Any = None,
) -> List[TopResult]:
    """Return top-k architectures ranked by `sort_by`.

    Ranking rules:
    - `accuracy`: higher is better
    - `throughput`: higher is better
    - `latency`: lower is better
    """

    if search_space is None:
        search_space = EfficientNetSS()

    # Accuracy surrogate (default metric/device)
    acc_model = anb.ANBEnsemble(model, seed=seed).load_ensemble()

    # Throughput surrogate
    thr_model = anb.ANBEnsemble(
        model, device=throughput_device, metric="throughput", seed=seed
    ).load_ensemble()

    # Latency surrogate
    lat_model = anb.ANBEnsemble(
        model, device=latency_device, metric="latency", seed=seed
    ).load_ensemble()

    candidates: List[Any] = _as_list(search_space.random_sample(num_candidates))

    mean_acc, _ = acc_model.query(candidates)
    mean_thr, _ = thr_model.query(candidates)
    mean_lat, _ = lat_model.query(candidates)

    mean_acc = np.asarray(mean_acc, dtype=float)
    mean_thr = np.asarray(mean_thr, dtype=float)
    mean_lat = np.asarray(mean_lat, dtype=float)

    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    top_k = min(top_k, len(candidates))

    if sort_by == "latency":
        order = np.argsort(mean_lat)  # ascending
    elif sort_by == "accuracy":
        order = np.argsort(-mean_acc)  # descending
    elif sort_by == "throughput":
        order = np.argsort(-mean_thr)  # descending
    else:
        raise ValueError(f"Unsupported sort_by: {sort_by}")

    results: List[TopResult] = []
    for idx in order[:top_k]:
        i = int(idx)
        results.append(
            TopResult(
                config=candidates[i],
                accuracy=float(mean_acc[i]),
                throughput=float(mean_thr[i]),
                latency=float(mean_lat[i]),
            )
        )

    return results


def _print_topk(results: List[TopResult], *, sort_by: SortBy):
    print(f"Top {len(results)} architectures (sorted by {sort_by}):")
    print("rank\taccuracy\tthroughput\tlatency")
    for rank, r in enumerate(results, start=1):
        print(f"{rank}\t{r.accuracy:.6f}\t{r.throughput:.6f}\t{r.latency:.6f}")
        print(r.config)


def get_top_arch(args):
    results = get_top_architectures(
        seed=args.seed,
        num_candidates=args.num_candidates,
        top_k=args.top_k,
        sort_by=args.sort_by,
        throughput_device=args.throughput_device,
        latency_device=args.latency_device,
        model=args.model,
    )

    _print_topk(results, sort_by=args.sort_by)

