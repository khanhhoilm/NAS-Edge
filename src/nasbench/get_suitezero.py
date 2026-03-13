import argparse
import json
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

from nasbench.nas_bench_suite_zero.naslib.search_spaces import (
    NasBench101SearchSpace,
    NasBench201SearchSpace,
    NasBench301SearchSpace,
    TransBench101SearchSpaceMacro,
    TransBench101SearchSpaceMicro,
)
from nasbench.nas_bench_suite_zero.naslib.search_spaces.core.query_metrics import Metric
from nasbench.nas_bench_suite_zero.naslib.utils import get_dataset_api
from tqdm import tqdm
import random


SEARCH_SPACES = {
    "nasbench101": NasBench101SearchSpace,
    "nasbench201": NasBench201SearchSpace,
    "nasbench301": NasBench301SearchSpace,
    "transbench101_micro": TransBench101SearchSpaceMicro,
    "transbench101_macro": TransBench101SearchSpaceMacro,
}

TASKS = {
    "nasbench101": ["cifar10"],
    "nasbench201": ["cifar10", "cifar100", "ImageNet16-120"],
    "nasbench301": ["cifar10"],
    "transbench101_micro": [
        "class_scene",
        "class_object",
        "jigsaw",
        "room_layout",
        "segmentsemantic",
        "normal",
        "autoencoder",
    ],
    "transbench101_macro": [
        "class_scene",
        "class_object",
        "jigsaw",
        "room_layout",
        "segmentsemantic",
        "normal",
        "autoencoder",
    ],
}

SORTABLE_METRICS = {
    "VAL_ACCURACY": (Metric.VAL_ACCURACY, True),  # (metric, higher_is_better)
    "TEST_ACCURACY": (Metric.TEST_ACCURACY, True),
    "TRAIN_ACCURACY": (Metric.TRAIN_ACCURACY, True),
    "TRAIN_TIME": (Metric.TRAIN_TIME, False),
    "VAL_LOSS": (Metric.VAL_LOSS, False),
    "TEST_LOSS": (Metric.TEST_LOSS, False),
    "TRAIN_LOSS": (Metric.TRAIN_LOSS, False),
    "LATENCY": (Metric.LATENCY, False),
    "PARAMETERS": (Metric.PARAMETERS, False),
    "FLOPS": (Metric.FLOPS, False),
}


METRIC_HIGHER_IS_BETTER = {
    "val_acc": True,
    "test_acc": True,
    "train_time": False,
    "latency": False,
    "parameters": False,
    "flops": False,
}


def make_hashable(obj):
    """Convert unhashable objects to hashable equivalents"""
    if isinstance(obj, list):
        return tuple(make_hashable(item) for item in obj)
    elif isinstance(obj, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
    else:
        return obj


def safe_get_arch_id(search_space):
    """Best-effort architecture identifier.

    Some NASLib search spaces may not have a stable/working `get_hash()` in every
    configuration. This helper tries `get_hash()` first, then falls back to common
    attributes.
    """
    try:
        arch_id = search_space.get_hash()
        if arch_id is not None:
            return arch_id
    except Exception:
        pass

    for attr in ("compact", "spec", "op_indices"):
        try:
            arch_id = getattr(search_space, attr, None)
            if arch_id is not None:
                return arch_id
        except Exception:
            continue

    return None


def get_on_nb101(args):
    """Get top architectures from NASBench101"""
    print(f'\n{"="*80}')
    print(f'NASBench101 - Top {args.k} Architectures (Dataset: {args.dataset})')
    print(f'{"="*80}\n')
    
    # Initialize search space and dataset API
    dataset_api = get_dataset_api('nasbench101', args.dataset)
    
    print(f'Loading architectures from NASBench101...')
    
    # Get all architecture hashes
    search_space = NasBench101SearchSpace()
    arch_list = list(search_space.get_arch_iterator(dataset_api))[:100]
    
    # For NB101 we can query directly from the SuiteZero wrapper without
    # instantiating/mutating the NASLib graph, similar to the NB201 path.
    # The data file only contains results at epoch=108 (final) for 3 runs.
    nb101_data = dataset_api["nb101_data"]

    # Query architectures sequentially
    results = []
    
    for arch_hash in tqdm(arch_list, desc="Querying architectures"):
        try:
            fixed, computed = nb101_data.get_metrics_from_hash(arch_hash)

            matrix = fixed.get("module_adjacency")
            ops = fixed.get("module_operations")

            if hasattr(matrix, "tolist"):
                matrix = matrix.tolist()
            if ops is not None:
                ops = list(ops)

            # computed[108] is a list of 3 runs. Accuracies are stored as [0,1].
            runs = computed.get(108) or []
            if len(runs) == 0:
                continue

            test_acc = sum(r["final_test_accuracy"] for r in runs) / len(runs) * 100
            val_acc = sum(r["final_validation_accuracy"] for r in runs) / len(runs) * 100
            train_time = sum(r["final_training_time"] for r in runs) / len(runs)
            params = fixed.get("trainable_parameters")

            results.append({
                'architecture': {'matrix': matrix, 'ops': ops},
                'test_acc': test_acc,
                'val_acc': val_acc,
                'train_time': train_time,
                # NASBench101 (SuiteZero's nb101_data) does not include inference latency.
                'latency': None,
                'params': params
            })
        except Exception as e:
            print(f"Error querying {arch_hash}: {e}")
    
    metric_key = args.metric
    if metric_key == "latency":
        print("\n⚠️  NASBench101 does not provide inference latency in this benchmark; sorting by train_time instead.\n")
        metric_key = "train_time"

    reverse = METRIC_HIGHER_IS_BETTER.get(metric_key, True)
    results.sort(
        key=lambda x: float("inf") if x.get(metric_key) is None else x[metric_key],
        reverse=reverse,
    )
    
    # Display top K
    print(f'\n🏆 TOP {args.k} ARCHITECTURES BY :\n')
    for i, result in enumerate(results[:args.k], 1):
        print(f'{i:2d}. Architecture (NASBench101):', result["architecture"])
        print(f'    📊 Test Accuracy:  {result["test_acc"]:.4f}%')
        print(f'    📈 Val Accuracy:   {result["val_acc"]:.4f}%')
        print(f'    ⏱️  Train Time:     {result["train_time"]:.2f}s')
        if args.metric == "latency":
            print(f'    🕒 Latency:        N/A')
        print(f'    🔢 Parameters:     {result["params"]:.2f}M')
        print()
    return results[:args.k]


def get_on_nb201(args):
    """Get top architectures from NASBench201"""
    print(f'\n{"="*80}')
    print(f'NASBench201 - Top {args.k} Architectures (Dataset: {args.dataset})')
    print(f'{"="*80}\n')
    
    # Initialize dataset API
    dataset_api = get_dataset_api('nasbench201', args.dataset)
    
    print(f'Collecting all architectures from NASBench201...')
    
    from nasbench.nas_bench_suite_zero.naslib.search_spaces.nasbench201.conversions import (
        convert_op_indices_to_str,
    )
    
    # Get all architectures (as op_indices)
    search_space = NasBench201SearchSpace()
    arch_list = list(search_space.get_arch_iterator(dataset_api))

    # For NB201, repeatedly calling `set_spec()` on the same NASLib object can
    # mutate/discretize the graph in ways that break later conversions (e.g.
    # turning the expected cell graph into an `Identity` op). Since all queried
    # metrics are already present in `dataset_api['nb201_data']`, we can extract
    # them directly and avoid the mutation path entirely.
    dataset_key = "cifar10-valid" if args.dataset == "cifar10" else args.dataset
    
    # Query architectures sequentially
    results = []
    
    for op_indices in tqdm(arch_list, desc="Querying architectures"):
        try:
            arch_str = convert_op_indices_to_str(op_indices)
            query_results = dataset_api["nb201_data"][arch_str][dataset_key]

            # NB201 stores accuracies as percentages already.
            test_acc = query_results["eval_acc1es"][-1]
            val_acc = query_results["eval_acc1es"][-1]
            train_time = query_results["cost_info"]["train_time"]
            latency = query_results["cost_info"].get("latency")
            params = query_results["cost_info"]["params"]
            
            results.append({
                'arch_str': arch_str,
                'test_acc': test_acc,
                'val_acc': val_acc,
                'train_time': train_time,
                'latency': latency,
                'params': params
            })
        except Exception as e:
            print(f"Error querying {op_indices}: {e}")
    
    reverse = METRIC_HIGHER_IS_BETTER.get(args.metric, True)
    results.sort(key=lambda x: x[args.metric], reverse=reverse)
    
    # Display top K
    print(f'\nTOP {args.k} ARCHITECTURES BY {args.metric.upper()}:\n')
    for i, result in enumerate(results[:args.k], 1):
        print(f'    Architecture:    {result["arch_str"]}')
        print(f'    Test Accuracy:   {result["test_acc"]:.4f}%')
        print(f'    📈 Val Accuracy:   {result["val_acc"]:.4f}%')
        print(f'    ⏱️  Train Time:     {result["train_time"]:.2f}s')
        if result.get("latency") is not None:
            print(f'    🕒 Latency:        {result["latency"]:.6f}s')
        print(f'    🔢 Parameters:     {result["params"]:.2f}M')
        print()
    return results[:args.k]


def get_on_nb301(args):
    """Get top architectures from NASBench301 via random sampling"""
    print(f'\n{"="*80}')
    print(f'NASBench301 - Top {args.k} from {args.num_samples} Random Samples (Dataset: {args.dataset})')
    print(f'{"="*80}\n')
    
    # Initialize search space and dataset API
    search_space = NasBench301SearchSpace()
    dataset_api = get_dataset_api('nasbench301', args.dataset)

    # IMPORTANT:
    # The NASLib conversion path `compact -> genotype -> naslib` is destructive
    # (it deletes edges). Repeatedly sampling + converting on the same object can
    # therefore lead to invalid architectures and effectively "no progress".
    # For sampling/evaluating via the NB301 surrogates, we can avoid mutating the
    # graph entirely by sampling `compact`, converting `compact -> genotype`, and
    # querying the surrogate models directly.
    search_space.instantiate_model = False

    from nasbench.nas_bench_suite_zero.naslib.search_spaces.nasbench301.conversions import (
        convert_compact_to_genotype,
        make_compact_immutable,
    )

    print(f'Sampling {args.num_samples} unique random architectures...')

    results = []
    sampled_compacts = set()

    pbar = tqdm(total=args.num_samples, desc="Sampling architectures", unit="arch")

    collected = 0
    attempts = 0
    duplicates = 0
    errors = 0
    max_attempts = args.num_samples * 500

    while collected < args.num_samples and attempts < max_attempts:
        attempts += 1
        if attempts % 1000 == 0:
            pbar.set_postfix({
                "attempts": attempts,
                "unique": len(sampled_compacts),
                "dups": duplicates,
                "errors": errors,
            })

        try:
            search_space.sample_random_architecture(dataset_api=dataset_api)
            compact = make_compact_immutable(search_space.compact)
        except Exception:
            errors += 1
            continue

        if compact in sampled_compacts:
            duplicates += 1
            continue

        try:
            genotype = convert_compact_to_genotype(compact)
            val_acc = dataset_api["nb301_model"][0].predict(
                config=genotype, representation="genotype"
            )
            train_time = dataset_api["nb301_model"][1].predict(
                config=genotype, representation="genotype"
            )
        except Exception:
            errors += 1
            continue

        sampled_compacts.add(compact)
        results.append({
            'genotype': str(genotype),
            'val_acc': val_acc,
            'train_time': train_time,
        })

        collected += 1
        pbar.update(1)

    pbar.close()

    if attempts >= max_attempts and collected < args.num_samples:
        print(
            f"\nStopped early after reaching max_attempts={max_attempts}. "
            f"Collected {collected}/{args.num_samples} unique architectures."
        )

    if errors > 0:
        print(f'\nSkipped {errors} architectures due to errors during sampling/prediction')
    if duplicates > 0:
        print(f'Skipped {duplicates} duplicate architectures during sampling')

    if len(results) == 0:
        print('\nNo valid architectures found. Try increasing --num_samples or check the benchmark setup.')
        return []

    reverse = METRIC_HIGHER_IS_BETTER.get(args.metric, True)
    results.sort(key=lambda x: x[args.metric], reverse=reverse)

    print(f'\nTOP {args.k} ARCHITECTURES BY {args.metric.upper()}:\n')
    for i, result in enumerate(results[:args.k], 1):
        print(f'{i:2d}. Genotype: {result["genotype"]}')
        print(f'    Val Accuracy:   {result["val_acc"]:.4f}%')
        print(f'    Train Time:     {result["train_time"]:.2f}s')
        print()

    return results[:args.k]


def get_on_transbench101_micro(args):
    """Get top architectures from TransBench101 Micro via random sampling"""
    print(f'\n{"="*80}')
    print(f'TransBench101-Micro - Top {args.k} from {args.num_samples} Random Samples (Task: {args.dataset})')
    print(f'{"="*80}\n')
    
    # Initialize search space and dataset API
    search_space = TransBench101SearchSpaceMicro()
    dataset_api = get_dataset_api('transbench101_micro', args.dataset)

    if dataset_api is None:
        valid_tasks = TASKS.get('transbench101_micro', [])
        print(
            f"\n❌ Unknown TransBench101 task '{args.dataset}'. "
            f"Valid tasks: {', '.join(valid_tasks)}\n"
            f"Example: python main.py suitezero --search_space transbench101_micro --dataset {valid_tasks[0] if valid_tasks else 'class_scene'} --k {args.k} --num_samples {args.num_samples}"
        )
        return []
    
    print(f'Sampling {args.num_samples} unique random architectures...')
    
    # Collect random architectures
    results = []
    sampled_hashes = set()
    
    pbar = tqdm(total=args.num_samples, desc="Sampling architectures", unit="arch")
    
    i = 0
    attempts = 0
    max_attempts = args.num_samples * 100
    
    while i < args.num_samples and attempts < max_attempts:
        attempts += 1
        
        # Sample random architecture
        search_space.sample_random_architecture(dataset_api=dataset_api)
        arch_hash = safe_get_arch_id(search_space)

        if arch_hash is None:
            continue
        
        # Convert to hashable type
        arch_key = make_hashable(arch_hash)
        
        # Skip if already sampled
        if arch_key in sampled_hashes:
            continue
        
        sampled_hashes.add(arch_key)
        
        # Query metrics
        test_acc = search_space.query(
            metric=Metric.TEST_ACCURACY,
            dataset=args.dataset,
            dataset_api=dataset_api
        )
        
        val_acc = search_space.query(
            metric=Metric.VAL_ACCURACY,
            dataset=args.dataset,
            dataset_api=dataset_api
        )
        
        train_time = search_space.query(
            metric=Metric.TRAIN_TIME,
            dataset=args.dataset,
            dataset_api=dataset_api
        )
        
        results.append({
            'hash': arch_hash,
            'test_acc': test_acc,
            'val_acc': val_acc,
            'train_time': train_time,
        })
        
        i += 1
        pbar.update(1)
    
    pbar.close()
    
    reverse = METRIC_HIGHER_IS_BETTER.get(args.metric, True)
    results.sort(key=lambda x: x[args.metric], reverse=reverse)
    
    # Display top K
    print(f'\n🏆 TOP {args.k} ARCHITECTURES BY {args.metric.upper()}:\n')
    for i, result in enumerate(results[:args.k], 1):
        print(f'{i:2d}. Architecture Hash: {result["hash"]}')
        print(f'    📊 Test Accuracy:  {result["test_acc"]:.4f}%')
        print(f'    📈 Val Accuracy:   {result["val_acc"]:.4f}%')
        print(f'    ⏱️  Train Time:     {result["train_time"]:.2f}s')
        print()
    return results[:args.k]


def get_on_transbench101_macro(args):
    """Get top architectures from TransBench101 Macro via random sampling"""
    print(f'\n{"="*80}')
    print(f'TransBench101-Macro - Top {args.k} from {args.num_samples} Random Samples (Task: {args.dataset})')
    print(f'{"="*80}\n')
    
    # Initialize search space and dataset API
    search_space = TransBench101SearchSpaceMacro()
    dataset_api = get_dataset_api('transbench101_macro', args.dataset)

    if dataset_api is None:
        valid_tasks = TASKS.get('transbench101_macro', [])
        print(
            f"\n❌ Unknown TransBench101 task '{args.dataset}'. "
            f"Valid tasks: {', '.join(valid_tasks)}\n"
            f"Example: python main.py suitezero --search_space transbench101_macro --dataset {valid_tasks[0] if valid_tasks else 'class_scene'} --k {args.k} --num_samples {args.num_samples}"
        )
        return []
    
    print(f'Sampling {args.num_samples} unique random architectures...')
    
    # Collect random architectures
    results = []
    sampled_hashes = set()
    
    pbar = tqdm(total=args.num_samples, desc="Sampling architectures", unit="arch")
    
    i = 0
    attempts = 0
    max_attempts = args.num_samples * 100
    
    while i < args.num_samples and attempts < max_attempts:
        attempts += 1
        
        # Sample random architecture
        search_space.sample_random_architecture(dataset_api=dataset_api)
        arch_hash = safe_get_arch_id(search_space)

        if arch_hash is None:
            continue
        
        # Convert to hashable type
        arch_key = make_hashable(arch_hash)
        
        # Skip if already sampled
        if arch_key in sampled_hashes:
            continue
        
        sampled_hashes.add(arch_key)
        
        # Query metrics
        test_acc = search_space.query(
            metric=Metric.TEST_ACCURACY,
            dataset=args.dataset,
            dataset_api=dataset_api
        )
        
        val_acc = search_space.query(
            metric=Metric.VAL_ACCURACY,
            dataset=args.dataset,
            dataset_api=dataset_api
        )
        
        train_time = search_space.query(
            metric=Metric.TRAIN_TIME,
            dataset=args.dataset,
            dataset_api=dataset_api
        )
        
        results.append({
            'hash': arch_hash,
            'test_acc': test_acc,
            'val_acc': val_acc,
            'train_time': train_time,
        })
        
        i += 1
        pbar.update(1)
    
    pbar.close()
    
    reverse = METRIC_HIGHER_IS_BETTER.get(args.metric, True)
    results.sort(key=lambda x: x[args.metric], reverse=reverse)
    
    # Display top K
    print(f'\n🏆 TOP {args.k} ARCHITECTURES BY TEST ACCURACY:\n')
    for i, result in enumerate(results[:args.k], 1):
        print(f'{i:2d}. Architecture Hash: {result["hash"]}')
        print(f'    📊 Test Accuracy:  {result["test_acc"]:.4f}%')
        print(f'    📈 Val Accuracy:   {result["val_acc"]:.4f}%')
        print(f'    ⏱️  Train Time:     {result["train_time"]:.2f}s')
        print()
    return results[:args.k]



def get_top_arch(args):
    if args.search_space == "nasbench101":
        get_on_nb101(args)
    elif args.search_space == "nasbench201":
        get_on_nb201(args)
    elif args.search_space == "nasbench301":
        get_on_nb301(args)
    elif args.search_space == "transbench101_micro":
        get_on_transbench101_micro(args)
    elif args.search_space == "transbench101_macro":
        get_on_transbench101_macro(args)
    else:
        raise ValueError(f"Unsupported search space: {args.search_space}")

    