#!/usr/bin/env python
"""
Demo script for NAS-Bench-Graph
Automatically queries and compares different GNN architectures
"""
import sys
import os


from nasbench.nas_bench_graph.readbench import light_read
from nasbench.nas_bench_graph.architecture import Arch
import random
from tqdm import tqdm

def demo_random_search(args):
    """Random search demo"""
    print(f'\n{"="*60}')
    print(f'Random Architecture Search - Dataset: {args.dataset.upper()}')
    print(f'{"="*60}\n')
    
    bench = light_read(args.dataset)
    
    link_options = [
        [0,0,0,0], [0,0,0,1], [0,0,1,1], [0,0,1,2],
        [0,0,1,3], [0,1,1,1], [0,1,1,2], [0,1,2,2], [0,1,2,3]
    ]
    
    op_options = ['gat', 'gcn', 'gin', 'cheb', 'sage', 'arma', 'graph', 'fc', 'skip']
    
    best_acc = 0
    best_arch = None
    sampled_hashes = set()  # Track sampled architectures to avoid duplicates
    
    print(f'Sampling {args.num_samples} unique random architectures...\n')
    
    i = 0
    attempts = 0
    max_attempts = args.num_samples * 100  # Prevent infinite loop
    
    pbar = tqdm(total=args.num_samples, desc="Sampling architectures", unit="arch")
    
    while i < args.num_samples and attempts < max_attempts:
        attempts += 1
        links = random.choice(link_options)
        ops = [random.choice(op_options) for _ in range(4)]
        
        arch = Arch(links, ops)
        arch_hash = arch.valid_hash()
        
        # Skip if already sampled
        if arch_hash in sampled_hashes:
            continue
        
        sampled_hashes.add(arch_hash)
        info = bench[arch_hash]
        
        tqdm.write(f'{i+1:2d}. {ops} | Acc: {info["perf"]:.4f}')
        
        if info['perf'] > best_acc:
            best_acc = info['perf']
            best_arch = (links, ops, info)
        
        i += 1
        pbar.update(1)
    
    pbar.close()
    
    print(f'\n{"="*60}')
    print(f'🎯 Best found: Accuracy {best_acc:.4f}')
    print(f'   Links: {best_arch[0]}')
    print(f'   Operations: {best_arch[1]}')
    print(f'{"="*60}\n')

def get_top_arch(args):
    """Get top K architectures with highest accuracy"""
    print(f'\n{"="*60}')
    print(f'Top {args.k} Architectures - Dataset: {args.dataset.upper()}')
    print(f'{"="*60}\n')
    
    # Load benchmark
    print(f'Loading {args.dataset} benchmark...')
    bench = light_read(args.dataset)
    print(f'✓ Loaded {len(bench)} architectures\n')
    
    # Collect all architectures with their performance
    all_results = []
    for arch_hash, info in bench.items():
        all_results.append({
            'hash': arch_hash,
            'test_acc': info['perf'],
            'val_acc': info['valid_perf'],
            'latency': info['latency'],
            'params': info['para']
        })
    
    # Sort by test accuracy
    all_results.sort(key=lambda x: x['test_acc'], reverse=True)
    
    # Display top K
    print(f'🏆 TOP {args.k} ARCHITECTURES BY TEST ACCURACY:\n')
    for i, result in enumerate(all_results[:args.k], 1):
        print(f'{i:2d}. Architecture Hash: {result["hash"]}')
        print(f'    📊 Test Accuracy:  {result["test_acc"]:.6f}')
        print(f'    📈 Val Accuracy:   {result["val_acc"]:.6f}')
        print(f'    ⏱️  Latency:        {result["latency"]:.6f}s')
        print(f'    🔢 Parameters:     {result["params"]:.4f}M')
        print()
    
    # Show statistics
    best = all_results[0]
    worst = all_results[-1]
    avg_acc = sum(r['test_acc'] for r in all_results) / len(all_results)
    
    print(f'{"="*60}')
    print(f'📈 STATISTICS')
    print(f'{"="*60}')
    print(f'Best Test Accuracy:    {best["test_acc"]:.6f}')
    print(f'Worst Test Accuracy:   {worst["test_acc"]:.6f}')
    print(f'Average Test Accuracy: {avg_acc:.6f}')
    print(f'Total Architectures:   {len(all_results)}')
    print()
    
    return all_results[:args.k]

