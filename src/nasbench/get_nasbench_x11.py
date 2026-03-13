from nasbench.nas_bench_x11.api import load_ensemble
import numpy as np
from ConfigSpace.read_and_write import json as cs_json
from tqdm import tqdm

def get_on_nb101(args):
    # Import NASBench101 dependencies only when needed
    from nasbench.nas_bench_x11.encodings.encodings_nb101 import encode_adj
    from nasbench.nasbench101.api import NASBench
    
    nb111_surrogate_model = load_ensemble('nasbench/nas_bench_x11/models/nb111-v0.5')

    nasbench101_path = 'nasbench/nasbench101/nasbench_full.tfrecord'
    nasbench = NASBench(nasbench101_path)

    # Lấy tất cả các hash của architectures
    all_hashes = list(nasbench.hash_iterator())
    arch_results = []
    
    num_to_process = min(args.num_samples, len(all_hashes)) if args.num_samples > 0 else len(all_hashes)
    print(f"Processing {num_to_process} / {len(all_hashes)} architectures...")
    processed = 0
    for i, arch_hash in enumerate(all_hashes):
        if processed >= num_to_process:
            break
        # Lấy thông tin architecture từ hash
        fixed_metrics, computed_metrics = nasbench.get_metrics_from_hash(arch_hash)
        
        # Lấy matrix và ops từ fixed_metrics
        matrix = fixed_metrics['module_adjacency']
        ops = fixed_metrics['module_operations']
        
        # Skip pruned architectures where NUM_VERTICES != 7
        if matrix.shape[0] != 7:
            continue
        
        # Encode architecture (26 features)
        encoding = encode_adj(matrix, ops)
        
        # Lấy accuracies tại epochs [4, 12, 36, 108] (4 features)
        accs = []
        for epoch in [4, 12, 36, 108]:
            # Lấy validation accuracy trung bình từ 3 runs
            epoch_accs = [computed_metrics[epoch][j]['final_validation_accuracy'] 
                         for j in range(len(computed_metrics[epoch]))]
            accs.append(np.mean(epoch_accs))
        
        # Tạo full_encoding = encoding + accuracies (26 + 4 = 30 features)
        full_encoding = np.array([*encoding, *accs])
        
        # Predict với surrogate model
        predicted_curve = nb111_surrogate_model.predict(
            config=full_encoding, 
            representation="compact", 
            with_noise=False,
            search_space='nb101'
        )
        
        # Lưu kết quả
        arch_results.append({
            'hash': arch_hash,
            'encoding': encoding,
            'accuracies': accs,
            'full_encoding': full_encoding,
            'predicted_curve': predicted_curve,
            'final_test_accuracy': computed_metrics[108][0]['final_test_accuracy']
        })
        processed += 1
        
    
    # Sắp xếp theo final test accuracy
    arch_results.sort(key=lambda x: x['final_test_accuracy'], reverse=True)
    
    # In top-k architectures
    print(f"\nTop-{args.k} architectures:")
    for i, result in enumerate(arch_results[:args.k]):
        print(f"\n{i+1}. Hash: {result['hash']}")
        print(f"   Final Test Accuracy: {result['final_test_accuracy']:.4f}")
        print(f"   Validation Accs [4,12,36,108]: {[f'{a:.4f}' for a in result['accuracies']]}")
        print(f"   Predicted Curve: {result['predicted_curve']}")
    
    return arch_results

def get_on_nb201(args):
    nb211_surrogate_model = load_ensemble('nasbench/nas_bench_x11/models/nb211-v0.5')
    from nasbench.nasbench201.api_201 import NASBench201API
    from tqdm import tqdm
    
    # Load NASBench-201 API để lấy tất cả architectures
    nasbench201_path = 'nasbench/nasbench201/NAS-Bench-201-v1_1-096897.pth'
    print(f"Loading NASBench-201 from {nasbench201_path}...")
    api = NASBench201API(nasbench201_path, verbose=False)
    
    # Lấy tất cả architecture strings
    all_archs = api.meta_archs  # List of all architecture strings
    total_archs = len(all_archs)
    
    print(f"Found {total_archs} architectures in NASBench-201")
    print(f"Processing {'all' if args.num_samples >= total_archs else args.num_samples} architectures...")
    
    # Giới hạn số lượng nếu num_samples < total
    num_to_process = min(args.num_samples, total_archs)
    archs_to_process = all_archs[:num_to_process]
    
    arch_results = []
    
    for i, arch_str in enumerate(tqdm(archs_to_process, desc="Evaluating architectures")):
        # Query performance using NASBench-X11 surrogate model
        learning_curve = nb211_surrogate_model.predict(
            config=arch_str, 
            representation="arch_str",
            with_noise=False,
            search_space='nb201'
        )
        
        accuracy = learning_curve[args.epoch - 1] if isinstance(learning_curve, (list, np.ndarray)) else learning_curve
        
        
        arch_results.append({
            'arch_str': arch_str,
            'arch_index': i,
            'learning_curve': learning_curve,
            'accuracy': accuracy,
        })
    
    print(f"\nSuccessfully processed {num_to_process} architectures.")
    
    # Sắp xếp theo accuracy
    arch_results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    # In top-k architectures
    print(f"\n{'='*70}")
    print(f"Top-{args.k} architectures (NASBench-201, nas_bench_x11 model):")
    print(f"{'='*70}")
    
    for i, result in enumerate(arch_results[:args.k]):
        print(f"\n[Rank {i+1}] Predicted Accuracy: {result['accuracy']:.4f}%")
        print(f"Architecture (index {result['arch_index']}): {result['arch_str']}")
    
    return arch_results

def get_on_nb301(args):
    nb311_surrogate_model = load_ensemble('nasbench/nas_bench_x11/models/nb311-v0.5')
    from collections import namedtuple
    import random
    Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
    
    # Các operations có sẵn trong DARTS search space
    OPERATIONS = [
        'max_pool_3x3',
        'avg_pool_3x3', 
        'skip_connect',
        'sep_conv_3x3',
        'sep_conv_5x5',
        'dil_conv_3x3',
        'dil_conv_5x5'
    ]
    
    def random_genotype():
        """Tạo random genotype cho DARTS"""
        def random_cell():
            cell = []
            # Node 2: nhận input từ node 0 và 1
            cell.append((random.choice(OPERATIONS), 0))
            cell.append((random.choice(OPERATIONS), 1))
            
            # Node 3: chọn 2 inputs từ {0, 1, 2}
            inputs_3 = random.sample([0, 1, 2], 2)
            for inp in sorted(inputs_3):
                cell.append((random.choice(OPERATIONS), inp))
            
            # Node 4: chọn 2 inputs từ {0, 1, 2, 3}
            inputs_4 = random.sample([0, 1, 2, 3], 2)
            for inp in sorted(inputs_4):
                cell.append((random.choice(OPERATIONS), inp))
            
            # Node 5: chọn 2 inputs từ {0, 1, 2, 3, 4}
            inputs_5 = random.sample([0, 1, 2, 3, 4], 2)
            for inp in sorted(inputs_5):
                cell.append((random.choice(OPERATIONS), inp))
            
            return cell
        
        def random_concat():
            # Random chọn subset từ {2, 3, 4, 5}, ít nhất 1 node
            available_nodes = [2, 3, 4, 5]
            num_concat = random.randint(1, 4)
            return sorted(random.sample(available_nodes, num_concat))
        
        normal = random_cell()
        reduce = random_cell()
        normal_concat = random_concat()
        reduce_concat = random_concat()
        
        return Genotype(
            normal=normal, 
            normal_concat=normal_concat,
            reduce=reduce,  
            reduce_concat=reduce_concat
        )
    
    arch_results = []
    seen_genotypes = set()
    print(f"Sampling {args.num_samples} unique random architectures using nas_bench_x11 model...")
    
    i = 0
    attempts = 0
    max_attempts = args.num_samples * 100
    
    for _ in tqdm(range(args.num_samples), desc="Sampling unique architectures"):
        if i >= args.num_samples:
            break
            
        attempts += 1
        genotype = random_genotype()
        
        # Convert genotype thành dạng hashable để kiểm tra duplicate
        genotype_key = (tuple(genotype.normal), tuple(genotype.normal_concat), 
                       tuple(genotype.reduce), tuple(genotype.reduce_concat))
        
        # Kiểm tra xem đã tạo genotype này chưa
        if genotype_key in seen_genotypes:
            continue
        
        seen_genotypes.add(genotype_key)
        i += 1
        
        # Query performance
        learning_curve = nb311_surrogate_model.predict(config=genotype, representation="genotype", with_noise=True)
        
        accuracy = learning_curve[args.epoch - 1] if isinstance(learning_curve, (list, np.ndarray)) else learning_curve
        
        arch_results.append({
            'genotype': genotype,
            'learning_curve': learning_curve,
            'accuracy': accuracy
        })
    
    if attempts >= max_attempts:
        print(f"\nWarning: Reached max attempts ({max_attempts}). Only generated {i} unique architectures.")
    else:
        print(f"\nSuccessfully generated {args.num_samples} unique architectures with {attempts} total attempts.")
    
    # Sắp xếp theo final accuracy
    arch_results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    # In top-k architectures
    print(f"\n{'='*70}")
    print(f"Top-{args.k} architectures (nas_bench_x11 model):")
    print(f"{'='*70}")
    
    for i, result in enumerate(arch_results[:args.k]):
        print(f"\n[Rank {i+1}] Accuracy: {result['accuracy']:.4f}%")
        # In Genotype
        genotype = result['genotype']
        print(f"\nGenotype:")
        print(f"  Normal: {genotype.normal}")
        print(f"  Reduce: {genotype.reduce}")
    return arch_results

def get_on_nbnlp(args):
    nbnlp_surrogate_model = load_ensemble('nasbench/nas_bench_x11/models/nbnlp-v0.5')
    import random
    from tqdm import tqdm
    
    # NLP operations: 0-padding, 1-linear, 2-blend, 3-elementwise_prod, 4-elementwise_sum, 5-activation_tanh, 6-activation_sigm, 7-activation_leaky_relu
    OPS_NLP = list(range(8))
    MAX_NODES = 12  # Maximum number of nodes in NLP search space
    
    def random_nlp_arch(max_nodes=8):
        """
        Generate random NLP architecture in compact format
        compact = (edges, ops, hidden_states)
        - edges: list of tuples (from_node, to_node)
        - ops: list of operations for each node
        - hidden_states: list of node indices that are used as hidden states (max 3)
        """
        num_nodes = random.randint(4, max_nodes)  # Random number of nodes between 4 and max_nodes
        
        # Generate edges - must form a DAG (Directed Acyclic Graph)
        edges = []
        # Each node can connect to subsequent nodes
        for i in range(num_nodes - 1):
            # Randomly connect to some future nodes
            num_connections = random.randint(1, min(3, num_nodes - i - 1))
            targets = random.sample(range(i + 1, num_nodes), num_connections)
            for target in targets:
                edges.append((i, target))
        
        # Ensure the last node is reachable (output node)
        if not any(edge[1] == num_nodes - 1 for edge in edges):
            source = random.randint(0, num_nodes - 2)
            edges.append((source, num_nodes - 1))
        
        # Generate operations for each node
        ops = [random.choice(OPS_NLP) for _ in range(num_nodes)]
        
        # Select hidden states (indices of nodes to use as hidden states, max 3)
        num_hidden = random.randint(1, min(3, num_nodes - 1))
        # Hidden states should not include the output node
        hidden_states = sorted(random.sample(range(num_nodes - 1), num_hidden))
        
        return (edges, ops, hidden_states)
    
    arch_results = []
    seen_archs = set()
    print(f"Sampling {args.num_samples} unique random NLP architectures...")
    
    i = 0
    attempts = 0
    max_attempts = args.num_samples * 100
    
    with tqdm(total=args.num_samples, desc="Evaluating NLP architectures") as pbar:
        while i < args.num_samples and attempts < max_attempts:
            attempts += 1
            compact = random_nlp_arch(max_nodes=MAX_NODES)
            
            # Convert to hashable format for duplicate checking
            compact_key = (tuple(compact[0]), tuple(compact[1]), tuple(compact[2]))
            
            if compact_key in seen_archs:
                continue
            
            seen_archs.add(compact_key)
            i += 1
            
            # Predict with surrogate model
            # NLP model expects compact representation with 3 accuracies for learning curve
            from nasbench.nas_bench_x11.encodings.encodings_nlp import encode_nlp
            
            # Use 3 placeholder accuracies (model expects 188 features = 185 + 3)
            accs = [0.0, 0.0, 0.0]
            encoding = encode_nlp(compact, max_nodes=MAX_NODES, accs=accs, one_hot=False, lc_feature=True, only_accs=False)
            
            learning_curve = nbnlp_surrogate_model.predict(
                config=encoding,
                representation="compact",
                with_noise=True,
                search_space='nlp'
            )
            
            # Get accuracy at the final epoch
            accuracy = learning_curve[args.epochs - 1] if isinstance(learning_curve, (list, np.ndarray)) else learning_curve
            
            arch_results.append({
                'compact': compact,
                'encoding': encoding,
                'learning_curve': learning_curve,
                'accuracy': accuracy,
                'num_nodes': len(compact[1]),
                'num_edges': len(compact[0])
            })
            
            pbar.update(1)
            pbar.set_postfix({'attempts': attempts, 'duplicates': attempts - i})
    
    if attempts >= max_attempts:
        print(f"\nWarning: Reached max attempts ({max_attempts}). Only generated {i} unique architectures.")
    else:
        print(f"\nSuccessfully generated {args.num_samples} unique architectures with {attempts} total attempts.")
    
    # Sort by accuracy
    arch_results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    # Print top-k architectures
    print(f"\n{'='*70}")
    print(f"Top-{args.k} architectures (NAS-Bench-NLP, nas_bench_x11 model):")
    print(f"{'='*70}")
    
    for i, result in enumerate(arch_results[:args.k]):
        print(f"\n[Rank {i+1}] Accuracy: {result['accuracy']:.4f}%")
        print(f"Nodes: {result['num_nodes']}, Edges: {result['num_edges']}")
        print(f"Architecture (compact format):")
        print(f"  Edges: {result['compact'][0]}")
        print(f"  Ops: {result['compact'][1]}")
        print(f"  Hidden States: {result['compact'][2]}")
    
    return arch_results
    


def get_top_arch(args):
    if args.search_space == 'nb101':
        get_on_nb101(args)
    elif args.search_space == 'nb201':
        get_on_nb201(args)
    elif args.search_space == 'nb301':
        get_on_nb301(args)
    elif args.search_space == 'nbnlp':
        get_on_nbnlp(args)
    else:
        raise NotImplementedError(f"Search space {args.search_space} not implemented yet.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate NAS-Bench-111 Surrogate Model")
    parser.add_argument('--search-space', type=str, default='nb101', choices=['nb101','nb201','nb301','nbnlp'], help='Search space')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset name')
    parser.add_argument('--k', type=int, default=10, help='Top-k architectures to display')
    parser.add_argument('--epochs', type=int, default=108, help='Epochs for learning curve')
    parser.add_argument('--num-samples', type=int, default=10000, help='Number of samples for nb301')
    args = parser.parse_args()
    main(args)