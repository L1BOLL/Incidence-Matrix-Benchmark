#!/usr/bin/env python3
"""
Graph Benchmark: Enhanced Incidence Matrix vs NetworkX
Tests weighted edges, parallel edges, attributes, and node-edge connections.
"""

import sys
import os
from datetime import datetime
import traceback
import random
import pandas as pd
import matplotlib.pyplot as plt
import platform
import psutil
import json

def get_environment_info():
    import sys, numpy, scipy, networkx, pandas
    info = {
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "networkx": networkx.__version__,
        "pandas": pandas.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.uname().processor,
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_freq_mhz": getattr(psutil.cpu_freq(), "current", None),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
    }
    return info


# Import our implementations
from incidence_graph import IncidenceGraph
from benchmark_framework import (
    BenchmarkRunner, generate_graph_nx, convert_nx_to_incidence, 
    add_node_edge_connections, TEST_OPERATIONS, SINGLE_ARG_OPS, 
    DUAL_ARG_OPS, NO_ARG_OPS, MemoryError
)

def run_comprehensive_benchmark():
    """Run the complete enhanced benchmark suite."""
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    env = get_environment_info()
    env_path = f"enhanced_graph_benchmark_{timestamp}_env.json"
    with open(env_path, "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2)
    print(f"[env] wrote {env_path}: {env}")
    BASE_SEED = 1337
    dataset_counter = 0



    # Test configurations
    NODE_SIZES = [100, 1_000, 10_000] # due to machine limitations eplaced [100, 1_000, 10_000, 100_000, 1_000_000]
    
    # Density definitions
    DENSITY_CONFIGS = {
        'sparse': lambda n: n * 2,      # ~4 avg degree
        'medium': lambda n: n * 10,     # ~20 avg degree  
        'dense': lambda n: min(n * n // 4, n * 50)  # Cap to avoid explosion
    }
    
    # Topologies (only for smaller graphs due to complexity)
    TOPOLOGIES = {
        100: ['erdos_renyi', 'barabasi_albert', 'watts_strogatz', 'grid_2d', 'star', 'complete'],
        1_000: ['erdos_renyi', 'barabasi_albert', 'watts_strogatz', 'grid_2d', 'star'],
        10_000: ['erdos_renyi', 'barabasi_albert']
        # 100_000: ['erdos_renyi'],
        # 1_000_000: ['erdos_renyi']
    }
    
    runner = BenchmarkRunner(max_memory_gb=12)
    all_results = []
    
    print("=" * 70)
    print("ENHANCED GRAPH BENCHMARK: Incidence Matrix vs NetworkX")
    print("Features: Weighted edges, Parallel edges, Attributes, Node-edge connections")
    print("=" * 70)
    print(f"Started: {datetime.now()}")
    print(f"Memory limit: 12GB")
    print()
    
    for num_nodes in NODE_SIZES:
        print(f"\n🧪 Testing {num_nodes:,} nodes...")
        
        for density_name, density_func in DENSITY_CONFIGS.items():
            num_edges = density_func(num_nodes)
            
            # Memory check before attempting
            if runner.should_skip_test(num_nodes, num_edges):
                print(f"  ⚠️  Skipping {density_name} density - estimated memory > 12GB")
                continue
            
            print(f"  📊 {density_name} density ({num_edges:,} edges)")
            
            for topology in TOPOLOGIES[num_nodes]:
                print(f"    🔗 {topology}...")
                
                try:
                    # Generate base graph with weights and attributes
                    nx_graph = generate_graph_nx(topology, num_nodes, num_edges, 
                                               directed=True, weighted=True)
                    actual_edges = nx_graph.number_of_edges()
                    
                    # Convert to incidence matrix
                    with runner.memory_monitor():
                        # Add parallel edges for smaller graphs to test the feature
                        add_parallel = (num_nodes <= 1000)
                        inc_graph = convert_nx_to_incidence(nx_graph, directed=True, 
                                                          add_parallel_edges=add_parallel)
                        
                        # Add node-edge connections for very small graphs
                        if num_nodes <= 100:
                            add_node_edge_connections(inc_graph, probability=0.1)
                    
                    conversion_memory = runner.last_memory_usage / (1024**2)  # MB
                    final_edges = inc_graph.number_of_edges()
                    
                    print(f"      📈 NX edges: {actual_edges:,}, Final edges: {final_edges:,}")
                    print(f"      💾 Conversion memory: {conversion_memory:.1f}MB")
                    
                    # Test each operation
                    for op_name, operations in TEST_OPERATIONS.items():
                        
                        # Prepare test arguments based on operation type
                        if op_name in DUAL_ARG_OPS:
                            # Use nodes that exist in both graphs
                            nodes = list(nx_graph.nodes())
                            if len(nodes) >= 2:
                                test_args = (nodes[0], nodes[1])
                                if op_name == 'add_weighted_edge':
                                    test_args = (nodes[0], nodes[1])
                                elif op_name in ['add_edge']:
                                    # Add weight argument
                                    test_args = (nodes[0], nodes[1], random.uniform(1, 5))
                            else:
                                continue  # Skip if insufficient nodes
                                
                        elif op_name in SINGLE_ARG_OPS:
                            test_args = (list(nx_graph.nodes())[0],)
                            
                        elif op_name in NO_ARG_OPS:
                            test_args = ()
                            
                        else:
                            test_args = ()
                        
                        try:
                            result = runner.benchmark_operation(
                                op_name, 
                                operations['nx'], 
                                operations['inc'],
                                nx_graph, 
                                inc_graph, 
                                *test_args,
                                runs=3
                            )
                            
                            # Add metadata
                            result.update({
                                'num_nodes': num_nodes,
                                'num_edges_nx': actual_edges,
                                'num_edges_inc': final_edges,
                                'density': density_name,
                                'topology': topology,
                                'conversion_memory_mb': conversion_memory,
                                'has_parallel_edges': add_parallel,
                                'has_node_edge_connections': (num_nodes <= 100),
                                'timestamp': datetime.now().isoformat()
                            })
                            
                            all_results.append(result)
                            
                            # Quick feedback with more detailed info
                            speedup = result['speedup']
                            nx_time = result['nx_time_mean']
                            inc_time = result['inc_time_mean']
                            
                            if result['inc_error']:
                                print(f"        💥 {op_name}: FAILED - {result['inc_error'][:30]}...")
                            elif speedup > 1:
                                print(f"        ✅ {op_name}: {speedup:.2f}x faster ({inc_time*1000:.1f}ms vs {nx_time*1000:.1f}ms)")
                            elif speedup > 0:
                                print(f"        ❌ {op_name}: {1/speedup:.2f}x slower ({inc_time*1000:.1f}ms vs {nx_time*1000:.1f}ms)")
                            else:
                                print(f"        💀 {op_name}: Complete failure")
                                
                        except Exception as e:
                            print(f"        💥 {op_name}: CRASHED - {str(e)[:50]}...")
                            
                            # Record failure
                            failed_result = {
                                'operation': op_name,
                                'num_nodes': num_nodes,
                                'num_edges_nx': actual_edges,
                                'num_edges_inc': final_edges,
                                'density': density_name,
                                'topology': topology,
                                'nx_time_mean': 0,
                                'inc_time_mean': float('inf'),
                                'speedup': 0,
                                'inc_error': str(e),
                                'timestamp': datetime.now().isoformat()
                            }
                            all_results.append(failed_result)
                
                except MemoryError as e:
                    print(f"      💀 {topology}: Memory limit exceeded")
                    break
                    
                except Exception as e:
                    print(f"      ❌ {topology}: Setup failed - {str(e)[:50]}...")
                    traceback.print_exc()
                    continue
    
    # Save results
    if all_results:
        df = pd.DataFrame(all_results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = f"enhanced_graph_benchmark_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        print(f"\n💾 Results saved to: {csv_file}")
        
        # Generate summary report
        generate_enhanced_summary_report(df, timestamp)
    else:
        print("\n💀 No results to save - all tests failed!")
    
    return all_results

def generate_enhanced_summary_report(df: pd.DataFrame, timestamp: str):
    """Generate enhanced summary analysis and plots."""
    
    print("\n" + "=" * 70)
    print("ENHANCED BENCHMARK SUMMARY")
    print("=" * 70)
    
    # Overall performance summary
    successful_tests = df[df['inc_error'].isna() & (df['speedup'] > 0)]
    failed_tests = df[(df['inc_error'].notna()) | (df['speedup'] == 0)]
    
    print(f"Total tests: {len(df)}")
    print(f"Successful tests: {len(successful_tests)}")
    print(f"Failed tests: {len(failed_tests)}")
    print(f"Success rate: {len(successful_tests)/len(df)*100:.1f}%")
    
    if len(successful_tests) > 0:
        avg_speedup = successful_tests['speedup'].mean()
        median_speedup = successful_tests['speedup'].median()
        
        print(f"\nPerformance Metrics:")
        print(f"Average speedup: {avg_speedup:.3f}x")
        print(f"Median speedup: {median_speedup:.3f}x")
        print(f"Best speedup: {successful_tests['speedup'].max():.3f}x")
        print(f"Worst speedup: {successful_tests['speedup'].min():.3f}x")
        
        # Performance by operation
        print("\nPerformance by Operation:")
        op_summary = successful_tests.groupby('operation')['speedup'].agg(['mean', 'std', 'count', 'min', 'max'])
        print(op_summary.round(3))
        
        # Performance by graph size
        print("\nPerformance by Graph Size:")
        size_summary = successful_tests.groupby('num_nodes')['speedup'].agg(['mean', 'count'])
        print(size_summary.round(3))
        
        # Memory analysis
        print("\nMemory Usage Analysis:")
        memory_summary = successful_tests.groupby(['num_nodes', 'density']).agg({
            'conversion_memory_mb': 'mean',
            'inc_memory_mb': 'mean',
            'nx_memory_mb': 'mean'
        }).round(1)
        print(memory_summary)
        
        # Feature impact analysis
        if 'has_parallel_edges' in df.columns:
            print("\nParallel Edges Impact:")
            parallel_impact = successful_tests.groupby('has_parallel_edges')['speedup'].agg(['mean', 'count'])
            print(parallel_impact.round(3))
        
        if 'has_node_edge_connections' in df.columns:
            print("\nNode-Edge Connections Impact:")
            hybrid_impact = successful_tests.groupby('has_node_edge_connections')['speedup'].agg(['mean', 'count'])
            print(hybrid_impact.round(3))
    
    # Failure analysis
    if len(failed_tests) > 0:
        print(f"\nFailure Analysis:")
        print("Failures by graph size:")
        failure_by_size = failed_tests.groupby('num_nodes').size()
        print(failure_by_size)
        
        print("\nFailures by operation:")
        failure_by_op = failed_tests.groupby('operation').size().sort_values(ascending=False)
        print(failure_by_op.head(10))
        
        print("\nCommon error types:")
        if 'inc_error' in failed_tests.columns:
            error_types = failed_tests['inc_error'].value_counts().head(5)
            print(error_types)
    
    # Critical findings
    print(f"\n" + "="*50)
    print("CRITICAL FINDINGS")
    print("="*50)
    
    if len(successful_tests) > 0:
        fastest_ops = successful_tests.groupby('operation')['speedup'].mean().sort_values(ascending=False).head(3)
        slowest_ops = successful_tests.groupby('operation')['speedup'].mean().sort_values(ascending=True).head(3)
        
        print("Fastest operations (Incidence Matrix advantage):")
        for op, speedup in fastest_ops.items():
            print(f"  {op}: {speedup:.3f}x faster")
        
        print("\nSlowest operations (NetworkX advantage):")
        for op, speedup in slowest_ops.items():
            print(f"  {op}: {1/speedup:.3f}x slower")
        
        # Find breaking point
        breaking_points = successful_tests.groupby('num_nodes')['speedup'].mean()
        usable_sizes = breaking_points[breaking_points > 0.1]  # At least 10% of NX performance
        
        if len(usable_sizes) > 0:
            max_usable = usable_sizes.index.max()
            print(f"\nMax usable graph size: {max_usable:,} nodes")
        else:
            print(f"\nNo usable graph sizes found (all >10x slower than NetworkX)")
    
    # Generate plots
    try:
        create_enhanced_benchmark_plots(df, timestamp)
        print(f"\n📊 Enhanced plots saved with timestamp: {timestamp}")
    except Exception as e:
        print(f"⚠️  Plot generation failed: {e}")

def create_enhanced_benchmark_plots(df: pd.DataFrame, timestamp: str):
    """Create enhanced visualization plots for benchmark results."""
    
    successful_df = df[df['inc_error'].isna() & (df['speedup'] > 0)]
    
    if len(successful_df) == 0:
        print("No successful tests to plot")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Enhanced Graph Benchmark: Incidence Matrix vs NetworkX', fontsize=16)
    
    # Plot 1: Speedup by graph size
    ax1 = axes[0, 0]
    for density in successful_df['density'].unique():
        subset = successful_df[successful_df['density'] == density]
        if len(subset) > 0:
            ax1.loglog(subset['num_nodes'], subset['speedup'], 'o-', label=density, alpha=0.7)
    
    ax1.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Equal performance')
    ax1.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, label='10x slower threshold')
    ax1.set_xlabel('Number of Nodes')
    ax1.set_ylabel('Speedup (>1 = Incidence faster)')
    ax1.set_title('Performance vs Graph Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Memory usage comparison
    ax2 = axes[0, 1]
    if 'inc_memory_mb' in successful_df.columns and 'nx_memory_mb' in successful_df.columns:
        ax2.loglog(successful_df['num_nodes'], successful_df['inc_memory_mb'], 'ro', alpha=0.6, label='Incidence Matrix')
        ax2.loglog(successful_df['num_nodes'], successful_df['nx_memory_mb'], 'bo', alpha=0.6, label='NetworkX')
        ax2.set_xlabel('Number of Nodes')
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.set_title('Memory Consumption Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Operation comparison
    ax3 = axes[0, 2]
    op_means = successful_df.groupby('operation')['speedup'].mean().sort_values()
    bars = ax3.barh(range(len(op_means)), op_means.values)
    ax3.set_yticks(range(len(op_means)))
    ax3.set_yticklabels(op_means.index, fontsize=8)
    ax3.axvline(x=1, color='red', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Average Speedup')
    ax3.set_title('Performance by Operation')
    ax3.set_xscale('log')
    
    # Color bars
    for i, bar in enumerate(bars):
        if op_means.iloc[i] > 1:
            bar.set_color('green')
        elif op_means.iloc[i] > 0.1:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    # Plot 4: Success rate by size
    ax4 = axes[1, 0]
    success_rate = (
        df.groupby('num_nodes')
        .apply(lambda x: (x['inc_error'].isna()).mean())
        .sort_index()
    )
    ax4.semilogx(success_rate.index, success_rate.values, marker='o')
    ax4.set_title('Success Rate by Size')
    ax4.set_xlabel('Nodes')
    ax4.set_ylabel('Success rate')

    plt.tight_layout()
    outfile = f"enhanced_benchmark_{timestamp}.png"
    plt.savefig(outfile, dpi=150)
    print(f"[plot] saved {outfile}")




if __name__ == "__main__":
    run_comprehensive_benchmark()