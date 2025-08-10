import time
import psutil
import gc
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Callable
from contextlib import contextmanager
import random

class MemoryError(Exception):
    """Custom exception for memory limit exceeded."""
    pass

class BenchmarkRunner:
    """
    Graph benchmark framework comparing incidence matrix vs NetworkX.
    Updated to handle weighted edges, parallel edges, and attributes.
    """
    
    def __init__(self, max_memory_gb: float = 12): # as a function of machine
        self.max_memory_bytes = max_memory_gb * (1024 ** 3)
        self.results = []
        
    def estimate_incidence_memory(self, num_nodes: int, num_edges: int) -> int:
        """Estimate memory usage for incidence matrix (bytes)."""
        # DOK matrix: 3 arrays (data, row, col) + overhead
        nnz = num_edges * 2  # source + target per edge
        data_bytes = nnz * 4  # float32
        indices_bytes = nnz * 8  # int32 for row + col indices
        
        # Dictionary overhead for entity/edge mappings
        dict_overhead = (num_nodes + num_edges) * 200  # increased for more mappings
        
        # DataFrame overhead for attributes (rough estimate)
        df_overhead = (num_nodes + num_edges) * 100
        
        return data_bytes + indices_bytes + dict_overhead + df_overhead
    
    def should_skip_test(self, num_nodes: int, num_edges: int) -> bool:
        """Check if test should be skipped due to memory constraints."""
        estimated = self.estimate_incidence_memory(num_nodes, num_edges)
        return estimated > self.max_memory_bytes
    
    @contextmanager
    def memory_monitor(self):
        """Context manager to monitor memory usage."""
        process = psutil.Process()
        gc.collect()
        mem_before = process.memory_info().rss
        
        try:
            yield
        finally:
            mem_after = process.memory_info().rss
            self.last_memory_usage = mem_after - mem_before
    
    def time_operation(self, operation: Callable, *args, **kwargs) -> float:
        """Time a single operation."""
        gc.collect()
        start_time = time.perf_counter()
        result = operation(*args, **kwargs)
        end_time = time.perf_counter()
        return end_time - start_time, result
    
    def benchmark_operation(self, operation_name: str, nx_op: Callable, inc_op: Callable, 
                          nx_graph, inc_graph, *args, runs: int = 3) -> Dict[str, Any]:
        """Benchmark a single operation on both graph types."""
        
        # NetworkX timing
        nx_times = []
        nx_memory = 0
        for _ in range(runs):
            with self.memory_monitor():
                try:
                    duration, _ = self.time_operation(nx_op, nx_graph, *args)
                    nx_times.append(duration)
                    nx_memory = max(nx_memory, self.last_memory_usage)
                except Exception:
                    # Some operations might not be supported in NetworkX
                    nx_times.append(float('inf'))
        
        # Incidence matrix timing  
        inc_times = []
        inc_memory = 0
        inc_error = None
        
        try:
            for _ in range(runs):
                with self.memory_monitor():
                    duration, _ = self.time_operation(inc_op, inc_graph, *args)
                    inc_times.append(duration)
                    inc_memory = max(inc_memory, self.last_memory_usage)
        except Exception as e:
            inc_error = str(e)
            inc_times = [float('inf')] * runs
        
        nx_mean = np.mean(nx_times)
        inc_mean = np.mean(inc_times)
        
        return {
            'operation': operation_name,
            'nx_time_mean': nx_mean,
            'nx_time_std': np.std(nx_times),
            'inc_time_mean': inc_mean,
            'inc_time_std': np.std(inc_times),
            'speedup': nx_mean / inc_mean if inc_mean > 0 and inc_mean != float('inf') else 0,
            'nx_memory_mb': nx_memory / (1024**2),
            'inc_memory_mb': inc_memory / (1024**2),
            'inc_error': inc_error
        }

def generate_graph_nx(topology: str, num_nodes: int, num_edges: int, directed: bool = True, 
                      weighted: bool = True) -> nx.Graph:
    """Generate NetworkX graph with specified topology and weights."""
    
    if topology == 'erdos_renyi':
        p = num_edges / (num_nodes * (num_nodes - 1) / (2 if not directed else 1))
        p = min(p, 1.0)  # Cap probability
        if directed:
            G = nx.erdos_renyi_graph(num_nodes, p, directed=True)
        else:
            G = nx.erdos_renyi_graph(num_nodes, p, directed=False)
    
    elif topology == 'barabasi_albert':
        m = max(1, num_edges // num_nodes)  # Average degree
        m = min(m, num_nodes - 1)  # Cap at max possible
        G = nx.barabasi_albert_graph(num_nodes, m)
        if directed:
            G = G.to_directed()
    
    elif topology == 'watts_strogatz':
        k = max(2, min(num_nodes - 1, 2 * (num_edges // num_nodes)))  # Ensure even k
        if k % 2 == 1:
            k -= 1
        k = max(2, k)
        p = 0.1  # Rewiring probability
        G = nx.watts_strogatz_graph(num_nodes, k, p)
        if directed:
            G = G.to_directed()
    
    elif topology == 'grid_2d':
        side = int(np.sqrt(num_nodes))
        G = nx.grid_2d_graph(side, side)
        # Relabel nodes to integers
        G = nx.convert_node_labels_to_integers(G)
        if directed:
            G = G.to_directed()
    
    elif topology == 'star':
        G = nx.star_graph(num_nodes - 1)
        if directed:
            G = G.to_directed()
    
    elif topology == 'complete':
        G = nx.complete_graph(num_nodes)
        if directed:
            G = G.to_directed()
    
    else:
        raise ValueError(f"Unknown topology: {topology}")
    
    # Add weights to edges
    if weighted:
        for u, v in G.edges():
            G[u][v]['weight'] = random.uniform(0.1, 10.0)
    
    # Add node attributes
    for node in G.nodes():
        G.nodes[node]['node_attr'] = f"node_{node}_value"
        G.nodes[node]['numeric_attr'] = random.uniform(0, 100)
    
    return G

def convert_nx_to_incidence(nx_graph, directed: bool = True, add_parallel_edges: bool = False):
    """Convert NetworkX graph to incidence matrix representation."""
    from incidence_graph import IncidenceGraph  # Import our class
    
    inc_graph = IncidenceGraph(directed=directed)
    
    # Set graph attributes
    inc_graph.set_graph_attribute('name', 'converted_from_nx')
    inc_graph.set_graph_attribute('original_type', type(nx_graph).__name__)
    
    # Add all nodes with attributes
    for node in nx_graph.nodes():
        attrs = dict(nx_graph.nodes[node])
        inc_graph.add_node(node, **attrs)
    
    # Add all edges with weights and attributes
    for source, target in nx_graph.edges():
        edge_data = nx_graph.get_edge_data(source, target)
        weight = edge_data.get('weight', 1.0)
        
        # Remove weight from attributes to avoid duplication
        attrs = {k: v for k, v in edge_data.items() if k != 'weight'}
        
        edge_id = inc_graph.add_edge(source, target, weight=weight, **attrs)
        
        # Add parallel edge for testing (10% chance for small graphs)
        if add_parallel_edges and random.random() < 0.1 and inc_graph.number_of_edges() < 1000:
            parallel_weight = random.uniform(0.1, 5.0)
            inc_graph.add_parallel_edge(source, target, weight=parallel_weight, 
                                      edge_type='parallel', test_attr='parallel_edge')
    
    return inc_graph

def add_node_edge_connections(inc_graph, probability: float = 0.05):
    """Add node-edge hybrid connections to test advanced features."""
    edges = inc_graph.edges()
    nodes = inc_graph.nodes()
    
    # Add a few edge entities and connect them to nodes
    for i, edge_id in enumerate(edges[:min(10, len(edges))]):  # Limit to avoid explosion
        if random.random() < probability:
            # Create node-edge connection
            random_node = random.choice(nodes)
            hybrid_edge_id = inc_graph.add_edge(
                random_node, edge_id, 
                weight=random.uniform(0.1, 2.0),
                edge_type='node_edge',
                hybrid_type='node_to_edge'
            )

# Enhanced test operations dictionary
TEST_OPERATIONS = {
    'add_edge': {
        'nx': lambda g, s, t, w: g.add_edge(s, t, weight=w),
        'inc': lambda g, s, t, w: g.add_edge(s, t, weight=w)
    },
    'add_weighted_edge': {
        'nx': lambda g, s, t: g.add_edge(s, t, weight=random.uniform(1, 10)),
        'inc': lambda g, s, t: g.add_edge(s, t, weight=random.uniform(1, 10))
    },
    'has_edge': {
        'nx': lambda g, s, t: g.has_edge(s, t),
        'inc': lambda g, s, t: g.has_edge(s, t)
    },
    'neighbors': {
        'nx': lambda g, node: list(g.neighbors(node)),
        'inc': lambda g, node: g.neighbors(node)
    },
    'degree': {
        'nx': lambda g, node: g.degree(node),
        'inc': lambda g, node: g.degree(node)
    },
    'copy': {
        'nx': lambda g: g.copy(),
        'inc': lambda g: g.copy()
    },
    'get_edge_weight': {
        'nx': lambda g, s, t: g.get_edge_data(s, t, {}).get('weight', 1.0) if g.has_edge(s, t) else None,
        'inc': lambda g, s, t: g.edge_weights.get(g.get_edge_ids(s, t)[0], None) if g.has_edge(s, t) and g.get_edge_ids(s, t) else None
    },
    'node_attributes': {
        'nx': lambda g, node: dict(g.nodes[node]) if node in g.nodes else {},
        'inc': lambda g, node: {attr: g.get_node_attribute(node, attr) for attr in ['node_attr', 'numeric_attr']} if node in g.nodes() else {}
    },
    'add_parallel_edge': {
        'nx': lambda g, s, t: None,  # NetworkX doesn't natively support parallel edges in simple graphs
        'inc': lambda g, s, t: g.add_parallel_edge(s, t, weight=random.uniform(1, 5))
    },
    'edge_list': {
        'nx': lambda g: list(g.edges(data=True)),
        'inc': lambda g: g.edge_list()
    },
    'memory_usage': {
        'nx': lambda g: estimate_nx_memory(g),
        'inc': lambda g: g.memory_usage()
    }
}

def estimate_nx_memory(nx_graph):
    """Rough estimate of NetworkX graph memory usage."""
    # Very rough estimation
    num_nodes = nx_graph.number_of_nodes()
    num_edges = nx_graph.number_of_edges()
    
    # NetworkX uses dict-of-dicts, very rough estimate
    node_memory = num_nodes * 200  # dict overhead + attributes
    edge_memory = num_edges * 150  # edge data structures
    
    return node_memory + edge_memory

# Operations that work on single arguments
SINGLE_ARG_OPS = ['neighbors', 'degree', 'node_attributes']
DUAL_ARG_OPS = ['add_edge', 'add_weighted_edge', 'has_edge', 'get_edge_weight', 'add_parallel_edge']
NO_ARG_OPS = ['copy', 'edge_list', 'memory_usage']