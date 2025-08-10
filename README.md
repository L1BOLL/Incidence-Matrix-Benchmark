# 1. Approach

Compared an **incidence matrix graph representation** with the standard **NetworkX DiGraph (Directed Graph)** on many graph sizes, densities, and shapes.  
Goal: measure speed (runtime) and memory use, find where the incidence form is better, and see how it scales.

# 2. Incidence Representation (Internal Design)

The custom `IncidenceGraph` stores a graph as a **sparse incidence matrix**.
- **Main structure**: a SciPy DOK (Dictionary Of Keys) sparse matrix `M` with shape _(n_nodes, n_edges)_.
    - `M[u, e]` = +w if edge `e` starts at node `u` (weight _w_), and −w if it ends at `u`.
    - A column can have many nonzeros, so it supports **parallel edges**.
    - Optional “hybrid” links let nodes connect to edges as first-class items.
- **Edge metadata**: a Pandas DataFrame for attributes (weights, IDs, flags).
- **Indexing**: node ID ↔ matrix row; edge ID ↔ matrix column (both mapped with Python dicts).
- **Memory estimate**: a simple linear model, trained on small samples, that predicts full-graph RAM from n_nodes, n_edges, and nnz (number of non-zeros).

This design is good for edge-focused work (edge listing, global walks). It has extra cost for node-focused work (degree, neighbors).

# 3. Benchmark Framework

In `benchmark_framework.py`:
- **Parameter grid**:
    - Sizes: 100, 1 000, 10 000 nodes (could not go larger on this machine).
    - Densities: sparse (≈2n edges), medium (≈10n), dense (up to min(n²/4, 50n)).
    - Topologies: Erdős–Rényi, Barabási–Albert, Watts–Strogatz, 2D grid, star, complete (subset changes with size).
    - Feature flags: parallel edges (n ≤ 1 000), hybrid node–edge links (n ≤ 100).
- **Implementations**: `IncidenceGraph` vs. NetworkX `DiGraph`.
- **Timed operations**:
    - Two-argument: `add_edge`, `add_weighted_edge`, `has_edge`, `get_edge_weight`, `add_parallel_edge`.
    - One-argument: `neighbors`, `degree`, `node_attributes`.
    - No-argument: `copy`, `edge_list`, `memory_usage`.
- **Metrics**:
    - Runtime: mean and standard deviation over 3 runs per operation.
    - Memory: RSS (Resident Set Size) change before/after each operation + model estimates.
    - Speedup = (nx_time / inc_time). Values >1 mean incidence is faster.

# 4. Testing Method and Why

- **Graph creation**: each case is built fresh with NetworkX generators to avoid caching.
- **Repeats**: 3 runs per operation to reduce noise from the Python VM and the OS scheduler.
- **Skip rule**: predicted memory is checked before running; very large cases are skipped to avoid >12 GB use  (current machine limit).
- **Measurement split**: time with `time.perf_counter`; memory with `psutil.Process().memory_info().rss` (RSS).
- **Reasoning**:
    - Mix of edge-heavy and node-heavy tasks.
    - Different densities and topologies to see if structure changes the result.
    - Larger graphs (10 000 nodes) to view scaling and memory trends.

# 5. Results
**Performance** (Speedup = nx_time / inc_time; >1 means incidence wins):
- Only **edge_list** was often faster with incidence (gmean (geometric mean) ≈ 1.42×; 82% of runs faster).
- All other operations were slower, often much slower: `degree` (~270× slower), `copy` (~196×), `get_edge_weight` (~80×), `node_attributes` (~73×), `neighbors` (~38×).
- The gap grew with size: gmean (geometric mean) speedup fell from 0.068× (100 nodes) to 0.028× (10 000 nodes).

**Memory**
- **Model**: incidence memory rises much faster with size (example: 10 000 nodes median ≈ 68 MB vs ~0 MB for NetworkX).
- **Measured RSS deltas**: noisy and often near zero (RSS has coarse steps). Use the model for true trends.

---

- **The directory names "2" has the files: incidence_graph.py where the incidence matrix class is defined, benchmark_framework for becnhmark layout and benchmark_main.**

- **The "Enhanced Memory Models" is the directory where linear memory estimators are used and output plots are implemented. The files serve the same purposes.**
