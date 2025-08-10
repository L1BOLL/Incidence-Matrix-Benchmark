import numpy as np
import scipy.sparse as sp
import pandas as pd

class IncidenceGraph:
    """
    Graph implementation using sparse incidence matrix representation.
    Rows = nodes + node-edge hybrid entities
    Columns = edges (including parallel edges)
    
    Matrix[i,j] = +weight if node i is source of edge j
    Matrix[i,j] = -weight if node i is target of edge j  
    Matrix[i,j] = 0 if node i not incident to edge j
    
    Supports:
    - Weighted edges
    - Parallel edges (same nodes, different edge IDs)
    - Node-edge hybrid edges (edges can connect to other edges)
    - Polars DataFrames for attributes
    """
    
    def __init__(self, directed=True):
        self.directed = directed
        
        # Entity mappings (nodes + node-edge hybrids)
        self.entity_to_idx = {}  # entity_id -> row index
        self.idx_to_entity = {}  # row index -> entity_id
        self.entity_types = {}   # entity_id -> 'node' or 'edge'
        
        # Edge mappings (supports parallel edges)
        self.edge_to_idx = {}    # edge_id -> column index
        self.idx_to_edge = {}    # column index -> edge_id
        self.edge_definitions = {}  # edge_id -> (source, target, edge_type)
        self.edge_weights = {}   # edge_id -> weight
        
        # Sparse incidence matrix
        self._matrix = sp.dok_matrix((0, 0), dtype=np.float32)
        self._num_entities = 0
        self._num_edges = 0
        
        # Attribute storage using polars DataFrames
        self.node_attributes = pd.DataFrame()  # Using pandas for now, can switch to polars
        self.edge_attributes = pd.DataFrame()
        self.graph_attributes = {}
        
        # Edge ID counter for parallel edges
        self._next_edge_id = 0
    
    def _get_next_edge_id(self) -> str:
        """Generate unique edge ID for parallel edges."""
        edge_id = f"edge_{self._next_edge_id}"
        self._next_edge_id += 1
        return edge_id
    
    def add_node(self, node_id, **attributes):
        """Add a node to the graph with optional attributes."""
        if node_id not in self.entity_to_idx:
            idx = self._num_entities
            self.entity_to_idx[node_id] = idx
            self.idx_to_entity[idx] = node_id
            self.entity_types[node_id] = 'node'
            self._num_entities += 1
            
            # Resize matrix to add new row
            if self._num_edges > 0:
                self._matrix.resize((self._num_entities, self._num_edges))
            else:
                self._matrix.resize((self._num_entities, 0))
            
            # Add node attributes
            if attributes:
                new_row = pd.DataFrame([{'node_id': node_id, **attributes}])
                if self.node_attributes.empty:
                    self.node_attributes = new_row
                else:
                    self.node_attributes = pd.concat([self.node_attributes, new_row], ignore_index=True)
    
    def add_edge(self, source, target, weight=1.0, edge_id=None, edge_type='regular', **attributes):
        """
        Add an edge to the graph.
        
        Parameters:
        - source, target: can be node IDs or edge IDs (for node-edge connections)
        - weight: edge weight
        - edge_id: custom edge ID (auto-generated if None, enables parallel edges)
        - edge_type: 'regular', 'node_edge', etc.
        - **attributes: edge attributes
        """
        # Ensure source and target exist as entities
        if source not in self.entity_to_idx:
            if edge_type == 'node_edge' and source.startswith('edge_'):
                # This is a node-edge hybrid - add as edge entity
                self._add_edge_entity(source)
            else:
                self.add_node(source)
        
        if target not in self.entity_to_idx:
            if edge_type == 'node_edge' and target.startswith('edge_'):
                self._add_edge_entity(target)
            else:
                self.add_node(target)
        
        # Generate edge ID if not provided (enables parallel edges)
        if edge_id is None:
            edge_id = self._get_next_edge_id()
        
        # Check if edge already exists
        if edge_id in self.edge_to_idx:
            # Update existing edge
            col_idx = self.edge_to_idx[edge_id]
            self.edge_weights[edge_id] = weight
            
            # Update matrix values
            source_idx = self.entity_to_idx[source]
            target_idx = self.entity_to_idx[target]
            
            # Clear old values
            self._matrix[:, col_idx] = 0
            
            # Set new values
            self._matrix[source_idx, col_idx] = weight
            if source != target:  # avoid double-counting self-loops
                self._matrix[target_idx, col_idx] = -weight if self.directed else weight
            
            return edge_id
        
        # Add new edge
        col_idx = self._num_edges
        self.edge_to_idx[edge_id] = col_idx
        self.idx_to_edge[col_idx] = edge_id
        self.edge_definitions[edge_id] = (source, target, edge_type)
        self.edge_weights[edge_id] = weight
        self._num_edges += 1
        
        # Resize matrix to add new column
        self._matrix.resize((self._num_entities, self._num_edges))
        
        # Set incidence values
        source_idx = self.entity_to_idx[source]
        target_idx = self.entity_to_idx[target]
        
        self._matrix[source_idx, col_idx] = weight
        if source != target:  # avoid double-counting self-loops
            self._matrix[target_idx, col_idx] = -weight if self.directed else weight
        
        # Add edge attributes
        if attributes:
            new_row = pd.DataFrame([{'edge_id': edge_id, 'source': source, 'target': target, 
                                   'weight': weight, 'edge_type': edge_type, **attributes}])
            if self.edge_attributes.empty:
                self.edge_attributes = new_row
            else:
                self.edge_attributes = pd.concat([self.edge_attributes, new_row], ignore_index=True)
        
        return edge_id
    
    def _add_edge_entity(self, edge_id):
        """Add an edge as an entity (for node-edge hybrid connections)."""
        if edge_id not in self.entity_to_idx:
            idx = self._num_entities
            self.entity_to_idx[edge_id] = idx
            self.idx_to_entity[idx] = edge_id
            self.entity_types[edge_id] = 'edge'
            self._num_entities += 1
            
            # Resize matrix
            self._matrix.resize((self._num_entities, self._num_edges))
    
    def add_parallel_edge(self, source, target, weight=1.0, **attributes):
        """Add a parallel edge (same nodes, different edge ID)."""
        return self.add_edge(source, target, weight=weight, edge_id=None, **attributes)
    
    def remove_edge(self, edge_id):
        """Remove an edge from the graph."""
        if edge_id not in self.edge_to_idx:
            raise KeyError(f"Edge {edge_id} not found")
        
        col_idx = self.edge_to_idx[edge_id]
        
        # Convert to CSR for efficient column removal
        csr_matrix = self._matrix.tocsr()
        
        # Create mask to remove column
        mask = np.ones(self._num_edges, dtype=bool)
        mask[col_idx] = False
        
        # Remove column
        csr_matrix = csr_matrix[:, mask]
        self._matrix = csr_matrix.todok()
        
        # Update mappings
        del self.edge_to_idx[edge_id]
        del self.edge_definitions[edge_id]
        del self.edge_weights[edge_id]
        
        # Reindex remaining edges
        new_edge_to_idx = {}
        new_idx_to_edge = {}
        
        new_idx = 0
        for old_idx in range(self._num_edges):
            if old_idx != col_idx:
                edge_id_old = self.idx_to_edge[old_idx]
                new_edge_to_idx[edge_id_old] = new_idx
                new_idx_to_edge[new_idx] = edge_id_old
                new_idx += 1
        
        self.edge_to_idx = new_edge_to_idx
        self.idx_to_edge = new_idx_to_edge
        self._num_edges -= 1
        
        # Remove from edge attributes
        if not self.edge_attributes.empty:
            self.edge_attributes = self.edge_attributes[self.edge_attributes['edge_id'] != edge_id]
    
    def remove_node(self, node_id):
        """Remove a node and all incident edges."""
        if node_id not in self.entity_to_idx:
            raise KeyError(f"Node {node_id} not found")
        
        entity_idx = self.entity_to_idx[node_id]
        
        # Find edges incident to this entity
        edges_to_remove = []
        for edge_id, (source, target, _) in self.edge_definitions.items():
            if source == node_id or target == node_id:
                edges_to_remove.append(edge_id)
        
        # Remove edges
        for edge_id in edges_to_remove:
            self.remove_edge(edge_id)
        
        # Convert to CSR for efficient row removal
        csr_matrix = self._matrix.tocsr()
        
        # Remove entity row from matrix
        mask = np.ones(self._num_entities, dtype=bool)
        mask[entity_idx] = False
        csr_matrix = csr_matrix[mask, :]
        self._matrix = csr_matrix.todok()
        
        # Update entity mappings
        del self.entity_to_idx[node_id]
        del self.entity_types[node_id]
        
        # Reindex remaining entities
        new_entity_to_idx = {}
        new_idx_to_entity = {}
        
        new_idx = 0
        for old_idx in range(self._num_entities):
            if old_idx != entity_idx:
                entity_id = self.idx_to_entity[old_idx]
                new_entity_to_idx[entity_id] = new_idx
                new_idx_to_entity[new_idx] = entity_id
                new_idx += 1
        
        self.entity_to_idx = new_entity_to_idx
        self.idx_to_entity = new_idx_to_entity
        self._num_entities -= 1
        
        # Remove from node attributes
        if not self.node_attributes.empty:
            self.node_attributes = self.node_attributes[self.node_attributes['node_id'] != node_id]
    
    def has_edge(self, source, target, edge_id=None):
        """Check if edge exists. If edge_id specified, check that specific edge."""
        if edge_id:
            return edge_id in self.edge_to_idx
        
        # Check any edge between source and target
        for eid, (src, tgt, _) in self.edge_definitions.items():
            if src == source and tgt == target:
                return True
        return False
    
    def get_edge_ids(self, source, target):
        """Get all edge IDs between two nodes (for parallel edges)."""
        edge_ids = []
        for eid, (src, tgt, _) in self.edge_definitions.items():
            if src == source and tgt == target:
                edge_ids.append(eid)
        return edge_ids
    
    def neighbors(self, entity_id):
        """Get neighbors of a node or edge entity."""
        if entity_id not in self.entity_to_idx:
            return []
        
        neighbors = []
        for edge_id, (source, target, _) in self.edge_definitions.items():
            if source == entity_id:
                neighbors.append(target)
            elif target == entity_id and not self.directed:
                neighbors.append(source)
        
        return list(set(neighbors))  # Remove duplicates
    
    def degree(self, entity_id):
        """Get degree of a node or edge entity."""
        if entity_id not in self.entity_to_idx:
            return 0
        
        entity_idx = self.entity_to_idx[entity_id]
        row = self._matrix.getrow(entity_idx)
        return len(row.nonzero()[1])
    
    def nodes(self):
        """Get all node IDs (excluding edge entities)."""
        return [eid for eid, etype in self.entity_types.items() if etype == 'node']
    
    def edges(self):
        """Get all edge IDs."""
        return list(self.edge_to_idx.keys())
    
    def edge_list(self):
        """Get list of (source, target, edge_id, weight) tuples."""
        edges = []
        for edge_id, (source, target, edge_type) in self.edge_definitions.items():
            weight = self.edge_weights[edge_id]
            edges.append((source, target, edge_id, weight))
        return edges
    
    def number_of_nodes(self):
        """Get number of nodes (excluding edge entities)."""
        return len([e for e in self.entity_types.values() if e == 'node'])
    
    def number_of_edges(self):
        """Get number of edges."""
        return self._num_edges
    
    def copy(self):
        """Create a deep copy of the graph."""
        new_graph = IncidenceGraph(directed=self.directed)
        
        # Copy graph attributes
        new_graph.graph_attributes = self.graph_attributes.copy()
        
        # Copy all nodes with attributes
        for node_id in self.nodes():
            node_attrs = {}
            if not self.node_attributes.empty:
                node_row = self.node_attributes[self.node_attributes['node_id'] == node_id]
                if not node_row.empty:
                    node_attrs = node_row.iloc[0].to_dict()
                    del node_attrs['node_id']  # Remove the ID from attributes
            new_graph.add_node(node_id, **node_attrs)
        
        # Copy all edges with attributes
        for edge_id, (source, target, edge_type) in self.edge_definitions.items():
            weight = self.edge_weights[edge_id]
            edge_attrs = {}
            if not self.edge_attributes.empty:
                edge_row = self.edge_attributes[self.edge_attributes['edge_id'] == edge_id]
                if not edge_row.empty:
                    edge_attrs = edge_row.iloc[0].to_dict()
                    # Remove metadata from attributes
                    for key in ['edge_id', 'source', 'target', 'weight', 'edge_type']:
                        edge_attrs.pop(key, None)
            
            new_graph.add_edge(source, target, weight=weight, edge_id=edge_id, 
                             edge_type=edge_type, **edge_attrs)
        
        return new_graph
    
    def memory_usage(self):
        """Estimate memory usage in bytes."""
        matrix_bytes = self._matrix.nnz * (4 + 4 + 4)  # data + row_ind + col_ind
        dict_bytes = (len(self.entity_to_idx) + len(self.edge_to_idx) + len(self.edge_weights)) * 100
        df_bytes = self.node_attributes.memory_usage(deep=True).sum() if not self.node_attributes.empty else 0
        df_bytes += self.edge_attributes.memory_usage(deep=True).sum() if not self.edge_attributes.empty else 0
        return matrix_bytes + dict_bytes + df_bytes
    
    def get_node_attribute(self, node_id, attribute):
        """Get specific node attribute."""
        if self.node_attributes.empty:
            return None
        row = self.node_attributes[self.node_attributes['node_id'] == node_id]
        if row.empty:
            return None
        return row.iloc[0].get(attribute)
    
    def get_edge_attribute(self, edge_id, attribute):
        """Get specific edge attribute."""
        if self.edge_attributes.empty:
            return None
        row = self.edge_attributes[self.edge_attributes['edge_id'] == edge_id]
        if row.empty:
            return None
        return row.iloc[0].get(attribute)
    
    def set_graph_attribute(self, key, value):
        """Set graph-level attribute."""
        self.graph_attributes[key] = value
    
    def get_graph_attribute(self, key, default=None):
        """Get graph-level attribute."""
        return self.graph_attributes.get(key, default)