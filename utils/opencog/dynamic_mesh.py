"""
Dynamic Mesh Integration for ECAN Attention Allocation

This module implements the dynamic mesh topology that integrates ECAN attention
allocation with the existing tensor fragment architecture, enabling real-time
bidirectional synchronization between attention values and hypergraph patterns.

The dynamic mesh provides:
- Bidirectional attention-hypergraph synchronization
- Dynamic topology updates based on attention flow
- Resource mesh reconfiguration for optimal allocation
- Emergent pattern recognition through attention clustering
"""

import asyncio
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import networkx as nx
from enum import Enum
import threading

from .ecan_attention import ECANAttentionManager, AttentionValue, ResourceAllocation
from .tensor_fragments import TensorFragment, TensorSignature


class MeshTopology(Enum):
    """Types of dynamic mesh topologies."""
    HIERARCHICAL = "hierarchical"      # Tree-like attention hierarchy
    LATERAL = "lateral"                # Peer-to-peer attention network
    HUB_SPOKE = "hub_spoke"           # Central attention hubs
    SMALL_WORLD = "small_world"       # Small-world network topology
    SCALE_FREE = "scale_free"         # Scale-free attention network


@dataclass
class AttentionFlow:
    """
    Represents attention flow between nodes in the dynamic mesh.
    """
    source_node: str
    target_node: str
    flow_strength: float
    flow_type: str = "spreading"  # spreading, focusing, inhibiting
    timestamp: float = field(default_factory=time.time)
    persistence: float = 1.0  # How long this flow persists
    
    def decay(self, decay_rate: float = 0.95):
        """Apply decay to the attention flow."""
        self.flow_strength *= decay_rate
        self.persistence *= decay_rate


@dataclass
class MeshNode:
    """
    Represents a node in the dynamic attention mesh.
    """
    node_id: str
    attention_value: AttentionValue
    tensor_fragment: Optional[TensorFragment] = None
    connections: Set[str] = field(default_factory=set)
    incoming_flows: List[AttentionFlow] = field(default_factory=list)
    outgoing_flows: List[AttentionFlow] = field(default_factory=list)
    mesh_position: Tuple[float, float] = (0.0, 0.0)  # 2D position for visualization
    cluster_id: Optional[str] = None
    
    def get_total_incoming_attention(self) -> float:
        """Calculate total incoming attention flow."""
        return sum(flow.flow_strength for flow in self.incoming_flows)
    
    def get_total_outgoing_attention(self) -> float:
        """Calculate total outgoing attention flow."""
        return sum(flow.flow_strength for flow in self.outgoing_flows)
    
    def get_attention_balance(self) -> float:
        """Calculate attention balance (incoming - outgoing)."""
        return self.get_total_incoming_attention() - self.get_total_outgoing_attention()


class DynamicAttentionMesh:
    """
    Dynamic mesh for attention allocation and resource distribution.
    
    This system maintains a dynamic network topology that adapts based on
    attention flow patterns, enabling efficient resource allocation and
    emergent cognitive pattern recognition.
    """
    
    def __init__(self, attention_manager: ECANAttentionManager, 
                 tensor_arch=None, initial_topology: MeshTopology = MeshTopology.SMALL_WORLD):
        """
        Initialize the dynamic attention mesh.
        
        Args:
            attention_manager: ECAN attention manager
            tensor_arch: Tensor fragment architecture
            initial_topology: Initial mesh topology
        """
        self.attention_manager = attention_manager
        self.tensor_arch = tensor_arch
        self.topology_type = initial_topology
        
        # Core mesh components
        self.nodes: Dict[str, MeshNode] = {}
        self.attention_flows: List[AttentionFlow] = []
        self.mesh_graph = nx.Graph()
        
        # Dynamic reconfiguration parameters
        self.topology_update_threshold = 0.1  # Threshold for topology changes
        self.flow_decay_rate = 0.98
        self.connection_strength_threshold = 0.05
        self.max_connections_per_node = 10
        
        # Clustering and pattern recognition
        self.attention_clusters: Dict[str, Set[str]] = {}
        self.pattern_templates: Dict[str, Dict] = {}
        self.emergent_patterns: List[Dict] = []
        
        # Performance tracking
        self.mesh_updates = 0
        self.topology_reconfigurations = 0
        self.attention_propagation_efficiency = 0.0
        
        # Threading for async operations
        self.running = False
        self.mesh_thread = None
        self.lock = threading.RLock()
        
        # Callbacks for mesh events
        self.topology_change_callbacks: List[Callable] = []
        self.pattern_discovery_callbacks: List[Callable] = []
        self.attention_flow_callbacks: List[Callable] = []
    
    def add_node(self, node_id: str, attention_value: AttentionValue = None,
                tensor_fragment: TensorFragment = None) -> bool:
        """
        Add a node to the dynamic mesh.
        
        Args:
            node_id: Unique identifier for the node
            attention_value: Initial attention value
            tensor_fragment: Associated tensor fragment
            
        Returns:
            True if node added successfully
        """
        with self.lock:
            if node_id in self.nodes:
                return False
            
            if attention_value is None:
                attention_value = AttentionValue()
            
            # Create mesh node
            node = MeshNode(
                node_id=node_id,
                attention_value=attention_value,
                tensor_fragment=tensor_fragment
            )
            
            self.nodes[node_id] = node
            self.mesh_graph.add_node(node_id, attention=attention_value.sti)
            
            # Update topology if needed
            self._update_node_connections(node_id)
            
            return True
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a node from the mesh."""
        with self.lock:
            if node_id not in self.nodes:
                return False
            
            # Clean up connections
            node = self.nodes[node_id]
            for connected_id in node.connections:
                if connected_id in self.nodes:
                    self.nodes[connected_id].connections.discard(node_id)
            
            # Remove from graph and data structures
            self.mesh_graph.remove_node(node_id)
            del self.nodes[node_id]
            
            # Clean up attention flows
            self.attention_flows = [
                flow for flow in self.attention_flows
                if flow.source_node != node_id and flow.target_node != node_id
            ]
            
            return True
    
    def update_node_attention(self, node_id: str, new_attention: AttentionValue) -> bool:
        """
        Update attention value for a node and propagate changes.
        
        Args:
            node_id: Node identifier
            new_attention: New attention value
            
        Returns:
            True if updated successfully
        """
        with self.lock:
            if node_id not in self.nodes:
                return False
            
            old_attention = self.nodes[node_id].attention_value
            self.nodes[node_id].attention_value = new_attention
            
            # Update graph node attributes
            self.mesh_graph.nodes[node_id]['attention'] = new_attention.sti
            
            # Synchronize with ECAN attention manager
            self.attention_manager.set_attention_value(node_id, new_attention)
            
            # Synchronize with tensor fragment if available
            if self.nodes[node_id].tensor_fragment:
                self._sync_attention_to_tensor(node_id, new_attention)
            
            # Trigger attention flow propagation if significant change
            sti_change = abs(new_attention.sti - old_attention.sti)
            if sti_change > self.topology_update_threshold:
                self._propagate_attention_change(node_id, sti_change)
            
            return True
    
    def create_attention_flow(self, source_id: str, target_id: str, 
                            flow_strength: float, flow_type: str = "spreading") -> bool:
        """
        Create an attention flow between two nodes.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            flow_strength: Strength of the attention flow
            flow_type: Type of flow (spreading, focusing, inhibiting)
            
        Returns:
            True if flow created successfully
        """
        with self.lock:
            if source_id not in self.nodes or target_id not in self.nodes:
                return False
            
            # Create attention flow
            flow = AttentionFlow(
                source_node=source_id,
                target_node=target_id,
                flow_strength=flow_strength,
                flow_type=flow_type
            )
            
            self.attention_flows.append(flow)
            
            # Add to node flow lists
            self.nodes[source_id].outgoing_flows.append(flow)
            self.nodes[target_id].incoming_flows.append(flow)
            
            # Update graph edge
            if self.mesh_graph.has_edge(source_id, target_id):
                current_weight = self.mesh_graph[source_id][target_id].get('weight', 0)
                self.mesh_graph[source_id][target_id]['weight'] = current_weight + flow_strength
            else:
                self.mesh_graph.add_edge(source_id, target_id, weight=flow_strength)
            
            # Trigger callbacks
            for callback in self.attention_flow_callbacks:
                try:
                    callback(flow)
                except Exception as e:
                    print(f"Attention flow callback error: {e}")
            
            return True
    
    def propagate_attention(self, iterations: int = 3) -> Dict[str, float]:
        """
        Propagate attention through the mesh network.
        
        Args:
            iterations: Number of propagation iterations
            
        Returns:
            Dictionary of node_id -> final_attention_value
        """
        results = {}
        
        with self.lock:
            for iteration in range(iterations):
                attention_changes = {}
                
                # Calculate attention changes for each node
                for node_id, node in self.nodes.items():
                    total_incoming = 0.0
                    total_outgoing = 0.0
                    
                    # Sum incoming attention flows
                    for flow in node.incoming_flows:
                        if flow.flow_type == "spreading":
                            total_incoming += flow.flow_strength
                        elif flow.flow_type == "inhibiting":
                            total_incoming -= flow.flow_strength
                    
                    # Sum outgoing attention flows  
                    for flow in node.outgoing_flows:
                        total_outgoing += flow.flow_strength
                    
                    # Calculate net change
                    net_change = total_incoming - (total_outgoing * 0.1)  # Small outgoing cost
                    attention_changes[node_id] = net_change
                
                # Apply attention changes
                for node_id, change in attention_changes.items():
                    if abs(change) > 0.01:  # Only apply significant changes
                        node = self.nodes[node_id]
                        new_sti = node.attention_value.sti + change
                        
                        new_attention = AttentionValue(
                            sti=new_sti,
                            lti=node.attention_value.lti,
                            vlti=node.attention_value.vlti,
                            confidence=node.attention_value.confidence
                        )
                        
                        self.update_node_attention(node_id, new_attention)
                        results[node_id] = new_sti
                
                # Decay attention flows
                self._decay_attention_flows()
        
        return results
    
    def detect_attention_clusters(self, min_cluster_size: int = 3) -> Dict[str, Set[str]]:
        """
        Detect clusters of nodes with high attention correlation.
        
        Args:
            min_cluster_size: Minimum size for a valid cluster
            
        Returns:
            Dictionary mapping cluster_id to set of node_ids
        """
        with self.lock:
            # Use attention values as node features for clustering
            attention_matrix = []
            node_list = list(self.nodes.keys())
            
            for node_id in node_list:
                node = self.nodes[node_id]
                attention_vector = [
                    node.attention_value.sti,
                    node.attention_value.lti,
                    node.get_total_incoming_attention(),
                    node.get_total_outgoing_attention()
                ]
                attention_matrix.append(attention_vector)
            
            if len(attention_matrix) < min_cluster_size:
                return {}
            
            # Simple clustering based on attention similarity
            clusters = {}
            cluster_id = 0
            
            attention_array = np.array(attention_matrix)
            if attention_array.size == 0:
                return {}
                
            # Normalize attention values
            if attention_array.std() > 0:
                attention_array = (attention_array - attention_array.mean()) / attention_array.std()
            
            # Find clusters using simple distance-based grouping
            unassigned = set(range(len(node_list)))
            
            while unassigned:
                # Start new cluster with highest attention node
                max_attention_idx = max(unassigned, key=lambda i: self.nodes[node_list[i]].attention_value.sti)
                current_cluster = {max_attention_idx}
                unassigned.remove(max_attention_idx)
                
                # Add similar nodes to cluster
                base_vector = attention_array[max_attention_idx]
                
                for candidate_idx in list(unassigned):
                    candidate_vector = attention_array[candidate_idx]
                    
                    # Calculate cosine similarity
                    if np.linalg.norm(base_vector) > 0 and np.linalg.norm(candidate_vector) > 0:
                        similarity = np.dot(base_vector, candidate_vector) / (
                            np.linalg.norm(base_vector) * np.linalg.norm(candidate_vector)
                        )
                        
                        if similarity > 0.7:  # High similarity threshold
                            current_cluster.add(candidate_idx)
                            unassigned.remove(candidate_idx)
                
                # Create cluster if large enough
                if len(current_cluster) >= min_cluster_size:
                    cluster_nodes = {node_list[i] for i in current_cluster}
                    clusters[f"cluster_{cluster_id}"] = cluster_nodes
                    cluster_id += 1
                    
                    # Update node cluster assignments
                    for node_id in cluster_nodes:
                        self.nodes[node_id].cluster_id = f"cluster_{cluster_id-1}"
            
            self.attention_clusters = clusters
            return clusters
    
    def reconfigure_topology(self, new_topology: MeshTopology = None) -> bool:
        """
        Reconfigure the mesh topology based on attention patterns.
        
        Args:
            new_topology: Target topology type (None for automatic selection)
            
        Returns:
            True if reconfiguration successful
        """
        with self.lock:
            if new_topology is None:
                new_topology = self._select_optimal_topology()
            
            old_topology = self.topology_type
            self.topology_type = new_topology
            
            # Clear existing connections
            for node in self.nodes.values():
                node.connections.clear()
            
            self.mesh_graph.clear_edges()
            
            # Apply new topology
            if new_topology == MeshTopology.HIERARCHICAL:
                self._create_hierarchical_topology()
            elif new_topology == MeshTopology.LATERAL:
                self._create_lateral_topology()
            elif new_topology == MeshTopology.HUB_SPOKE:
                self._create_hub_spoke_topology()
            elif new_topology == MeshTopology.SMALL_WORLD:
                self._create_small_world_topology()
            elif new_topology == MeshTopology.SCALE_FREE:
                self._create_scale_free_topology()
            
            self.topology_reconfigurations += 1
            
            # Trigger callbacks
            for callback in self.topology_change_callbacks:
                try:
                    callback(old_topology, new_topology)
                except Exception as e:
                    print(f"Topology change callback error: {e}")
            
            return True
    
    def get_mesh_statistics(self) -> Dict[str, Any]:
        """Get comprehensive mesh statistics."""
        with self.lock:
            # Calculate network metrics
            if len(self.mesh_graph.nodes()) > 1:
                try:
                    avg_clustering = nx.average_clustering(self.mesh_graph)
                    density = nx.density(self.mesh_graph)
                    if nx.is_connected(self.mesh_graph):
                        avg_path_length = nx.average_shortest_path_length(self.mesh_graph)
                    else:
                        avg_path_length = float('inf')
                except:
                    avg_clustering = 0.0
                    density = 0.0
                    avg_path_length = float('inf')
            else:
                avg_clustering = 0.0
                density = 0.0
                avg_path_length = 0.0
            
            # Calculate attention statistics
            total_sti = sum(node.attention_value.sti for node in self.nodes.values())
            total_lti = sum(node.attention_value.lti for node in self.nodes.values())
            active_flows = len([f for f in self.attention_flows if f.flow_strength > 0.01])
            
            return {
                'total_nodes': len(self.nodes),
                'total_edges': len(self.mesh_graph.edges()),
                'active_attention_flows': active_flows,
                'total_attention_flows': len(self.attention_flows),
                'topology_type': self.topology_type.value,
                'mesh_updates': self.mesh_updates,
                'topology_reconfigurations': self.topology_reconfigurations,
                'attention_clusters': len(self.attention_clusters),
                'total_sti': total_sti,
                'total_lti': total_lti,
                'network_density': density,
                'average_clustering': avg_clustering,
                'average_path_length': avg_path_length,
                'attention_propagation_efficiency': self.attention_propagation_efficiency
            }
    
    def start_dynamic_updates(self):
        """Start the dynamic mesh update daemon."""
        if self.running:
            return
        
        self.running = True
        self.mesh_thread = threading.Thread(target=self._mesh_update_loop)
        self.mesh_thread.daemon = True
        self.mesh_thread.start()
    
    def stop_dynamic_updates(self):
        """Stop the dynamic mesh update daemon."""
        self.running = False
        if self.mesh_thread:
            self.mesh_thread.join()
    
    # Private helper methods
    
    def _update_node_connections(self, node_id: str):
        """Update connections for a node based on current topology."""
        if self.topology_type == MeshTopology.SMALL_WORLD:
            self._connect_small_world_node(node_id)
        elif self.topology_type == MeshTopology.SCALE_FREE:
            self._connect_scale_free_node(node_id)
        # Add other topology-specific connection logic
    
    def _propagate_attention_change(self, node_id: str, change_magnitude: float):
        """Propagate attention changes to connected nodes."""
        node = self.nodes[node_id]
        propagation_strength = change_magnitude * 0.1  # 10% propagation
        
        for connected_id in node.connections:
            if connected_id in self.nodes:
                self.create_attention_flow(
                    node_id, connected_id, propagation_strength, "spreading"
                )
    
    def _sync_attention_to_tensor(self, node_id: str, attention_value: AttentionValue):
        """Synchronize attention value to tensor fragment."""
        node = self.nodes[node_id]
        if not node.tensor_fragment:
            return
        
        # Update tensor signature with attention values
        norm_sti, norm_lti, confidence = attention_value.normalize_for_tensor()
        
        node.tensor_fragment.signature.salience = norm_sti
        node.tensor_fragment.signature.context = norm_lti
        node.tensor_fragment.signature.autonomy_index = confidence
        
        # Update in atomspace if available
        if self.tensor_arch and self.tensor_arch.atomspace and node.tensor_fragment.atom_handle:
            atom = self.tensor_arch.atomspace.get_atom(node.tensor_fragment.atom_handle)
            if atom and 'tensor_fragment' in atom:
                atom['tensor_fragment']['signature'] = {
                    'modality': node.tensor_fragment.signature.modality,
                    'depth': node.tensor_fragment.signature.depth,
                    'context': norm_lti,
                    'salience': norm_sti,
                    'autonomy_index': confidence
                }
    
    def _decay_attention_flows(self):
        """Apply decay to attention flows."""
        decayed_flows = []
        
        for flow in self.attention_flows:
            flow.decay(self.flow_decay_rate)
            
            if flow.flow_strength > 0.001:  # Keep flows above threshold
                decayed_flows.append(flow)
        
        self.attention_flows = decayed_flows
    
    def _select_optimal_topology(self) -> MeshTopology:
        """Select optimal topology based on current attention patterns."""
        # Simple heuristic: use clustering coefficient to select topology
        if len(self.nodes) < 5:
            return MeshTopology.LATERAL
        
        clusters = self.detect_attention_clusters()
        if len(clusters) > len(self.nodes) * 0.3:  # Many clusters
            return MeshTopology.HUB_SPOKE
        elif len(clusters) > 2:
            return MeshTopology.HIERARCHICAL
        else:
            return MeshTopology.SMALL_WORLD
    
    def _create_hierarchical_topology(self):
        """Create hierarchical mesh topology."""
        node_list = sorted(self.nodes.keys(), 
                          key=lambda x: self.nodes[x].attention_value.sti, 
                          reverse=True)
        
        # Connect each node to the next higher attention node
        for i in range(1, len(node_list)):
            parent_id = node_list[i-1]
            child_id = node_list[i]
            
            self.nodes[parent_id].connections.add(child_id)
            self.nodes[child_id].connections.add(parent_id)
            self.mesh_graph.add_edge(parent_id, child_id)
    
    def _create_lateral_topology(self):
        """Create lateral (fully connected) mesh topology."""
        node_list = list(self.nodes.keys())
        
        for i, node_id in enumerate(node_list):
            for j, other_id in enumerate(node_list):
                if i != j:
                    self.nodes[node_id].connections.add(other_id)
                    if not self.mesh_graph.has_edge(node_id, other_id):
                        self.mesh_graph.add_edge(node_id, other_id)
    
    def _create_hub_spoke_topology(self):
        """Create hub-and-spoke mesh topology."""
        node_list = list(self.nodes.keys())
        if not node_list:
            return
        
        # Select highest attention node as hub
        hub_id = max(node_list, key=lambda x: self.nodes[x].attention_value.sti)
        
        # Connect all other nodes to hub
        for node_id in node_list:
            if node_id != hub_id:
                self.nodes[hub_id].connections.add(node_id)
                self.nodes[node_id].connections.add(hub_id)
                self.mesh_graph.add_edge(hub_id, node_id)
    
    def _create_small_world_topology(self):
        """Create small-world mesh topology."""
        node_list = list(self.nodes.keys())
        n = len(node_list)
        
        if n < 3:
            self._create_lateral_topology()
            return
        
        # Create ring topology first
        for i in range(n):
            next_i = (i + 1) % n
            node_id = node_list[i]
            next_id = node_list[next_i]
            
            self.nodes[node_id].connections.add(next_id)
            self.nodes[next_id].connections.add(node_id)
            self.mesh_graph.add_edge(node_id, next_id)
        
        # Add random shortcuts based on attention similarity
        for i in range(n):
            if np.random.random() < 0.3:  # 30% chance of shortcut
                j = np.random.randint(0, n)
                if i != j and abs(i - j) > 1:
                    node_i = node_list[i]
                    node_j = node_list[j]
                    
                    self.nodes[node_i].connections.add(node_j)
                    self.nodes[node_j].connections.add(node_i)
                    self.mesh_graph.add_edge(node_i, node_j)
    
    def _create_scale_free_topology(self):
        """Create scale-free mesh topology using preferential attachment."""
        node_list = list(self.nodes.keys())
        n = len(node_list)
        
        if n < 2:
            return
        
        # Start with two connected nodes
        self.nodes[node_list[0]].connections.add(node_list[1])
        self.nodes[node_list[1]].connections.add(node_list[0])
        self.mesh_graph.add_edge(node_list[0], node_list[1])
        
        # Add remaining nodes with preferential attachment
        for i in range(2, n):
            new_node = node_list[i]
            
            # Calculate connection probabilities based on degree and attention
            connection_probs = []
            for j in range(i):
                existing_node = node_list[j]
                degree = len(self.nodes[existing_node].connections)
                attention = self.nodes[existing_node].attention_value.sti
                prob = (degree + 1) * (attention + 1)
                connection_probs.append(prob)
            
            # Normalize probabilities
            total_prob = sum(connection_probs)
            if total_prob > 0:
                connection_probs = [p / total_prob for p in connection_probs]
            
                # Select nodes to connect to (preferential attachment)
                num_connections = min(2, i)  # Connect to 1-2 existing nodes
                selected_indices = np.random.choice(
                    i, size=num_connections, replace=False, p=connection_probs
                )
                
                for idx in selected_indices:
                    existing_node = node_list[idx]
                    self.nodes[new_node].connections.add(existing_node)
                    self.nodes[existing_node].connections.add(new_node)
                    self.mesh_graph.add_edge(new_node, existing_node)
    
    def _connect_small_world_node(self, node_id: str):
        """Connect a node using small-world principles."""
        # Connect to nodes with similar attention values
        node_attention = self.nodes[node_id].attention_value.sti
        
        candidates = []
        for other_id, other_node in self.nodes.items():
            if other_id != node_id:
                attention_diff = abs(other_node.attention_value.sti - node_attention)
                candidates.append((other_id, attention_diff))
        
        # Sort by attention similarity and connect to closest matches
        candidates.sort(key=lambda x: x[1])
        max_connections = min(3, len(candidates))
        
        for i in range(max_connections):
            other_id = candidates[i][0]
            self.nodes[node_id].connections.add(other_id)
            self.nodes[other_id].connections.add(node_id)
            self.mesh_graph.add_edge(node_id, other_id)
    
    def _connect_scale_free_node(self, node_id: str):
        """Connect a node using scale-free preferential attachment."""
        # Connect based on degree and attention (preferential attachment)
        candidates = []
        for other_id, other_node in self.nodes.items():
            if other_id != node_id:
                degree = len(other_node.connections)
                attention = other_node.attention_value.sti
                score = (degree + 1) * (attention + 1)
                candidates.append((other_id, score))
        
        if not candidates:
            return
        
        # Sort by score (degree * attention) and connect to top candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        max_connections = min(2, len(candidates))
        
        for i in range(max_connections):
            other_id = candidates[i][0]
            self.nodes[node_id].connections.add(other_id)
            self.nodes[other_id].connections.add(node_id)
            self.mesh_graph.add_edge(node_id, other_id)
    
    def _mesh_update_loop(self):
        """Background daemon for dynamic mesh updates."""
        while self.running:
            try:
                # Synchronize with ECAN attention manager
                self._sync_with_ecan_manager()
                
                # Propagate attention through the mesh
                self.propagate_attention(iterations=1)
                
                # Detect attention clusters
                self.detect_attention_clusters()
                
                # Check if topology reconfiguration is needed
                if self.mesh_updates % 10 == 0:  # Every 10 updates
                    self._check_topology_reconfiguration()
                
                self.mesh_updates += 1
                
                # Sleep for next cycle
                time.sleep(0.5)  # 0.5 second cycle
                
            except Exception as e:
                print(f"Mesh update daemon error: {e}")
    
    def _sync_with_ecan_manager(self):
        """Synchronize mesh nodes with ECAN attention manager."""
        with self.lock:
            # Update attention values from ECAN manager
            for node_id in self.nodes:
                ecan_attention = self.attention_manager.get_attention_value(node_id)
                if ecan_attention:
                    self.nodes[node_id].attention_value = ecan_attention
                    self.mesh_graph.nodes[node_id]['attention'] = ecan_attention.sti
    
    def _check_topology_reconfiguration(self):
        """Check if topology reconfiguration is beneficial."""
        current_efficiency = self._calculate_attention_efficiency()
        
        # If efficiency is low, try a different topology
        if current_efficiency < 0.5:
            optimal_topology = self._select_optimal_topology()
            if optimal_topology != self.topology_type:
                self.reconfigure_topology(optimal_topology)
    
    def _calculate_attention_efficiency(self) -> float:
        """Calculate current attention propagation efficiency."""
        if not self.attention_flows:
            return 0.0
        
        # Measure how well attention flows reach high-attention nodes
        total_flow_efficiency = 0.0
        valid_flows = 0
        
        for flow in self.attention_flows:
            if flow.target_node in self.nodes:
                target_attention = self.nodes[flow.target_node].attention_value.sti
                # Higher efficiency if flow reaches high-attention nodes
                efficiency = min(flow.flow_strength * target_attention / 100.0, 1.0)
                total_flow_efficiency += efficiency
                valid_flows += 1
        
        if valid_flows == 0:
            return 0.0
        
        efficiency = total_flow_efficiency / valid_flows
        self.attention_propagation_efficiency = efficiency
        return efficiency