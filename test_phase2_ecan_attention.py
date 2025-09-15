#!/usr/bin/env python3
"""
Comprehensive test suite for Phase 2: ECAN Attention Allocation & Resource Kernel

This test suite verifies the implementation of:
- ECAN Attention Manager functionality
- Dynamic Mesh Integration
- Resource Kernel operations
- Tensor Fragment integration
- Real-world verification scenarios

Tensor signature mapping: [tasks, attention, priority, resources]
"""

import sys
import os
import time
import asyncio
import numpy as np
import pytest
from unittest.mock import Mock, patch

# Add the project root to Python path
sys.path.append('/home/runner/work/Archon/Archon')

from utils.opencog import opencog
from utils.opencog.ecan_attention import (
    ECANAttentionManager, AttentionValue, ResourceAllocation, 
    ECANResourceKernel, create_ecan_enhanced_tensor_fragment
)
from utils.opencog.dynamic_mesh import (
    DynamicAttentionMesh, MeshTopology, AttentionFlow, MeshNode
)
from utils.opencog.tensor_fragments import (
    TensorSignature, TensorFragment, TensorFragmentArchitecture
)


class TestECANAttentionManager:
    """Test suite for ECAN Attention Manager functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.atomspace = opencog.atomspace()
        self.attention_manager = ECANAttentionManager(
            atomspace=self.atomspace,
            initial_sti_budget=1000.0,
            initial_lti_budget=1000.0
        )
    
    def test_attention_value_normalization(self):
        """Test attention value normalization for tensor integration."""
        av = AttentionValue(sti=500.0, lti=750.0, confidence=0.8)
        norm_sti, norm_lti, confidence = av.normalize_for_tensor()
        
        # STI normalization: [-1000, 1000] -> [0, 1]
        expected_sti = (500.0 + 1000) / 2000  # Should be 0.75
        assert abs(norm_sti - expected_sti) < 0.001
        
        # LTI normalization: [0, 1000] -> [0, 1]
        expected_lti = 750.0 / 1000  # Should be 0.75
        assert abs(norm_lti - expected_lti) < 0.001
        
        # Confidence should remain the same
        assert confidence == 0.8
    
    def test_attention_value_from_tensor(self):
        """Test creation of attention values from normalized tensor values."""
        av = AttentionValue.from_tensor_values(0.75, 0.75, 0.8)
        
        # Expected STI: (0.75 * 2000) - 1000 = 500
        assert abs(av.sti - 500.0) < 0.001
        
        # Expected LTI: 0.75 * 1000 = 750
        assert abs(av.lti - 750.0) < 0.001
        
        assert av.confidence == 0.8
    
    def test_set_and_get_attention_value(self):
        """Test setting and retrieving attention values."""
        atom_handle = "test_atom_1"
        av = AttentionValue(sti=100.0, lti=200.0, confidence=0.9)
        
        # Set attention value
        success = self.attention_manager.set_attention_value(atom_handle, av)
        assert success == True
        
        # Get attention value
        retrieved_av = self.attention_manager.get_attention_value(atom_handle)
        assert retrieved_av is not None
        assert retrieved_av.sti == 100.0
        assert retrieved_av.lti == 200.0
        assert retrieved_av.confidence == 0.9
    
    def test_resource_allocation_and_deallocation(self):
        """Test resource allocation functionality."""
        task_id = "test_task_1"
        resource_req = ResourceAllocation(
            cpu_cycles=100,
            memory_allocation=50,
            priority_level=7,
            task_type="reasoning"
        )
        
        # Allocate resources
        success = self.attention_manager.allocate_resources(task_id, resource_req)
        assert success == True
        
        # Check allocation
        assert self.attention_manager.allocated_cpu == 100
        assert self.attention_manager.allocated_memory == 50
        
        # Deallocate resources
        success = self.attention_manager.deallocate_resources(task_id)
        assert success == True
        
        # Check deallocation
        assert self.attention_manager.allocated_cpu == 0
        assert self.attention_manager.allocated_memory == 0
    
    def test_resource_allocation_budget_constraint(self):
        """Test resource allocation respects budget constraints."""
        task_id = "test_task_overflow"
        resource_req = ResourceAllocation(
            cpu_cycles=2000,  # Exceeds budget
            memory_allocation=50,
            priority_level=5
        )
        
        # Should fail due to budget constraint
        success = self.attention_manager.allocate_resources(task_id, resource_req)
        assert success == False
        
        # No resources should be allocated
        assert self.attention_manager.allocated_cpu == 0
        assert self.attention_manager.allocated_memory == 0
    
    def test_attention_spreading(self):
        """Test attention spreading mechanism."""
        # Set up connected atoms in atomspace
        atom1 = self.atomspace.add_node("ConceptNode", "atom1")
        atom2 = self.atomspace.add_node("ConceptNode", "atom2")
        atom3 = self.atomspace.add_node("ConceptNode", "atom3")
        
        # Create links between atoms
        link1 = self.atomspace.add_link("InheritanceLink", [atom1, atom2])
        link2 = self.atomspace.add_link("InheritanceLink", [atom2, atom3])
        
        # Set high attention for source atom
        source_av = AttentionValue(sti=200.0, lti=100.0)
        self.attention_manager.set_attention_value(atom1, source_av)
        
        # Spread attention
        affected_count = self.attention_manager.spread_attention(atom1, spread_amount=50.0)
        
        # Should affect connected atoms
        assert affected_count > 0
        
        # Check that connected atoms received attention
        atom2_av = self.attention_manager.get_attention_value(atom2)
        if atom2_av:
            assert atom2_av.sti > 0  # Should have received some attention
    
    def test_high_sti_atoms_retrieval(self):
        """Test retrieval of high STI atoms."""
        # Create atoms with different STI values
        atoms_data = [
            ("atom1", 500.0),
            ("atom2", 300.0),
            ("atom3", 800.0),
            ("atom4", 100.0),
            ("atom5", 600.0)
        ]
        
        for atom_id, sti_value in atoms_data:
            av = AttentionValue(sti=sti_value, lti=50.0)
            self.attention_manager.set_attention_value(atom_id, av)
        
        # Get top 3 high STI atoms
        high_sti_atoms = self.attention_manager.get_high_sti_atoms(count=3)
        
        assert len(high_sti_atoms) == 3
        
        # Should be sorted by STI in descending order
        assert high_sti_atoms[0][0] == "atom3"  # Highest STI (800)
        assert high_sti_atoms[1][0] == "atom5"  # Second highest (600)
        assert high_sti_atoms[2][0] == "atom1"  # Third highest (500)
    
    def test_attention_statistics(self):
        """Test attention statistics calculation."""
        # Add some atoms with attention values
        av1 = AttentionValue(sti=100.0, lti=200.0)
        av2 = AttentionValue(sti=150.0, lti=100.0)
        
        self.attention_manager.set_attention_value("atom1", av1)
        self.attention_manager.set_attention_value("atom2", av2)
        
        # Allocate some resources
        resource_req = ResourceAllocation(cpu_cycles=100, memory_allocation=50)
        self.attention_manager.allocate_resources("task1", resource_req)
        
        # Get statistics
        stats = self.attention_manager.get_attention_statistics()
        
        assert stats['total_atoms'] == 2
        assert stats['total_sti'] == 250.0  # 100 + 150
        assert stats['total_lti'] == 300.0  # 200 + 100
        assert stats['cpu_utilization'] == 0.1  # 100/1000
        assert stats['memory_utilization'] == 0.05  # 50/1000
        assert stats['active_tasks'] == 1


class TestECANResourceKernel:
    """Test suite for ECAN Resource Kernel functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.atomspace = opencog.atomspace()
        self.attention_manager = ECANAttentionManager(atomspace=self.atomspace)
        self.resource_kernel = ECANResourceKernel(self.attention_manager)
    
    @pytest.mark.asyncio
    async def test_cognitive_task_scheduling(self):
        """Test cognitive task scheduling functionality."""
        task_executed = False
        
        async def test_task():
            nonlocal task_executed
            task_executed = True
            return "task_completed"
        
        task_id = "test_cognitive_task"
        resource_req = ResourceAllocation(
            cpu_cycles=50,
            memory_allocation=25,
            priority_level=8,
            task_type="reasoning"
        )
        
        # Schedule the task
        success = await self.resource_kernel.schedule_cognitive_task(
            task_id, test_task, resource_req
        )
        
        assert success == True
        
        # Wait for task execution
        await asyncio.sleep(0.1)  # Small delay for task execution
        
        # Check that task was executed
        assert task_executed == True
        
        # Check task history
        assert self.resource_kernel.total_tasks_processed >= 1
    
    @pytest.mark.asyncio
    async def test_task_with_attention_boost(self):
        """Test task scheduling with attention boost."""
        async def test_task():
            return "boosted_task_completed"
        
        task_id = "boosted_task"
        resource_req = ResourceAllocation(cpu_cycles=30, memory_allocation=15, priority_level=5)
        
        # Schedule task with attention boost
        success = await self.resource_kernel.schedule_cognitive_task(
            task_id, test_task, resource_req, attention_boost=100.0
        )
        
        assert success == True
        
        # Check that attention was boosted
        attention_value = self.attention_manager.get_attention_value(task_id)
        assert attention_value is not None
        assert attention_value.sti >= 100.0
    
    def test_kernel_statistics(self):
        """Test resource kernel statistics."""
        stats = self.resource_kernel.get_kernel_statistics()
        
        assert 'active_tasks' in stats
        assert 'total_tasks_processed' in stats
        assert 'average_completion_time' in stats
        assert 'resource_efficiency' in stats
        assert 'max_concurrent_tasks' in stats
        
        # Initial values should be reasonable
        assert stats['active_tasks'] >= 0
        assert stats['total_tasks_processed'] >= 0
        assert stats['max_concurrent_tasks'] > 0


class TestDynamicAttentionMesh:
    """Test suite for Dynamic Attention Mesh functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.atomspace = opencog.atomspace()
        self.attention_manager = ECANAttentionManager(atomspace=self.atomspace)
        self.tensor_arch = TensorFragmentArchitecture(self.atomspace)
        self.mesh = DynamicAttentionMesh(
            self.attention_manager, 
            self.tensor_arch,
            MeshTopology.SMALL_WORLD
        )
    
    def test_node_addition_and_removal(self):
        """Test adding and removing nodes from the mesh."""
        node_id = "test_node_1"
        attention_value = AttentionValue(sti=100.0, lti=50.0)
        
        # Add node
        success = self.mesh.add_node(node_id, attention_value)
        assert success == True
        assert node_id in self.mesh.nodes
        assert self.mesh.nodes[node_id].attention_value.sti == 100.0
        
        # Remove node
        success = self.mesh.remove_node(node_id)
        assert success == True
        assert node_id not in self.mesh.nodes
    
    def test_attention_value_update(self):
        """Test updating attention values in the mesh."""
        node_id = "test_node_update"
        initial_av = AttentionValue(sti=50.0, lti=25.0)
        updated_av = AttentionValue(sti=150.0, lti=75.0)
        
        # Add node
        self.mesh.add_node(node_id, initial_av)
        
        # Update attention value
        success = self.mesh.update_node_attention(node_id, updated_av)
        assert success == True
        
        # Verify update
        node = self.mesh.nodes[node_id]
        assert node.attention_value.sti == 150.0
        assert node.attention_value.lti == 75.0
    
    def test_attention_flow_creation(self):
        """Test creating attention flows between nodes."""
        # Add two nodes
        node1_id = "flow_source"
        node2_id = "flow_target"
        av1 = AttentionValue(sti=200.0, lti=100.0)
        av2 = AttentionValue(sti=50.0, lti=25.0)
        
        self.mesh.add_node(node1_id, av1)
        self.mesh.add_node(node2_id, av2)
        
        # Create attention flow
        success = self.mesh.create_attention_flow(
            node1_id, node2_id, flow_strength=25.0, flow_type="spreading"
        )
        
        assert success == True
        
        # Verify flow was created
        source_node = self.mesh.nodes[node1_id]
        target_node = self.mesh.nodes[node2_id]
        
        assert len(source_node.outgoing_flows) == 1
        assert len(target_node.incoming_flows) == 1
        
        flow = source_node.outgoing_flows[0]
        assert flow.source_node == node1_id
        assert flow.target_node == node2_id
        assert flow.flow_strength == 25.0
    
    def test_attention_propagation(self):
        """Test attention propagation through the mesh."""
        # Create a small network
        nodes_data = [
            ("node1", AttentionValue(sti=100.0, lti=50.0)),
            ("node2", AttentionValue(sti=20.0, lti=10.0)),
            ("node3", AttentionValue(sti=30.0, lti=15.0))
        ]
        
        for node_id, av in nodes_data:
            self.mesh.add_node(node_id, av)
        
        # Create attention flows
        self.mesh.create_attention_flow("node1", "node2", 10.0)
        self.mesh.create_attention_flow("node1", "node3", 15.0)
        
        # Propagate attention
        results = self.mesh.propagate_attention(iterations=2)
        
        # Verify propagation occurred
        assert len(results) > 0
        
        # Target nodes should have received attention
        node2 = self.mesh.nodes["node2"]
        node3 = self.mesh.nodes["node3"]
        
        # These nodes should have higher attention than initial values
        assert node2.attention_value.sti >= 20.0
        assert node3.attention_value.sti >= 30.0
    
    def test_attention_cluster_detection(self):
        """Test attention cluster detection."""
        # Create nodes with similar attention patterns
        cluster1_nodes = [
            ("cluster1_node1", AttentionValue(sti=100.0, lti=50.0)),
            ("cluster1_node2", AttentionValue(sti=95.0, lti=48.0)),
            ("cluster1_node3", AttentionValue(sti=105.0, lti=52.0))
        ]
        
        cluster2_nodes = [
            ("cluster2_node1", AttentionValue(sti=200.0, lti=100.0)),
            ("cluster2_node2", AttentionValue(sti=195.0, lti=98.0)),
            ("cluster2_node3", AttentionValue(sti=205.0, lti=102.0))
        ]
        
        # Add all nodes
        for node_id, av in cluster1_nodes + cluster2_nodes:
            self.mesh.add_node(node_id, av)
        
        # Detect clusters
        clusters = self.mesh.detect_attention_clusters(min_cluster_size=2)
        
        # Should detect clusters
        assert len(clusters) > 0
        
        # Verify cluster assignments
        for cluster_id, node_set in clusters.items():
            assert len(node_set) >= 2
    
    def test_topology_reconfiguration(self):
        """Test mesh topology reconfiguration."""
        # Add some nodes
        for i in range(5):
            node_id = f"topo_node_{i}"
            av = AttentionValue(sti=float(i * 20), lti=float(i * 10))
            self.mesh.add_node(node_id, av)
        
        # Initial topology
        initial_topology = self.mesh.topology_type
        
        # Reconfigure to different topology
        new_topology = MeshTopology.HUB_SPOKE
        success = self.mesh.reconfigure_topology(new_topology)
        
        assert success == True
        assert self.mesh.topology_type == new_topology
        assert self.mesh.topology_reconfigurations >= 1
    
    def test_mesh_statistics(self):
        """Test mesh statistics calculation."""
        # Add some nodes and connections
        for i in range(3):
            node_id = f"stats_node_{i}"
            av = AttentionValue(sti=float(i * 50), lti=float(i * 25))
            self.mesh.add_node(node_id, av)
        
        # Create some flows
        self.mesh.create_attention_flow("stats_node_0", "stats_node_1", 10.0)
        self.mesh.create_attention_flow("stats_node_1", "stats_node_2", 15.0)
        
        # Get statistics
        stats = self.mesh.get_mesh_statistics()
        
        # Verify statistics structure
        required_keys = [
            'total_nodes', 'total_edges', 'active_attention_flows',
            'topology_type', 'mesh_updates', 'total_sti', 'total_lti'
        ]
        
        for key in required_keys:
            assert key in stats
        
        # Verify reasonable values
        assert stats['total_nodes'] == 3
        assert stats['active_attention_flows'] >= 0
        assert stats['total_sti'] >= 0
        assert stats['total_lti'] >= 0


class TestTensorFragmentIntegration:
    """Test suite for ECAN-Tensor Fragment integration."""
    
    def setup_method(self):
        """Set up test environment."""
        self.atomspace = opencog.atomspace()
        self.tensor_arch = TensorFragmentArchitecture(self.atomspace)
        self.attention_manager = ECANAttentionManager(atomspace=self.atomspace)
    
    def test_ecan_enhanced_tensor_fragment_creation(self):
        """Test creation of ECAN-enhanced tensor fragments."""
        content = "test_cognitive_primitive"
        attention_value = AttentionValue(sti=150.0, lti=300.0, confidence=0.9)
        resource_req = ResourceAllocation(
            cpu_cycles=100,
            memory_allocation=50,
            priority_level=8,
            task_type="reasoning"
        )
        
        # Create enhanced tensor fragment
        fragment = create_ecan_enhanced_tensor_fragment(
            self.tensor_arch, content, attention_value, resource_req
        )
        
        assert fragment is not None
        assert fragment.content == content
        
        # Check tensor signature mapping
        signature = fragment.signature
        
        # Verify ECAN values are properly mapped
        norm_sti, norm_lti, confidence = attention_value.normalize_for_tensor()
        assert abs(signature.salience - norm_sti) < 0.001
        assert abs(signature.context - norm_lti) < 0.001
        assert abs(signature.autonomy_index - confidence) < 0.001
        
        # Verify metadata
        assert 'ecan_attention' in fragment.metadata
        assert 'resource_allocation' in fragment.metadata
        assert fragment.metadata['ecan_attention'] == attention_value
        assert fragment.metadata['resource_allocation'] == resource_req
    
    def test_resource_allocation_tensor_mapping(self):
        """Test ResourceAllocation to tensor signature mapping."""
        resource_req = ResourceAllocation(
            cpu_cycles=500,
            memory_allocation=200,
            priority_level=9,
            task_type="perception",
            allocated_time=2.5
        )
        
        # Get tensor signature
        modality, depth, salience, autonomy = resource_req.to_tensor_signature()
        
        # Verify mappings
        assert 0.0 <= modality <= 1.0
        assert 0.0 <= depth <= 1.0
        assert 0.0 <= salience <= 1.0
        assert 0.0 <= autonomy <= 1.0
        
        # Check specific mappings
        assert salience == 0.9  # priority_level 9 / 10
        assert modality == 0.5   # "perception" task type
    
    def test_bidirectional_attention_sync(self):
        """Test bidirectional synchronization between ECAN and tensor fragments."""
        # Create a tensor fragment
        signature = TensorSignature(0.3, 0.6, 0.8, 0.7, 0.9)
        content = "sync_test_primitive"
        fragment = TensorFragment(signature, content)
        
        # Encode to hypergraph
        atom_handle = fragment.encode_to_hypergraph(self.atomspace)
        
        # Create corresponding ECAN attention value
        attention_value = AttentionValue.from_tensor_values(
            signature.salience,        # STI from salience
            signature.context,         # LTI from context
            signature.autonomy_index   # Confidence from autonomy
        )
        
        # Set in attention manager
        self.attention_manager.set_attention_value(atom_handle, attention_value)
        
        # Verify synchronization
        retrieved_av = self.attention_manager.get_attention_value(atom_handle)
        assert retrieved_av is not None
        
        # Check that values match within tolerance
        norm_sti, norm_lti, confidence = retrieved_av.normalize_for_tensor()
        assert abs(norm_sti - signature.salience) < 0.001
        assert abs(norm_lti - signature.context) < 0.001
        assert abs(confidence - signature.autonomy_index) < 0.001


class TestRealWorldScenarios:
    """Test suite for real-world verification scenarios."""
    
    def setup_method(self):
        """Set up comprehensive test environment."""
        self.atomspace = opencog.atomspace()
        self.tensor_arch = TensorFragmentArchitecture(self.atomspace)
        self.attention_manager = ECANAttentionManager(
            atomspace=self.atomspace,
            initial_sti_budget=5000.0,
            initial_lti_budget=5000.0
        )
        self.resource_kernel = ECANResourceKernel(self.attention_manager)
        self.mesh = DynamicAttentionMesh(
            self.attention_manager,
            self.tensor_arch,
            MeshTopology.SMALL_WORLD
        )
    
    @pytest.mark.asyncio
    async def test_cognitive_workflow_scenario(self):
        """Test a complete cognitive workflow with ECAN attention allocation."""
        
        # Simulate cognitive tasks with different priorities
        tasks = [
            ("perception_task", "perception", 8, 100, 50),
            ("reasoning_task", "reasoning", 9, 200, 100),
            ("memory_task", "memory", 6, 150, 75),
            ("action_task", "action", 7, 120, 60)
        ]
        
        # Schedule all tasks
        for task_id, task_type, priority, cpu, memory in tasks:
            async def task_function():
                await asyncio.sleep(0.01)  # Simulate work
                return f"{task_id}_completed"
            
            resource_req = ResourceAllocation(
                cpu_cycles=cpu,
                memory_allocation=memory,
                priority_level=priority,
                task_type=task_type
            )
            
            success = await self.resource_kernel.schedule_cognitive_task(
                task_id, task_function, resource_req
            )
            assert success == True
        
        # Wait for task completion
        await asyncio.sleep(0.1)
        
        # Verify task execution
        stats = self.resource_kernel.get_kernel_statistics()
        assert stats['total_tasks_processed'] >= len(tasks)
    
    def test_attention_allocation_efficiency(self):
        """Test attention allocation efficiency under various loads."""
        
        # Create a network of cognitive primitives
        primitives = []
        for i in range(20):
            primitive_id = f"primitive_{i}"
            
            # Vary attention values
            sti = np.random.uniform(-500, 500)
            lti = np.random.uniform(0, 1000)
            av = AttentionValue(sti=sti, lti=lti, confidence=0.8)
            
            # Add to attention manager
            self.attention_manager.set_attention_value(primitive_id, av)
            
            # Add to mesh
            self.mesh.add_node(primitive_id, av)
            
            primitives.append(primitive_id)
        
        # Create some attention flows
        for i in range(10):
            source = np.random.choice(primitives)
            target = np.random.choice(primitives)
            if source != target:
                flow_strength = np.random.uniform(5, 25)
                self.mesh.create_attention_flow(source, target, flow_strength)
        
        # Propagate attention multiple times
        for _ in range(5):
            self.mesh.propagate_attention(iterations=2)
        
        # Check efficiency metrics
        stats = self.mesh.get_mesh_statistics()
        assert stats['total_nodes'] == 20
        assert stats['active_attention_flows'] >= 0
        
        # Attention should be properly distributed
        high_sti_atoms = self.attention_manager.get_high_sti_atoms(count=5)
        assert len(high_sti_atoms) > 0
    
    def test_dynamic_topology_adaptation(self):
        """Test dynamic topology adaptation based on attention patterns."""
        
        # Create nodes with clustered attention patterns
        cluster_centers = [100, 300, 500]  # Three attention clusters
        
        for i, center in enumerate(cluster_centers):
            for j in range(5):  # 5 nodes per cluster
                node_id = f"cluster_{i}_node_{j}"
                # Add some noise around cluster center
                sti = center + np.random.uniform(-20, 20)
                av = AttentionValue(sti=sti, lti=50.0, confidence=0.8)
                
                self.mesh.add_node(node_id, av)
        
        # Initial topology
        initial_topology = self.mesh.topology_type
        
        # Detect clusters
        clusters = self.mesh.detect_attention_clusters(min_cluster_size=3)
        
        # Should detect approximately 3 clusters
        assert len(clusters) >= 2  # At least 2 clusters should be detected
        
        # Test topology reconfiguration
        if len(clusters) > 2:
            # Should select hub-spoke for many clusters
            optimal_topology = self.mesh._select_optimal_topology()
            assert optimal_topology in [MeshTopology.HUB_SPOKE, MeshTopology.HIERARCHICAL]
    
    def test_resource_contention_handling(self):
        """Test handling of resource contention scenarios."""
        
        # Fill up most of the resource budget
        high_resource_tasks = []
        for i in range(5):
            task_id = f"high_resource_task_{i}"
            resource_req = ResourceAllocation(
                cpu_cycles=180,  # Total: 900, close to 1000 budget
                memory_allocation=180,
                priority_level=7,
                task_type="reasoning"
            )
            
            success = self.attention_manager.allocate_resources(task_id, resource_req)
            if success:
                high_resource_tasks.append(task_id)
        
        # Now try to allocate a task that should fail
        overflow_task = ResourceAllocation(
            cpu_cycles=200,  # Would exceed budget
            memory_allocation=200,
            priority_level=9
        )
        
        success = self.attention_manager.allocate_resources("overflow_task", overflow_task)
        assert success == False  # Should fail due to resource constraints
        
        # Deallocate one task and try again
        if high_resource_tasks:
            self.attention_manager.deallocate_resources(high_resource_tasks[0])
            
            # Now the allocation should succeed
            success = self.attention_manager.allocate_resources("retry_task", overflow_task)
            assert success == True
    
    def test_attention_decay_and_maintenance(self):
        """Test attention decay and maintenance mechanisms."""
        
        # Create atoms with various attention levels
        atoms_data = [
            ("decaying_atom_1", 100.0),
            ("decaying_atom_2", 50.0),
            ("decaying_atom_3", 10.0),
            ("decaying_atom_4", 1.0)  # Very low attention
        ]
        
        for atom_id, sti_value in atoms_data:
            av = AttentionValue(sti=sti_value, lti=25.0, confidence=0.8)
            self.attention_manager.set_attention_value(atom_id, av)
        
        # Simulate multiple decay cycles
        for _ in range(5):
            self.attention_manager._decay_attention_values()
        
        # Check that high attention atoms still exist but with lower values
        atom1_av = self.attention_manager.get_attention_value("decaying_atom_1")
        assert atom1_av is not None
        assert atom1_av.sti < 100.0  # Should have decayed
        assert atom1_av.sti > 10.0   # But not too much
        
        # Very low attention atoms might be cleaned up
        stats = self.attention_manager.get_attention_statistics()
        assert stats['total_atoms'] <= len(atoms_data)  # Some may have been cleaned up
    
    @pytest.mark.asyncio
    async def test_concurrent_attention_operations(self):
        """Test concurrent attention operations for thread safety."""
        
        async def attention_updater(atom_prefix: str, iterations: int):
            """Concurrently update attention values."""
            for i in range(iterations):
                atom_id = f"{atom_prefix}_{i % 5}"  # Reuse atoms
                sti = np.random.uniform(-100, 100)
                lti = np.random.uniform(0, 100)
                av = AttentionValue(sti=sti, lti=lti, confidence=0.8)
                
                self.attention_manager.set_attention_value(atom_id, av)
                await asyncio.sleep(0.001)  # Small delay
        
        async def resource_allocator(task_prefix: str, iterations: int):
            """Concurrently allocate and deallocate resources."""
            for i in range(iterations):
                task_id = f"{task_prefix}_{i}"
                resource_req = ResourceAllocation(
                    cpu_cycles=np.random.randint(10, 50),
                    memory_allocation=np.random.randint(5, 25),
                    priority_level=np.random.randint(1, 10)
                )
                
                success = self.attention_manager.allocate_resources(task_id, resource_req)
                if success:
                    await asyncio.sleep(0.002)
                    self.attention_manager.deallocate_resources(task_id)
        
        # Run concurrent operations
        tasks = [
            attention_updater("concurrent_atom_a", 10),
            attention_updater("concurrent_atom_b", 10),
            resource_allocator("concurrent_task_a", 5),
            resource_allocator("concurrent_task_b", 5)
        ]
        
        await asyncio.gather(*tasks)
        
        # Verify system remains consistent
        stats = self.attention_manager.get_attention_statistics()
        assert stats['total_atoms'] >= 0
        assert stats['active_tasks'] >= 0
        assert stats['cpu_utilization'] >= 0
        assert stats['memory_utilization'] >= 0


def run_phase2_tests():
    """Run all Phase 2 ECAN tests and return results."""
    
    print("🧠 Phase 2: ECAN Attention Allocation & Resource Kernel - Test Suite")
    print("=" * 75)
    
    # Test categories
    test_categories = [
        ("ECAN Attention Manager", TestECANAttentionManager),
        ("ECAN Resource Kernel", TestECANResourceKernel),
        ("Dynamic Attention Mesh", TestDynamicAttentionMesh),
        ("Tensor Fragment Integration", TestTensorFragmentIntegration),
        ("Real-World Scenarios", TestRealWorldScenarios)
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for category_name, test_class in test_categories:
        print(f"\n🔬 Testing {category_name}")
        print("-" * 50)
        
        # Get all test methods
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        for test_method_name in test_methods:
            total_tests += 1
            
            try:
                # Create test instance
                test_instance = test_class()
                test_instance.setup_method()
                
                # Get test method
                test_method = getattr(test_instance, test_method_name)
                
                # Run test (handle async tests)
                if asyncio.iscoroutinefunction(test_method):
                    asyncio.run(test_method())
                else:
                    test_method()
                
                print(f"  ✅ {test_method_name}")
                passed_tests += 1
                
            except Exception as e:
                print(f"  ❌ {test_method_name}: {e}")
                failed_tests.append((category_name, test_method_name, str(e)))
    
    # Print summary
    print(f"\n📊 Test Summary")
    print("=" * 40)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_tests)}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if failed_tests:
        print(f"\n❌ Failed Tests:")
        for category, test_name, error in failed_tests:
            print(f"  {category} -> {test_name}: {error}")
    
    return {
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'failed_tests': failed_tests,
        'success_rate': (passed_tests/total_tests)*100 if total_tests > 0 else 0
    }


if __name__ == "__main__":
    # Run the test suite
    test_results = run_phase2_tests()
    
    # Exit with appropriate code
    if test_results['success_rate'] == 100.0:
        print("\n🎉 All tests passed!")
        exit(0)
    else:
        print(f"\n⚠️  Some tests failed. Success rate: {test_results['success_rate']:.1f}%")
        exit(1)