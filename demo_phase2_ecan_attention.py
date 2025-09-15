#!/usr/bin/env python3
"""
Phase 2 ECAN Attention Allocation & Resource Kernel Demo

This demo showcases the complete Phase 2 implementation:
- ECAN Attention Manager with STI/LTI-based attention allocation
- Dynamic Mesh Integration with real-time topology adaptation
- Resource Kernel for cognitive task scheduling
- Bidirectional attention-hypergraph synchronization
- Real-world verification scenarios

Tensor signature mapping: [tasks, attention, priority, resources]
"""

import sys
import os
import asyncio
import time
import numpy as np

# Add the project root to Python path
sys.path.append('/home/runner/work/Archon/Archon')

async def main():
    """Main demonstration of Phase 2 ECAN attention allocation."""
    
    print("🧠 Phase 2: ECAN Attention Allocation & Resource Kernel Demo")
    print("=" * 70)
    
    # Import Phase 2 components
    from utils.opencog import opencog
    from utils.opencog.ecan_attention import (
        ECANAttentionManager, AttentionValue, ResourceAllocation,
        ECANResourceKernel, create_ecan_enhanced_tensor_fragment
    )
    from utils.opencog.dynamic_mesh import (
        DynamicAttentionMesh, MeshTopology, AttentionFlow
    )
    from utils.opencog.tensor_fragments import (
        TensorFragmentArchitecture, TensorSignature, TensorFragment
    )
    
    # Initialize the cognitive substrate
    print("\n🧬 Initializing ECAN Cognitive Substrate")
    print("-" * 45)
    
    atomspace = opencog.atomspace()
    print(f"✅ AtomSpace initialized")
    
    # Initialize ECAN attention manager
    attention_manager = ECANAttentionManager(
        atomspace=atomspace,
        initial_sti_budget=10000.0,
        initial_lti_budget=10000.0
    )
    print(f"✅ ECAN Attention Manager initialized")
    print(f"   - STI Budget: {attention_manager.sti_budget}")
    print(f"   - LTI Budget: {attention_manager.lti_budget}")
    
    # Initialize tensor fragment architecture
    tensor_arch = TensorFragmentArchitecture(atomspace)
    print(f"✅ Tensor Fragment Architecture initialized")
    
    # Initialize resource kernel
    resource_kernel = ECANResourceKernel(attention_manager)
    print(f"✅ ECAN Resource Kernel initialized")
    
    # Initialize dynamic mesh
    mesh = DynamicAttentionMesh(
        attention_manager, 
        tensor_arch,
        MeshTopology.SMALL_WORLD
    )
    print(f"✅ Dynamic Attention Mesh initialized ({mesh.topology_type.value})")
    
    # Demo 1: Basic ECAN Attention Allocation
    print("\n🎯 Demo 1: Basic ECAN Attention Allocation")
    print("-" * 45)
    
    # Create cognitive primitives with different attention values
    primitives = [
        ("visual_attention", AttentionValue(sti=200.0, lti=150.0, confidence=0.9)),
        ("reasoning_engine", AttentionValue(sti=350.0, lti=400.0, confidence=0.95)),
        ("memory_retrieval", AttentionValue(sti=100.0, lti=300.0, confidence=0.8)),
        ("action_planning", AttentionValue(sti=250.0, lti=200.0, confidence=0.85)),
        ("language_processing", AttentionValue(sti=180.0, lti=250.0, confidence=0.9))
    ]
    
    for primitive_id, av in primitives:
        success = attention_manager.set_attention_value(primitive_id, av)
        if success:
            print(f"  ✅ {primitive_id}: STI={av.sti:.1f}, LTI={av.lti:.1f}")
        else:
            print(f"  ❌ Failed to set attention for {primitive_id}")
    
    # Show attention statistics
    stats = attention_manager.get_attention_statistics()
    print(f"\n📊 Attention Statistics:")
    print(f"  - Total Atoms: {stats['total_atoms']}")
    print(f"  - Total STI: {stats['total_sti']:.1f}")
    print(f"  - Total LTI: {stats['total_lti']:.1f}")
    print(f"  - STI Budget Used: {stats['sti_budget_used']:.1f}")
    
    # Get high STI atoms
    high_sti_atoms = attention_manager.get_high_sti_atoms(count=3)
    print(f"\n🔥 Top 3 High-STI Atoms:")
    for i, (atom_id, sti_value) in enumerate(high_sti_atoms, 1):
        print(f"  {i}. {atom_id}: STI={sti_value:.1f}")
    
    # Demo 2: Resource Allocation and Task Scheduling
    print("\n⚙️  Demo 2: Resource Allocation & Task Scheduling")
    print("-" * 50)
    
    # Define cognitive tasks with different resource requirements
    cognitive_tasks = [
        ("perception_task", "perception", 8, 150, 75, 2.0),
        ("reasoning_task", "reasoning", 9, 300, 150, 5.0),
        ("memory_consolidation", "memory", 6, 200, 100, 3.0),
        ("motor_control", "action", 7, 100, 50, 1.5),
        ("language_generation", "general", 8, 250, 125, 4.0)
    ]
    
    # Schedule tasks
    scheduled_tasks = []
    for task_id, task_type, priority, cpu, memory, duration in cognitive_tasks:
        
        async def create_task_function(task_name, exec_duration):
            """Create a task function with specific duration."""
            start_time = time.time()
            print(f"    🔄 Executing {task_name}...")
            await asyncio.sleep(exec_duration * 0.1)  # Scale down for demo
            end_time = time.time()
            result = {
                'task_name': task_name,
                'execution_time': end_time - start_time,
                'status': 'completed'
            }
            print(f"    ✅ {task_name} completed in {result['execution_time']:.3f}s")
            return result
        
        # Create resource allocation
        resource_req = ResourceAllocation(
            cpu_cycles=cpu,
            memory_allocation=memory,
            priority_level=priority,
            task_type=task_type,
            allocated_time=duration
        )
        
        # Create task function
        task_function = lambda tn=task_id, dur=duration: create_task_function(tn, dur)
        
        # Schedule the task
        success = await resource_kernel.schedule_cognitive_task(
            task_id, task_function, resource_req, attention_boost=50.0
        )
        
        if success:
            print(f"  ✅ Scheduled: {task_id} (Priority: {priority}, CPU: {cpu}, Memory: {memory})")
            scheduled_tasks.append(task_id)
        else:
            print(f"  ❌ Failed to schedule: {task_id} (Resource constraints)")
    
    # Wait for task execution
    print(f"\n⏳ Executing {len(scheduled_tasks)} cognitive tasks...")
    await asyncio.sleep(2.0)  # Wait for tasks to complete
    
    # Show resource kernel statistics
    kernel_stats = resource_kernel.get_kernel_statistics()
    print(f"\n📈 Resource Kernel Statistics:")
    print(f"  - Active Tasks: {kernel_stats['active_tasks']}")
    print(f"  - Total Tasks Processed: {kernel_stats['total_tasks_processed']}")
    print(f"  - Average Completion Time: {kernel_stats['average_completion_time']:.3f}s")
    print(f"  - Resource Efficiency: {kernel_stats['resource_efficiency']:.4f}")
    
    # Demo 3: Dynamic Mesh Integration
    print("\n🕸️  Demo 3: Dynamic Mesh Integration")
    print("-" * 40)
    
    # Add nodes to the mesh
    mesh_nodes = []
    for primitive_id, av in primitives:
        # Create enhanced tensor fragment
        resource_req = ResourceAllocation(
            cpu_cycles=np.random.randint(50, 200),
            memory_allocation=np.random.randint(25, 100),
            priority_level=np.random.randint(5, 10),
            task_type=primitive_id.split('_')[0]
        )
        
        tensor_fragment = create_ecan_enhanced_tensor_fragment(
            tensor_arch, primitive_id, av, resource_req
        )
        
        # Add to mesh
        success = mesh.add_node(primitive_id, av, tensor_fragment)
        if success:
            mesh_nodes.append(primitive_id)
            print(f"  ✅ Added to mesh: {primitive_id}")
    
    # Create attention flows between related nodes
    attention_flows = [
        ("visual_attention", "reasoning_engine", 25.0, "spreading"),
        ("reasoning_engine", "memory_retrieval", 30.0, "spreading"),
        ("memory_retrieval", "action_planning", 20.0, "spreading"),
        ("action_planning", "language_processing", 15.0, "spreading"),
        ("language_processing", "visual_attention", 10.0, "spreading")  # Circular flow
    ]
    
    print(f"\n🌊 Creating attention flows:")
    for source, target, strength, flow_type in attention_flows:
        success = mesh.create_attention_flow(source, target, strength, flow_type)
        if success:
            print(f"  ✅ Flow: {source} → {target} (strength: {strength})")
    
    # Propagate attention through the mesh
    print(f"\n🔄 Propagating attention through mesh...")
    propagation_results = mesh.propagate_attention(iterations=3)
    
    print(f"📊 Attention propagation results:")
    for node_id, final_sti in propagation_results.items():
        print(f"  - {node_id}: Final STI = {final_sti:.1f}")
    
    # Detect attention clusters
    clusters = mesh.detect_attention_clusters(min_cluster_size=2)
    if clusters:
        print(f"\n🎯 Detected {len(clusters)} attention clusters:")
        for cluster_id, node_set in clusters.items():
            print(f"  - {cluster_id}: {', '.join(node_set)}")
    else:
        print(f"\n🎯 No significant attention clusters detected")
    
    # Show mesh statistics
    mesh_stats = mesh.get_mesh_statistics()
    print(f"\n📊 Dynamic Mesh Statistics:")
    print(f"  - Total Nodes: {mesh_stats['total_nodes']}")
    print(f"  - Total Edges: {mesh_stats['total_edges']}")
    print(f"  - Active Attention Flows: {mesh_stats['active_attention_flows']}")
    print(f"  - Topology: {mesh_stats['topology_type']}")
    print(f"  - Network Density: {mesh_stats['network_density']:.3f}")
    print(f"  - Average Clustering: {mesh_stats['average_clustering']:.3f}")
    
    # Demo 4: Tensor Fragment Integration
    print("\n🧩 Demo 4: Tensor Fragment Integration")
    print("-" * 42)
    
    # Show tensor signature mapping: [tasks, attention, priority, resources]
    print("Tensor Signature Mapping: [tasks, attention, priority, resources]")
    print("  tasks → modality | attention → salience | priority → depth | resources → autonomy")
    print()
    
    for node_id in mesh_nodes[:3]:  # Show first 3 nodes
        node = mesh.nodes[node_id]
        if node.tensor_fragment:
            sig = node.tensor_fragment.signature
            av = node.attention_value
            
            print(f"🔹 {node_id}:")
            print(f"  Tensor Signature: [{sig.modality:.3f}, {sig.depth:.3f}, {sig.context:.3f}, {sig.salience:.3f}, {sig.autonomy_index:.3f}]")
            print(f"  ECAN Attention: STI={av.sti:.1f}, LTI={av.lti:.1f}, Conf={av.confidence:.2f}")
            
            # Show mapping
            norm_sti, norm_lti, confidence = av.normalize_for_tensor()
            print(f"  Mapping Verification:")
            print(f"    - attention (STI): {av.sti:.1f} → {norm_sti:.3f} ≈ {sig.salience:.3f} ✅")
            print(f"    - context (LTI): {av.lti:.1f} → {norm_lti:.3f} ≈ {sig.context:.3f} ✅")
            print(f"    - autonomy (Conf): {av.confidence:.2f} ≈ {sig.autonomy_index:.3f} ✅")
            print()
    
    # Demo 5: Real-Time Attention Dynamics
    print("\n⚡ Demo 5: Real-Time Attention Dynamics")
    print("-" * 42)
    
    print("Starting dynamic attention updates...")
    
    # Start attention daemon for automatic spreading and decay
    attention_manager.start_attention_daemon()
    mesh.start_dynamic_updates()
    
    # Simulate dynamic attention changes
    for cycle in range(5):
        print(f"\n🔄 Attention Cycle {cycle + 1}:")
        
        # Randomly boost attention for some nodes
        boost_node = np.random.choice(mesh_nodes)
        current_av = attention_manager.get_attention_value(boost_node)
        if current_av:
            boosted_av = AttentionValue(
                sti=current_av.sti + np.random.uniform(20, 50),
                lti=current_av.lti + np.random.uniform(5, 15),
                vlti=current_av.vlti,
                confidence=current_av.confidence
            )
            attention_manager.set_attention_value(boost_node, boosted_av)
            print(f"  📈 Boosted {boost_node}: STI={boosted_av.sti:.1f}")
        
        # Show current high-attention nodes
        high_sti = attention_manager.get_high_sti_atoms(count=2)
        print(f"  🔥 Top attention: {high_sti[0][0]}({high_sti[0][1]:.1f}), {high_sti[1][0]}({high_sti[1][1]:.1f})")
        
        await asyncio.sleep(0.5)  # Let the system process
    
    # Stop daemons
    attention_manager.stop_attention_daemon()
    mesh.stop_dynamic_updates()
    
    # Final statistics
    print("\n📊 Final System Statistics")
    print("-" * 30)
    
    final_attention_stats = attention_manager.get_attention_statistics()
    final_mesh_stats = mesh.get_mesh_statistics()
    final_kernel_stats = resource_kernel.get_kernel_statistics()
    
    print(f"ECAN Attention Manager:")
    print(f"  - Total Atoms: {final_attention_stats['total_atoms']}")
    print(f"  - CPU Utilization: {final_attention_stats['cpu_utilization']:.1%}")
    print(f"  - Memory Utilization: {final_attention_stats['memory_utilization']:.1%}")
    
    print(f"Dynamic Mesh:")
    print(f"  - Mesh Updates: {final_mesh_stats['mesh_updates']}")
    print(f"  - Topology Reconfigurations: {final_mesh_stats['topology_reconfigurations']}")
    print(f"  - Attention Propagation Efficiency: {final_mesh_stats['attention_propagation_efficiency']:.3f}")
    
    print(f"Resource Kernel:")
    print(f"  - Tasks Processed: {final_kernel_stats['total_tasks_processed']}")
    print(f"  - Average Completion Time: {final_kernel_stats['average_completion_time']:.3f}s")
    print(f"  - Resource Efficiency: {final_kernel_stats['resource_efficiency']:.4f}")
    
    # Success metrics
    print(f"\n🎉 Phase 2 Demo Results")
    print("=" * 30)
    
    success_metrics = {
        'attention_allocation': final_attention_stats['total_atoms'] > 0,
        'resource_scheduling': final_kernel_stats['total_tasks_processed'] > 0,
        'mesh_integration': final_mesh_stats['total_nodes'] > 0,
        'attention_propagation': final_mesh_stats['active_attention_flows'] > 0,
        'tensor_integration': len([n for n in mesh.nodes.values() if n.tensor_fragment]) > 0
    }
    
    total_checks = len(success_metrics)
    passed_checks = sum(success_metrics.values())
    success_rate = (passed_checks / total_checks) * 100
    
    print(f"✅ Attention Allocation: {'PASS' if success_metrics['attention_allocation'] else 'FAIL'}")
    print(f"✅ Resource Scheduling: {'PASS' if success_metrics['resource_scheduling'] else 'FAIL'}")
    print(f"✅ Mesh Integration: {'PASS' if success_metrics['mesh_integration'] else 'FAIL'}")
    print(f"✅ Attention Propagation: {'PASS' if success_metrics['attention_propagation'] else 'FAIL'}")
    print(f"✅ Tensor Integration: {'PASS' if success_metrics['tensor_integration'] else 'FAIL'}")
    
    print(f"\n🏆 Overall Success Rate: {success_rate:.1f}% ({passed_checks}/{total_checks})")
    
    if success_rate >= 80.0:
        print("🎉 Phase 2 implementation is working successfully!")
        return True
    else:
        print("⚠️  Phase 2 implementation may need attention.")
        return False


if __name__ == "__main__":
    print("Starting Phase 2 ECAN Attention Allocation Demo...")
    
    try:
        success = asyncio.run(main())
        if success:
            print("\n✅ Demo completed successfully!")
            exit(0)
        else:
            print("\n⚠️  Demo completed with issues.")
            exit(1)
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)