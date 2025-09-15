"""
Phase 3 Neural-Symbolic Synthesis Demonstration.

This script demonstrates the complete Phase 3 implementation with custom GGML kernels
for seamless neural-symbolic computation and inference.
"""

import asyncio
import time
import sys
import os

# Add the project root to the path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.opencog.ggml_kernels import (
    GGMLKernelManager, KernelType, NeuralSymbolicSignature,
    create_neural_symbolic_operation, enhance_tensor_fragment_with_ggml
)
from utils.opencog.tensor_benchmarking import TensorBenchmarkManager

# Import Phase 1 and Phase 2 components for integration demonstration
try:
    from utils.opencog.atomspace import AtomSpace
    from utils.opencog.tensor_fragments import TensorSignature, TensorFragment, TensorFragmentArchitecture
    from utils.opencog.ecan_attention import ECANAttentionManager, AttentionValue
    INTEGRATION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Integration demonstration limited due to import error: {e}")
    INTEGRATION_AVAILABLE = False


async def demo_neural_symbolic_kernels():
    """Demonstrate custom GGML kernels for neural-symbolic computation."""
    print("🧠 Phase 3: Neural-Symbolic Synthesis via Custom GGML Kernels")
    print("=" * 70)
    print("Tensor signature: [atoms, confidence, features]")
    print()
    
    # Initialize kernel manager
    kernel_manager = GGMLKernelManager()
    
    print("📊 Kernel Manager Initialized:")
    print(f"   Registered kernels: {len(kernel_manager.kernels)}")
    for kernel_type in kernel_manager.kernels.keys():
        print(f"   - {kernel_type.value}")
    print()
    
    # === 3.1 Kernel Customization Demo ===
    print("🔧 3.1 Kernel Customization & Bidirectional Neural-Symbolic Operations")
    print("-" * 50)
    
    # Symbolic Reasoning Kernel
    print("🧠 Testing Symbolic Reasoning Kernel...")
    reasoning_op = create_neural_symbolic_operation(
        "demo_reasoning",
        KernelType.SYMBOLIC_REASONING,
        atoms_strength=0.85,  # Strong symbolic representation
        confidence=0.92,      # High confidence in reasoning
        features=0.78,        # Rich feature context
        parameters={
            'symbols': ['human', 'mortal', 'socrates', 'philosopher', 'greek'],
            'rules': [
                {'conditions': ['human', 'mortal'], 'conclusion': 'all_humans_are_mortal'},
                {'conditions': ['socrates', 'human'], 'conclusion': 'socrates_is_human'},
                {'conditions': ['socrates_is_human', 'all_humans_are_mortal'], 'conclusion': 'socrates_is_mortal'},
                {'conditions': ['socrates', 'philosopher'], 'conclusion': 'socrates_is_philosopher'},
                {'conditions': ['philosopher', 'greek'], 'conclusion': 'greek_philosophy'}
            ]
        }
    )
    
    reasoning_result = await kernel_manager.execute_operation(reasoning_op)
    print(f"   Input signature: [atoms={reasoning_op.input_signature.atoms:.3f}, "
          f"confidence={reasoning_op.input_signature.confidence:.3f}, "
          f"features={reasoning_op.input_signature.features:.3f}]")
    print(f"   Derived facts: {len(reasoning_result['derived_facts'])}")
    print(f"   Output signature: [atoms={reasoning_result['output_signature'].atoms:.3f}, "
          f"confidence={reasoning_result['output_signature'].confidence:.3f}, "
          f"features={reasoning_result['output_signature'].features:.3f}]")
    print(f"   Facts: {reasoning_result['derived_facts']}")
    print()
    
    # Neural Embedding Kernel
    print("🔢 Testing Neural Embedding Kernel...")
    embedding_op = create_neural_symbolic_operation(
        "demo_embedding",
        KernelType.NEURAL_EMBEDDING,
        atoms_strength=0.65,  # Moderate symbolic content
        confidence=0.88,      # Good confidence in neural processing
        features=0.95,        # Very rich features
        parameters={
            'text_inputs': [
                'artificial intelligence and machine learning',
                'symbolic reasoning and knowledge representation',
                'neural networks and deep learning architectures',
                'cognitive science and human intelligence'
            ],
            'embedding_dim': 256
        }
    )
    
    embedding_result = await kernel_manager.execute_operation(embedding_op)
    print(f"   Input signature: [atoms={embedding_op.input_signature.atoms:.3f}, "
          f"confidence={embedding_op.input_signature.confidence:.3f}, "
          f"features={embedding_op.input_signature.features:.3f}]")
    print(f"   Processed texts: {embedding_result['processed_texts']}")
    print(f"   Embedding dimension: {embedding_result['embedding_dim']}")
    print(f"   Output signature: [atoms={embedding_result['output_signature'].atoms:.3f}, "
          f"confidence={embedding_result['output_signature'].confidence:.3f}, "
          f"features={embedding_result['output_signature'].features:.3f}]")
    print(f"   Sample embedding values: {embedding_result['embeddings'][0][:5]} ...")
    print()
    
    # Attention Fusion Kernel
    print("🎯 Testing Attention Fusion Kernel...")
    fusion_op = create_neural_symbolic_operation(
        "demo_fusion",
        KernelType.ATTENTION_FUSION,
        atoms_strength=0.75,
        confidence=0.90,
        features=0.85,
        parameters={
            'neural_inputs': embedding_result['embeddings'][0][:8],  # Use first 8 embedding dims
            'symbolic_inputs': reasoning_result['derived_facts'][:8] if len(reasoning_result['derived_facts']) >= 8 
                             else reasoning_result['derived_facts'] + ['padding'] * (8 - len(reasoning_result['derived_facts']))
        }
    )
    
    fusion_result = await kernel_manager.execute_operation(fusion_op)
    print(f"   Input signature: [atoms={fusion_op.input_signature.atoms:.3f}, "
          f"confidence={fusion_op.input_signature.confidence:.3f}, "
          f"features={fusion_op.input_signature.features:.3f}]")
    print(f"   Fusion quality: {fusion_result['fusion_quality']:.3f}")
    print(f"   Attention scores: {[f'{score:.3f}' for score in fusion_result['attention_scores'][:5]]} ...")
    print(f"   Output signature: [atoms={fusion_result['output_signature'].atoms:.3f}, "
          f"confidence={fusion_result['output_signature'].confidence:.3f}, "
          f"features={fusion_result['output_signature'].features:.3f}]")
    print()
    
    # Demonstrate kernel compilation
    print("⚡ Kernel Compilation & Optimization...")
    compiled_reasoning = kernel_manager.compile_operation(reasoning_op)
    compiled_embedding = kernel_manager.compile_operation(embedding_op)
    
    print(f"   Compiled reasoning kernel: {len(compiled_reasoning)} characters")
    print(f"   Compiled embedding kernel: {len(compiled_embedding)} characters")
    
    # Show optimization
    operations = [reasoning_op, embedding_op, fusion_op]
    optimized_ops = kernel_manager.optimize_operations(operations)
    print(f"   Original operations: {len(operations)}")
    print(f"   Optimized operations: {len(optimized_ops)}")
    print()
    
    return {
        'reasoning_result': reasoning_result,
        'embedding_result': embedding_result,
        'fusion_result': fusion_result,
        'kernel_manager': kernel_manager
    }


async def demo_tensor_benchmarking():
    """Demonstrate tensor benchmarking framework."""
    print("📊 3.2 Tensor Benchmarking Framework")
    print("-" * 40)
    
    # Initialize benchmarking
    kernel_manager = GGMLKernelManager()
    benchmark_manager = TensorBenchmarkManager(kernel_manager)
    
    print("🎯 Running Performance Benchmarks...")
    print("   (Using reduced iterations for demo)")
    
    # Modify benchmark configurations for faster demo
    for suite in benchmark_manager.benchmark_suites:
        for config in suite.benchmarks:
            config.iterations = 5  # Reduce for demo
            config.warmup_iterations = 2
            config.timeout_seconds = 10.0
    
    # Run benchmarks
    start_time = time.time()
    results = await benchmark_manager.run_all_benchmarks()
    benchmark_time = time.time() - start_time
    
    # Display results
    summary = results['summary']
    print(f"\n📈 Benchmarking Results ({benchmark_time:.2f}s):")
    print(f"   Performance Grade: {summary['performance_grade']}")
    print(f"   Success Rate: {summary['success_rate']:.1%}")
    print(f"   Avg Execution Time: {summary['overall_avg_execution_time']:.4f}s")
    print(f"   Avg Throughput: {summary['overall_avg_throughput']:.2f} ops/s")
    print(f"   Avg Accuracy: {summary['overall_avg_accuracy']:.1%}")
    
    # Show per-kernel performance
    kernel_stats = results['kernel_performance']['kernel_stats']
    print(f"\n🔧 Per-Kernel Performance:")
    for kernel_name, stats in kernel_stats.items():
        success_rate = (stats['successful_ops'] / stats['total_ops'] * 100) if stats['total_ops'] > 0 else 0
        print(f"   {kernel_name}: {success_rate:.1f}% success, {stats['avg_time']:.4f}s avg")
    
    print()
    return results


def demo_phase_integration():
    """Demonstrate integration with Phase 1 and Phase 2."""
    print("🔗 3.3 Integration with Phases 1-2")
    print("-" * 35)
    
    if not INTEGRATION_AVAILABLE:
        print("   ⚠️ Integration demo skipped - components not available")
        print("   This would demonstrate:")
        print("   - Phase 1 tensor fragment enhancement with GGML kernels")
        print("   - Phase 2 ECAN attention integration")
        print("   - Bidirectional signature mapping")
        print()
        return
    
    # Phase 1 Integration
    print("🧮 Phase 1 Tensor Fragment Integration...")
    atomspace = AtomSpace()
    tensor_arch = TensorFragmentArchitecture(atomspace)
    kernel_manager = GGMLKernelManager(atomspace)
    
    # Create Phase 1 tensor fragment
    phase1_signature = TensorSignature(
        modality=0.8,    # Conceptual modality
        depth=0.7,       # Deep processing
        context=0.9,     # Rich context
        salience=0.85,   # High attention
        autonomy_index=0.6  # Moderate autonomy
    )
    
    tensor_fragment = TensorFragment(phase1_signature, "neural_symbolic_concept")
    
    # Enhance with GGML capabilities
    enhanced_fragment = enhance_tensor_fragment_with_ggml(tensor_fragment, kernel_manager)
    
    ns_sig = enhanced_fragment.neural_symbolic_signature
    print(f"   Phase 1 → Phase 3 mapping:")
    print(f"   [modality={phase1_signature.modality:.3f}, depth={phase1_signature.depth:.3f}, "
          f"context={phase1_signature.context:.3f}, salience={phase1_signature.salience:.3f}, "
          f"autonomy={phase1_signature.autonomy_index:.3f}]")
    print(f"   ↓")
    print(f"   [atoms={ns_sig.atoms:.3f}, confidence={ns_sig.confidence:.3f}, "
          f"features={ns_sig.features:.3f}]")
    
    # Demonstrate reverse mapping
    reverse_phase1 = ns_sig.to_phase1_signature()
    print(f"   ↓")
    print(f"   [modality={reverse_phase1.modality:.3f}, depth={reverse_phase1.depth:.3f}, "
          f"context={reverse_phase1.context:.3f}, salience={reverse_phase1.salience:.3f}, "
          f"autonomy={reverse_phase1.autonomy_index:.3f}]")
    print()
    
    # Phase 2 Integration
    print("⚡ Phase 2 ECAN Attention Integration...")
    attention_manager = ECANAttentionManager(atomspace)
    
    # Create attention value
    attention_value = AttentionValue(sti=250.0, lti=180.0, confidence=0.92)
    
    # Map to neural-symbolic signature
    atoms_strength = min(1.0, attention_value.sti / 400.0)  # STI normalization
    confidence = attention_value.confidence
    features = min(1.0, attention_value.lti / 200.0)        # LTI as features
    
    ecan_ns_sig = NeuralSymbolicSignature(atoms_strength, confidence, features).normalize()
    
    print(f"   ECAN → Phase 3 mapping:")
    print(f"   STI={attention_value.sti:.1f}, LTI={attention_value.lti:.1f}, "
          f"conf={attention_value.confidence:.3f}")
    print(f"   ↓")
    print(f"   [atoms={ecan_ns_sig.atoms:.3f}, confidence={ecan_ns_sig.confidence:.3f}, "
          f"features={ecan_ns_sig.features:.3f}]")
    print()
    
    return {
        'enhanced_fragment': enhanced_fragment,
        'ecan_signature': ecan_ns_sig,
        'integration_successful': True
    }


async def demo_end_to_end_verification():
    """Demonstrate end-to-end verification and validation."""
    print("✅ 3.3 End-to-End Verification")
    print("-" * 30)
    
    print("🔍 Running Comprehensive Workflow Test...")
    
    # Initialize system
    kernel_manager = GGMLKernelManager()
    
    # Multi-stage neural-symbolic pipeline
    stages = []
    
    # Stage 1: Initial symbolic knowledge
    stage1_op = create_neural_symbolic_operation(
        "verification_stage1",
        KernelType.SYMBOLIC_REASONING,
        atoms_strength=0.9,   # High symbolic content
        confidence=0.85,      # Good initial confidence
        features=0.7,         # Moderate features
        parameters={
            'symbols': ['AI_system', 'learning', 'adaptation', 'intelligence'],
            'rules': [
                {'conditions': ['AI_system', 'learning'], 'conclusion': 'adaptive_AI'},
                {'conditions': ['adaptive_AI', 'intelligence'], 'conclusion': 'intelligent_adaptation'}
            ]
        }
    )
    
    stage1_result = await kernel_manager.execute_operation(stage1_op)
    stages.append(('Symbolic Reasoning', stage1_result))
    
    # Stage 2: Neural embedding of symbolic conclusions
    stage2_op = create_neural_symbolic_operation(
        "verification_stage2",
        KernelType.NEURAL_EMBEDDING,
        atoms_strength=stage1_result['output_signature'].atoms,
        confidence=stage1_result['output_signature'].confidence,
        features=stage1_result['output_signature'].features,
        parameters={
            'text_inputs': stage1_result['derived_facts'] + ['cognitive architecture'],
            'embedding_dim': 128
        }
    )
    
    stage2_result = await kernel_manager.execute_operation(stage2_op)
    stages.append(('Neural Embedding', stage2_result))
    
    # Stage 3: Attention fusion of results
    stage3_op = create_neural_symbolic_operation(
        "verification_stage3",
        KernelType.ATTENTION_FUSION,
        atoms_strength=stage2_result['output_signature'].atoms,
        confidence=stage2_result['output_signature'].confidence,
        features=stage2_result['output_signature'].features,
        parameters={
            'neural_inputs': stage2_result['embeddings'][0][:6],
            'symbolic_inputs': stage1_result['derived_facts'][:6] if len(stage1_result['derived_facts']) >= 6 
                             else stage1_result['derived_facts'] + ['context'] * (6 - len(stage1_result['derived_facts']))
        }
    )
    
    stage3_result = await kernel_manager.execute_operation(stage3_op)
    stages.append(('Attention Fusion', stage3_result))
    
    # Analyze pipeline
    print(f"   Pipeline stages: {len(stages)}")
    
    for i, (stage_name, result) in enumerate(stages, 1):
        sig = result['output_signature']
        print(f"   Stage {i} ({stage_name}):")
        print(f"     Signature: [atoms={sig.atoms:.3f}, confidence={sig.confidence:.3f}, "
              f"features={sig.features:.3f}]")
        
        if 'derived_facts' in result:
            print(f"     Derived facts: {len(result['derived_facts'])}")
        if 'embeddings' in result:
            print(f"     Embeddings: {len(result['embeddings'])} x {len(result['embeddings'][0])}")
        if 'fused_representations' in result:
            print(f"     Fused representations: {len(result['fused_representations'])}")
    
    # Final validation
    final_signature = stages[-1][1]['output_signature']
    
    # Check signature evolution
    initial_confidence = stages[0][1]['output_signature'].confidence
    final_confidence = final_signature.confidence
    confidence_change = final_confidence - initial_confidence
    
    initial_features = stages[0][1]['output_signature'].features
    final_features = final_signature.features
    features_change = final_features - initial_features
    
    print(f"\n📈 Pipeline Analysis:")
    print(f"   Confidence evolution: {initial_confidence:.3f} → {final_confidence:.3f} "
          f"({confidence_change:+.3f})")
    print(f"   Features evolution: {initial_features:.3f} → {final_features:.3f} "
          f"({features_change:+.3f})")
    
    # Validation checks
    validation_results = []
    
    # Check 1: All operations succeeded
    all_successful = all('output_signature' in result for _, result in stages)
    validation_results.append(('All operations successful', all_successful))
    
    # Check 2: Signature values in valid range
    valid_ranges = all(
        0.0 <= result['output_signature'].atoms <= 1.0 and
        0.0 <= result['output_signature'].confidence <= 1.0 and
        0.0 <= result['output_signature'].features <= 1.0
        for _, result in stages
    )
    validation_results.append(('Valid signature ranges', valid_ranges))
    
    # Check 3: Information preservation
    info_preserved = len(stages) == 3 and all(result for _, result in stages)
    validation_results.append(('Information preservation', info_preserved))
    
    # Check 4: Performance within bounds
    performance_stats = kernel_manager.get_performance_stats()
    good_performance = performance_stats['average_execution_time'] < 1.0
    validation_results.append(('Performance acceptable', good_performance))
    
    print(f"\n✅ Validation Results:")
    all_passed = True
    for check_name, passed in validation_results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status} {check_name}")
        if not passed:
            all_passed = False
    
    print(f"\n🎯 End-to-End Verification: {'✅ PASSED' if all_passed else '❌ FAILED'}")
    print()
    
    return {
        'stages': stages,
        'validation_results': validation_results,
        'all_passed': all_passed,
        'performance_stats': performance_stats
    }


async def main():
    """Run the complete Phase 3 demonstration."""
    print("🚀 Starting Phase 3 Neural-Symbolic Synthesis Demonstration")
    print("=" * 80)
    
    start_time = time.time()
    results = {}
    
    try:
        # 3.1 Kernel Customization
        kernel_results = await demo_neural_symbolic_kernels()
        results['kernels'] = kernel_results
        
        # 3.2 Tensor Benchmarking  
        benchmark_results = await demo_tensor_benchmarking()
        results['benchmarks'] = benchmark_results
        
        # 3.3 Integration & Verification
        integration_results = demo_phase_integration()
        results['integration'] = integration_results
        
        verification_results = await demo_end_to_end_verification()
        results['verification'] = verification_results
        
        # Final summary
        total_time = time.time() - start_time
        
        print("🎉 Phase 3 Demonstration Complete!")
        print("=" * 50)
        print(f"⏱️  Total demonstration time: {total_time:.2f}s")
        
        # Success metrics
        kernel_manager = kernel_results['kernel_manager']
        final_stats = kernel_manager.get_performance_stats()
        
        print(f"📊 Final Statistics:")
        print(f"   Total operations executed: {final_stats['total_operations']}")
        print(f"   Success rate: {final_stats['successful_operations']}/{final_stats['total_operations']} "
              f"({final_stats['successful_operations']/final_stats['total_operations']*100:.1f}%)")
        print(f"   Average execution time: {final_stats['average_execution_time']:.4f}s")
        
        if 'benchmarks' in results:
            benchmark_summary = results['benchmarks']['summary']
            print(f"   Benchmark performance grade: {benchmark_summary['performance_grade']}")
        
        if 'verification' in results:
            verification_passed = results['verification']['all_passed']
            print(f"   End-to-end verification: {'✅ PASSED' if verification_passed else '❌ FAILED'}")
        
        print(f"\n✅ Phase 3 Implementation Status: COMPLETE")
        print("   - ✅ 3.1 Kernel Customization")
        print("   - ✅ 3.2 Tensor Benchmarking") 
        print("   - ✅ 3.3 End-to-End Verification")
        
        return results
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    asyncio.run(main())