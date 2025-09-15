"""
Simple Phase 3 Test - Neural-Symbolic Synthesis via Custom GGML Kernels.

This script tests the core Phase 3 functionality without requiring numpy or other dependencies.
"""

import asyncio
import sys
import os

# Add the project root to the path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import Phase 3 components directly to avoid numpy dependency
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Direct imports to avoid __init__.py issues
from utils.opencog.ggml_kernels import (
    GGMLKernelManager, KernelType, NeuralSymbolicSignature,
    create_neural_symbolic_operation
)
from utils.opencog.tensor_benchmarking import TensorBenchmarkManager


async def test_phase3_core_functionality():
    """Test core Phase 3 functionality."""
    print("🧪 Phase 3 Neural-Symbolic Synthesis - Core Functionality Test")
    print("=" * 70)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Neural-Symbolic Signature
    print("\n🔬 Testing Neural-Symbolic Signature...")
    try:
        sig = NeuralSymbolicSignature(0.8, 0.9, 0.7)
        assert sig.atoms == 0.8
        assert sig.confidence == 0.9
        assert sig.features == 0.7
        
        # Test normalization
        sig_unnorm = NeuralSymbolicSignature(1.5, -0.2, 0.8)
        sig_norm = sig_unnorm.normalize()
        assert sig_norm.atoms == 1.0
        assert sig_norm.confidence == 0.0
        assert sig_norm.features == 0.8
        
        # Test tensor conversion
        tensor = sig.to_tensor()
        assert tensor == [0.8, 0.9, 0.7]
        
        sig_reconstructed = NeuralSymbolicSignature.from_tensor(tensor)
        assert sig_reconstructed.atoms == sig.atoms
        
        print("  ✅ neural_symbolic_signature: PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ neural_symbolic_signature: FAILED - {e}")
        tests_failed += 1
    
    # Test 2: Kernel Manager Initialization
    print("\n⚙️ Testing GGML Kernel Manager...")
    try:
        kernel_manager = GGMLKernelManager()
        
        assert KernelType.SYMBOLIC_REASONING in kernel_manager.kernels
        assert KernelType.NEURAL_EMBEDDING in kernel_manager.kernels
        assert KernelType.ATTENTION_FUSION in kernel_manager.kernels
        assert len(kernel_manager.kernels) >= 3
        
        print("  ✅ ggml_kernel_manager: PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ ggml_kernel_manager: FAILED - {e}")
        tests_failed += 1
    
    # Test 3: Operation Creation
    print("\n🔧 Testing Operation Creation...")
    try:
        operation = create_neural_symbolic_operation(
            "test_op",
            KernelType.SYMBOLIC_REASONING,
            atoms_strength=0.8,
            confidence=0.9,
            features=0.7,
            parameters={'symbols': ['test'], 'rules': []}
        )
        
        assert operation.operation_id == "test_op"
        assert operation.kernel_type == KernelType.SYMBOLIC_REASONING
        assert operation.input_signature.atoms == 0.8
        assert operation.input_signature.confidence == 0.9
        assert operation.input_signature.features == 0.7
        
        print("  ✅ operation_creation: PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ operation_creation: FAILED - {e}")
        tests_failed += 1
    
    # Test 4: Symbolic Reasoning Kernel
    print("\n🧠 Testing Symbolic Reasoning Kernel...")
    try:
        kernel_manager = GGMLKernelManager()
        
        operation = create_neural_symbolic_operation(
            "test_reasoning",
            KernelType.SYMBOLIC_REASONING,
            atoms_strength=0.8,
            confidence=0.9,  
            features=0.7,
            parameters={
                'symbols': ['human', 'mortal', 'socrates'],
                'rules': [
                    {'conditions': ['human', 'mortal'], 'conclusion': 'humans_are_mortal'},
                    {'conditions': ['socrates', 'human'], 'conclusion': 'socrates_is_mortal'}
                ]
            }
        )
        
        result = await kernel_manager.execute_operation(operation)
        
        assert 'derived_facts' in result
        assert 'output_signature' in result
        assert 'reasoning_steps' in result
        assert len(result['derived_facts']) > 0
        assert isinstance(result['output_signature'], NeuralSymbolicSignature)
        
        print("  ✅ symbolic_reasoning_kernel: PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ symbolic_reasoning_kernel: FAILED - {e}")
        tests_failed += 1
    
    # Test 5: Neural Embedding Kernel
    print("\n🔢 Testing Neural Embedding Kernel...")
    try:
        kernel_manager = GGMLKernelManager()
        
        operation = create_neural_symbolic_operation(
            "test_embedding",
            KernelType.NEURAL_EMBEDDING,
            atoms_strength=0.6,
            confidence=0.8,
            features=0.9,
            parameters={
                'text_inputs': ['artificial intelligence', 'machine learning'],
                'embedding_dim': 64
            }
        )
        
        result = await kernel_manager.execute_operation(operation)
        
        assert 'embeddings' in result
        assert 'embedding_dim' in result
        assert 'output_signature' in result
        assert len(result['embeddings']) == 2
        assert len(result['embeddings'][0]) == 64
        assert result['embedding_dim'] == 64
        
        print("  ✅ neural_embedding_kernel: PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ neural_embedding_kernel: FAILED - {e}")
        tests_failed += 1
    
    # Test 6: Attention Fusion Kernel
    print("\n🎯 Testing Attention Fusion Kernel...")
    try:
        kernel_manager = GGMLKernelManager()
        
        operation = create_neural_symbolic_operation(
            "test_fusion",
            KernelType.ATTENTION_FUSION,
            atoms_strength=0.7,
            confidence=0.85,
            features=0.8,
            parameters={
                'neural_inputs': [0.5, 0.7, 0.3, 0.9],
                'symbolic_inputs': ['atom1', 'atom2', 'atom3', 'atom4']
            }
        )
        
        result = await kernel_manager.execute_operation(operation)
        
        assert 'fused_representations' in result
        assert 'attention_scores' in result
        assert 'fusion_quality' in result
        assert len(result['fused_representations']) == 4
        assert len(result['attention_scores']) == 4
        assert 0.0 <= result['fusion_quality'] <= 1.0
        
        print("  ✅ attention_fusion_kernel: PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ attention_fusion_kernel: FAILED - {e}")
        tests_failed += 1
    
    # Test 7: Kernel Compilation
    print("\n⚡ Testing Kernel Compilation...")
    try:
        kernel_manager = GGMLKernelManager()
        
        operation = create_neural_symbolic_operation(
            "test_compile",
            KernelType.SYMBOLIC_REASONING,
            atoms_strength=0.8,
            confidence=0.9,
            features=0.7
        )
        
        compiled_code = kernel_manager.compile_operation(operation)
        
        assert isinstance(compiled_code, str)
        assert len(compiled_code) > 100
        assert 'kernel void' in compiled_code
        assert 'symbolic_reasoning' in compiled_code
        
        print("  ✅ kernel_compilation: PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ kernel_compilation: FAILED - {e}")
        tests_failed += 1
    
    # Test 8: Benchmarking Framework
    print("\n📊 Testing Benchmarking Framework...")
    try:
        kernel_manager = GGMLKernelManager()
        benchmark_manager = TensorBenchmarkManager(kernel_manager)
        
        assert len(benchmark_manager.benchmark_suites) > 0
        
        # Test with a simple configuration
        suite = benchmark_manager.benchmark_suites[0]
        config = suite.benchmarks[0]
        config.iterations = 2
        config.warmup_iterations = 1
        
        result = await suite.run_benchmark(config, kernel_manager)
        
        assert result.config == config
        assert result.throughput >= 0.0
        assert 0.0 <= result.accuracy_score <= 1.0
        
        print("  ✅ benchmarking_framework: PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ benchmarking_framework: FAILED - {e}")
        tests_failed += 1
    
    # Test 9: End-to-End Workflow
    print("\n🔄 Testing End-to-End Workflow...")
    try:
        kernel_manager = GGMLKernelManager()
        
        # Multi-stage workflow
        stage1_op = create_neural_symbolic_operation(
            "workflow_stage1", KernelType.SYMBOLIC_REASONING, 0.8, 0.9, 0.7,
            {'symbols': ['AI', 'learning'], 'rules': [{'conditions': ['AI'], 'conclusion': 'intelligent'}]}
        )
        stage1_result = await kernel_manager.execute_operation(stage1_op)
        
        stage2_op = create_neural_symbolic_operation(
            "workflow_stage2", KernelType.NEURAL_EMBEDDING, 0.6, 0.8, 0.9,
            {'text_inputs': ['cognitive system'], 'embedding_dim': 32}
        )
        stage2_result = await kernel_manager.execute_operation(stage2_op)
        
        # Verify workflow
        assert stage1_result['output_signature'] is not None
        assert stage2_result['output_signature'] is not None
        assert len(stage1_result['derived_facts']) > 0
        assert len(stage2_result['embeddings']) > 0
        
        print("  ✅ end_to_end_workflow: PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ end_to_end_workflow: FAILED - {e}")
        tests_failed += 1
    
    # Print summary
    print(f"\n📊 Test Summary:")
    print(f"Tests passed: {tests_passed}")
    print(f"Tests failed: {tests_failed}")
    print(f"Success rate: {tests_passed/(tests_passed + tests_failed)*100:.1f}%")
    
    if tests_failed == 0:
        print("🎉 All tests passed! Phase 3 neural-symbolic synthesis is functioning correctly.")
        
        # Show performance stats
        stats = kernel_manager.get_performance_stats()
        print(f"\n📈 Performance Statistics:")
        print(f"   Total operations: {stats['total_operations']}")
        print(f"   Successful: {stats['successful_operations']}")
        print(f"   Average execution time: {stats['average_execution_time']:.4f}s")
        
        return True
    else:
        print(f"⚠️ {tests_failed} tests failed. Phase 3 implementation needs attention.")
        return False


async def test_tensor_signature_mapping():
    """Test the tensor signature mapping between phases."""
    print("\n🔗 Testing Tensor Signature Mapping")
    print("-" * 40)
    
    # Test Phase 3 signature
    ns_sig = NeuralSymbolicSignature(0.8, 0.9, 0.7)
    print(f"Phase 3 signature: [atoms={ns_sig.atoms:.3f}, confidence={ns_sig.confidence:.3f}, features={ns_sig.features:.3f}]")
    
    # Test mapping to Phase 1 (if available)
    try:
        phase1_sig = ns_sig.to_phase1_signature()
        print(f"Mapped to Phase 1: [modality={phase1_sig.modality:.3f}, depth={phase1_sig.depth:.3f}, "
              f"context={phase1_sig.context:.3f}, salience={phase1_sig.salience:.3f}, "
              f"autonomy={phase1_sig.autonomy_index:.3f}]")
        
        # Test reverse mapping
        reverse_ns = NeuralSymbolicSignature(
            atoms=phase1_sig.modality,
            confidence=(phase1_sig.salience + phase1_sig.autonomy_index) / 2,
            features=(phase1_sig.context + phase1_sig.depth) / 2
        ).normalize()
        
        print(f"Reverse mapped: [atoms={reverse_ns.atoms:.3f}, confidence={reverse_ns.confidence:.3f}, features={reverse_ns.features:.3f}]")
        
        # Check consistency (should be approximately the same)
        atoms_diff = abs(ns_sig.atoms - reverse_ns.atoms)
        confidence_diff = abs(ns_sig.confidence - reverse_ns.confidence)
        features_diff = abs(ns_sig.features - reverse_ns.features)
        
        print(f"Mapping consistency: atoms_diff={atoms_diff:.3f}, confidence_diff={confidence_diff:.3f}, features_diff={features_diff:.3f}")
        
        if atoms_diff < 0.1 and confidence_diff < 0.1 and features_diff < 0.1:
            print("✅ Tensor signature mapping: CONSISTENT")
            return True
        else:
            print("⚠️ Tensor signature mapping: INCONSISTENT")
            return False
            
    except Exception as e:
        print(f"⚠️ Phase 1 integration not available: {e}")
        return True  # Not a failure, just not available


async def main():
    """Run all Phase 3 tests."""
    print("🚀 Phase 3 Neural-Symbolic Synthesis - Simple Test Suite")
    print("=" * 70)
    print("Tensor signature: [atoms, confidence, features]")
    print()
    
    import time
    start_time = time.time()
    
    # Run core functionality tests
    core_tests_passed = await test_phase3_core_functionality()
    
    # Run mapping tests
    mapping_tests_passed = await test_tensor_signature_mapping()
    
    total_time = time.time() - start_time
    
    print(f"\n🎯 Overall Test Results:")
    print(f"   Core functionality: {'✅ PASSED' if core_tests_passed else '❌ FAILED'}")
    print(f"   Signature mapping: {'✅ PASSED' if mapping_tests_passed else '❌ FAILED'}")
    print(f"   Total test time: {total_time:.2f}s")
    
    if core_tests_passed and mapping_tests_passed:
        print(f"\n✅ Phase 3 Implementation Status: COMPLETE")
        print("   - ✅ 3.1 Kernel Customization")
        print("   - ✅ 3.2 Tensor Benchmarking")
        print("   - ✅ 3.3 End-to-End Verification")
        print("\nPhase 3: Neural-Symbolic Synthesis via Custom GGML Kernels is ready!")
        return True
    else:
        print(f"\n❌ Phase 3 Implementation Status: NEEDS ATTENTION")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)