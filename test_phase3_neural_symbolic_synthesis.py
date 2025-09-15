"""
Comprehensive Test Suite for Phase 3: Neural-Symbolic Synthesis via Custom GGML Kernels.

This module provides end-to-end verification of neural-symbolic computation capabilities
with tensor signature [atoms, confidence, features] and integration with Phases 1-2.
"""

import asyncio
import pytest
import time
import sys
import os

# Add the project root to the path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.opencog.ggml_kernels import (
    GGMLKernelManager, KernelOperation, KernelType, NeuralSymbolicSignature,
    create_neural_symbolic_operation, SymbolicReasoningKernel, 
    NeuralEmbeddingKernel, AttentionFusionKernel, enhance_tensor_fragment_with_ggml
)
from utils.opencog.tensor_benchmarking import (
    TensorBenchmarkManager, BenchmarkConfig, BenchmarkType
)

# Import Phase 1 and Phase 2 components for integration testing
try:
    from utils.opencog.atomspace import AtomSpace
    from utils.opencog.tensor_fragments import TensorSignature, TensorFragment, TensorFragmentArchitecture
    from utils.opencog.ecan_attention import ECANAttentionManager, AttentionValue
    INTEGRATION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Integration testing limited due to import error: {e}")
    INTEGRATION_AVAILABLE = False


class TestNeuralSymbolicSignature:
    """Test neural-symbolic tensor signature [atoms, confidence, features]."""
    
    def test_signature_creation(self):
        """Test basic signature creation and validation."""
        sig = NeuralSymbolicSignature(0.8, 0.9, 0.7)
        assert sig.atoms == 0.8
        assert sig.confidence == 0.9
        assert sig.features == 0.7
    
    def test_signature_normalization(self):
        """Test signature normalization to [0,1] range."""
        sig = NeuralSymbolicSignature(1.5, -0.2, 0.8)
        normalized = sig.normalize()
        
        assert normalized.atoms == 1.0
        assert normalized.confidence == 0.0
        assert normalized.features == 0.8
    
    def test_tensor_conversion(self):
        """Test conversion to/from tensor representation."""
        original = NeuralSymbolicSignature(0.6, 0.8, 0.9)
        tensor = original.to_tensor()
        reconstructed = NeuralSymbolicSignature.from_tensor(tensor)
        
        assert tensor == [0.6, 0.8, 0.9]
        assert reconstructed.atoms == original.atoms
        assert reconstructed.confidence == original.confidence
        assert reconstructed.features == original.features
    
    @pytest.mark.skipif(not INTEGRATION_AVAILABLE, reason="Phase 1 integration not available")
    def test_phase1_signature_conversion(self):
        """Test conversion to Phase 1 tensor signature format."""
        ns_sig = NeuralSymbolicSignature(0.8, 0.9, 0.7)
        phase1_sig = ns_sig.to_phase1_signature()
        
        assert hasattr(phase1_sig, 'modality')
        assert hasattr(phase1_sig, 'depth')
        assert hasattr(phase1_sig, 'context')
        assert hasattr(phase1_sig, 'salience')
        assert hasattr(phase1_sig, 'autonomy_index')
        
        # Check mapping logic
        assert phase1_sig.modality == 0.8  # atoms → modality
        assert phase1_sig.context == 0.7   # features → context
        assert 0.0 <= phase1_sig.depth <= 1.0
        assert 0.0 <= phase1_sig.salience <= 1.0
        assert 0.0 <= phase1_sig.autonomy_index <= 1.0


class TestGGMLKernels:
    """Test individual GGML kernel implementations."""
    
    @pytest.fixture
    def kernel_manager(self):
        """Create a kernel manager for testing."""
        return GGMLKernelManager()
    
    def test_kernel_registration(self, kernel_manager):
        """Test kernel registration and retrieval."""
        assert KernelType.SYMBOLIC_REASONING in kernel_manager.kernels
        assert KernelType.NEURAL_EMBEDDING in kernel_manager.kernels
        assert KernelType.ATTENTION_FUSION in kernel_manager.kernels
        
        reasoning_kernel = kernel_manager.kernels[KernelType.SYMBOLIC_REASONING]
        assert isinstance(reasoning_kernel, SymbolicReasoningKernel)
    
    @pytest.mark.asyncio
    async def test_symbolic_reasoning_kernel(self, kernel_manager):
        """Test symbolic reasoning kernel execution."""
        operation = create_neural_symbolic_operation(
            "test_reasoning",
            KernelType.SYMBOLIC_REASONING,
            atoms_strength=0.8,
            confidence=0.9,
            features=0.7,
            parameters={
                'symbols': ['concept_A', 'concept_B', 'relation_X'],
                'rules': [
                    {'conditions': ['concept_A', 'relation_X'], 'conclusion': 'derived_concept'},
                    {'conditions': ['concept_B'], 'conclusion': 'simple_conclusion'}
                ]
            }
        )
        
        result = await kernel_manager.execute_operation(operation)
        
        assert 'derived_facts' in result
        assert 'output_signature' in result
        assert 'reasoning_steps' in result
        assert 'confidence_decay' in result
        
        output_sig = result['output_signature']
        assert isinstance(output_sig, NeuralSymbolicSignature)
        assert 0.0 <= output_sig.atoms <= 1.0
        assert 0.0 <= output_sig.confidence <= 1.0
        assert 0.0 <= output_sig.features <= 1.0
        
        # Should have derived some facts
        assert len(result['derived_facts']) > 0
        assert result['reasoning_steps'] == 2
    
    @pytest.mark.asyncio
    async def test_neural_embedding_kernel(self, kernel_manager):
        """Test neural embedding kernel execution."""
        operation = create_neural_symbolic_operation(
            "test_embedding",
            KernelType.NEURAL_EMBEDDING,
            atoms_strength=0.6,
            confidence=0.8,
            features=0.9,
            parameters={
                'text_inputs': ['neural network', 'symbolic reasoning', 'cognitive agent'],
                'embedding_dim': 128
            }
        )
        
        result = await kernel_manager.execute_operation(operation)
        
        assert 'embeddings' in result
        assert 'embedding_dim' in result
        assert 'output_signature' in result
        assert 'processed_texts' in result
        
        embeddings = result['embeddings']
        assert len(embeddings) == 3  # Three input texts
        assert len(embeddings[0]) == 128  # Embedding dimension
        assert result['embedding_dim'] == 128
        assert result['processed_texts'] == 3
        
        # Check embedding values are in reasonable range
        for embedding in embeddings:
            for value in embedding:
                assert -1.0 <= value <= 1.0
    
    @pytest.mark.asyncio
    async def test_attention_fusion_kernel(self, kernel_manager):
        """Test attention fusion kernel execution."""
        operation = create_neural_symbolic_operation(
            "test_fusion",
            KernelType.ATTENTION_FUSION,
            atoms_strength=0.7,
            confidence=0.85,
            features=0.8,
            parameters={
                'neural_inputs': [0.5, 0.7, 0.3, 0.9, 0.2],
                'symbolic_inputs': ['atom1', 'atom2', 'atom3', 'atom4', 'atom5']
            }
        )
        
        result = await kernel_manager.execute_operation(operation)
        
        assert 'fused_representations' in result
        assert 'attention_scores' in result
        assert 'fusion_quality' in result
        assert 'output_signature' in result
        
        fused_reps = result['fused_representations']
        attention_scores = result['attention_scores']
        
        assert len(fused_reps) == 5  # Five input pairs
        assert len(attention_scores) == 5
        
        # Check attention scores are in [0,1] range
        for score in attention_scores:
            assert 0.0 <= score <= 1.0
        
        # Check fusion quality
        assert 0.0 <= result['fusion_quality'] <= 1.0
    
    def test_kernel_compilation(self, kernel_manager):
        """Test kernel compilation functionality."""
        operation = create_neural_symbolic_operation(
            "test_compile",
            KernelType.SYMBOLIC_REASONING,
            atoms_strength=0.8,
            confidence=0.9,
            features=0.7,
            parameters={'symbols': ['test'], 'rules': []}
        )
        
        compiled_code = kernel_manager.compile_operation(operation)
        
        assert isinstance(compiled_code, str)
        assert len(compiled_code) > 100  # Should be substantial code
        assert 'kernel void' in compiled_code
        assert 'symbolic_reasoning' in compiled_code
        assert operation.operation_id in compiled_code
    
    def test_operation_optimization(self, kernel_manager):
        """Test operation optimization."""
        operations = [
            create_neural_symbolic_operation(f"test_opt_{i}", KernelType.SYMBOLIC_REASONING, 0.8, 0.9, 0.7)
            for i in range(5)
        ]
        
        optimized = kernel_manager.optimize_operations(operations)
        
        assert len(optimized) == len(operations)  # Should preserve count
        assert all(isinstance(op, KernelOperation) for op in optimized)


class TestTensorBenchmarking:
    """Test tensor benchmarking framework."""
    
    @pytest.fixture
    def benchmark_manager(self):
        """Create a benchmark manager for testing."""
        kernel_manager = GGMLKernelManager()
        return TensorBenchmarkManager(kernel_manager)
    
    @pytest.mark.asyncio 
    async def test_benchmark_execution(self, benchmark_manager):
        """Test basic benchmark execution."""
        # Run a subset of benchmarks with reduced iterations for testing
        suite = benchmark_manager.benchmark_suites[0]
        
        # Modify first benchmark for faster testing
        config = suite.benchmarks[0]
        config.iterations = 3
        config.warmup_iterations = 1
        
        result = await suite.run_benchmark(config, benchmark_manager.kernel_manager)
        
        assert result.config == config
        assert len(result.execution_times) <= config.iterations
        assert result.throughput >= 0.0
        assert 0.0 <= result.accuracy_score <= 1.0
    
    def test_performance_grading(self, benchmark_manager):
        """Test performance grading system."""
        # Test with mock data
        execution_times = [0.001, 0.002, 0.0015]  # Fast execution times
        throughputs = [500.0, 600.0, 550.0]       # Good throughput
        accuracies = [0.95, 0.98, 0.96]           # High accuracy
        
        grade = benchmark_manager._calculate_performance_grade(execution_times, throughputs, accuracies)
        
        assert grade in ['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'F']
        # With good metrics, should get a decent grade
        assert grade in ['A+', 'A', 'A-', 'B+', 'B']


class TestIntegrationPhases:
    """Test integration with Phase 1 and Phase 2 components."""
    
    @pytest.mark.skipif(not INTEGRATION_AVAILABLE, reason="Integration components not available")
    def test_phase1_integration(self):
        """Test integration with Phase 1 tensor fragments."""
        atomspace = AtomSpace()
        tensor_arch = TensorFragmentArchitecture(atomspace)
        kernel_manager = GGMLKernelManager(atomspace)
        
        # Create a Phase 1 tensor fragment
        phase1_sig = TensorSignature(0.8, 0.7, 0.9, 0.6, 0.5)
        tensor_fragment = TensorFragment(phase1_sig, "test_content")
        
        # Enhance with GGML capabilities
        enhanced_fragment = enhance_tensor_fragment_with_ggml(tensor_fragment, kernel_manager)
        
        assert hasattr(enhanced_fragment, 'neural_symbolic_signature')
        assert hasattr(enhanced_fragment, 'kernel_manager')
        assert hasattr(enhanced_fragment, 'compiled_operations')
        
        ns_sig = enhanced_fragment.neural_symbolic_signature
        assert isinstance(ns_sig, NeuralSymbolicSignature)
        assert 0.0 <= ns_sig.atoms <= 1.0
        assert 0.0 <= ns_sig.confidence <= 1.0
        assert 0.0 <= ns_sig.features <= 1.0
    
    @pytest.mark.skipif(not INTEGRATION_AVAILABLE, reason="Integration components not available")
    def test_phase2_integration(self):
        """Test integration with Phase 2 ECAN attention system."""
        atomspace = AtomSpace()
        attention_manager = ECANAttentionManager(atomspace)
        kernel_manager = GGMLKernelManager(atomspace)
        
        # Create attention value
        attention_value = AttentionValue(sti=200.0, lti=150.0, confidence=0.9)
        
        # Map to neural-symbolic signature
        # STI [0, 400] → atoms [0, 1]
        atoms_strength = min(1.0, attention_value.sti / 400.0)
        confidence = attention_value.confidence
        features = 0.7  # Default feature value
        
        ns_sig = NeuralSymbolicSignature(atoms_strength, confidence, features).normalize()
        
        assert 0.0 <= ns_sig.atoms <= 1.0
        assert ns_sig.confidence == 0.9
        assert ns_sig.features == 0.7
        
        # Should map back consistently
        assert abs(ns_sig.atoms - 0.5) < 0.1  # STI 200/400 = 0.5


class TestEndToEndWorkflow:
    """Test complete end-to-end neural-symbolic workflows."""
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self):
        """Test a complete neural-symbolic computation workflow."""
        # Initialize system
        kernel_manager = GGMLKernelManager()
        
        # Step 1: Neural embedding of concepts
        embedding_op = create_neural_symbolic_operation(
            "workflow_embedding",
            KernelType.NEURAL_EMBEDDING,
            atoms_strength=0.6,
            confidence=0.8,
            features=0.9,
            parameters={
                'text_inputs': ['cognitive architecture', 'neural networks'],
                'embedding_dim': 64
            }
        )
        
        embedding_result = await kernel_manager.execute_operation(embedding_op)
        
        # Step 2: Symbolic reasoning on concepts
        reasoning_op = create_neural_symbolic_operation(
            "workflow_reasoning",
            KernelType.SYMBOLIC_REASONING,
            atoms_strength=embedding_result['output_signature'].atoms,
            confidence=embedding_result['output_signature'].confidence,
            features=embedding_result['output_signature'].features,
            parameters={
                'symbols': ['cognitive_architecture', 'neural_networks', 'learning'],
                'rules': [
                    {'conditions': ['cognitive_architecture', 'neural_networks'], 'conclusion': 'hybrid_system'},
                    {'conditions': ['hybrid_system', 'learning'], 'conclusion': 'adaptive_cognition'}
                ]
            }
        )
        
        reasoning_result = await kernel_manager.execute_operation(reasoning_op)
        
        # Step 3: Attention fusion of results
        fusion_op = create_neural_symbolic_operation(
            "workflow_fusion",
            KernelType.ATTENTION_FUSION,
            atoms_strength=reasoning_result['output_signature'].atoms,
            confidence=reasoning_result['output_signature'].confidence,
            features=reasoning_result['output_signature'].features,
            parameters={
                'neural_inputs': embedding_result['embeddings'][0][:4],  # First 4 dims
                'symbolic_inputs': reasoning_result['derived_facts'][:4] if len(reasoning_result['derived_facts']) >= 4 else reasoning_result['derived_facts'] + ['padding'] * (4 - len(reasoning_result['derived_facts']))
            }
        )
        
        fusion_result = await kernel_manager.execute_operation(fusion_op)
        
        # Verify workflow consistency
        assert embedding_result['output_signature'] is not None
        assert reasoning_result['output_signature'] is not None  
        assert fusion_result['output_signature'] is not None
        
        # Check information flow through pipeline
        final_signature = fusion_result['output_signature']
        assert 0.0 <= final_signature.atoms <= 1.0
        assert 0.0 <= final_signature.confidence <= 1.0
        assert 0.0 <= final_signature.features <= 1.0
        
        # Should have processed information at each stage
        assert len(embedding_result['embeddings']) > 0
        assert len(reasoning_result['derived_facts']) > 0
        assert len(fusion_result['fused_representations']) > 0
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self):
        """Test system performance under concurrent load."""
        kernel_manager = GGMLKernelManager()
        
        # Create multiple concurrent operations
        operations = []
        for i in range(10):
            op = create_neural_symbolic_operation(
                f"load_test_{i}",
                KernelType.SYMBOLIC_REASONING,
                atoms_strength=0.7 + 0.1 * (i % 3),
                confidence=0.8 + 0.1 * (i % 2),
                features=0.6 + 0.1 * (i % 4),
                parameters={
                    'symbols': [f'symbol_{j}' for j in range(5 + i)],
                    'rules': [{'conditions': [f'symbol_{j}'], 'conclusion': f'result_{j}'} for j in range(2 + i % 3)]
                }
            )
            operations.append(op)
        
        # Execute all operations concurrently
        start_time = time.time()
        tasks = [kernel_manager.execute_operation(op) for op in operations]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        # Check results
        successful_results = [r for r in results if not isinstance(r, Exception)]
        failed_results = [r for r in results if isinstance(r, Exception)]
        
        assert len(successful_results) >= len(operations) * 0.8  # At least 80% success rate
        assert end_time - start_time < 5.0  # Should complete within 5 seconds
        
        # Check performance stats
        stats = kernel_manager.get_performance_stats()
        assert stats['total_operations'] >= len(operations)
        assert stats['successful_operations'] >= len(successful_results)


def run_verification_suite():
    """Run the complete verification suite manually."""
    print("🧪 Phase 3 Neural-Symbolic Synthesis Verification Suite")
    print("=" * 70)
    
    # Track test results
    tests_passed = 0
    tests_failed = 0
    test_results = []
    
    async def run_async_tests():
        nonlocal tests_passed, tests_failed
        
        # Test 1: Neural-Symbolic Signature
        print("\n🔬 Testing Neural-Symbolic Signature...")
        try:
            sig = NeuralSymbolicSignature(0.8, 0.9, 0.7)
            normalized = sig.normalize()
            tensor = sig.to_tensor()
            reconstructed = NeuralSymbolicSignature.from_tensor(tensor)
            
            assert sig.atoms == 0.8
            assert reconstructed.atoms == sig.atoms
            
            print("  ✅ neural_symbolic_signature: PASSED")
            tests_passed += 1
            test_results.append(("neural_symbolic_signature", True, None))
        except Exception as e:
            print(f"  ❌ neural_symbolic_signature: FAILED - {e}")
            tests_failed += 1
            test_results.append(("neural_symbolic_signature", False, str(e)))
        
        # Test 2: GGML Kernel Manager
        print("\n⚙️ Testing GGML Kernel Manager...")
        try:
            kernel_manager = GGMLKernelManager()
            
            assert KernelType.SYMBOLIC_REASONING in kernel_manager.kernels
            assert KernelType.NEURAL_EMBEDDING in kernel_manager.kernels
            assert KernelType.ATTENTION_FUSION in kernel_manager.kernels
            
            print("  ✅ ggml_kernel_manager: PASSED")
            tests_passed += 1
            test_results.append(("ggml_kernel_manager", True, None))
        except Exception as e:
            print(f"  ❌ ggml_kernel_manager: FAILED - {e}")
            tests_failed += 1
            test_results.append(("ggml_kernel_manager", False, str(e)))
        
        # Test 3: Symbolic Reasoning Kernel
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
                    'symbols': ['concept_A', 'concept_B'],
                    'rules': [{'conditions': ['concept_A'], 'conclusion': 'derived_fact'}]
                }
            )
            
            result = await kernel_manager.execute_operation(operation)
            
            assert 'derived_facts' in result
            assert 'output_signature' in result
            assert len(result['derived_facts']) > 0
            
            print("  ✅ symbolic_reasoning_kernel: PASSED")
            tests_passed += 1
            test_results.append(("symbolic_reasoning_kernel", True, None))
        except Exception as e:
            print(f"  ❌ symbolic_reasoning_kernel: FAILED - {e}")
            tests_failed += 1
            test_results.append(("symbolic_reasoning_kernel", False, str(e)))
        
        # Test 4: Neural Embedding Kernel  
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
                    'text_inputs': ['test text', 'another text'],
                    'embedding_dim': 64
                }
            )
            
            result = await kernel_manager.execute_operation(operation)
            
            assert 'embeddings' in result
            assert len(result['embeddings']) == 2
            assert len(result['embeddings'][0]) == 64
            
            print("  ✅ neural_embedding_kernel: PASSED")
            tests_passed += 1
            test_results.append(("neural_embedding_kernel", True, None))
        except Exception as e:
            print(f"  ❌ neural_embedding_kernel: FAILED - {e}")
            tests_failed += 1
            test_results.append(("neural_embedding_kernel", False, str(e)))
        
        # Test 5: Attention Fusion Kernel
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
                    'neural_inputs': [0.5, 0.7, 0.3],
                    'symbolic_inputs': ['atom1', 'atom2', 'atom3']
                }
            )
            
            result = await kernel_manager.execute_operation(operation)
            
            assert 'fused_representations' in result
            assert 'attention_scores' in result
            assert len(result['fused_representations']) == 3
            
            print("  ✅ attention_fusion_kernel: PASSED")
            tests_passed += 1
            test_results.append(("attention_fusion_kernel", True, None))
        except Exception as e:
            print(f"  ❌ attention_fusion_kernel: FAILED - {e}")
            tests_failed += 1
            test_results.append(("attention_fusion_kernel", False, str(e)))
        
        # Test 6: Kernel Compilation
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
            
            print("  ✅ kernel_compilation: PASSED")
            tests_passed += 1
            test_results.append(("kernel_compilation", True, None))
        except Exception as e:
            print(f"  ❌ kernel_compilation: FAILED - {e}")
            tests_failed += 1
            test_results.append(("kernel_compilation", False, str(e)))
        
        # Test 7: Benchmarking Framework
        print("\n📊 Testing Benchmarking Framework...")
        try:
            kernel_manager = GGMLKernelManager()
            benchmark_manager = TensorBenchmarkManager(kernel_manager)
            
            # Test with minimal config
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
            test_results.append(("benchmarking_framework", True, None))
        except Exception as e:
            print(f"  ❌ benchmarking_framework: FAILED - {e}")
            tests_failed += 1
            test_results.append(("benchmarking_framework", False, str(e)))
        
        # Test 8: End-to-End Workflow
        print("\n🔄 Testing End-to-End Workflow...")
        try:
            kernel_manager = GGMLKernelManager()
            
            # Multi-step workflow
            embed_op = create_neural_symbolic_operation(
                "workflow_embed", KernelType.NEURAL_EMBEDDING, 0.6, 0.8, 0.9,
                {'text_inputs': ['test'], 'embedding_dim': 32}
            )
            embed_result = await kernel_manager.execute_operation(embed_op)
            
            reason_op = create_neural_symbolic_operation(
                "workflow_reason", KernelType.SYMBOLIC_REASONING, 0.8, 0.9, 0.7,
                {'symbols': ['test'], 'rules': [{'conditions': ['test'], 'conclusion': 'result'}]}
            )
            reason_result = await kernel_manager.execute_operation(reason_op)
            
            assert embed_result['output_signature'] is not None
            assert reason_result['output_signature'] is not None
            
            print("  ✅ end_to_end_workflow: PASSED")
            tests_passed += 1
            test_results.append(("end_to_end_workflow", True, None))
        except Exception as e:
            print(f"  ❌ end_to_end_workflow: FAILED - {e}")
            tests_failed += 1
            test_results.append(("end_to_end_workflow", False, str(e)))
    
    # Run async tests
    asyncio.run(run_async_tests())
    
    # Print summary
    print(f"\n📊 Verification Summary:")
    print(f"Tests passed: {tests_passed}")
    print(f"Tests failed: {tests_failed}")
    print(f"Success rate: {tests_passed/(tests_passed + tests_failed)*100:.1f}%")
    
    if tests_failed == 0:
        print("🎉 All tests passed! Neural-symbolic synthesis is functioning correctly.")
    else:
        print(f"⚠️ {tests_failed} tests failed. Check implementation.")
        for test_name, success, error in test_results:
            if not success:
                print(f"   ❌ {test_name}: {error}")
    
    return tests_passed, tests_failed, test_results


if __name__ == "__main__":
    # Run verification suite directly
    run_verification_suite()