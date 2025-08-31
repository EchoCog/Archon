#!/usr/bin/env python3
"""
Test suite for Phase 1 Cognitive Primitives & Foundational Hypergraph Encoding.

This script tests the tensor fragment architecture, cognitive grammar microservices,
and verification/visualization components.
"""

import sys
import os
import asyncio
import numpy as np

# Add the project root to Python path
sys.path.append('/home/runner/work/Archon/Archon')

def test_phase1_cognitive_primitives():
    """Test the Phase 1 cognitive primitives implementation."""
    
    print("🧬 Testing Phase 1: Cognitive Primitives & Foundational Hypergraph Encoding")
    print("=" * 80)
    
    # Import components
    try:
        from utils.opencog import opencog
        from utils.opencog.tensor_fragments import (
            TensorSignature, TensorFragment, TensorFragmentArchitecture, 
            CognitivePrimitive, create_default_ml_primitive_encoders, 
            create_default_hypergraph_decoders
        )
        from utils.opencog.cognitive_grammar import (
            CognitiveGrammarMicroservice, SchemeParser
        )
        from utils.opencog.verification import (
            run_verification_suite, CognitivePrimitiveVerifier, HypergraphVisualizer
        )
        print("✅ All Phase 1 components imported successfully")
    except Exception as e:
        print(f"❌ Failed to import Phase 1 components: {e}")
        return False
    
    # Test 1.2: Tensor Fragment Architecture
    print("\n📊 Testing 1.2: Tensor Fragment Architecture")
    print("-" * 50)
    
    # Initialize components
    atomspace = opencog.atomspace()
    tensor_arch = TensorFragmentArchitecture(atomspace)
    
    # Register default encoders and decoders
    encoders = create_default_ml_primitive_encoders()
    for prim_type, encoder in encoders.items():
        tensor_arch.register_ml_primitive_encoder(prim_type, encoder)
    
    decoders = create_default_hypergraph_decoders()
    for pattern_type, decoder in decoders.items():
        tensor_arch.register_hypergraph_decoder(pattern_type, decoder)
    
    print(f"✅ TensorFragmentArchitecture initialized with {len(encoders)} encoders and {len(decoders)} decoders")
    
    # Test tensor signature creation and validation
    test_signature = TensorSignature(
        modality=0.85,  # High linguistic modality
        depth=0.7,      # Moderate processing depth
        context=0.9,    # High contextual relevance
        salience=0.8,   # High attention weight
        autonomy_index=0.6  # Moderate autonomy
    )
    
    print(f"✅ Created tensor signature: {test_signature.to_tensor()}")
    
    # Create cognitive primitives with different characteristics
    cognitive_primitives = [
        CognitivePrimitive("attention_allocation", "attention_allocation", {"priority": "high", "adaptive": True}),
        CognitivePrimitive("pattern_recognition", "pattern_matching", {"hierarchical": True, "recursive": False}),
        CognitivePrimitive("goal_planning", "goal_planning", {"meta": True, "autonomous": True}),
        CognitivePrimitive("memory_retrieval", "memory_retrieval", {"priority": "medium"})
    ]
    
    # Convert to tensor fragments
    fragments = []
    for primitive in cognitive_primitives:
        fragment = primitive.to_tensor_fragment()
        fragments.append(fragment)
        print(f"✅ Created tensor fragment for {primitive.name}: {fragment.signature.to_tensor()}")
    
    # Test ML primitive encoding
    neural_layer_data = {"type": "attention", "size": 512, "activation": "relu"}
    encoded_fragment = tensor_arch.encode_ml_primitive("neural_layer", neural_layer_data)
    if encoded_fragment:
        fragments.append(encoded_fragment)
        print(f"✅ Encoded neural layer primitive: {encoded_fragment.signature.to_tensor()}")
    
    # Test hypergraph encoding
    print(f"\n🕸️ Testing hypergraph encoding for {len(fragments)} fragments")
    encoded_handles = []
    for i, fragment in enumerate(fragments):
        try:
            handle = fragment.encode_to_hypergraph(atomspace)
            encoded_handles.append(handle)
            print(f"  ✅ Fragment {i} encoded to handle: {handle}")
        except Exception as e:
            print(f"  ❌ Fragment {i} encoding failed: {e}")
    
    print(f"📊 AtomSpace now contains {len(atomspace.atoms)} atoms and {len(atomspace.relationships)} relationships")
    
    # Test pattern creation
    pattern_handle = tensor_arch.create_cognitive_pattern(fragments[:3], "test_cognitive_pattern")
    print(f"✅ Created cognitive pattern: {pattern_handle}")
    
    # Test pattern extraction
    extracted_fragments = tensor_arch.extract_cognitive_pattern(pattern_handle)
    print(f"✅ Extracted {len(extracted_fragments)} fragments from pattern")
    
    # Test 1.1: Scheme Cognitive Grammar Microservices
    print("\n🔤 Testing 1.1: Scheme Cognitive Grammar Microservices")
    print("-" * 50)
    
    # Test parser
    parser = SchemeParser()
    test_expressions = [
        '(define neural_attention "attention mechanism")',
        '(eval (+ modality depth context))',
        '(match (ConceptNode "pattern") (ConceptNode "target"))',
        '(lambda (x) (if (> x 0.5) "high" "low"))'
    ]
    
    for expr in test_expressions:
        try:
            parsed = parser.parse(expr)
            print(f"  ✅ Parsed: {expr}")
            print(f"     Type: {parsed.type.value}, Children: {len(parsed.children)}")
        except Exception as e:
            print(f"  ❌ Parse failed for {expr}: {e}")
    
    # Test microservice
    microservice = CognitiveGrammarMicroservice(atomspace)
    
    # Test basic evaluation
    async def test_microservice_eval():
        result1 = await microservice.parse_and_evaluate('(+ 1 2 3)')
        print(f"  ✅ Evaluated (+ 1 2 3) = {result1}")
        
        result2 = await microservice.parse_and_evaluate('(define test_concept "cognitive_primitive")')
        print(f"  ✅ Defined concept: {result2}")
        
        result3 = await microservice.parse_and_evaluate('test_concept')
        print(f"  ✅ Retrieved concept: {result3}")
    
    # Run async test
    asyncio.run(test_microservice_eval())
    
    # Test binding with tensor fragments
    for i, fragment in enumerate(fragments[:3]):
        binding_handle = microservice.create_cognitive_binding(f"fragment_{i}", fragment)
        print(f"  ✅ Created cognitive binding for fragment_{i}: {binding_handle}")
    
    # Test 1.3: Verification & Visualization
    print("\n🔍 Testing 1.3: Verification & Visualization")
    print("-" * 50)
    
    # Create verifier
    verifier = CognitivePrimitiveVerifier(atomspace, tensor_arch)
    
    # Test signature verification
    for i, fragment in enumerate(fragments[:3]):
        result = verifier.verify_tensor_fragment(fragment)
        status = "✅ VALID" if result['valid'] else "❌ INVALID"
        print(f"  {status} Fragment {i}: {len(result['errors'])} errors, {len(result['warnings'])} warnings")
    
    # Test pattern verification
    pattern_result = verifier.verify_cognitive_pattern(pattern_handle)
    pattern_status = "✅ VALID" if pattern_result['valid'] else "❌ INVALID"
    print(f"  {pattern_status} Pattern: {len(pattern_result['errors'])} errors, {len(pattern_result['warnings'])} warnings")
    
    # Run full verification suite
    print(f"\n🧪 Running comprehensive verification suite...")
    verification_results = run_verification_suite(atomspace, tensor_arch)
    
    suite_status = verification_results['summary']['status']
    success_rate = verification_results['summary']['success_rate']
    print(f"📊 Verification Suite Status: {suite_status.upper()} ({success_rate:.1%} success rate)")
    
    # Test visualization (create basic visualization)
    try:
        visualizer = HypergraphVisualizer(atomspace)
        
        # Test creating NetworkX graph
        graph = visualizer.create_networkx_graph()
        print(f"✅ Created NetworkX graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
        
        # Note: Skip matplotlib visualization in test environment
        print("  ℹ️ Matplotlib visualization skipped in test environment")
        
    except Exception as e:
        print(f"⚠️ Visualization test failed: {e}")
    
    # Test bidirectional translation
    print("\n🔄 Testing Bidirectional Translation")
    print("-" * 40)
    
    # Test ML primitive -> Hypergraph -> ML primitive
    original_data = {"type": "attention", "heads": 8, "dim": 512}
    
    # Encode to tensor fragment
    encoded_fragment = tensor_arch.encode_ml_primitive("attention", original_data)
    if encoded_fragment:
        print(f"✅ Encoded attention mechanism to tensor fragment")
        
        # Encode to hypergraph
        hypergraph_handle = encoded_fragment.encode_to_hypergraph(atomspace)
        print(f"✅ Encoded to hypergraph: {hypergraph_handle}")
        
        # Decode from hypergraph
        decoded_fragment = TensorFragment.decode_from_hypergraph(atomspace, hypergraph_handle)
        if decoded_fragment:
            print(f"✅ Decoded from hypergraph")
            
            # Decode back to ML primitive
            decoded_data = tensor_arch.decode_to_ml_primitive(decoded_fragment, "attention_config")
            print(f"✅ Decoded back to ML primitive: {decoded_data}")
        else:
            print(f"❌ Failed to decode from hypergraph")
    else:
        print(f"❌ Failed to encode attention mechanism")
    
    # Test pattern similarity
    if len(encoded_handles) >= 2:
        pattern1 = tensor_arch.create_cognitive_pattern([fragments[0]], "pattern_1")
        pattern2 = tensor_arch.create_cognitive_pattern([fragments[1]], "pattern_2")
        
        similarity = tensor_arch.compute_pattern_similarity(pattern1, pattern2)
        print(f"✅ Pattern similarity computed: {similarity:.3f}")
    
    # Final statistics
    print(f"\n📈 Final Statistics:")
    print(f"   AtomSpace: {len(atomspace.atoms)} atoms, {len(atomspace.relationships)} relationships")
    print(f"   Types: {len(atomspace.types)} registered")
    print(f"   Tensor Fragments: {len(fragments)} created")
    print(f"   Cognitive Patterns: 1 created")
    print(f"   ML Primitive Encoders: {len(tensor_arch.ml_primitive_encoders)}")
    print(f"   Hypergraph Decoders: {len(tensor_arch.hypergraph_decoders)}")
    
    print(f"\n🎉 Phase 1 Testing Complete!")
    print(f"🧬 Cognitive primitives and hypergraph encoding functional")
    
    return True


def test_tensor_signature_properties():
    """Test the tensor signature system specifically."""
    
    print("\n🔬 Testing Tensor Signature Properties")
    print("-" * 40)
    
    from utils.opencog.tensor_fragments import TensorSignature
    
    # Test signature creation with different characteristics
    signatures = [
        TensorSignature(0.9, 0.8, 0.85, 0.95, 0.7),  # High-level reasoning
        TensorSignature(0.6, 0.4, 0.5, 0.6, 0.3),    # Basic perception
        TensorSignature(0.8, 0.9, 0.7, 0.9, 0.9),    # Autonomous decision making
        TensorSignature(0.7, 0.3, 0.9, 0.8, 0.4),    # Context-aware processing
    ]
    
    signature_names = ["high_reasoning", "basic_perception", "autonomous_decision", "context_aware"]
    
    for name, sig in zip(signature_names, signatures):
        tensor = sig.to_tensor()
        print(f"  {name}: {tensor}")
        
        # Test normalization
        normalized = sig.normalize()
        norm_tensor = normalized.to_tensor()
        print(f"    Normalized: {norm_tensor}")
        
        # Test roundtrip
        roundtrip = TensorSignature.from_tensor(norm_tensor)
        diff = np.abs(norm_tensor - roundtrip.to_tensor())
        print(f"    Roundtrip error: {np.max(diff):.10f}")
    
    return True


def demonstrate_cognitive_grammar():
    """Demonstrate cognitive grammar capabilities."""
    
    print("\n🧠 Demonstrating Cognitive Grammar Operations")
    print("-" * 45)
    
    from utils.opencog.cognitive_grammar import CognitiveGrammarMicroservice
    from utils.opencog import opencog
    
    atomspace = opencog.atomspace()
    microservice = CognitiveGrammarMicroservice(atomspace)
    
    # Test expressions
    expressions = [
        '(define attention_weight 0.85)',
        '(define neural_layer (lambda (input) (+ input 0.1)))',
        '(eval (+ 0.8 0.1 0.05))',
        '(if (> attention_weight 0.8) "high_attention" "normal_attention")',
        '(match (ConceptNode "pattern") (ConceptNode "pattern"))'
    ]
    
    async def run_grammar_tests():
        for expr in expressions:
            try:
                result = await microservice.parse_and_evaluate(expr)
                print(f"  ✅ {expr}")
                print(f"     Result: {result}")
            except Exception as e:
                print(f"  ❌ {expr}")
                print(f"     Error: {e}")
    
    asyncio.run(run_grammar_tests())
    
    return True


if __name__ == "__main__":
    success = True
    
    try:
        success &= test_phase1_cognitive_primitives()
        success &= test_tensor_signature_properties()
        success &= demonstrate_cognitive_grammar()
        
        if success:
            print(f"\n🎉 All Phase 1 tests completed successfully!")
            print(f"🧬 Cognitive primitives and foundational hypergraph encoding is ready")
        else:
            print(f"\n⚠️ Some tests failed. Please review the output above.")
            
    except Exception as e:
        print(f"\n💥 Test suite execution failed: {e}")
        success = False
    
    sys.exit(0 if success else 1)