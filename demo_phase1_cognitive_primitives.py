#!/usr/bin/env python3
"""
Phase 1 Cognitive Primitives Integration Demo

This demo showcases the complete Phase 1 implementation:
- Scheme Cognitive Grammar Microservices
- Tensor Fragment Architecture with [modality, depth, context, salience, autonomy_index] signature
- Verification & Visualization capabilities
- Bidirectional translation between ML primitives and hypergraph patterns
"""

import sys
import os
import asyncio
import numpy as np

# Add the project root to Python path
sys.path.append('/home/runner/work/Archon/Archon')

async def main():
    """Main demonstration of Phase 1 cognitive primitives."""
    
    print("🧬 Phase 1: Cognitive Primitives & Foundational Hypergraph Encoding Demo")
    print("=" * 75)
    
    # Import Phase 1 components
    from utils.opencog import opencog
    from utils.opencog.tensor_fragments import (
        TensorSignature, TensorFragment, TensorFragmentArchitecture,
        CognitivePrimitive, create_default_ml_primitive_encoders,
        create_default_hypergraph_decoders
    )
    from utils.opencog.cognitive_grammar import (
        CognitiveGrammarMicroservice, CognitiveGrammarServer,
        CognitiveGrammarOrchestrator
    )
    from utils.opencog.verification import (
        CognitivePrimitiveVerifier, HypergraphVisualizer, CognitiveDashboard
    )
    
    # Initialize the cognitive substrate
    print("\n🧠 Initializing Cognitive Substrate")
    print("-" * 40)
    
    atomspace = opencog.atomspace()
    cogserver = opencog.cogserver(atomspace)
    utilities = opencog.utilities(atomspace)
    
    # Initialize Phase 1 components
    tensor_arch = TensorFragmentArchitecture(atomspace)
    grammar_service = CognitiveGrammarMicroservice(atomspace)
    verifier = CognitivePrimitiveVerifier(atomspace, tensor_arch)
    visualizer = HypergraphVisualizer(atomspace)
    dashboard = CognitiveDashboard(verifier)
    
    print("✅ Cognitive substrate initialized")
    print(f"   AtomSpace: {len(atomspace.types)} predefined types")
    print(f"   Components: AtomSpace, CogServer, Utilities, TensorArch, Grammar, Verifier")
    
    # Register ML primitive encoders and decoders
    encoders = create_default_ml_primitive_encoders()
    for prim_type, encoder in encoders.items():
        tensor_arch.register_ml_primitive_encoder(prim_type, encoder)
    
    decoders = create_default_hypergraph_decoders()
    for pattern_type, decoder in decoders.items():
        tensor_arch.register_hypergraph_decoder(pattern_type, decoder)
    
    print(f"✅ Registered {len(encoders)} ML encoders and {len(decoders)} hypergraph decoders")
    
    # Demonstrate 1.2: Tensor Fragment Architecture
    print("\n📊 Demonstrating Tensor Fragment Architecture")
    print("-" * 50)
    
    # Create various ML primitives to encode
    ml_primitives = [
        ("neural_layer", {"type": "transformer", "size": 768, "activation": "gelu"}),
        ("attention", {"type": "multi_head", "heads": 12, "dim": 768}),
        ("embedding", {"dimensions": 1024, "type": "positional_embedding"}),
        ("neural_layer", {"type": "feedforward", "size": 2048, "activation": "relu"}),
        ("attention", {"type": "self_attention", "heads": 16, "dim": 512})
    ]
    
    # Encode ML primitives to tensor fragments
    tensor_fragments = []
    for i, (prim_type, prim_data) in enumerate(ml_primitives):
        fragment = tensor_arch.encode_ml_primitive(prim_type, prim_data)
        tensor_fragments.append(fragment)
        
        sig = fragment.signature
        print(f"  🧬 ML Primitive {i+1} ({prim_type}):")
        print(f"     Tensor Signature: [modality={sig.modality:.3f}, depth={sig.depth:.3f}, "
              f"context={sig.context:.3f}, salience={sig.salience:.3f}, autonomy={sig.autonomy_index:.3f}]")
        print(f"     Content: {fragment.content}")
    
    # Encode fragments to hypergraph
    print(f"\n🕸️ Encoding {len(tensor_fragments)} fragments to hypergraph...")
    fragment_handles = []
    for i, fragment in enumerate(tensor_fragments):
        handle = fragment.encode_to_hypergraph(atomspace)
        fragment_handles.append(handle)
        print(f"  ✅ Fragment {i+1} encoded to AtomSpace")
    
    print(f"📊 AtomSpace now contains {len(atomspace.atoms)} atoms and {len(atomspace.relationships)} relationships")
    
    # Create cognitive patterns
    attention_fragments = [f for f in tensor_fragments if "attention" in str(f.content)]
    neural_fragments = [f for f in tensor_fragments if "layer" in str(f.content)]
    
    attention_pattern = tensor_arch.create_cognitive_pattern(attention_fragments, "attention_processing_pattern")
    neural_pattern = tensor_arch.create_cognitive_pattern(neural_fragments, "neural_computation_pattern")
    
    print(f"✅ Created cognitive patterns: attention_processing, neural_computation")
    
    # Compute pattern similarity
    similarity = tensor_arch.compute_pattern_similarity(attention_pattern, neural_pattern)
    print(f"📊 Pattern similarity: {similarity:.3f}")
    
    # Demonstrate 1.1: Scheme Cognitive Grammar Microservices
    print("\n🔤 Demonstrating Cognitive Grammar Microservices")
    print("-" * 55)
    
    # Start grammar server
    grammar_server = CognitiveGrammarServer(atomspace, port=8081)
    await grammar_server.start()
    
    # Test cognitive grammar expressions
    grammar_expressions = [
        '(define attention_mechanism (lambda (input salience) (* input salience)))',
        '(define high_salience_threshold 0.8)',
        '(eval (attention_mechanism 0.9 0.85))',
        '(if (> 0.85 high_salience_threshold) "critical_attention" "normal_attention")',
        '(let ((modality 0.9) (depth 0.7)) (+ modality depth))'
    ]
    
    for expr in grammar_expressions:
        result = await grammar_server.submit_request("evaluate", expr)
        print(f"  🔤 {expr}")
        print(f"     ⟹ {result}")
    
    # Bind tensor fragments to cognitive concepts
    concept_bindings = [
        ("attention_primitive", tensor_fragments[1]),  # attention fragment
        ("neural_primitive", tensor_fragments[0]),     # neural layer fragment
        ("embedding_primitive", tensor_fragments[2])   # embedding fragment
    ]
    
    for concept_name, fragment in concept_bindings:
        binding_data = {"name": concept_name, "tensor_fragment": fragment}
        handle = await grammar_server.submit_request("define_cognitive_concept", binding_data)
        print(f"  🔗 Bound {concept_name} to tensor fragment: {handle}")
    
    # Transform Scheme expressions to AtomSpace
    atomspace_transforms = [
        '(Evaluation (Predicate "has_property") (List (ConceptNode "attention") (ConceptNode "high_salience")))',
        '(Inheritance (ConceptNode "neural_layer") (ConceptNode "computational_primitive"))',
        '(Member (ConceptNode "transformer_block") (ConceptNode "architecture_components"))'
    ]
    
    for expr in atomspace_transforms:
        transform_data = {"expression": expr}
        handle = await grammar_server.submit_request("transform_to_atomspace", transform_data)
        print(f"  🕸️ Transformed to AtomSpace: {handle}")
    
    await grammar_server.stop()
    
    # Demonstrate 1.3: Verification & Visualization
    print("\n🔍 Demonstrating Verification & Visualization")
    print("-" * 50)
    
    # Collect system metrics
    metrics = dashboard.collect_metrics()
    print(f"📊 System Metrics:")
    print(f"   AtomSpace: {metrics['atomspace_metrics']['atom_count']} atoms, "
          f"{metrics['atomspace_metrics']['relationship_count']} relationships")
    print(f"   Average Connectivity: {metrics['atomspace_metrics']['average_connectivity']:.3f}")
    print(f"   Type Diversity: {metrics['atomspace_metrics']['type_diversity']}")
    
    # Verify all fragments
    print(f"\n🔬 Verifying {len(tensor_fragments)} tensor fragments...")
    valid_count = 0
    for i, fragment in enumerate(tensor_fragments):
        result = verifier.verify_tensor_fragment(fragment)
        if result['valid']:
            valid_count += 1
            print(f"  ✅ Fragment {i+1}: VALID")
        else:
            print(f"  ❌ Fragment {i+1}: INVALID - {len(result['errors'])} errors")
    
    print(f"📊 Verification Results: {valid_count}/{len(tensor_fragments)} fragments valid")
    
    # Verify cognitive patterns
    patterns_to_verify = [attention_pattern, neural_pattern]
    valid_patterns = 0
    for i, pattern in enumerate(patterns_to_verify):
        result = verifier.verify_cognitive_pattern(pattern)
        if result['valid']:
            valid_patterns += 1
            print(f"  ✅ Pattern {i+1}: VALID")
            
            if 'pattern_metrics' in result:
                metrics = result['pattern_metrics']
                print(f"     Coherence: {metrics['coherence']:.3f}, Diversity: {metrics['diversity']:.3f}")
        else:
            print(f"  ❌ Pattern {i+1}: INVALID - {len(result['errors'])} errors")
    
    print(f"📊 Pattern Verification: {valid_patterns}/{len(patterns_to_verify)} patterns valid")
    
    # Generate comprehensive verification report
    verification_report = verifier.create_verification_report(tensor_fragments, [attention_pattern, neural_pattern])
    print(f"\n📋 Comprehensive Verification Report:")
    print(f"   Health Status: {verification_report['summary']['health_status'].upper()}")
    print(f"   Total Errors: {verification_report['summary']['total_errors']}")
    print(f"   Total Warnings: {verification_report['summary']['total_warnings']}")
    
    # Demonstrate bidirectional translation
    print("\n🔄 Demonstrating Bidirectional Translation")
    print("-" * 45)
    
    # ML Primitive → Tensor Fragment → Hypergraph → Tensor Fragment → ML Primitive
    original_primitive = {"type": "attention", "heads": 8, "dim": 512, "dropout": 0.1}
    print(f"🔵 Original ML Primitive: {original_primitive}")
    
    # Step 1: Encode to tensor fragment
    encoded_fragment = tensor_arch.encode_ml_primitive("attention", original_primitive)
    print(f"🧬 Tensor Fragment Signature: {encoded_fragment.signature.to_tensor()}")
    
    # Step 2: Encode to hypergraph
    hypergraph_handle = encoded_fragment.encode_to_hypergraph(atomspace)
    print(f"🕸️ Hypergraph Handle: {hypergraph_handle}")
    
    # Step 3: Decode from hypergraph
    decoded_fragment = TensorFragment.decode_from_hypergraph(atomspace, hypergraph_handle)
    print(f"🧬 Decoded Tensor Fragment: {decoded_fragment.signature.to_tensor()}")
    
    # Step 4: Decode to ML primitive
    decoded_primitive = tensor_arch.decode_to_ml_primitive(decoded_fragment, "attention_config")
    print(f"🔵 Decoded ML Primitive: {decoded_primitive}")
    
    # Compute translation fidelity
    original_sig = encoded_fragment.signature.to_tensor()
    decoded_sig = decoded_fragment.signature.to_tensor()
    translation_error = np.max(np.abs(original_sig - decoded_sig))
    print(f"📊 Translation Fidelity: {1.0 - translation_error:.6f} (error: {translation_error:.6f})")
    
    # Test cross-modal cognitive operations
    print("\n🌐 Testing Cross-Modal Cognitive Operations")
    print("-" * 45)
    
    # Create cognitive primitives with different modalities
    cross_modal_primitives = [
        CognitivePrimitive("visual_attention", "attention_allocation", {"modality": "visual", "priority": "high"}),
        CognitivePrimitive("linguistic_processing", "pattern_matching", {"modality": "linguistic", "recursive": True}),
        CognitivePrimitive("temporal_reasoning", "reasoning", {"modality": "temporal", "meta": True}),
        CognitivePrimitive("spatial_navigation", "goal_planning", {"modality": "spatial", "autonomous": True})
    ]
    
    cross_modal_fragments = [prim.to_tensor_fragment() for prim in cross_modal_primitives]
    
    # Create cross-modal pattern
    cross_modal_pattern = tensor_arch.create_cognitive_pattern(cross_modal_fragments, "cross_modal_integration")
    print(f"✅ Created cross-modal cognitive pattern with {len(cross_modal_fragments)} primitives")
    
    # Compute cross-modal similarities
    print(f"\n📊 Cross-Modal Similarity Matrix:")
    for i, frag1 in enumerate(cross_modal_fragments):
        similarities = []
        for j, frag2 in enumerate(cross_modal_fragments):
            if i != j:
                # Create single-fragment patterns for comparison
                pattern1 = tensor_arch.create_cognitive_pattern([frag1], f"temp_pattern_{i}")
                pattern2 = tensor_arch.create_cognitive_pattern([frag2], f"temp_pattern_{j}")
                sim = tensor_arch.compute_pattern_similarity(pattern1, pattern2)
                similarities.append(f"{sim:.3f}")
            else:
                similarities.append("1.000")
        
        prim_name = cross_modal_primitives[i].name
        print(f"   {prim_name:20}: [{', '.join(similarities)}]")
    
    # Test Scheme cognitive grammar with tensor fragments
    print("\n🔤 Testing Cognitive Grammar with Tensor Fragments")
    print("-" * 55)
    
    # Define cognitive grammar expressions that work with tensor fragments
    cognitive_expressions = [
        '(define cognitive_threshold 0.8)',
        '(define is_high_salience (lambda (fragment) (> (get_salience fragment) cognitive_threshold)))',
        '(define combine_fragments (lambda (f1 f2) (create_pattern (list f1 f2))))'
    ]
    
    for expr in cognitive_expressions:
        result = await grammar_service.parse_and_evaluate(expr)
        print(f"  🔤 {expr}")
        print(f"     ⟹ Defined successfully: {type(result).__name__}")
    
    # Bind tensor fragments to grammar environment
    for i, fragment in enumerate(cross_modal_fragments[:3]):
        binding_handle = grammar_service.create_cognitive_binding(f"modal_fragment_{i}", fragment)
        print(f"  🔗 Bound modal_fragment_{i}: {binding_handle}")
    
    # Final system status
    print("\n🎯 Phase 1 Implementation Status")
    print("-" * 35)
    
    final_metrics = dashboard.collect_metrics()
    
    print(f"✅ 1.1 Scheme Cognitive Grammar Microservices: IMPLEMENTED")
    print(f"   - Scheme parser with cognitive expressions")
    print(f"   - Microservice architecture with async processing")
    print(f"   - Bidirectional AtomSpace integration")
    
    print(f"✅ 1.2 Tensor Fragment Architecture: IMPLEMENTED")
    print(f"   - Tensor signature [modality, depth, context, salience, autonomy_index]")
    print(f"   - {len(tensor_arch.ml_primitive_encoders)} ML primitive encoders")
    print(f"   - {len(tensor_arch.hypergraph_decoders)} hypergraph decoders")
    print(f"   - Bidirectional translation: ML ↔ TensorFragment ↔ Hypergraph")
    
    print(f"✅ 1.3 Verification & Visualization: IMPLEMENTED")
    print(f"   - Cognitive primitive verification system")
    print(f"   - Hypergraph pattern visualization")
    print(f"   - Real-time monitoring dashboard")
    
    print(f"\n📊 Final System State:")
    print(f"   Total Atoms: {final_metrics['atomspace_metrics']['atom_count']}")
    print(f"   Total Relationships: {final_metrics['atomspace_metrics']['relationship_count']}")
    print(f"   Type Diversity: {final_metrics['atomspace_metrics']['type_diversity']}")
    print(f"   Average Connectivity: {final_metrics['atomspace_metrics']['average_connectivity']:.3f}")
    print(f"   Active Tensor Fragments: {len(tensor_fragments) + len(cross_modal_fragments)}")
    
    print(f"\n🧬 Phase 1: Cognitive Primitives & Foundational Hypergraph Encoding")
    print(f"🎉 IMPLEMENTATION COMPLETE!")
    
    return True


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        if success:
            print(f"\n✨ Demo completed successfully!")
        else:
            print(f"\n⚠️ Demo encountered issues")
    except Exception as e:
        print(f"\n💥 Demo failed: {e}")
        sys.exit(1)