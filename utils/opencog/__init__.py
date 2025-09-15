"""
OpenCog components implementation for Archon.
"""

# Import the custom implementations
from utils.opencog.atomspace import AtomSpace
from utils.opencog.cogserver import CogServer
from utils.opencog.utilities import Utilities
from utils.opencog.tensor_fragments import (
    TensorFragmentArchitecture, 
    TensorSignature, 
    TensorFragment, 
    CognitivePrimitive,
    create_default_ml_primitive_encoders,
    create_default_hypergraph_decoders
)
from utils.opencog.cognitive_grammar import (
    CognitiveGrammarMicroservice,
    CognitiveGrammarServer, 
    CognitiveGrammarOrchestrator,
    SchemeParser
)
from utils.opencog.verification import (
    CognitivePrimitiveVerifier,
    HypergraphVisualizer,
    CognitiveDashboard,
    run_verification_suite
)

# Define the opencog namespace
class opencog:
    """OpenCog namespace for component organization."""
    
    # Define submodules
    atomspace = None
    cogserver = None
    utilities = None
    tensor_fragments = None
    cognitive_grammar = None
    verification = None
    
# Attach the component classes to the opencog namespace
opencog.atomspace = AtomSpace
opencog.cogserver = CogServer
opencog.utilities = Utilities

# Attach Phase 1 cognitive primitives components
opencog.tensor_fragments = TensorFragmentArchitecture
opencog.cognitive_grammar = CognitiveGrammarMicroservice
opencog.verification = CognitivePrimitiveVerifier

# Phase 2: ECAN Attention Allocation & Resource Kernel
from utils.opencog.ecan_attention import (
    ECANAttentionManager,
    AttentionValue,
    ResourceAllocation,
    ECANResourceKernel,
    AttentionType,
    create_ecan_enhanced_tensor_fragment,
    sync_tensor_fragments_with_ecan
)
from utils.opencog.dynamic_mesh import (
    DynamicAttentionMesh,
    MeshTopology,
    AttentionFlow,
    MeshNode
)

# Attach Phase 2 ECAN components to opencog namespace
opencog.ecan_attention = ECANAttentionManager
opencog.dynamic_mesh = DynamicAttentionMesh

# Export key classes for direct import
__all__ = [
    'AtomSpace', 'CogServer', 'Utilities',
    'TensorFragmentArchitecture', 'TensorSignature', 'TensorFragment', 'CognitivePrimitive',
    'CognitiveGrammarMicroservice', 'CognitiveGrammarServer', 'CognitiveGrammarOrchestrator',
    'CognitivePrimitiveVerifier', 'HypergraphVisualizer', 'CognitiveDashboard',
    'SchemeParser', 'run_verification_suite',
    'create_default_ml_primitive_encoders', 'create_default_hypergraph_decoders',
    # Phase 2 ECAN components
    'ECANAttentionManager', 'AttentionValue', 'ResourceAllocation', 'ECANResourceKernel',
    'DynamicAttentionMesh', 'MeshTopology', 'AttentionFlow', 'MeshNode',
    'create_ecan_enhanced_tensor_fragment', 'sync_tensor_fragments_with_ecan'
]
