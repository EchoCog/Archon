"""
Tensor Fragment Architecture for OpenCog integration.

This module implements the tensor signature system [modality, depth, context, salience, autonomy_index]
for encoding agentic kernel ML primitives and their bidirectional translation with AtomSpace hypergraph patterns.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum


class ModalityType(Enum):
    """Enumeration of cognitive modalities."""
    LINGUISTIC = "linguistic"
    VISUAL = "visual"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    CONCEPTUAL = "conceptual"
    BEHAVIORAL = "behavioral"
    EMOTIONAL = "emotional"
    SENSORY = "sensory"


@dataclass
class TensorSignature:
    """
    Tensor signature representation: [modality, depth, context, salience, autonomy_index]
    
    Attributes:
        modality: Type of cognitive modality (0.0-1.0 normalized)
        depth: Processing depth level (0.0-1.0, deeper = more abstract)
        context: Contextual relevance (0.0-1.0)
        salience: Attention weight (0.0-1.0)
        autonomy_index: Autonomous behavior potential (0.0-1.0)
    """
    modality: float
    depth: float
    context: float
    salience: float
    autonomy_index: float
    
    def to_tensor(self) -> np.ndarray:
        """Convert to numpy tensor."""
        return np.array([self.modality, self.depth, self.context, self.salience, self.autonomy_index])
    
    @classmethod
    def from_tensor(cls, tensor: np.ndarray) -> 'TensorSignature':
        """Create from numpy tensor."""
        if len(tensor) != 5:
            raise ValueError("Tensor signature must have exactly 5 dimensions")
        return cls(*tensor.tolist())
    
    def normalize(self) -> 'TensorSignature':
        """Ensure all values are in [0.0, 1.0] range."""
        return TensorSignature(
            modality=np.clip(self.modality, 0.0, 1.0),
            depth=np.clip(self.depth, 0.0, 1.0),
            context=np.clip(self.context, 0.0, 1.0),
            salience=np.clip(self.salience, 0.0, 1.0),
            autonomy_index=np.clip(self.autonomy_index, 0.0, 1.0)
        )


class TensorFragment:
    """
    Represents a tensor fragment with cognitive primitives encoding.
    
    This is the core abstraction for bidirectional translation between
    agentic ML primitives and AtomSpace hypergraph patterns.
    """
    
    def __init__(self, signature: TensorSignature, content: Any, atom_handle: Optional[str] = None):
        """
        Initialize a tensor fragment.
        
        Args:
            signature: The tensor signature for this fragment
            content: The actual content/data being encoded
            atom_handle: Optional AtomSpace handle for this fragment
        """
        self.signature = signature.normalize()
        self.content = content
        self.atom_handle = atom_handle
        self.metadata = {}
    
    def encode_to_hypergraph(self, atomspace) -> str:
        """
        Encode this tensor fragment into AtomSpace hypergraph representation.
        
        Args:
            atomspace: The AtomSpace instance to encode into
            
        Returns:
            The atom handle for the encoded fragment
        """
        # Create primary concept node for the content
        if isinstance(self.content, str):
            content_node = atomspace.add_node("ConceptNode", self.content)
        else:
            content_node = atomspace.add_node("ConceptNode", str(self.content))
        
        # Create tensor signature node
        sig_node = atomspace.add_node("TensorSignatureNode", f"sig_{id(self)}")
        
        # Create modality nodes
        modality_node = atomspace.add_node("ModalityNode", f"modality_{self.signature.modality:.3f}")
        depth_node = atomspace.add_node("DepthNode", f"depth_{self.signature.depth:.3f}")
        context_node = atomspace.add_node("ContextNode", f"context_{self.signature.context:.3f}")
        salience_node = atomspace.add_node("SalienceNode", f"salience_{self.signature.salience:.3f}")
        autonomy_node = atomspace.add_node("AutonomyNode", f"autonomy_{self.signature.autonomy_index:.3f}")
        
        # Create signature structure as a ListLink
        signature_list = atomspace.add_link("ListLink", [
            modality_node, depth_node, context_node, salience_node, autonomy_node
        ])
        
        # Link signature to signature node
        atomspace.add_link("HasSignatureLink", [sig_node, signature_list])
        
        # Link content to signature
        fragment_link = atomspace.add_link("TensorFragmentLink", [content_node, sig_node])
        
        self.atom_handle = fragment_link
        return fragment_link
    
    @classmethod
    def decode_from_hypergraph(cls, atomspace, atom_handle: str) -> Optional['TensorFragment']:
        """
        Decode a tensor fragment from AtomSpace hypergraph representation.
        
        Args:
            atomspace: The AtomSpace instance to decode from
            atom_handle: The atom handle to decode
            
        Returns:
            The decoded TensorFragment or None if decoding fails
        """
        # Get the fragment link
        fragment_link = atomspace.relationships.get(atom_handle)
        if not fragment_link or fragment_link["type"] != "TensorFragmentLink":
            return None
        
        outgoing_set = fragment_link["outgoing_set"]
        if len(outgoing_set) != 2:
            return None
        
        content_handle, sig_handle = outgoing_set
        
        # Get content
        content_atom = atomspace.get_atom(content_handle)
        if not content_atom:
            return None
        content = content_atom["name"]
        
        # Get signature
        sig_incoming = atomspace.get_incoming_set(sig_handle)
        signature_link = None
        for link_handle in sig_incoming:
            link = atomspace.relationships.get(link_handle)
            if link and link["type"] == "HasSignatureLink":
                signature_link = link
                break
        
        if not signature_link:
            return None
        
        # Extract signature values from the list
        sig_list_handle = signature_link["outgoing_set"][1]
        sig_list = atomspace.relationships.get(sig_list_handle)
        if not sig_list or sig_list["type"] != "ListLink":
            return None
        
        sig_nodes = sig_list["outgoing_set"]
        if len(sig_nodes) != 5:
            return None
        
        # Parse signature values from node names
        signature_values = []
        for node_handle in sig_nodes:
            node = atomspace.get_atom(node_handle)
            if node:
                # Extract numeric value from node name (e.g., "modality_0.750" -> 0.750)
                node_name = node["name"]
                value_str = node_name.split("_")[-1]
                try:
                    value = float(value_str)
                    signature_values.append(value)
                except ValueError:
                    signature_values.append(0.0)
            else:
                signature_values.append(0.0)
        
        if len(signature_values) == 5:
            signature = TensorSignature(*signature_values)
            return cls(signature, content, atom_handle)
        
        return None


class CognitivePrimitive:
    """
    Represents an atomic cognitive operation that can be encoded as tensor fragments.
    """
    
    def __init__(self, name: str, operation_type: str, parameters: Dict[str, Any]):
        """
        Initialize a cognitive primitive.
        
        Args:
            name: Name of the primitive
            operation_type: Type of cognitive operation
            parameters: Parameters for the operation
        """
        self.name = name
        self.operation_type = operation_type
        self.parameters = parameters
    
    def to_tensor_fragment(self) -> TensorFragment:
        """
        Convert this cognitive primitive to a tensor fragment.
        
        Returns:
            A TensorFragment representation of this primitive
        """
        # Determine modality based on operation type
        modality_map = {
            'concept_learning': 0.8,
            'pattern_matching': 0.6,
            'reasoning': 0.9,
            'memory_retrieval': 0.7,
            'goal_planning': 0.85,
            'attention_allocation': 0.75,
            'behavior_generation': 0.65
        }
        
        modality = modality_map.get(self.operation_type, 0.5)
        
        # Calculate depth based on complexity
        complexity_indicators = ['hierarchical', 'recursive', 'meta', 'abstract']
        depth = 0.3 + 0.2 * sum(1 for indicator in complexity_indicators 
                                if indicator in str(self.parameters).lower())
        depth = min(depth, 1.0)
        
        # Context based on parameter richness
        context = min(0.1 + 0.1 * len(self.parameters), 1.0)
        
        # Salience based on priority or importance
        salience = self.parameters.get('priority', 0.5)
        if isinstance(salience, str):
            salience_map = {'low': 0.2, 'medium': 0.5, 'high': 0.8, 'critical': 1.0}
            salience = salience_map.get(salience.lower(), 0.5)
        
        # Autonomy based on self-directed capability
        autonomy_keywords = ['autonomous', 'self', 'independent', 'adaptive']
        autonomy = 0.3 + 0.2 * sum(1 for keyword in autonomy_keywords 
                                  if keyword in str(self.parameters).lower())
        autonomy = min(autonomy, 1.0)
        
        signature = TensorSignature(modality, depth, context, salience, autonomy)
        return TensorFragment(signature, self.name)


class TensorFragmentArchitecture:
    """
    Main architecture class for managing tensor fragments and their hypergraph encoding.
    """
    
    def __init__(self, atomspace=None):
        """
        Initialize the tensor fragment architecture.
        
        Args:
            atomspace: The AtomSpace instance to use for hypergraph operations
        """
        self.atomspace = atomspace
        self.fragments = {}
        self.ml_primitive_encoders = {}
        self.hypergraph_decoders = {}
    
    def register_ml_primitive_encoder(self, primitive_type: str, encoder_func):
        """
        Register an encoder for a specific ML primitive type.
        
        Args:
            primitive_type: The type of ML primitive
            encoder_func: Function that converts the primitive to TensorFragment
        """
        self.ml_primitive_encoders[primitive_type] = encoder_func
    
    def register_hypergraph_decoder(self, pattern_type: str, decoder_func):
        """
        Register a decoder for a specific hypergraph pattern type.
        
        Args:
            pattern_type: The type of hypergraph pattern
            decoder_func: Function that converts hypergraph to ML primitive
        """
        self.hypergraph_decoders[pattern_type] = decoder_func
    
    def encode_ml_primitive(self, primitive_type: str, primitive_data: Any) -> Optional[TensorFragment]:
        """
        Encode an ML primitive into a tensor fragment.
        
        Args:
            primitive_type: The type of ML primitive
            primitive_data: The primitive data to encode
            
        Returns:
            The encoded TensorFragment or None if encoding fails
        """
        if primitive_type in self.ml_primitive_encoders:
            encoder = self.ml_primitive_encoders[primitive_type]
            return encoder(primitive_data)
        
        # Default encoding for unknown primitives
        signature = TensorSignature(0.5, 0.5, 0.5, 0.5, 0.5)
        return TensorFragment(signature, primitive_data)
    
    def decode_to_ml_primitive(self, fragment: TensorFragment, target_type: str) -> Optional[Any]:
        """
        Decode a tensor fragment back to an ML primitive.
        
        Args:
            fragment: The TensorFragment to decode
            target_type: The target ML primitive type
            
        Returns:
            The decoded ML primitive or None if decoding fails
        """
        if target_type in self.hypergraph_decoders:
            decoder = self.hypergraph_decoders[target_type]
            return decoder(fragment)
        
        # Default decoding returns the content
        return fragment.content
    
    def create_cognitive_pattern(self, fragments: List[TensorFragment], pattern_name: str) -> str:
        """
        Create a cognitive pattern from multiple tensor fragments.
        
        Args:
            fragments: List of TensorFragments to combine
            pattern_name: Name for the pattern
            
        Returns:
            The AtomSpace handle for the created pattern
        """
        if not self.atomspace:
            raise ValueError("AtomSpace required for hypergraph operations")
        
        # Encode all fragments to hypergraph
        fragment_handles = []
        for fragment in fragments:
            handle = fragment.encode_to_hypergraph(self.atomspace)
            fragment_handles.append(handle)
        
        # Create pattern node
        pattern_node = self.atomspace.add_node("PatternNode", pattern_name)
        
        # Create pattern structure
        pattern_list = self.atomspace.add_link("ListLink", fragment_handles)
        pattern_link = self.atomspace.add_link("PatternLink", [pattern_node, pattern_list])
        
        return pattern_link
    
    def extract_cognitive_pattern(self, pattern_handle: str) -> List[TensorFragment]:
        """
        Extract tensor fragments from a cognitive pattern.
        
        Args:
            pattern_handle: The AtomSpace handle for the pattern
            
        Returns:
            List of TensorFragments that comprise the pattern
        """
        if not self.atomspace:
            return []
        
        # Get pattern link
        pattern_link = self.atomspace.relationships.get(pattern_handle)
        if not pattern_link or pattern_link["type"] != "PatternLink":
            return []
        
        outgoing_set = pattern_link["outgoing_set"]
        if len(outgoing_set) != 2:
            return []
        
        pattern_list_handle = outgoing_set[1]
        pattern_list = self.atomspace.relationships.get(pattern_list_handle)
        if not pattern_list or pattern_list["type"] != "ListLink":
            return []
        
        # Decode each fragment
        fragments = []
        for fragment_handle in pattern_list["outgoing_set"]:
            fragment = TensorFragment.decode_from_hypergraph(self.atomspace, fragment_handle)
            if fragment:
                fragments.append(fragment)
        
        return fragments
    
    def compute_pattern_similarity(self, pattern1_handle: str, pattern2_handle: str) -> float:
        """
        Compute similarity between two cognitive patterns based on their tensor signatures.
        
        Args:
            pattern1_handle: Handle for first pattern
            pattern2_handle: Handle for second pattern
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        fragments1 = self.extract_cognitive_pattern(pattern1_handle)
        fragments2 = self.extract_cognitive_pattern(pattern2_handle)
        
        if not fragments1 or not fragments2:
            return 0.0
        
        # Compute average signature similarity
        total_similarity = 0.0
        comparisons = 0
        
        for f1 in fragments1:
            for f2 in fragments2:
                # Compute cosine similarity between tensor signatures
                tensor1 = f1.signature.to_tensor()
                tensor2 = f2.signature.to_tensor()
                
                dot_product = np.dot(tensor1, tensor2)
                norm1 = np.linalg.norm(tensor1)
                norm2 = np.linalg.norm(tensor2)
                
                if norm1 > 0 and norm2 > 0:
                    similarity = dot_product / (norm1 * norm2)
                    total_similarity += similarity
                    comparisons += 1
        
        return total_similarity / comparisons if comparisons > 0 else 0.0


def create_default_ml_primitive_encoders() -> Dict[str, callable]:
    """
    Create default encoders for common ML primitives.
    
    Returns:
        Dictionary mapping primitive types to encoder functions
    """
    
    def encode_neural_network_layer(layer_data):
        """Encode a neural network layer as a tensor fragment."""
        layer_type = layer_data.get('type', 'unknown')
        layer_size = layer_data.get('size', 1)
        activation = layer_data.get('activation', 'linear')
        
        # High modality for neural processing
        modality = 0.9
        # Depth based on layer complexity  
        depth = min(0.3 + layer_size * 0.001, 1.0)
        # Context based on activation function
        context_map = {'relu': 0.6, 'sigmoid': 0.7, 'tanh': 0.8, 'softmax': 0.9}
        context = context_map.get(activation, 0.5)
        # High salience for neural components
        salience = 0.8
        # Moderate autonomy for layers
        autonomy = 0.6
        
        signature = TensorSignature(modality, depth, context, salience, autonomy)
        return TensorFragment(signature, f"{layer_type}_layer_{layer_size}")
    
    def encode_attention_mechanism(attention_data):
        """Encode an attention mechanism as a tensor fragment."""
        attention_type = attention_data.get('type', 'self_attention')
        num_heads = attention_data.get('heads', 8)
        
        # Very high modality for attention
        modality = 0.95
        # Deep processing for attention
        depth = 0.85
        # High context relevance
        context = 0.9
        # Critical salience
        salience = 0.95
        # High autonomy for attention allocation
        autonomy = 0.8
        
        signature = TensorSignature(modality, depth, context, salience, autonomy)
        return TensorFragment(signature, f"{attention_type}_{num_heads}heads")
    
    def encode_embedding_vector(embedding_data):
        """Encode an embedding vector as a tensor fragment."""
        dimensions = embedding_data.get('dimensions', 768)
        embedding_type = embedding_data.get('type', 'word_embedding')
        
        # High conceptual modality
        modality = 0.85
        # Medium depth for embeddings
        depth = 0.6
        # Context based on dimensions
        context = min(0.4 + dimensions * 0.0005, 1.0)
        # Medium salience
        salience = 0.6
        # Low autonomy for static embeddings
        autonomy = 0.3
        
        signature = TensorSignature(modality, depth, context, salience, autonomy)
        return TensorFragment(signature, f"{embedding_type}_{dimensions}d")
    
    return {
        'neural_layer': encode_neural_network_layer,
        'attention': encode_attention_mechanism,
        'embedding': encode_embedding_vector
    }


def create_default_hypergraph_decoders() -> Dict[str, callable]:
    """
    Create default decoders for hypergraph patterns to ML primitives.
    
    Returns:
        Dictionary mapping pattern types to decoder functions
    """
    
    def decode_to_neural_config(fragment: TensorFragment):
        """Decode a tensor fragment to neural network configuration."""
        # Parse layer information from content
        content = str(fragment.content)
        parts = content.split('_')
        
        config = {
            'type': parts[0] if len(parts) > 0 else 'dense',
            'activation': 'relu' if fragment.signature.context < 0.7 else 'sigmoid',
            'size': int(fragment.signature.depth * 1000) + 64,
            'attention_weight': fragment.signature.salience
        }
        return config
    
    def decode_to_attention_config(fragment: TensorFragment):
        """Decode a tensor fragment to attention mechanism configuration."""
        content = str(fragment.content)
        
        config = {
            'type': 'multi_head_attention',
            'num_heads': max(1, int(fragment.signature.autonomy_index * 16)),
            'key_dim': int(fragment.signature.depth * 512) + 64,
            'dropout_rate': 1.0 - fragment.signature.salience
        }
        return config
    
    def decode_to_embedding_config(fragment: TensorFragment):
        """Decode a tensor fragment to embedding configuration."""
        config = {
            'dimensions': int(fragment.signature.context * 1000) + 128,
            'vocab_size': int(fragment.signature.depth * 50000) + 1000,
            'trainable': fragment.signature.autonomy_index > 0.5
        }
        return config
    
    return {
        'neural_config': decode_to_neural_config,
        'attention_config': decode_to_attention_config,
        'embedding_config': decode_to_embedding_config
    }