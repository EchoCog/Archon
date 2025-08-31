"""
Verification and Visualization tools for cognitive primitives and hypergraph patterns.

This module provides tools to verify the correctness of cognitive primitives
and visualize hypergraph patterns for debugging and analysis.
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime
import io
import base64


class CognitivePrimitiveVerifier:
    """
    Verifies the correctness and consistency of cognitive primitives.
    """
    
    def __init__(self, atomspace=None, tensor_architecture=None):
        """
        Initialize the verifier.
        
        Args:
            atomspace: AtomSpace instance for verification
            tensor_architecture: TensorFragmentArchitecture instance
        """
        self.atomspace = atomspace
        self.tensor_architecture = tensor_architecture
        self.verification_results = []
    
    def verify_tensor_signature(self, signature) -> Dict[str, Any]:
        """
        Verify that a tensor signature is well-formed.
        
        Args:
            signature: TensorSignature to verify
            
        Returns:
            Verification result dictionary
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'metrics': {}
        }
        
        # Check value ranges
        tensor = signature.to_tensor()
        for i, (dim_name, value) in enumerate(zip(
            ['modality', 'depth', 'context', 'salience', 'autonomy_index'], 
            tensor
        )):
            if not (0.0 <= value <= 1.0):
                result['valid'] = False
                result['errors'].append(f"{dim_name} value {value} out of range [0.0, 1.0]")
        
        # Check for reasonable value distributions
        if np.std(tensor) < 0.1:
            result['warnings'].append("Low variance in tensor signature - may indicate poor encoding")
        
        # Check for extreme values
        if np.max(tensor) - np.min(tensor) > 0.9:
            result['warnings'].append("Very high dynamic range - verify encoding logic")
        
        # Compute metrics
        result['metrics'] = {
            'mean': float(np.mean(tensor)),
            'std': float(np.std(tensor)),
            'range': float(np.max(tensor) - np.min(tensor)),
            'entropy': float(-np.sum(tensor * np.log(tensor + 1e-10)))  # Information entropy
        }
        
        return result
    
    def verify_tensor_fragment(self, fragment) -> Dict[str, Any]:
        """
        Verify a tensor fragment and its hypergraph encoding.
        
        Args:
            fragment: TensorFragment to verify
            
        Returns:
            Verification result dictionary
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'signature_verification': {},
            'hypergraph_verification': {}
        }
        
        # Verify signature
        sig_result = self.verify_tensor_signature(fragment.signature)
        result['signature_verification'] = sig_result
        if not sig_result['valid']:
            result['valid'] = False
            result['errors'].extend(sig_result['errors'])
        result['warnings'].extend(sig_result['warnings'])
        
        # Verify hypergraph encoding if AtomSpace is available
        if self.atomspace:
            try:
                # Encode to hypergraph
                handle = fragment.encode_to_hypergraph(self.atomspace)
                
                # Try to decode back
                decoded_fragment = fragment.decode_from_hypergraph(self.atomspace, handle)
                
                if decoded_fragment:
                    # Compare signatures
                    original_tensor = fragment.signature.to_tensor()
                    decoded_tensor = decoded_fragment.signature.to_tensor()
                    
                    diff = np.abs(original_tensor - decoded_tensor)
                    max_diff = np.max(diff)
                    
                    if max_diff > 0.01:  # Allow small floating point errors
                        result['warnings'].append(f"Signature encoding/decoding difference: {max_diff}")
                    
                    result['hypergraph_verification'] = {
                        'encode_success': True,
                        'decode_success': True,
                        'roundtrip_error': float(max_diff),
                        'atom_handle': handle
                    }
                else:
                    result['errors'].append("Failed to decode tensor fragment from hypergraph")
                    result['valid'] = False
                    result['hypergraph_verification'] = {
                        'encode_success': True,
                        'decode_success': False
                    }
                    
            except Exception as e:
                result['errors'].append(f"Hypergraph encoding error: {str(e)}")
                result['valid'] = False
                result['hypergraph_verification'] = {
                    'encode_success': False,
                    'error': str(e)
                }
        
        return result
    
    def verify_cognitive_pattern(self, pattern_handle: str) -> Dict[str, Any]:
        """
        Verify a cognitive pattern in the AtomSpace.
        
        Args:
            pattern_handle: AtomSpace handle for the pattern
            
        Returns:
            Verification result dictionary
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'pattern_metrics': {}
        }
        
        if not self.atomspace:
            result['errors'].append("AtomSpace required for pattern verification")
            result['valid'] = False
            return result
        
        if not self.tensor_architecture:
            result['errors'].append("TensorFragmentArchitecture required for pattern verification")
            result['valid'] = False
            return result
        
        try:
            # Extract fragments from pattern
            fragments = self.tensor_architecture.extract_cognitive_pattern(pattern_handle)
            
            if not fragments:
                result['errors'].append("No tensor fragments found in pattern")
                result['valid'] = False
                return result
            
            # Verify each fragment
            fragment_results = []
            for i, fragment in enumerate(fragments):
                frag_result = self.verify_tensor_fragment(fragment)
                fragment_results.append(frag_result)
                if not frag_result['valid']:
                    result['valid'] = False
                    result['errors'].append(f"Fragment {i} verification failed")
            
            # Compute pattern-level metrics
            signatures = [f.signature.to_tensor() for f in fragments]
            if signatures:
                signature_matrix = np.array(signatures)
                
                result['pattern_metrics'] = {
                    'fragment_count': len(fragments),
                    'signature_mean': signature_matrix.mean(axis=0).tolist(),
                    'signature_std': signature_matrix.std(axis=0).tolist(),
                    'coherence': float(1.0 - np.mean(np.std(signature_matrix, axis=0))),
                    'diversity': float(np.mean(np.std(signature_matrix, axis=1)))
                }
            
            result['fragment_verifications'] = fragment_results
            
        except Exception as e:
            result['errors'].append(f"Pattern verification error: {str(e)}")
            result['valid'] = False
        
        return result
    
    def run_full_verification_suite(self) -> Dict[str, Any]:
        """
        Run a full verification suite on the cognitive architecture.
        
        Returns:
            Complete verification report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'atomspace_status': {},
            'tensor_architecture_status': {},
            'pattern_verifications': [],
            'overall_health': 'unknown'
        }
        
        # Check AtomSpace status
        if self.atomspace:
            report['atomspace_status'] = {
                'available': True,
                'atom_count': len(self.atomspace.atoms),
                'relationship_count': len(self.atomspace.relationships),
                'type_count': len(self.atomspace.types)
            }
        else:
            report['atomspace_status'] = {'available': False}
        
        # Check tensor architecture status
        if self.tensor_architecture:
            report['tensor_architecture_status'] = {
                'available': True,
                'fragment_count': len(self.tensor_architecture.fragments),
                'encoder_count': len(self.tensor_architecture.ml_primitive_encoders),
                'decoder_count': len(self.tensor_architecture.hypergraph_decoders)
            }
        else:
            report['tensor_architecture_status'] = {'available': False}
        
        # Determine overall health
        errors = 0
        warnings = 0
        
        for result in self.verification_results:
            if not result.get('valid', True):
                errors += 1
            warnings += len(result.get('warnings', []))
        
        if errors == 0:
            report['overall_health'] = 'healthy' if warnings == 0 else 'warning'
        else:
            report['overall_health'] = 'error'
        
        report['error_count'] = errors
        report['warning_count'] = warnings
        
        return report
    
    def create_verification_report(self, fragments: List, patterns: List[str] = None) -> Dict[str, Any]:
        """
        Create a comprehensive verification report.
        
        Args:
            fragments: List of TensorFragments to verify
            patterns: List of pattern handles to verify
            
        Returns:
            Comprehensive verification report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {},
            'fragment_verifications': [],
            'pattern_verifications': [],
            'visualizations': {}
        }
        
        # Verify fragments
        valid_fragments = 0
        total_errors = 0
        total_warnings = 0
        
        for i, fragment in enumerate(fragments):
            result = self.verify_tensor_fragment(fragment)
            result['fragment_id'] = i
            report['fragment_verifications'].append(result)
            
            if result['valid']:
                valid_fragments += 1
            total_errors += len(result['errors'])
            total_warnings += len(result['warnings'])
        
        # Verify patterns
        valid_patterns = 0
        if patterns and self.tensor_architecture:
            for pattern_handle in patterns:
                result = self.verify_cognitive_pattern(pattern_handle)
                result['pattern_handle'] = pattern_handle
                report['pattern_verifications'].append(result)
                
                if result['valid']:
                    valid_patterns += 1
                total_errors += len(result['errors'])
                total_warnings += len(result['warnings'])
        
        # Summary
        report['summary'] = {
            'total_fragments': len(fragments),
            'valid_fragments': valid_fragments,
            'total_patterns': len(patterns) if patterns else 0,
            'valid_patterns': valid_patterns,
            'total_errors': total_errors,
            'total_warnings': total_warnings,
            'health_status': 'healthy' if total_errors == 0 else 'error' if total_errors > 5 else 'warning'
        }
        
        return report


class HypergraphVisualizer:
    """
    Provides visualization capabilities for hypergraph patterns and cognitive structures.
    """
    
    def __init__(self, atomspace=None):
        """
        Initialize the visualizer.
        
        Args:
            atomspace: AtomSpace instance for visualization
        """
        self.atomspace = atomspace
        self.color_schemes = self._create_color_schemes()
    
    def _create_color_schemes(self) -> Dict[str, Dict[str, str]]:
        """Create color schemes for different atom types."""
        return {
            'default': {
                'ConceptNode': '#E3F2FD',
                'PredicateNode': '#F3E5F5',
                'VariableNode': '#E8F5E8',
                'SchemaNode': '#FFF3E0',
                'TensorSignatureNode': '#FFEBEE',
                'ModalityNode': '#E0F2F1',
                'DepthNode': '#E8EAF6',
                'ContextNode': '#F1F8E9',
                'SalienceNode': '#FFF8E1',
                'AutonomyNode': '#FCE4EC',
                'EvaluationLink': '#BBDEFB',
                'InheritanceLink': '#C8E6C9',
                'ListLink': '#DCEDC8',
                'TensorFragmentLink': '#FFCDD2',
                'PatternLink': '#D1C4E9'
            }
        }
    
    def create_networkx_graph(self, filter_types: Optional[List[str]] = None) -> nx.Graph:
        """
        Create a NetworkX graph from AtomSpace representation.
        
        Args:
            filter_types: Optional list of atom types to include
            
        Returns:
            NetworkX Graph object
        """
        if not self.atomspace:
            return nx.Graph()
        
        G = nx.Graph()
        
        # Add nodes
        for handle, atom in self.atomspace.atoms.items():
            if filter_types and atom['type'] not in filter_types:
                continue
            
            G.add_node(handle, 
                      label=atom['name'],
                      type=atom['type'],
                      color=self.color_schemes['default'].get(atom['type'], '#FFFFFF'))
        
        # Add edges (links)
        for handle, link in self.atomspace.relationships.items():
            if filter_types and link['type'] not in filter_types:
                continue
            
            outgoing_set = link['outgoing_set']
            
            # Add link node
            G.add_node(handle,
                      label=link['type'],
                      type=link['type'],
                      color=self.color_schemes['default'].get(link['type'], '#F5F5F5'),
                      shape='box')
            
            # Connect link to its atoms
            for atom_handle in outgoing_set:
                if atom_handle in G.nodes:
                    G.add_edge(handle, atom_handle)
        
        return G
    
    def visualize_hypergraph(self, title: str = "AtomSpace Hypergraph", 
                           filter_types: Optional[List[str]] = None,
                           layout: str = 'spring',
                           save_path: Optional[str] = None) -> Optional[str]:
        """
        Visualize the AtomSpace hypergraph.
        
        Args:
            title: Title for the visualization
            filter_types: Optional atom types to include
            layout: Layout algorithm ('spring', 'circular', 'random')
            save_path: Optional path to save the visualization
            
        Returns:
            Base64 encoded image if save_path is None, otherwise None
        """
        G = self.create_networkx_graph(filter_types)
        
        if len(G.nodes) == 0:
            print("No nodes to visualize")
            return None
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(G, k=2, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'random':
            pos = nx.random_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        # Get node colors and shapes
        node_colors = [G.nodes[node].get('color', '#FFFFFF') for node in G.nodes]
        node_shapes = [G.nodes[node].get('shape', 'circle') for node in G.nodes]
        
        # Draw nodes with different shapes
        circle_nodes = [node for node in G.nodes if G.nodes[node].get('shape', 'circle') == 'circle']
        box_nodes = [node for node in G.nodes if G.nodes[node].get('shape', 'circle') == 'box']
        
        if circle_nodes:
            circle_colors = [G.nodes[node].get('color', '#FFFFFF') for node in circle_nodes]
            nx.draw_networkx_nodes(G, pos, nodelist=circle_nodes, node_color=circle_colors,
                                 node_shape='o', node_size=800, alpha=0.8)
        
        if box_nodes:
            box_colors = [G.nodes[node].get('color', '#F5F5F5') for node in box_nodes]
            nx.draw_networkx_nodes(G, pos, nodelist=box_nodes, node_color=box_colors,
                                 node_shape='s', node_size=1200, alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.6, edge_color='gray', width=1)
        
        # Draw labels
        labels = {node: G.nodes[node].get('label', node)[:15] + ('...' if len(G.nodes[node].get('label', node)) > 15 else '') 
                 for node in G.nodes}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return None
        else:
            # Return as base64 encoded image
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            return image_base64
    
    def visualize_tensor_signature_space(self, fragments: List, 
                                       title: str = "Tensor Signature Space") -> Optional[str]:
        """
        Visualize tensor fragments in the 5D signature space using dimensionality reduction.
        
        Args:
            fragments: List of TensorFragments to visualize
            title: Title for the visualization
            
        Returns:
            Base64 encoded image
        """
        if len(fragments) < 2:
            print("Need at least 2 fragments for visualization")
            return None
        
        # Extract signatures as matrix
        signatures = np.array([f.signature.to_tensor() for f in fragments])
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Dimension names
        dims = ['Modality', 'Depth', 'Context', 'Salience', 'Autonomy']
        
        # Plot pairwise projections
        plot_idx = 0
        for i in range(5):
            for j in range(i + 1, 5):
                if plot_idx >= 6:
                    break
                
                ax = axes[plot_idx // 3, plot_idx % 3]
                
                # Scatter plot
                ax.scatter(signatures[:, i], signatures[:, j], 
                          alpha=0.7, s=60, c=range(len(signatures)), cmap='viridis')
                
                ax.set_xlabel(dims[i])
                ax.set_ylabel(dims[j])
                ax.set_title(f"{dims[i]} vs {dims[j]}")
                ax.grid(True, alpha=0.3)
                
                # Add fragment labels
                for k, fragment in enumerate(fragments):
                    if hasattr(fragment, 'content'):
                        label = str(fragment.content)[:10]
                        ax.annotate(label, (signatures[k, i], signatures[k, j]), 
                                  xytext=(5, 5), textcoords='offset points', 
                                  fontsize=8, alpha=0.7)
                
                plot_idx += 1
        
        # Remove empty subplots
        while plot_idx < 6:
            axes[plot_idx // 3, plot_idx % 3].remove()
            plot_idx += 1
        
        plt.tight_layout()
        
        # Return as base64 encoded image
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        return image_base64
    
    def create_verification_report(self, fragments: List, patterns: List[str] = None) -> Dict[str, Any]:
        """
        Create a comprehensive verification report.
        
        Args:
            fragments: List of TensorFragments to verify
            patterns: List of pattern handles to verify
            
        Returns:
            Comprehensive verification report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {},
            'fragment_verifications': [],
            'pattern_verifications': [],
            'visualizations': {}
        }
        
        # Verify fragments
        valid_fragments = 0
        total_errors = 0
        total_warnings = 0
        
        for i, fragment in enumerate(fragments):
            result = self.verify_tensor_fragment(fragment)
            result['fragment_id'] = i
            report['fragment_verifications'].append(result)
            
            if result['valid']:
                valid_fragments += 1
            total_errors += len(result['errors'])
            total_warnings += len(result['warnings'])
        
        # Verify patterns
        valid_patterns = 0
        if patterns and self.tensor_architecture:
            for pattern_handle in patterns:
                result = self.verify_cognitive_pattern(pattern_handle)
                result['pattern_handle'] = pattern_handle
                report['pattern_verifications'].append(result)
                
                if result['valid']:
                    valid_patterns += 1
                total_errors += len(result['errors'])
                total_warnings += len(result['warnings'])
        
        # Create visualizations
        if len(fragments) >= 2:
            try:
                sig_vis = self.visualize_tensor_signature_space(fragments)
                if sig_vis:
                    report['visualizations']['signature_space'] = sig_vis
            except Exception as e:
                print(f"Signature space visualization failed: {e}")
        
        if self.atomspace and len(self.atomspace.atoms) > 0:
            try:
                hypergraph_vis = self.visualize_hypergraph()
                if hypergraph_vis:
                    report['visualizations']['hypergraph'] = hypergraph_vis
            except Exception as e:
                print(f"Hypergraph visualization failed: {e}")
        
        # Summary
        report['summary'] = {
            'total_fragments': len(fragments),
            'valid_fragments': valid_fragments,
            'total_patterns': len(patterns) if patterns else 0,
            'valid_patterns': valid_patterns,
            'total_errors': total_errors,
            'total_warnings': total_warnings,
            'health_status': 'healthy' if total_errors == 0 else 'error' if total_errors > 5 else 'warning'
        }
        
        return report


class CognitiveDashboard:
    """
    Dashboard for monitoring cognitive primitive health and performance.
    """
    
    def __init__(self, verifier: CognitivePrimitiveVerifier):
        """
        Initialize the dashboard.
        
        Args:
            verifier: CognitivePrimitiveVerifier instance
        """
        self.verifier = verifier
        self.metrics_history = []
    
    def collect_metrics(self) -> Dict[str, Any]:
        """
        Collect current system metrics.
        
        Returns:
            Current metrics dictionary
        """
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'atomspace_metrics': {},
            'tensor_metrics': {},
            'performance_metrics': {}
        }
        
        # AtomSpace metrics
        if self.verifier.atomspace:
            metrics['atomspace_metrics'] = {
                'atom_count': len(self.verifier.atomspace.atoms),
                'relationship_count': len(self.verifier.atomspace.relationships),
                'type_diversity': len(self.verifier.atomspace.types),
                'average_connectivity': self._compute_average_connectivity()
            }
        
        # Tensor architecture metrics
        if self.verifier.tensor_architecture:
            metrics['tensor_metrics'] = {
                'fragment_count': len(self.verifier.tensor_architecture.fragments),
                'encoder_count': len(self.verifier.tensor_architecture.ml_primitive_encoders),
                'decoder_count': len(self.verifier.tensor_architecture.hypergraph_decoders)
            }
        
        self.metrics_history.append(metrics)
        return metrics
    
    def _compute_average_connectivity(self) -> float:
        """Compute average connectivity of atoms in the hypergraph."""
        if not self.verifier.atomspace:
            return 0.0
        
        total_connections = 0
        atom_count = len(self.verifier.atomspace.atoms)
        
        for atom_handle in self.verifier.atomspace.atoms:
            incoming = self.verifier.atomspace.get_incoming_set(atom_handle)
            total_connections += len(incoming)
        
        return total_connections / atom_count if atom_count > 0 else 0.0
    
    def generate_dashboard_html(self) -> str:
        """
        Generate an HTML dashboard for cognitive primitive monitoring.
        
        Returns:
            HTML string for the dashboard
        """
        latest_metrics = self.metrics_history[-1] if self.metrics_history else {}
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Cognitive Primitives Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .header {{ background: #2196F3; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
                .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
                .metric-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .metric-title {{ font-size: 18px; font-weight: bold; margin-bottom: 15px; color: #333; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2196F3; }}
                .metric-label {{ font-size: 14px; color: #666; margin-top: 5px; }}
                .status-good {{ color: #4CAF50; }}
                .status-warning {{ color: #FF9800; }}
                .status-error {{ color: #F44336; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🧠 Cognitive Primitives Dashboard</h1>
                    <p>Real-time monitoring of tensor fragments and hypergraph patterns</p>
                    <p>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-title">AtomSpace Status</div>
                        <div class="metric-value">{latest_metrics.get('atomspace_metrics', {}).get('atom_count', 0)}</div>
                        <div class="metric-label">Total Atoms</div>
                        <div class="metric-value">{latest_metrics.get('atomspace_metrics', {}).get('relationship_count', 0)}</div>
                        <div class="metric-label">Relationships</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-title">Tensor Architecture</div>
                        <div class="metric-value">{latest_metrics.get('tensor_metrics', {}).get('fragment_count', 0)}</div>
                        <div class="metric-label">Active Fragments</div>
                        <div class="metric-value">{latest_metrics.get('tensor_metrics', {}).get('encoder_count', 0)}</div>
                        <div class="metric-label">ML Encoders</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-title">Connectivity</div>
                        <div class="metric-value">{latest_metrics.get('atomspace_metrics', {}).get('average_connectivity', 0.0):.2f}</div>
                        <div class="metric-label">Avg Connections</div>
                        <div class="metric-value">{latest_metrics.get('atomspace_metrics', {}).get('type_diversity', 0)}</div>
                        <div class="metric-label">Type Diversity</div>
                    </div>
                </div>
                
                <div class="metric-card" style="margin-top: 20px;">
                    <div class="metric-title">System Health</div>
                    <div id="health-status">Monitoring...</div>
                </div>
            </div>
            
            <script>
                // Simple health status indicator
                const healthElement = document.getElementById('health-status');
                const atomCount = {latest_metrics.get('atomspace_metrics', {}).get('atom_count', 0)};
                const fragmentCount = {latest_metrics.get('tensor_metrics', {}).get('fragment_count', 0)};
                
                if (atomCount > 0 && fragmentCount > 0) {{
                    healthElement.innerHTML = '<span class="status-good">✅ System Healthy</span>';
                }} else if (atomCount > 0 || fragmentCount > 0) {{
                    healthElement.innerHTML = '<span class="status-warning">⚠️ Partial Functionality</span>';
                }} else {{
                    healthElement.innerHTML = '<span class="status-error">❌ System Not Initialized</span>';
                }}
            </script>
        </body>
        </html>
        """
        
        return html
    
    def export_metrics_json(self) -> str:
        """
        Export metrics history as JSON.
        
        Returns:
            JSON string with metrics history
        """
        return json.dumps(self.metrics_history, indent=2)


def create_verification_test_suite():
    """
    Create a test suite for verifying cognitive primitives functionality.
    
    Returns:
        Dictionary with test functions
    """
    
    def test_tensor_signature_encoding():
        """Test tensor signature encoding and decoding."""
        from utils.opencog.tensor_fragments import TensorSignature
        
        # Create test signature
        sig = TensorSignature(0.8, 0.6, 0.7, 0.9, 0.5)
        
        # Convert to tensor and back
        tensor = sig.to_tensor()
        decoded_sig = TensorSignature.from_tensor(tensor)
        
        # Verify roundtrip
        original_tensor = sig.to_tensor()
        decoded_tensor = decoded_sig.to_tensor()
        
        diff = np.abs(original_tensor - decoded_tensor)
        max_diff = np.max(diff)
        
        return {
            'test_name': 'tensor_signature_encoding',
            'passed': max_diff < 1e-10,
            'details': f"Max difference: {max_diff}",
            'original': original_tensor.tolist(),
            'decoded': decoded_tensor.tolist()
        }
    
    def test_hypergraph_encoding():
        """Test hypergraph encoding of tensor fragments."""
        from utils.opencog.tensor_fragments import TensorSignature, TensorFragment
        from utils.opencog.atomspace import AtomSpace
        
        # Create test components
        atomspace = AtomSpace()
        sig = TensorSignature(0.9, 0.7, 0.8, 0.6, 0.5)
        fragment = TensorFragment(sig, "test_concept")
        
        # Test encoding
        try:
            handle = fragment.encode_to_hypergraph(atomspace)
            decoded = TensorFragment.decode_from_hypergraph(atomspace, handle)
            
            success = decoded is not None
            if success:
                # Check content preservation
                content_match = decoded.content == fragment.content
                
                # Check signature preservation (allow small floating point differences)
                sig_diff = np.max(np.abs(fragment.signature.to_tensor() - decoded.signature.to_tensor()))
                sig_match = sig_diff < 0.1
                
                success = content_match and sig_match
            
            return {
                'test_name': 'hypergraph_encoding',
                'passed': success,
                'details': f"Atoms created: {len(atomspace.atoms)}, Relationships: {len(atomspace.relationships)}",
                'content_match': success and decoded.content == fragment.content,
                'signature_diff': float(sig_diff) if success else None
            }
            
        except Exception as e:
            return {
                'test_name': 'hypergraph_encoding',
                'passed': False,
                'details': f"Exception: {str(e)}"
            }
    
    def test_cognitive_grammar_parsing():
        """Test cognitive grammar parsing."""
        from utils.opencog.cognitive_grammar import SchemeParser
        
        parser = SchemeParser()
        
        test_expressions = [
            '(define concept_learning "ML primitive")',
            '(eval (+ 1 2 3))',
            '(lambda (x y) (+ x y))',
            '(if (> x 0) "positive" "non-positive")'
        ]
        
        results = []
        for expr in test_expressions:
            try:
                parsed = parser.parse(expr)
                results.append({
                    'expression': expr,
                    'parsed_successfully': True,
                    'type': parsed.type.value,
                    'children_count': len(parsed.children)
                })
            except Exception as e:
                results.append({
                    'expression': expr,
                    'parsed_successfully': False,
                    'error': str(e)
                })
        
        passed = all(r['parsed_successfully'] for r in results)
        
        return {
            'test_name': 'cognitive_grammar_parsing',
            'passed': passed,
            'details': f"Parsed {len([r for r in results if r['parsed_successfully']])}/{len(results)} expressions",
            'results': results
        }
    
    return {
        'tensor_signature_encoding': test_tensor_signature_encoding,
        'hypergraph_encoding': test_hypergraph_encoding,
        'cognitive_grammar_parsing': test_cognitive_grammar_parsing
    }


def run_verification_suite(atomspace=None, tensor_architecture=None) -> Dict[str, Any]:
    """
    Run the complete verification suite for cognitive primitives.
    
    Args:
        atomspace: Optional AtomSpace instance
        tensor_architecture: Optional TensorFragmentArchitecture instance
        
    Returns:
        Complete verification results
    """
    print("🧪 Running Cognitive Primitives Verification Suite")
    print("=" * 60)
    
    test_suite = create_verification_test_suite()
    results = {}
    
    for test_name, test_func in test_suite.items():
        print(f"Running {test_name}...")
        try:
            result = test_func()
            results[test_name] = result
            
            if result['passed']:
                print(f"  ✅ {test_name}: PASSED")
            else:
                print(f"  ❌ {test_name}: FAILED - {result.get('details', 'Unknown error')}")
                
        except Exception as e:
            results[test_name] = {
                'test_name': test_name,
                'passed': False,
                'details': f"Test execution error: {str(e)}"
            }
            print(f"  💥 {test_name}: ERROR - {str(e)}")
    
    # Summary
    passed_tests = sum(1 for r in results.values() if r['passed'])
    total_tests = len(results)
    
    print("\n📊 Verification Summary:")
    print("-" * 30)
    print(f"Tests passed: {passed_tests}/{total_tests}")
    print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Cognitive primitives are functioning correctly.")
    elif passed_tests > 0:
        print("⚠️ Some tests failed. Please review the failed tests above.")
    else:
        print("❌ All tests failed. Please check the implementation.")
    
    return {
        'summary': {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': passed_tests/total_tests,
            'status': 'healthy' if passed_tests == total_tests else 'degraded'
        },
        'test_results': results,
        'timestamp': datetime.now().isoformat()
    }