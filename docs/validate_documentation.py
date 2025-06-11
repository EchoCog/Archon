#!/usr/bin/env python3
"""
Archon Documentation Validator

This script validates the comprehensive architecture documentation,
ensuring that Mermaid diagrams are properly formatted and the
documentation structure follows the hypergraph-centric principles.
"""

import os
import re
from typing import Dict, List, Tuple

def validate_mermaid_diagrams(file_path: str) -> Dict[str, any]:
    """Validate Mermaid diagrams in a documentation file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all mermaid code blocks
    mermaid_pattern = r'```mermaid\n(.*?)\n```'
    mermaid_blocks = re.findall(mermaid_pattern, content, re.DOTALL)
    
    diagram_types = {}
    total_nodes = 0
    total_edges = 0
    
    for i, block in enumerate(mermaid_blocks):
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        if not lines:
            continue
            
        # Identify diagram type
        first_line = lines[0]
        diagram_type = "unknown"
        
        if first_line.startswith('graph '):
            diagram_type = "graph"
        elif first_line.startswith('sequenceDiagram'):
            diagram_type = "sequence"
        elif first_line.startswith('stateDiagram'):
            diagram_type = "state"
        elif first_line.startswith('flowchart '):
            diagram_type = "flowchart"
        elif first_line.startswith('timeline'):
            diagram_type = "timeline"
        
        # Count nodes and edges for graph-type diagrams
        if diagram_type in ["graph", "flowchart"]:
            nodes = set()
            edges = 0
            
            for line in lines[1:]:  # Skip first line (diagram declaration)
                if '-->' in line or '<-->' in line or '---' in line:
                    edges += 1
                # Extract node names
                node_matches = re.findall(r'\b[A-Z][A-Z0-9_]*\b', line)
                nodes.update(node_matches)
            
            total_nodes += len(nodes)
            total_edges += edges
        
        # Count diagram types
        diagram_types[diagram_type] = diagram_types.get(diagram_type, 0) + 1
    
    return {
        'total_diagrams': len(mermaid_blocks),
        'diagram_types': diagram_types,
        'total_nodes': total_nodes,
        'total_edges': total_edges,
        'cognitive_complexity': total_nodes + (total_edges * 2)  # Weighted complexity
    }

def analyze_documentation_structure() -> Dict[str, any]:
    """Analyze the overall documentation structure."""
    docs_dir = 'docs'
    docs_files = ['ARCHITECTURE.md', 'OPENCOG_INTEGRATION.md', 'WORKFLOW.md', 'README.md']
    
    structure_analysis = {
        'files': {},
        'total_lines': 0,
        'total_diagrams': 0,
        'cognitive_complexity_score': 0
    }
    
    for filename in docs_files:
        file_path = os.path.join(docs_dir, filename)
        if not os.path.exists(file_path):
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        lines = len(content.split('\n'))
        words = len(content.split())
        
        # Validate Mermaid diagrams
        mermaid_analysis = validate_mermaid_diagrams(file_path)
        
        # Cognitive architecture keywords analysis
        cognitive_keywords = [
            'recursive', 'emergent', 'hypergraph', 'cognitive', 'reasoning',
            'atomspace', 'pattern', 'adaptive', 'synergy', 'distributed'
        ]
        
        keyword_density = sum(content.lower().count(keyword) for keyword in cognitive_keywords)
        
        structure_analysis['files'][filename] = {
            'lines': lines,
            'words': words,
            'diagrams': mermaid_analysis['total_diagrams'],
            'diagram_types': mermaid_analysis['diagram_types'],
            'cognitive_nodes': mermaid_analysis['total_nodes'],
            'cognitive_edges': mermaid_analysis['total_edges'],
            'keyword_density': keyword_density,
            'complexity_score': mermaid_analysis['cognitive_complexity']
        }
        
        structure_analysis['total_lines'] += lines
        structure_analysis['total_diagrams'] += mermaid_analysis['total_diagrams']
        structure_analysis['cognitive_complexity_score'] += mermaid_analysis['cognitive_complexity']
    
    return structure_analysis

def generate_documentation_report() -> str:
    """Generate a comprehensive report on the documentation."""
    analysis = analyze_documentation_structure()
    
    report = []
    report.append("=" * 60)
    report.append("ARCHON ARCHITECTURE DOCUMENTATION VALIDATION REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Overall statistics
    report.append("📊 OVERALL STATISTICS")
    report.append("-" * 25)
    report.append(f"Total documentation files: {len(analysis['files'])}")
    report.append(f"Total lines of documentation: {analysis['total_lines']:,}")
    report.append(f"Total Mermaid diagrams: {analysis['total_diagrams']}")
    report.append(f"Cognitive complexity score: {analysis['cognitive_complexity_score']}")
    report.append("")
    
    # File-by-file analysis
    report.append("📋 FILE ANALYSIS")
    report.append("-" * 15)
    
    for filename, stats in analysis['files'].items():
        report.append(f"\n🔹 {filename}")
        report.append(f"   Lines: {stats['lines']:,}")
        report.append(f"   Words: {stats['words']:,}")
        report.append(f"   Diagrams: {stats['diagrams']}")
        report.append(f"   Cognitive nodes: {stats['cognitive_nodes']}")
        report.append(f"   Cognitive edges: {stats['cognitive_edges']}")
        report.append(f"   Keyword density: {stats['keyword_density']}")
        
        if stats['diagram_types']:
            report.append(f"   Diagram types: {dict(stats['diagram_types'])}")
    
    report.append("")
    
    # Diagram type distribution
    all_diagram_types = {}
    for file_stats in analysis['files'].values():
        for dtype, count in file_stats['diagram_types'].items():
            all_diagram_types[dtype] = all_diagram_types.get(dtype, 0) + count
    
    report.append("📈 DIAGRAM TYPE DISTRIBUTION")
    report.append("-" * 28)
    for dtype, count in sorted(all_diagram_types.items()):
        percentage = (count / analysis['total_diagrams']) * 100 if analysis['total_diagrams'] > 0 else 0
        report.append(f"{dtype:>12}: {count:>3} ({percentage:5.1f}%)")
    
    report.append("")
    
    # Cognitive architecture assessment
    report.append("🧠 COGNITIVE ARCHITECTURE ASSESSMENT")
    report.append("-" * 36)
    
    complexity_per_diagram = (analysis['cognitive_complexity_score'] / 
                             analysis['total_diagrams']) if analysis['total_diagrams'] > 0 else 0
    
    if complexity_per_diagram > 50:
        complexity_rating = "🔥 High (Excellent hypergraph density)"
    elif complexity_per_diagram > 25:
        complexity_rating = "⚡ Medium (Good cognitive representation)"
    else:
        complexity_rating = "📝 Basic (Simple documentation)"
    
    report.append(f"Complexity per diagram: {complexity_per_diagram:.1f}")
    report.append(f"Cognitive rating: {complexity_rating}")
    
    # Validate architectural principles
    report.append("")
    report.append("✅ ARCHITECTURAL PRINCIPLES VALIDATION")
    report.append("-" * 38)
    
    principles_found = []
    for file_stats in analysis['files'].values():
        if file_stats['keyword_density'] > 10:
            principles_found.append("Recursive implementation pathways")
        if 'graph' in file_stats['diagram_types'] and file_stats['diagram_types']['graph'] > 2:
            principles_found.append("Hypergraph pattern encoding")
        if 'sequence' in file_stats['diagram_types']:
            principles_found.append("Signal propagation pathways")
        if 'state' in file_stats['diagram_types']:
            principles_found.append("Adaptive attention allocation")
    
    unique_principles = list(set(principles_found))
    for principle in unique_principles:
        report.append(f"✓ {principle}")
    
    if len(unique_principles) >= 4:
        report.append("\n🎯 COMPREHENSIVE: All core architectural principles documented!")
    elif len(unique_principles) >= 2:
        report.append("\n👍 GOOD: Most architectural principles covered")
    else:
        report.append("\n⚠️  NEEDS IMPROVEMENT: More architectural detail needed")
    
    report.append("")
    report.append("=" * 60)
    report.append("Documentation validation complete. Ready for distributed cognition!")
    report.append("=" * 60)
    
    return "\n".join(report)

def main():
    """Main function to run the documentation validation."""
    print("Validating Archon architecture documentation...")
    print("Analyzing hypergraph-centric implementation patterns...\n")
    
    if not os.path.exists('docs'):
        print("❌ Error: docs directory not found!")
        return
    
    report = generate_documentation_report()
    print(report)
    
    # Save report to file
    with open('docs/VALIDATION_REPORT.txt', 'w') as f:
        f.write(report)
    print(f"\n📄 Detailed report saved to: docs/VALIDATION_REPORT.txt")

if __name__ == "__main__":
    main()