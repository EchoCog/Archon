#!/usr/bin/env python
"""
OpenCog Components Demo

This script demonstrates the basic functionality of our custom OpenCog components,
showing how they can be used for knowledge representation and reasoning.
"""

import os
import sys

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.opencog import opencog

def demo_atomspace():
    """Demonstrate AtomSpace functionality for knowledge representation."""
    print("\n=== AtomSpace Demo ===")
    
    # Initialize an AtomSpace
    atomspace = opencog.atomspace.AtomSpace()
    
    # Create some nodes representing concepts
    print("Creating nodes representing concepts...")
    person = atomspace.add_node("ConceptNode", "Person")
    alice = atomspace.add_node("ConceptNode", "Alice")
    bob = atomspace.add_node("ConceptNode", "Bob")
    programmer = atomspace.add_node("ConceptNode", "Programmer")
    manager = atomspace.add_node("ConceptNode", "Manager")
    
    # Create relationships between nodes
    print("Creating relationships between nodes...")
    atomspace.add_link("InheritanceLink", [alice, person])
    atomspace.add_link("InheritanceLink", [bob, person])
    atomspace.add_link("InheritanceLink", [alice, programmer])
    atomspace.add_link("InheritanceLink", [bob, manager])
    
    # Demonstrate querying the AtomSpace
    print("\nQuerying the AtomSpace...")
    alice_node = atomspace.get_atom(alice)
    print(f"Alice node: {alice_node}")
    
    # Find all Person instances
    print("\nFinding all Person instances...")
    person_links = atomspace.get_incoming_set(person)
    for link in person_links:
        outgoing = atomspace.get_outgoing_set(link)
        if outgoing and len(outgoing) >= 1:
            instance = atomspace.get_atom(outgoing[0])
            print(f"Person instance: {instance['name']}")
    
    # Find all nodes of a specific type
    print("\nFinding all ConceptNodes...")
    concept_nodes = atomspace.get_atoms_by_type("ConceptNode")
    print(f"ConceptNodes: {len(concept_nodes)}")
    for node in concept_nodes:
        node_data = atomspace.get_atom(node)
        print(f"  {node_data['name']}")

def demo_cogserver():
    """Demonstrate CogServer functionality for agent execution."""
    print("\n=== CogServer Demo ===")
    
    # Initialize an AtomSpace and CogServer
    atomspace = opencog.atomspace.AtomSpace()
    cogserver = opencog.cogserver.CogServer(atomspace)
    
    # Create a simple agent that adds facts to the AtomSpace
    def knowledge_agent(atomspace):
        print("Knowledge Agent: Adding facts to the AtomSpace...")
        
        # Adding facts about programming languages
        python = atomspace.add_node("ConceptNode", "Python")
        language = atomspace.add_node("ConceptNode", "ProgrammingLanguage")
        atomspace.add_link("InheritanceLink", [python, language])
        
        dynamically_typed = atomspace.add_node("ConceptNode", "DynamicallyTyped")
        atomspace.add_link("EvaluationLink", [dynamically_typed, python])
        
        # Agent output
        print("Knowledge Agent: Facts added successfully")
        return True
    
    # Create a simple agent that queries the AtomSpace
    def query_agent(atomspace):
        print("Query Agent: Searching for programming languages...")
        
        # Find programming languages
        language = None
        for node in atomspace.atoms.values():
            if node["type"] == "ConceptNode" and node["name"] == "ProgrammingLanguage":
                language = node
                break
        
        if language:
            # Find instances of programming languages
            programming_languages = []
            
            for link in atomspace.relationships.values():
                if link["type"] == "InheritanceLink":
                    outgoing = link["outgoing_set"]
                    if len(outgoing) == 2:
                        # Check if second element is the ProgrammingLanguage concept
                        target = outgoing[1]
                        target_node = atomspace.get_atom(target)
                        
                        if target_node and target_node["name"] == "ProgrammingLanguage":
                            # First element is a programming language
                            lang = atomspace.get_atom(outgoing[0])
                            if lang:
                                programming_languages.append(lang["name"])
            
            print(f"Query Agent: Found {len(programming_languages)} programming languages:")
            for lang in programming_languages:
                print(f"  - {lang}")
        else:
            print("Query Agent: No programming language concept found")
        
        return True
    
    # Register the agents
    print("Registering agents...")
    cogserver.register_agent("knowledge_agent", knowledge_agent, priority=1)
    cogserver.register_agent("query_agent", query_agent, priority=0)
    
    # Start the CogServer
    print("Starting the CogServer...")
    cogserver.start()
    
    try:
        # Start the agents
        print("Starting agents...")
        cogserver.start_agent("knowledge_agent")
        
        # Wait a moment for the knowledge agent to execute
        import time
        time.sleep(0.5)
        
        # Then start the query agent
        cogserver.start_agent("query_agent")
        
        # Wait for the query agent to execute
        time.sleep(0.5)
        
        # Show agent status
        print("\nAgent status:")
        for name, status in cogserver.get_agent_status().items():
            print(f"  {name}: {status}")
    
    finally:
        # Stop the CogServer
        print("Stopping the CogServer...")
        cogserver.stop()

def demo_utilities():
    """Demonstrate Utilities functionality for advanced reasoning."""
    print("\n=== Utilities Demo ===")
    
    # Initialize AtomSpace and Utilities
    atomspace = opencog.atomspace.AtomSpace()
    utilities = opencog.utilities.Utilities(atomspace)
    
    # Create a simple knowledge graph using the utilities
    print("Creating a knowledge graph...")
    triples = [
        ("Python", "is_a", "ProgrammingLanguage"),
        ("Python", "has_feature", "DynamicTyping"),
        ("Python", "has_feature", "GarbageCollection"),
        ("Java", "is_a", "ProgrammingLanguage"),
        ("Java", "has_feature", "StaticTyping"),
        ("Java", "has_feature", "GarbageCollection"),
        ("C++", "is_a", "ProgrammingLanguage"),
        ("C++", "has_feature", "StaticTyping"),
        ("C++", "has_feature", "ManualMemoryManagement")
    ]
    
    utilities.create_knowledge_graph(triples)
    
    # Extract and display the knowledge graph
    print("\nExtracted knowledge graph:")
    extracted_triples = utilities.extract_knowledge_graph()
    for subj, pred, obj in extracted_triples:
        print(f"  {subj} -- {pred} --> {obj}")
    
    # Demonstrate pattern matching using a simple reasoner
    print("\nDemonstrating pattern matching with a simple reasoner...")
    
    def language_feature_reasoner(atomspace, feature_name):
        """Find all languages that have a specific feature"""
        results = []
        
        # Get all triples from the knowledge graph
        triples = utilities.extract_knowledge_graph()
        
        # Filter triples to find languages with the specified feature
        for subj, pred, obj in triples:
            if pred == "has_feature" and obj == feature_name:
                results.append(subj)
                
        return results
    
    # Register the reasoner
    utilities.register_reasoner("language_feature", language_feature_reasoner)
    
    # Apply the reasoner to find languages with garbage collection
    gc_languages = utilities.apply_reasoner("language_feature", "GarbageCollection")
    print(f"Languages with garbage collection: {', '.join(gc_languages)}")
    
    # Apply the reasoner to find languages with static typing
    static_languages = utilities.apply_reasoner("language_feature", "StaticTyping")
    print(f"Languages with static typing: {', '.join(static_languages)}")
    
    # Demonstrate Atomese expression parsing
    print("\nDemonstrating Atomese expression parsing...")
    
    # Create a simple Atomese expression
    expression = "(EvaluationLink (PredicateNode \"writes\") (ListLink (ConceptNode \"Programmer\") (ConceptNode \"Code\")))"
    result = utilities.create_atomese_expression(expression)
    
    print(f"Created Atomese expression, resulting in handle: {result}")
    
    # Use query functionality
    print("\nDemonstrating query functionality...")
    programming_languages = utilities.query_atoms(type_filter="ConceptNode")
    print("ConceptNodes found:")
    for handle in programming_languages:
        node = atomspace.get_atom(handle)
        if node:
            print(f"  {node['name']}")

def main():
    """Main demo function."""
    print("OpenCog Components Demo")
    print("======================")
    
    demo_atomspace()
    demo_cogserver()
    demo_utilities()
    
    print("\nDemo complete!")

if __name__ == "__main__":
    main()
