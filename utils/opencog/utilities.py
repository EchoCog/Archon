"""
Utilities implementation for OpenCog integration.

This module provides a custom implementation of the Utilities component
for optimizing data queries and enhancing reasoning capabilities as specified in PLAN.md.
"""

from typing import List, Dict, Any, Optional, Callable

class Utilities:
    """
    Utilities provides helper functions for working with AtomSpace data,
    optimizing data queries, and implementing advanced reasoning capabilities.
    """
    
    def __init__(self, atomspace=None):
        """
        Initialize a new Utilities instance.
        
        Args:
            atomspace: The AtomSpace instance to associate with these utilities
        """
        self.atomspace = atomspace
        self._registered_reasoners = {}
        self._pattern_matchers = {}
    
    def query_atoms(self, type_filter=None, name_filter=None, value_filter=None):
        """
        Query atoms based on filters.
        
        Args:
            type_filter: Filter by atom type
            name_filter: Filter by atom name
            value_filter: Filter by atom value
            
        Returns:
            List of matching atom handles
        """
        if not self.atomspace:
            return []
        
        results = []
        for handle, atom in self.atomspace.atoms.items():
            match = True
            
            if type_filter and atom["type"] != type_filter:
                match = False
            
            if name_filter and atom["name"] != name_filter:
                match = False
            
            if value_filter and atom["value"] != value_filter:
                match = False
            
            if match:
                results.append(handle)
        
        return results
    
    def register_reasoner(self, name: str, reasoning_function: Callable):
        """
        Register a reasoner component.
        
        Args:
            name: The name of the reasoner
            reasoning_function: The function implementing the reasoning logic
        """
        self._registered_reasoners[name] = reasoning_function
    
    def apply_reasoner(self, reasoner_name: str, *args, **kwargs):
        """
        Apply a registered reasoner to the AtomSpace.
        
        Args:
            reasoner_name: The name of the reasoner to apply
            *args, **kwargs: Arguments to pass to the reasoner
            
        Returns:
            The result of the reasoner application
        """
        if reasoner_name not in self._registered_reasoners:
            raise ValueError(f"Reasoner '{reasoner_name}' not found")
        
        reasoner = self._registered_reasoners[reasoner_name]
        return reasoner(self.atomspace, *args, **kwargs)
    
    def register_pattern_matcher(self, name: str, pattern_matcher_function: Callable):
        """
        Register a pattern matcher.
        
        Args:
            name: The name of the pattern matcher
            pattern_matcher_function: The function implementing the pattern matching logic
        """
        self._pattern_matchers[name] = pattern_matcher_function
    
    def match_pattern(self, matcher_name: str, pattern: Any, *args, **kwargs):
        """
        Apply a registered pattern matcher to find patterns in the AtomSpace.
        
        Args:
            matcher_name: The name of the pattern matcher to apply
            pattern: The pattern to match
            *args, **kwargs: Additional arguments for the matcher
            
        Returns:
            The result of the pattern matching
        """
        if matcher_name not in self._pattern_matchers:
            raise ValueError(f"Pattern matcher '{matcher_name}' not found")
        
        matcher = self._pattern_matchers[matcher_name]
        return matcher(self.atomspace, pattern, *args, **kwargs)
    
    def create_atomese_expression(self, expression_str: str):
        """
        Parse an Atomese expression string and create the corresponding atoms in the AtomSpace.
        
        Args:
            expression_str: The Atomese expression string
            
        Returns:
            The handle of the root atom of the expression
        """
        # A simple parser for basic Atomese expressions
        # This is a placeholder for a more sophisticated implementation
        if not expression_str:
            return None
            
        parts = expression_str.strip("()").split()
        if not parts:
            return None
            
        link_type = parts[0]
        
        # For simplicity, assume the rest are node references or nested expressions
        outgoing_set = []
        current_part = ""
        nesting = 0
        
        for part in parts[1:]:
            if "(" in part:
                nesting += part.count("(")
                current_part += " " + part if current_part else part
            elif ")" in part:
                nesting -= part.count(")")
                current_part += " " + part
                if nesting == 0 and current_part:
                    # Process nested expression
                    nested_handle = self.create_atomese_expression(current_part)
                    outgoing_set.append(nested_handle)
                    current_part = ""
            elif nesting > 0:
                current_part += " " + part
            else:
                # This is a simple node reference, assume ConceptNode
                node_handle = self.atomspace.add_node("ConceptNode", part)
                outgoing_set.append(node_handle)
        
        # Create the link
        if outgoing_set:
            return self.atomspace.add_link(link_type, outgoing_set)
        return None
    
    def evaluate_query(self, query: str):
        """
        Evaluate a query against the AtomSpace.
        
        Args:
            query: The query string in a simple query format
            
        Returns:
            The query results
        """
        # A simple query evaluator - would be more complex in a real implementation
        
        # Split query parts
        parts = query.split()
        if len(parts) < 3:
            return []
        
        query_type = parts[0].lower()
        
        if query_type == "get":
            entity_type = parts[1]
            condition_parts = parts[2:]
            
            # Simple condition parsing
            if len(condition_parts) >= 3 and condition_parts[0] == "where":
                attribute = condition_parts[1]
                op = condition_parts[2]
                value = " ".join(condition_parts[3:])
                
                # Remove quotes if present
                value = value.strip("'\"")
                
                # Handle different conditions
                if attribute == "type" and op == "=":
                    return self.atomspace.get_atoms_by_type(value)
                elif attribute == "name" and op == "=":
                    return self.query_atoms(name_filter=value)
                    
        return []
        
    def create_knowledge_graph(self, triples: List[tuple]):
        """
        Create a knowledge graph from a list of (subject, predicate, object) triples.
        
        Args:
            triples: List of (subject, predicate, object) triples
            
        Returns:
            List of created link handles
        """
        links = []
        for subj, pred, obj in triples:
            # Create nodes for subject and object
            subj_handle = self.atomspace.add_node("ConceptNode", subj)
            obj_handle = self.atomspace.add_node("ConceptNode", obj)
            
            # Create a link with the predicate type
            link_handle = self.atomspace.add_link(pred, [subj_handle, obj_handle])
            links.append(link_handle)
        
        return links
    
    def extract_knowledge_graph(self):
        """
        Extract a knowledge graph representation from the AtomSpace.
        
        Returns:
            List of (subject, predicate, object) triples
        """
        triples = []
        
        for link_handle, link in self.atomspace.relationships.items():
            pred = link["type"]
            outgoing = link["outgoing_set"]
            
            if len(outgoing) == 2:
                subj_handle = outgoing[0]
                obj_handle = outgoing[1]
                
                subj_atom = self.atomspace.get_atom(subj_handle)
                obj_atom = self.atomspace.get_atom(obj_handle)
                
                if subj_atom and obj_atom:
                    subj = subj_atom["name"]
                    obj = obj_atom["name"]
                    triples.append((subj, pred, obj))
        
        return triples
