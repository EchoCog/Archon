"""
AtomSpace implementation for OpenCog integration.

This module provides a custom implementation of the AtomSpace component
for data representation and storage as specified in PLAN.md.
"""

class AtomSpace:
    """
    AtomSpace is OpenCog's data representation and storage component.
    It helps in managing complex data structures and relationships.
    """
    
    def __init__(self):
        """Initialize a new AtomSpace instance."""
        self.atoms = {}
        self.relationships = {}
        self.types = set()
    
    def add_node(self, type_name, name, value=None):
        """
        Add a node to the AtomSpace.
        
        Args:
            type_name: The type of the node
            name: The name identifier for the node
            value: Optional value associated with the node
            
        Returns:
            The node handle
        """
        node_id = f"{type_name}:{name}"
        if node_id not in self.atoms:
            self.atoms[node_id] = {"type": type_name, "name": name, "value": value}
            self.types.add(type_name)
        return node_id
    
    def add_link(self, type_name, outgoing_set):
        """
        Add a link between nodes in the AtomSpace.
        
        Args:
            type_name: The type of the link
            outgoing_set: List of node handles that this link connects
            
        Returns:
            The link handle
        """
        link_id = f"{type_name}:{','.join(outgoing_set)}"
        if link_id not in self.relationships:
            self.relationships[link_id] = {
                "type": type_name,
                "outgoing_set": outgoing_set
            }
            self.types.add(type_name)
        return link_id
    
    def get_atom(self, handle):
        """
        Get an atom by its handle.
        
        Args:
            handle: The handle of the atom
            
        Returns:
            The atom data or None if not found
        """
        return self.atoms.get(handle)
    
    def get_outgoing_set(self, link_handle):
        """
        Get the outgoing set of a link.
        
        Args:
            link_handle: The handle of the link
            
        Returns:
            List of node handles or None if not found
        """
        link = self.relationships.get(link_handle)
        if link:
            return link["outgoing_set"]
        return None
    
    def get_atoms_by_type(self, type_name):
        """
        Get all atoms of a specific type.
        
        Args:
            type_name: The type to filter by
            
        Returns:
            List of atom handles of the specified type
        """
        return [handle for handle, atom in self.atoms.items() 
                if atom["type"] == type_name]
    
    def get_links_by_type(self, type_name):
        """
        Get all links of a specific type.
        
        Args:
            type_name: The type to filter by
            
        Returns:
            List of link handles of the specified type
        """
        return [handle for handle, link in self.relationships.items() 
                if link["type"] == type_name]
    
    def get_incoming_set(self, node_handle):
        """
        Get all links that contain this node in their outgoing set.
        
        Args:
            node_handle: The handle of the node
            
        Returns:
            List of link handles that reference this node
        """
        incoming = []
        for handle, link in self.relationships.items():
            if node_handle in link["outgoing_set"]:
                incoming.append(handle)
        return incoming
