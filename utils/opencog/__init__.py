"""
OpenCog components implementation for Archon.
"""

# Import the custom implementations
from utils.opencog.atomspace import AtomSpace
from utils.opencog.cogserver import CogServer
from utils.opencog.utilities import Utilities

# Define the opencog namespace
class opencog:
    """OpenCog namespace for component organization."""
    
    # Define submodules
    atomspace = None
    cogserver = None
    utilities = None
    
# Attach the component classes to the opencog namespace
opencog.atomspace = AtomSpace
opencog.cogserver = CogServer
opencog.utilities = Utilities
