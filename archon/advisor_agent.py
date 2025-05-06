from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import os
import sys
import json
from typing import List, Any
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import Client

# Add the parent directory to sys.path to allow importing from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import get_env_var
from archon.agent_prompts import advisor_prompt
from archon.agent_tools import get_file_content_tool

# Import custom OpenCog dependencies
from utils.opencog import opencog

load_dotenv()

provider = get_env_var('LLM_PROVIDER') or 'OpenAI'
llm = get_env_var('PRIMARY_MODEL') or 'gpt-4o-mini'
base_url = get_env_var('BASE_URL') or 'https://api.openai.com/v1'
api_key = get_env_var('LLM_API_KEY') or 'no-llm-api-key-provided'

model = AnthropicModel(llm, api_key=api_key) if provider == "Anthropic" else OpenAIModel(llm, base_url=base_url, api_key=api_key)

logfire.configure(send_to_logfire='if-token-present')

@dataclass
class AdvisorDeps:
    file_list: List[str]
    atomspace: Any = None
    cogserver: Any = None
    utilities: Any = None
    
    def __post_init__(self):
        # Initialize OpenCog components if they haven't been provided
        if self.atomspace is None:
            self.atomspace = opencog.atomspace()
        if self.cogserver is None:
            self.cogserver = opencog.cogserver(self.atomspace)
        if self.utilities is None:
            self.utilities = opencog.utilities(self.atomspace)

advisor_agent = Agent(
    model,
    system_prompt=advisor_prompt,
    deps_type=AdvisorDeps,
    retries=2
)

@advisor_agent.system_prompt  
def add_file_list(ctx: RunContext[str]) -> str:
    joined_files = "\n".join(ctx.deps.file_list)
    return f"""
    
    Here is the list of all the files that you can pull the contents of with the
    'get_file_content' tool if the example/tool/MCP server is relevant to the
    agent the user is trying to build:

    {joined_files}
    """

@advisor_agent.tool_plain
def get_file_content(ctx: RunContext[AdvisorDeps], file_path: str) -> str:
    """
    Retrieves the content of a specific file. Use this to get the contents of an example, tool, config for an MCP server
    
    Args:
        ctx: The context including OpenCog components
        file_path: The path to the file
        
    Returns:
        The raw contents of the file
    """
    content = get_file_content_tool(file_path)
    
    # If OpenCog components are available, use them to enhance understanding of the file content
    if ctx.deps.atomspace and ctx.deps.utilities:
        atomspace = ctx.deps.atomspace
        utilities = ctx.deps.utilities
        
        # Create a representation of the file in the AtomSpace
        file_node = atomspace.add_node("ConceptNode", file_path)
        content_node = atomspace.add_node("ConceptNode", "FileContent")
        
        # Extract key concepts from the content for improved reasoning
        # This is a simplified implementation - a more sophisticated approach would use NLP
        lines = content.split('\n')
        for i, line in enumerate(lines[:20]):  # Process first 20 lines for efficiency
            if line.strip():
                # Store important lines like imports, class definitions, and function definitions
                if any(keyword in line for keyword in ["import ", "class ", "def ", "@", "function"]):
                    line_node = atomspace.add_node("ConceptNode", line.strip())
                    utilities.create_atomese_expression(
                        f"(Evaluation (Predicate \"has_line\") (List {file_node} {line_node}))"
                    )
        
        # Create a link between the file and its content
        utilities.create_atomese_expression(
            f"(Evaluation (Predicate \"has_content\") (List {file_node} {content_node}))"
        )
    
    return content

@advisor_agent.tool_plain
def reason_with_opencog(ctx: RunContext[AdvisorDeps], query: str) -> str:
    """
    Utilizes OpenCog's reasoning capabilities to analyze a query and provide insights.
    
    Args:
        ctx: The context including OpenCog components
        query: The query to reason about
        
    Returns:
        Results of the reasoning process
    """
    if not ctx.deps.atomspace or not ctx.deps.utilities:
        return "OpenCog components are not properly initialized."
    
    atomspace = ctx.deps.atomspace
    utilities = ctx.deps.utilities
    
    # Create a query node in the AtomSpace
    query_node = atomspace.add_node("ConceptNode", query)
    
    # Define a simple collaborative reasoning function
    def collaborative_reasoning(atomspace, query):
        """Performs collaborative reasoning on the query using AtomSpace knowledge"""
        results = []
        
        # Get all file nodes in the AtomSpace
        file_nodes = utilities.query_atoms(type_filter="ConceptNode")
        
        # Find relevant knowledge based on the query
        query_terms = set(query.lower().split())
        
        for file_handle in file_nodes:
            file_data = atomspace.get_atom(file_handle)
            if not file_data or "name" not in file_data or not isinstance(file_data["name"], str):
                continue
                
            file_path = file_data["name"]
            
            # Skip nodes that aren't file paths
            if not os.path.exists(file_path):
                continue
                
            # Get all lines associated with this file
            incoming_links = atomspace.get_incoming_set(file_handle)
            lines = []
            
            for link_handle in incoming_links:
                link_data = atomspace.get_atom(link_handle)
                if not link_data:
                    continue
                    
                outgoing_set = atomspace.get_outgoing_set(link_handle)
                if outgoing_set and len(outgoing_set) >= 2:
                    line_handle = outgoing_set[1]
                    line_data = atomspace.get_atom(line_handle)
                    if line_data and "name" in line_data:
                        lines.append(line_data["name"])
            
            # Calculate relevance of this file to the query
            relevance = 0
            
            # Check if any query terms appear in the file path
            for term in query_terms:
                if term in file_path.lower():
                    relevance += 5
                    
            # Check if any query terms appear in the extracted lines
            for line in lines:
                for term in query_terms:
                    if term in line.lower():
                        relevance += 2
            
            if relevance > 0:
                result_entry = {
                    "file": os.path.basename(file_path),
                    "path": file_path,
                    "relevance": relevance,
                    "key_lines": lines[:5]  # Include up to 5 key lines
                }
                results.append(result_entry)
        
        # Sort by relevance
        results.sort(key=lambda x: x["relevance"], reverse=True)
        return results[:5]  # Return top 5 most relevant results
    
    # Register and apply the collaborative reasoning function
    utilities.register_reasoner("collaborative", collaborative_reasoning)
    reasoning_results = utilities.apply_reasoner("collaborative", query)
    
    # Format the results
    if not reasoning_results:
        return "No relevant information found based on the current knowledge."
    
    formatted_results = "Reasoning Results:\n\n"
    for i, result in enumerate(reasoning_results, 1):
        formatted_results += f"{i}. File: {result['file']}\n"
        formatted_results += f"   Path: {result['path']}\n"
        if result.get('key_lines'):
            formatted_results += "   Key concepts:\n"
            for line in result['key_lines']:
                formatted_results += f"   - {line}\n"
        formatted_results += "\n"
    
    return formatted_results
