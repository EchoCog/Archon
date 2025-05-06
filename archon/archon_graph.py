from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai import Agent, RunContext
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, List, Any
from langgraph.config import get_stream_writer
from langgraph.types import interrupt
from dotenv import load_dotenv
from openai import AsyncOpenAI
from supabase import Client
import logfire
import os
import sys

# Import the message classes from Pydantic AI
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter
)

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from archon.pydantic_ai_coder import pydantic_ai_coder, PydanticAIDeps
from archon.advisor_agent import advisor_agent, AdvisorDeps
from archon.refiner_agents.prompt_refiner_agent import prompt_refiner_agent
from archon.refiner_agents.tools_refiner_agent import tools_refiner_agent, ToolsRefinerDeps
from archon.refiner_agents.agent_refiner_agent import agent_refiner_agent, AgentRefinerDeps
from archon.agent_tools import list_documentation_pages_tool
from utils.utils import get_env_var, get_clients

# Import custom OpenCog dependencies
from utils.opencog import opencog

# Load environment variables
load_dotenv()

# Configure logfire to suppress warnings (optional)
logfire.configure(send_to_logfire='never')

provider = get_env_var('LLM_PROVIDER') or 'OpenAI'
base_url = get_env_var('BASE_URL') or 'https://api.openai.com/v1'
api_key = get_env_var('LLM_API_KEY') or 'no-llm-api-key-provided'

is_anthropic = provider == "Anthropic"
is_openai = provider == "OpenAI"

reasoner_llm_model_name = get_env_var('REASONER_MODEL') or 'o3-mini'
reasoner_llm_model = AnthropicModel(reasoner_llm_model_name, api_key=api_key) if is_anthropic else OpenAIModel(reasoner_llm_model_name, base_url=base_url, api_key=api_key)

reasoner = Agent(  
    reasoner_llm_model,
    system_prompt='You are an expert at coding AI agents with Pydantic AI and defining the scope for doing so.',  
)

primary_llm_model_name = get_env_var('PRIMARY_MODEL') or 'gpt-4o-mini'
primary_llm_model = AnthropicModel(primary_llm_model_name, api_key=api_key) if is_anthropic else OpenAIModel(primary_llm_model_name, base_url=base_url, api_key=api_key)

router_agent = Agent(  
    primary_llm_model,
    system_prompt='Your job is to route the user message either to the end of the conversation or to continue coding the AI agent.',  
)

end_conversation_agent = Agent(  
    primary_llm_model,
    system_prompt='Your job is to end a conversation for creating an AI agent by giving instructions for how to execute the agent and they saying a nice goodbye to the user.',  
)

# Initialize clients
embedding_client, supabase = get_clients()

# Define state schema
class AgentState(TypedDict):
    latest_user_message: str
    messages: Annotated[List[bytes], lambda x, y: x + y]

    scope: str
    advisor_output: str
    file_list: List[str]

    refined_prompt: str
    refined_tools: str
    refined_agent: str

    # Add OpenCog components
    atomspace: Any
    cogserver: Any
    utilities: Any

# Scope Definition Node with Reasoner LLM
async def define_scope_with_reasoner(state: AgentState):
    # Initialize OpenCog components
    atomspace = opencog.atomspace.AtomSpace()
    cogserver = opencog.cogserver.CogServer(atomspace)
    utilities = opencog.utilities.Utilities(atomspace)

    # First, get the documentation pages so the reasoner can decide which ones are necessary
    documentation_pages = await list_documentation_pages_tool(supabase)
    documentation_pages_str = "\n".join(documentation_pages)

    # Create knowledge representation for the user request in AtomSpace
    # This enables complex reasoning patterns via Atomese
    request_node = atomspace.add_node("ConceptNode", "UserRequest")
    request_content_node = atomspace.add_node("ConceptNode", state['latest_user_message'])
    utilities.create_atomese_expression(f"(Evaluation (Predicate \"has_content\") (List {request_node} {request_content_node}))")
    
    # Add documentation pages to AtomSpace for reasoning
    doc_list_node = atomspace.add_node("ConceptNode", "DocumentationPages")
    for doc_page in documentation_pages:
        page_node = atomspace.add_node("ConceptNode", doc_page)
        utilities.create_atomese_expression(f"(Member {page_node} {doc_list_node})")

    # Register a simple reasoner to identify relevant documentation based on content similarity
    def doc_relevance_reasoner(atomspace, request_text):
        """Simple reasoner to identify relevant documentation pages"""
        relevant_docs = []
        request_words = set(request_text.lower().split())
        
        # Get all documentation pages from AtomSpace
        doc_list = atomspace.get_atom(doc_list_node)
        if not doc_list:
            return []
            
        # Find relevant docs based on simple word overlap
        for doc_handle in utilities.query_atoms(type_filter="ConceptNode"):
            doc = atomspace.get_atom(doc_handle)
            if doc and "name" in doc:
                doc_name = doc["name"]
                # Simple relevance check - could be enhanced with more sophisticated NLP
                if any(word in doc_name.lower() for word in request_words):
                    relevant_docs.append(doc_name)
                    
        return relevant_docs
    
    # Register the reasoner
    utilities.register_reasoner("doc_relevance", doc_relevance_reasoner)
    
    # Apply the reasoner to find relevant documentation
    relevant_docs = utilities.apply_reasoner("doc_relevance", state['latest_user_message'])
    
    # Then, use the reasoner to define the scope, enhanced with OpenCog reasoning
    prompt = f"""
    User AI Agent Request: {state['latest_user_message']}
    
    Create detailed scope document for the AI agent including:
    - Architecture diagram
    - Core components
    - External dependencies
    - Testing strategy

    Also based on these documentation pages available:

    {documentation_pages_str}
    
    Based on advanced reasoning, these documentation pages are particularly relevant:
    {', '.join(relevant_docs) if relevant_docs else 'No specific pages identified as highly relevant'}

    Include a list of documentation pages that are relevant to creating this agent for the user in the scope document.
    """

    result = await reasoner.run(prompt)
    scope = result.data

    # Store the agent requirements in AtomSpace for future reasoning
    scope_node = atomspace.add_node("ConceptNode", "AgentScope")
    scope_content_node = atomspace.add_node("ConceptNode", scope)
    utilities.create_atomese_expression(f"(Evaluation (Predicate \"has_content\") (List {scope_node} {scope_content_node}))")
    utilities.create_atomese_expression(f"(Inheritance {scope_node} {request_node})")

    # Get the directory one level up from the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    scope_path = os.path.join(parent_dir, "workbench", "scope.md")
    os.makedirs(os.path.join(parent_dir, "workbench"), exist_ok=True)

    with open(scope_path, "w", encoding="utf-8") as f:
        f.write(scope)
    
    return {"scope": scope, "atomspace": atomspace, "cogserver": cogserver, "utilities": utilities}

# Advisor agent - create a starting point based on examples and prebuilt tools/MCP servers
async def advisor_with_examples(state: AgentState):
    # Get the directory one level up from the current file (archon_graph.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    # The agent-resources folder is adjacent to the parent folder of archon_graph.py
    agent_resources_dir = os.path.join(parent_dir, "agent-resources")
    
    # Get a list of all files in the agent-resources directory and its subdirectories
    file_list = []
    
    for root, dirs, files in os.walk(agent_resources_dir):
        for file in files:
            # Get the full path to the file
            file_path = os.path.join(root, file)
            # Use the full path instead of relative path
            file_list.append(file_path)
    
    # Access the OpenCog components from the state if they exist,
    # otherwise create new instances
    atomspace = state.get('atomspace')
    cogserver = state.get('cogserver')
    utilities = state.get('utilities')
    
    if not atomspace or not cogserver or not utilities:
        # Create new OpenCog components if they don't exist
        atomspace = opencog.atomspace.AtomSpace()
        cogserver = opencog.cogserver.CogServer(atomspace)
        utilities = opencog.utilities.Utilities(atomspace)
    
    # Leverage AtomSpace for organizing and reasoning about example files
    # Create a node representing the example files category
    examples_node = atomspace.add_node("ConceptNode", "ExampleFiles")
    tools_node = atomspace.add_node("ConceptNode", "ToolFiles")
    mcp_node = atomspace.add_node("ConceptNode", "MCPFiles")
    
    # Add file entries to AtomSpace with appropriate categorization
    for file_path in file_list:
        file_node = atomspace.add_node("ConceptNode", file_path)
        
        # Categorize files based on path
        if "/examples/" in file_path:
            utilities.create_atomese_expression(f"(Member {file_node} {examples_node})")
        elif "/tools/" in file_path:
            utilities.create_atomese_expression(f"(Member {file_node} {tools_node})")
        elif "/mcps/" in file_path:
            utilities.create_atomese_expression(f"(Member {file_node} {mcp_node})")
    
    # Add user request to AtomSpace
    request_node = atomspace.add_node("ConceptNode", "UserRequest")
    request_content_node = atomspace.add_node("ConceptNode", state['latest_user_message'])
    utilities.create_atomese_expression(f"(Evaluation (Predicate \"has_content\") (List {request_node} {request_content_node}))")
    
    # Define a simple file relevance reasoner based on keyword matching
    def file_relevance_reasoner(atomspace, request_text, utilities):
        """Reasoner to identify relevant example files for a user request"""
        request_keywords = set(request_text.lower().split())
        relevance_scores = {}
        
        # Get all file nodes from AtomSpace
        for file_handle in utilities.query_atoms(type_filter="ConceptNode"):
            file_data = atomspace.get_atom(file_handle)
            if not file_data or "name" not in file_data:
                continue
                
            file_path = file_data["name"]
            if not isinstance(file_path, str) or not os.path.exists(file_path):
                continue
                
            # Calculate relevance based on path components and filename
            score = 0
            path_parts = file_path.lower().split('/')
            
            # Check filename relevance
            filename = path_parts[-1]
            for keyword in request_keywords:
                if keyword in filename:
                    score += 10
            
            # Check path relevance
            for part in path_parts:
                for keyword in request_keywords:
                    if keyword in part:
                        score += 5
            
            # Assign higher scores to examples that match request type
            if any(kw in request_text.lower() for kw in ["web", "search", "browser"]):
                if "web_search" in file_path.lower() or "brave_search" in file_path.lower():
                    score += 15
            
            if any(kw in request_text.lower() for kw in ["github", "git", "repo", "repository"]):
                if "github" in file_path.lower() or "git" in file_path.lower():
                    score += 15
                    
            if any(kw in request_text.lower() for kw in ["mcp", "model", "context", "protocol"]):
                if "/mcps/" in file_path:
                    score += 15
            
            # If the score is positive, consider it relevant
            if score > 0:
                relevance_scores[file_path] = score
                
        # Sort by relevance score
        sorted_files = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)
        return [file_path for file_path, score in sorted_files]
    
    # Register the reasoner
    utilities.register_reasoner("file_relevance", file_relevance_reasoner)
    
    # Apply the reasoner to find relevant example files
    relevant_files = utilities.apply_reasoner("file_relevance", state['latest_user_message'], utilities)
    
    # Update the file list to prioritize relevant files
    # Put the most relevant files first in the list, followed by the rest
    file_list = relevant_files + [f for f in file_list if f not in relevant_files]
    
    # Then, prompt the advisor with the enhanced list of files it can use for examples and tools
    deps = AdvisorDeps(
        file_list=file_list,
        atomspace=atomspace,
        cogserver=cogserver,
        utilities=utilities
    )
    
    result = await advisor_agent.run(state['latest_user_message'], deps=deps)
    advisor_output = result.data
    
    return {"file_list": file_list, "advisor_output": advisor_output, "atomspace": atomspace, "cogserver": cogserver, "utilities": utilities}

# Coding Node with Feedback Handling
async def coder_agent(state: AgentState, writer):    
    # Prepare dependencies
    deps = PydanticAIDeps(
        supabase=supabase,
        embedding_client=embedding_client,
        reasoner_output=state['scope'],
        advisor_output=state['advisor_output']
    )

    # Get the message history into the format for Pydantic AI
    message_history: list[ModelMessage] = []
    for message_row in state['messages']:
        message_history.extend(ModelMessagesTypeAdapter.validate_json(message_row))

    # The prompt either needs to be the user message (initial agent request or feedback)
    # or the refined prompt/tools/agent if we are in that stage of the agent creation process
    if 'refined_prompt' in state and state['refined_prompt']:
        prompt = f"""
        I need you to refine the agent you created. 
        
        Here is the refined prompt:\n
        {state['refined_prompt']}\n\n

        Here are the refined tools:\n
        {state['refined_tools']}\n

        And finally, here are the changes to the agent definition to make if any:\n
        {state['refined_agent']}\n\n

        Output any changes necessary to the agent code based on these refinements.
        """
    else:
        prompt = state['latest_user_message']

    # Run the agent in a stream
    if not is_openai:
        writer = get_stream_writer()
        result = await pydantic_ai_coder.run(prompt, deps=deps, message_history=message_history)
        writer(result.data)
    else:
        async with pydantic_ai_coder.run_stream(
            state['latest_user_message'],
            deps=deps,
            message_history=message_history
        ) as result:
            # Stream partial text as it arrives
            async for chunk in result.stream_text(delta=True):
                writer(chunk)

    # print(ModelMessagesTypeAdapter.validate_json(result.new_messages_json()))

    # Add the new conversation history (including tool calls)
    # Reset the refined properties in case they were just used to refine the agent
    return {
        "messages": [result.new_messages_json()],
        "refined_prompt": "",
        "refined_tools": "",
        "refined_agent": ""
    }

# Interrupt the graph to get the user's next message
def get_next_user_message(state: AgentState):
    value = interrupt({})

    # Set the user's latest message for the LLM to continue the conversation
    return {
        "latest_user_message": value
    }

# Determine if the user is finished creating their AI agent or not
async def route_user_message(state: AgentState):
    prompt = f"""
    The user has sent a message: 
    
    {state['latest_user_message']}

    If the user wants to end the conversation, respond with just the text "finish_conversation".
    If the user wants to continue coding the AI agent and gave feedback, respond with just the text "coder_agent".
    If the user asks specifically to "refine" the agent, respond with just the text "refine".
    """

    result = await router_agent.run(prompt)
    
    if result.data == "finish_conversation": return "finish_conversation"
    if result.data == "refine": return ["refine_prompt", "refine_tools", "refine_agent"]
    return "coder_agent"

# Refines the prompt for the AI agent
async def refine_prompt(state: AgentState):
    # Get the message history into the format for Pydantic AI
    message_history: list[ModelMessage] = []
    for message_row in state['messages']:
        message_history.extend(ModelMessagesTypeAdapter.validate_json(message_row))

    prompt = "Based on the current conversation, refine the prompt for the agent."

    # Run the agent to refine the prompt for the agent being created
    result = await prompt_refiner_agent.run(prompt, message_history=message_history)

    return {"refined_prompt": result.data}

# Refines the tools for the AI agent
async def refine_tools(state: AgentState):
    # Prepare dependencies
    deps = ToolsRefinerDeps(
        supabase=supabase,
        embedding_client=embedding_client,
        file_list=state['file_list']
    )

    # Get the message history into the format for Pydantic AI
    message_history: list[ModelMessage] = []
    for message_row in state['messages']:
        message_history.extend(ModelMessagesTypeAdapter.validate_json(message_row))

    prompt = "Based on the current conversation, refine the tools for the agent."

    # Run the agent to refine the tools for the agent being created
    result = await tools_refiner_agent.run(prompt, deps=deps, message_history=message_history)

    return {"refined_tools": result.data}

# Refines the defintion for the AI agent
async def refine_agent(state: AgentState):
    # Prepare dependencies
    deps = AgentRefinerDeps(
        supabase=supabase,
        embedding_client=embedding_client
    )

    # Get the message history into the format for Pydantic AI
    message_history: list[ModelMessage] = []
    for message_row in state['messages']:
        message_history.extend(ModelMessagesTypeAdapter.validate_json(message_row))

    prompt = "Based on the current conversation, refine the agent definition."

    # Run the agent to refine the definition for the agent being created
    result = await agent_refiner_agent.run(prompt, deps=deps, message_history=message_history)

    return {"refined_agent": result.data}

# End of conversation agent to give instructions for executing the agent
async def finish_conversation(state: AgentState, writer):    
    # Get the message history into the format for Pydantic AI
    message_history: list[ModelMessage] = []
    for message_row in state['messages']:
        message_history.extend(ModelMessagesTypeAdapter.validate_json(message_row))

    # Run the agent in a stream
    if not is_openai:
        writer = get_stream_writer()
        result = await end_conversation_agent.run(state['latest_user_message'], message_history= message_history)
        writer(result.data)   
    else: 
        async with end_conversation_agent.run_stream(
            state['latest_user_message'],
            message_history= message_history
        ) as result:
            # Stream partial text as it arrives
            async for chunk in result.stream_text(delta=True):
                writer(chunk)

    return {"messages": [result.new_messages_json()]}        

# Build workflow
builder = StateGraph(AgentState)

# Add nodes
builder.add_node("define_scope_with_reasoner", define_scope_with_reasoner)
builder.add_node("advisor_with_examples", advisor_with_examples)
builder.add_node("coder_agent", coder_agent)
builder.add_node("get_next_user_message", get_next_user_message)
builder.add_node("refine_prompt", refine_prompt)
builder.add_node("refine_tools", refine_tools)
builder.add_node("refine_agent", refine_agent)
builder.add_node("finish_conversation", finish_conversation)

# Set edges
builder.add_edge(START, "define_scope_with_reasoner")
builder.add_edge(START, "advisor_with_examples")
builder.add_edge("define_scope_with_reasoner", "coder_agent")
builder.add_edge("advisor_with_examples", "coder_agent")
builder.add_edge("coder_agent", "get_next_user_message")
builder.add_conditional_edges(
    "get_next_user_message",
    route_user_message,
    ["coder_agent", "finish_conversation", "refine_prompt", "refine_tools", "refine_agent"]
)
builder.add_edge("refine_prompt", "coder_agent")
builder.add_edge("refine_tools", "coder_agent")
builder.add_edge("refine_agent", "coder_agent")
builder.add_edge("finish_conversation", END)

# Configure persistence
memory = MemorySaver()
agentic_flow = builder.compile(checkpointer=memory)
