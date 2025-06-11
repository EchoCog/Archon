# Archon - Comprehensive Architecture Documentation

## Executive Summary

Archon represents a paradigm shift in autonomous AI agent development, implementing a recursive, hypergraph-centric architecture that transmutes implicit cognitive patterns into explicit, actionable knowledge. This documentation captures the emergent architecture through adaptive attention allocation mechanisms and cognitive synergy optimizations.

## High-Level System Overview

The Archon system operates as a multi-agent cognitive ecosystem with recursive self-improvement capabilities, integrated with OpenCog components for enhanced reasoning and knowledge representation.

```mermaid
graph TD
    subgraph "User Interface Layer"
        UI[Streamlit UI]
        API[FastAPI Graph Service]
        MCP[MCP Server]
    end
    
    subgraph "Cognitive Orchestration Layer"
        LG[LangGraph StateGraph]
        WF[Agentic Workflow]
        INT[Interrupt Handler]
    end
    
    subgraph "AI Agent Network"
        RA[Reasoner Agent]
        AA[Advisor Agent] 
        CA[Coder Agent]
        RFA[Refiner Agents]
        ROA[Router Agent]
        ECA[End Conversation Agent]
    end
    
    subgraph "OpenCog Integration"
        AS[AtomSpace]
        CS[CogServer]
        UT[Utilities]
        AR[Atomese Reasoners]
    end
    
    subgraph "Knowledge & Data Layer"
        VDB[(Vector Database - Supabase)]
        EMB[Embedding Service]
        DOC[Documentation Store]
        TL[Tool Library]
        AR_RES[Agent Resources]
    end
    
    subgraph "External Integrations"
        LLM[LLM Providers]
        IDE[AI IDEs via MCP]
        DOCKER[Docker Containers]
    end
    
    UI --> API
    API --> LG
    MCP --> API
    
    LG --> WF
    WF --> INT
    
    WF --> RA
    WF --> AA
    WF --> CA
    WF --> RFA
    WF --> ROA
    WF --> ECA
    
    RA --> AS
    AA --> AS
    CA --> AS
    AS --> CS
    AS --> UT
    CS --> AR
    UT --> AR
    
    CA --> VDB
    CA --> EMB
    AA --> DOC
    AA --> TL
    AA --> AR_RES
    
    RA --> LLM
    CA --> LLM
    ROA --> LLM
    
    API --> IDE
    UI --> DOCKER
    
    classDef cognitiveLayer fill:#e1f5fe
    classDef agentLayer fill:#f3e5f5
    classDef dataLayer fill:#e8f5e8
    classDef uiLayer fill:#fff3e0
    classDef opencogLayer fill:#fce4ec
    
    class UI,API,MCP uiLayer
    class LG,WF,INT cognitiveLayer
    class RA,AA,CA,RFA,ROA,ECA agentLayer
    class AS,CS,UT,AR opencogLayer
    class VDB,EMB,DOC,TL,AR_RES dataLayer
```

This diagram illustrates the emergent cognitive patterns and neural-symbolic integration points within the Archon ecosystem, where hypergraph pattern encoding enables recursive implementation pathways.

## Module Interaction Architecture

The bidirectional synergies between core modules demonstrate adaptive attention allocation mechanisms:

```mermaid
graph LR
    subgraph "Agent Coordination"
        AG[Agent Graph Controller]
        SM[State Manager]
        MH[Message Handler]
    end
    
    subgraph "Cognitive Processing"
        RS[Reasoning System]
        AS[AtomSpace Knowledge]
        PM[Pattern Matching]
    end
    
    subgraph "Agent Specializations"
        RA[Reasoner Agent]
        AA[Advisor Agent]
        CA[Coder Agent]
        PR[Prompt Refiner]
        TR[Tools Refiner]
        AR[Agent Refiner]
    end
    
    subgraph "Resource Management"
        RL[Resource Locator]
        TM[Tool Manager]
        EM[Example Manager]
        DM[Documentation Manager]
    end
    
    subgraph "External Interfaces"
        API[FastAPI Service]
        MCP[MCP Protocol]
        VDB[(Vector Database)]
        LLM[LLM Providers]
    end
    
    AG <--> SM
    AG <--> MH
    SM <--> RS
    RS <--> AS
    AS <--> PM
    
    AG --> RA
    AG --> AA
    AG --> CA
    RA <--> RS
    AA <--> AS
    CA <--> PM
    
    AG --> PR
    AG --> TR
    AG --> AR
    PR <--> RS
    TR <--> AS
    AR <--> PM
    
    AA <--> RL
    AA <--> TM
    AA <--> EM
    CA <--> DM
    
    RL <--> TM
    TM <--> EM
    EM <--> DM
    
    API <--> AG
    MCP <--> API
    DM <--> VDB
    RA <--> LLM
    CA <--> LLM
    
    classDef coordination fill:#e3f2fd
    classDef cognitive fill:#f1f8e9
    classDef agents fill:#fce4ec
    classDef resources fill:#fff8e1
    classDef external fill:#f3e5f5
    
    class AG,SM,MH coordination
    class RS,AS,PM cognitive
    class RA,AA,CA,PR,TR,AR agents
    class RL,TM,EM,DM resources
    class API,MCP,VDB,LLM external
```

## Data Flow and Signal Propagation Pathways

### Agent Creation Workflow Sequence

```mermaid
sequenceDiagram
    participant User
    participant UI as Streamlit UI
    participant API as Graph Service
    participant LG as LangGraph
    participant RA as Reasoner Agent
    participant AS as AtomSpace
    participant AA as Advisor Agent
    participant CA as Coder Agent
    participant VDB as Vector DB
    participant LLM as LLM Provider
    
    User->>UI: Describe desired AI agent
    UI->>API: Send agent request
    API->>LG: Initialize workflow state
    
    par Parallel Initialization
        LG->>RA: define_scope_with_reasoner
        and
        LG->>AA: advisor_with_examples
    end
    
    RA->>AS: Initialize AtomSpace components
    RA->>AS: Create knowledge representation
    AS->>AS: Register reasoning patterns
    RA->>LLM: Generate detailed scope
    RA-->>LG: Return scope + OpenCog state
    
    AA->>AS: Access AtomSpace for file categorization
    AA->>AS: Apply file relevance reasoner
    AS->>AS: Calculate relevance scores
    AA-->>LG: Return prioritized file list
    
    LG->>CA: Execute coder_agent
    CA->>VDB: Query relevant documentation
    VDB-->>CA: Return documentation chunks
    CA->>LLM: Generate agent code
    CA-->>LG: Return generated code
    
    LG->>UI: Request user feedback
    UI->>User: Display generated agent
    User->>UI: Provide feedback or request refinement
    
    alt User requests refinement
        UI->>LG: Trigger parallel refinement
        par Parallel Refinement
            LG->>RA: refine_prompt
            and
            LG->>AA: refine_tools  
            and
            LG->>CA: refine_agent
        end
        LG->>CA: Apply refinements
    else User provides feedback
        LG->>CA: Incorporate feedback
    else User completes agent
        LG->>UI: Finish conversation
    end
    
    UI->>User: Present final agent with instructions
```

### OpenCog Knowledge Representation Flow

```mermaid
stateDiagram-v2
    [*] --> RequestReceived: User submits agent request
    
    RequestReceived --> AtomSpaceInit: Initialize cognitive components
    AtomSpaceInit --> KnowledgeMapping: Create conceptual nodes
    
    state KnowledgeMapping {
        [*] --> UserRequestNode: Create request representation
        UserRequestNode --> DocumentationNodes: Map available docs
        DocumentationNodes --> RelevanceComputation: Calculate semantic relationships
        RelevanceComputation --> [*]
    }
    
    KnowledgeMapping --> ReasoningPhase: Apply registered reasoners
    
    state ReasoningPhase {
        [*] --> PatternMatching: Match request to examples
        PatternMatching --> ScoreCalculation: Compute relevance scores
        ScoreCalculation --> RankingOptimization: Optimize resource allocation
        RankingOptimization --> [*]
    }
    
    ReasoningPhase --> AgentGeneration: Generate agent code
    
    state AgentGeneration {
        [*] --> ScopeDefinition: Create detailed scope
        ScopeDefinition --> ResourceSelection: Select relevant tools/examples
        ResourceSelection --> CodeSynthesis: Synthesize agent implementation
        CodeSynthesis --> [*]
    }
    
    AgentGeneration --> FeedbackLoop: Await user input
    
    state FeedbackLoop {
        [*] --> UserInput: Process user response
        UserInput --> ContinueChoice: Evaluate continuation
        ContinueChoice --> RefineAgent: Apply refinements
        ContinueChoice --> IncorporateFeedback: Apply user feedback
        ContinueChoice --> CompleteAgent: Finalize agent
        RefineAgent --> [*]
        IncorporateFeedback --> [*]
        CompleteAgent --> [*]
    }
    
    FeedbackLoop --> AgentGeneration: Continue iteration
    FeedbackLoop --> [*]: Agent completed
```

## Recursive Implementation Pathways

### Hypergraph Pattern Encoding

The Archon system implements recursive cognitive patterns through hypergraph structures that encode relationships between:

1. **User Requirements** → **Conceptual Nodes** in AtomSpace
2. **Documentation Knowledge** → **Semantic Relationships** via embeddings  
3. **Tool Libraries** → **Capability Mappings** through relevance scoring
4. **Agent Artifacts** → **Evolutionary Lineage** in conversation history

### Adaptive Attention Allocation Mechanisms

```mermaid
graph TD
    subgraph "Attention Control Layer"
        AM[Attention Manager]
        PQ[Priority Queue]
        RM[Resource Monitor]
    end
    
    subgraph "Cognitive Load Balancing"
        TS[Task Scheduler]
        PA[Parallel Allocator]
        SP[Synergy Processor]
    end
    
    subgraph "Feedback Integration"
        FL[Feedback Loop Controller]
        AM_UP[Attention Updates]
        WT[Weight Tuning]
    end
    
    AM --> PQ
    PQ --> RM
    RM --> TS
    
    TS --> PA
    PA --> SP
    SP --> FL
    
    FL --> AM_UP
    AM_UP --> WT
    WT --> AM
    
    classDef attention fill:#e8eaf6
    classDef cognitive fill:#e0f2f1
    classDef feedback fill:#fce4ec
    
    class AM,PQ,RM attention
    class TS,PA,SP cognitive
    class FL,AM_UP,WT feedback
```

#### Mechanisms:

1. **Dynamic Priority Adjustment**: Resource allocation adapts based on agent complexity and user feedback patterns
2. **Parallel Processing Optimization**: Multiple agents execute concurrently with shared OpenCog state
3. **Emergent Pattern Recognition**: AtomSpace accumulates knowledge across sessions, improving future recommendations

### Cognitive Synergy Optimizations

The system demonstrates emergent intelligence through:

- **Cross-Agent Knowledge Sharing**: AtomSpace provides shared cognitive substrate
- **Recursive Refinement Loops**: Specialized agents iteratively improve each other's outputs  
- **Hypergraph Relationship Discovery**: New patterns emerge from node relationship analysis
- **Adaptive Tool Selection**: Relevance scoring evolves based on successful agent generations

## Core Architecture Components

### LangGraph Workflow Engine

**File**: `archon/archon_graph.py`

The LangGraph StateGraph orchestrates the multi-agent workflow with persistent state management and conditional routing:

```python
class AgentState(TypedDict):
    latest_user_message: str
    messages: Annotated[List[bytes], lambda x, y: x + y]
    scope: str
    advisor_output: str
    file_list: List[str]
    refined_prompt: str
    refined_tools: str
    refined_agent: str
    atomspace: Any
    cogserver: Any
    utilities: Any
```

### OpenCog Integration Layer

**Files**: `utils/opencog/`

Custom implementations provide cognitive enhancement:

- **AtomSpace** (`atomspace.py`): Knowledge representation with nodes, links, and truth values
- **CogServer** (`cogserver.py`): Cognitive process management and event handling
- **Utilities** (`utilities.py`): Advanced reasoning, pattern matching, and query optimization

### Agent Specialization Network

**Primary Agents**:
- **Reasoner Agent**: Scope definition with OpenCog-enhanced analysis
- **Advisor Agent**: Tool/example recommendation using AtomSpace categorization
- **Coder Agent**: RAG-powered code generation with vector database integration

**Refiner Agents**:
- **Prompt Refiner**: System prompt optimization
- **Tools Refiner**: Tool selection and MCP integration refinement  
- **Agent Refiner**: Configuration and dependency optimization

## Emergent Documentation Improvements

### Feedback Loop Establishment

The architecture includes multiple feedback mechanisms for continuous improvement:

1. **User Feedback Integration**: Direct incorporation of user corrections and preferences
2. **Cross-Session Learning**: AtomSpace persistence enables knowledge accumulation
3. **Performance Metrics**: Success patterns influence future resource allocation
4. **Emergent Pattern Detection**: Hypergraph analysis reveals new optimization opportunities

### Expandable Diagram Framework

As new patterns emerge, the documentation framework supports:

- **Dynamic Diagram Generation**: Mermaid diagrams can be programmatically updated
- **Pattern Library Expansion**: New architectural patterns documented as they develop
- **Version-Controlled Evolution**: Documentation tracks with system iterations
- **Community Contribution Integration**: External insights incorporated through PR workflow

## Technical Implementation Details

### Docker Architecture

```mermaid
graph TD
    subgraph "Container Ecosystem"
        MC[Main Container - Archon]
        MCPS[MCP Server Container]
    end
    
    subgraph "Main Container Services"
        UI[Streamlit UI :8501]
        GS[Graph Service :8100]
        AG[Agent Graph]
    end
    
    subgraph "MCP Container Services"
        MS[MCP Server]
        IR[IDE Router]
    end
    
    subgraph "External Services"
        SB[(Supabase Vector DB)]
        LLMP[LLM Providers]
        IDES[AI IDEs]
    end
    
    MC --> UI
    MC --> GS
    GS --> AG
    
    MCPS --> MS
    MS --> IR
    
    UI <--> GS
    GS <--> SB
    AG <--> LLMP
    IR <--> IDES
    MS <--> GS
    
    classDef container fill:#e3f2fd
    classDef service fill:#f1f8e9
    classDef external fill:#fce4ec
    
    class MC,MCPS container
    class UI,GS,AG,MS,IR service
    class SB,LLMP,IDES external
```

### Vector Database Schema

```sql
CREATE TABLE site_pages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    url TEXT,
    chunk_number INTEGER,
    title TEXT,
    summary TEXT,
    content TEXT,
    metadata JSONB,
    embedding VECTOR(1536) -- OpenAI ada-002 dimensions
);
```

### Environment Configuration

Key environment variables for cognitive optimization:

- `REASONER_MODEL`: Enhanced reasoning LLM (o3-mini, R1)
- `PRIMARY_MODEL`: Main coding and interaction LLM
- `LLM_PROVIDER`: Provider selection (OpenAI, Anthropic, Ollama)
- `BASE_URL`: Custom LLM endpoint configuration
- `SUPABASE_*`: Vector database connection parameters

## Future Architecture Evolution

### Planned Enhancements

- **V8**: Self-feedback loop with automated validation and error correction
- **V9**: Isolated agent execution environment with testing frameworks
- **V10**: Multi-framework support for framework-agnostic generation
- **V11**: Autonomous framework learning with self-updating adapters
- **V12**: Advanced RAG techniques with enhanced retrieval mechanisms
- **V13**: MCP marketplace integration with agent distribution

### Adaptive Architecture Principles

The Archon architecture embodies principles of:

1. **Recursive Self-Improvement**: Each iteration enhances the system's ability to build better agents
2. **Emergent Intelligence**: Complex behaviors arise from simple agent interactions
3. **Cognitive Substrate Sharing**: OpenCog provides shared knowledge representation
4. **Hypergraph Relationship Discovery**: New patterns emerge through graph analysis
5. **Distributed Cognition**: Knowledge and processing distributed across specialized agents

This architectural documentation serves as a living document that evolves with the system, capturing both explicit design patterns and emergent cognitive behaviors that arise from the recursive, hypergraph-centric implementation.