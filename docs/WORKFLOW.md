# Archon Workflow and Data Flow Documentation

## Cognitive Workflow Overview

The Archon system implements a sophisticated multi-agent workflow that demonstrates emergent cognitive patterns through recursive implementation pathways and adaptive attention allocation mechanisms.

## LangGraph State Management

### State Schema and Transitions

```mermaid
stateDiagram-v2
    [*] --> WorkflowInit: Initialize AgentState
    
    state WorkflowInit {
        [*] --> StateCreation: Create empty state
        StateCreation --> MessageInit: Initialize message history
        MessageInit --> OpenCogInit: Initialize cognitive components
        OpenCogInit --> [*]
    }
    
    WorkflowInit --> ParallelPhase: Parallel initialization
    
    state ParallelPhase {
        [*] --> ScopeDefinition: Reasoner agent
        [*] --> AdvisorAnalysis: Advisor agent
        ScopeDefinition --> ScopeComplete: Scope ready
        AdvisorAnalysis --> AdvisorComplete: Resources ready
        ScopeComplete --> [*]
        AdvisorComplete --> [*]
    }
    
    ParallelPhase --> CodingPhase: Both phases complete
    
    state CodingPhase {
        [*] --> CodeGeneration: Generate initial agent
        CodeGeneration --> UserInteraction: Present to user
        UserInteraction --> FeedbackEvaluation: Evaluate response
        FeedbackEvaluation --> ContinueCoding: Apply feedback
        FeedbackEvaluation --> RefinementPhase: Trigger refinement
        FeedbackEvaluation --> Completion: Finish agent
        ContinueCoding --> CodeGeneration
        RefinementPhase --> CodeGeneration
        Completion --> [*]
    }
    
    CodingPhase --> [*]: Agent completed
    
    state RefinementPhase {
        [*] --> PromptRefinement: Refine system prompt
        [*] --> ToolsRefinement: Refine tools selection
        [*] --> AgentRefinement: Refine agent config
        PromptRefinement --> [*]
        ToolsRefinement --> [*] 
        AgentRefinement --> [*]
    }
```

### State Data Structure

```python
class AgentState(TypedDict):
    # User interaction state
    latest_user_message: str
    messages: Annotated[List[bytes], lambda x, y: x + y]
    
    # Agent artifacts
    scope: str
    advisor_output: str
    file_list: List[str]
    
    # Refinement outputs
    refined_prompt: str
    refined_tools: str
    refined_agent: str
    
    # OpenCog cognitive state
    atomspace: Any
    cogserver: Any
    utilities: Any
```

## Agent Interaction Patterns

### Multi-Agent Coordination

```mermaid
graph TD
    subgraph "Coordination Layer"
        LG[LangGraph Controller]
        SM[State Manager]
        EH[Event Handler]
    end
    
    subgraph "Primary Agents"
        RA[Reasoner Agent]
        AA[Advisor Agent]
        CA[Coder Agent]
    end
    
    subgraph "Specialist Agents"
        PR[Prompt Refiner]
        TR[Tools Refiner]
        AR[Agent Refiner]
    end
    
    subgraph "Support Agents"
        ROA[Router Agent]
        ECA[End Conversation Agent]
    end
    
    subgraph "Shared Resources"
        AS[AtomSpace]
        VDB[(Vector Database)]
        TL[Tool Library]
    end
    
    LG --> SM
    SM --> EH
    
    LG --> RA
    LG --> AA
    LG --> CA
    
    LG --> PR
    LG --> TR
    LG --> AR
    
    LG --> ROA
    LG --> ECA
    
    RA <--> AS
    AA <--> AS
    CA <--> VDB
    AA <--> TL
    
    EH --> AS
    SM --> VDB
    
    classDef coordination fill:#e3f2fd
    classDef primary fill:#e8f5e8
    classDef specialist fill:#fff3e0
    classDef support fill:#fce4ec
    classDef resources fill:#f3e5f5
    
    class LG,SM,EH coordination
    class RA,AA,CA primary
    class PR,TR,AR specialist
    class ROA,ECA support
    class AS,VDB,TL resources
```

## Detailed Workflow Sequences

### Agent Creation Complete Sequence

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
    participant VDB as Vector Database
    participant LLM as LLM Provider
    participant PR as Prompt Refiner
    participant TR as Tools Refiner
    participant AR as Agent Refiner
    
    Note over User, AR: Initial Agent Request Phase
    User->>UI: "Create a web scraping agent"
    UI->>API: POST /chat with request
    API->>LG: initialize_workflow(user_message)
    
    Note over LG, AR: Parallel Initialization Phase
    par Scope Definition
        LG->>RA: define_scope_with_reasoner(state)
        RA->>AS: Initialize AtomSpace components
        RA->>AS: add_node("ConceptNode", "UserRequest")
        RA->>AS: create_atomese_expression(request_mapping)
        AS-->>RA: Return node handles
        RA->>LLM: Generate detailed scope with context
        LLM-->>RA: Return comprehensive scope
        RA-->>LG: Return {scope, atomspace, cogserver, utilities}
    and Advisor Analysis
        LG->>AA: advisor_with_examples(state)
        AA->>AS: Access shared AtomSpace
        AA->>AS: Categorize agent resources by type
        AA->>AS: Apply file relevance reasoner
        AS-->>AA: Return prioritized file list
        AA->>LLM: Generate recommendations with context
        LLM-->>AA: Return advisor output
        AA-->>LG: Return {file_list, advisor_output}
    end
    
    Note over LG, AR: Code Generation Phase
    LG->>CA: coder_agent(state, writer)
    CA->>VDB: query_documentation("web scraping")
    VDB-->>CA: Return relevant doc chunks
    CA->>LLM: Generate agent code with RAG context
    
    loop Streaming Response
        LLM-->>CA: Stream code chunks
        CA-->>API: Write partial response
        API-->>UI: Stream to user interface
    end
    
    CA-->>LG: Return {messages, new_conversation_state}
    LG->>UI: interrupt() for user feedback
    UI->>User: Display generated agent code
    
    Note over User, AR: User Feedback Loop
    User->>UI: "Please refine this agent"
    UI->>LG: Continue workflow with feedback
    LG->>ROA: route_user_message(state)
    ROA->>LLM: Analyze user intent
    LLM-->>ROA: Return "refine"
    ROA-->>LG: Return ["refine_prompt", "refine_tools", "refine_agent"]
    
    Note over LG, AR: Parallel Refinement Phase
    par Prompt Refinement
        LG->>PR: refine_prompt(state)
        PR->>LLM: Optimize system prompt based on conversation
        LLM-->>PR: Return refined prompt
        PR-->>LG: Return {refined_prompt}
    and Tools Refinement
        LG->>TR: refine_tools(state)
        TR->>VDB: Query for better tool options
        VDB-->>TR: Return enhanced tool suggestions
        TR->>LLM: Refine tool selection
        LLM-->>TR: Return refined tools
        TR-->>LG: Return {refined_tools}
    and Agent Refinement
        LG->>AR: refine_agent(state)
        AR->>LLM: Optimize agent configuration
        LLM-->>AR: Return refined agent config
        AR-->>LG: Return {refined_agent}
    end
    
    Note over LG, AR: Apply Refinements
    LG->>CA: coder_agent(state_with_refinements, writer)
    CA->>LLM: Apply all refinements to agent
    
    loop Stream Refined Agent
        LLM-->>CA: Stream refined code
        CA-->>API: Write response
        API-->>UI: Stream to user
    end
    
    CA-->>LG: Return updated conversation
    LG->>UI: interrupt() for final feedback
    UI->>User: Display refined agent
    
    Note over User, AR: Completion Phase
    User->>UI: "This looks perfect!"
    UI->>LG: Continue with completion message
    LG->>ROA: route_user_message(state)
    ROA-->>LG: Return "finish_conversation"
    LG->>ECA: finish_conversation(state, writer)
    ECA->>LLM: Generate completion instructions
    LLM-->>ECA: Return final instructions
    ECA-->>UI: Stream final response
    UI->>User: Present completed agent with usage instructions
```

### OpenCog Knowledge Integration Flow

```mermaid
flowchart TD
    subgraph "Knowledge Input Processing"
        INPUT[User Request Input]
        PARSE[Request Parsing]
        CONCEPT[Concept Extraction]
    end
    
    subgraph "AtomSpace Operations"
        NODE_CREATE[Node Creation]
        LINK_FORM[Link Formation]
        TRUTH_ASSIGN[Truth Value Assignment]
        INDEX_UPDATE[Index Updates]
    end
    
    subgraph "Reasoning Pipeline"
        PATTERN_REG[Pattern Registration]
        REASONER_APPLY[Reasoner Application]
        RESULT_RANK[Result Ranking]
        KNOWLEDGE_MERGE[Knowledge Merging]
    end
    
    subgraph "Agent Integration"
        CONTEXT_BUILD[Context Building]
        PROMPT_ENHANCE[Prompt Enhancement]
        CODE_INFLUENCE[Code Generation Influence]
        FEEDBACK_INTEGRATE[Feedback Integration]
    end
    
    subgraph "Knowledge Output"
        SCOPE_ENHANCE[Enhanced Scope]
        RESOURCE_PRIORITIZE[Prioritized Resources]
        CODE_OPTIMIZE[Optimized Code]
        LEARNING_CAPTURE[Learning Capture]
    end
    
    INPUT --> PARSE
    PARSE --> CONCEPT
    CONCEPT --> NODE_CREATE
    
    NODE_CREATE --> LINK_FORM
    LINK_FORM --> TRUTH_ASSIGN
    TRUTH_ASSIGN --> INDEX_UPDATE
    
    INDEX_UPDATE --> PATTERN_REG
    PATTERN_REG --> REASONER_APPLY
    REASONER_APPLY --> RESULT_RANK
    RESULT_RANK --> KNOWLEDGE_MERGE
    
    KNOWLEDGE_MERGE --> CONTEXT_BUILD
    CONTEXT_BUILD --> PROMPT_ENHANCE
    PROMPT_ENHANCE --> CODE_INFLUENCE
    CODE_INFLUENCE --> FEEDBACK_INTEGRATE
    
    FEEDBACK_INTEGRATE --> SCOPE_ENHANCE
    FEEDBACK_INTEGRATE --> RESOURCE_PRIORITIZE
    FEEDBACK_INTEGRATE --> CODE_OPTIMIZE
    FEEDBACK_INTEGRATE --> LEARNING_CAPTURE
    
    classDef input fill:#e8f5e8
    classDef atomspace fill:#e3f2fd
    classDef reasoning fill:#fff3e0
    classDef integration fill:#fce4ec
    classDef output fill:#f3e5f5
    
    class INPUT,PARSE,CONCEPT input
    class NODE_CREATE,LINK_FORM,TRUTH_ASSIGN,INDEX_UPDATE atomspace
    class PATTERN_REG,REASONER_APPLY,RESULT_RANK,KNOWLEDGE_MERGE reasoning
    class CONTEXT_BUILD,PROMPT_ENHANCE,CODE_INFLUENCE,FEEDBACK_INTEGRATE integration
    class SCOPE_ENHANCE,RESOURCE_PRIORITIZE,CODE_OPTIMIZE,LEARNING_CAPTURE output
```

## Data Propagation Pathways

### Vector Database Integration

```mermaid
sequenceDiagram
    participant CA as Coder Agent
    participant VDB as Vector Database
    participant EMB as Embedding Service
    participant DOC as Documentation Store
    
    Note over CA, DOC: Documentation Retrieval Process
    CA->>VDB: query_documentation(user_request)
    VDB->>EMB: generate_embedding(query_text)
    EMB-->>VDB: Return query embedding
    VDB->>DOC: similarity_search(embedding, k=10)
    DOC-->>VDB: Return matching chunks
    VDB-->>CA: Return ranked documentation
    
    Note over CA, DOC: Context Enhancement
    CA->>CA: combine_context(docs, scope, advisor_output)
    CA->>LLM: generate_code(enhanced_context)
```

### Tool Library Interaction

```mermaid
graph LR
    subgraph "Tool Discovery"
        AA[Advisor Agent]
        FS[File System Scanner]
        CAT[Categorizer]
    end
    
    subgraph "Relevance Computation"
        AS[AtomSpace Reasoner]
        SCORE[Scoring Engine]
        RANK[Ranking System]
    end
    
    subgraph "Tool Integration"
        SEL[Tool Selector]
        TEMPL[Template Processor]
        CODE[Code Injector]
    end
    
    subgraph "Quality Assurance"
        VAL[Validator]
        TEST[Tester]
        OPT[Optimizer]
    end
    
    AA --> FS
    FS --> CAT
    CAT --> AS
    
    AS --> SCORE
    SCORE --> RANK
    RANK --> SEL
    
    SEL --> TEMPL
    TEMPL --> CODE
    CODE --> VAL
    
    VAL --> TEST
    TEST --> OPT
    OPT --> AA
    
    classDef discovery fill:#e8f5e8
    classDef computation fill:#e3f2fd
    classDef integration fill:#fff3e0
    classDef quality fill:#fce4ec
    
    class AA,FS,CAT discovery
    class AS,SCORE,RANK computation
    class SEL,TEMPL,CODE integration
    class VAL,TEST,OPT quality
```

## Recursive Feedback Mechanisms

### Learning Loop Implementation

```mermaid
stateDiagram-v2
    [*] --> InitialRequest: User submits request
    
    InitialRequest --> KnowledgeRetrieval: Query existing knowledge
    
    state KnowledgeRetrieval {
        [*] --> AtomSpaceQuery: Search AtomSpace
        AtomSpaceQuery --> RelevanceCheck: Check relevance
        RelevanceCheck --> ContextBuilding: Build context
        ContextBuilding --> [*]
    }
    
    KnowledgeRetrieval --> AgentGeneration: Generate agent
    
    state AgentGeneration {
        [*] --> ScopeCreation: Create scope
        ScopeCreation --> ResourceSelection: Select resources
        ResourceSelection --> CodeSynthesis: Synthesize code
        CodeSynthesis --> [*]
    }
    
    AgentGeneration --> UserFeedback: Present to user
    
    state UserFeedback {
        [*] --> FeedbackAnalysis: Analyze feedback
        FeedbackAnalysis --> SuccessPattern: Success pattern
        FeedbackAnalysis --> ImprovementNeeded: Needs improvement
        SuccessPattern --> LearningCapture: Capture success
        ImprovementNeeded --> RefinementTrigger: Trigger refinement
        LearningCapture --> [*]
        RefinementTrigger --> [*]
    }
    
    UserFeedback --> KnowledgeUpdate: Update knowledge base
    
    state KnowledgeUpdate {
        [*] --> PatternExtraction: Extract patterns
        PatternExtraction --> AtomSpaceUpdate: Update AtomSpace
        AtomSpaceUpdate --> IndexRebuild: Rebuild indices
        IndexRebuild --> [*]
    }
    
    KnowledgeUpdate --> AgentGeneration: Improve next iteration
    KnowledgeUpdate --> [*]: Complete cycle
```

### Adaptive Attention Allocation

```mermaid
graph TD
    subgraph "Attention Sources"
        UF[User Feedback]
        SP[Success Patterns]
        FP[Failure Patterns]
        RP[Resource Patterns]
    end
    
    subgraph "Attention Processing"
        WC[Weight Calculator]
        PA[Priority Allocator]
        DM[Dynamic Modifier]
    end
    
    subgraph "Resource Allocation"
        TA[Task Allocator]
        RA[Resource Allocator]
        PA_PROC[Process Allocator]
    end
    
    subgraph "Feedback Integration"
        PM[Performance Monitor]
        AM[Adjustment Mechanism]
        LC[Learning Consolidator]
    end
    
    UF --> WC
    SP --> WC
    FP --> WC
    RP --> WC
    
    WC --> PA
    PA --> DM
    DM --> TA
    
    TA --> RA
    RA --> PA_PROC
    PA_PROC --> PM
    
    PM --> AM
    AM --> LC
    LC --> WC
    
    classDef sources fill:#e8f5e8
    classDef processing fill:#e3f2fd
    classDef allocation fill:#fff3e0
    classDef feedback fill:#fce4ec
    
    class UF,SP,FP,RP sources
    class WC,PA,DM processing
    class TA,RA,PA_PROC allocation
    class PM,AM,LC feedback
```

## Emergent Pattern Recognition

### Pattern Evolution Tracking

```mermaid
timeline
    title Agent Generation Pattern Evolution
    
    section Initial State
        Basic Request : Single agent request
                    : Simple LLM generation
                    : No context awareness
    
    section Knowledge Integration
        Context Building : Documentation retrieval
                        : Tool library scanning
                        : Example identification
    
    section OpenCog Enhancement
        Cognitive Substrate : AtomSpace integration
                           : Reasoning mechanisms
                           : Knowledge representation
    
    section Adaptive Learning
        Pattern Recognition : Success pattern capture
                           : Failure analysis
                           : Resource optimization
    
    section Emergent Intelligence
        Predictive Behavior : Anticipatory resource selection
                           : Context-aware generation
                           : Self-improving workflows
```

### Hypergraph Pattern Discovery

```mermaid
graph TB
    subgraph "Pattern Input Sources"
        UR[User Requests]
        AR[Agent Results]
        FR[Feedback Responses]
        SR[Success Rates]
    end
    
    subgraph "Hypergraph Analysis"
        NE[Node Extraction]
        RE[Relationship Extraction]
        HF[Hypergraph Formation]
        PA[Pattern Analysis]
    end
    
    subgraph "Pattern Types"
        SP[Structural Patterns]
        BP[Behavioral Patterns]
        TP[Temporal Patterns]
        CP[Causal Patterns]
    end
    
    subgraph "Pattern Application"
        PR[Pattern Recognition]
        PS[Pattern Selection]
        PO[Pattern Optimization]
        PI[Pattern Integration]
    end
    
    UR --> NE
    AR --> RE
    FR --> HF
    SR --> PA
    
    NE --> SP
    RE --> BP
    HF --> TP
    PA --> CP
    
    SP --> PR
    BP --> PS
    TP --> PO
    CP --> PI
    
    PR --> UR
    PS --> AR
    PO --> FR
    PI --> SR
    
    classDef input fill:#e8f5e8
    classDef analysis fill:#e3f2fd
    classDef patterns fill:#fff3e0
    classDef application fill:#fce4ec
    
    class UR,AR,FR,SR input
    class NE,RE,HF,PA analysis
    class SP,BP,TP,CP patterns
    class PR,PS,PO,PI application
```

## Performance Optimization Pathways

### Parallel Processing Architecture

```mermaid
graph TD
    subgraph "Request Processing Pool"
        RP1[Request Processor 1]
        RP2[Request Processor 2]
        RP3[Request Processor 3]
        RPn[Request Processor N]
    end
    
    subgraph "Shared Resources"
        SAS[Shared AtomSpace]
        SVD[Shared Vector DB]
        STL[Shared Tool Library]
    end
    
    subgraph "Coordination Layer"
        LB[Load Balancer]
        SC[Session Controller]
        RC[Resource Coordinator]
    end
    
    subgraph "Optimization Services"
        CC[Computation Cache]
        MC[Memory Controller]
        PC[Performance Counter]
    end
    
    LB --> RP1
    LB --> RP2
    LB --> RP3
    LB --> RPn
    
    RP1 <--> SAS
    RP2 <--> SVD
    RP3 <--> STL
    RPn <--> SAS
    
    SC --> LB
    RC --> SAS
    RC --> SVD
    RC --> STL
    
    CC --> RP1
    MC --> SAS
    PC --> LB
    
    classDef processing fill:#e8f5e8
    classDef shared fill:#e3f2fd
    classDef coordination fill:#fff3e0
    classDef optimization fill:#fce4ec
    
    class RP1,RP2,RP3,RPn processing
    class SAS,SVD,STL shared
    class LB,SC,RC coordination
    class CC,MC,PC optimization
```

This comprehensive workflow documentation captures the recursive implementation pathways and emergent cognitive patterns that arise from the hypergraph-centric architecture, demonstrating how adaptive attention allocation mechanisms enable distributed cognition across the Archon agent ecosystem.