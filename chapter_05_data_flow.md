# Chapter 5: Data Flow & Execution Flow

> *"Understanding the path data travels through your system is the first step to understanding where it can go wrong."*

---

## 5.1 The Request Lifecycle

Before diving into specific flows, let's trace what happens to a single user request from the moment it enters the system to the moment a response returns.

```
╔══════════════════════════════════════════════════════════════════╗
║                    COMPLETE REQUEST LIFECYCLE                    ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  T=0ms   User sends: "What is the capital of France?"           ║
║                                │                                ║
║  T=1ms   ┌─────────────────────▼────────────────────────────┐  ║
║          │ SECURITY LAYER: Prompt Injection Defense          │  ║
║          │ • Pattern match: No injection detected ✓          │  ║
║          │ • LLM classifier: Score 0.02 (safe) ✓             │  ║
║          └─────────────────────┬────────────────────────────┘  ║
║                                │ Clean input                    ║
║  T=2ms   ┌─────────────────────▼────────────────────────────┐  ║
║          │ CACHE LAYER: TieredMemoryCache                    │  ║
║          │ • L1 check: cache hit! ✓                          │  ║
║          └─────────────────────┬────────────────────────────┘  ║
║                                │ Cached answer: "Paris"         ║
║  T=3ms   Response returned to user                             ║
║                                                                 ║
║  ─────────────── OR (on cache miss) ────────────────────────── ║
║                                                                 ║
║  T=2ms   ┌─────────────────────▼────────────────────────────┐  ║
║          │ MEMORY LAYER: ConversationMemoryManager           │  ║
║          │ • Retrieve relevant conversation history          │  ║
║          │ • Apply compression if history > threshold        │  ║
║          └─────────────────────┬────────────────────────────┘  ║
║                                │ Enriched context               ║
║  T=5ms   ┌─────────────────────▼────────────────────────────┐  ║
║          │ ROUTING LAYER: ModelRouter                        │  ║
║          │ • Complexity score: 0.1 (simple factual query)   │  ║
║          │ • Selected model: llama3 (local, free)            │  ║
║          └─────────────────────┬────────────────────────────┘  ║
║                                │ Routed to llama3               ║
║  T=6ms   ┌─────────────────────▼────────────────────────────┐  ║
║          │ LLM INFERENCE: Ollama (llama3)                    │  ║
║          │ • Generating response...                          │  ║
║          └─────────────────────┬────────────────────────────┘  ║
║                                │ Raw response: "Paris"          ║
║  T=200ms ┌─────────────────────▼────────────────────────────┐  ║
║          │ EVALUATION LAYER: ContinuousEvaluator             │  ║
║          │ • Quality score: 0.97 ✓                           │  ║
║          │ • Stored in eval_history                          │  ║
║          └─────────────────────┬────────────────────────────┘  ║
║                                │ Validated response             ║
║  T=250ms ┌─────────────────────▼────────────────────────────┐  ║
║          │ COST TRACKING: CostTracker                        │  ║
║          │ • Logged: model=llama3, cost=$0.00, 15 tokens     │  ║
║          └─────────────────────┬────────────────────────────┘  ║
║                                │                                ║
║  T=251ms Response returned to user: "Paris"                    ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 5.2 Data Flow in the Cost Optimization Pipeline

The cost optimization pipeline's data flow is centered on two transformations: **token counting** and **model selection**.

```
INPUT: User Query + Context
       │
       ▼
┌─────────────────────────────────────────────────┐
│          TOKEN COUNTING STAGE                   │
│                                                 │
│  tokenize(query + context + system_prompt)      │
│  → input_token_count: int                       │
│  → estimated_output_tokens: int (heuristic)     │
│  → total_estimated_tokens: int                  │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│          COMPLEXITY SCORING STAGE               │
│                                                 │
│  score_complexity(query, context)               │
│  → complexity_signals: dict                     │
│     • length_score: 0.0 - 1.0                  │
│     • reasoning_score: 0.0 - 1.0               │
│     • domain_score: 0.0 - 1.0                  │
│  → final_complexity: max(signals)               │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│          MODEL ROUTING STAGE                    │
│                                                 │
│  complexity < 0.3  → llama3 (local, $0.00)      │
│  complexity < 0.6  → gpt-4o-mini ($0.001)       │
│  complexity ≥ 0.6  → gpt-4o ($0.015)            │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│          INFERENCE STAGE                        │
│                                                 │
│  response = llm_call(selected_model, prompt)    │
│  → response_text: str                           │
│  → actual_input_tokens: int                     │
│  → actual_output_tokens: int                    │
│  → latency_ms: float                           │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│          COST RECORDING STAGE                   │
│                                                 │
│  tracker.track_call(model, in_tok, out_tok, lat)│
│  → CostRecord stored in call_history            │
│  → Cumulative totals updated                    │
│  → Alert if daily budget exceeded               │
└─────────────────────────────────────────────────┘

OUTPUT: Response text + CostRecord
```

---

## 5.3 Data Flow in the RAG Pipeline

The RAG pipeline has two distinct phases with very different data flows:

### Phase 1: Indexing (Offline, One-Time)

```
DOCUMENTS (PDFs, text files, URLs)
    │
    ▼
┌─────────────────────────────────────────┐
│           CHUNKING STAGE                │
│                                         │
│  Input:  document_text (potentially     │
│          100,000+ tokens)               │
│                                         │
│  Process:                               │
│  1. Split by sentence boundaries        │
│  2. Merge into chunks of ~500 tokens    │
│  3. Add overlap (50 tokens) between     │
│     adjacent chunks for context         │
│                                         │
│  Output: chunks[] (list of strings)     │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│           EMBEDDING STAGE               │
│                                         │
│  For each chunk:                        │
│  embed(chunk) → vector[768]             │
│  (768-dimensional float array)          │
│                                         │
│  These numbers capture semantic         │
│  meaning — similar text has similar     │
│  vectors (high cosine similarity)       │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│           VECTOR STORE STAGE            │
│                                         │
│  Store: (vector, chunk_text, metadata)  │
│  Indexed for fast nearest-neighbor      │
│  search (FAISS, Chroma, Pinecone, etc.) │
└─────────────────────────────────────────┘
```

### Phase 2: Query (Online, Per-Request)

```
USER QUERY: "What are Python decorators?"
    │
    ▼
embed(query) → query_vector[768]
    │
    ▼
┌──────────────────────────────────────────┐
│         SIMILARITY SEARCH                │
│                                          │
│  For all stored vectors:                 │
│  similarity = cosine(query_vec, doc_vec) │
│                                          │
│  Return top-K most similar chunks        │
│  (typically K=3-5)                       │
└──────────────────┬───────────────────────┘
                   │ Top K chunks retrieved
                   ▼
┌──────────────────────────────────────────┐
│         CONTEXT INJECTION                │
│                                          │
│  prompt = f"""                           │
│  Using the following context:            │
│  {retrieved_chunks}                      │
│                                          │
│  Answer this question: {user_query}      │
│  """                                     │
└──────────────────┬───────────────────────┘
                   │
                   ▼
              LLM INFERENCE
                   │
                   ▼
              RESPONSE TO USER
```

---

## 5.4 Data Flow in the Agent Loop

The agent's ReAct loop has a cyclical data flow that continues until a terminal condition is reached:

```
START: User Goal
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                   AGENT STATE                           │
│  {                                                      │
│    goal: "Research AI trends and write a summary",      │
│    step_count: 0,                                       │
│    history: [],                                         │
│    tools_available: ["web_search", "read_file",         │
│                       "run_code", "write_file"]         │
│  }                                                      │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │  THINK STAGE   │◄──────────────────────────┐
              │                │                           │
              │ LLM receives:  │                           │
              │ • Current goal │                           │
              │ • History      │                           │
              │ • Available    │                           │
              │   tools        │                           │
              │                │                           │
              │ LLM outputs:   │                           │
              │ • Thought      │                           │
              │ • Action       │                           │
              │   (tool call)  │                           │
              └────────┬───────┘                           │
                       │                                   │
                       ▼                                   │
              ┌────────────────┐                           │
              │   ACT STAGE    │                           │
              │                │                           │
              │ Execute the    │                           │
              │ chosen tool    │                           │
              │ with given     │                           │
              │ parameters     │                           │
              └────────┬───────┘                           │
                       │                                   │
                       ▼                                   │
              ┌────────────────┐                           │
              │ OBSERVE STAGE  │                           │
              │                │                           │
              │ Tool result    │                           │
              │ added to       │                           │
              │ history        │                           │
              └────────┬───────┘                           │
                       │                                   │
                       ▼                                   │
              ┌────────────────┐                           │
              │ TERMINAL CHECK │                           │
              │                │                           │
              │ Is goal        │                           │
              │ achieved? OR   ├── No ─────────────────────┘
              │ Max steps hit? │
              └────────┬───────┘
                       │ Yes
                       ▼
                 FINAL ANSWER
```

### State at Each Stage

The agent state object grows with each loop iteration:

```python
# After step 1:
state = {
    "goal": "Research AI trends and write a summary",
    "step_count": 1,
    "history": [
        {"type": "think", "content": "I need to search for recent AI trends"},
        {"type": "action", "tool": "web_search", "args": {"query": "AI trends 2024"}},
        {"type": "observe", "result": "Found 10 articles about LLMs, multimodal AI..."}
    ]
}

# After step 3:
state = {
    "goal": "Research AI trends and write a summary",  
    "step_count": 3,
    "history": [
        # ... 3 think/act/observe cycles ...
    ],
    # Token count is now 3× larger than after step 1
    # Memory management kicks in here to compress old steps
}
```

---

## 5.5 Data Flow in Multi-Agent Systems

Multi-agent systems have a more complex data flow because information moves between agents:

```
USER TASK: "Research quantum computing advances and create a report"
    │
    ▼
┌────────────────────────────────────────────────────────────────┐
│                      ORCHESTRATOR                              │
│                                                                │
│  1. Parse task                                                │
│  2. Create execution plan:                                    │
│     Step 1: ResearchAgent → gather information                │
│     Step 2: AnalysisAgent → analyze & synthesize             │
│     Step 3: WritingAgent → format as report                   │
└──────────────────┬─────────────────────────────────────────────┘
                   │
     ┌─────────────┴──────────────────────┐
     │                                    │
     ▼                                    │
┌─────────────┐                          │
│  RESEARCH   │                          │
│   AGENT     │                          │
│             │                          │
│ web_search  │                          │
│ → results[] │                          │
│             │                          │
│ Output:     │                          │
│ research_   │                          │
│ notes.txt   │                          │
└──────┬──────┘                          │
       │ Research complete               │
       ▼                                 │
┌─────────────┐                          │
│  ANALYSIS   │                          │
│   AGENT     │                          │
│             │                          │
│ Reads:      │◄─────────────────────────┘
│ research_   │ (also receives original task context)
│ notes.txt   │
│             │
│ Synthesizes │
│ key themes  │
│             │
│ Output:     │
│ analysis.   │
│ json        │
└──────┬──────┘
       │ Analysis complete
       ▼
┌─────────────┐
│  WRITING    │
│   AGENT     │
│             │
│ Reads:      │
│ analysis.   │
│ json        │
│             │
│ Formats as  │
│ readable    │
│ report      │
│             │
│ Output:     │
│ report.md   │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│                       ORCHESTRATOR                              │
│                                                                 │
│  All steps complete → synthesize final response → return        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5.6 The Evaluation Data Pipeline

The evaluation pipeline runs in parallel with every request, collecting quality data over time:

```
Every request produces:
┌──────────────────────────────────────────────────────────────┐
│  EvalRecord {                                                │
│    request_id: "abc-123",                                    │
│    timestamp: "2024-01-15T14:30:00",                        │
│    question: "What is the capital of France?",              │
│    response: "Paris",                                        │
│    model: "llama3",                                          │
│    scores: {                                                 │
│      accuracy: 1.0,                                          │
│      relevance: 1.0,                                         │
│      completeness: 0.8,  // didn't say "Île-de-France"      │
│      clarity: 1.0                                            │
│    },                                                        │
│    overall: 0.96,                                            │
│    latency_ms: 180                                           │
│  }                                                           │
└──────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────┐
│                EVAL HISTORY STORE                            │
│  (growing list of EvalRecords)                               │
│                                                              │
│  Windowed quality analysis:                                  │
│  • Rolling 20-request average                               │
│  • Trend detection (improving/degrading/stable)             │
│  • Per-model breakdowns                                      │
│  • Per-query-type breakdowns                                 │
└──────────────────────────────────────────────────────────────┘
         │
         ├──── Quality < threshold ──► ALERT
         │
         └──── Periodic reporting ──► Dashboard/Logs
```

---

## 5.7 Sequence Diagrams for Key Flows

### Sequence: Cache Hit (Best Case)

```
User        Security     Cache        CostTracker
 │               │           │              │
 │──────────────►│           │              │
 │   query       │ clean     │              │
 │               │──────────►│              │
 │               │           │◄─── L1 hit   │
 │               │           │  "Paris"     │
 │◄──────────────────────────│              │
 │   "Paris"     │           │              │
 │  (< 3ms!)     │           │              │
```

### Sequence: Cache Miss → LLM → Eval (Normal Case)

```
User     Security    Cache    Router    Ollama    Evaluator   Tracker
 │           │          │        │         │           │          │
 │──────────►│          │        │         │           │          │
 │           │─────────►│        │         │           │          │
 │           │          │ miss   │         │           │          │
 │           │          │───────►│         │           │          │
 │           │          │        │─────────►           │          │
 │           │          │        │         │ response  │          │
 │           │          │        │         │──────────►│          │
 │           │          │        │         │           │ score    │
 │           │          │        │         │           │─────────►│
 │           │          │        │         │           │          │ record
 │◄──────────────────────────────────────── response  │          │
```

### Sequence: Quality Failure → Retry

```
User     Agent    Evaluator   Fallback
 │          │         │           │
 │─────────►│         │           │
 │          │─────────►           │
 │          │    score=0.45 ✗     │
 │          │         │           │
 │          │ retry with stronger │
 │          │─────────────────────►
 │          │                     │ better response
 │          │◄────────────────────│
 │          │─────────►           │
 │          │    score=0.89 ✓     │
 │◄─────────│         │           │
```

---

## 5.8 Summary

The data flows in this system are more complex than they appear from a single notebook, but the pattern is consistent:

1. **Input** is cleaned and validated at the boundary
2. **Context** is enriched from memory/cache
3. **Routing** selects the optimal model
4. **Inference** generates the response
5. **Evaluation** validates quality
6. **Telemetry** records everything for analysis

Every optimization in the lab targets one of these stages. In the next chapter, we examine the design decisions behind these choices.


---

## 5.9 The Full Six-Stage Request Pipeline

Every production LLM request passes through six transformation stages. This is the detailed view of what happens inside the "request lifecycle" from Section 5.1.

```
INCOMING REQUEST
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1: AUTHENTICATION & RATE LIMITING           ⏱ < 5ms     │
│  • Validate API key / JWT token                                 │
│  • Check user's rate limit (requests/min, tokens/day)           │
│  • Check global circuit breaker state                           │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2: INPUT VALIDATION & GUARDRAILS            ⏱ < 50ms    │
│  • Schema validation                                            │
│  • Length limits (anti token-stuffing)                          │
│  • PII detection and redaction                                  │
│  • Prompt injection detection (regex + LLM classifier)          │
│  • Content policy check                                         │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 3: CONTEXT ASSEMBLY                         ⏱ < 100ms   │
│  • Load conversation history from session store                 │
│  • RAG: retrieve relevant documents                             │
│  • Apply memory compression if history too long                 │
│  • Inject system prompt and few-shot examples                   │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 4: COST & ROUTING DECISION                  ⏱ < 20ms    │
│  • Count tokens in assembled context                            │
│  • Check daily budget remaining                                 │
│  • Select optimal model (complexity + budget)                   │
│  • Check cache (semantic similarity)                            │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
           ┌───────────────┴───────────────┐
      Cache Hit                        Cache Miss
           │                               │
           ▼                               ▼
┌──────────────────┐       ┌─────────────────────────────────────┐
│ SERVE FROM CACHE │       │  STAGE 5: INFERENCE        ⏱ 100ms– │
│ ⏱ < 10ms         │       │  • Send to LLM (cloud or local)     │
└──────────────────┘       │  • Stream or wait for completion    │
                           │  • Handle timeout / retry           │
                           └────────────────────────────────────-┘
                                             │
                                             ▼
                          ┌──────────────────────────────────────┐
                          │  STAGE 6: OUTPUT PROCESSING ⏱ <30ms │
                          │  • Parse structured output           │
                          │  • Run output guardrails             │
                          │  • Store to cache & history          │
                          │  • Log metrics                       │
                          └──────────────────┬───────────────────┘
                                             ▼
                                       RESPONSE TO USER
```

---

## 5.10 The LLM Gateway Data Flow

At scale, all LLM calls should flow through a centralised **LLM Gateway** rather than directly from application code.

```
WITHOUT GATEWAY (antipattern at scale):
┌──────────────────────────────────────────────────────┐
│  Service A ──────────────────────► OpenAI API        │
│  Service B ──────────────────────► OpenAI API        │  Each service has its own
│  Service C ──────────────────────► Anthropic         │  auth, retry, rate limits.
│  Agent D   ──────────────────────► OpenAI API        │  No central enforcement.
└──────────────────────────────────────────────────────┘

WITH GATEWAY (recommended at scale):
┌────────────────────────────────────────────────────────────────┐
│  Service A ─┐                                                  │
│  Service B  ├──► LLM GATEWAY ──► [OpenAI / Anthropic / Local] │
│  Agent C  ──┘    • Unified auth        • Response caching      │
│                  • Rate limiting       • Full request logging   │
│                  • Budget enforcement  • Cost attribution       │
│                  • Failover between providers                   │
└────────────────────────────────────────────────────────────────┘
```

The gateway's internal data flow:

```
Request in
    │
    ├─1─► Rate limit check (Redis counter, atomic)
    │
    ├─2─► Cache lookup (semantic similarity)
    │        Hit → return cached response
    │
    ├─3─► Model selection (complexity score → routing table)
    │
    ├─4─► Budget check (per-user, per-tenant, system-wide)
    │        Over budget → downgrade to local model
    │
    ├─5─► Inference with failover chain
    │        Primary model → fallback provider → local model
    │
    ├─6─► Cache write (store response for future hits)
    │
    └─7─► Telemetry (cost, latency, quality recorded)

Response out
```

---

## 5.11 Summary

Data flows in this system follow a consistent pattern:

1. **Input** is cleaned and validated at the boundary (Stages 1–2)
2. **Context** is enriched from memory and knowledge (Stage 3)
3. **Routing** selects the optimal model (Stage 4)
4. **Inference** generates the response (Stage 5)
5. **Output** is validated and stored (Stage 6)
6. **Telemetry** records everything for analysis (runs throughout)

The LLM Gateway centralises cross-cutting concerns so that no application service needs to implement them independently.

---

*Next: [Chapter 6 — Key Design Decisions →](./chapter_06_design_decisions.md)*
