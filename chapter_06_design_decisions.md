# Chapter 6: Key Design Decisions

> *"There are no solutions. There are only trade-offs."*
> — Thomas Sowell

---

Every non-trivial design decision in software involves choosing between competing goods. This chapter explains the significant design choices made in the LLM Engineering Optimization Lab — not just *what* was chosen, but *why*, and what was given up in exchange.

---

## 6.1 Why Ollama Instead of Cloud APIs?

### The Decision

The lab uses Ollama for local LLM inference rather than the OpenAI, Anthropic, or Google APIs.

### The Case For This Decision

**Cost-zero experimentation:** During development and learning, you run hundreds or thousands of experiments. At even $0.001/request, costs accumulate quickly. At $0/request with Ollama, you can iterate freely.

**No data exposure:** Every prompt you send to a cloud API is potentially logged, used for training (depending on agreements), and subject to data retention policies. Local inference keeps all data on your machine.

**No rate limits:** Cloud APIs have rate limits that slow down high-throughput experiments. Local inference has no such ceiling (just hardware limits).

**Forcing function for efficiency:** Because local models are generally less capable than frontier models like GPT-4, using them forces you to write better prompts and build better systems. If your system works well on llama3, it will work *great* on GPT-4.

### The Case Against (What Was Given Up)

| Limitation | Mitigation |
|---|---|
| Local models are less capable | Design prompts that work with smaller models |
| Hardware requirements (8GB+ RAM) | Most developer machines qualify |
| First setup takes time | One-time cost; 15-minute setup |
| No access to GPT-4V, DALL-E, etc. | Use cloud APIs for specific multimodal tasks |

### The Deeper Design Principle

This decision reflects a broader philosophy: **develop locally, deploy globally.** The patterns learned with local models transfer directly to cloud APIs. Once you've learned to optimize prompt efficiency with llama3, you'll save real money when you switch to GPT-4.

```python
# This code works identically with Ollama and OpenAI
class LLMClient:
    def __init__(self, backend="ollama"):
        self.backend = backend
    
    def generate(self, prompt: str, model: str = "llama3") -> str:
        if self.backend == "ollama":
            return self._call_ollama(prompt, model)
        elif self.backend == "openai":
            return self._call_openai(prompt, model)
```

---

## 6.2 Why Notebooks Instead of a Package?

### The Decision

The lab is structured as independent Jupyter notebooks, not as a Python package with importable modules.

### The Case For This Decision

**Transparency:** In notebooks, every cell shows its output. You can see exactly what each operation produces. This is invaluable for learning and debugging LLM systems, where the output is often surprising.

**Iterative exploration:** LLM engineering is inherently experimental. You need to try a prompt, see the response, modify it, and try again. Notebooks excel at this; Python packages don't.

**Self-documentation:** A well-written notebook is simultaneously code AND documentation. The `#` cells explain what the code does; the output cells show proof it worked.

**Lower barrier to entry:** Not everyone who wants to understand LLM engineering is ready to navigate a complex Python package. Notebooks meet learners where they are.

### The Case Against

| Limitation | Mitigation |
|---|---|
| Not importable by other projects | Chapter 9 shows how to extract to modules |
| State can get corrupted (run cells out of order) | Each notebook designed to run top-to-bottom |
| Testing is harder | Integration tests demonstrate correctness in-notebook |
| Not deployable as-is | Notebooks are the *reference*; production code is separate |

### The Migration Path

The lab is designed with *eventual productionization* in mind. Each notebook's core classes can be extracted to Python modules with minimal changes:

```
NOTEBOOK:                    PRODUCTION:
┌────────────────────┐       ┌────────────────────┐
│ class CostTracker: │  →    │ # cost_tracker.py  │
│     ...            │       │ class CostTracker: │
│                    │       │     ...            │
└────────────────────┘       └────────────────────┘
```

---

## 6.3 The Tiered Memory Decision

### The Decision

Use a three-tier memory hierarchy (RAM → Vector DB → LLM) rather than a single cache layer or no cache at all.

### Why Three Tiers?

**Single tier is insufficient because:** Different queries have different retrieval needs. An exact match is handled perfectly by a hash map. A semantically similar query needs vector search. A novel query needs generation. One tier can't handle all three cases optimally.

```
QUERY TYPE         | OPTIMAL TIER    | WHY
─────────────────────────────────────────────────────────
"capital of France"| L1 (RAM)        | Exact match, seen before
"France's capital" | L2 (vector)     | Same meaning, different words
"Largest city in   | L3 (LLM)        | Novel — never asked before
 the Île-de-France │                 │ AND slightly wrong (it's Paris
 region?"          │                 │ but context matters)
```

**Two tiers would miss the semantic middle ground:** An L1 (exact) + L3 (LLM) system would miss all the cases where a semantically similar answer exists. Given that users rephrase the same questions constantly, the L2 layer captures a significant portion of requests.

### Cache Invalidation

One famous problem: "There are only two hard things in computer science: cache invalidation and naming things."

The lab uses a **TTL (Time-To-Live)** approach:

```python
class TieredMemoryCache:
    DEFAULT_TTL = {
        "factual": 86400 * 7,    # 7 days for facts (stable)
        "current_events": 3600,  # 1 hour for news (volatile)
        "computation": 86400,    # 1 day for calculations
        "opinion": 3600,         # 1 hour for opinions (model may change)
    }
```

---

## 6.4 Choosing an Evaluation Strategy

### The Decision

Use **LLM-as-judge** rather than traditional NLP metrics (BLEU, ROUGE) or pure human evaluation.

### Why Not Traditional Metrics?

BLEU and ROUGE measure token overlap between a generated response and a reference answer. They're great for machine translation but terrible for open-ended LLM evaluation:

```
QUESTION: "What's the capital of France?"
GROUND TRUTH: "Paris"
LLM ANSWER: "The capital of France is Paris."

BLEU score: ~0.2 (poor — different words)
Human score: 1.0 (perfect — correct answer)
```

### Why Not Pure Human Evaluation?

Human evaluation is the gold standard, but it doesn't scale:
- 10,000 requests/day × human evaluation = prohibitively expensive
- Human evaluators have inconsistent standards
- Latency: humans can't evaluate in real-time

### Why LLM-as-Judge Works

A second LLM evaluating outputs is:
- **Consistent**: Same rubric every time
- **Scalable**: Runs automatically on every response
- **Fast**: Local model evaluation ~100ms
- **Surprisingly accurate**: Correlates 0.85+ with human judges (per academic benchmarks)

```
LLM JUDGE SCORE CORRELATION WITH HUMANS:
┌─────────────────────────────────────────┐
│                                         │
│ Human-Human Agreement:       ~0.80      │
│ (humans don't even agree)               │
│                                         │
│ LLM Judge vs. Human:         ~0.78-0.87 │
│ (comparable to human-human!)            │
│                                         │
│ Traditional NLP vs. Human:   ~0.30-0.50 │
│ (poor for open-ended responses)         │
└─────────────────────────────────────────┘
```

---

## 6.5 Stateless vs. Stateful Agents

### The Decision

Agents in the lab are **stateless by default** with **explicit state management**.

This means: agents don't automatically remember previous conversations. State must be explicitly passed in and out.

### Why Stateless?

**Predictability:** Stateful agents behave differently depending on history in ways that are hard to debug. Stateless agents with explicit state are easier to reason about.

**Testability:** You can test a stateless agent by giving it any state and checking its output. Testing a stateful agent requires reproducing the exact sequence of interactions that led to that state.

**Scalability:** Stateless agents can run on any server in a cluster. Stateful agents are tied to the server (or storage) holding their state.

**Crash recovery:** If a stateless agent crashes, the next request starts fresh with the provided state. If a stateful agent crashes, its state may be lost.

### The Explicit State Pattern

```python
# IMPLICIT STATE (antipattern - don't do this)
class BadAgent:
    def __init__(self):
        self.memory = []  # Hidden state
    
    def chat(self, message: str) -> str:
        self.memory.append(message)  # State mutation hidden from caller
        return self.llm.generate(str(self.memory))

# EXPLICIT STATE (recommended pattern)
class GoodAgent:
    def chat(self, message: str, 
             history: list) -> tuple[str, list]:
        # State comes IN as parameter
        new_history = history + [{"role": "user", "content": message}]
        response = self.llm.generate(new_history)
        new_history.append({"role": "assistant", "content": response})
        # State goes OUT as return value
        return response, new_history
```

---

## 6.6 The MCP Protocol Choice

### The Decision

Use the **Model Context Protocol (MCP)** for tool definitions rather than a custom JSON schema.

### Why Standardize on MCP?

**Portability:** Tools defined in MCP format work across different LLM providers. A tool you build for llama3 works with GPT-4 and Claude without modification.

**Ecosystem:** The MCP ecosystem is growing rapidly. Standardizing means you can use community-built tools rather than building everything from scratch.

**Validation:** MCP provides schemas that validate tool inputs/outputs. This catches integration errors early.

**Introspection:** MCP tools are self-describing. Agents can ask "what tools do you have?" and get machine-readable, actionable answers.

### MCP vs. Custom Tool Format

```python
# CUSTOM FORMAT (brittle, non-portable)
def search(query):
    """Search the web. Pass query as a string."""
    return web_search(query)

# MCP FORMAT (standard, portable, validated)
SEARCH_TOOL = {
    "name": "web_search",
    "description": "Search the web for current information. "
                   "Use this when you need facts not in training data.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Specific search query (3-8 words for best results)",
                "maxLength": 500
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results to return",
                "default": 5,
                "minimum": 1,
                "maximum": 20
            }
        },
        "required": ["query"]
    }
}
```

---

## 6.7 Summary

The design decisions in this lab reflect a consistent set of values:

| Value | Expressed As |
|---|---|
| Learnability | Notebooks over packages |
| Cost-consciousness | Local inference, smart routing |
| Predictability | Stateless agents, explicit state |
| Measurability | Continuous evaluation on every request |
| Portability | MCP standard over custom schemas |
| Efficiency | Three-tier caching, memory compression |

None of these decisions are "right" in the abstract. They're appropriate for a *learning and optimization lab* that needs to be approachable, affordable, and rigorous. Production systems built from this lab might make different choices in specific areas — and Chapter 9 addresses exactly those evolution paths.


---

## 6.7 Inference Infrastructure — The Serving Stack Decision

### The Three Paths

```
PATH 1: MANAGED CLOUD API
Your App → OpenAI / Anthropic / Google API
Pros: No infrastructure, latest models, instant start
Cons: Highest cost, data sent externally, rate limits
Best for: Early stage, low volume, privacy-insensitive

PATH 2: SELF-HOSTED ON CLOUD GPU
Your App → vLLM / TGI on AWS/GCP/Azure GPU instance
Pros: Full control, lower cost at scale, data stays in your cloud
Cons: GPU management, model deployment complexity
Best for: Mid-to-large scale, need cost control

PATH 3: ON-PREMISES / EDGE (the lab's choice)
Your App → Ollama / llama.cpp on your hardware
Pros: Zero data egress, lowest latency, no per-token cost
Cons: Hardware capex, model size limited by local RAM/GPU
Best for: Regulated industries, air-gapped environments, development
```

### Quantisation: Fitting Large Models on Limited Hardware

Quantisation reduces model precision (e.g., 32-bit → 4-bit), dramatically reducing memory at a small quality cost:

```
MODEL SIZE (Llama 3 8B):
FP32    → ~32 GB VRAM │ No quality loss   │ Baseline speed
FP16    → ~16 GB VRAM │ Negligible loss   │ 1.5× faster
INT8    →  ~8 GB VRAM │ Minimal loss      │ 2× faster
INT4 (Q4_K_M) → ~4 GB │ Noticeable loss   │ 3× faster  ← practical sweet spot
```

```python
# Using quantised models with Ollama
# Pull quantised model (4-bit = ~4.7 GB instead of 16 GB)
# ollama pull llama3:8b-instruct-q4_K_M

response = requests.post("http://localhost:11434/api/chat",
    json={"model": "llama3:8b-instruct-q4_K_M",
          "messages": [{"role": "user", "content": "Explain quantisation"}],
          "stream": False})
```

---

## 6.8 Multi-Tenancy — Isolation by Design

When serving multiple customers or teams, isolation must be designed in from the start, not bolted on later.

### What Must Be Isolated

```
DATA ISOLATION:    Tenant A cannot see Tenant B's history or vector store
BUDGET ISOLATION:  Each tenant has its own token/cost budget
RATE ISOLATION:    One tenant's heavy use cannot starve others
MODEL ISOLATION:   Different tenants may get different system prompts or models
```

### The Isolation Mechanism

```python
class MultiTenantService:
    """
    Isolation via scoped keys — tenant ID injected at every layer.
    """
    def process(self, message, tenant_id, user_id, session_id):
        # Session key includes tenant_id → prevents cross-tenant access
        session_key = f"tenant:{tenant_id}:user:{user_id}:{session_id}"

        # Vector search scoped to tenant's namespace
        docs = self.vector_store.search(
            query=message,
            namespace=f"tenant-{tenant_id}"  # never crosses tenant boundary
        )

        # Gateway call with tenant as caller_id for budget attribution
        response = self.gateway.complete(
            messages=self._build_messages(message, docs),
            caller_id=f"tenant:{tenant_id}",
            model=self.tenant_config(tenant_id).preferred_model
        )
        return response
```

---

## 6.9 Versioning and A/B Testing — Safe Change Management

LLM systems change constantly: new models, new prompts, new RAG configurations. Without versioning and testing, every change is a gamble.

### Three Things to Version

```
1. PROMPT VERSIONS       "Be concise" added → 15% shorter responses, 8% quality drop?
2. MODEL VERSIONS        gpt-4-turbo → gpt-4o → different behaviour even with same prompt
3. RAG CONFIGURATION     chunk_size=500 → chunk_size=300 → better precision, lower recall?
```

### The A/B Testing Pattern

```python
class LLMABTester:
    def get_config_for_request(self, experiment: str, user_id: str):
        """
        Deterministic assignment: same user always gets same variant.
        WHY: Random per-request creates inconsistent UX.
        """
        hash_val = int(hashlib.md5(f"{experiment}:{user_id}".encode()).hexdigest(), 16)
        hash_val = hash_val / (2**128)

        exp = self.experiments[experiment]
        if hash_val < exp["traffic_split"]:
            return exp["treatment"], "treatment"
        return exp["control"], "control"

    def get_results(self, experiment: str) -> dict:
        """Compute statistical significance of quality difference."""
        from scipy import stats
        exp = self.experiments[experiment]
        ctrl = [r["quality"] for r in exp["results"]["control"]]
        trt  = [r["quality"] for r in exp["results"]["treatment"]]

        if len(ctrl) < 30 or len(trt) < 30:
            return {"status": "insufficient_data", "min_required": 30}

        _, p_value = stats.ttest_ind(ctrl, trt)
        improvement = (sum(trt)/len(trt)) - (sum(ctrl)/len(ctrl))

        return {
            "improvement": round(improvement, 4),
            "p_value": round(p_value, 4),
            "statistically_significant": p_value < 0.05,
            "recommendation": "deploy_treatment" if (p_value < 0.05 and improvement > 0)
                              else "keep_control"
        }
```

---

## 6.10 Summary

Updated design decision values:

| Value | Expressed As |
|---|---|
| Learnability | Notebooks over packages |
| Cost-consciousness | Local inference, smart routing, quantisation |
| Predictability | Stateless agents, explicit state |
| Measurability | Continuous evaluation + A/B testing |
| Portability | MCP standard over custom schemas |
| Efficiency | Three-tier caching, memory compression |
| Safety | Multi-layer guardrails, prompt injection defence |
| Isolation | Tenant-scoped keys and vector namespaces |
| Change safety | Shadow mode, canary, A/B testing before full rollout |

---

*Next: [Chapter 7 — Scaling, Trade-offs, and Performance →](./chapter_07_scaling.md)*
