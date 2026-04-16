# Chapter 2: High-Level Architecture

> *"Architecture is the art of knowing which complexity to hide."*

---

## 2.1 The Bird's-Eye View

The LLM Engineering Optimization Lab isn't eight disconnected scripts — it's a carefully layered system. Think of it like a well-designed kitchen:

- **Infrastructure layer** (Ollama): The stove — generates heat (tokens) on demand
- **Optimization layer** (cost, memory, cache): The mise en place — ingredients prepared efficiently
- **Agent layer** (reliability, multi-agent): The chef — orchestrates components into a coherent meal
- **Quality layer** (evaluation, security): The health inspector — ensures what leaves is safe and good

### The Full Production-Grade Topology

```
╔══════════════════════════════════════════════════════════════════════════╗
║               PRODUCTION LLM SYSTEM — COMPLETE TOPOLOGY                 ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  ┌────────────────────────────────────────────────────────────────────┐  ║
║  │                        CLIENT LAYER                                │  ║
║  │    Web App │ Mobile App │ API Consumers │ Internal Tools            │  ║
║  └────────────────────────────┬───────────────────────────────────────┘  ║
║                               │  HTTPS / WebSocket                       ║
║  ┌────────────────────────────▼───────────────────────────────────────┐  ║
║  │                     CDN / EDGE LAYER                               │  ║
║  │     Rate Limiting │ DDoS Protection │ TLS Termination              │  ║
║  └────────────────────────────┬───────────────────────────────────────┘  ║
║                               │                                          ║
║  ┌────────────────────────────▼───────────────────────────────────────┐  ║
║  │                   API GATEWAY / LOAD BALANCER                      │  ║
║  │  Auth (JWT/API Key) │ Routing │ Request Throttling │ Versioning    │  ║
║  └──────┬─────────────────────────────────────────────────┬───────────┘  ║
║         │                                                 │              ║
║  ┌──────▼──────┐                                 ┌────────▼────────┐    ║
║  │  Sync API   │                                 │  Async Worker   │    ║
║  │  (FastAPI)  │                                 │  (Celery/RQ)    │    ║
║  │  <2s tasks  │                                 │  long tasks     │    ║
║  └──────┬──────┘                                 └────────┬────────┘    ║
║         │                                                 │              ║
║  ┌──────▼─────────────────────────────────────────────────▼───────────┐  ║
║  │                      MIDDLEWARE LAYER                               │  ║
║  │  Guardrails │ Prompt Builder │ Context Manager │ Cost Gate          │  ║
║  └────────────────────────────────────┬────────────────────────────────┘  ║
║                                       │                                   ║
║  ┌────────────────────────────────────▼────────────────────────────────┐  ║
║  │                      INFERENCE LAYER                                │  ║
║  │  Cloud LLM (OpenAI/Anthropic) │ Self-Hosted (vLLM/Ollama)          │  ║
║  └─────────────────────────────────────────────────────────────────────┘  ║
║                                                                           ║
║  ┌─────────────────────────────────────────────────────────────────────┐  ║
║  │                       DATA LAYER                                    │  ║
║  │  Vector DB │ Cache (Redis) │ RDBMS │ Object Store │ Message Queue   │  ║
║  └─────────────────────────────────────────────────────────────────────┘  ║
║                                                                           ║
║  ┌─────────────────────────────────────────────────────────────────────┐  ║
║  │                   OBSERVABILITY LAYER                               │  ║
║  │  Metrics (Prometheus) │ Logs (ELK) │ Traces (Jaeger) │ LLM Tracing  │  ║
║  └─────────────────────────────────────────────────────────────────────┘  ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## 2.2 The Role of Local Inference with Ollama

The entire lab is built on **Ollama**, a local LLM serving tool. Understanding why illuminates the architectural philosophy.

```python
import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={"model": "llama3", "prompt": "Explain RAG in one sentence.", "stream": False}
)
print(response.json()["response"])
```

This is nearly identical to calling the OpenAI API — except it runs on your laptop, costs $0 per token, and your data never leaves your machine.

| Consideration | Cloud API | Local (Ollama) |
|---|---|---|
| Cost per token | $0.001–$0.015 | $0.000 |
| Privacy | Data sent to provider | Data stays local |
| Latency | 200–2000ms (network) | 50–500ms (local GPU/CPU) |
| Rate limits | Hard caps, throttling | None |
| Availability | Dependent on provider | Always available |
| Fine-tuning | Expensive/restricted | Full control |

---

## 2.3 Notebook-as-Microservice: The Lab's Architectural Philosophy

Each notebook in the lab is designed as an independent, self-documenting microservice.

```
MONOLITHIC (what beginners build):
┌────────────────────────────────────────────────────┐
│  One Big LLM Application                          │
│  prompting + memory + cost + eval + security      │
│  (all tangled together, untestable)               │
└────────────────────────────────────────────────────┘

COMPOSABLE (what the lab teaches):
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│  Memory  │  │   Cost   │  │  Eval    │  │ Security │
│ Manager  │  │ Tracker  │  │ Loop     │  │ Guard    │
└────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
     └──────────────┴─────────────┴──────────────┘
                            │
                    ┌───────▼──────┐
                    │  LLM Agent   │
                    └──────────────┘
```

---

## 2.4 Component Map: Eight Notebooks, One System

```
┌──────────────────────────────────────────────────────────────────────┐
│                        NOTEBOOK COMPONENT MAP                        │
├──────────────────────────┬─────────────────────────┬─────────────────┤
│ Notebook                 │ Primary Concern         │ Key Output      │
├──────────────────────────┼─────────────────────────┼─────────────────┤
│ llm_cost_optimization    │ Token efficiency         │ Cost metrics    │
│ agent_memory_optimization│ Context management       │ Memory footprint│
│ agent_continuous_eval    │ Quality measurement      │ Quality scores  │
│ agent_reliability        │ Error handling           │ Uptime metrics  │
│ multi_agent_mcp          │ Agent coordination       │ Task completion │
│ production_ai_engineering│ Production patterns      │ Deployment guide│
│ prompt_injection_defense │ Attack detection         │ Security score  │
│ tiered_memory_cache      │ Multi-level caching      │ Cache hit rates │
└──────────────────────────┴─────────────────────────┴─────────────────┘
```

---

## 2.5 How the Components Interact

```
                      USER REQUEST
                           │
                           ▼
              ┌────────────────────────┐
              │  Security Guard        │
              │  (Prompt Injection     │
              │   Defense)             │
              └────────────┬───────────┘
                           │  Clean input
                           ▼
              ┌────────────────────────┐
              │  Memory Manager        │◄─── Retrieves relevant
              │  (Tiered Cache +       │     conversation history
              │   Memory Optimization) │
              └────────────┬───────────┘
                           │  Enriched context
                           ▼
              ┌────────────────────────┐
              │  Cost Optimizer        │◄─── Picks cheapest
              │  (Model Router)        │     capable model
              └────────────┬───────────┘
                           │  Optimized prompt
                           ▼
              ┌────────────────────────┐
              │  Agent / LLM           │
              │  (with Reliability     │
              │   Wrappers)            │
              └────────────┬───────────┘
                           │  Raw response
                           ▼
              ┌────────────────────────┐
              │  Evaluation Loop       │◄─── Scores quality,
              │  (Continuous Eval)     │     flags if low
              └────────────┬───────────┘
                           │  Validated response
                           ▼
                      USER RESPONSE
```

---

## 2.6 The Evaluation Loop: The Heart of the Lab

The continuous evaluation component is architecturally central because it is the feedback mechanism that makes all other optimisations *measurable*.

```
Step 1: Request comes in
Step 2: System generates a response
Step 3: Evaluator scores the response
         ├── Score ≥ threshold? → Serve to user ✓
         └── Score < threshold? →
                    ├── Log the failure
                    ├── Retry with stronger model
                    └── Alert if persistent
Step 4: Metrics stored for analysis
Step 5: Aggregate reports generated
```

---

## 2.7 The LLMOps Lifecycle

Just as DevOps systematised software deployment, **LLMOps** systematises the deployment and operation of LLM systems. Every architecture decision in the lab is designed with this full lifecycle in mind.

```
╔═════════════════════════════════════════════════════════════════╗
║                      LLMOps LIFECYCLE                          ║
╠═════════════════════════════════════════════════════════════════╣
║                                                                 ║
║  ┌─────────┐    ┌─────────┐    ┌──────────┐    ┌───────────┐  ║
║  │  BUILD  │───►│  TEST   │───►│  DEPLOY  │───►│  MONITOR  │  ║
║  └────┬────┘    └────┬────┘    └─────┬────┘    └─────┬─────┘  ║
║       │              │               │                │        ║
║  Prompt eng   Offline eval     A/B testing      Quality trend  ║
║  RAG setup    Golden set       Shadow mode      Cost tracking  ║
║  Model sel    Regression       Canary release   Latency SLOs   ║
║               Red-teaming      Rollback                        ║
║                                                                 ║
║  ◄──────────────── Continuous Feedback Loop ─────────────────► ║
╚═════════════════════════════════════════════════════════════════╝
```

Each stage maps to a notebook in the lab:
- **Build** → Cost Opt, Memory Opt, Multi-Agent MCP
- **Test** → Continuous Eval, Agent Reliability
- **Deploy** → Production AI Engineering
- **Monitor** → Continuous Eval, Tiered Cache (hit rates), Cost Tracker
- **Secure** → Prompt Injection Defense

---

## 2.8 Summary

The LLM Engineering Optimization Lab's architecture is built around five principles:

1. **Local-first**: Ollama eliminates cost and privacy concerns during development
2. **Composable**: Each notebook is an independent, testable component
3. **Measurable**: Every optimisation is validated by metrics
4. **Layered**: Security → Optimisation → Agent → Evaluation, in that order
5. **Production-minded**: Full topology from CDN to observability, not just the model call

**Key Takeaways:**
- The lab is a layered system, not a collection of scripts
- Every production LLM system needs all seven layers: client, edge, API gateway, middleware, inference, data, and observability
- The evaluation loop is the architectural core that makes optimisation measurable
- LLMOps is the operational discipline that keeps the system healthy over time

---

*Next: [Chapter 3 — Core Concepts & Building Blocks →](./chapter_03_core_concepts.md)*
