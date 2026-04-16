# Table of Contents

---

## Front Matter

- [Title Page](./00_title_page.md)
- [Preface](./01_preface.md)

---

## Part I: Foundations

### [Chapter 1: Problem Statement & Motivation](./chapter_01_problem_statement.md)
- 1.1 The LLM Engineering Crisis Nobody Talks About
- 1.2 Why "Good Enough" Models Fail in Production
- 1.3 The Cost Problem: When Demos Become Disasters
- 1.4 The Hallucination Problem: Confident Wrongness
  - Four Types of Hallucination
  - Hallucination Risk by Task Type
  - Mitigation Strategies (RAG grounding, self-consistency, citation enforcement)
- 1.5 The Reliability Problem: Flaky Systems at Scale
- 1.6 What the Lab Solves
- 1.7 Summary

### [Chapter 2: High-Level Architecture](./chapter_02_architecture.md)
- 2.1 The Bird's-Eye View
- 2.2 The Full Production-Grade Topology (7-Layer Diagram)
- 2.3 The Role of Local Inference with Ollama
- 2.4 Notebook-as-Microservice: The Lab's Architectural Philosophy
- 2.5 Component Map: Eight Notebooks, One System
- 2.6 How the Components Interact
- 2.7 The Evaluation Loop: The Heart of the Lab
- 2.8 The LLMOps Lifecycle (Build → Test → Deploy → Monitor)
- 2.9 Summary

---

## Part II: Core Concepts

### [Chapter 3: Core Concepts & Building Blocks](./chapter_03_core_concepts.md)
- 3.1 Tokens, Context Windows, and Cost
- 3.2 Prompt Engineering Fundamentals
  - Anatomy of a Good Prompt
  - Chain-of-Thought, Structured Output, Role Specification
- 3.3 Retrieval-Augmented Generation (RAG)
  - Indexing Phase and Query Phase
  - Connection to Tiered Memory Cache
- 3.4 Agent Architecture: Tools, Memory, and Reasoning
  - ReAct Pattern, Three Memory Types
- 3.5 Evaluation Frameworks for LLM Output
  - LLM-as-Judge Pattern
- 3.6 Model Context Protocol (MCP)
- 3.7 Tiered Memory Architectures
- 3.8 Prompt Injection: The LLM Security Problem
- 3.9 Summary (original)
- **3.10 Embeddings Deep Dive**
  - What Embeddings Are, Distance Metrics (cosine, dot-product)
  - Choosing an Embedding Model, Dimensionality Trade-offs
  - Batch Embedding with Ollama
- **3.11 Temperature and Sampling — Controlling LLM Randomness**
  - What Temperature Does, top-p vs top-k
  - Parameter Reference by Task Type
- **3.12 Fine-tuning vs. RAG vs. Prompting — Decision Framework**
  - Decision Tree, Approach Comparison Table
  - "Try Prompting First" Rule
- **3.11 Vector Database Selection**
  - Comparison: FAISS, Chroma, Weaviate, Pinecone, Qdrant, pgvector
  - Quick Selection Guide
- **3.14 Chunking Strategies**
  - Five Strategies: Fixed-size, Sentence, Recursive, Semantic, Document-Structure
  - Starting Point Defaults
- 3.15 Summary (updated)

---

## Part III: The Code

### [Chapter 4: Code Walkthrough (Module by Module)](./chapter_04_code_walkthrough.md)
- 4.1 `llm_cost_optimization_lab_part1.ipynb` — Cost Control
- 4.2 `agent_memory_optimization_ollama.ipynb` — Memory Management
- 4.3 `agent_continuous_eval_ollama.ipynb` — Continuous Evaluation
- 4.4 `agent_reliability_ollama.ipynb` — Reliability Patterns
- 4.5 `multi_agent_mcp_ollama.ipynb` — Multi-Agent Coordination
- 4.6 `production_ai_engineering.ipynb` — Production Patterns
- 4.7 `prompt_injection_defense_ollama.ipynb` — Security
- 4.8 `tiered_memory_cache_ollama.ipynb` — Caching & Memory Tiers
- **4.9 Guardrails: Input and Output Safety (Production Pattern)**
  - Input Guardrail Stack (4 layers), Output Guardrail Stack
- **4.10 Streaming Architecture**
  - SSE Streaming Endpoint, Buffered Streaming with Guardrails
- **4.11 Budget Enforcement (Production Pattern)**
  - Three-Level Redis Enforcement, Atomic Check-and-Reserve
- 4.12 Summary (updated)

### [Chapter 5: Data Flow & Execution Flow](./chapter_05_data_flow.md)
- 5.1 The Request Lifecycle
- 5.2 Data Flow in the Cost Optimization Pipeline
- 5.3 Data Flow in the RAG Pipeline
- 5.4 Data Flow in the Agent Loop
- 5.5 Data Flow in Multi-Agent Systems
- 5.6 The Evaluation Data Pipeline
- 5.7 Sequence Diagrams for Key Flows
- 5.8 Summary (original)
- **5.9 The Full Six-Stage Request Pipeline**
  - Stage-by-stage breakdown with timing targets
- **5.10 The LLM Gateway Data Flow**
  - Without vs. with gateway, internal flow diagram
- 5.11 Summary (updated)

---

## Part IV: Design & Engineering Decisions

### [Chapter 6: Key Design Decisions](./chapter_06_design_decisions.md)
- 6.1 Why Ollama Instead of Cloud APIs?
- 6.2 Why Notebooks Instead of a Package?
- 6.3 The Tiered Memory Decision
- 6.4 Choosing an Evaluation Strategy
- 6.5 Stateless vs. Stateful Agents
- 6.6 The MCP Protocol Choice
- 6.7 Summary (original)
- **6.7 Inference Infrastructure — The Serving Stack Decision**
  - Three Paths (Cloud API, Self-Hosted GPU, On-Premises)
  - Quantisation Trade-offs Table
- **6.8 Multi-Tenancy — Isolation by Design**
  - What Must Be Isolated, Tenant-Scoped Request Pattern
- **6.9 Versioning and A/B Testing — Safe Change Management**
  - Three Things to Version, Deterministic Assignment, Statistical Significance
- 6.10 Summary (updated)

### [Chapter 7: Scaling, Trade-offs, and Performance](./chapter_07_scaling.md)
- 7.1 Cost vs. Quality: The Core Trade-off
- 7.2 Latency vs. Accuracy in RAG Systems
- 7.3 Memory Tier Trade-offs
- 7.4 Single Agent vs. Multi-Agent
- 7.5 Model Size vs. Response Quality
- 7.6 Caching Strategies and Their Costs
- 7.7 Observability and Its Overhead
- 7.8 Summary (original)
- **7.9 Resilience and Failure Trade-offs**
  - Failure Mode Matrix, Circuit Breaker Implementation
- **7.10 Async Patterns and Throughput**
  - Async LLM Client with Semaphore, Batch Parallelism
- **7.11 Token Optimisation Techniques**
  - Prompt Compression, KV Cache Prefix Strategy, Structured Output
- **7.12 Context Management Trade-offs**
  - Four Strategies Compared, Summary + Recent Pattern
- 7.11 Summary (updated — complete performance reference table)

---

## Part V: Real World & Future

### [Chapter 8: Real-world Use Cases](./chapter_08_use_cases.md)
- 8.1 Customer Support Automation
- 8.2 Internal Knowledge Management
- 8.3 Code Review and Generation
- 8.4 Document Analysis Pipelines
- 8.5 Autonomous Research Agents
- 8.6 Production Monitoring Dashboards
- 8.7 Summary (original)
- **8.8 Compliance-Critical Applications (Healthcare / Finance / Legal)**
  - PII Redaction, Local-Inference-Only, Human-in-the-Loop Gate, Immutable Audit
- **8.9 Continuous Improvement via User Feedback**
  - Feedback Flywheel, Explicit + Implicit Signals, Training Data Generation
- 8.10 Summary (updated)

### [Chapter 9: Improvements & Extensions](./chapter_09_improvements.md)
- 9.1 From Notebooks to Production Services
- 9.2 Adding Persistent Storage
- 9.3 Integrating Observability (Langfuse, Langsmith)
- 9.4 Fine-tuning for Domain Specificity
- 9.5 Extending the Evaluation Framework
- 9.6 Multi-modal Extensions
- 9.7 Distributed Agent Architectures
- 9.8 Summary (original)
- **9.9 Prompt Regression Testing and CI/CD**
  - Golden Dataset Pattern, Compare Prompts, GitHub Actions Integration
- **9.10 Shadow Mode Deployment**
  - Async Shadow Evaluation, Promotion Path
- **9.11 Building the Feedback-Driven Improvement Pipeline**
  - Feedback to Fine-tuning, JSONL Dataset Generation
- 9.12 Summary (updated — complete 8-stage evolution path)

### [Chapter 10: Interview & Discussion Questions](./chapter_10_interview_questions.md)
- 10.1 System Design Questions (Q1–Q4)
- 10.2 LLM Engineering Concepts (Q5–Q8)
- 10.3 Optimization & Trade-off Questions (Q9–Q10)
- 10.4 Security Questions (Q11–Q12)
- 10.5 Debugging Scenarios (Q11–Q14)
- 10.6 Behavioral / Experience Questions (Q15–Q16)
- **10.7 Production Readiness Questions (Q17–Q20)**
  - Request pipeline layers, LLM Gateway, shadow mode, production checklist
- **10.8 Advanced Concepts Questions (Q21–Q24)**
  - Hallucination mitigation, fine-tune vs RAG, chunking strategies, temperature
- **10.9 Summary: The Complete Competency Map**

### [Chapter 11: The Missing Essentials](./chapter_11_missing_essentials.md)
- **11.1 The Full LLM Gateway Implementation**
  - Complete `LLMGateway` class with failover chain
  - `_build_failover_chain`: ordered provider fallback (OpenAI → Anthropic → Local)
  - Gateway as microservice deployment pattern
- **11.2 LLM Observability Stack — The Four Pillars**
  - Pillar 1: System Metrics (Prometheus) — LLM-specific metric definitions
  - Pillar 2: Structured Logs — mandatory JSON fields for every LLM call
  - Pillar 3: Distributed Traces (OpenTelemetry / Jaeger) — end-to-end flow
  - Pillar 4: LLM-Specific Tracing (Langfuse / LangSmith) — prompt-level visibility
  - `instrument_llm_call` decorator — auto-instruments every LLM call
  - Prometheus alert rules (quality degraded, cost spike, high error rate)
  - Langfuse integration with user feedback attachment
- **11.3 Advanced RAG Patterns**
  - HyDE: Hypothetical Document Embeddings (20–40% recall improvement)
  - Two-Stage Reranking with CrossEncoder (20–40% precision improvement)
  - Parent-Document Retrieval (small chunks for retrieval, large for generation)
  - Hybrid Search: BM25 + Semantic with Reciprocal Rank Fusion
  - Pattern selection guide: when to use each
- **11.4 Production Readiness Checklist**
  - 30-point canonical go/no-go checklist across 6 categories
  - Infrastructure, Pipeline, Resilience, Observability, Quality, Compliance
- **11.5 The Complete LLM Engineering Knowledge Map**
  - Full topic → chapter cross-reference for the entire book

---

## Appendices

- Appendix A: Ollama Quick Reference
- Appendix B: Token Cost Calculator
- Appendix C: Evaluation Metrics Reference
- Appendix D: Recommended Reading

---

*Total estimated reading time: 12–15 hours*
*Total estimated hands-on time: 18–24 hours*
