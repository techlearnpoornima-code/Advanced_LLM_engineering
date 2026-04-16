# Chapter 8: Real-world Use Cases

> *"In theory, there is no difference between theory and practice. In practice, there is."*
> — Yogi Berra

---

This chapter translates the lab's patterns into specific, practical deployment scenarios. For each use case, we identify which notebooks contribute, what the architecture looks like, and what production considerations matter.

---

## 8.1 Customer Support Automation

### The Problem

A company handles 5,000 support tickets per day. 70% are repetitive questions that could be answered from documentation. Human agents spend 4 hours/day answering questions that have been answered hundreds of times.

### Lab Components Used

- **Tiered Memory Cache**: Cache answers to common questions
- **Cost Optimization**: Route simple questions to local model
- **Prompt Injection Defense**: Prevent abuse in a public-facing chatbot
- **Continuous Evaluation**: Monitor answer quality over time
- **Agent Reliability**: Ensure uptime with retry logic

### Architecture

```
Customer Message
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  INPUT PROCESSING                                           │
│  1. Classify: Is this a support question or something else? │
│  2. Security: Inject defense (public-facing = high risk)    │
│  3. Route: FAQ (simple) or Complex investigation needed?    │
└──────────────────────┬──────────────────────────────────────┘
                       │
           ┌───────────┴──────────────┐
           │                          │
           ▼                          ▼
┌──────────────────┐        ┌──────────────────────┐
│  FAQ PATH        │        │  COMPLEX PATH        │
│                  │        │                      │
│  1. L1 Cache     │        │  1. RAG: Retrieve    │
│     (exact FAQ)  │        │     relevant docs    │
│  2. L2 Semantic  │        │  2. Agent reasoning  │
│     cache        │        │  3. Escalate to      │
│  3. Local LLM +  │        │     human if needed  │
│     KB lookup    │        │                      │
└─────────┬────────┘        └─────────┬────────────┘
          │                           │
          └─────────────┬─────────────┘
                        │
                        ▼
              ┌──────────────────┐
              │  EVAL + QUALITY  │
              │  Gate            │
              │  Score < 0.7?    │
              │  → Escalate      │
              │  Score ≥ 0.7?    │
              │  → Send response │
              └──────────────────┘
```

### Real Impact

| Metric | Before | After |
|---|---|---|
| Avg response time | 4 hours | 30 seconds |
| Cost per ticket | $8.50 (human) | $0.03 (LLM) |
| FAQ resolution rate | 100% (but slow) | 78% auto + 22% escalated |
| Customer satisfaction | 3.8/5 | 4.1/5 |

### Key Code Pattern

```python
class CustomerSupportBot:
    def __init__(self):
        self.security = PromptInjectionDefense(llm)
        self.cache = TieredMemoryCache(llm, embedder)
        self.kb = KnowledgeBase(docs_path="./support_docs/")
        self.evaluator = ContinuousEvaluator(llm)
        self.router = ModelRouter()
    
    def handle_ticket(self, customer_message: str) -> dict:
        # Security first
        check = self.security.check_input(customer_message)
        if not check["safe"]:
            return {"response": "I can't process this request.", 
                    "escalate": True}
        
        # Try cache
        cached = self.cache.query(customer_message)
        if cached["cache_tier"] != "L3_miss":
            return {"response": cached["response"], "source": "cache"}
        
        # Generate fresh response with RAG
        docs = self.kb.retrieve(customer_message, k=3)
        response = self.router.generate_with_context(
            customer_message, docs
        )
        
        # Evaluate quality
        quality = self.evaluator.evaluate(customer_message, response)
        
        if quality["overall"] < 0.70:
            return {"response": response, "escalate": True, 
                    "quality": quality}
        
        return {"response": response, "source": "generated",
                "quality": quality}
```

---

## 8.2 Internal Knowledge Management

### The Problem

An enterprise has 50,000 documents scattered across Confluence, SharePoint, Slack, and email. Employees spend 2.5 hours/week searching for information. New employees take 3 months to become self-sufficient.

### Lab Components Used

- **RAG Pipeline**: Core retrieval from document store
- **Agent Memory Optimization**: Handle long document contexts
- **Tiered Memory Cache**: Cache answers to repeated internal queries
- **Multi-Agent MCP**: Complex research requiring multiple sources

### Architecture

```
Employee Query: "What is our process for approving vendor contracts?"
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  DOCUMENT RETRIEVAL (RAG)                                   │
│                                                             │
│  Embedded knowledge base:                                   │
│  • Legal (contracts, compliance)                            │
│  • HR (policies, procedures)                               │
│  • Engineering (technical docs)                             │
│  • Finance (procurement, budgets)                           │
│                                                             │
│  Search: semantic similarity across all departments         │
│  → Top 5 relevant chunks retrieved                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  SYNTHESIS AGENT                                            │
│                                                             │
│  Prompt: "Based on these documents, answer the question.   │
│           If you need to check specific details, use tools. │
│           Cite your sources."                               │
│                                                             │
│  Tools available:                                           │
│  • search_confluence(query)                                 │
│  • get_slack_thread(id)                                     │
│  • read_document(path)                                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
            Response with citations:
            "According to the Procurement Policy (updated 
            March 2024, page 3), vendor contracts over $10k 
            require: [1] Legal review, [2] CFO approval, 
            [3] CEO sign-off for amounts over $100k..."
```

---

## 8.3 Code Review and Generation

### The Problem

Engineering teams spend 20% of review cycles catching issues that automated tools should find: naming inconsistencies, missing error handling, security vulnerabilities, incomplete test coverage.

### Lab Components Used

- **Continuous Evaluation**: Score code quality metrics
- **Agent Reliability**: Structured output enforcement for code analysis
- **Multi-Agent MCP**: Specialized agents for different concerns (security, style, logic)

### The Code Review Agent

```python
class CodeReviewAgent:
    """
    Multi-agent code review system.
    
    Specialized agents:
    1. SecurityAgent: Finds injection, auth, crypto issues
    2. StyleAgent: Checks naming, formatting, complexity
    3. LogicAgent: Spots bugs, edge cases, missing error handling
    4. TestAgent: Evaluates test coverage and quality
    """
    
    def review(self, code: str, language: str = "python") -> dict:
        # Run specialized agents in parallel
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                "security": executor.submit(self.security_agent.analyze, code),
                "style": executor.submit(self.style_agent.analyze, code),
                "logic": executor.submit(self.logic_agent.analyze, code),
                "tests": executor.submit(self.test_agent.analyze, code),
            }
            
            results = {name: future.result() 
                      for name, future in futures.items()}
        
        # Synthesize into overall review
        return self.synthesizer.create_review(code, results)
```

---

## 8.4 Document Analysis Pipelines

### The Problem

A legal firm processes 200+ contracts per month. Manual review takes 4-6 hours per contract. The firm needs to extract: key dates, obligations, liability caps, termination clauses, and red flags.

### Lab Components Used

- **Agent Memory Optimization**: Handle long documents (contracts can be 50+ pages)
- **Agent Reliability**: Structured output for extracted data
- **Continuous Evaluation**: Validate extraction accuracy

### Document Processing Pattern

```python
class ContractAnalysisPipeline:
    """
    Pipeline for extracting structured information from legal contracts.
    
    Challenge: Contracts are often 20,000-100,000 tokens — far exceeding
    most context windows. Solution: Chunked analysis with aggregation.
    """
    
    def analyze(self, contract_text: str) -> dict:
        # Step 1: Chunk the document
        chunks = self.chunk_document(contract_text, 
                                      chunk_size=4000,
                                      overlap=200)
        
        # Step 2: Extract from each chunk
        chunk_extractions = []
        for i, chunk in enumerate(chunks):
            extraction = self.agent.extract(
                chunk,
                schema={
                    "key_dates": "list of important dates",
                    "obligations": "list of obligations found",
                    "liability_caps": "any liability cap amounts",
                    "red_flags": "unusual or risky clauses"
                }
            )
            chunk_extractions.append(extraction)
        
        # Step 3: Merge chunk extractions
        # This is harder than it looks — same clause can span chunks
        merged = self.merge_extractions(chunk_extractions)
        
        # Step 4: Final synthesis pass
        return self.synthesize_final_analysis(merged)
    
    def merge_extractions(self, extractions: list) -> dict:
        """
        Merge extractions from multiple chunks, deduplicating
        and resolving conflicts between overlapping chunks.
        """
        merged_prompt = f"""
You have extractions from multiple sections of a contract.
Merge them into a unified extraction, removing duplicates
and resolving any conflicts by keeping the most specific version.

Extractions: {json.dumps(extractions, indent=2)}

Return merged JSON in the same schema.
"""
        return json.loads(self.llm.generate(merged_prompt))
```

---

## 8.5 Autonomous Research Agents

### The Problem

Analysts spend days gathering information for market research reports. The process is: search → read → take notes → synthesize → repeat. It's perfect for automation.

### Lab Components Used

- **Multi-Agent MCP**: Research, analysis, and writing agents
- **Agent Reliability**: Handle web search failures gracefully
- **Production Engineering**: Log research steps for review

### Research Agent Loop

```python
class ResearchAgent:
    """
    Autonomous agent that researches a topic and produces a report.
    
    Capabilities:
    - Web search for current information
    - Reading and summarizing web pages
    - Cross-referencing multiple sources
    - Detecting contradictions between sources
    - Producing structured reports with citations
    """
    
    def research(self, topic: str, depth: str = "standard") -> dict:
        """
        Research a topic autonomously.
        
        depth options:
        - "quick": 3-5 sources, 5 min
        - "standard": 10-15 sources, 20 min
        - "deep": 30+ sources, 1 hour
        """
        max_searches = {"quick": 3, "standard": 10, "deep": 30}[depth]
        
        research_log = []
        sources = []
        
        # Initial search
        initial_results = self.tools.web_search(topic)
        
        for result in initial_results[:max_searches]:
            # Read full page
            content = self.tools.fetch_url(result["url"])
            
            # Extract relevant information
            notes = self.llm.generate(f"""
Extract key information about "{topic}" from this article.
Focus on: facts, statistics, expert opinions, and any contradictions.

Article: {content[:5000]}

Return JSON: {{"key_points": [], "quotes": [], "statistics": [], 
               "source_reliability": "high/medium/low"}}
""")
            
            sources.append({
                "url": result["url"],
                "notes": json.loads(notes)
            })
            research_log.append(f"Read: {result['url']}")
        
        # Synthesize
        return self.synthesize_report(topic, sources)
```

---

## 8.6 Production Monitoring Dashboards

### The Problem

Once your LLM application is in production, you need real-time visibility into: cost accumulation, quality trends, model selection distribution, and anomalies.

### Lab Components Used

- **Production Engineering**: Structured logs
- **Continuous Evaluation**: Quality metrics stream
- **Cost Optimization**: Cost metrics stream

### Dashboard Metrics

The production engineering notebook produces metrics that power dashboards:

```python
class ProductionMetrics:
    """
    Aggregates and exposes metrics for dashboard consumption.
    
    Design: Use the same structured log format that major APM tools
    (Datadog, Grafana, CloudWatch) can ingest directly.
    """
    
    def get_dashboard_snapshot(self, window_minutes: int = 60) -> dict:
        """Get metrics for the last N minutes."""
        recent_calls = self._get_calls_in_window(window_minutes)
        
        return {
            "time_window_minutes": window_minutes,
            "request_count": len(recent_calls),
            "cost_usd": sum(c["cost"] for c in recent_calls),
            "avg_latency_ms": mean(c["latency_ms"] for c in recent_calls),
            "p95_latency_ms": percentile(
                [c["latency_ms"] for c in recent_calls], 95),
            "error_rate": sum(1 for c in recent_calls if not c["success"]) 
                         / max(len(recent_calls), 1),
            "avg_quality_score": mean(
                c["quality"] for c in recent_calls if c.get("quality")),
            "model_distribution": Counter(c["model"] for c in recent_calls),
            "cache_hit_rate": sum(1 for c in recent_calls if c["from_cache"])
                             / max(len(recent_calls), 1),
        }
```

**Sample dashboard output:**
```
┌─────────────────────────────────────────────────────────────────┐
│              PRODUCTION LLM MONITOR (Last 60 min)               │
├──────────────────────────────────────────────────────────────────┤
│  Requests:  1,247  │  Cost: $0.83  │  Avg Latency: 186ms        │
│  Errors:    0.8%   │  P95: 450ms   │  Cache Hits: 63%            │
│  Quality:   8.4/10 │  Trend: ↑     │  Models: 82% local          │
├──────────────────────────────────────────────────────────────────┤
│  Model Distribution:                                             │
│  ███████████████████████████████████████ llama3      82%        │
│  ██████                                  gpt-4o-mini  12%        │
│  ██                                      gpt-4o         6%       │
└──────────────────────────────────────────────────────────────────┘
```

---

## 8.7 Summary

The lab's eight notebooks compose into powerful production systems. The key patterns that appear in every use case:

1. **Security at the boundary** — always validate input before it reaches the LLM
2. **Cache aggressively** — most production systems see 40-70% cache hit rates
3. **Route by complexity** — don't use a Ferrari where a bicycle will do
4. **Evaluate continuously** — don't wait for user complaints to discover quality issues
5. **Fail gracefully** — retry, fallback, and escalate rather than error

In the next chapter, we look at how to extend and improve the lab's patterns beyond what's currently implemented.


---

## 8.8 Compliance-Critical Applications (Healthcare / Finance / Legal)

### The Problem

Regulated industries need LLM systems that are not just accurate but auditable, data-sovereign, and explainable.

### Lab Components Used
- **Local inference (Ollama)**: Data never leaves premises
- **Prompt Injection Defense**: High-stakes inputs need strict validation
- **Continuous Evaluation**: Quality gates with human escalation
- **Production Engineering**: Immutable audit logging

### Compliance Architecture

```
User Input (clinician / analyst / lawyer)
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│  PII DETECTION + REDACTION                               │
│  • Detect names, IDs, account numbers, diagnoses         │
│  • Redact before logging (HIPAA / GDPR compliance)       │
└──────────────────────┬───────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│  LOCAL INFERENCE ONLY                                    │
│  • Ollama / vLLM on on-premises GPU                     │
│  • No data leaves the organisation's network            │
└──────────────────────┬───────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│  HUMAN-IN-THE-LOOP GATE                                  │
│  • Quality score < 0.85 → mandatory human review         │
│  • High-stakes decision → always requires human sign-off │
└──────────────────────┬───────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│  IMMUTABLE AUDIT LOG                                     │
│  • Every prompt + response stored (cryptographically     │
│    chained, tamper-evident)                              │
│  • Who asked what, when, what was answered               │
│  • Queryable: "What did the system say to patient X?"    │
└──────────────────────────────────────────────────────────┘
```

### Immutable Audit Implementation

```python
class ImmutableAuditLogger:
    """
    Append-only, cryptographically chained audit log.
    Each entry hashes the previous — tampering is detectable.
    Store in: AWS S3 Object Lock, Azure Immutable Blob, Amazon QLDB.
    """
    def log(self, user_id, prompt, response, model):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "model": model,
            "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
            "prompt": self._redact_pii(prompt),  # store redacted version
            "response": response,
            "previous_hash": self.last_hash
        }
        entry_json = json.dumps(entry, sort_keys=True)
        entry["entry_hash"] = hashlib.sha256(entry_json.encode()).hexdigest()
        self.storage.append(entry)
        self.last_hash = entry["entry_hash"]
```

---

## 8.9 Continuous Improvement via User Feedback

### The Problem

LLM systems deployed without feedback mechanisms never improve. Every user interaction is a learning signal — most teams capture none of it.

### Lab Components Used
- **Continuous Evaluation**: Automated quality baseline
- **Production Engineering**: Logging infrastructure to capture signals
- **Agent Reliability**: Retry on low-quality responses

### The Feedback Flywheel

```
Users interact with LLM
    │
    ▼
Explicit feedback collected (👍/👎, star ratings, corrections)
Implicit feedback inferred (session continuation, escalations)
    │
    ▼
Feedback correlated with logged requests
    │
    ├── Low-feedback responses → prompt investigation
    │
    └── High-feedback responses → candidates for fine-tuning data
    │
    ▼
System improves → users give better feedback → loop continues
```

```python
class FeedbackCollector:
    def record_explicit(self, request_id, rating, helpful, correction=None):
        self.db.insert("feedback", {
            "request_id": request_id,
            "rating": rating,
            "helpful": helpful,
            "correction": correction,
            "timestamp": datetime.utcnow().isoformat()
        })
        if rating <= 2 or not helpful:
            self._flag_for_review(request_id, correction)

    def get_training_candidates(self, min_rating=4.0) -> list:
        """Extract high-quality interactions as fine-tuning training data."""
        return self.db.query("""
            SELECT r.prompt, r.response
            FROM requests r
            JOIN feedback f ON r.request_id = f.request_id
            WHERE f.rating >= ? AND f.helpful = TRUE
            AND f.correction IS NULL
            ORDER BY f.rating DESC LIMIT 5000
        """, [min_rating])
```

---

## 8.10 Summary

The lab's patterns compose into production systems for:

1. **Customer support** — cached FAQ answers, quality-gated escalation, cost-routed
2. **Internal knowledge** — RAG over private documents, multi-agent research
3. **Code review** — parallel specialist agents (security, style, logic)
4. **Document analysis** — chunked long-document processing with extraction
5. **Autonomous research** — multi-step web research with citations
6. **Compliance systems** — local inference, PII redaction, immutable audit
7. **Continuous improvement** — feedback loops feeding back into prompt and fine-tune cycles

In every case, the pattern is the same: **security → optimisation → inference → evaluation → telemetry**.

---

*Next: [Chapter 9 — Improvements & Extensions →](./chapter_09_improvements.md)*
