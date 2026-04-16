# Chapter 7: Scaling, Trade-offs, and Performance

> *"Premature optimization is the root of all evil — but knowing your trade-offs is not premature, it's engineering."*

---

This chapter is about the trade-offs that matter when you move from a working system to a well-performing one. We'll look at concrete numbers, not abstract principles.

---

## 7.1 Cost vs. Quality: The Core Trade-off

The fundamental tension in LLM engineering is that more capable models are more expensive. But "more expensive" doesn't always mean "better results."

### The Quality-Cost Curve

```
Quality
(0-10)
  │
10│                                          ●  GPT-4o
  │                                    ●  GPT-4o + RAG
  │                              ●  Claude 3.5
  │                        ●  GPT-4o-mini + RAG
  │                  ●  GPT-4o-mini
  │            ●  Llama3 + RAG
  │      ●  Llama3
  │
  └──────────────────────────────────────────────► Cost
     $0  $0.001  $0.01  $0.1   $1    $10   (per 1k requests)
```

**Key insight:** Adding RAG to a cheaper model often outperforms the next tier up at a fraction of the cost. This is why the cost optimization notebook focuses on *system optimization* rather than just *model selection*.

### Benchmarking on Real Tasks

The cost optimization notebook includes a benchmark harness:

```python
def run_cost_quality_benchmark(test_cases: list) -> pd.DataFrame:
    """
    Run benchmark across models and strategies.
    
    Returns DataFrame with columns:
    model, strategy, quality_score, cost_usd, latency_ms
    
    Use this to find your personal cost-quality sweet spot.
    """
    results = []
    
    configurations = [
        ("llama3", "baseline"),
        ("llama3", "with_rag"),
        ("llama3", "with_rag_cot"),      # Chain-of-thought
        ("gpt-4o-mini", "baseline"),
        ("gpt-4o-mini", "with_rag"),
        ("gpt-4o", "baseline"),
    ]
    
    for model, strategy in configurations:
        for test_case in test_cases:
            response = run_with_strategy(model, strategy, test_case["query"])
            quality = evaluate_quality(response, test_case["ground_truth"])
            cost = compute_cost(model, response)
            
            results.append({
                "model": model,
                "strategy": strategy,
                "quality": quality,
                "cost_usd": cost,
                "quality_per_dollar": quality / max(cost, 0.0001)
            })
    
    return pd.DataFrame(results).sort_values("quality_per_dollar", ascending=False)
```

**Typical results you might see:**

| Model | Strategy | Quality | Cost/1k req | Quality/$ |
|---|---|---|---|---|
| llama3 | baseline | 6.2 | $0.00 | ∞ |
| llama3 | + RAG | 7.8 | $0.02 | 390 |
| gpt-4o-mini | + RAG | 8.5 | $0.90 | 9.4 |
| gpt-4o | baseline | 9.1 | $12.00 | 0.76 |

*The local model with RAG provides dramatically better quality per dollar for most use cases.*

---

## 7.2 Latency vs. Accuracy in RAG Systems

RAG introduces a latency vs. accuracy trade-off at the retrieval stage.

### The K Parameter

The number of chunks to retrieve (`k`) is the key control:

```
k=1: Fast retrieval, may miss context
k=3: Balanced (most common default)
k=5: More context, slower, more tokens
k=10: Maximum context, slowest, highest cost
```

```python
def benchmark_rag_k(query: str, vector_store, llm, ground_truth: str):
    """Measure quality vs. latency tradeoff for different k values."""
    for k in [1, 2, 3, 5, 8, 10]:
        start = time.time()
        
        chunks = vector_store.search(query, k=k)
        context = "\n\n".join(chunks)
        response = llm.generate(f"Context: {context}\n\nQuestion: {query}")
        
        latency = (time.time() - start) * 1000
        quality = evaluate_quality(response, ground_truth)
        tokens = count_tokens(context)
        
        print(f"k={k:2d} | quality={quality:.2f} | "
              f"latency={latency:.0f}ms | tokens={tokens}")
```

**Typical output:**
```
k= 1 | quality=0.61 | latency=120ms  | tokens=450
k= 2 | quality=0.74 | latency=145ms  | tokens=880
k= 3 | quality=0.81 | latency=170ms  | tokens=1350  ← sweet spot
k= 5 | quality=0.83 | latency=225ms  | tokens=2200
k= 8 | quality=0.83 | latency=320ms  | tokens=3500
k=10 | quality=0.82 | latency=410ms  | tokens=4400  ← diminishing returns
```

*The quality curve flattens after k=3-5, but latency and cost continue rising.*

---

## 7.3 Memory Tier Trade-offs

Each tier of the memory cache trades cost against richness of matching:

```
┌────────────────────────────────────────────────────────────────────┐
│                     CACHE TIER COMPARISON                         │
├──────────┬────────────┬────────────┬─────────────────────────────┤
│  Tier    │ Latency    │ Cost       │ Match Type                  │
├──────────┼────────────┼────────────┼─────────────────────────────┤
│ L1 (RAM) │ < 1ms      │ ~$0/req    │ Exact string match          │
│          │            │            │ "capital of France?"        │
│          │            │            │ = "capital of France?"      │
├──────────┼────────────┼────────────┼─────────────────────────────┤
│ L2       │ 5-50ms     │ ~$0.001/req│ Semantic similarity         │
│ (Vector) │            │ (embedding)│ "capital of France?"        │
│          │            │            │ ≈ "France's capital city?"  │
├──────────┼────────────┼────────────┼─────────────────────────────┤
│ L3       │ 100-5000ms │ $0-$0.015  │ Any query                   │
│ (LLM)    │            │ per request│ "What city does the French  │
│          │            │            │  president live in?"        │
└──────────┴────────────┴────────────┴─────────────────────────────┘
```

### Cache Hit Rate vs. Threshold

The L2 cache has a tunable similarity threshold. Setting it too low causes wrong answers; too high causes unnecessary LLM calls:

```python
# Simulate cache effectiveness at different thresholds
def analyze_cache_threshold(test_queries: list, vector_cache):
    thresholds = [0.99, 0.97, 0.95, 0.90, 0.85, 0.80]
    
    for threshold in thresholds:
        hits = 0
        false_positives = 0
        
        for query in test_queries:
            result = vector_cache.search(query, threshold=threshold)
            if result:
                hits += 1
                if not is_semantically_appropriate(query, result["response"]):
                    false_positives += 1
        
        hit_rate = hits / len(test_queries)
        accuracy = 1 - (false_positives / max(hits, 1))
        
        print(f"threshold={threshold:.2f} | "
              f"hit_rate={hit_rate:.0%} | "
              f"accuracy={accuracy:.0%} | "
              f"cost_saved={hit_rate * 0.015:.4f}$/1k")
```

**Typical output:**
```
threshold=0.99 | hit_rate= 8% | accuracy=99% | cost_saved=$0.0012/1k
threshold=0.97 | hit_rate=23% | accuracy=97% | cost_saved=$0.0035/1k
threshold=0.95 | hit_rate=41% | accuracy=94% | cost_saved=$0.0062/1k  ← sweet spot
threshold=0.90 | hit_rate=65% | accuracy=87% | cost_saved=$0.0098/1k
threshold=0.85 | hit_rate=78% | accuracy=79% | cost_saved=$0.0117/1k  ← too many errors
threshold=0.80 | hit_rate=89% | accuracy=68% | cost_saved=$0.0134/1k
```

---

## 7.4 Single Agent vs. Multi-Agent

Multi-agent systems are more powerful but come with overhead:

```
SINGLE AGENT:
Query → Agent → Response
Latency: 1× (single LLM call)
Cost: 1× (single LLM call)
Complexity: Low
Failure modes: Single point of failure, context window limits

MULTI-AGENT (sequential):
Query → Orchestrator → Agent1 → Agent2 → Agent3 → Synthesize → Response
Latency: 4-6× (multiple LLM calls)
Cost: 4-6× (multiple LLM calls)
Complexity: High
Failure modes: Agent communication errors, state propagation bugs

MULTI-AGENT (parallel):
Query → Orchestrator → [Agent1 + Agent2 + Agent3] → Synthesize → Response
Latency: 2× (parallel execution + synthesis)
Cost: 4-6× (multiple LLM calls, but with parallelism)
Complexity: Very high
Failure modes: Race conditions, partial failure handling
```

### When Multi-Agent Is Worth It

```python
USE_MULTI_AGENT_WHEN = [
    "Task requires specialized knowledge from different domains",
    "Task can be decomposed into parallel workstreams",
    "Task is too complex for a single context window",
    "Different parts of the task require different model capabilities",
    "You need independent validation of results",
]

USE_SINGLE_AGENT_WHEN = [
    "Task fits comfortably in one context window",
    "Speed is critical (< 200ms latency required)",
    "Budget is tight",
    "Task is sequential and cannot be parallelized",
    "You need maximum simplicity for debugging",
]
```

---

## 7.5 Model Size vs. Response Quality

The relationship between model size and quality is non-linear:

```
Quality
(0-10)
  10│
    │                              ●●● 70B models
   8│                        ●● 13B models
    │               ●● 7B models
   6│          ● 3B models
    │     ● 1B models
   4│
    └────────────────────────────────────────► Parameters
        1B    3B    7B    13B   70B

Key observations:
• 7B → 13B: +1-2 quality points (~1.5x cost)
• 13B → 70B: +1-2 quality points (~5x cost)  
• Quality gains diminish rapidly above 13B for most tasks
• For structured tasks (JSON output), 7B is often sufficient
• For complex reasoning, 13B+ makes a meaningful difference
```

---

## 7.6 Caching Strategies and Their Costs

Beyond the tiered memory cache, there are other caching strategies with different trade-off profiles:

| Strategy | What It Caches | Hit Rate | Freshness Risk |
|---|---|---|---|
| Exact cache | Full response | Low (5-15%) | None |
| Semantic cache | Response by meaning | Medium (20-40%) | Low |
| Prefix cache | Shared prompt prefixes | High (60-80%) | None |
| Result cache | Tool call results | High for same tools | Medium |
| Embedding cache | Vector embeddings | Very high (95%+) | Very low |

**Prefix caching** is particularly effective for systems with long system prompts:

```python
# Without prefix caching:
# Every call sends: system_prompt (2000 tokens) + user_query (100 tokens)
# Cost = 2100 tokens × N requests = expensive

# With prefix caching (supported by OpenAI, Anthropic):
# First call: cache the 2000-token system prompt
# Subsequent calls: send only user_query (100 tokens), reference cached prefix
# Cost = 100 tokens × N requests + 2000 tokens once = 95% cost reduction!
```

---

## 7.7 Observability and Its Overhead

Observability (logging, metrics, tracing) has a cost. The question is: how much, and is it worth it?

### Overhead Breakdown

| Observability Feature | Latency Added | CPU Overhead | Worth It? |
|---|---|---|---|
| Structured logging | 0.1-0.5ms | < 1% | Always |
| Token counting | 0.5-2ms | ~1% | Always |
| Quality evaluation | 50-200ms | ~5% | For non-cached paths |
| Full request tracing | 1-5ms | ~2% | In production |
| Distributed tracing (Jaeger) | 5-20ms | ~3% | At scale |

**The rule of thumb:** Observability costs ~1-5% of total latency and ~2-5% CPU. For a system where LLM inference takes 200-5000ms, adding 10ms of observability is always justified by the debugging and optimization value it provides.

```python
# MINIMAL viable observability (always include these):
class MinimalObservability:
    def log_request(self, request_id: str, model: str, 
                    tokens_in: int, tokens_out: int, 
                    latency_ms: float, success: bool):
        """
        This 5-field log answers 90% of production debugging questions:
        - request_id: "Which request failed?"
        - model: "Is one model worse than others?"
        - tokens: "Why is cost high?"
        - latency_ms: "Where is the slowness?"
        - success: "What's my error rate?"
        """
        print(json.dumps({
            "request_id": request_id,
            "model": model,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "latency_ms": latency_ms,
            "success": success,
            "timestamp": datetime.utcnow().isoformat()
        }))
```

---

## 7.8 Summary

The key performance relationships in the lab:

```
COST OPTIMIZATION:
  Local model → near-zero cost at the expense of raw capability
  Model routing → 60-80% cost reduction vs. always-GPT-4
  Caching → 40-70% additional cost reduction
  
QUALITY OPTIMIZATION:
  RAG → +1-3 quality points on knowledge-intensive tasks
  Chain-of-thought → +1-2 quality points on reasoning tasks
  Evaluation loop → enables continuous quality monitoring
  
LATENCY OPTIMIZATION:
  L1 cache → < 1ms (vs. 200-5000ms LLM)
  L2 semantic cache → 5-50ms for semantically similar queries
  Local model → 50-500ms (vs. 200-2000ms cloud API)
  
RELIABILITY OPTIMIZATION:
  Retry logic → ~99% success rate even with 5% base failure rate
  Fallback models → 99.9% availability
```

In the next chapter, we apply these trade-offs to real-world scenarios.


---

## 7.9 Resilience and Failure Trade-offs

LLM systems fail in predictable ways. Engineering resilience means anticipating each failure mode and designing a response.

```
FAILURE MODE        │ PROBABILITY │ IMPACT    │ MITIGATION
────────────────────┼─────────────┼───────────┼──────────────────────────────────
Provider outage     │ Low (1-5%)  │ High      │ Circuit breaker + fallback model
Rate limit hit      │ Medium      │ Medium    │ Per-service limiting, backoff, queue
Model latency spike │ Medium      │ Medium    │ Timeout + retry with faster model
GPU OOM             │ Low-Medium  │ High      │ Request queue + backpressure
Bad model response  │ Low-Medium  │ Medium    │ Output guardrails + retry
Cost spike          │ Low         │ Very High │ Budget enforcement hard stop
Prompt injection    │ Low         │ High      │ Multi-layer input guardrails
Context overflow    │ Medium      │ Medium    │ Memory compression before threshold
```

### Circuit Breaker: Fail Fast, Not Slowly

```python
class LLMCircuitBreaker:
    """
    Without circuit breaker: failing provider causes every request to hang
    for the full timeout (e.g., 30s) before failing.
    With circuit breaker: once failure threshold is hit, fail immediately.

    States: CLOSED (normal) → OPEN (failing, reject fast) → HALF_OPEN (testing recovery)
    """
    FAILURE_THRESHOLD = 5
    RECOVERY_TIMEOUT  = 60   # seconds before attempting recovery

    def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            elapsed = time.time() - self.last_failure_time
            if elapsed < self.RECOVERY_TIMEOUT:
                raise CircuitBreakerOpen(f"Retry in {self.RECOVERY_TIMEOUT - elapsed:.0f}s")
            self.state = "HALF_OPEN"
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.FAILURE_THRESHOLD:
            self.state = "OPEN"
            logger.warning("circuit_breaker_opened", failures=self.failure_count)
```

---

## 7.10 Async Patterns and Throughput

Synchronous LLM calls limit throughput to 1× latency. Async enables N× parallelism.

```
SYNCHRONOUS (blocks thread):
User 1 → LLM (2s) → Response
User 2 → [waits 2s] → LLM → Response
Throughput: 0.5 req/s

ASYNC (concurrent):
User 1 ─┐
User 2  ├─► All LLM calls in parallel → all finish ~2s total
User 3 ─┘
Throughput: ~5 req/s (10× improvement)
```

```python
import asyncio
from asyncio import Semaphore

class AsyncLLMClient:
    def __init__(self, base_url: str, max_concurrent: int = 10):
        self.semaphore = Semaphore(max_concurrent)  # prevents overwhelming the LLM

    async def generate(self, prompt: str) -> str:
        async with self.semaphore:
            async with self.session.post(
                f"{self.base_url}/api/generate",
                json={"model": "llama3", "prompt": prompt, "stream": False}
            ) as resp:
                return (await resp.json())["response"]

    async def generate_batch(self, prompts: list) -> list:
        """All prompts run concurrently — total time ≈ single request time."""
        return await asyncio.gather(*[self.generate(p) for p in prompts])
```

---

## 7.11 Token Optimisation Techniques

Beyond model routing and caching, these techniques reduce token consumption within individual requests.

### Prompt Compression

```python
class PromptCompressor:
    """Remove redundant tokens without losing meaning. Target: 20-40% reduction."""

    FILLER_PATTERNS = [
        (r"As an AI language model, I ", "I "),
        (r"Certainly! Here'?s? ", ""),
        (r"Of course! I'd be happy to ", ""),
        (r"Great question! ", ""),
        (r"Please note that ", ""),
    ]

    def compress(self, prompt: str) -> str:
        for pattern, replacement in self.FILLER_PATTERNS:
            prompt = re.sub(pattern, replacement, prompt, flags=re.IGNORECASE)
        prompt = re.sub(r'\n{3,}', '\n\n', prompt)  # max 2 blank lines
        prompt = re.sub(r' {2,}', ' ', prompt)       # no double spaces
        return prompt
```

### KV Cache — Put Static Content First

```python
# WRONG: Variable content at start defeats KV cache
def bad_prompt(user_name, query, static_system_prompt):
    return f"Hello {user_name},\n\n{static_system_prompt}\n\nQuestion: {query}"
    # Cache miss every call — system_prompt never cached

# RIGHT: Static content first maximises cache hits
def good_prompt(user_name, query, static_system_prompt):
    return f"{static_system_prompt}\n\nUser: {user_name}\nQuestion: {query}"
    # 2000-token system prompt cached after first call
    # Subsequent calls pay for only ~50 new tokens
    # Savings: 97%+ on input token cost for repeated users
```

### Structured Output Compression

```
PROSE OUTPUT   (~80 tokens):
"The overall sentiment is mixed. The reviewer has a positive view of the
 product itself, describing it as 'great', but negative regarding shipping..."

STRUCTURED OUTPUT   (~25 tokens):
{"product_sentiment": "positive", "shipping_sentiment": "negative",
 "overall": "mixed"}

Result: 69% fewer output tokens. At 10,000 req/day this is significant.
```

---

## 7.12 Context Management Trade-offs

Four strategies for managing conversation history as it grows:

```
STRATEGY          │ TOKEN COST  │ ACCURACY  │ IMPLEMENTATION
──────────────────┼─────────────┼───────────┼──────────────────────────────
Sliding window    │ Fixed low   │ Loses old │ Simple — drop oldest turns
Summary + recent  │ Low-medium  │ Good      │ Summarise old, keep recent  ← recommended
Selective memory  │ Low         │ Highest   │ Extract facts, retrieve by relevance
Full + compress   │ High        │ Perfect   │ Expensive, use for critical systems
```

**Sweet spot — Summary + Recent:**

```
[Summary of turns 1-10: 200 tokens]     ← compressed history
[Turns 11-15 verbatim: ~800 tokens]     ← exact recent context
[Current query: 100 tokens]             ← fresh input
Total: ~1100 tokens  (vs 3000+ without compression: 63% savings)
```

---

## 7.13 Summary

Complete performance trade-off table for reference:

```
DIMENSION          │ TECHNIQUE                        │ TYPICAL IMPROVEMENT
───────────────────┼──────────────────────────────────┼──────────────────────
COST               │ Local model (Ollama)              │ ~100% cost reduction
                   │ Model routing                     │ 60–80% cost reduction
                   │ Semantic caching                  │ 40–70% additional
                   │ KV cache / prefix sharing         │ Up to 97% on repeated prompt
                   │ Prompt compression                │ 20–40% token reduction
───────────────────┼──────────────────────────────────┼──────────────────────
LATENCY            │ L1 cache (RAM)                    │ < 1ms vs 200–5000ms
                   │ L2 semantic cache                 │ 5–50ms
                   │ Local model                       │ 50–500ms (vs 200–2000ms cloud)
───────────────────┼──────────────────────────────────┼──────────────────────
QUALITY            │ RAG grounding                     │ +1–3 quality points
                   │ Chain-of-thought                  │ +1–2 quality points
                   │ Reranking (two-stage RAG)         │ +20–40% retrieval recall
───────────────────┼──────────────────────────────────┼──────────────────────
RELIABILITY        │ Retry + fallback                  │ 99%+ success rate
                   │ Circuit breaker                   │ Instant fail vs 30s hang
                   │ Output guardrails                 │ Catches schema + safety issues
───────────────────┼──────────────────────────────────┼──────────────────────
THROUGHPUT         │ Async concurrency                 │ Up to 10× vs synchronous
                   │ Request batching                  │ 2–5× GPU utilisation
```

---

*Next: [Chapter 8 — Real-world Use Cases →](./chapter_08_use_cases.md)*
