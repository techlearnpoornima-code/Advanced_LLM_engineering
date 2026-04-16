# Chapter 9: Improvements & Extensions

> *"Software is never finished. It is only released."*

---

The lab, as designed, is an excellent learning foundation. But production systems need more. This chapter is your roadmap for taking the lab's patterns beyond notebooks and into real, scalable, maintainable deployments.

---

## 9.1 From Notebooks to Production Services

The most important transformation is extracting notebook code into importable Python packages and deployable services. Here's the systematic migration path:

### Step 1: Extract Classes into Modules

```
CURRENT (notebook):                PRODUCTION (package):
llm_engineering_lab/
  llm_cost_optimization_lab_part1.ipynb   →   src/
  agent_memory_optimization_ollama.ipynb       ├── cost/
  agent_continuous_eval_ollama.ipynb           │   ├── __init__.py
  agent_reliability_ollama.ipynb               │   ├── tracker.py
  multi_agent_mcp_ollama.ipynb                 │   └── router.py
  production_ai_engineering.ipynb          ├── memory/
  prompt_injection_defense_ollama.ipynb    │   ├── __init__.py
  tiered_memory_cache_ollama.ipynb         │   ├── manager.py
                                           │   └── cache.py
                                           ├── evaluation/
                                           │   ├── __init__.py
                                           │   └── evaluator.py
                                           ├── security/
                                           │   ├── __init__.py
                                           │   └── defense.py
                                           ├── agents/
                                           │   ├── __init__.py
                                           │   ├── base.py
                                           │   ├── reliable.py
                                           │   └── multi_agent.py
                                           └── config.py
```

### Step 2: Add Configuration Management

Notebooks hardcode values. Production code should not:

```python
# config.py
from pydantic import BaseSettings

class LLMLabConfig(BaseSettings):
    """
    Configuration management using Pydantic + environment variables.
    All settings can be overridden via environment variables or .env file.
    """
    
    # Inference
    ollama_base_url: str = "http://localhost:11434"
    default_model: str = "llama3"
    fallback_model: str = "mistral"
    
    # Cost thresholds
    daily_budget_usd: float = 10.0
    alert_threshold_pct: float = 0.80  # Alert at 80% of budget
    
    # Cache settings
    l1_cache_max_size: int = 10_000
    l2_similarity_threshold: float = 0.95
    cache_ttl_seconds: int = 86400
    
    # Evaluation
    quality_threshold: float = 0.75
    eval_sample_rate: float = 1.0    # Evaluate 100% of requests
    
    # Security
    enable_injection_defense: bool = True
    injection_llm_check: bool = True  # Use LLM for second layer
    
    class Config:
        env_file = ".env"
        env_prefix = "LLM_LAB_"

# Usage:
# LLM_LAB_DAILY_BUDGET_USD=50.0 python myapp.py
# Or in .env: LLM_LAB_DEFAULT_MODEL=gpt-4o-mini
config = LLMLabConfig()
```

### Step 3: Wrap in a FastAPI Service

```python
# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="LLM Engineering Lab API")

class QueryRequest(BaseModel):
    query: str
    session_id: str | None = None
    use_rag: bool = True

class QueryResponse(BaseModel):
    response: str
    model_used: str
    cache_tier: str
    quality_score: float
    cost_usd: float
    latency_ms: float

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Main query endpoint with all optimizations applied.
    """
    # Security
    safety = security_guard.check_input(request.query)
    if not safety["safe"]:
        raise HTTPException(status_code=400, 
                          detail="Request flagged by security filter")
    
    # Process
    result = await agent.process(
        query=request.query,
        session_id=request.session_id,
        use_rag=request.use_rag
    )
    
    return QueryResponse(**result)

@app.get("/metrics")
async def get_metrics():
    """Expose production metrics for monitoring."""
    return metrics.get_dashboard_snapshot(window_minutes=60)

@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers."""
    ollama_ok = await check_ollama_connection()
    return {"status": "ok" if ollama_ok else "degraded",
            "ollama": ollama_ok}
```

### Step 4: Containerize with Docker

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/
COPY main.py .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s \
    CMD curl -f http://localhost:8000/health || exit 1

# Run
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 9.2 Adding Persistent Storage

The lab uses in-memory storage. Production needs persistence across restarts:

### Database Schema for Conversation History

```sql
-- conversations.sql
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES conversations(id),
    role VARCHAR(50) NOT NULL,  -- 'user', 'assistant', 'system'
    content TEXT NOT NULL,
    token_count INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE eval_records (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id VARCHAR(255) UNIQUE,
    model VARCHAR(100),
    quality_score FLOAT,
    accuracy FLOAT,
    relevance FLOAT,
    completeness FLOAT,
    clarity FLOAT,
    latency_ms FLOAT,
    cost_usd FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX idx_messages_conversation ON messages(conversation_id, created_at);
CREATE INDEX idx_eval_model ON eval_records(model, created_at);
CREATE INDEX idx_eval_quality ON eval_records(quality_score, created_at);
```

### Persistent Cache with Redis

```python
import redis
import json

class PersistentL1Cache:
    """
    Replace in-memory L1 cache with Redis for persistence
    and multi-instance sharing.
    
    WHY Redis:
    - Survives application restarts
    - Works across multiple instances of your app
    - Built-in TTL support
    - ~0.5ms latency (still much faster than LLM)
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(redis_url)
        self.default_ttl = 86400  # 24 hours
    
    def get(self, key: str) -> str | None:
        value = self.redis.get(f"llm_cache:{key}")
        return value.decode() if value else None
    
    def set(self, key: str, value: str, ttl: int = None):
        self.redis.setex(
            f"llm_cache:{key}",
            ttl or self.default_ttl,
            value
        )
    
    def get_stats(self) -> dict:
        info = self.redis.info("stats")
        return {
            "hits": info["keyspace_hits"],
            "misses": info["keyspace_misses"],
            "hit_rate": info["keyspace_hits"] / 
                       max(info["keyspace_hits"] + info["keyspace_misses"], 1)
        }
```

---

## 9.3 Integrating Observability (Langfuse, Langsmith)

The lab's structured logging is a start. Production systems benefit from specialized LLM observability platforms:

### Langfuse Integration

```python
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context

langfuse = Langfuse()

class ObservableLLMAgent:
    """
    LLM agent instrumented with Langfuse for:
    - Full trace visibility (see every prompt and response)
    - Cost tracking across sessions
    - Quality scoring attached to traces
    - User feedback collection
    """
    
    @observe(name="llm_query")
    def query(self, user_input: str, session_id: str) -> str:
        # This decorator automatically:
        # - Records start/end time
        # - Captures input/output
        # - Computes token counts if model is known
        
        langfuse_context.update_current_trace(
            user_id=session_id,
            tags=["production"],
        )
        
        response = self.agent.process(user_input)
        
        # Attach quality score to the trace
        quality = self.evaluator.evaluate(user_input, response)
        langfuse_context.update_current_observation(
            metadata={"quality_score": quality["overall"]}
        )
        
        return response
    
    def record_user_feedback(self, trace_id: str, score: int, comment: str = ""):
        """Allow users to rate responses (1-5 stars)."""
        langfuse.score(
            trace_id=trace_id,
            name="user_feedback",
            value=score,
            comment=comment
        )
```

---

## 9.4 Fine-tuning for Domain Specificity

When a general-purpose model consistently fails on your domain, fine-tuning is the answer. The lab can generate training data for fine-tuning:

### Generating Fine-tuning Data from the Lab

```python
class FineTuningDataGenerator:
    """
    Use the lab's evaluation system to curate high-quality
    training examples for fine-tuning.
    
    The insight: you already have a running system generating
    responses. Filter for the best ones (high eval scores)
    and use them as training data for a smaller, faster model.
    
    This is "knowledge distillation" — teaching a small model
    to behave like a large model on your specific domain.
    """
    
    def __init__(self, eval_history: list, quality_threshold: float = 0.90):
        self.eval_history = eval_history
        self.threshold = quality_threshold
    
    def generate_training_set(self) -> list:
        """
        Extract high-quality examples from eval history
        in OpenAI fine-tuning format.
        """
        high_quality = [
            record for record in self.eval_history
            if record["overall"] >= self.threshold
        ]
        
        training_examples = []
        for record in high_quality:
            training_examples.append({
                "messages": [
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": record["question"]},
                    {"role": "assistant", "content": record["answer"]}
                ]
            })
        
        return training_examples
    
    def save_as_jsonl(self, output_path: str):
        """Save training data in JSONL format (required by OpenAI)."""
        examples = self.generate_training_set()
        with open(output_path, 'w') as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')
        
        print(f"Saved {len(examples)} training examples to {output_path}")
        print(f"Quality threshold: {self.threshold} "
              f"({len(examples)/len(self.eval_history):.0%} of total)")
```

---

## 9.5 Extending the Evaluation Framework

The current evaluator scores four dimensions. Production systems often need domain-specific metrics:

```python
class DomainEvaluator(ContinuousEvaluator):
    """
    Extends the base evaluator with domain-specific metrics.
    
    Add your own dimensions based on what matters for your use case:
    - Legal accuracy for contract analysis
    - Code correctness for code generation
    - Medical safety for health applications
    - Citation accuracy for research assistants
    """
    
    def evaluate_with_domain(self, question: str, answer: str,
                             domain: str = "general") -> dict:
        base_result = self.evaluate(question, answer)
        
        if domain == "code":
            domain_score = self._eval_code_correctness(answer)
        elif domain == "legal":
            domain_score = self._eval_legal_accuracy(answer)
        elif domain == "medical":
            domain_score = self._eval_medical_safety(answer)
        else:
            domain_score = None
        
        return {**base_result, "domain_score": domain_score}
    
    def _eval_code_correctness(self, code: str) -> dict:
        """
        Test if code actually runs and passes basic checks.
        Real code evaluation: extract code blocks and execute them.
        """
        code_blocks = self._extract_code_blocks(code)
        
        results = []
        for block in code_blocks:
            try:
                exec(block, {"__builtins__": {}})  # Sandboxed execution
                results.append({"runs": True, "error": None})
            except Exception as e:
                results.append({"runs": False, "error": str(e)})
        
        runnable = sum(1 for r in results if r["runs"])
        return {
            "runnable_blocks": runnable,
            "total_blocks": len(results),
            "correctness_score": runnable / max(len(results), 1)
        }
```

---

## 9.6 Multi-modal Extensions

The lab is currently text-only. Many production use cases need images, audio, or structured data:

```python
class MultiModalAgent:
    """
    Extends the lab's agent patterns to handle images and audio.
    
    Requires models with multimodal capabilities:
    - LLaVA (vision, runs locally via Ollama)
    - GPT-4o (vision + audio, cloud)
    - Whisper (speech-to-text, runs locally)
    """
    
    def process_with_image(self, text_query: str, 
                           image_path: str) -> str:
        """
        Process a query that includes an image.
        Automatically routes to a vision-capable model.
        """
        import base64
        
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()
        
        # LLaVA format for Ollama
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llava",    # Vision model
                "prompt": text_query,
                "images": [image_data],
                "stream": False
            }
        )
        return response.json()["response"]
    
    def process_audio_query(self, audio_path: str) -> str:
        """
        Convert speech to text, then process with LLM.
        """
        import whisper
        
        # Transcribe audio
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        text_query = result["text"]
        
        # Now process as normal text query
        return self.agent.process(text_query)
```

---

## 9.7 Distributed Agent Architectures

For high-throughput production systems, a single-machine deployment isn't sufficient. Here's the distributed pattern:

```
DISTRIBUTED MULTI-AGENT ARCHITECTURE:

┌─────────────────────────────────────────────────────────────────┐
│                      LOAD BALANCER                              │
│                    (nginx / AWS ALB)                            │
└───────────────────┬─────────────────────────────────────────────┘
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
┌──────────┐  ┌──────────┐  ┌──────────┐
│  API     │  │  API     │  │  API     │
│  Server  │  │  Server  │  │  Server  │
│  (inst 1)│  │  (inst 2)│  │  (inst 3)│
└──────┬───┘  └──────┬───┘  └──────┬───┘
       │              │              │
       └──────────────┼──────────────┘
                      │
          ┌───────────┼──────────────┐
          ▼           ▼              ▼
┌──────────────┐  ┌──────────┐  ┌───────────┐
│  Message     │  │  Redis   │  │ Postgres  │
│  Queue       │  │  Cache   │  │ (history, │
│  (Celery/    │  │          │  │  eval)    │
│   RabbitMQ)  │  │          │  │           │
└──────┬───────┘  └──────────┘  └───────────┘
       │
  ┌────┴────┐
  │         │
  ▼         ▼
┌────────┐ ┌────────┐
│ Ollama │ │ Ollama │   (multiple GPU machines)
│ GPU 1  │ │ GPU 2  │
└────────┘ └────────┘
```

### Task Queue Pattern for Long-running Agent Tasks

```python
from celery import Celery

celery_app = Celery('llm_lab', broker='redis://localhost:6379/0')

@celery_app.task(bind=True, max_retries=3)
def run_research_agent(self, task_id: str, query: str) -> dict:
    """
    Run a long research task asynchronously.
    
    WHY async: Multi-step research can take 30-120 seconds.
    No web user wants to wait that long on an HTTP connection.
    Use async tasks + webhooks or polling for results.
    """
    try:
        agent = ResearchAgent()
        result = agent.research(query, depth="standard")
        
        # Store result
        store_task_result(task_id, result)
        return {"status": "complete", "task_id": task_id}
        
    except Exception as exc:
        # Celery handles retries with exponential backoff
        raise self.retry(exc=exc, countdown=2 ** self.request.retries)
```

---

## 9.8 Summary

The lab provides all the patterns you need. This chapter showed you the evolution path:

```
LEARNING PATH:
Notebooks → Python modules → FastAPI service → Docker container

STORAGE PATH:
In-memory dicts → Redis cache → PostgreSQL persistence

OBSERVABILITY PATH:
Print statements → Structured logs → Langfuse/Langsmith

SCALE PATH:
Single machine → Load balanced API → Message queue + workers

QUALITY PATH:
Manual evaluation → Automated eval → Fine-tuned domain model
```

Each step in this evolution is independent. You don't need to do all of them at once. Start with the one that solves your most pressing problem, and build from there.


---

## 9.9 Prompt Regression Testing and CI/CD

Prompts are code. They must be tested like code. A "small prompt tweak" can silently break quality for 20% of your traffic.

### The Golden Dataset Pattern

```python
class PromptTestSuite:
    """
    Regression testing for LLM prompts.
    Before any prompt change, run the new prompt against curated test cases
    with known-good answers. Only ship if quality is maintained or improved.

    golden_dataset format:
    [{"id": "t001", "input": "...", "must_contain": ["Paris"],
      "must_not_contain": ["London"], "quality_threshold": 0.9}]
    """

    def run(self, prompt_template: str, model: str = "llama3") -> TestReport:
        results = []
        for test in self.golden_dataset:
            response = llm.generate(prompt_template.format(query=test["input"]),
                                    model=model)
            # Hard constraints
            for required in test.get("must_contain", []):
                if required.lower() not in response.lower():
                    results.append(TestResult(test["id"], passed=False,
                                              reason=f"Missing: '{required}'"))
                    continue
            # Quality score
            quality = self.evaluator.evaluate(test["input"], response)
            passed = quality["overall"] >= test.get("quality_threshold", 0.75)
            results.append(TestResult(test["id"], passed=passed,
                                      quality_score=quality["overall"]))
        return TestReport(results)

    def compare_prompts(self, prompt_a: str, prompt_b: str) -> dict:
        """Returns which prompt performs better and by how much."""
        report_a, report_b = self.run(prompt_a), self.run(prompt_b)
        return {
            "winner": "B" if report_b.pass_rate > report_a.pass_rate else "A",
            "improvement": report_b.avg_quality - report_a.avg_quality,
            "regressions": report_b.find_regressions(report_a)
        }
```

### CI/CD Integration

```yaml
# .github/workflows/prompt_regression.yml
name: LLM Prompt Regression Test
on:
  pull_request:
    paths: ['prompts/**', 'src/**']

jobs:
  prompt_regression:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Start Ollama + pull model
        run: |
          curl -fsSL https://ollama.ai/install.sh | sh
          ollama serve &
          sleep 5 && ollama pull llama3
      - name: Run prompt regression tests
        run: python -m pytest tests/prompt_tests/ -v
      - name: Quality gate
        run: |
          python scripts/check_quality_gate.py \
            --min-pass-rate 0.95 \
            --min-avg-quality 0.80 \
            --fail-on-regression
```

---

## 9.10 Shadow Mode Deployment

Before exposing users to a new LLM configuration, run it in **shadow mode**: it receives all production traffic, its responses are evaluated, but users only see the current production response.

```python
class ShadowModeDeployer:
    """
    Shadow mode workflow:
    1. New config receives same requests as production
    2. New config's responses are NOT shown to users
    3. Both responses are evaluated and compared
    4. After N requests with satisfactory metrics → promote to A/B test
    5. After A/B test proves improvement → promote to production
    """

    async def process(self, request, production_config, shadow_config):
        # Run both in parallel
        prod_task   = asyncio.create_task(self._call(production_config, request))
        shadow_task = asyncio.create_task(self._call(shadow_config, request))

        # Return production response immediately (user is waiting)
        prod_response = await prod_task

        # Evaluate shadow asynchronously (doesn't block user)
        asyncio.create_task(self._evaluate_shadow(shadow_task, request, prod_response))

        return prod_response

    async def _evaluate_shadow(self, shadow_task, request, prod_response):
        shadow_response = await shadow_task
        prod_q   = self.evaluator.evaluate(request.query, prod_response)
        shadow_q = self.evaluator.evaluate(request.query, shadow_response)
        logger.info("shadow_comparison",
                    prod_quality=prod_q["overall"],
                    shadow_quality=shadow_q["overall"],
                    shadow_better=shadow_q["overall"] > prod_q["overall"])
```

---

## 9.11 Building the Feedback-Driven Improvement Pipeline

Connect user feedback to a continuous fine-tuning pipeline:

```
PIPELINE OVERVIEW:
─────────────────────────────────────────────────────
Step 1: Capture feedback (ratings, corrections, implicit signals)
Step 2: Filter high-quality interactions (rating ≥ 4, no corrections needed)
Step 3: Generate JSONL fine-tuning dataset
Step 4: Fine-tune a smaller model on your domain data
Step 5: A/B test fine-tuned model vs. general model
Step 6: Deploy if improvement confirmed
─────────────────────────────────────────────────────
```

```python
class FeedbackToFineTuning:
    """
    Converts accumulated user feedback into a fine-tuning dataset.
    This is knowledge distillation: teaching a small model to behave
    like the large model on YOUR specific domain.
    """
    def generate_training_set(self, min_rating=4.0) -> list:
        high_quality = self.feedback_db.query(
            "SELECT prompt, response FROM requests r "
            "JOIN feedback f ON r.request_id = f.request_id "
            "WHERE f.rating >= ? AND f.helpful = TRUE "
            "AND f.correction IS NULL", [min_rating]
        )
        return [
            {"messages": [
                {"role": "user",      "content": row["prompt"]},
                {"role": "assistant", "content": row["response"]}
            ]}
            for row in high_quality
        ]

    def save_as_jsonl(self, path: str):
        examples = self.generate_training_set()
        with open(path, 'w') as f:
            for ex in examples:
                f.write(json.dumps(ex) + '\n')
        print(f"Saved {len(examples)} training examples to {path}")
```

---

## 9.12 Summary: The Complete Evolution Path

```
STAGE 1 — LEARNING:       Notebooks (this lab)
STAGE 2 — MODULARISE:     Python package (src/ layout, config.py)
STAGE 3 — SERVE:          FastAPI + Docker
STAGE 4 — PERSIST:        Redis (cache) + Postgres (history, eval)
STAGE 5 — OBSERVE:        Prometheus + Langfuse/LangSmith
STAGE 6 — SAFE DEPLOY:    Prompt regression tests + shadow mode + A/B
STAGE 7 — IMPROVE:        Feedback collection → fine-tuning pipeline
STAGE 8 — SCALE:          Load balancer + Celery workers + multi-GPU vLLM
```

Each stage is independent. Start with the one that solves your most pressing problem today.

---

*Next: [Chapter 10 — Interview & Discussion Questions →](./chapter_10_interview_questions.md)*
