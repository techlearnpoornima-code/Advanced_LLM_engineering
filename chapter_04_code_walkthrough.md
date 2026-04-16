# Chapter 4: Code Walkthrough (Module by Module)

> *"Code is like humor. When you have to explain it, it's bad."*
> — Cory House

*But here, we explain it anyway — because understanding why code is written a certain way is the difference between copying patterns and internalizing them.*

---

## 4.1 `llm_cost_optimization_lab_part1.ipynb` — Cost Control

### Purpose

This notebook establishes the foundation of the entire lab: **measuring and minimizing the cost of every LLM interaction.** Everything else in the lab builds on top of this cost-awareness.

### Key Pattern: The Cost Tracker

The first major construct is a `CostTracker` class that wraps every LLM call:

```python
class CostTracker:
    """
    Wraps LLM calls to track token usage and compute cost.
    
    WHY: Without tracking, you have no baseline to optimize against.
    This class makes cost a first-class observable metric.
    """
    
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0
        self.call_history = []
    
    def track_call(self, model: str, input_tokens: int, 
                   output_tokens: int, latency_ms: float):
        """Record a single LLM call's metrics."""
        cost = self._compute_cost(model, input_tokens, output_tokens)
        
        self.call_history.append({
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": latency_ms,
            "cost_usd": cost,
            "timestamp": datetime.now().isoformat()
        })
        
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.call_count += 1
    
    def _compute_cost(self, model: str, 
                      input_tokens: int, output_tokens: int) -> float:
        """
        Compute USD cost for a given model and token count.
        
        WHY these specific rates: These are representative 2024 pricing
        for common cloud models. In production, fetch live rates from
        provider APIs to avoid stale pricing.
        """
        rates = {
            "gpt-4o": {"input": 5.0e-6, "output": 15.0e-6},
            "gpt-4o-mini": {"input": 0.15e-6, "output": 0.60e-6},
            "claude-3-5-sonnet": {"input": 3.0e-6, "output": 15.0e-6},
            "llama3": {"input": 0.0, "output": 0.0},  # Local = free
        }
        r = rates.get(model, {"input": 0.0, "output": 0.0})
        return (input_tokens * r["input"]) + (output_tokens * r["output"])
    
    def report(self) -> dict:
        """Generate a cost summary."""
        return {
            "total_calls": self.call_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": sum(c["cost_usd"] for c in self.call_history),
            "avg_cost_per_call": (
                sum(c["cost_usd"] for c in self.call_history) / self.call_count
                if self.call_count > 0 else 0
            )
        }
```

### Key Pattern: Model Routing

The second major innovation is **model routing** — automatically choosing the cheapest model capable of handling each query:

```python
class ModelRouter:
    """
    Routes queries to the cheapest model that can handle them.
    
    WHY: Not all queries are equal. "What's 2+2?" doesn't need GPT-4.
    "Analyze this 50-page legal document" might. Smart routing is
    the single highest-leverage cost optimization available.
    
    The routing decision uses query complexity signals:
    - Token count (longer = harder)
    - Presence of specialized domain terms
    - Whether structured output is required
    - Historical success rate of each model on similar queries
    """
    
    def route(self, query: str, context: str = "") -> str:
        """Return the recommended model name for this query."""
        complexity = self._score_complexity(query, context)
        
        if complexity < 0.3:
            return "llama3"           # Local, free
        elif complexity < 0.6:
            return "gpt-4o-mini"      # Cheap cloud
        else:
            return "gpt-4o"           # Powerful cloud
    
    def _score_complexity(self, query: str, context: str) -> float:
        """
        Score query complexity from 0.0 (trivial) to 1.0 (very complex).
        
        Signals used:
        - Length (more tokens = more context = harder)
        - Reasoning keywords (analyze, explain why, compare...)
        - Domain-specific terms (legal, medical, financial...)
        - Presence of code or mathematical notation
        """
        signals = []
        
        # Length signal
        token_count = len(query.split()) + len(context.split())
        signals.append(min(token_count / 1000, 1.0))
        
        # Reasoning signal
        reasoning_words = ["analyze", "compare", "explain why", 
                          "evaluate", "critique", "design"]
        has_reasoning = any(w in query.lower() for w in reasoning_words)
        signals.append(0.7 if has_reasoning else 0.0)
        
        # Domain signal
        specialist_domains = ["legal", "medical", "financial", 
                             "regulatory", "clinical", "statutory"]
        has_domain = any(d in query.lower() for d in specialist_domains)
        signals.append(0.9 if has_domain else 0.0)
        
        return max(signals)  # Worst case wins
```

### Visualizing the Savings

The notebook includes visualization code to make cost differences immediately visible:

```python
def compare_models_on_task(query: str, context: str, tracker: CostTracker):
    """
    Run the same query through multiple models and compare:
    - Response quality (scored by evaluator)
    - Token usage
    - Cost
    - Latency
    
    This generates a comparison table showing exactly how much
    is saved by intelligent model routing.
    """
    models = ["llama3", "gpt-4o-mini", "gpt-4o"]
    results = []
    
    for model in models:
        start = time.time()
        response = call_llm(model, query, context)
        latency = (time.time() - start) * 1000
        
        tokens_in = count_tokens(query + context)
        tokens_out = count_tokens(response)
        
        tracker.track_call(model, tokens_in, tokens_out, latency)
        
        results.append({
            "model": model,
            "response": response,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "latency_ms": latency,
            "cost_usd": tracker._compute_cost(model, tokens_in, tokens_out)
        })
    
    return results
```

---

## 4.2 `agent_memory_optimization_ollama.ipynb` — Memory Management

### Purpose

As conversations grow longer, they consume more tokens. This notebook implements strategies to keep memory footprint manageable without losing important context.

### Key Pattern: Conversation Summarization

```python
class ConversationMemoryManager:
    """
    Manages conversation history to stay within token budgets.
    
    WHY: A 10-turn conversation can easily reach 5,000+ tokens.
    At 128k context windows, this isn't a technical problem yet —
    but it IS a cost problem. Every request re-sends the entire
    history. With 1,000 users, you're paying for 5 million tokens
    of context per message.
    
    Strategy: Keep recent turns verbatim, summarize older turns.
    This preserves immediate context while reducing distant history
    to a compact summary.
    """
    
    RECENT_TURNS_TO_KEEP = 5        # Keep last 5 exchanges verbatim
    SUMMARY_TOKEN_BUDGET = 200       # Summarize everything older than 5 turns
    
    def __init__(self, llm_client):
        self.llm = llm_client
        self.full_history = []
        self.summary = ""
    
    def add_turn(self, role: str, content: str):
        """Add a new conversation turn."""
        self.full_history.append({
            "role": role, 
            "content": content,
            "tokens": count_tokens(content)
        })
        
        # Check if we need to summarize
        if len(self.full_history) > self.RECENT_TURNS_TO_KEEP * 2:
            self._compress_old_turns()
    
    def get_context_for_next_call(self) -> list:
        """
        Build the message history to send on the next LLM call.
        
        Returns:
        - If we have a summary: [summary_message, ...recent_turns]
        - Otherwise: [...all_turns]
        """
        recent = self.full_history[-self.RECENT_TURNS_TO_KEEP * 2:]
        
        if self.summary:
            summary_message = {
                "role": "system",
                "content": f"[Conversation summary so far]: {self.summary}"
            }
            return [summary_message] + recent
        
        return recent
    
    def _compress_old_turns(self):
        """
        Summarize turns older than RECENT_TURNS_TO_KEEP.
        
        This is where the magic happens: we use the LLM itself
        to compress older parts of its own context.
        """
        old_turns = self.full_history[:-self.RECENT_TURNS_TO_KEEP * 2]
        old_text = "\n".join(
            f"{t['role']}: {t['content']}" for t in old_turns
        )
        
        summary_prompt = f"""
        Summarize this conversation excerpt in under 200 words.
        Focus on: key decisions made, information established, 
        user preferences noted.
        
        Conversation:
        {old_text}
        
        Summary:
        """
        
        new_summary = self.llm.generate(summary_prompt)
        
        # If we already have a summary, combine it with the new one
        if self.summary:
            combine_prompt = f"""
            Merge these two summaries into one coherent summary of 
            under 200 words:
            
            Earlier summary: {self.summary}
            Recent summary: {new_summary}
            
            Combined summary:
            """
            self.summary = self.llm.generate(combine_prompt)
        else:
            self.summary = new_summary
        
        # Keep only recent turns in full_history
        self.full_history = self.full_history[-self.RECENT_TURNS_TO_KEEP * 2:]
```

### Memory Token Savings Visualization

```
Before memory optimization (10-turn conversation):
┌────────────────────────────────────────────────────────┐
│ Turn 1-2   │ Turn 3-4   │ Turn 5-6   │ Turn 7-8 │ Q10│
│ 800 tokens │ 600 tokens │ 700 tokens │ 500 tok  │ 400│
│                                              Total: 3000│
└────────────────────────────────────────────────────────┘
(Sent on EVERY request after turn 10)

After memory optimization:
┌────────────────────────────────────────────────────────┐
│ Summary    │ Turn 7-8   │ Q10                          │
│ 200 tokens │ 500 tokens │ 400 tokens                   │
│                                          Total: 1100    │
└────────────────────────────────────────────────────────┘
Savings: 63% token reduction!
```

---

## 4.3 `agent_continuous_eval_ollama.ipynb` — Continuous Evaluation

### Purpose

Continuously measure response quality so you know immediately when things degrade.

### Key Pattern: The Evaluation Loop

```python
class ContinuousEvaluator:
    """
    Runs quality evaluation on every LLM response.
    
    WHY: Without continuous eval, quality degradation is invisible
    until users complain. By then, hundreds of bad responses have
    already been served. Early detection enables fast correction.
    
    The evaluator uses an LLM-as-judge pattern: a second LLM 
    (typically the same local model) scores each response against
    a rubric. This is cheap (local model), fast (~200ms), and
    surprisingly accurate compared to human judges.
    """
    
    QUALITY_THRESHOLD = 0.75  # Below this, alert and potentially retry
    
    def __init__(self, llm_client, alert_handler=None):
        self.llm = llm_client
        self.alert_handler = alert_handler
        self.eval_history = []
    
    def evaluate(self, question: str, answer: str, 
                 ground_truth: str = None) -> dict:
        """
        Evaluate a question-answer pair.
        
        Args:
            question: The original user question
            answer: The LLM's response to evaluate
            ground_truth: Known correct answer (if available)
        
        Returns:
            dict with scores and reasoning
        """
        eval_prompt = self._build_eval_prompt(question, answer, ground_truth)
        
        eval_response = self.llm.generate(eval_prompt)
        scores = self._parse_eval_response(eval_response)
        
        result = {
            "question": question,
            "answer": answer,
            "scores": scores,
            "overall": scores.get("overall", 0.0),
            "timestamp": datetime.now().isoformat()
        }
        
        self.eval_history.append(result)
        
        # Alert if quality drops below threshold
        if result["overall"] < self.QUALITY_THRESHOLD:
            self._handle_low_quality(result)
        
        return result
    
    def _build_eval_prompt(self, question: str, answer: str, 
                            ground_truth: str = None) -> str:
        """
        Build a structured evaluation prompt.
        
        WHY this specific rubric: These four dimensions cover the
        key quality concerns in LLM responses:
        - Accuracy: Is it correct?
        - Relevance: Does it answer the actual question?
        - Completeness: Is anything important missing?
        - Clarity: Is it understandable?
        
        Weighting accuracy highest (0.4) because wrong answers
        are worse than incomplete or verbose correct ones.
        """
        gt_section = f"\nGround Truth: {ground_truth}" if ground_truth else ""
        
        return f"""
You are a strict quality evaluator. Assess this Q&A pair.

Question: {question}
Answer: {answer}{gt_section}

Score each dimension from 0.0 to 1.0:
- accuracy (weight 0.40): Factually correct?
- relevance (weight 0.30): Directly addresses the question?
- completeness (weight 0.20): Nothing important missing?
- clarity (weight 0.10): Easy to understand?

Respond ONLY with valid JSON:
{{
  "accuracy": 0.0,
  "relevance": 0.0,
  "completeness": 0.0,
  "clarity": 0.0,
  "overall": 0.0,
  "reasoning": "brief explanation"
}}
"""
    
    def get_quality_trend(self, window: int = 20) -> dict:
        """
        Compute quality trend over recent evaluations.
        
        WHY window=20: 20 samples gives a statistically meaningful
        average while being recent enough to detect fresh regressions.
        """
        recent = self.eval_history[-window:]
        if not recent:
            return {"trend": "no_data", "avg_score": 0.0}
        
        scores = [e["overall"] for e in recent]
        avg = sum(scores) / len(scores)
        
        # Detect trend: compare first half to second half
        if len(scores) >= 4:
            first_half = sum(scores[:len(scores)//2]) / (len(scores)//2)
            second_half = sum(scores[len(scores)//2:]) / (len(scores)//2)
            trend = "improving" if second_half > first_half + 0.05 else \
                    "degrading" if second_half < first_half - 0.05 else \
                    "stable"
        else:
            trend = "insufficient_data"
        
        return {"trend": trend, "avg_score": avg, "sample_count": len(scores)}
```

---

## 4.4 `agent_reliability_ollama.ipynb` — Reliability Patterns

### Purpose

Make agents that fail gracefully and recover automatically.

### Key Pattern: Retry with Fallback

```python
class ReliableAgent:
    """
    Wraps LLM calls with retry logic, fallback models, and
    structured output validation.
    
    WHY: LLMs fail in non-obvious ways:
    1. They refuse to answer (safety filters triggered)
    2. They return malformed JSON (structured output failure)
    3. They return incomplete answers (context window exceeded)
    4. They time out (model under load)
    
    Each failure mode needs a different recovery strategy.
    """
    
    def __init__(self, primary_model: str, fallback_model: str):
        self.primary = primary_model
        self.fallback = fallback_model
        self.retry_delays = [1, 2, 4]  # Exponential backoff in seconds
    
    def call_with_retry(self, prompt: str, 
                        output_schema: dict = None) -> dict:
        """
        Call LLM with automatic retry and fallback.
        
        Retry Strategy:
        - Attempt 1: Primary model, no delay
        - Attempt 2: Primary model, 1s delay (transient failure?)
        - Attempt 3: Primary model, 2s delay (persistent issue?)
        - Attempt 4: FALLBACK model (model-level issue?)
        """
        last_error = None
        
        # Try primary model with retries
        for attempt, delay in enumerate(self.retry_delays):
            if attempt > 0:
                time.sleep(delay)
            
            try:
                response = self._call_llm(self.primary, prompt)
                
                # Validate structure if schema provided
                if output_schema:
                    validated = self._validate_output(response, output_schema)
                    if validated:
                        return {"success": True, "response": validated, 
                                "model": self.primary, "attempts": attempt + 1}
                else:
                    return {"success": True, "response": response,
                            "model": self.primary, "attempts": attempt + 1}
                    
            except (TimeoutError, ConnectionError) as e:
                last_error = e
                continue
            except OutputValidationError as e:
                # Invalid output — prompt modification might help
                prompt = self._add_format_reminder(prompt, output_schema)
                last_error = e
                continue
        
        # Primary failed — try fallback
        try:
            response = self._call_llm(self.fallback, prompt)
            return {"success": True, "response": response,
                    "model": self.fallback, "attempts": len(self.retry_delays) + 1,
                    "used_fallback": True}
        except Exception as e:
            return {"success": False, "error": str(e), 
                    "last_error": str(last_error)}
    
    def _validate_output(self, response: str, schema: dict) -> dict | None:
        """
        Validate that LLM output matches expected JSON schema.
        
        WHY: LLMs frequently return almost-valid JSON with issues like:
        - Trailing commas
        - Single quotes instead of double quotes
        - Missing required fields
        - Extra text before/after JSON
        
        We try to fix common issues before declaring failure.
        """
        # Try direct parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from surrounding text
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                # Check required fields
                if all(k in parsed for k in schema.get("required", [])):
                    return parsed
            except json.JSONDecodeError:
                pass
        
        raise OutputValidationError(f"Could not parse valid JSON from: {response[:200]}")
```

---

## 4.5 `multi_agent_mcp_ollama.ipynb` — Multi-Agent Coordination

### Purpose

Implement systems where multiple specialized agents collaborate on complex tasks using the Model Context Protocol.

### Key Pattern: The Orchestrator-Worker Pattern

```python
class MultiAgentOrchestrator:
    """
    Coordinates multiple specialized agents via MCP.
    
    WHY multi-agent: Some tasks are too complex for a single agent.
    Breaking them into specialized roles improves:
    - Quality (each agent specializes in one thing)
    - Parallelism (agents can work concurrently)
    - Debuggability (failures are isolated to one agent)
    
    Architecture:
    - Orchestrator: Plans the task, delegates sub-tasks
    - Specialist agents: Execute specific sub-tasks
    - MCP: Provides structured tool interface between agents
    
    Analogy: Like a project manager coordinating specialist
    contractors — each does their best work; the PM ensures
    the pieces fit together.
    """
    
    def __init__(self, llm_client):
        self.llm = llm_client
        self.agents = {}
        self.mcp_tools = self._register_mcp_tools()
    
    def register_agent(self, name: str, agent, description: str):
        """Register a specialist agent with the orchestrator."""
        self.agents[name] = {
            "agent": agent,
            "description": description,
            "call_count": 0,
            "success_rate": 1.0
        }
    
    def run_task(self, task: str) -> str:
        """
        Decompose a complex task and coordinate agent execution.
        
        Steps:
        1. Orchestrator creates execution plan
        2. Each sub-task assigned to best available agent
        3. Results collected and synthesized
        4. Final response returned
        """
        # Step 1: Plan
        plan = self._create_plan(task)
        
        # Step 2: Execute (potentially in parallel)
        results = {}
        for step in plan["steps"]:
            agent_name = self._select_agent(step["task_type"])
            result = self.agents[agent_name]["agent"].execute(step["task"])
            results[step["id"]] = result
        
        # Step 3: Synthesize
        return self._synthesize_results(task, plan, results)
    
    def _create_plan(self, task: str) -> dict:
        """Have the orchestrator LLM create an execution plan."""
        available_agents = "\n".join(
            f"- {name}: {info['description']}"
            for name, info in self.agents.items()
        )
        
        plan_prompt = f"""
You are a task orchestrator. Break this task into sub-tasks.

Available agents:
{available_agents}

Task: {task}

Create an execution plan as JSON:
{{
  "steps": [
    {{
      "id": "step_1",
      "task": "specific sub-task description",
      "task_type": "agent_name",
      "depends_on": []  // step ids this step needs first
    }}
  ]
}}
"""
        response = self.llm.generate(plan_prompt)
        return json.loads(response)
```

### MCP Tool Registration

```python
def _register_mcp_tools(self) -> list:
    """
    Register tools in MCP format.
    
    MCP tool schema (compatible with OpenAI function calling):
    {
        "name": str,
        "description": str,
        "parameters": {JSON Schema}
    }
    
    WHY this format: MCP's standardized schema means the same
    tool definition works across different LLM providers without
    modification.
    """
    return [
        {
            "name": "web_search",
            "description": "Search the web for current information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "read_file",
            "description": "Read contents of a local file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"}
                },
                "required": ["path"]
            }
        },
        {
            "name": "run_code",
            "description": "Execute Python code and return output",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to run"}
                },
                "required": ["code"]
            }
        }
    ]
```

---

## 4.6 `production_ai_engineering.ipynb` — Production Patterns

### Purpose

Demonstrate the patterns that separate a toy LLM application from a production-grade one.

### Key Pattern: Structured Logging

```python
import structlog

log = structlog.get_logger()

class ProductionLLMWrapper:
    """
    Production wrapper around LLM calls with:
    - Structured logging (machine-parseable)
    - Request correlation IDs (trace calls across services)
    - Latency tracking
    - Error classification
    
    WHY: In production, you need answers to questions like:
    "Why did request abc-123 fail?"
    "Which users are experiencing high latency?"
    "Is quality dropping for a specific type of query?"
    
    Structured logs (JSON format) let you answer these questions
    by querying your log aggregator (Datadog, CloudWatch, etc.)
    """
    
    def call(self, prompt: str, model: str = "llama3", 
             request_id: str = None) -> dict:
        request_id = request_id or str(uuid.uuid4())
        
        log.info("llm_call_start", 
                 request_id=request_id,
                 model=model,
                 prompt_tokens=count_tokens(prompt))
        
        start_time = time.time()
        
        try:
            response = self._do_call(model, prompt)
            latency_ms = (time.time() - start_time) * 1000
            
            log.info("llm_call_success",
                     request_id=request_id,
                     model=model,
                     latency_ms=round(latency_ms, 2),
                     response_tokens=count_tokens(response),
                     total_tokens=count_tokens(prompt) + count_tokens(response))
            
            return {"success": True, "response": response, 
                    "request_id": request_id, "latency_ms": latency_ms}
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            
            log.error("llm_call_failed",
                      request_id=request_id,
                      model=model,
                      error_type=type(e).__name__,
                      error_message=str(e),
                      latency_ms=round(latency_ms, 2))
            
            raise
```

---

## 4.7 `prompt_injection_defense_ollama.ipynb` — Security

### Key Pattern: The Defense Classifier

```python
class PromptInjectionDefense:
    """
    Detects and blocks prompt injection attacks.
    
    WHY this matters in production:
    If your LLM application processes user input that gets injected
    into prompts (chatbots, Q&A systems, document processors), it is
    vulnerable to users overriding your system instructions.
    
    This is especially dangerous in:
    - Customer service bots (get it to say competitor is better)
    - Document processing (inject instructions into PDFs)
    - RAG systems (poison the knowledge base)
    """
    
    # Known injection patterns (updated regularly)
    INJECTION_PATTERNS = [
        r"ignore (all |previous |prior )?(instructions?|prompts?)",
        r"you are now",
        r"disregard (your |all )?(previous |prior )?(instructions?|training)",
        r"(new|updated) (system |)prompt",
        r"act as (a |an |)?(different|unrestricted|jailbroken)",
        r"DAN\b",  # "Do Anything Now" jailbreak
        r"developer mode",
        r"override (safety|restrictions?|guidelines?)",
    ]
    
    def __init__(self, llm_client):
        self.llm = llm_client
        self.compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS
        ]
    
    def check_input(self, user_input: str) -> dict:
        """
        Multi-layer injection check.
        
        Returns:
            {"safe": bool, "risk_score": float, "reason": str}
        """
        # Layer 1: Fast regex check (< 1ms)
        for pattern in self.compiled_patterns:
            if pattern.search(user_input):
                return {
                    "safe": False,
                    "risk_score": 0.95,
                    "reason": f"Pattern match: {pattern.pattern}",
                    "layer": "regex"
                }
        
        # Layer 2: LLM classification (10-100ms)
        # Only run if regex passes — don't pay for every request
        if len(user_input) > 50:  # Only check longer inputs
            risk = self._llm_classify(user_input)
            if risk > 0.7:
                return {
                    "safe": False,
                    "risk_score": risk,
                    "reason": "LLM classifier flagged as potential injection",
                    "layer": "llm"
                }
        
        return {"safe": True, "risk_score": 0.0, "reason": "passed", "layer": "none"}
    
    def _llm_classify(self, user_input: str) -> float:
        """
        Use a small LLM to classify input as injection or not.
        
        WHY LLM-as-classifier: Regex catches known patterns.
        LLM catches *novel* patterns — new jailbreak techniques
        that haven't been added to the regex list yet.
        
        We use the smallest available model for speed.
        """
        classify_prompt = f"""
You are a security classifier. Determine if this text is a 
prompt injection attack — an attempt to override AI system instructions.

Text: "{user_input}"

Respond ONLY with a JSON: {{"is_injection": true/false, "confidence": 0.0-1.0, "reason": "brief"}}
"""
        response = self.llm.generate(classify_prompt, model="phi3")  # Use small, fast model
        
        try:
            result = json.loads(response)
            if result.get("is_injection"):
                return result.get("confidence", 0.8)
            return 0.0
        except:
            return 0.0  # Parse failure = assume safe
```

---

## 4.8 `tiered_memory_cache_ollama.ipynb` — Caching & Memory Tiers

### Key Pattern: The Cache Hierarchy

```python
class TieredMemoryCache:
    """
    Multi-level cache for LLM responses.
    
    WHY tiered: Different queries benefit from different cache strategies:
    - Exact same query: Serve from RAM instantly
    - Semantically similar query: Serve from vector cache
    - New query: Generate fresh, cache result
    
    This mirrors how computer memory works:
    L1 cache (RAM): Fast, small, exact match
    L2 cache (SSD): Medium, larger, near-match
    L3 "cache" (LLM): Slow, unlimited, any query
    """
    
    def __init__(self, llm_client, embed_client):
        self.llm = llm_client
        self.embedder = embed_client
        
        # L1: In-memory exact cache
        self.l1_cache = {}  # query_hash -> response
        
        # L2: Semantic vector cache
        self.l2_cache = []  # list of (embedding, query, response)
        self.l2_threshold = 0.95  # cosine similarity threshold
        
        # Metrics
        self.stats = {"l1_hits": 0, "l2_hits": 0, "misses": 0}
    
    def query(self, user_query: str) -> dict:
        """
        Query the tiered cache, falling through each level.
        """
        # L1: Exact match
        query_hash = hashlib.md5(user_query.lower().strip().encode()).hexdigest()
        if query_hash in self.l1_cache:
            self.stats["l1_hits"] += 1
            return {"response": self.l1_cache[query_hash], 
                    "cache_tier": "L1", "latency_ms": 0.1}
        
        # L2: Semantic similarity
        query_embedding = self.embedder.embed(user_query)
        similar = self._find_similar(query_embedding)
        
        if similar and similar["similarity"] >= self.l2_threshold:
            self.stats["l2_hits"] += 1
            # Also cache in L1 for future exact matches
            self.l1_cache[query_hash] = similar["response"]
            return {"response": similar["response"],
                    "cache_tier": "L2",
                    "similarity": similar["similarity"],
                    "latency_ms": 15.0}
        
        # L3: Fresh LLM generation
        self.stats["misses"] += 1
        start = time.time()
        response = self.llm.generate(user_query)
        latency = (time.time() - start) * 1000
        
        # Store in both cache tiers
        self.l1_cache[query_hash] = response
        self.l2_cache.append({
            "embedding": query_embedding,
            "query": user_query,
            "response": response
        })
        
        return {"response": response, "cache_tier": "L3_miss", 
                "latency_ms": latency}
    
    def get_cache_stats(self) -> dict:
        """Return cache performance metrics."""
        total = sum(self.stats.values())
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            "l1_hit_rate": self.stats["l1_hits"] / total,
            "l2_hit_rate": self.stats["l2_hits"] / total,
            "miss_rate": self.stats["misses"] / total,
            "effective_cost_reduction": 1 - (self.stats["misses"] / total)
        }
```

---

## 4.9 Summary

We've walked through all eight notebooks and their core patterns:

| Notebook | Core Pattern | What It Optimizes |
|---|---|---|
| Cost Optimization | CostTracker + ModelRouter | Token spend |
| Memory Optimization | ConversationMemoryManager | Context window usage |
| Continuous Eval | ContinuousEvaluator | Response quality |
| Reliability | ReliableAgent with retry | Uptime and correctness |
| Multi-Agent MCP | Orchestrator-Worker | Task complexity |
| Production Eng | Structured logging | Observability |
| Prompt Injection | Defense classifier | Security |
| Tiered Cache | TieredMemoryCache | Latency and cost |

In the next chapter, we'll trace how data flows through these components end-to-end.


---

## 4.9 Guardrails: Input and Output Safety (Production Pattern)

Guardrails are non-optional in production. They prevent bad things going in and bad things coming out.

### Input Guardrail Stack

```python
class InputGuardrails:
    """
    Composable input guardrail stack.
    Fail fast: check cheapest/fastest guards first.
    Only run expensive LLM-based checks if cheaper ones pass.
    """

    def check(self, user_input: str) -> GuardrailResult:
        # Layer 1: Length check (< 1ms)
        if len(user_input) > self.config.max_input_chars:
            return GuardrailResult.fail("INPUT_TOO_LONG")

        # Layer 2: PII detection — regex-based (~5ms)
        if self.config.detect_pii:
            pii_found = self.pii_detector.detect(user_input)
            if pii_found and self.config.block_pii:
                return GuardrailResult.fail("PII_DETECTED", pii_found.types)
            elif pii_found:
                user_input = self.pii_detector.redact(user_input)  # sanitise, don't block

        # Layer 3: Prompt injection — fast regex (~2ms)
        injection = self.injection_defense.check_input(user_input)
        if not injection["safe"] and injection["layer"] == "regex":
            return GuardrailResult.fail("INJECTION_DETECTED", injection["reason"])

        # Layer 4: Content policy — LLM classifier (~50ms, only if above pass)
        if self.config.enable_content_policy:
            policy = self.content_classifier.classify(user_input)
            if policy["violates_policy"]:
                return GuardrailResult.fail("POLICY_VIOLATION", policy["category"])

        return GuardrailResult.pass_(sanitized_input=user_input)
```

### Output Guardrail Stack

```python
class OutputGuardrails:
    """
    Common output failures caught here:
    1. Schema violations (asked for JSON, got prose)
    2. Hallucinated URLs or citations
    3. Leaked system prompt content
    4. Off-topic or harmful content in response
    5. PII echoed back from user input
    """

    def check(self, output: str, expected_schema=None,
              original_query: str = None) -> OutputGuardrailResult:
        failures = []

        # Structured output validation
        if expected_schema:
            try:
                parsed = json.loads(output)
                if not validate_json_schema(parsed, expected_schema):
                    failures.append(("schema", "Output does not match required schema"))
            except json.JSONDecodeError as e:
                failures.append(("schema", str(e)))

        # URL hallucination check
        urls = re.findall(r'https?://[^\s]+', output)
        for url in urls:
            if not self._url_likely_real(url):
                failures.append(("url_validation", f"Potentially hallucinated URL: {url}"))

        # System prompt leakage
        if self._contains_system_prompt_content(output):
            failures.append(("no_leak", "System prompt content detected in output"))

        if failures:
            return OutputGuardrailResult.fail(failures)
        return OutputGuardrailResult.pass_()
```

---

## 4.10 Streaming Architecture

Streaming returns tokens to the user as they are generated — dramatically improving perceived performance.

```
WITHOUT STREAMING:
User sends query → [LLM generates: 8 seconds] → Full response displayed
                                                  (8 seconds of nothing)

WITH STREAMING:
User sends query → Token 1 (0.2s) → Token 2 (0.4s) → ... → Token 200 (8s)
                   "The"             "answer"              (tokens appear as typed)
```

```python
from fastapi.responses import StreamingResponse
import json

@app.post("/chat/stream")
async def stream_chat(request: ChatRequest):
    """
    Streaming endpoint using Server-Sent Events (SSE).
    SSE is simpler than WebSocket for one-way streaming
    and is natively supported by browsers.
    """
    async def generate_tokens():
        async for token in llm_client.stream_complete(
            messages=request.messages,
            model=request.model
        ):
            yield f"data: {json.dumps({'token': token, 'done': False})}\n\n"
        yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"

    return StreamingResponse(
        generate_tokens(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )
```

**Buffered streaming with guardrails:** Collect the full stream first, run output guardrails, then re-stream to the client. Delay is usually imperceptible (< 100ms) but guarantees safety checks run on the complete response.

---

## 4.11 Budget Enforcement (Production Pattern)

```python
class BudgetEnforcementLayer:
    """
    Three-level budget enforcement.
    Uses Redis atomic operations to prevent race conditions
    in multi-instance deployments.

    Level 1: Per-request (max_tokens hard cap)
    Level 2: Per-user/tenant daily and monthly limits
    Level 3: System-wide daily hard stop
    """

    def check_and_reserve(self, user_id, tenant_id,
                           estimated_tokens, estimated_cost_usd):
        today = datetime.utcnow().strftime("%Y-%m-%d")
        month = datetime.utcnow().strftime("%Y-%m")

        user_used  = int(self.redis.get(f"budget:user:{user_id}:{today}") or 0)
        tenant_used = float(self.redis.get(f"budget:tenant:{tenant_id}:{month}") or 0)
        system_used = float(self.redis.get(f"budget:system:{today}") or 0)

        limits = self.get_limits(user_id, tenant_id)

        # System-wide hard stop
        if system_used + estimated_cost_usd > limits["system_daily_usd"]:
            return BudgetDecision.REJECTED("System daily budget exhausted")

        # Tenant monthly limit
        if tenant_used + estimated_cost_usd > limits["tenant_monthly_usd"]:
            return BudgetDecision.REJECTED("Tenant monthly budget exhausted")

        # Graceful degradation at 80% — switch to free local model
        usage_pct = (tenant_used + estimated_cost_usd) / limits["tenant_monthly_usd"]
        if usage_pct > 0.8:
            return BudgetDecision.DOWNGRADE("Approaching budget limit", model="llama3")

        # User daily token limit
        if user_used + estimated_tokens > limits["user_daily_tokens"]:
            return BudgetDecision.REJECTED("User daily token limit reached")

        # All checks passed — atomically reserve the budget
        pipe = self.redis.pipeline()
        pipe.incrby(f"budget:user:{user_id}:{today}", estimated_tokens)
        pipe.expire(f"budget:user:{user_id}:{today}", 86400)
        pipe.incrbyfloat(f"budget:tenant:{tenant_id}:{month}", estimated_cost_usd)
        pipe.incrbyfloat(f"budget:system:{today}", estimated_cost_usd)
        pipe.execute()

        return BudgetDecision.APPROVED()
```

---

## 4.12 Summary

We have now walked through all eight notebooks **plus** the three production patterns every deployment needs:

| Notebook / Pattern | Core Pattern | What It Optimises |
|---|---|---|
| Cost Optimization | CostTracker + ModelRouter | Token spend |
| Memory Optimization | ConversationMemoryManager | Context window usage |
| Continuous Eval | ContinuousEvaluator | Response quality |
| Reliability | ReliableAgent with retry | Uptime and correctness |
| Multi-Agent MCP | Orchestrator-Worker | Task complexity |
| Production Eng | Structured logging | Observability |
| Prompt Injection | Defense classifier | Security |
| Tiered Cache | TieredMemoryCache | Latency and cost |
| **Guardrails** | Input + Output validation | Safety |
| **Streaming** | SSE generator | Perceived performance |
| **Budget Enforcement** | 3-level Redis enforcement | Cost control |

---

*Next: [Chapter 5 — Data Flow & Execution Flow →](./chapter_05_data_flow.md)*
