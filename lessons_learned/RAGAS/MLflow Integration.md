# Lessons Learned: Integrating RAGAS 0.4.3 with MLflow and Local LLMs

## Context

This document captures bugs, API mismatches, and workarounds encountered while integrating
RAGAS 0.4.3 into a RAG evaluation pipeline using MLflow for tracking and Ollama for local LLM evaluation.

---

## 1. `@experiment` Decorator API

**Problem:** Using `@experiment` as a bare decorator and calling `.arun()` on the result fails with:
```
AttributeError: 'function' object has no attribute 'arun'
```

**Root cause:** In RAGAS 0.4.3, `@experiment` is a decorator **factory** — it must be called with parentheses.
Using it bare returns a plain function with no `.arun()` method.

**Fix:**
```python
# Wrong
@experiment
async def run_experiment(row): ...

# Correct
@experiment()
async def run_experiment(row): ...

result = await run_experiment.arun(dataset, name="my_eval")
```

---

## 2. `LangchainLLMWrapper` is Deprecated

**Problem:** `LangchainLLMWrapper` raises a deprecation warning and a `ValueError` in 0.4.3.

**Root cause:** The public class is a shim. The real class is `_LangchainLLMWrapper` (with underscore),
but it is also unstable. More importantly, RAGAS 0.4.3 moved to a native async client model.

**Fix:** Do not use `LangchainLLMWrapper` at all. Use `llm_factory()` with a native async client instead:
```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory

evaluator_llm = llm_factory("gpt-4o-mini", provider="openai", client=AsyncOpenAI(...))
```

---

## 3. `llm_factory` Requires a Native Async Client

**Problem:** Passing a LangChain LLM (e.g. `ChatOllama`, `ChatCohere`) to `llm_factory` as `client`
causes:
```
Cannot use agenerate() with a synchronous client. Use generate() instead.
```

**Root cause:** `llm_factory` wraps the client with Instructor for structured output. It checks
`_check_client_async()` which inspects the client class name and methods. LangChain wrappers
do not expose the expected async interface (`AsyncInstructor`, `acompletion`, or async `chat.completions.create`).

**Fix:** Always pass a **native** async client to `llm_factory`, never a LangChain wrapper:
```python
# Wrong
evaluator_llm = llm_factory("command-r-plus", provider="cohere", client=ChatCohere(...))

# Correct
from cohere import AsyncClient
evaluator_llm = llm_factory("command-r-plus", provider="cohere", client=AsyncClient(...))
```

---

## 4. Cohere Native Client Not Compatible with Instructor

**Problem:** Using `cohere.AsyncClient` with `llm_factory` raises:
```
Failed to patch cohere client with Instructor: 'AsyncClient' object has no attribute 'messages'
```

**Root cause:** Instructor expects an OpenAI-style `.messages` interface. Cohere's native client
uses `.chat()` instead, which Instructor cannot patch.

**Fix:** Use OpenAI, Groq, or a LiteLLM-based client for the RAGAS evaluator LLM instead of Cohere.
Your main pipeline LLM can remain Cohere — only the evaluator needs to change.

---

## 5. Using Local Ollama Models via LiteLLM

**Problem:** RAGAS has no direct Ollama support in `llm_factory` or `embedding_factory`.

**Fix:** Route through LiteLLM with Instructor in JSON mode:

```python
import instructor
from litellm import AsyncOpenAI as LiteLLMAsync
from ragas.llms import llm_factory

litellm_async_client = LiteLLMAsync(
    base_url="http://localhost:11434/v1",  # Note: must include /v1
    api_key="ollama"
)

async_instructor_client = instructor.from_openai(
    litellm_async_client,
    mode=instructor.Mode.JSON  # Required for local models — see issue #6
)

evaluator_llm = llm_factory(
    model="mistral:latest",       # No "ollama/" prefix when pointing directly at Ollama /v1
    provider="ollama",
    client=async_instructor_client,
    adapter="litellm"
)
```

**Key details:**
- Base URL must be `http://localhost:11434/v1`, not `http://localhost:11434`
- Model name must be `mistral:latest`, not `ollama/mistral:latest` (that prefix is for LiteLLM proxy routing, not direct Ollama)
- `instructor.from_openai(AsyncOpenAI(...))` produces `AsyncInstructor`, which `_check_client_async()` recognizes as async

---

## 6. Instructor Tool Calls Not Supported by Local Models

**Problem:** Even when the model returns valid JSON, RAGAS evaluation fails with:
```
Instructor does not support multiple tool calls, use List[Model] instead
```

**Root cause:** By default, Instructor uses OpenAI-style tool/function calling to extract structured output.
Smaller local models (Mistral, LLaMA3) return JSON in the message content but do not support
the tool call protocol properly.

**Fix:** Force Instructor into JSON mode, which parses the message content directly:
```python
async_instructor_client = instructor.from_openai(
    litellm_async_client,
    mode=instructor.Mode.JSON
)
```

---

## 7. `embedding_factory` Does Not Support Ollama

**Problem:** Passing `"ollama"` as provider to `embedding_factory` raises:
```
ValueError: Unsupported provider: ollama. Supported providers: openai, google, litellm, huggingface
```

**Fix:** Use HuggingFace embeddings locally instead — no API key required:
```python
from ragas.embeddings import HuggingFaceEmbeddings

evaluator_embeddings = HuggingFaceEmbeddings(model="BAAI/bge-base-en-v1.5")
```

Note: Import directly from `ragas.embeddings`, not `ragas.embeddings.base`, to avoid deprecation warnings:
```python
# Deprecated
from ragas.embeddings import embedding_factory

# Correct
from ragas.embeddings import HuggingFaceEmbeddings
```

---

## 8. `result` from `.arun()` is an `Experiment` Object, Not a Dict

**Problem:** Accessing `result["metric_name"]` after `.arun()` raises:
```
TypeError: list indices must be integers or slices, not str
```

**Root cause:** `.arun()` returns a RAGAS `Experiment` object (backed by a list), not a dict.

**Fix:** Convert to pandas first, then aggregate:
```python
df = result.to_pandas()
avg_score = df[config["metric_name"]].mean()
metrics = {config["metric_name"]: avg_score}
```

---

## 9. `Experiment` Object is Not JSON Serializable

**Problem:** Passing the raw `result` object to `json.dump()` raises:
```
TypeError: Object of type Experiment is not JSON serializable
```

**Fix:** Serialize the pandas DataFrame instead:
```python
save_json(df.to_dict(orient="records"), output_dir / "results.json")
```

---

## 10. Ollama Model Name Mismatch

**Problem:** Ollama registers models as `mistral:latest` but configs often pass `"mistral"`, causing:
```
Ollama model not found: mistral
```

**Fix:** Either always use the full tag in configs, or normalize in `build_llm()`:
```python
def build_llm(model_name):
    if ":" not in model_name:
        model_name = f"{model_name}:latest"
    ...
```

---

## Summary Table

| # | Error | Root Cause | Fix |
|---|-------|------------|-----|
| 1 | `'function' has no attribute 'arun'` | `@experiment` used without parentheses | Use `@experiment()` |
| 2 | `LangchainLLMWrapper` deprecated | Removed in 0.4.3 | Use `llm_factory` with native async client |
| 3 | `Cannot use agenerate() with sync client` | LangChain wrappers fail async check | Pass native async client to `llm_factory` |
| 4 | Cohere `AsyncClient` has no `.messages` | Instructor expects OpenAI-style interface | Use OpenAI/Groq/LiteLLM for evaluator |
| 5 | No Ollama support in `llm_factory` | RAGAS only supports OpenAI-style providers | Route through LiteLLM + Instructor |
| 6 | `Instructor does not support multiple tool calls` | Local models don't support tool call protocol | Use `instructor.Mode.JSON` |
| 7 | `Unsupported provider: ollama` in `embedding_factory` | Ollama not a supported embedding provider | Use `HuggingFaceEmbeddings` locally |
| 8 | `list indices must be integers, not str` | `.arun()` returns `Experiment` object not dict | Use `result.to_pandas()` first |
| 9 | `Experiment is not JSON serializable` | `Experiment` object can't be serialized directly | Use `df.to_dict(orient="records")` |
| 10 | `Ollama model not found: mistral` | Missing `:latest` tag in model name | Normalize tag in `build_llm()` |

---

## Recommended Evaluator Setup for Local Ollama (RAGAS 0.4.3)

```python
import instructor
from litellm import AsyncOpenAI as LiteLLMAsync
from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings

# LLM
litellm_client = LiteLLMAsync(base_url="http://localhost:11434/v1", api_key="ollama")
instructor_client = instructor.from_openai(litellm_client, mode=instructor.Mode.JSON)
evaluator_llm = llm_factory(
    model="mistral:latest",
    provider="ollama",
    client=instructor_client,
    adapter="litellm"
)

# Embeddings
evaluator_embeddings = HuggingFaceEmbeddings(model="BAAI/bge-base-en-v1.5")
```

**Dependencies:**
```bash
pip install litellm instructor langchain-ollama ragas
ollama pull mistral
ollama pull nomic-embed-text  # optional, if using Ollama embeddings via LiteLLM
```