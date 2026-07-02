# Understanding the Stack: Async, Instructor, and LiteLLM

This document explains the concepts behind the RAGAS evaluator setup,
built up from first principles.

---

## 1. Each LLM Has Its Own API

Every LLM provider invented their own way to receive requests:

```python
# OpenAI
client.chat.completions.create(model=..., messages=...)

# Anthropic
client.messages.create(model=..., messages=...)

# Cohere
client.chat(model=..., message=...)   # singular "message", not "messages"

# HuggingFace
pipeline("text-generation", model=...)(prompt)  # completely different concept
```

Different method names, different parameter names, different response structures.

---

## 2. Unification Libraries Wrap All of Them

Libraries like LangChain and LiteLLM solve this by providing one unified interface:

```python
# LangChain unifies behind .invoke()
llm.invoke(messages)   # works for OpenAI, Ollama, Cohere, all of them

# LiteLLM unifies behind completion()
litellm.completion(model="ollama/mistral", messages=...)

# RAGAS llm_factory unifies behind .generate() and .agenerate()
evaluator_llm.agenerate(prompt, ResponseModel)
```

They operate at different levels:

```
Your code
    ↓
LangChain (.invoke)       ← high level: chains, agents, memory, retrievers
    ↓
LiteLLM (.completion)     ← mid level: unified API calls + provider routing
    ↓
Native clients            ← raw provider SDKs (openai, anthropic, cohere)
    ↓
HTTP requests             ← what actually goes over the network
    ↓
LLM provider servers      ← OpenAI, Anthropic, Ollama, etc.
```

LangChain sits higher because it handles the full RAG pipeline.
LiteLLM sits lower and only cares about making the API call uniform.

---

## 3. Why Async Was Needed

### The problem async solves

Synchronous code blocks while waiting:

```python
# Sync — each sample waits for the previous one to finish
sample 1 → wait 5s → sample 2 → wait 5s → sample 3 → wait 5s = 15s total
```

Async runs concurrently:

```python
# Async — all samples run at the same time
sample 1 start → sample 2 start → sample 3 start → all finish ≈ 5s total
```

### How async works

`async def` declares a function that can be paused and resumed.
`await` marks the pause points — "start this, go do other things, come back when done":

```python
async def evaluate_sample(sample):
    score = await call_llm(sample)   # pause here, don't block other samples
    return score
```

### Why RAGAS uses async

RAGAS evaluates multiple samples concurrently to save time.
So it always calls `agenerate()` (async) not `generate()` (sync):

```python
# Inside RAGAS metric scoring
score = await evaluator_llm.agenerate(prompt, ResponseModel)
```

### Why the client must also be async

`agenerate()` internally awaits the client:

```python
async def agenerate(self, prompt, response_model):
    result = await self.client.chat.completions.create(...)  # ← await here
    return result
```

If `.create()` is a regular sync function, you cannot `await` it — that is
exactly the error you hit:

```
Cannot use agenerate() with a synchronous client. Use generate() instead.
```

`AsyncOpenAI` has an async `.create()`, so it works.
Plain `OpenAI` has a sync `.create()`, so it fails.

---

## 4. What Instructor Does

### The problem it solves

LLMs return free text. In code you usually want structured data:

```python
# What LLM returns
"The patient is John, 35 years old, diagnosed with flu"

# What you actually want
Patient(name="John", age=35, diagnosis="flu")
```

### How Instructor works

You define a Pydantic model, pass it to Instructor, it handles everything:

```python
import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel

client = instructor.from_openai(AsyncOpenAI())

class Patient(BaseModel):
    name: str
    age: int
    diagnosis: str

patient = await client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "John is 35 and has flu"}],
    response_model=Patient,
)

print(patient.name)       # "John"    ← always a string
print(patient.age)        # 35        ← always an int
print(patient.diagnosis)  # "flu"     ← always a string
```

If the LLM returns invalid output, Instructor automatically retries with
the validation error as feedback until it gets a valid response.

### Two modes

**Tool call mode (default)** — tells the LLM to call a structured tool:
```
Instructor → LLM: "Use tool Patient(name, age, diagnosis)"
LLM → Instructor: {tool_call: {name: "Patient", args: {name: "John", ...}}}
```
Works well with OpenAI and Claude. Local models like Mistral don't support it reliably.

**JSON mode** — tells the LLM to return raw JSON:
```
Instructor → LLM: "Respond only with JSON: {name: ..., age: ..., diagnosis: ...}"
LLM → Instructor: {"name": "John", "age": 35, "diagnosis": "flu"}
```
Any model that can follow instructions can do this. That's why you needed:

```python
instructor.from_openai(client, mode=instructor.Mode.JSON)
```

### Why RAGAS uses Instructor

RAGAS metrics need structured responses from the evaluator LLM.
For example, `AnswerRelevancy` internally defines something like:

```python
class RelevancyOutput(BaseModel):
    question: str      # rephrased version of the original question
    noncommittal: int  # 0 = real answer, 1 = dodges the question
```

Rather than parsing free text, RAGAS uses Instructor to guarantee it always
gets back a properly typed object. That is why `llm_factory` wraps your
client with Instructor — it is the structured output layer all metrics depend on.

---

## 5. Why LiteLLM Was Needed

### Ollama runs a local server

When Ollama runs, it starts a web server on your machine:
```
http://localhost:11434
```

### Ollama speaks OpenAI's language

Ollama copied OpenAI's API format exactly:
```
OpenAI:  https://api.openai.com/v1/chat/completions
Ollama:  http://localhost:11434/v1/chat/completions
```

Same path, same request format, same response format.
This is what "OpenAI-compatible endpoint" means.

### `openai.AsyncOpenAI` points at OpenAI by default

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key="sk-...")
# internally: base_url = "https://api.openai.com"
```

It doesn't know about your local machine. You have to redirect it:

```python
client = AsyncOpenAI(
    base_url="http://localhost:11434/v1",  # ← redirect to local Ollama
    api_key="ollama"                        # ← Ollama doesn't need a real key
)
```

### Why `llm_factory` needed the LiteLLM adapter

`llm_factory` has two adapters:

```python
llm_factory(model=..., client=..., adapter="instructor")  # default
llm_factory(model=..., client=..., adapter="litellm")     # what you used
```

**Instructor adapter (default)** patches the client directly:
```
your_client → instructor patches it → structured output
```
Instructor's patching mechanism looks for specific OpenAI-style attributes.
Local model clients and non-OpenAI providers have subtle differences that
cause patching to fail. This is exactly the error you hit with Cohere:
```
Failed to patch cohere client with Instructor: 'AsyncClient' object has no attribute 'messages'
```

**LiteLLM adapter** adds a smoothing layer first:
```
your_client → LiteLLM completion layer → instructor → structured output
```
LiteLLM normalizes all provider differences before Instructor ever sees the
response. So Instructor always gets a clean, consistent, OpenAI-style response
regardless of what provider is underneath.

### Why LiteLLM's own `AsyncOpenAI` was used

The LiteLLM adapter inside `llm_factory` was designed and tested around
LiteLLM's own client. It guarantees the response format the adapter expects,
and adds extra handling for retries, timeouts, and local model quirks:

```python
from litellm import AsyncOpenAI as LiteLLMAsync

client = LiteLLMAsync(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)
```

---

## 6. The Full Picture

```
RAGAS metric (needs structured score)
    ↓
llm_factory(adapter="litellm")
    ↓
LiteLLMStructuredLLM.agenerate()        ← async, awaits the client
    ↓
Instructor (Mode.JSON)                  ← structured output layer
    ↓
LiteLLM AsyncOpenAI client              ← async, OpenAI-style interface
    ↓
http://localhost:11434/v1               ← Ollama's OpenAI-compatible endpoint
    ↓
mistral:latest                          ← local model
```

Each layer depends on the one below it:
- RAGAS needs async → `agenerate()` needs async client
- RAGAS needs structured output → Instructor needs OpenAI-style client
- Instructor default patching fails for local models → LiteLLM adapter needed
- LiteLLM adapter needs LiteLLM's own client → `LiteLLMAsync` with `base_url`
- Ollama accepts OpenAI-style requests → compatible with LiteLLM client

Getting any one layer wrong produces a different error —
which is exactly the sequence of errors you debugged through.