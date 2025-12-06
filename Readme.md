# RAG Architecture Diagram

This document explains the Retrieval-Augmented Generation (RAG) architecture in a practical, implementation-focused way and includes a diagram-like flow, component responsibilities, data shapes, and integration notes. Use this as a blueprint when building RAG systems with LangChain.

---

## 1. High-level Overview

RAG marries a retriever (search over a knowledge base) with a generator (LLM). The retriever finds relevant context; the generator conditions on that context plus the user query to produce an answer.

**High-level flow:**

1. Document Ingestion → 2. Text Splitting → 3. Embeddings → 4. Vector Store → 5. Retriever → 6. (Optional) Reranker → 7. Prompt/Chain → 8. LLM → 9. Postprocessing/QA

---

## 2. Component Breakdown

### 2.1 Document Ingestion

* **Responsibility:** Read files (PDF, DOCX, HTML, txt, web pages).
* **Typical classes (LangChain):** `PyPDFLoader`, `TextLoader`, `WebBaseLoader`, `DirectoryLoader`.
* **Output:** `List[Document]` where `Document` contains `page_content` and `metadata`.

### 2.2 Text Splitter

* **Responsibility:** Chunk documents into coherent passages (chunks) with overlap.
* **Typical classes:** `RecursiveCharacterTextSplitter`, `TokenTextSplitter`.
* **Key config:** `chunk_size`, `chunk_overlap`.
* **Output:** `List[Document]` (smaller chunks).

### 2.3 Embeddings

* **Responsibility:** Convert text chunks into vectors.
* **Typical providers:** `OpenAIEmbeddings`, `HuggingFaceEmbeddings`, `InstructorEmbeddings`.
* **Output:** `ndarray` or list of vectors paired with document IDs/metadata.

### 2.4 Vector Store

* **Responsibility:** Store vectors, perform nearest-neighbor search.
* **Typical stores:** `FAISS`, `Chroma`, `Milvus`, `Weaviate` (via integrations).
* **Important:** Persist index to disk or managed service; track metadata for retrieval.

### 2.5 Retriever

* **Responsibility:** Given a query, return top-k most relevant document chunks (and optionally scores).
* **Settings:** `search_k`, `fetch_k`, `search_type` (similarity/dense/sparse hybrid).
* **Output:** `List[Document]` with `score` in metadata.

### 2.6 (Optional) Reranker / Re-rank stage

* **Responsibility:** Reorder or filter retrieved docs using a stronger cross-encoder model or heuristics.
* **Why use:** Improve precision, reduce hallucinations, surface most factual context.

### 2.7 Prompt + Chain

* **Responsibility:** Format the retrieved chunks plus the query into a prompt for the LLM.
* **Patterns:** `context + question` (direct), `map-reduce`, `refine`, or `staged summarization`.
* **LCEL:** Use `PromptTemplate` and `RunnableSequence` to combine retriever → prompt → llm.

### 2.8 LLM

* **Responsibility:** Generate answer conditioned on prompt + context.
* **Choices:** `ChatOpenAI`, `Anthropic`, `local LLMs` (Llama, Mistral) via providers.
* **Output:** Raw text, or structured output if using output parsers.

### 2.9 Postprocessing / QA

* **Responsibility:** Clean, parse, and optionally verify LLM output. Implement citation insertion, confidence scoring.

---

## 3. Data Shapes & Example

* `Document`:

```py
{
  "page_content": "...",
  "metadata": {"source": "doc.pdf", "page": 2}
}
```

* `Embeddings`:

```py
[0.003, -0.12, ...]  # float vector
```

* `Retrieval result`:

```py
[{ "document": Document, "score": 0.87 }, ...]
```

---

## 4. Common RAG Patterns

1. **Single-pass RAG (context-window)**

   * Retrieve top-k chunks, concatenate into prompt, call LLM. Best for short context sizes.

2. **Map-Reduce (answer synthesis)**

   * Generate per-chunk partial answers, then combine them in a final pass.

3. **Refine**

   * Iteratively refine an answer as more chunks are processed.

4. **Rerank + Synthesize**

   * Rerank retrieved docs with a cross-encoder then synthesize.

5. **Hybrid retrieval**

   * Combine sparse (BM25) + dense (embeddings) retrieval for broader recall.

---

## 5. Practical Notes / Best Practices

* **Chunk appropriately:** Use smaller chunks for shorter LLM context windows, larger for longer windows.
* **Metadata matters:** Save source, page, timestamps so you can cite results.
* **Pin versions:** Use langchain 1.x and matching community integrations.
* **Re‑ranking reduces hallucination:** Use cross-encoder reranker if accuracy is critical.
* **Cache embeddings:** Recompute only when documents change.

---

## 6. Minimal Code Sketch (LCEL style)

```py
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Load
loader = PyPDFLoader("docs.pdf")
docs = loader.load()

# Split
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# Embeddings + store
emb = OpenAIEmbeddings()
db = FAISS.from_documents(chunks, emb)
retriever = db.as_retriever(search_k=10)

# Prompt + LLM
prompt = PromptTemplate(input_variables=["context","question"], template="{context}\n\nQ: {question}\nA:")
llm = ChatOpenAI(model="gpt-4o-mini")

# RAG (conceptual using LCEL)
rag = ({"context": retriever, "question": RunnablePassthrough()} | prompt) | llm
```

---

## 7. Diagram (ASCII)

```
[Files/PDFs/Web] -> DocumentLoader -> [Doc objects]
                         |
                         v
                   TextSplitter -> [Chunks]
                         |
                         v
                    Embeddings -> [Vectors]
                         |
                         v
                      VectorStore  <-- persist
                         |
                         v
                       Retriever --- optional Reranker
                         |
                         v
                  PromptTemplate (context + query)
                         |
                         v
                         LLM
                         |
                         v
                    Postprocess / Cite / Return
```

---

## 8. Further Reading

* RAG patterns: single-pass, map-reduce, refine.
* LangChain docs: how to wire retrievers to LCEL runnables.

---

# Agents Cheat Sheet (LangChain 1.x)

A concise reference for building and debugging agents with LangChain (post-1.0). Covers core agent types, tools, memory strategies, safety, and example setups.

---

## 1. What is an Agent?

An *Agent* is a program that uses an LLM to decide which actions (tools) to run, in what order, and when to stop. Agents allow chaining tool calls, external data access, and planning.

Agents = *LLM (reasoning) + Tools (actions) + Memory (state) + Controller (policy)*.

---

## 2. Core Agent Types

### 2.1 ReAct-style (Reason + Act)

* Agent produces reasoning traces + tool calls in the same output.
* Good for interactive, stepwise problem solving.

### 2.2 Zero-shot / Few-shot Tool-Use Agent

* Uses prompt templates to teach LLM how to call tools (no training).
* Simpler, but prompt engineering is key.

### 2.3 Planner + Executor (Hierarchical)

* Planner: breaks goal into sub-tasks.
* Executor: runs tools for each subtask.
* Useful for complex multi-step workflows.

### 2.4 Retrieval-Augmented Agent

* Agent uses a retriever as a tool to fetch facts/contexts before deciding further actions.

---

## 3. Tools

* Tools are callable functions that the LLM can trigger.
* Typical tools: `search`, `code_executor`, `calculator`, `file_reader`, `api_caller`.
* In LangChain: tools are usually subclasses of `Tool` or wrappers provided in `langchain_community.tools`.

**Design tip:** Give tools clear `name`, `description`, and deterministic outputs.

---

## 4. Memory Strategies

* **No memory:** stateless agents for single query tasks.
* **Short-term buffer:** `ConversationBufferMemory` — keep last N interactions.
* **Summarization memory:** `ConversationSummaryMemory` — keep condensed history.
* **Long-term/DB memory:** Persist important facts to DB (Redis, PG) and load as context when needed.

Memory classes are in `langchain_community.memory` (post-1.0).

---

## 5. Safety & Guardrails

* **Tool whitelisting:** restrict tools available in production.
* **Action limits / step limits:** prevent runaway loops (max_steps).
* **Output validation:** use output parsers and validators.
* **Rate limiting & timeouts:** for external services.
* **Human-in-the-loop:** for high-risk decisions.

---

## 6. Debugging Patterns

* **1. Reproduce with fixed seed / stub LLM:** Use deterministic LLM or mock responses.
* **2. Log action trace:** Capture LLM thoughts, tool calls, responses.
* **3. Unit-test tools separately.**
* **4. Validate tool outputs before feeding back to LLM.**

---

## 7. Example: ReAct-style Agent (conceptual)

```py
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain_community.memory import ConversationBufferMemory
from langchain_experimental.agents import ReActAgent  # conceptual

# Tools
search_tool = Tool(
    name="serp",
    func=duckduckgo_search,
    description="Search the web and return top results"
)

# Memory
memory = ConversationBufferMemory(k=6)

# LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Agent (pseudo-class)
agent = ReActAgent(llm=llm, tools=[search_tool], memory=memory, max_steps=6)

# Run
agent.run("What's the latest on space launches this month?")
```

---

## 8. Agent Patterns & Tips

* **Few-shot prompts:** include tool examples and expected formats.
* **Chain-of-thought control:** either encourage or suppress verbose reasoning depending on token budget.
* **Tool result summarization:** summarize long tool outputs before returning to agent.
* **Fail-safe responses:** if tools fail, agent should have fallback behaviors.

---

## 9. Common Pitfalls & Fixes

* **LLM hallucination of tool results:** validate tool outputs and include source evidence.
* **Tool name mismatches in prompts:** ensure prompt names exactly match code tool names.
* **Agent loops:** impose step limits and timeouts.
* **Memory bloat:** summarize or prune old entries.

---

## 10. Libraries & Imports (Quick list)

```
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain_community.memory import ConversationBufferMemory
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.agents import create_pandas_dataframe_agent
```

---

## 11. Further Reading & Next Steps

* Implement a small ReAct demo with 2 tools: `search` + `calculator`.
* Add a retriever tool that queries your RAG index.
* Build unit tests for tools and agent steps.

---

*End of Agents Cheat Sheet.*

