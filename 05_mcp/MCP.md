# 🔌 Unit 5 — The Model Context Protocol (MCP)

The first four units built *representations of meaning* — n-gram counts, hidden
states, embeddings, attention. Every one of them lived **inside** a model. This
unit is about the moment that internal meaning has to **reach out and act**: how
a language model looks something up, counts something exactly, or touches the
world. The standard way it does so is the **Model Context Protocol (MCP)**.

> **The philosophical hook.** A language model is a master of *distribution* —
> it knows "search" co-occurs with "find", "query", "results". But knowing the
> word's company is not the same as *being able to search*. MCP is where the
> symbol-grounding problem gets cashed out operationally: a **tool call** is a
> tiny, checkable bridge from the model's statistical sense of a word to a real,
> deterministic action in the world. The model *proposes* a call; the tool
> *disposes* of it. Meaning, finally, has consequences.

---

## 🧠 What MCP actually is

An LLM, on its own, can only emit text. It cannot read a file, query a
database, or add two numbers reliably. **MCP is a standard "plug"** that lets a
model use external **tools** — ordinary functions with typed inputs and a
description.

It is **declarative** (a tool is defined by its schema, not its code),
**deterministic** (the function runs in real code, so results are exact and
reproducible), and **model-agnostic** (the same server works with Claude,
GPT, Gemini…). It is the same protocol that Claude Desktop, Claude Code, and
Cursor speak — what we build here is *real*, not a teaching mock-up.

### The four components

```
   USER  ──►  CLIENT / LLM  ──►  MCP SERVER  ──►  TOOLS (real Python)
              (decides what          (registry +        (do the actual
               tool to call)          dispatcher)        work, return data)
                    ▲                                          │
                    └──────────  structured result  ◄──────────┘
```

| Component | Role | In this unit |
|---|---|---|
| **LLM / Client** | Decides *which* tool to call with *what* arguments | `client_demo.py` (we play the model's role by hand) |
| **MCP Server** | Lists tool schemas; dispatches calls to functions | `server.py` (`FastMCP("philnlp_mcp")`) |
| **Tools** | Plain functions that do real work, return JSON | the three `@mcp.tool` functions |

**Key idea:** the model never runs our code and never sees our data. It sees
only the tool *interface* — name, description, input schema. That indirection
is the whole point: we can swap a JSON list for a SQL database and the model
notices nothing.

---

## 🔁 The tool-call loop

A real conversation with tools runs like this:

1. The client sends the user's message **+ the tool schemas** to the model.
2. The model replies not with prose but with a structured request:
   *"call `philnlp_corpus_search` with `{query: "distribution"}`."*
3. The **client** (not the model) executes that call against the server.
4. The server runs the real Python function and returns structured data.
5. The result is fed back to the model as a new message.
6. The model either calls another tool (back to step 2) or writes the final
   answer. The client controls when the loop stops.

`client_demo.py` makes steps 1–4 visible without an LLM in the loop: **we**
choose the calls, so you can watch `list_tools()` (step 1) and `call_tool()`
(steps 3–4) happen over the real protocol.

---

## 📁 Files in this unit

| File | What it is |
|---|---|
| [`server.py`](server.py) | A real `FastMCP` server exposing three tools over stdio |
| [`client_demo.py`](client_demo.py) | A client that launches the server and drives the protocol |
| `MCP.md` | This document |

### The three tools (each echoes an earlier unit)

| Tool | Does | Connects to |
|---|---|---|
| `philnlp_tokenize` | Splits text into word/char tokens | 🔤 Unit 2 — tokenization |
| `philnlp_word_frequencies` | Counts the most frequent words | 🔤 Unit 1 — meaning as counts |
| `philnlp_corpus_search` | Searches a corpus of philosophy-of-language quotes | the canonical "JSON-as-database" MCP pattern |

---

## ⚙️ Setup

This unit has **one dependency** — the official MCP Python SDK (`mcp`) — kept in
a local virtual environment so it doesn't disturb the rest of the course.

```bash
# from the repo root
python -m venv 05_mcp/.venv
05_mcp/.venv/bin/python -m pip install "mcp>=1.0"
```

(Or `pip install -r requirements.txt`, which now includes `mcp`.) On Windows the
interpreter is `05_mcp\.venv\Scripts\python.exe`.

---

## ▶️ Running it

### 1. The demo client (start here)

```bash
cd 05_mcp
.venv/bin/python client_demo.py
```

You'll see the protocol handshake, the **tool schemas the LLM would receive**,
three tool calls with their structured results, and a deliberately invalid call
that the server rejects *before* the function runs (validation is part of the
contract, not an afterthought). Read the script top-to-bottom while it runs.

### 2. The MCP Inspector (optional, needs Node)

The official SDK ships a web UI for poking a server by hand:

```bash
cd 05_mcp
.venv/bin/mcp dev server.py     # opens the Inspector in your browser
```

### 3. Using it from a real client (Claude Desktop / Claude Code)

This is the real payoff — let an actual LLM call these tools. Point the client
at the server with the venv's Python (use **absolute paths**).

**Claude Code** (from the repo root):

```bash
claude mcp add philnlp -- /ABS/PATH/philnlp_course/05_mcp/.venv/bin/python \
    /ABS/PATH/philnlp_course/05_mcp/server.py
```

**Claude Desktop** — add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "philnlp": {
      "command": "/ABS/PATH/philnlp_course/05_mcp/.venv/bin/python",
      "args": ["/ABS/PATH/philnlp_course/05_mcp/server.py"]
    }
  }
}
```

Restart the client, then ask in plain English: *"Search the philnlp corpus for
quotes about distribution"* or *"What are the 5 most frequent words in this
paragraph?"* — and watch the model choose the tool calls itself.

---

## 🛠️ How `server.py` works (the parts that matter)

- **`FastMCP("philnlp_mcp")`** — the server. The name follows the MCP convention
  `{service}_mcp`.
- **`@mcp.tool(...)`** — registers a function as a tool. FastMCP reads the
  function's **type hints** and **docstring** to build the JSON schema the model
  sees, so the docstring is not just documentation — it's the model's only guide
  to *when* to use the tool. Write it like a prompt.
- **`Annotated[str, Field(description=..., min_length=...)]`** — per-parameter
  description **and** validation. We expose fields as *direct parameters* (not
  wrapped in one model object) so the schema stays flat: the model sees `text`
  and `method`, not a nested blob. Bad arguments are rejected before our code
  runs.
- **`annotations={"readOnlyHint": True, ...}`** — behavioral *hints* to the
  client (read-only? destructive? idempotent?). Hints, not security guarantees.
- **`mcp.run()`** — serves over **stdio** (JSON-RPC on stdin/stdout). This is
  why a stdio server must never `print()` to stdout — that channel carries the
  protocol. Use stderr for logging. For a remote, multi-client server you'd use
  `mcp.run(transport="streamable_http")` instead.

---

## 🧩 Where to go next

- **Swap the backend.** Replace the in-memory `CORPUS` list with a SQLite table.
  The tool *interface* shouldn't change at all — proof of the abstraction.
- **Add a tool.** Define a function, decorate it with `@mcp.tool`, restart the
  server. Three steps. Try `philnlp_ngram_counts(text, n)` to extend Unit 1.
- **Let the model drive.** Register the server with Claude Desktop/Code and ask
  questions in natural language — now the *model* chooses the calls, closing the
  loop this unit opened by hand.

---

## ✅ The takeaway

MCP is the bridge from *representing* meaning to *acting* on it. The model
contributes language understanding and the decision of *what* to do; the tool
contributes a deterministic, verifiable *doing*. Everything earlier in this
course was about the first half. This unit is about the seam — and the seam is
where statistical "understanding" finally has to make contact with the world.
