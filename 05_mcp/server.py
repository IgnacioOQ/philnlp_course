"""
🔌 A Real MCP Server, From Scratch
==================================

The practical companion to MCP.md. We build a *real* Model Context Protocol
(MCP) server using the official Python SDK (`FastMCP`) and expose three small
tools that an LLM can call. Nothing here is a toy reimplementation of the
protocol — this is the actual wire protocol that Claude Desktop, Claude Code,
Cursor, and friends speak.

How to run it
-------------
This file is a *server*. It does not "do" anything when run by a human — it
waits on stdin/stdout for an MCP client to connect and speak the protocol. So
there are two ways to exercise it:

    # 1. The self-contained demo client in this folder (recommended first run):
    python client_demo.py

    # 2. The official MCP Inspector (needs Node/npx), a web UI to poke the tools:
    mcp dev server.py

    # 3. Register it with a real LLM client (Claude Desktop / Claude Code) —
    #    see MCP.md § "Using it from a real client".

What is MCP, in one breath
--------------------------
An LLM is brilliant at *language* but cannot, by itself, look anything up,
count anything exactly, or touch the world. MCP is the standard plug that lets
it. The server declares a set of **tools** — ordinary functions with typed
inputs and a description. The client hands those tool *schemas* to the model.
The model, mid-conversation, emits a structured "call `tool_x` with these
arguments" request; the client executes the real function here and feeds the
result back. The model never runs our code and never sees our data directly —
it only ever sees the tool interface. That indirection is the whole idea.

Philosophical hook
------------------
Every earlier unit asked what *kind of meaning* a representation captures, and
each answer was internal to the model — co-occurrence counts, latent states,
geometry, attention. MCP is where that internal meaning finally has to *cash
out in action*. The model's symbols only mean something here if `tokenize`
really tokenizes and `corpus_search` really returns the Wittgenstein quote.
This is the symbol-grounding problem made operational: a tool call is a tiny,
checkable bridge between the model's distributional "understanding" of the word
"search" and an actual, deterministic search happening in the world. The model
proposes; the tool disposes.

The three tools (each echoes an earlier unit)
---------------------------------------------
- philnlp_tokenize        → Unit 2: turning a string into discrete symbols.
- philnlp_word_frequencies→ Unit 1: meaning as raw co-occurrence / counts.
- philnlp_corpus_search   → the canonical "JSON-as-database" MCP pattern.
"""

# =====================================================================
# 0. Setup
# =====================================================================
#
# `FastMCP` is the high-level server class from the official MCP Python SDK.
# It does the heavy lifting: it reads each tool's type hints + docstring to
# build the JSON schema the LLM sees, validates incoming arguments, and handles
# the JSON-RPC handshake over stdio. We only write plain Python functions and
# decorate them.
#
# We annotate each parameter with `Annotated[type, Field(...)]`. `Field` carries
# the human description and the validation constraints (min_length, ge/le, …);
# FastMCP turns them into the JSON Schema the model reads AND enforces them
# before our function runs. Writing the fields as *direct parameters* (rather
# than wrapping them in one Pydantic model argument) keeps the generated schema
# flat — the model sees `text` and `method` directly, which is exactly what we
# want it to reason about.
from collections import Counter
from typing import Annotated, Literal

from mcp.server.fastmcp import FastMCP
from pydantic import Field

# The server name follows the MCP convention `{service}_mcp` (snake_case, no
# version, descriptive of the service). This name is what a client lists the
# server under.
mcp = FastMCP("philnlp_mcp")


# =====================================================================
# 1. The "database": a tiny corpus of philosophy-of-language quotes
# =====================================================================
#
# The canonical MCP data pattern is "JSON-as-database": keep structured data in
# plain Python/JSON and expose *search* over it as a tool. The LLM never sees
# this list — it only sees the `philnlp_corpus_search` tool. We could swap this
# for a SQLite table or a real API later and the tool interface would not
# change one bit. That stability of interface is the point of the abstraction.
CORPUS: list[dict] = [
    {
        "author": "J. R. Firth",
        "year": 1957,
        "quote": "You shall know a word by the company it keeps.",
        "theme": "distributional semantics",
    },
    {
        "author": "Ludwig Wittgenstein",
        "year": 1953,
        "quote": "The meaning of a word is its use in the language.",
        "theme": "meaning as use",
    },
    {
        "author": "Gottlob Frege",
        "year": 1892,
        "quote": "Only in the context of a sentence do words have meaning.",
        "theme": "compositionality",
    },
    {
        "author": "Alan Turing",
        "year": 1950,
        "quote": "We can only see a short distance ahead, but we can see plenty there that needs to be done.",
        "theme": "machine intelligence",
    },
    {
        "author": "Zellig Harris",
        "year": 1954,
        "quote": "Difference of meaning correlates with difference of distribution.",
        "theme": "distributional semantics",
    },
    {
        "author": "Richard Montague",
        "year": 1970,
        "quote": "There is no important theoretical difference between natural languages and the artificial languages of logicians.",
        "theme": "formal semantics",
    },
]


# =====================================================================
# 2. Tool 1 — philnlp_tokenize  (echoes Unit 2: tokenization)
# =====================================================================
#
# Annotations (the dict in the decorator) are *hints* to the client about how a
# tool behaves. This tool reads input and returns output with no side effects,
# so it is read-only, non-destructive, idempotent, and self-contained
# (openWorldHint=False — it touches no external system).
@mcp.tool(
    name="philnlp_tokenize",
    annotations={
        "title": "Tokenize Text",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
def philnlp_tokenize(
    text: Annotated[
        str,
        Field(description="The text to tokenize.", min_length=1, max_length=5000),
    ],
    method: Annotated[
        Literal["whitespace", "char"],
        Field(
            description="'whitespace' splits on spaces into words; "
            "'char' yields one token per character."
        ),
    ] = "whitespace",
) -> dict:
    """Split text into discrete tokens — the first step of every NLP pipeline.

    Use this when you need to count tokens, see how a string breaks into words
    or characters, or demonstrate tokenization. Do NOT use it to compute word
    frequencies (use philnlp_word_frequencies for that).

    Args:
        text: the text to tokenize.
        method: 'whitespace' (split on spaces) or 'char' (one token per char).

    Returns:
        dict: {
            "method": str,        # the method used
            "n_tokens": int,      # how many tokens were produced
            "tokens": list[str],  # the tokens themselves (capped at 200)
        }
    """
    if method == "char":
        tokens = list(text)
    else:
        tokens = text.split()

    # Return concise data: a client's context window is finite, so we cap the
    # token list. We still report the true count so nothing is misleading.
    return {
        "method": method,
        "n_tokens": len(tokens),
        "tokens": tokens[:200],
    }


# =====================================================================
# 3. Tool 2 — philnlp_word_frequencies  (echoes Unit 1: counts)
# =====================================================================
@mcp.tool(
    name="philnlp_word_frequencies",
    annotations={
        "title": "Most Frequent Words",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
def philnlp_word_frequencies(
    text: Annotated[
        str,
        Field(description="The text to analyze.", min_length=1, max_length=20000),
    ],
    top_k: Annotated[
        int,
        Field(description="How many of the most frequent words to return.", ge=1, le=100),
    ] = 10,
) -> dict:
    """Count the most frequent words in a text — the n-gram unit's core move.

    This is the distributional hypothesis in miniature: meaning approximated by
    nothing but frequency. Lowercases and splits on whitespace, then counts.

    Args:
        text: the text to analyze.
        top_k: number of top words to return (1-100, default 10).

    Returns:
        dict: {
            "n_tokens": int,                 # total word tokens
            "n_unique": int,                 # distinct word types
            "top": list[[str, int]],         # [word, count], most frequent first
        }
    """
    words = text.lower().split()
    counts = Counter(words)
    return {
        "n_tokens": len(words),
        "n_unique": len(counts),
        # Counter.most_common returns tuples; JSON has no tuples, so we hand back
        # lists. The LLM gets clean [word, count] pairs in frequency order.
        "top": [[w, c] for w, c in counts.most_common(top_k)],
    }


# =====================================================================
# 4. Tool 3 — philnlp_corpus_search  (the JSON-as-database pattern)
# =====================================================================
@mcp.tool(
    name="philnlp_corpus_search",
    annotations={
        "title": "Search the Quote Corpus",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
def philnlp_corpus_search(
    query: Annotated[
        str,
        Field(
            description="Search term, matched (case-insensitively) against "
            "author, quote text, and theme.",
            min_length=1,
            max_length=200,
        ),
    ],
) -> dict:
    """Search a small corpus of philosophy-of-language quotes.

    The corpus holds quotes from Firth, Wittgenstein, Frege, Turing, Harris,
    and Montague, tagged by author, year, and theme. Use this to find who said
    what about meaning, distribution, use, compositionality, etc. Matching is a
    case-insensitive substring search across author, quote, and theme.

    Args:
        query: the search term.

    Returns:
        dict: {
            "query": str,
            "count": int,                # number of matches
            "results": list[dict],       # each: author, year, quote, theme
        }
    """
    q = query.lower()
    results = [
        entry
        for entry in CORPUS
        if q in entry["author"].lower()
        or q in entry["quote"].lower()
        or q in entry["theme"].lower()
    ]
    return {"query": query, "count": len(results), "results": results}


# =====================================================================
# 5. Run the server
# =====================================================================
#
# `mcp.run()` with no arguments uses the **stdio** transport: the server reads
# JSON-RPC messages from stdin and writes responses to stdout. That is exactly
# why a stdio MCP server must never `print()` to stdout for logging — stdout is
# the message channel. (Use stderr if you must log.) stdio is the right
# transport for a local, single-client tool like this one; for a remote,
# multi-client server you would use `mcp.run(transport="streamable_http")`.
if __name__ == "__main__":
    mcp.run()
