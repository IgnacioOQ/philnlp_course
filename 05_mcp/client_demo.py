"""
🔁 Watching the MCP Tool-Call Loop Happen
=========================================

`server.py` is an MCP *server* — it just waits to be spoken to. This file is the
*client*: it launches the server as a subprocess, speaks the real MCP protocol
to it over stdio, asks what tools exist, calls them, and prints the round-trip.

This is the payoff of the unit: you get to *see* the two halves of the protocol
that every earlier diagram talked about —

    list_tools()  →  the client discovers the tool *schemas* (what an LLM sees)
    call_tool()   →  the client invokes a tool and gets a structured result back

— without needing an LLM, an API key, or Claude Desktop. Here *we* play the
role the language model would normally play: we decide which tool to call and
with what arguments. A real LLM client does exactly this, just with the model
choosing the calls instead of us.

Run it end-to-end:

    python client_demo.py

It is intended to be read while running. Each numbered section is one step of
the protocol, with prints showing exactly what crosses the wire.
"""

# =====================================================================
# 0. Setup
# =====================================================================
#
# `asyncio` — the MCP client SDK is async, because in real use a client juggles
#   streaming responses, progress notifications, and multiple servers at once.
# `stdio_client` + `StdioServerParameters` — launch a server as a subprocess and
#   get read/write streams to it.
# `ClientSession` — the high-level object that does the JSON-RPC handshake and
#   exposes `list_tools()` / `call_tool()`.
import asyncio
import json
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# We launch the server with *this same Python interpreter* (sys.executable), so
# the demo uses whatever venv you ran it from — no PATH guessing. The server
# file sits next to this one.
SERVER = StdioServerParameters(
    command=sys.executable,
    args=["server.py"],
)


def show(title: str, payload) -> None:
    """Pretty-print a labelled chunk of JSON crossing the wire."""
    print(f"\n{'─' * 70}\n{title}\n{'─' * 70}")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def unwrap(result):
    """Pull the tool's return value out of a CallToolResult.

    A tool result can arrive two ways depending on SDK/server version:
      - `.structuredContent`: the dict our tool returned, already parsed; or
      - `.content`: a list of text blocks (here, the dict serialized to JSON).
    We prefer structured; otherwise we parse the JSON text so the printout is a
    clean dict rather than an escaped string.
    """
    if result.structuredContent is not None:
        return result.structuredContent
    texts = [c.text for c in result.content if hasattr(c, "text")]
    try:
        return [json.loads(t) for t in texts] if len(texts) != 1 else json.loads(texts[0])
    except (json.JSONDecodeError, ValueError):
        return texts


async def main() -> None:
    # =================================================================
    # 1. Connect: launch the server and open a session
    # =================================================================
    #
    # `stdio_client(SERVER)` spawns `python server.py` and wires its stdin/stdout
    # to us. `ClientSession` then performs the MCP *initialize* handshake — the
    # client and server exchange protocol versions and capabilities. Nothing
    # else can happen before `initialize()` returns.
    print("🚀 Launching server.py as a subprocess and connecting over stdio…")
    async with stdio_client(SERVER) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            print(f"✅ Connected to server: '{init.serverInfo.name}' "
                  f"(protocol {init.protocolVersion})")

            # =========================================================
            # 2. Discovery: list_tools() — exactly what an LLM is given
            # =========================================================
            #
            # The client asks the server for its tool catalog. Each tool comes
            # back with a name, a description (built from our docstring), and an
            # inputSchema (built from our Pydantic model). THIS is the entire
            # surface the language model sees — never our Python source, never
            # the corpus. The model decides what to call from these schemas
            # alone, which is why good names + descriptions matter so much.
            tools = await session.list_tools()
            show(
                "STEP 1 — list_tools(): the schemas the LLM would receive",
                [
                    {
                        "name": t.name,
                        "description": (t.description or "").split("\n")[0],
                        "input_properties": list(
                            (t.inputSchema.get("properties") or {}).keys()
                        ),
                    }
                    for t in tools.tools
                ],
            )

            # =========================================================
            # 3. Invocation: call_tool() — three times, one per tool
            # =========================================================
            #
            # In a real session the model emits these calls. Here we hand-write
            # them to see each tool fire. `call_tool` returns a result whose
            # `.structuredContent` holds the dict our tool returned (FastMCP
            # serializes it for us).
            calls = [
                ("philnlp_tokenize",
                 {"text": "you shall know a word", "method": "whitespace"}),
                ("philnlp_word_frequencies",
                 {"text": "the cat sat on the mat the cat ran", "top_k": 3}),
                ("philnlp_corpus_search",
                 {"query": "distribution"}),
            ]

            for name, args in calls:
                result = await session.call_tool(name, args)
                show(f"STEP 2 — call_tool('{name}', {args})", unwrap(result))

            # =========================================================
            # 4. Validation: the protocol rejects a bad call
            # =========================================================
            #
            # Because the tool input is a Pydantic model with extra='forbid' and
            # a min_length, a malformed call is rejected *before* our function
            # runs. This is the "deterministic" guarantee from the lectures:
            # the tool layer enforces its own contract, the model cannot smuggle
            # in nonsense. We expect this call to come back flagged as an error.
            bad = await session.call_tool("philnlp_tokenize", {"text": ""})
            show(
                "STEP 3 — a deliberately invalid call (empty text) is rejected",
                {"isError": bad.isError,
                 "content": [c.text for c in bad.content if hasattr(c, "text")]},
            )

    print("\n🏁 Done. The subprocess server has been shut down cleanly.")


if __name__ == "__main__":
    asyncio.run(main())
