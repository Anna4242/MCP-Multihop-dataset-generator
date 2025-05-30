# mcp_demo.py
import asyncio, json, os, time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from mcp_use import MCPAgent, MCPClient

# ───────────────────  CONFIG  ─────────────────── #
CONFIG = {
    "inputs": [
        { "type":"promptString", "id":"github_token",
          "description":"GitHub PAT", "password":True }
    ],
    "mcpServers": {
        "github": {
            "connector":"stdio","command":"cmd",
            "args":["/c","npx","-y","@modelcontextprotocol/server-github"],
            "env":{"GITHUB_PERSONAL_ACCESS_TOKEN":"${input:github_token}"}
        },
        "filesystem": {
            "connector":"stdio","command":"cmd",
            "args":["/c","npx","-y","@modelcontextprotocol/server-filesystem",
                    r"D:\one drive\study\ARCEE AI INTERNSHIP\mcp data gen minimal"]
        },
        "memory": {
            "connector":"stdio","command":"npx",
            "args":["-y","@modelcontextprotocol/server-memory"]
        },
        "playwright": {
            "connector":"stdio","command":"npx",
            "args":["@playwright/mcp@latest"],
            "env":{"DISPLAY":":1"}
        },
        "sequential-thinking": {
            "connector":"stdio","command":"cmd",
            "args":["/c","npx","-y",
                    "@modelcontextprotocol/server-sequential-thinking"]
        }
    }
}

TASKS = [
    ("github",
     "In the repo YOUR‑USER/YOUR‑REPO on branch main, create TEST.md "
     "with the content 'Hello MCP'."),
    ("filesystem",
     "List the files and directories in the current directory."),
    ("memory",
     "Store the sentence 'Memory works' under the key test_key then retrieve it."),
    ("playwright",
     "Open https://example.org and give me the page title."),
    ("sequential-thinking",
     "Plan in two steps: first say 'step‑one complete', then say 'step‑two complete'.")
]

# ──────────────  JSON‑L Logger  ────────────── #
class JSONLLogger(BaseCallbackHandler):
    def __init__(self, path: Path):
        self.f = path.open("a", encoding="utf-8")
    async def on_tool_end(self, tool_name, output, **kw):
        self.f.write(json.dumps({
            "ts": round(time.time(),3),
            "tool": tool_name,
            "args": kw.get("input_args", {}),
            "output": output
        })+"\n")
        self.f.flush()
    def close(self): self.f.close()

# ───────────────────  MAIN  ─────────────────── #
async def main():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY missing in .env")

    client = MCPClient.from_dict(CONFIG)

    llm   = ChatOpenAI(model="gpt-4o")
    agent = MCPAgent(llm=llm, client=client, max_steps=30)

    # Initialise once so we can attach callbacks
    await agent.initialize()

    log_dir = Path("logs"); log_dir.mkdir(exist_ok=True)
    logger  = JSONLLogger(log_dir / "tool_calls.jsonl")
    for t in getattr(agent, "tools", []):
        if hasattr(t, "callbacks"): t.callbacks.append(logger)

    try:
        for server, prompt in TASKS:
            print(f"\n▶ {server}: {prompt}")
            try:
                ans = await agent.run(prompt, manage_connector=False)
                print("   →", ans)
            except Exception as e:
                print("   ✗ ERROR:", e)
    finally:
        # graceful shutdown (no Windows warnings)
        try:
            await client.close_all_sessions()
        finally:
            logger.close()
        await asyncio.sleep(0)   # allow transports to close

    print("\nTrace written to", logger.f.name)

if __name__ == "__main__":
    asyncio.run(main())



