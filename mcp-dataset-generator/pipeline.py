#!/usr/bin/env python3
"""
pipeline.py
──────────────────────────────────────────────────────────────────────────────
1.  Connect to MCP → extra_info/<server>.json
2.  Generate dataset rows  → datasets/*.csv
3.  DeepEval scoring       → console
"""

from __future__ import annotations
import asyncio, json
from pathlib import Path
from datetime import datetime

from settings import OPENAI_API_KEY, MCP_CONFIG
from env_generator import get_mcp_tools, generate_env_entry
from dataset_generator import DatasetGenerator
from evaluate_mcp_dataset import MCPDatasetEvaluator

EXTRA_INFO_DIR = Path("extra_info")
DATASET_DIR    = Path("datasets")
USE_REAL_TOOLS = True

# ── 1. build env JSON ─────────────────────────────────────────────────────
async def build_env(config_file: str):
    EXTRA_INFO_DIR.mkdir(exist_ok=True)
    tools, client, server = await get_mcp_tools(config_file)
    if not tools:
        raise RuntimeError("No MCP tools returned.")

    env_entry = await generate_env_entry(tools, use_real_tools=USE_REAL_TOOLS)
    env_path  = EXTRA_INFO_DIR / f"{server}.json"
    env_path.write_text(json.dumps(env_entry, indent=2, ensure_ascii=False))
    print(f"📦  Env JSON written to {env_path}")

    if hasattr(client, "close"):
        try:
            await client.close()
        except Exception:
            pass
    return env_path, tools

# ── 2. build dataset CSV ──────────────────────────────────────────────────
async def build_dataset(tools, env_path: Path):
    DATASET_DIR.mkdir(exist_ok=True)
    gen = DatasetGenerator()

    entries = await gen.generate_entries(
        tools=tools,
        num_entries=5,
        data_source="syntool_re_call"
    )

    # (Optional) embed env code into each row
    # env_blob = json.loads(env_path.read_text())
    # for e in entries:
    #     e["extra_info"]["env"] = env_blob["env"]

    csv_name = f"mcp_dataset_{datetime.now():%Y%m%d_%H%M%S}"
    csv_path = await gen.save_dataset_csv(entries, DATASET_DIR, csv_name)
    return Path(csv_path)

# ── 3. orchestrate ────────────────────────────────────────────────────────
async def main():
    env_path, tools = await build_env(MCP_CONFIG)
    csv_path        = await build_dataset(tools, env_path)

    print("\n🔍  DeepEval scoring …\n")
    MCPDatasetEvaluator(csv_path).run()

if __name__ == "__main__":
    asyncio.run(main())

