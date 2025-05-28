#!/usr/bin/env python3
"""
run_queries_with_agent.py
-------------------------
Re‑use your existing one‑liner pattern:

    llm   = ChatOpenAI(model="gpt-4o")
    agent = MCPAgent(llm=llm, client=client, max_steps=30)
    answer = await agent.run(prompt)

… but do it for every prompt stored in the JSON produced by the
multi‑hop generator.

The script prints each answer to stdout and stops there – no extra
logging / saving unless you turn `SAVE_JSON = True` below.
"""

import asyncio
import contextlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from mcp_use import MCPAgent, MCPClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────
#  CONFIG – edit the two paths if needed
# ────────────────────────────────────────────────────────────────
MCP_CONFIG   = Path(
    r"D:\one drive\study\ARCEE AI INTERNSHIP\mcp data gen minimal\mcp-dataset-generator\tests\mcp_servers.json"
)
QUERIES_FILE = Path(
    r"D:\one drive\study\ARCEE AI INTERNSHIP\mcp data gen minimal\queries\cross_server_multi_hop_queries_20250520_213935.json"
)

SAVE_JSON = True           # set False if you do NOT want a .json output
OUT_DIR   = Path("execution_results")

# Timeout settings (in seconds)
QUERY_TIMEOUT = 120        # Maximum time to wait for a single query
CLEANUP_TIMEOUT = 10       # Maximum time to wait during cleanup

# ────────────────────────────────────────────────────────────────
#  HELPER
# ────────────────────────────────────────────────────────────────
def check_file_exists(path: Path) -> bool:
    """Check if file exists and log appropriate message."""
    if not path.exists():
        logger.error(f"File not found: {path}")
        return False
    return True

def read_queries(path: Path) -> List[Dict]:
    """Load the JSON list; accept 'query' or 'question' field"""
    if not check_file_exists(path):
        return []
    
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError(f"{path} must contain a JSON array")
        
        # Validate queries
        valid_queries = []
        for i, q in enumerate(data):
            if not (q.get("query") or q.get("question")):
                logger.warning(f"Skipping item {i+1}: missing 'query' or 'question' field")
                continue
            valid_queries.append(q)
        
        return valid_queries
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in file: {path}")
        return []
    except Exception as e:
        logger.error(f"Error reading queries from {path}: {e}")
        return []

def save_results(results: List[Dict], out_dir: Path) -> Optional[Path]:
    """Save results to JSON file with timestamp."""
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = out_dir / f"answers_{ts}.json"
        out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        return out_file
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        return None

# ────────────────────────────────────────────────────────────────
#  MAIN
# ────────────────────────────────────────────────────────────────
async def run_query(agent: MCPAgent, prompt: str, tag: str, idx: int, total: int) -> Dict:
    """Run a single query with timeout and error handling."""
    rec = {"id": tag, "prompt": prompt, "answer": None, "error": None}
    
    try:
        # Apply timeout to each query
        answer = await asyncio.wait_for(agent.run(prompt), timeout=QUERY_TIMEOUT)
        rec["answer"] = answer
        print(f"   → {answer[:120]}{'...' if len(answer) > 120 else ''}")
    except asyncio.TimeoutError:
        error_msg = f"Query timed out after {QUERY_TIMEOUT} seconds"
        rec["error"] = error_msg
        logger.error(f"Query {idx}/{total} timed out")
        print(f"   ✗ TIMEOUT: {error_msg}")
    except Exception as exc:
        rec["error"] = str(exc)
        logger.error(f"Query {idx}/{total} failed: {exc}")
        print(f"   ✗ ERROR: {exc}")
    
    return rec

async def run_all() -> int:
    """Main function to run all queries."""
    client = None
    
    try:
        # Load environment variables
        load_dotenv()
        
        # Check if GitHub token is set
        github_token = os.environ.get("MCP_INPUT_github_token")
        if not github_token:
            logger.warning("GitHub token (MCP_INPUT_github_token) not found in environment")
            print("⚠️  Warning: GitHub token not found in environment variables")
            
        # Check if OpenAI API key is set
        openai_key = os.environ.get("OPENAI_API_KEY")
        if not openai_key:
            logger.error("OpenAI API key (OPENAI_API_KEY) not found in environment")
            print("❌ Error: OpenAI API key not found in environment variables")
            return 1
        
        # Check if config files exist
        if not check_file_exists(MCP_CONFIG):
            return 1
            
        # 1) Spin up servers
        print(f"🔄 Initializing MCP client from {MCP_CONFIG}")
        client = MCPClient.from_config_file(str(MCP_CONFIG))
        
        # 2) Build agent
        print("🤖 Building agent with GPT-4o")
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        agent = MCPAgent(llm=llm, client=client, max_steps=30)
        
        # 3) Read prompts
        queries = read_queries(QUERIES_FILE)
        if not queries:
            logger.error(f"No valid queries found in {QUERIES_FILE}")
            print(f"❌ No valid queries found in {QUERIES_FILE}")
            return 1
            
        print(f"📋 Loaded {len(queries)} prompts from {QUERIES_FILE}")
        
        # 4) Process each query
        results = []
        for idx, q in enumerate(queries, 1):
            prompt = q.get("query") or q.get("question") or ""
            tag = q.get("id", f"query_{idx}")
            
            print(f"\n▶ {idx}/{len(queries)}  {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
            
            rec = await run_query(agent, prompt, tag, idx, len(queries))
            results.append(rec)
        
        # 5) Save results if needed
        if SAVE_JSON and results:
            path = save_results(results, OUT_DIR)
            if path:
                print(f"\n💾 Results saved to {path.resolve()}")
        
        return 0
    
    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled by user")
        return 130
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)
        print(f"\n❌ Critical error: {e}")
        return 1
    finally:
        # Proper cleanup with timeout
        
            print("🧹 Cleaning up MCP client sessions...")
           

def main():
    """Entry point with proper asyncio handling."""
    try:
        # Create new event loop and run the main function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        exit_code = loop.run_until_complete(run_all())
        
        # Cleanup
        pending = asyncio.all_tasks(loop)
        if pending:
            print(f"Cancelling {len(pending)} pending tasks...")
            for task in pending:
                task.cancel()
            
            # Allow cancelled tasks to complete with a timeout
            loop.run_until_complete(
                asyncio.wait(pending, timeout=CLEANUP_TIMEOUT, loop=loop)
            )
        
        loop.close()
        return exit_code
    
    except Exception as e:
        print(f"Setup error: {e}")
        return 1

if __name__ == "__main__":
    import os
    sys.exit(main())

