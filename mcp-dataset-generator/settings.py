"""
settings.py
───────────
Centralised configuration loader.

• Place a `.env` file in the project root (same folder as this file).
• All scripts should simply:   from settings import OPENAI_API_KEY, MCP_CONFIG, CSV_DATASET
"""

from pathlib import Path
import os

from dotenv import load_dotenv

# ------------------------------------------------------------------ #
# 1.  Load variables from `.env` (only once for the whole project)
# ------------------------------------------------------------------ #
REPO_ROOT = Path(__file__).parent            # folder containing settings.py
load_dotenv(dotenv_path=REPO_ROOT / ".env", override=False)

# ------------------------------------------------------------------ #
# 2.  Mandatory secret
# ------------------------------------------------------------------ #
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError(
        "OPENAI_API_KEY is not set. "
        "Add it to your .env file or export it in the shell."
    )

# ------------------------------------------------------------------ #
# 3.  Optional convenience variables
# ------------------------------------------------------------------ #
# Path (or name) of the MCP config file you want to use by default
MCP_CONFIG  = os.getenv("MCP_CONFIG", "browser_mcp.json")

# Default CSV dataset path for the evaluator (can be overridden per call)
CSV_DATASET = os.getenv("CSV_DATASET", "")

# ------------------------------------------------------------------ #
# 4.  Make sure the key is also present in `os.environ` for libraries
# ------------------------------------------------------------------ #
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY     # so deepeval / langchain see it
