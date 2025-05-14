#!/usr/bin/env python3
"""
evaluate_mcp_dataset.py
──────────────────────────────────────────────────────────────────────────────
Importable evaluator + optional CLI.

Example:
    from evaluate_mcp_dataset import MCPDatasetEvaluator
    report = MCPDatasetEvaluator("datasets/my.csv").run()

CLI:
    python evaluate_mcp_dataset.py datasets/my.csv
"""

from __future__ import annotations
import json, os, re, sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd
from deepeval import evaluate
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.metrics import TaskCompletionMetric, ToolCorrectnessMetric

from settings import OPENAI_API_KEY          # loads key from .env

# ── helpers ────────────────────────────────────────────────────────────────
_TOOL_RE = re.compile(r'["\']([A-Za-z0-9 _-]{3,80})["\']')

def _parse_tool_calls(code: str, expected: List[str]) -> List[ToolCall]:
    found = set(_TOOL_RE.findall(code))
    matched = [t for t in expected if t in found] or expected
    return [ToolCall(name=t) for t in matched]

def _load_csv(path: Path, n: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, nrows=n)
    for col in ["reward_model", "extra_info.function_schemas"]:
        df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    return df

# ── evaluator class ────────────────────────────────────────────────────────
class MCPDatasetEvaluator:
    """DeepEval Task‑Completion & Tool‑Correctness for an MCP dataset CSV."""

    TASK_THRESHOLD = 0.70
    JUDGE_MODEL    = "gpt-4o"

    def __init__(self,
                 csv_path: str | Path,
                 max_rows: int | None = None,
                 openai_api_key: str | None = None):
        self.csv_path = Path(csv_path)
        self.max_rows = max_rows
        self.api_key  = openai_api_key or OPENAI_API_KEY

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        # DeepEval looks for the key in the environment
        os.environ["OPENAI_API_KEY"] = self.api_key

        self.task_metric = TaskCompletionMetric(
            threshold=self.TASK_THRESHOLD,
            model=self.JUDGE_MODEL
        )
        self.tool_metric = ToolCorrectnessMetric()

    # ------------------------------------------------------------------
    def _build_test_cases(self) -> List[LLMTestCase]:
        df = _load_csv(self.csv_path, self.max_rows)
        cases: List[LLMTestCase] = []

        for _, row in df.iterrows():
            question  = str(row["question"])
            answer    = (row["reward_model"] or [""])[0]
            rule_code = str(row["extra_info.rule"])

            schemas = row["extra_info.function_schemas"]
            expected_names = (
                list(schemas.keys()) if isinstance(schemas, dict)
                else [d["name"] for d in schemas if isinstance(d, dict)]
                if isinstance(schemas, list) else []
            )

            cases.append(
                LLMTestCase(
                    input=question,
                    actual_output=answer,
                    tools_called=_parse_tool_calls(rule_code, expected_names),
                    expected_tools=[ToolCall(name=n) for n in expected_names],
                )
            )
        self._n_cases = len(cases)
        return cases

    # ------------------------------------------------------------------
    def run(self) -> Dict[str, float]:
        """Runs evaluation, prints per‑row and average scores, returns dict."""
        tests   = self._build_test_cases()
        results = evaluate(tests, metrics=[self.task_metric, self.tool_metric])

        old_api = hasattr(results[0], "metric_scores")
        print(f"\n✅  Evaluating {self._n_cases} rows ({self.csv_path})\n")

        for idx, res in enumerate(results):
            if old_api:
                tscore = res.metric_scores[self.task_metric]
                cscore = res.metric_scores[self.tool_metric]
            else:
                tscore, cscore = res
            print(f"[{idx:03}]  TaskCompletion={tscore:.2f}   ToolCorrectness={cscore:.2f}")

        avg_task  = sum((r.metric_scores[self.task_metric] if old_api else r[0]) for r in results) / len(results)
        avg_tools = sum((r.metric_scores[self.tool_metric] if old_api else r[1]) for r in results) / len(results)

        print("\n───── Averages ─────")
        print(f"Task‑Completion : {avg_task:.3f}")
        print(f"Tool‑Correctness: {avg_tools:.3f}")
        print("────────────────────")

        return {"task_completion": avg_task, "tool_correctness": avg_tools}

    # optional one‑liner wrapper --------------------------------------
    @staticmethod
    def run_eval(csv_path: str | Path, max_rows: int | None = None):
        """Convenience static wrapper (sync)."""
        return MCPDatasetEvaluator(csv_path, max_rows=max_rows).run()

# ── CLI entry‑point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage:  python evaluate_mcp_dataset.py  <csv_path>")
    csv = Path(sys.argv[1])
    MCPDatasetEvaluator(csv).run()

