#!/usr/bin/env python3
"""
step_reward_evaluator.py
------------------------
Add this to your MCP pipeline to evaluate each tool execution step-by-step.
Integrates with MCPQueryExecutor to provide SWIRL-style rewards.
"""

import csv
import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import openai
from dataclasses import dataclass, asdict


@dataclass
class StepReward:
    """Reward components for a single tool execution step."""
    tool_relevance: float      # 0.0-1.0: Is this the right tool?
    progress_made: float       # -1.0-1.0: Forward/backward progress
    output_quality: float      # 0.0-1.0: Quality of tool output
    efficiency: float          # 0.0-1.0: Could better tool be used?
    total: float              # Weighted sum of components
    explanation: str          # LLM's reasoning


class StepRewardEvaluator:
    """Evaluates each tool execution step using LLM-as-judge approach."""
    
    def __init__(
        self,
        api_key: str,
        api_base: str = "https://openrouter.ai/api/v1",
        model: str = "openai/gpt-4o-mini",
        weights: Dict[str, float] = None
    ):
        """
        Initialize the step reward evaluator.
        
        Args:
            api_key: OpenAI API key
            api_base: API base URL
            model: Model to use for evaluation
            weights: Component weights for total score
        """
        self.client = openai.OpenAI(api_key=api_key, base_url=api_base)
        self.model = model
        self.weights = weights or {
            "tool_relevance": 0.3,
            "progress_made": 0.3,
            "output_quality": 0.2,
            "efficiency": 0.2
        }
    
    async def evaluate_step(
        self,
        query: str,
        task_context: str,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_output: Any,
        previous_steps: List[Dict],
        available_tools: List[str]
    ) -> StepReward:
        """
        Evaluate a single tool execution step.
        
        Args:
            query: Original user query
            task_context: Current task state/context
            tool_name: Name of tool that was executed
            tool_input: Input provided to the tool
            tool_output: Output from the tool
            previous_steps: List of previous tool executions
            available_tools: List of available tool names
            
        Returns:
            StepReward with component scores
        """
        # Build evaluation prompt
        prompt = self._build_evaluation_prompt(
            query, task_context, tool_name, tool_input,
            tool_output, previous_steps, available_tools
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            # Parse response
            content = response.choices[0].message.content
            return self._parse_evaluation(content)
            
        except Exception as e:
            # Fallback to heuristic evaluation
            return self._heuristic_evaluation(
                tool_name, tool_output, previous_steps
            )
    
    def _get_system_prompt(self) -> str:
        """System prompt for the evaluator."""
        return """You are an expert evaluator for AI tool usage. Your job is to evaluate each tool execution step.

For each step, provide scores for:
1. tool_relevance (0.0-1.0): Is this the right tool for the current task state?
2. progress_made (-1.0 to 1.0): Does this move toward (-1=away, 0=neutral, 1=toward) task completion?
3. output_quality (0.0-1.0): Is the output well-formed and useful?
4. efficiency (0.0-1.0): Is this the most efficient tool choice?

Respond in JSON format:
{
    "tool_relevance": 0.8,
    "progress_made": 0.6,
    "output_quality": 0.9,
    "efficiency": 0.7,
    "explanation": "Brief explanation of scores"
}"""
    
    def _build_evaluation_prompt(
        self, query: str, task_context: str, tool_name: str,
        tool_input: Dict, tool_output: Any, previous_steps: List[Dict],
        available_tools: List[str]
    ) -> str:
        """Build the evaluation prompt."""
        # Summarize previous steps
        prev_summary = ""
        if previous_steps:
            prev_summary = "Previous steps:\n"
            for i, step in enumerate(previous_steps[-3:], 1):  # Last 3 steps
                prev_summary += f"{i}. {step['tool_name']} -> {str(step['tool_output'])[:100]}...\n"
        
        return f"""Evaluate this tool execution step:

ORIGINAL QUERY: {query}

CURRENT CONTEXT: {task_context}

{prev_summary}

CURRENT STEP:
Tool: {tool_name}
Input: {json.dumps(tool_input, indent=2)}
Output: {str(tool_output)[:500]}...

AVAILABLE TOOLS: {', '.join(available_tools[:10])}

Evaluate the quality of this tool choice and execution."""
    
    def _parse_evaluation(self, content: str) -> StepReward:
        """Parse LLM evaluation response."""
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                
                # Calculate total score
                total = sum(
                    data.get(k, 0) * self.weights.get(k, 0)
                    for k in self.weights
                )
                
                return StepReward(
                    tool_relevance=data.get("tool_relevance", 0.5),
                    progress_made=data.get("progress_made", 0),
                    output_quality=data.get("output_quality", 0.5),
                    efficiency=data.get("efficiency", 0.5),
                    total=total,
                    explanation=data.get("explanation", "No explanation provided")
                )
        except:
            pass
        
        # Fallback
        return self._heuristic_evaluation("", None, [])
    
    def _heuristic_evaluation(
        self, tool_name: str, tool_output: Any, previous_steps: List[Dict]
    ) -> StepReward:
        """Fallback heuristic evaluation."""
        # Simple heuristics
        relevance = 0.7  # Assume somewhat relevant
        progress = 0.5 if tool_output and not isinstance(tool_output, dict) or (
            isinstance(tool_output, dict) and not tool_output.get("error")
        ) else -0.2
        quality = 0.8 if tool_output else 0.2
        
        # Check for repeated tools (inefficiency)
        recent_tools = [s['tool_name'] for s in previous_steps[-3:]]
        efficiency = 0.4 if tool_name in recent_tools else 0.8
        
        total = (
            relevance * self.weights["tool_relevance"] +
            progress * self.weights["progress_made"] +
            quality * self.weights["output_quality"] +
            efficiency * self.weights["efficiency"]
        )
        
        return StepReward(
            tool_relevance=relevance,
            progress_made=progress,
            output_quality=quality,
            efficiency=efficiency,
            total=total,
            explanation="Heuristic evaluation (LLM evaluation failed)"
        )


class RewardTracker:
    """Tracks and saves step-by-step rewards for GRPO training."""
    
    def __init__(self, output_dir: Path = Path("reward_trajectories")):
        """Initialize the reward tracker."""
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trajectories = []
        self.current_trajectory = None
    
    def start_trajectory(self, query_id: str, query_text: str):
        """Start tracking a new query trajectory."""
        self.current_trajectory = {
            "query_id": query_id,
            "query_text": query_text,
            "steps": [],
            "total_reward": 0,
            "timestamp": datetime.now().isoformat()
        }
    
    def add_step(
        self,
        step_num: int,
        tool_name: str,
        tool_input: Dict,
        tool_output: Any,
        reward: StepReward
    ):
        """Add a step to the current trajectory."""
        if not self.current_trajectory:
            return
        
        step_data = {
            "step": step_num,
            "tool_name": tool_name,
            "tool_input": json.dumps(tool_input),
            "tool_output": str(tool_output)[:1000],  # Truncate long outputs
            "reward_components": asdict(reward),
            "step_reward": reward.total
        }
        
        self.current_trajectory["steps"].append(step_data)
        self.current_trajectory["total_reward"] += reward.total
    
    def end_trajectory(self, final_answer: str = None):
        """End the current trajectory and save it."""
        if not self.current_trajectory:
            return
        
        self.current_trajectory["final_answer"] = final_answer
        self.current_trajectory["num_steps"] = len(self.current_trajectory["steps"])
        
        # Calculate average step reward
        if self.current_trajectory["steps"]:
            avg_reward = (
                self.current_trajectory["total_reward"] / 
                len(self.current_trajectory["steps"])
            )
            self.current_trajectory["avg_step_reward"] = avg_reward
        
        self.trajectories.append(self.current_trajectory)
        self.current_trajectory = None
    
    def save_to_csv(self, filename: str = None) -> Path:
        """Save all trajectories to CSV format for GRPO training."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reward_trajectories_{timestamp}.csv"
        
        filepath = self.output_dir / filename
        
        # Flatten trajectories for CSV
        rows = []
        for traj in self.trajectories:
            base_row = {
                "query_id": traj["query_id"],
                "query_text": traj["query_text"],
                "final_answer": traj.get("final_answer", ""),
                "total_reward": traj["total_reward"],
                "num_steps": traj["num_steps"],
                "avg_step_reward": traj.get("avg_step_reward", 0)
            }
            
            # Add each step as a separate row
            for step in traj["steps"]:
                row = {**base_row}
                row.update({
                    "step_num": step["step"],
                    "tool_name": step["tool_name"],
                    "tool_input": step["tool_input"],
                    "tool_output": step["tool_output"],
                    "step_reward": step["step_reward"],
                    "tool_relevance": step["reward_components"]["tool_relevance"],
                    "progress_made": step["reward_components"]["progress_made"],
                    "output_quality": step["reward_components"]["output_quality"],
                    "efficiency": step["reward_components"]["efficiency"],
                    "reward_explanation": step["reward_components"]["explanation"]
                })
                rows.append(row)
        
        # Write to CSV
        if rows:
            fieldnames = rows[0].keys()
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        
        return filepath
    
    def save_trajectories_json(self, filename: str = None) -> Path:
        """Save trajectories in JSON format for detailed analysis."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trajectories_detailed_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.trajectories, f, indent=2)
        
        return filepath


# Integration with MCPQueryExecutor
# Add this to your MCPQueryExecutor class modifications:

def integrate_step_rewards(executor_self):
    """
    Add this method to MCPQueryExecutor to integrate step rewards.
    Call this in __init__ after initializing other components.
    """
    # Initialize reward evaluator and tracker
    executor_self.reward_evaluator = StepRewardEvaluator(
        api_key=executor_self.api_key,
        api_base=executor_self.api_base
    )
    executor_self.reward_tracker = RewardTracker()
    
    # Store query context for reward evaluation
    executor_self.current_query_context = ""
    executor_self.previous_steps = []


async def enhanced_tool_callback(executor_self, tool_call_data):
    """
    Enhanced version of _tool_callback that includes reward evaluation.
    Replace the original _tool_callback with this version.
    """
    # Original tracking logic
    if not executor_self.current_query_id:
        return
    
    # Get current step number
    current_step = tool_call_data.get("step", 0)
    
    # Evaluate the tool execution step
    if hasattr(executor_self, 'reward_evaluator'):
        try:
            # Build task context from recent outputs
            task_context = executor_self.current_query_context
            if executor_self.previous_steps:
                recent_outputs = [str(s.get("tool_output", ""))[:200] 
                                for s in executor_self.previous_steps[-2:]]
                task_context += "\nRecent outputs: " + " | ".join(recent_outputs)
            
            # Evaluate step
            step_reward = await executor_self.reward_evaluator.evaluate_step(
                query=executor_self.current_query,
                task_context=task_context,
                tool_name=tool_call_data["tool_name"],
                tool_input=tool_call_data["input"],
                tool_output=tool_call_data["output"],
                previous_steps=executor_self.previous_steps,
                available_tools=list(executor_self.tools.keys())
            )
            
            # Add reward to tracking data
            tool_call_data["step_reward"] = asdict(step_reward)
            
            # Track reward
            executor_self.reward_tracker.add_step(
                step_num=current_step,
                tool_name=tool_call_data["tool_name"],
                tool_input=tool_call_data["input"],
                tool_output=tool_call_data["output"],
                reward=step_reward
            )
            
            # Update context
            executor_self.previous_steps.append(tool_call_data)
            
        except Exception as e:
            print(f"Warning: Step reward evaluation failed: {e}")
    
    # Continue with original tracking logic...
    # (Include the rest of the original _tool_callback code here)


# Usage in your pipeline:
"""
# In MCPQueryExecutor.__init__, add:
self.integrate_step_rewards()

# Replace _tool_callback with enhanced_tool_callback

# In run_query method, add:
# Before running query:
self.reward_tracker.start_trajectory(tag, prompt)
self.previous_steps = []

# After query completes:
self.reward_tracker.end_trajectory(rec.get("answer"))

# In execute_queries method, after all queries:
csv_path = self.reward_tracker.save_to_csv()
json_path = self.reward_tracker.save_trajectories_json()
print(f"💰 Rewards saved to: {csv_path}")
"""