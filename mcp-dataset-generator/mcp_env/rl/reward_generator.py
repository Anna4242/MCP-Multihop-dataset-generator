# mcp_env/rl/reward_generator.py
"""
Reward Generator for MCP Environment.

Provides both heuristic-based and LLM-based reward generation for the MCP environment,
allowing for process-based evaluation of tool use steps.
"""

import json
import os
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

# Optional LLM integration
try:
    import openai
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file
    HAS_LLM = True
except ImportError:
    HAS_LLM = False

@dataclass
class RewardConfig:
    """Configuration for reward generation."""
    # Basic rewards
    step_penalty: float = -0.01           # Small penalty for each step to encourage efficiency
    error_penalty: float = -0.3           # Penalty for errors
    max_steps_penalty: float = -0.1       # Penalty for reaching max steps
    invalid_action_penalty: float = -0.5  # Penalty for invalid actions
    
    # Answer rewards
    answer_base_reward: float = 0.5       # Base reward for providing an answer
    answer_quality_bonus: float = 0.3     # Bonus for good quality answer
    efficiency_bonus_factor: float = 0.1  # Multiplier for step efficiency bonus
    
    # Process rewards
    process_base_reward: float = 0.1      # Base reward for good process
    good_tool_selection_bonus: float = 0.2  # Bonus for appropriate tool selection
    good_args_bonus: float = 0.1          # Bonus for well-formed tool arguments
    good_query_bonus: float = 0.2         # Bonus for good search queries
    
    # LLM-based reward
    use_llm_rewards: bool = False         # Whether to use LLM for reward generation
    llm_reward_weight: float = 0.7        # Weight of LLM reward vs heuristic reward
    llm_model: str = "openai/gpt-4o-mini" # Model to use for LLM rewards
    llm_temperature: float = 0.1          # Temperature for LLM generation


class RewardGenerator:
    """Generates rewards for actions in the MCP environment."""
    
    def __init__(self, config: Optional[RewardConfig] = None):
        """Initialize the reward generator with given config."""
        self.config = config or RewardConfig()
        self.llm_client = None
        
        # Initialize LLM client if using LLM rewards
        if self.config.use_llm_rewards and HAS_LLM:
            api_key = os.getenv("OPENAI_API_KEY")
            api_base = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
            
            if api_key:
                self.llm_client = openai.OpenAI(
                    api_key=api_key,
                    base_url=api_base
                )
            else:
                print("⚠️ Warning: use_llm_rewards is enabled but OPENAI_API_KEY not found")
                self.config.use_llm_rewards = False
    
    def calculate_reward(
        self, 
        action: Dict[str, Any], 
        result: Dict[str, Any], 
        state: 'MCPState',  # Forward reference to avoid circular import
        done: bool, 
        is_answer: bool
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate reward for the given action and result.
        
        Args:
            action: The action taken (tool and args)
            result: The result of the action
            state: The current state of the environment
            done: Whether the episode is done
            is_answer: Whether this is an answer action
            
        Returns:
            Tuple of (reward, reward_info)
        """
        # Basic reward components
        heuristic_reward, reward_info = self._calculate_heuristic_reward(
            action, result, state, done, is_answer
        )
        
        # LLM-based process reward if enabled
        llm_reward = 0.0
        if self.config.use_llm_rewards and self.llm_client:
            llm_reward = self._calculate_llm_reward(action, result, state)
            reward_info["llm_reward"] = llm_reward
            
            # Combine rewards using weight
            final_reward = (
                (1 - self.config.llm_reward_weight) * heuristic_reward + 
                self.config.llm_reward_weight * llm_reward
            )
        else:
            final_reward = heuristic_reward
        
        reward_info["final_reward"] = final_reward
        return final_reward, reward_info
    
    def _calculate_heuristic_reward(
        self,
        action: Dict[str, Any],
        result: Dict[str, Any],
        state: 'MCPState',
        done: bool,
        is_answer: bool
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate reward based on heuristic rules."""
        reward_info = {
            "components": {}
        }
        
        # Start with step penalty
        reward = self.config.step_penalty
        reward_info["components"]["step_penalty"] = self.config.step_penalty
        
        # Check for errors
        if "error" in result:
            reward += self.config.error_penalty
            reward_info["components"]["error_penalty"] = self.config.error_penalty
            reward_info["error"] = result["error"]
        
        # Handle answer actions
        if is_answer:
            answer = action.get("args", {}).get("answer", "")
            answer_reward = self._calculate_answer_reward(answer, state)
            reward += answer_reward
            reward_info["components"]["answer_reward"] = answer_reward
            
            # Check answer quality
            answer_quality = self._evaluate_answer_quality(answer, state)
            reward_info["answer_quality"] = answer_quality
        
        # Handle tool actions (non-answer)
        else:
            tool_name = action.get("tool", "")
            args = action.get("args", {})
            
            # Evaluate tool selection
            tool_selection_score = self._evaluate_tool_selection(tool_name, state)
            if tool_selection_score > 0:
                tool_bonus = self.config.good_tool_selection_bonus * tool_selection_score
                reward += tool_bonus
                reward_info["components"]["tool_selection_bonus"] = tool_bonus
            
            # Evaluate arguments
            args_score = self._evaluate_args(tool_name, args, state)
            if args_score > 0:
                args_bonus = self.config.good_args_bonus * args_score
                reward += args_bonus
                reward_info["components"]["args_bonus"] = args_bonus
            
            # Special case for search queries
            if "search" in tool_name.lower() and "query" in args:
                query = args.get("query", "")
                query_score = self._evaluate_search_query(query, state)
                if query_score > 0:
                    query_bonus = self.config.good_query_bonus * query_score
                    reward += query_bonus
                    reward_info["components"]["query_bonus"] = query_bonus
        
        # Process quality bonus
        process_score = self._evaluate_process(action, result, state)
        if process_score > 0:
            process_bonus = self.config.process_base_reward * process_score
            reward += process_bonus
            reward_info["components"]["process_bonus"] = process_bonus
        
        # Check if max steps reached
        if done and state.step >= state.max_steps and not is_answer:
            reward += self.config.max_steps_penalty
            reward_info["components"]["max_steps_penalty"] = self.config.max_steps_penalty
        
        return reward, reward_info
    
    def _calculate_answer_reward(self, answer: str, state: 'MCPState') -> float:
        """Calculate reward for answer action."""
        base_reward = self.config.answer_base_reward
        
        # Bonus for reasonable answer length
        if 10 <= len(answer) <= 500:
            base_reward += self.config.answer_quality_bonus
        
        # Bonus for efficiency (fewer steps)
        efficiency_bonus = max(0, (state.max_steps - state.step) * self.config.efficiency_bonus_factor)
        
        return base_reward + efficiency_bonus
    
    def _evaluate_answer_quality(self, answer: str, state: 'MCPState') -> float:
        """Evaluate the quality of an answer based on heuristics."""
        # Simple length-based evaluation for now
        if len(answer) < 5:
            return 0.1  # Too short
        elif len(answer) > 1000:
            return 0.3  # Too verbose
        elif 100 <= len(answer) <= 500:
            return 0.8  # Good length
        else:
            return 0.5  # Acceptable
    
    def _evaluate_tool_selection(self, tool_name: str, state: 'MCPState') -> float:
        """Evaluate if the selected tool is appropriate for the current state."""
        # Simple implementation - can be expanded with more sophisticated logic
        if tool_name not in state.available_tools:
            return 0.0
            
        # Check if the tool is relevant to the task
        task_lower = state.task.lower()
        
        if "search" in tool_name.lower() and any(term in task_lower for term in ["search", "find", "look up", "research"]):
            return 1.0
        elif "file" in tool_name.lower() and any(term in task_lower for term in ["file", "read", "write", "directory"]):
            return 1.0
        elif "list" in tool_name.lower() and any(term in task_lower for term in ["list", "show", "display"]):
            return 1.0
        elif "query" in tool_name.lower() and any(term in task_lower for term in ["database", "query", "data", "sql"]):
            return 1.0
        elif "repo" in tool_name.lower() and any(term in task_lower for term in ["git", "github", "repository", "code"]):
            return 1.0
            
        # Default - somewhat appropriate
        return 0.5
    
    def _evaluate_args(self, tool_name: str, args: Dict[str, Any], state: 'MCPState') -> float:
        """Evaluate if the arguments are appropriate for the selected tool."""
        # Check if required args are present
        if "search" in tool_name.lower() and "query" not in args:
            return 0.0
        elif "file" in tool_name.lower() and "path" not in args and "filename" not in args:
            return 0.0
        elif "query" in tool_name.lower() and "sql" not in args:
            return 0.0
            
        # All required args present
        return 1.0
    
    def _evaluate_search_query(self, query: str, state: 'MCPState') -> float:
        """Evaluate the quality of a search query."""
        if not query or len(query) < 2:
            return 0.0
            
        # Check if query relates to task
        task_words = set(state.task.lower().split())
        query_words = set(query.lower().split())
        
        # Calculate word overlap
        overlap = len(task_words.intersection(query_words))
        if overlap > 0:
            return min(1.0, overlap / 3)  # Scale by overlap, max at 1.0
            
        return 0.5  # Default - somewhat related
    
    def _evaluate_process(self, action: Dict[str, Any], result: Dict[str, Any], state: 'MCPState') -> float:
        """Evaluate the overall process quality based on action and context."""
        # Simple implementation for now
        # Check conversation history for repeated actions
        if len(state.conversation_history) >= 4:
            # Check if this action is similar to a recent action
            tool_name = action.get("tool", "")
            args = action.get("args", {})
            
            for msg in state.conversation_history[-4:]:
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if tool_name in content and str(args) in content:
                        return 0.0  # Repeated action, no process bonus
        
        return 0.7  # Default - generally good process
    
    def _calculate_llm_reward(
        self, 
        action: Dict[str, Any], 
        result: Dict[str, Any], 
        state: 'MCPState'
    ) -> float:
        """Calculate reward using an LLM to evaluate process quality."""
        if not self.llm_client:
            return 0.0
            
        # Create prompt for the LLM
        system_prompt = """You are an AI assistant evaluating the quality of tool use in a multi-step reasoning process.
Your job is to assess whether the latest action is reasonable given the task and conversation history.
Rate the action on a scale from 0.0 to 1.0, where:
- 0.0: Completely irrelevant or harmful action
- 0.3: Poor action that doesn't help solve the task
- 0.5: Mediocre action with limited usefulness
- 0.7: Good action that makes progress toward the goal
- 1.0: Excellent action that directly advances solving the task

Provide your rating as a single number between 0.0 and 1.0."""

        # Format the user message
        user_message = f"""
TASK: {state.task}
STEP: {state.step}/{state.max_steps}

AVAILABLE TOOLS: {', '.join(state.available_tools)}

CONVERSATION HISTORY:
{json.dumps(state.conversation_history[-3:], indent=2)}

LATEST ACTION:
Tool: {action.get('tool', '')}
Arguments: {json.dumps(action.get('args', {}), indent=2)}

RESULT:
{json.dumps(result, indent=2)}

Rate this action on a scale from 0.0 to 1.0 based on:
1. Is this a reasonable tool to use given the task?
2. Are the arguments appropriate?
3. Does this action make progress toward solving the task?
4. Is this action efficient or wasteful?

Provide ONLY a number between 0.0 and 1.0 as your response.
"""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=self.config.llm_temperature,
                max_tokens=10
            )
            
            # Extract and parse the score
            response_text = response.choices[0].message.content.strip()
            
            try:
                # Extract number from response
                import re
                match = re.search(r'(\d+\.\d+|\d+)', response_text)
                if match:
                    score = float(match.group(1))
                    # Clamp to valid range
                    score = max(0.0, min(1.0, score))
                    return score
                else:
                    return 0.5  # Default if no valid score found
            except ValueError:
                return 0.5  # Default if parsing fails
                
        except Exception as e:
            print(f"⚠️ LLM reward calculation failed: {e}")
            return 0.5  # Default if LLM call fails