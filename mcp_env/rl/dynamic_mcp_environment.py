import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

from mcp_tool_executor import DynamicMCPToolExecutor
from dynamic_task_generator import DynamicTaskGenerator
from reward_generator import RewardGenerator, RewardConfig

@dataclass
class MCPState:
    """State representation for the MCP environment."""
    task: str
    conversation_history: List[Dict[str, str]]
    step: int
    max_steps: int
    available_tools: List[str]
    tool_executor: DynamicMCPToolExecutor
    
    def to_observation(self) -> str:
        """Convert state to observation string."""
        obs = {
            "task": self.task,
            "step": f"{self.step}/{self.max_steps}",
            "available_tools": self.available_tools,
            "conversation": self.conversation_history,
            "tool_info": self._get_tool_descriptions()
        }
        return json.dumps(obs, indent=2)
    
    def _get_tool_descriptions(self) -> Dict[str, str]:
        """Get descriptions of available tools."""
        descriptions = {}
        for tool_name in self.available_tools:
            if tool_name != "answer":
                tool_info = self.tool_executor.get_tool_info(tool_name)
                if tool_info:
                    descriptions[tool_name] = tool_info.description
            else:
                descriptions[tool_name] = "Provide final answer to complete the task"
        return descriptions

class DynamicMCPEnvironment:
    """Dynamic RL Environment that adapts to available MCP servers."""
    
    def __init__(
        self, 
        max_steps: int = 10, 
        servers_dir: Path = None,
        reward_config: RewardConfig = None,
        use_llm_rewards: bool = False
    ):
        self.max_steps = max_steps
        self.tool_executor = DynamicMCPToolExecutor(servers_dir)
        self.task_generator = DynamicTaskGenerator(self.tool_executor)
        self.state = None
        
        # Initialize reward generator
        if reward_config is None:
            reward_config = RewardConfig(use_llm_rewards=use_llm_rewards)
        self.reward_generator = RewardGenerator(reward_config)
        
        # Print initialization summary
        self._print_initialization_summary()
    
    def _print_initialization_summary(self):
        """Print a summary of what was discovered and loaded."""
        print(f"\n🎮 DYNAMIC MCP ENVIRONMENT INITIALIZED")
        print("=" * 50)
        
        servers = self.tool_executor.servers
        tools = self.tool_executor.tools
        
        print(f"📊 Summary:")
        print(f"   • Servers loaded: {len(servers)}")
        print(f"   • Total tools: {len(tools)}")
        print(f"   • Reward Generator: {'LLM-based' if self.reward_generator.config.use_llm_rewards else 'Heuristic'}")
        
        if servers:
            print(f"\n🖥️ Available Servers:")
            for server_name, server_info in servers.items():
                tool_names = [t.name for t in server_info.tools]
                print(f"   • {server_name}: {tool_names}")
        
        task_types = self.task_generator.get_available_task_types()
        print(f"\n🎯 Available Task Types: {task_types}")
        
        if not servers:
            print(f"\n⚠️ WARNING: No servers were loaded!")
            print(f"   Check that your servers directory exists and contains valid server files.")
    
    def reset(self, task: str = None, task_type: str = None) -> str:
        """Reset the environment and return initial observation."""
        if task is None:
            task = self.task_generator.generate_task(task_type)
        
        self.state = MCPState(
            task=task,
            conversation_history=[
                {"role": "system", "content": f"Task: {task}"},
                {"role": "system", "content": "Use the available tools to complete this task. When finished, use the 'answer' tool."}
            ],
            step=0,
            max_steps=self.max_steps,
            available_tools=self.tool_executor.get_available_tools(),
            tool_executor=self.tool_executor
        )
        
        return self.state.to_observation()
    
    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Execute an action and return (observation, reward, done, info)."""
        if not self.state:
            raise RuntimeError("Environment not initialized. Call reset() first.")
            
        if self.state.step >= self.max_steps:
            return self.state.to_observation(), -0.1, True, {"reason": "max_steps_reached"}
        
        self.state.step += 1
        
        try:
            # Parse action
            action_data = json.loads(action)
            tool_name = action_data.get("tool")
            args = action_data.get("args", {})
            
            if not tool_name:
                return self._handle_invalid_action("Missing tool name")
            
            # Check if this is an answer action
            is_answer = (tool_name == "answer")
            
            # Execute tool
            result = self.tool_executor.execute_tool(tool_name, args)
            
            # Add to conversation history
            self.state.conversation_history.append({
                "role": "assistant", 
                "content": f"Tool: {tool_name}, Args: {args}"
            })
            self.state.conversation_history.append({
                "role": "system", 
                "content": f"Result: {json.dumps(result)}"
            })
            
            # Calculate reward using the reward generator
            done = (is_answer or self.state.step >= self.max_steps)
            
            reward, reward_info = self.reward_generator.calculate_reward(
                action=action_data,
                result=result,
                state=self.state,
                done=done,
                is_answer=is_answer
            )
            
            # Add reward info to result
            info = {"result": result, "reward_info": reward_info}
            
            # Handle answer action
            if is_answer:
                info["answer"] = args.get("answer", "")
                info["task_completed"] = True
                info["steps_taken"] = self.state.step
            
            # Check for errors
            if "error" in result:
                info["error"] = result["error"]
            
            return self.state.to_observation(), reward, done, info
            
        except json.JSONDecodeError:
            return self._handle_invalid_action("Invalid JSON format")
        except Exception as e:
            return self._handle_invalid_action(f"Action execution error: {str(e)}")
    
    def _handle_invalid_action(self, error_msg: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Handle invalid actions."""
        self.state.conversation_history.append({
            "role": "system",
            "content": f"Error: {error_msg}"
        })
        
        # Use reward generator for invalid action
        reward = self.reward_generator.config.invalid_action_penalty
        
        return self.state.to_observation(), reward, True, {"error": error_msg}
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the environment."""
        servers_info = {}
        for server_name, server_info in self.tool_executor.servers.items():
            servers_info[server_name] = {
                "tools": [t.name for t in server_info.tools],
                "tool_descriptions": {t.name: t.description for t in server_info.tools}
            }
        
        return {
            "servers": servers_info,
            "total_tools": len(self.tool_executor.tools),
            "available_tools": self.tool_executor.get_available_tools(),
            "task_types": self.task_generator.get_available_task_types(),
            "action_format": '{"tool": "tool_name", "args": {"param": "value"}}',
            "max_steps": self.max_steps,
            "reward_type": "llm" if self.reward_generator.config.use_llm_rewards else "heuristic"
        }