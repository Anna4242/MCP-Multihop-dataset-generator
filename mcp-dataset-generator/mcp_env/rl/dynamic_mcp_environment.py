# mcp_env/rl/dynamic_mcp_environment.py
"""
Dynamic MCP RL Environment that adapts to available servers and provides
a reinforcement learning interface for training LLMs with tool use.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

try:
    from .mcp_tool_executor import DynamicMCPToolExecutor
    from .dynamic_task_generator import DynamicTaskGenerator
except ImportError:
    # Handle direct execution
    from mcp_tool_executor import DynamicMCPToolExecutor
    from dynamic_task_generator import DynamicTaskGenerator

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
    
    def __init__(self, max_steps: int = 10, servers_dir: Path = None):
        self.max_steps = max_steps
        self.tool_executor = DynamicMCPToolExecutor(servers_dir)
        self.task_generator = DynamicTaskGenerator(self.tool_executor)
        self.state = None
        
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
            
            # Check if this was an answer action
            if tool_name == "answer":
                return self._handle_answer_action(result, args)
            
            # Check for errors
            if "error" in result:
                reward = -0.3  # Penalty for errors
                return self.state.to_observation(), reward, False, {"error": result["error"]}
            
            # Normal step reward
            reward = -0.01  # Small step penalty to encourage efficiency
            done = self.state.step >= self.max_steps
            
            if done:
                reward = -0.1  # Penalty for not completing task
            
            return self.state.to_observation(), reward, done, {"result": result}
            
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
        return self.state.to_observation(), -0.5, True, {"error": error_msg}
    
    def _handle_answer_action(self, result: Dict[str, Any], args: Dict[str, Any]) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Handle answer actions and calculate final reward."""
        answer = args.get("answer", "")
        
        # Simple reward based on answer length and task completion
        reward = self._calculate_answer_reward(answer)
        
        return self.state.to_observation(), reward, True, {
            "answer": answer,
            "task_completed": True,
            "steps_taken": self.state.step
        }
    
    def _calculate_answer_reward(self, answer: str) -> float:
        """Calculate reward for the final answer."""
        base_reward = 0.5
        
        # Bonus for reasonable answer length
        if 10 <= len(answer) <= 500:
            base_reward += 0.3
        
        # Bonus for efficiency (fewer steps)
        efficiency_bonus = max(0, (self.max_steps - self.state.step) * 0.1)
        
        return base_reward + efficiency_bonus
    
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
            "max_steps": self.max_steps
        }

# Test function
def test_dynamic_environment():
    """Test the dynamic MCP environment."""
    print("🧪 TESTING DYNAMIC MCP ENVIRONMENT")
    print("=" * 60)
    
    # Create environment - it will auto-discover servers
    env = DynamicMCPEnvironment(max_steps=8)
    
    # Get environment info
    env_info = env.get_environment_info()
    print(f"\n📋 Environment Info:")
    print(f"   Total tools: {env_info['total_tools']}")
    print(f"   Available tools: {env_info['available_tools']}")
    
    # Test with different task types
    available_task_types = env_info['task_types']
    
    for task_type in available_task_types[:3]:  # Test first 3 task types
        print(f"\n🎯 Testing {task_type} task:")
        
        obs = env.reset(task_type=task_type)
        print(f"   Task generated successfully")
        
        # Try one action to verify everything works
        available_tools = env.tool_executor.get_available_tools()
        if available_tools:
            # Pick the first non-answer tool
            test_tool = next((t for t in available_tools if t != "answer"), "answer")
            
            if test_tool != "answer":
                # Create a simple test action
                if "search" in test_tool:
                    action = json.dumps({"tool": test_tool, "args": {"query": "test"}})
                elif "list" in test_tool:
                    action = json.dumps({"tool": test_tool, "args": {"path": "/"}})
                else:
                    action = json.dumps({"tool": test_tool, "args": {}})
            else:
                action = json.dumps({"tool": "answer", "args": {"answer": "Test completed"}})
            
            obs, reward, done, info = env.step(action)
            print(f"   Action executed: {test_tool}, Reward: {reward:.3f}")

if __name__ == "__main__":
    test_dynamic_environment()