# mcp_env/rl/__init__.py
"""
MCP RL Environment Package

Dynamic reinforcement learning environment for training LLMs with MCP tool use.
Automatically discovers and initializes tools based on available servers.
"""

from .mcp_tool_executor import DynamicMCPToolExecutor, ToolInfo, ServerInfo
from .dynamic_task_generator import DynamicTaskGenerator
from .dynamic_mcp_environment import DynamicMCPEnvironment, MCPState
from .reward_generator import RewardGenerator, RewardConfig

__all__ = [
    'DynamicMCPToolExecutor',
    'DynamicTaskGenerator', 
    'DynamicMCPEnvironment',
    'RewardGenerator',
    'RewardConfig',
    'ToolInfo',
    'ServerInfo',
    'MCPState'
]

__version__ = "1.0.0"