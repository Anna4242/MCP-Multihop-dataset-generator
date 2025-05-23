# mcp_env/rl/dynamic_task_generator.py
"""
Dynamic Task Generator for creating tasks based on available MCP tools.
"""

import random
from typing import Dict, List
from .mcp_tool_executor import DynamicMCPToolExecutor

class DynamicTaskGenerator:
    """Generates tasks dynamically based on available tools."""
    
    def __init__(self, tool_executor: DynamicMCPToolExecutor):
        self.tool_executor = tool_executor
        self.task_templates = self._generate_task_templates()
    
    def _generate_task_templates(self) -> Dict[str, List[str]]:
        """Generate task templates based on available tools and servers."""
        templates = {"general": []}
        tools_by_server = self.tool_executor.get_tools_by_server()
        
        for server_name, tools in tools_by_server.items():
            if server_name == "system":
                continue
                
            server_templates = []
            
            # Generate templates based on available tools
            if any("search" in tool for tool in tools):
                server_templates.extend([
                    f"Search for information about machine learning using {server_name}",
                    f"Find and summarize information about Python programming",
                    f"Look up current trends in artificial intelligence"
                ])
            
            if any("list" in tool for tool in tools):
                server_templates.extend([
                    f"List and explore available resources using {server_name}",
                    f"Discover and catalog available items"
                ])
            
            if any("read" in tool for tool in tools):
                server_templates.extend([
                    f"Read and analyze available documents using {server_name}",
                    f"Extract information from available files"
                ])
            
            if any("write" in tool for tool in tools):
                server_templates.extend([
                    f"Create a summary document using {server_name}",
                    f"Generate and save a report about your findings"
                ])
            
            if any("query" in tool for tool in tools):
                server_templates.extend([
                    f"Query and analyze data using {server_name}",
                    f"Extract insights from available data sources",
                    f"Generate a data summary report"
                ])
            
            if any("pull_request" in tool or "repo" in tool for tool in tools):
                server_templates.extend([
                    f"Analyze repository information using {server_name}",
                    f"Review and summarize project status"
                ])
            
            # Add mathematical tasks if math tools are available
            if any(op in tools for op in ["add", "subtract", "multiply", "divide"]):
                server_templates.extend([
                    f"Perform calculations using {server_name}",
                    f"Solve mathematical problems step by step"
                ])
            
            if server_templates:
                templates[server_name.lower()] = server_templates
                templates["general"].extend(server_templates)
        
        # Add multi-tool tasks if we have tools from multiple servers
        if len(tools_by_server) > 2:  # More than just system
            templates["multi_tool"] = [
                "Research a topic and create a comprehensive report using multiple tools",
                "Gather information from various sources and synthesize findings",
                "Complete a complex task that requires multiple types of operations",
                "Analyze data and present findings in a structured format"
            ]
            templates["general"].extend(templates["multi_tool"])
        
        return templates
    
    def generate_task(self, task_type: str = None) -> str:
        """Generate a task based on available tools."""
        if task_type and task_type in self.task_templates:
            return random.choice(self.task_templates[task_type])
        
        # Default to general tasks
        if self.task_templates["general"]:
            return random.choice(self.task_templates["general"])
        
        # Fallback
        return "Use the available tools to complete a useful task"
    
    def get_available_task_types(self) -> List[str]:
        """Get available task types."""
        return list(self.task_templates.keys())