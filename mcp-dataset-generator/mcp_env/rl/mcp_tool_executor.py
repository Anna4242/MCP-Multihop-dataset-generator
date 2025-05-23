# mcp_env/rl/mcp_tool_executor.py
"""
Dynamic MCP Tool Executor for discovering and executing tools from available servers.
"""

import sys
import json
import inspect
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass

@dataclass
class ToolInfo:
    """Information about a discovered tool."""
    name: str
    function: Callable
    server_name: str
    args_spec: Dict[str, Any]
    description: str

@dataclass
class ServerInfo:
    """Information about a discovered server."""
    name: str
    module: Any
    tools: List[ToolInfo]
    mcp_instance: Any

class DynamicMCPToolExecutor:
    """Dynamically discovers and executes tools from available MCP servers."""
    
    def __init__(self, servers_dir: Path = None):
        self.servers_dir = servers_dir or self._find_servers_directory()
        self.servers: Dict[str, ServerInfo] = {}
        self.tools: Dict[str, ToolInfo] = {}
        
        if self.servers_dir:
            self.discover_and_load_servers()
        else:
            print("⚠️ No servers directory found. Please set servers_dir manually.")
    
    def _find_servers_directory(self) -> Optional[Path]:
        """Find the servers directory automatically."""
        possible_locations = [
            Path("./servers"),
            Path("./mcp_env/servers"),
            Path("../servers"),
            Path("../mcp_env/servers"),
            Path("D:/one drive/study/ARCEE AI INTERNSHIP/mcp data gen minimal/mcp-dataset-generator/mcp_env/servers")
        ]
        
        for location in possible_locations:
            if location.exists() and location.is_dir():
                return location
        return None
    
    def discover_and_load_servers(self):
        """Discover and load all available MCP servers."""
        print(f"🔍 Discovering servers in: {self.servers_dir}")
        
        if not self.servers_dir.exists():
            print(f"❌ Servers directory not found: {self.servers_dir}")
            return
        
        # Find all server directories
        server_dirs = [d for d in self.servers_dir.iterdir() if d.is_dir()]
        
        for server_dir in server_dirs:
            self._load_server(server_dir)
        
        print(f"📊 Discovery complete: {len(self.servers)} servers, {len(self.tools)} tools")
    
    def _load_server(self, server_dir: Path):
        """Load a single server and discover its tools."""
        server_name = server_dir.name
        
        # Look for server.py or other Python files
        server_files = [
            server_dir / "server.py",
            server_dir / f"{server_name.lower()}_server.py",
            server_dir / f"{server_name.lower()}.py"
        ]
        
        # Also check for any .py file that might be the server
        for py_file in server_dir.glob("*.py"):
            if py_file.name not in ["__init__.py"] and py_file not in server_files:
                server_files.append(py_file)
        
        server_module = None
        loaded_from = None
        
        for server_file in server_files:
            if server_file.exists():
                try:
                    # Add server directory to path
                    sys.path.insert(0, str(server_dir))
                    
                    # Import the module
                    module_name = server_file.stem
                    server_module = __import__(module_name)
                    loaded_from = server_file.name
                    
                    # Clean up path
                    if str(server_dir) in sys.path:
                        sys.path.remove(str(server_dir))
                    
                    break
                    
                except Exception as e:
                    if str(server_dir) in sys.path:
                        sys.path.remove(str(server_dir))
                    continue
        
        if not server_module:
            print(f"⚠️ Could not load server from {server_name}")
            return
        
        # Discover tools in the server
        tools = self._discover_tools(server_module, server_name)
        
        if tools:
            # Get MCP instance if available
            mcp_instance = getattr(server_module, 'mcp', None)
            
            server_info = ServerInfo(
                name=server_name,
                module=server_module,
                tools=tools,
                mcp_instance=mcp_instance
            )
            
            self.servers[server_name] = server_info
            
            # Add tools to global tool registry
            for tool in tools:
                self.tools[tool.name] = tool
            
            print(f"✅ Loaded {server_name} ({loaded_from}): {len(tools)} tools")
        else:
            print(f"⚠️ No tools found in {server_name}")
    
    def _discover_tools(self, server_module: Any, server_name: str) -> List[ToolInfo]:
        """Discover tools in a server module."""
        tools = []
        
        # Method 1: Check MCP instance for tools
        if hasattr(server_module, 'mcp'):
            mcp = server_module.mcp
            if hasattr(mcp, 'tools'):
                mcp_tools = mcp.tools
                for tool in mcp_tools:
                    if hasattr(tool, 'name') and hasattr(tool, 'func'):
                        tool_info = self._create_tool_info(
                            tool.name, 
                            tool.func, 
                            server_name,
                            getattr(tool, 'description', f"Tool from {server_name}")
                        )
                        if tool_info:
                            tools.append(tool_info)
        
        # Method 2: Look for functions that match common tool patterns
        tool_functions = self._find_tool_functions(server_module)
        for func_name, func in tool_functions.items():
            if func_name not in [t.name for t in tools]:  # Avoid duplicates
                tool_info = self._create_tool_info(func_name, func, server_name)
                if tool_info:
                    tools.append(tool_info)
        
        return tools
    
    def _find_tool_functions(self, module: Any) -> Dict[str, Callable]:
        """Find functions that look like tools in a module."""
        tool_functions = {}
        
        # Common tool function names
        common_tools = [
            'list_dir', 'read_file', 'write_file',
            'web_search', 'search',
            'run_query', 'list_tables', 'execute_query',
            'list_pull_requests', 'get_pull_request', 'clone_repo',
            'add', 'subtract', 'multiply', 'divide',
            'get_weather', 'send_email', 'create_task'
        ]
        
        # Look for functions in the module
        for name, obj in inspect.getmembers(module):
            if (inspect.isfunction(obj) and 
                not name.startswith('_') and 
                (name in common_tools or self._looks_like_tool(name, obj))):
                tool_functions[name] = obj
        
        return tool_functions
    
    def _looks_like_tool(self, name: str, func: Callable) -> bool:
        """Heuristics to determine if a function looks like a tool."""
        try:
            sig = inspect.signature(func)
            # Tools usually have parameters and return something
            if len(sig.parameters) == 0:
                return False
            
            # Check docstring for tool-like keywords
            doc = inspect.getdoc(func) or ""
            tool_keywords = ['execute', 'query', 'search', 'list', 'get', 'create', 'read', 'write']
            if any(keyword in doc.lower() for keyword in tool_keywords):
                return True
            
            # Check function name patterns
            tool_patterns = ['_query', '_search', '_list', '_get', '_create', '_read', '_write']
            if any(pattern in name.lower() for pattern in tool_patterns):
                return True
                
        except Exception:
            pass
        
        return False
    
    def _create_tool_info(self, name: str, func: Callable, server_name: str, description: str = None) -> Optional[ToolInfo]:
        """Create ToolInfo from a function."""
        try:
            # Get function signature for args specification
            sig = inspect.signature(func)
            args_spec = {}
            
            for param_name, param in sig.parameters.items():
                param_info = {"name": param_name}
                
                # Get type annotation if available
                if param.annotation != inspect.Parameter.empty:
                    param_info["type"] = str(param.annotation)
                
                # Get default value if available
                if param.default != inspect.Parameter.empty:
                    param_info["default"] = param.default
                    param_info["required"] = False
                else:
                    param_info["required"] = True
                
                args_spec[param_name] = param_info
            
            # Generate description if not provided
            if not description:
                doc = inspect.getdoc(func)
                if doc:
                    description = doc.split('\n')[0]  # First line of docstring
                else:
                    description = f"{name} tool from {server_name}"
            
            return ToolInfo(
                name=name,
                function=func,
                server_name=server_name,
                args_spec=args_spec,
                description=description
            )
            
        except Exception as e:
            print(f"⚠️ Could not create tool info for {name}: {e}")
            return None
    
    def get_available_tools(self) -> List[str]:
        """Get list of all available tool names."""
        tools = list(self.tools.keys())
        tools.append("answer")  # Always available
        return sorted(tools)
    
    def get_tool_info(self, tool_name: str) -> Optional[ToolInfo]:
        """Get detailed information about a tool."""
        return self.tools.get(tool_name)
    
    def get_tools_by_server(self) -> Dict[str, List[str]]:
        """Get tools organized by server."""
        by_server = {}
        for tool_name, tool_info in self.tools.items():
            server_name = tool_info.server_name
            if server_name not in by_server:
                by_server[server_name] = []
            by_server[server_name].append(tool_name)
        
        # Add answer tool
        by_server["system"] = ["answer"]
        return by_server
    
    def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call and return the result."""
        try:
            # Handle answer tool specially
            if tool_name == "answer":
                return {"type": "answer", "content": args.get("answer", "")}
            
            # Find the tool
            tool_info = self.tools.get(tool_name)
            if not tool_info:
                return {"error": f"Unknown tool: {tool_name}"}
            
            # Validate arguments against tool specification
            validation_result = self._validate_args(tool_info, args)
            if validation_result is not True:
                return {"error": f"Invalid arguments: {validation_result}"}
            
            # Execute the tool function
            result = tool_info.function(**args)
            
            # Wrap result in a standard format
            return self._format_tool_result(tool_name, result, tool_info.server_name)
            
        except Exception as e:
            return {"error": f"Tool execution failed: {str(e)}"}
    
    def _validate_args(self, tool_info: ToolInfo, args: Dict[str, Any]) -> Any:
        """Validate arguments against tool specification."""
        try:
            # Check required arguments
            for param_name, param_info in tool_info.args_spec.items():
                if param_info.get("required", True) and param_name not in args:
                    return f"Missing required argument: {param_name}"
            
            return True
            
        except Exception as e:
            return f"Argument validation error: {str(e)}"
    
    def _format_tool_result(self, tool_name: str, result: Any, server_name: str) -> Dict[str, Any]:
        """Format tool result in a standard way."""
        formatted = {
            "type": f"{server_name.lower()}_{tool_name}",
            "tool": tool_name,
            "server": server_name,
            "raw_result": result
        }
        
        # Add specific formatting based on tool type
        if "search" in tool_name.lower():
            if isinstance(result, list):
                formatted["type"] = "search_results"
                formatted["results"] = result
                formatted["count"] = len(result)
        
        elif "list" in tool_name.lower():
            if isinstance(result, list):
                formatted["type"] = "list_result"
                formatted["items"] = result
                formatted["count"] = len(result)
        
        elif "query" in tool_name.lower():
            formatted["type"] = "query_result"
            if isinstance(result, list):
                formatted["rows"] = result
                formatted["count"] = len(result)
            else:
                formatted["result"] = result
        
        elif "read" in tool_name.lower():
            formatted["type"] = "file_content"
            formatted["content"] = str(result)
        
        elif "write" in tool_name.lower():
            formatted["type"] = "write_result"
            formatted["status"] = str(result)
        
        else:
            # Generic formatting
            formatted["result"] = result
        
        return formatted