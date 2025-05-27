#!/usr/bin/env python3
"""
improved_mcp_query_executor.py
------------------------------
Improved version of MCPQueryExecutor that tracks tool usage without monkey patching.
This version captures all tools called during execution, their inputs and outputs, and
verbose agent output while maintaining compatibility with existing code.

Usage as a script:
    python improved_mcp_query_executor.py

Usage as a module:
    from improved_mcp_query_executor import MCPQueryExecutor
    
    executor = MCPQueryExecutor(
        config_path="path/to/config.json",
        queries_path="path/to/queries.json"
    )
    await executor.execute_queries()
"""

import asyncio
import json
import logging
import os
import sys
import io
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from collections import defaultdict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from mcp_use import MCPAgent, MCPClient
from langchain_core.tools import BaseTool, Tool
from langchain_core.utils.input import get_color_mapping
from pydantic import Field, PrivateAttr

# Load environment variables
load_dotenv()

# Set OpenAI API key and base URL
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")  # Default to OpenRouter URL
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY missing in .env or environment")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("mcp_executor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Custom stream class to capture stdout
class CaptureStdout(io.StringIO):
    def __init__(self, original_stdout):
        super().__init__()
        self.original_stdout = original_stdout
        self.log_entries = []
        self.current_text = ""
        self.step_counter = 0
        self.current_tool = "initialization"
    
    def write(self, text):
        # Write to the original stdout so we still see output
        self.original_stdout.write(text)
        
        # Also capture in our buffer
        self.current_text += text
        
        # If we have a newline, consider it a complete log entry
        if '\n' in text:
            lines = self.current_text.split('\n')
            # Last line might be incomplete, so keep it
            self.current_text = lines[-1]
            # Add the complete lines to our log entries
            for line in lines[:-1]:
                if line.strip():  # Only add non-empty lines
                    self.log_entries.append({
                        "timestamp": datetime.now().isoformat(),
                        "step": self.step_counter,
                        "tool": self.current_tool,
                        "text": line
                    })
        
        # Also write to the StringIO object
        return super().write(text)
    
    def set_step(self, step_number, tool_name="unknown"):
        """Set the current step number and tool name."""
        self.step_counter = step_number
        self.current_tool = tool_name
    
    def increment_step(self, tool_name=None):
        """Increment the step counter and optionally set a new tool name."""
        self.step_counter += 1
        if tool_name:
            self.current_tool = tool_name
    
    def flush(self):
        self.original_stdout.flush()
        return super().flush()


# TrackingTool wrapper class to capture tool usage
class TrackingTool(BaseTool):
    """A wrapper tool that captures usage statistics for the wrapped tool."""
    
    # Define these as actual fields in the model
    server_name: str = Field(default="unknown", description="Server name this tool belongs to")
    
    # Use private attributes for things we don't want in the schema
    _wrapped_tool: BaseTool = PrivateAttr()
    _callback: Callable = PrivateAttr()
    
    def __init__(self, **data):
        """Initialize a tracking tool."""
        # Extract the wrapped_tool and callback from the data
        wrapped_tool = data.pop("wrapped_tool", None)
        callback = data.pop("callback", None)
        
        # If name is not provided, use the wrapped tool's name
        if "name" not in data and wrapped_tool:
            data["name"] = wrapped_tool.name
            
        # If description is not provided, use the wrapped tool's description  
        if "description" not in data and wrapped_tool:
            data["description"] = wrapped_tool.description
            
        # Initialize the Pydantic model
        super().__init__(**data)
        
        # Set the private attributes
        self._wrapped_tool = wrapped_tool
        self._callback = callback
        
        # Copy additional attributes from wrapped tool
        if wrapped_tool:
            # Handle args_schema
            if hasattr(wrapped_tool, "args_schema"):
                self.args_schema = wrapped_tool.args_schema
                
            # Handle return_direct
            if hasattr(wrapped_tool, "return_direct"):
                self.return_direct = wrapped_tool.return_direct
                
            # Handle handle_tool_error  
            if hasattr(wrapped_tool, "handle_tool_error"):
                self.handle_tool_error = wrapped_tool.handle_tool_error
    
    def _run(self, *args, **kwargs):
        """Run the tool synchronously, capturing usage data."""
        # This is a fallback, but MCP tools typically use async
        raise NotImplementedError("This tool only supports async execution")
    
    async def _arun(self, **kwargs):
        """Run the tool asynchronously, capturing usage data."""
        # Create a record for this tool call
        start_time = datetime.now()
        tool_call = {
            "timestamp": start_time.isoformat(),
            "tool_name": self.name,
            "server_name": self.server_name,
            "input": kwargs,
            "output": None,
            "success": False,
            "error": None,
            "execution_time_ms": None
        }
        
        try:
            # Execute the wrapped tool
            result = await self._wrapped_tool._arun(**kwargs)
            
            # Record success
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds() * 1000  # ms
            tool_call.update({
                "output": result,
                "success": True,
                "execution_time_ms": execution_time
            })
            
            # Call the callback with the tracking data
            if self._callback:
                await self._callback(tool_call)
            
            return result
        except Exception as e:
            # Record failure
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds() * 1000  # ms
            tool_call.update({
                "error": str(e),
                "success": False,
                "execution_time_ms": execution_time
            })
            
            # Call the callback with the tracking data
            if self._callback:
                await self._callback(tool_call)
            
            # Re-raise the exception if we don't handle errors
            if not (hasattr(self, "handle_tool_error") and self.handle_tool_error):
                raise
            return f"Error executing {self.name}: {str(e)}"


class MCPQueryExecutor:
    """
    Improved executor that captures tool sequences and verbose output without monkey patching.
    """
    
    def __init__(
        self,
        config_path: Union[str, Path] = None,
        queries_path: Union[str, Path] = None,
        save_results: bool = True,
        output_dir: Union[str, Path] = "execution_results",
        tool_logs_dir: Union[str, Path] = "tool_usage_logs",
        query_timeout: int = 180,
        cleanup_timeout: int = 10,
        model_name: str = "gpt-4o",
        temperature: float = 0,
        max_agent_steps: int = 30,
        api_key: str = None,
        **kwargs
    ):
        """
        Initialize the executor with compatibility for existing pipeline.
        
        Args:
            config_path: Path to MCP server configuration file
            queries_path: Path to queries JSON file
            save_results: Whether to save results (compatibility parameter)
            output_dir: Directory to save results (compatibility parameter)
            tool_logs_dir: Directory to save tool logs (compatibility parameter)
            query_timeout: Timeout in seconds for individual queries
            cleanup_timeout: Timeout for cleanup operations (compatibility parameter)
            model_name: LLM model to use
            temperature: LLM temperature setting
            max_agent_steps: Maximum steps the agent can take
            api_key: OpenAI API key (overrides environment variable)
            **kwargs: Additional arguments for compatibility
        """
        # Default paths if not provided
        self.config_path = Path(config_path or os.environ.get("MCP_CONFIG") or 
                               "mcp_servers.json")
        self.queries_path = Path(queries_path or os.environ.get("QUERIES_FILE") or
                                "queries.json")
        
        # API key
        self.api_key = api_key or OPENAI_API_KEY
        
        # Compatibility settings
        self.save_results = save_results
        self.output_dir = Path(output_dir)
        self.tool_logs_dir = Path(tool_logs_dir)
        self.cleanup_timeout = cleanup_timeout
        
        # Execution settings
        self.query_timeout = query_timeout
        self.model_name = model_name
        self.temperature = temperature
        self.max_agent_steps = max_agent_steps
        
        # Store additional kwargs for compatibility
        self.kwargs = kwargs
        
        # Runtime objects
        self.client = None
        self.agent = None
        self.run_start_time = None
        self.run_end_time = None
        self.results = []  # Added for compatibility
        
        # Storage for tool sequences
        self.query_tool_sequences = {}
        self.current_query_id = None
        self.current_query = None
        
        # Compatibility with original tool tracking structure
        self.tool_calls = []
        self.calls_by_tool = defaultdict(list)
        self.calls_by_server = defaultdict(list)
        self.calls_by_query = defaultdict(list)
        
        # Create log directory
        self.log_dir = Path("tool_sequence_logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Verbose output directory
        self.verbose_output_dir = Path("verbose_output")
        self.verbose_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output directories for compatibility
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tool_logs_dir.mkdir(parents=True, exist_ok=True)
        
        # For capturing stdout
        self.original_stdout = sys.stdout
        self.captured_stdout = None
        self.query_verbose_output = {}
    
    def _start_capturing_stdout(self):
        """Start capturing stdout."""
        if self.captured_stdout is None:
            self.captured_stdout = CaptureStdout(self.original_stdout)
            sys.stdout = self.captured_stdout
    
    def _stop_capturing_stdout(self):
        """Stop capturing stdout and return the captured output."""
        if self.captured_stdout is not None:
            # Get the captured log entries
            log_entries = self.captured_stdout.log_entries.copy()
            # Reset the capture
            sys.stdout = self.original_stdout
            self.captured_stdout = None
            return log_entries
        return []

    async def _tool_callback(self, tool_call_data):
        """Callback function for tracking tool usage."""
        # Skip if we're not tracking a query
        if not self.current_query_id:
            return
        
        # Get the current step number
        if self.captured_stdout:
            current_step = self.captured_stdout.step_counter
            self.captured_stdout.increment_step(tool_call_data["tool_name"])
        else:
            current_step = 0
        
        # Add step to the tool call data
        tool_call_data["step"] = current_step
        
        # Initialize sequence for this query if not exists
        if self.current_query_id not in self.query_tool_sequences:
            self.query_tool_sequences[self.current_query_id] = {
                "query_id": self.current_query_id,
                "query_text": None,  # Will fill this later
                "start_time": datetime.now().isoformat(),
                "end_time": None,
                "tool_calls": [],
                "result": None,
                "error": None,
                "verbose_output": []
            }
        
        # Add tool call to sequence
        self.query_tool_sequences[self.current_query_id]["tool_calls"].append(tool_call_data)
        
        # Update stdout to reflect current tool
        if self.captured_stdout:
            self.captured_stdout.set_step(current_step, tool_call_data["tool_name"])
        
        # Also update the compatibility structures
        compatibility_call = {
            "timestamp": tool_call_data["timestamp"],
            "step": current_step,
            "tool_name": tool_call_data["tool_name"],
            "server_name": tool_call_data["server_name"],
            "query_id": self.current_query_id,
            "query": self.current_query[:200] if self.current_query else None,
            "tool_input": self._serialize_data(tool_call_data["input"]),
            "tool_output": self._serialize_data(tool_call_data["output"]),
            "success": tool_call_data["success"],
            "error": tool_call_data["error"]
        }
        
        self.tool_calls.append(compatibility_call)
        self.calls_by_tool[tool_call_data["tool_name"]].append(compatibility_call)
        self.calls_by_server[tool_call_data["server_name"]].append(compatibility_call)
        self.calls_by_query[self.current_query_id].append(compatibility_call)
        
        # Save after each tool call to prevent data loss
        self._save_current_sequence()
    
    def _serialize_data(self, data):
        """
        Serialize data to ensure it can be saved to JSON.
        
        Args:
            data: Any data to serialize
            
        Returns:
            JSON-serializable representation
        """
        if data is None:
            return None
            
        try:
            # For simple types, return as is
            if isinstance(data, (str, int, float, bool)) or data is None:
                return data
                
            # For dict, list, tuple - try to convert to dict/list
            if isinstance(data, (dict, list, tuple)):
                return json.loads(json.dumps(data, default=str))
                
            # For other objects, convert to string representation
            return str(data)
        except:
            # Fallback to string representation
            try:
                return str(data)
            except:
                return "UNSERIALIZABLE_DATA"
    
    def _save_current_sequence(self):
        """Save the current tool sequence to prevent data loss."""
        if not self.current_query_id or self.current_query_id not in self.query_tool_sequences:
            return
            
        # Create a filename
        query_id_safe = str(self.current_query_id).replace("/", "_").replace("\\", "_")
        filename = f"sequence_{query_id_safe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path = self.log_dir / filename
        
        # Save the sequence
        try:
            # Create a version safe for JSON
            sequence = self.query_tool_sequences[self.current_query_id].copy()
            
            # Ensure all data is serializable
            for call in sequence["tool_calls"]:
                call["input"] = self._serialize_data(call["input"])
                call["output"] = self._serialize_data(call["output"])
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(sequence, f, indent=2)
                
            logger.debug(f"Saved sequence checkpoint for {self.current_query_id}")
        except Exception as e:
            logger.error(f"Error saving sequence checkpoint: {e}")
    
    def save_results_to_file(self, results: List[Dict]) -> Optional[Path]:
        """Save results to JSON file with timestamp. (Compatibility method)"""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_file = self.output_dir / f"answers_{ts}.json"
            out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
            return out_file
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return None
    
    def save_tool_logs(self, filename: Optional[str] = None) -> Path:
        """
        Save the collected tool usage data to a JSON file. (Compatibility method)
        
        Args:
            filename: Optional custom filename
            
        Returns:
            Path to the saved file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tool_usage_{timestamp}.json"
        
        output_path = self.tool_logs_dir / filename
        
        # Create the full log object using the compatibility structures
        from collections import Counter
        
        # Prepare summary
        tool_counts = Counter([call["tool_name"] for call in self.tool_calls])
        server_counts = Counter([call["server_name"] for call in self.tool_calls])
        query_counts = {
            query_id: len(calls) 
            for query_id, calls in self.calls_by_query.items()
        }
        
        log_data = {
            "run_metadata": {
                "start_time": self.run_start_time.isoformat() if self.run_start_time else None,
                "end_time": self.run_end_time.isoformat() if self.run_end_time else None,
                "total_tool_calls": len(self.tool_calls),
                "total_queries": len(self.calls_by_query),
                "unique_tools_used": len(self.calls_by_tool),
                "servers_used": list(self.calls_by_server.keys()),
            },
            "summary": {
                "tool_counts": dict(tool_counts),
                "server_counts": dict(server_counts),
                "tool_calls_per_query": query_counts,
            },
            "tool_calls": self.tool_calls
        }
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2)
        
        logger.info(f"Tool usage logs saved to {output_path}")
        return output_path
    
    def _save_verbose_output(self):
        """Save the verbose output for all queries."""
        if not self.query_verbose_output and not self.query_tool_sequences:
            logger.warning("No verbose output to save")
            return None
            
        try:
            # Create output file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"verbose_output_{timestamp}.json"
            path = self.verbose_output_dir / filename
            
            # Combine all verbose output
            verbose_data = {}
            
            # First add any directly captured verbose output
            for query_id, output in self.query_verbose_output.items():
                verbose_data[query_id] = output
            
            # Then add verbose output from tool sequences
            for query_id, sequence in self.query_tool_sequences.items():
                verbose_entries = []
                
                # Add the query's own verbose output if it exists
                if sequence.get("verbose_output"):
                    verbose_entries.extend(sequence["verbose_output"])
                
                # Sort by timestamp
                verbose_entries.sort(key=lambda x: x.get("timestamp", ""))
                
                # Add to the verbose data if not already present
                if query_id not in verbose_data:
                    verbose_data[query_id] = {
                        "query_id": query_id,
                        "query_text": sequence.get("query_text", ""),
                        "entries": verbose_entries
                    }
                else:
                    # Merge with existing entries
                    verbose_data[query_id]["entries"].extend(verbose_entries)
                    verbose_data[query_id]["entries"].sort(key=lambda x: x.get("timestamp", ""))
            
            # Add metadata
            output = {
                "metadata": {
                    "run_start_time": self.run_start_time.isoformat() if self.run_start_time else None,
                    "run_end_time": self.run_end_time.isoformat() if self.run_end_time else None,
                    "model_name": self.model_name,
                    "temperature": self.temperature,
                    "max_agent_steps": self.max_agent_steps,
                    "total_queries": len(verbose_data)
                },
                "verbose_output": verbose_data
            }
            
            # Save to file
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2)
                
            logger.info(f"Verbose output saved to {path}")
            print(f"📝 Verbose output saved to {path}")
            
            # Also save individual verbose output files
            for query_id, data in verbose_data.items():
                query_id_safe = str(query_id).replace("/", "_").replace("\\", "_")
                query_filename = f"verbose_{query_id_safe}.json"
                query_path = self.verbose_output_dir / query_filename
                
                with open(query_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                
                logger.info(f"Verbose output for query {query_id} saved to {query_path}")
            
            return path
        except Exception as e:
            logger.error(f"Error saving verbose output: {e}")
            return None
    
    def _save_all_sequences(self):
        """Save all tool sequences to a single file."""
        if not self.query_tool_sequences:
            logger.warning("No tool sequences to save")
            return None
            
        try:
            # Create output file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"all_sequences_{timestamp}.json"
            path = self.log_dir / filename
            
            # Create a version safe for JSON
            sequences = {}
            for query_id, sequence in self.query_tool_sequences.items():
                seq_copy = sequence.copy()
                
                # Ensure all data is serializable
                for call in seq_copy["tool_calls"]:
                    call["input"] = self._serialize_data(call["input"])
                    call["output"] = self._serialize_data(call["output"])
                    
                sequences[query_id] = seq_copy
            
            # Add metadata
            output = {
                "metadata": {
                    "run_start_time": self.run_start_time.isoformat() if self.run_start_time else None,
                    "run_end_time": self.run_end_time.isoformat() if self.run_end_time else None,
                    "model_name": self.model_name,
                    "temperature": self.temperature,
                    "max_agent_steps": self.max_agent_steps,
                    "total_queries": len(sequences)
                },
                "sequences": sequences
            }
            
            # Save to file
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2)
                
            logger.info(f"All tool sequences saved to {path}")
            print(f"📊 All tool sequences saved to {path}")
            return path
        except Exception as e:
            logger.error(f"Error saving all sequences: {e}")
            return None
    
    def save_per_query_sequences(self):
        """Save each query's tool sequence to a separate file."""
        if not self.query_tool_sequences:
            logger.warning("No tool sequences to save")
            return []
            
        paths = []
        for query_id, sequence in self.query_tool_sequences.items():
            try:
                # Create a safe filename
                query_id_safe = str(query_id).replace("/", "_").replace("\\", "_")
                filename = f"sequence_{query_id_safe}_final.json"
                path = self.log_dir / filename
                
                # Create a version safe for JSON
                seq_copy = sequence.copy()
                
                # Ensure all data is serializable
                for call in seq_copy["tool_calls"]:
                    call["input"] = self._serialize_data(call["input"])
                    call["output"] = self._serialize_data(call["output"])
                
                # Save to file
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(seq_copy, f, indent=2)
                    
                paths.append(path)
                logger.info(f"Sequence for query {query_id} saved to {path}")
            except Exception as e:
                logger.error(f"Error saving sequence for query {query_id}: {e}")
                
        print(f"📊 Individual query sequences saved to {self.log_dir}")
        return paths
    
    def generate_tool_usage_report(self, include_full_details: bool = False) -> str:
        """
        Generate a text report of tool usage. (Compatibility method)
        
        Args:
            include_full_details: Whether to include detailed logs of each tool call
            
        Returns:
            A formatted text report
        """
        total_calls = len(self.tool_calls)
        if total_calls == 0:
            return "No tool calls recorded."
            
        unique_tools = len(self.calls_by_tool)
        unique_servers = len(self.calls_by_server)
        
        # Get the most used tools and servers
        from collections import Counter
        tool_counts = Counter([call["tool_name"] for call in self.tool_calls])
        server_counts = Counter([call["server_name"] for call in self.tool_calls])
        most_common_tools = tool_counts.most_common(10)
        most_common_servers = server_counts.most_common(5)
        
        # Begin building the report
        report = ["# MCP Tool Usage Report", ""]
        
        # Run metadata
        report.append("## Run Information")
        if self.run_start_time:
            report.append(f"- Start time: {self.run_start_time}")
        if self.run_end_time:
            report.append(f"- End time: {self.run_end_time}")
            duration = self.run_end_time - self.run_start_time
            report.append(f"- Duration: {duration}")
        report.append(f"- Total tool calls: {total_calls}")
        report.append(f"- Unique tools used: {unique_tools}")
        report.append(f"- MCP servers used: {unique_servers}")
        report.append("")
        
        # Tool usage summary
        report.append("## Tool Usage Summary")
        report.append("### Most Frequently Used Tools")
        for tool, count in most_common_tools:
            percentage = (count / total_calls) * 100
            report.append(f"- {tool}: {count} calls ({percentage:.1f}%)")
        report.append("")
        
        # Server usage
        report.append("### Server Usage")
        for server, count in most_common_servers:
            percentage = (count / total_calls) * 100
            report.append(f"- {server}: {count} calls ({percentage:.1f}%)")
        report.append("")
        
        # Per-query analysis
        report.append("## Query Analysis")
        for query_id, calls in self.calls_by_query.items():
            query_text = calls[0].get("query", "Unknown") if calls else "Unknown"
            report.append(f"### {query_id}")
            report.append(f"Query: {query_text[:100]}..." if len(str(query_text)) > 100 else query_text)
            report.append(f"Total tool calls: {len(calls)}")
            
            # Get most used tools for this query
            query_tool_counts = Counter([call["tool_name"] for call in calls])
            report.append("Most used tools:")
            for tool, count in query_tool_counts.most_common(5):
                report.append(f"  - {tool}: {count} calls")
            
            # Tool sequence for this query
            if include_full_details:
                report.append("Tool sequence:")
                for i, call in enumerate(calls, 1):
                    report.append(f"  {i}. {call['tool_name']} ({call['server_name']})")
            
            report.append("")
        
        return "\n".join(report)
    
    def check_file_exists(self, path: Path) -> bool:
        """Check if file exists and log appropriate message."""
        if not path.exists():
            logger.error(f"File not found: {path}")
            return False
        return True
    
    def read_queries(self, path: Path) -> List[Dict]:
        """Load the JSON list; accept 'query' or 'question' field"""
        if not self.check_file_exists(path):
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
    
    def _wrap_tools_with_tracking(self, tools):
        """Wrap tools with tracking functionality."""
        wrapped_tools = []
        for tool in tools:
            server_name = getattr(tool, "server_name", "unknown")
            
            # Create the wrapped tool using valid fields
            wrapped_tool = TrackingTool(
                name=tool.name,
                description=tool.description,
                server_name=server_name,
                wrapped_tool=tool,
                callback=self._tool_callback
            )
            
            # Make sure to copy any schema
            if hasattr(tool, "args_schema"):
                wrapped_tool.args_schema = tool.args_schema
            
            # Copy other important attributes
            for attr in ["return_direct", "handle_tool_error"]:
                if hasattr(tool, attr):
                    setattr(wrapped_tool, attr, getattr(tool, attr))
            
            wrapped_tools.append(wrapped_tool)
        
        return wrapped_tools
    
    async def setup(self) -> bool:
        """Initialize the MCP client and agent."""
        try:
            # Record start time
            self.run_start_time = datetime.now()
            
            # Check if OpenAI API key is set
            if not self.api_key:
                logger.error("OpenAI API key not found")
                print("❌ Error: OpenAI API key not found")
                return False
            
            # Check if config files exist
            if not self.check_file_exists(self.config_path):
                return False
            
            # Start capturing stdout
            self._start_capturing_stdout()
            
            # Initialize MCP client
            print(f"🔄 Initializing MCP client from {self.config_path}")
            self.client = MCPClient.from_config_file(str(self.config_path))
            
            # Build agent
            print(f"🤖 Building agent with {self.model_name}")
            llm = ChatOpenAI(model=self.model_name, temperature=self.temperature, api_key=self.api_key)
            
            # Create the agent (but don't initialize yet - we need to wrap tools first)
            self.agent = MCPAgent(
                llm=llm, 
                client=self.client, 
                max_steps=self.max_agent_steps,
                verbose=True,
                auto_initialize=False
            )
            
            # Initialize the agent - this will create the tools
            await self.agent.initialize()
            
            # Get the original tools
            original_tools = self.agent._tools
            
            # Create wrapped tools with tracking functionality
            wrapped_tools = self._wrap_tools_with_tracking(original_tools)
            
            # Replace the original tools with wrapped tools
            self.agent._tools = wrapped_tools
            
            # Create the agent executor with wrapped tools
            self.agent._agent_executor = self.agent._create_agent()
            
            return True
        except Exception as e:
            logger.error(f"Error during setup: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def cleanup(self):
        """Clean up resources."""
        # Record end time
        self.run_end_time = datetime.now()
        
        # Stop capturing stdout
        captured_logs = self._stop_capturing_stdout()
        if captured_logs:
            # Save the final captured logs
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"final_output_{timestamp}.json"
            path = self.verbose_output_dir / filename
            
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(captured_logs, f, indent=2)
                logger.info(f"Final captured output saved to {path}")
            except Exception as e:
                logger.error(f"Error saving final captured output: {e}")
        
        print("🧹 Cleaning up MCP client sessions...")
        if self.client:
            try:
                await self.client.close_all_sessions()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
    
    async def run_query(self, query_data: Dict, idx: int, total: int) -> Dict:
        """Run a single query with timeout and error handling."""
        if not self.agent:
            raise ValueError("Agent not initialized. Call setup() first.")
        
        # Extract query information
        prompt = query_data.get("query") or query_data.get("question") or ""
        tag = query_data.get("id", f"query_{idx}")
        
        # Set current query ID to track tool sequence
        self.current_query_id = tag
        self.current_query = prompt
        
        # Initialize the record
        rec = {"id": tag, "prompt": prompt, "answer": None, "error": None}
        
        # Store the query text in the sequence data
        if tag in self.query_tool_sequences:
            self.query_tool_sequences[tag]["query_text"] = prompt
        else:
            self.query_tool_sequences[tag] = {
                "query_id": tag,
                "query_text": prompt,
                "start_time": datetime.now().isoformat(),
                "end_time": None,
                "tool_calls": [],
                "result": None,
                "error": None,
                "verbose_output": []
            }
        
        # Clear the stdout buffer and start capturing
        if self.captured_stdout:
            self.captured_stdout.log_entries = []
            self.captured_stdout.set_step(0, "query_start")
        else:
            self._start_capturing_stdout()
            if self.captured_stdout:
                self.captured_stdout.set_step(0, "query_start")
        
        print(f"\n▶ {idx}/{total}  {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        
        try:
            # Apply timeout to each query
            answer = await asyncio.wait_for(self.agent.run(prompt), timeout=self.query_timeout)
            rec["answer"] = answer
            print(f"   → {answer[:120]}{'...' if len(answer) > 120 else ''}")
            
            # Capture the verbose output
            if self.captured_stdout:
                self.captured_stdout.set_step(self.captured_stdout.step_counter, "query_complete")
            verbose_output = self._stop_capturing_stdout()
            self._start_capturing_stdout()  # Restart capturing
            
            # Store result in sequence data
            self.query_tool_sequences[tag]["result"] = answer
            self.query_tool_sequences[tag]["end_time"] = datetime.now().isoformat()
            self.query_tool_sequences[tag]["verbose_output"] = verbose_output
            
            # Also store verbose output separately
            self.query_verbose_output[tag] = {
                "query_id": tag,
                "query_text": prompt,
                "entries": verbose_output
            }
            
        except asyncio.TimeoutError:
            error_msg = f"Query timed out after {self.query_timeout} seconds"
            rec["error"] = error_msg
            logger.error(f"Query {idx}/{total} timed out")
            print(f"   ✗ TIMEOUT: {error_msg}")
            
            # Capture the verbose output
            if self.captured_stdout:
                self.captured_stdout.set_step(self.captured_stdout.step_counter, "query_timeout")
            verbose_output = self._stop_capturing_stdout()
            self._start_capturing_stdout()  # Restart capturing
            
            # Store error in sequence data
            self.query_tool_sequences[tag]["error"] = error_msg
            self.query_tool_sequences[tag]["end_time"] = datetime.now().isoformat()
            self.query_tool_sequences[tag]["verbose_output"] = verbose_output
            
            # Also store verbose output separately
            self.query_verbose_output[tag] = {
                "query_id": tag,
                "query_text": prompt,
                "entries": verbose_output
            }
            
        except Exception as exc:
            rec["error"] = str(exc)
            logger.error(f"Query {idx}/{total} failed: {exc}")
            print(f"   ✗ ERROR: {exc}")
            
            # Capture the verbose output
            if self.captured_stdout:
                self.captured_stdout.set_step(self.captured_stdout.step_counter, "query_error")
            verbose_output = self._stop_capturing_stdout()
            self._start_capturing_stdout()  # Restart capturing
            
            # Store error in sequence data
            self.query_tool_sequences[tag]["error"] = str(exc)
            self.query_tool_sequences[tag]["end_time"] = datetime.now().isoformat()
            self.query_tool_sequences[tag]["verbose_output"] = verbose_output
            
            # Also store verbose output separately
            self.query_verbose_output[tag] = {
                "query_id": tag,
                "query_text": prompt,
                "entries": verbose_output
            }
        
        # Reset current query ID
        self.current_query_id = None
        self.current_query = None
        
        # Save the completed query sequence
        self._save_current_sequence()
        
        return rec
    
    async def execute_queries(self) -> int:
        """Main function to run all queries."""
        try:
            # Setup client and agent
            if not await self.setup():
                return 1
            
            # Read prompts
            queries = self.read_queries(self.queries_path)
            if not queries:
                logger.error(f"No valid queries found in {self.queries_path}")
                print(f"❌ No valid queries found in {self.queries_path}")
                return 1
                
            print(f"📋 Loaded {len(queries)} prompts from {self.queries_path}")
            
            # Process each query
            self.results = []
            for idx, query_data in enumerate(queries, 1):
                rec = await self.run_query(query_data, idx, len(queries))
                self.results.append(rec)
            
            # Save all results to a file (compatibility)
            if self.save_results and self.results:
                results_file = self.save_results_to_file(self.results)
                if results_file:
                    print(f"\n💾 Results saved to {results_file}")
            
            # Save all tool sequences
            all_sequences_path = self._save_all_sequences()
            
            # Save individual query sequences
            self.save_per_query_sequences()
            
            # Save verbose output
            verbose_output_path = self._save_verbose_output()
            
            # Save tool usage logs (compatibility)
            if self.tool_calls:  # Only if we have tool calls
                tool_logs_path = self.save_tool_logs()
                print(f"🔍 Tool usage logs saved to {tool_logs_path}")
            
            # Print summary
            total_tool_calls = sum(len(seq["tool_calls"]) for seq in self.query_tool_sequences.values())
            print("\n===== EXECUTION SUMMARY =====")
            print(f"Total queries executed: {len(queries)}")
            print(f"Total tool calls: {total_tool_calls}")
            print(f"Tool calls per query: {total_tool_calls / len(queries):.1f}")
            print("============================")
            
            return 0
        
        except KeyboardInterrupt:
            print("\n⚠️ Operation cancelled by user")
            return 130
        except Exception as e:
            logger.critical(f"Unhandled exception: {e}", exc_info=True)
            print(f"\n❌ Critical error: {e}")
            import traceback
            traceback.print_exc()
            return 1
        finally:
            # Proper cleanup
            await self.cleanup()

    @classmethod
    async def run_all(cls, config_path: str = None, queries_path: str = None) -> int:
        """Class method to run the executor with default settings."""
        executor = cls(config_path=config_path, queries_path=queries_path)
        return await executor.execute_queries()


async def main_async():
    """Async entry point for the script."""
    # Parse command-line arguments if needed
    import argparse
    parser = argparse.ArgumentParser(description="Execute MCP queries with detailed tool sequence and verbose output logging")
    parser.add_argument("--config", type=str, help="Path to MCP server configuration file")
    parser.add_argument("--queries", type=str, help="Path to queries JSON file")
    parser.add_argument("--model", type=str, default="gpt-4o", help="LLM model to use")
    parser.add_argument("--timeout", type=int, default=180, help="Query timeout in seconds")
    parser.add_argument("--max-steps", type=int, default=30, help="Maximum agent steps")
    parser.add_argument("--api-key", type=str, help="OpenAI API key (overrides environment variable)")
    parser.add_argument("--save-results", action="store_true", help="Save results to file")
    parser.add_argument("--output-dir", type=str, default="execution_results", help="Directory to save results")
    
    args = parser.parse_args()
    
    # Create executor with command-line arguments
    executor = MCPQueryExecutor(
        config_path=args.config,
        queries_path=args.queries,
        model_name=args.model,
        query_timeout=args.timeout,
        max_agent_steps=args.max_steps,
        api_key=args.api_key,
        save_results=args.save_results,
        output_dir=args.output_dir
    )
    
    return await executor.execute_queries()


def main():
    """Entry point with proper asyncio handling."""
    try:
        # Create new event loop and run the main function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        exit_code = loop.run_until_complete(main_async())
        
        # Cleanup
        pending = asyncio.all_tasks(loop)
        if pending:
            print(f"Cancelling {len(pending)} pending tasks...")
            for task in pending:
                task.cancel()
            
            # Allow cancelled tasks to complete with a timeout
            loop.run_until_complete(
                asyncio.wait(pending, timeout=10)
            )
        
        loop.close()
        return exit_code
    
    except Exception as e:
        print(f"Setup error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())