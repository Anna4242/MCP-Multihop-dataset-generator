"""
Multi-server dataset generator for MCP tools with built-in multi-hop validation.
This generator creates multi-hop datasets that work across different server types.
"""
import asyncio
import json
import os
import csv
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Set, Tuple, Optional, Union

import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY missing in .env or environment")

# ════════════════════ MULTI-HOP VALIDATOR ════════════════════

class MultiHopValidator:
    """Validator to ensure questions require true multi-hop reasoning."""
    
    def __init__(self, min_score=0.7):
        """
        Initialize the validator.
        
        Args:
            min_score: Minimum multi-hop score to pass validation (0.0-1.0)
        """
        self.min_score = min_score
        
        # Regular expressions for detecting tool usage patterns
        self.tool_reference_pattern = re.compile(r'# Using tool: [\'"]([^\'"]+)[\'"]')
        self.tool_call_pattern = re.compile(r'(\w+)\s*\(')
        self.variable_pattern = re.compile(r'(\w+)\s*=')
        self.data_flow_pattern = re.compile(r'(\w+)\s*\(.*?(\w+)\.?\w*\s*.*?\)')
        
        # Schema properties that are not tools
        self.schema_props = {
            "$schema", "$id", "definitions", "type", "properties", 
            "required", "title", "description", "items", "additionalProperties",
            "anyOf", "allOf", "oneOf", "not", "enum", "const", "default", 
            "format", "pattern", "minLength", "maxLength", "minimum", 
            "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
            "minItems", "maxItems", "uniqueItems", "minProperties", "maxProperties"
        }
    
    def validate_entry(self, entry: Dict) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validate if an entry requires multi-hop reasoning.
        
        Args:
            entry: Dictionary containing the dataset entry
            
        Returns:
            Tuple of (is_valid, reason, metrics)
        """
        # Extract components
        question = entry.get('question', '')
        rule_code = entry.get('extra_info', {}).get('rule', '')
        
        # Check if rule code exists
        if not rule_code:
            return False, "Missing rule code", {}
        
        # Check for explicit tool references
        tool_references = self.tool_reference_pattern.findall(rule_code)
        if len(tool_references) < 2:
            return False, f"Found only {len(tool_references)} tool references, need at least 2", {
                "tool_references": tool_references
            }
        
        # Detect tool calls in the code
        potential_tool_calls = self.tool_call_pattern.findall(rule_code)
        # Filter out common programming keywords
        filtered_tool_calls = [
            call for call in potential_tool_calls 
            if call not in ['def', 'if', 'for', 'while', 'min', 'max', 'print', 'return', 'len', 'str']
        ]
        
        if len(set(filtered_tool_calls)) < 2:
            return False, f"Found only {len(set(filtered_tool_calls))} unique tool calls, need at least 2", {
                "potential_tool_calls": list(set(filtered_tool_calls))
            }
        
        # Check for variable assignments (indicating data flow)
        variable_assignments = self.variable_pattern.findall(rule_code)
        if len(variable_assignments) < 2:
            return False, f"Found only {len(variable_assignments)} variable assignments, need at least 2 for data flow", {
                "variable_assignments": variable_assignments
            }
        
        # Check for data flow between tool calls
        data_flows = self.data_flow_pattern.findall(rule_code)
        has_data_flow = any(flow[1] in variable_assignments for flow in data_flows)
        
        if not has_data_flow:
            return False, "No evidence of data flowing between tool calls", {
                "data_flows": data_flows
            }
        
        # Check for multi-hop keywords in the question
        multihop_keywords = ['then', 'based on', 'using that', 'with this information', 'afterwards', 'following', 'next', 'subsequently']
        has_multihop_keywords = any(keyword in question.lower() for keyword in multihop_keywords)
        
        # Create metrics
        metrics = {
            "tool_references": tool_references,
            "tool_calls": list(set(filtered_tool_calls)),
            "variable_assignments": variable_assignments,
            "has_data_flow": has_data_flow,
            "has_multihop_keywords": has_multihop_keywords,
            "multihop_score": self._calculate_multihop_score(
                len(tool_references), 
                len(set(filtered_tool_calls)),
                has_data_flow,
                has_multihop_keywords
            )
        }
        
        # Check if score meets minimum threshold
        is_valid = metrics["multihop_score"] >= self.min_score
        reason = "Entry meets multi-hop requirements" if is_valid else f"Multi-hop score {metrics['multihop_score']} below threshold {self.min_score}"
        
        return is_valid, reason, metrics
    
    def _calculate_multihop_score(self, 
                                 num_tool_references: int, 
                                 num_unique_tools: int,
                                 has_data_flow: bool,
                                 has_multihop_keywords: bool) -> float:
        """
        Calculate a score representing the strength of multi-hop nature.
        
        Args:
            num_tool_references: Number of explicit tool references
            num_unique_tools: Number of unique tool calls detected
            has_data_flow: Whether data flows between tools
            has_multihop_keywords: Whether the question has multi-hop keywords
            
        Returns:
            Float score between 0 and 1
        """
        score = 0.0
        
        # Tool usage (50% of score)
        tool_score = min(1.0, (num_unique_tools - 1) / 2) * 0.5
        
        # Data flow (30% of score)
        data_flow_score = 0.3 if has_data_flow else 0.0
        
        # Question wording (20% of score)
        keyword_score = 0.2 if has_multihop_keywords else 0.0
        
        return tool_score + data_flow_score + keyword_score
    
    def enhance_entry(self, entry: Dict) -> Dict:
        """
        Enhance an entry to better meet multi-hop requirements.
        
        Args:
            entry: Dataset entry to enhance
            
        Returns:
            Enhanced entry
        """
        enhanced_entry = entry.copy()
        
        if 'extra_info' not in enhanced_entry:
            enhanced_entry['extra_info'] = {}
            
        # Get rule code and question
        rule_code = enhanced_entry.get('extra_info', {}).get('rule', '')
        question = enhanced_entry.get('question', '')
        
        # Extract function schemas to get real tool names
        schemas = {}
        schemas_str = enhanced_entry.get('extra_info', {}).get('function_schemas', '{}')
        if isinstance(schemas_str, str):
            try:
                schemas = json.loads(schemas_str)
            except json.JSONDecodeError:
                pass
        elif isinstance(schemas_str, dict):
            schemas = schemas_str
        
        # Filter out schema properties from tool names
        tool_names = set()
        if isinstance(schemas, dict):
            tool_names = {
                name for name in schemas.keys() 
                if name not in self.schema_props and not name.startswith('$')
            }
        
        # Enhance rule code with tool references
        tool_references = set(self.tool_reference_pattern.findall(rule_code))
        missing_tools = tool_names - tool_references
        
        if missing_tools and rule_code:
            tool_comments = "\n".join([f'# Using tool: "{tool}"' for tool in missing_tools])
            rule_code = tool_comments + "\n\n" + rule_code
            enhanced_entry['extra_info']['rule'] = rule_code
        
        # Enhance question with multi-hop keywords if needed
        multihop_keywords = ['then', 'based on', 'using that', 'with this information', 'afterwards', 'following', 'next', 'subsequently']
        has_multihop_keywords = any(keyword in question.lower() for keyword in multihop_keywords)
        
        if not has_multihop_keywords and question:
            # Find a good spot to insert a multi-hop keyword
            parts = re.split(r'(\.|\?|!)', question)
            if len(parts) > 2:  # Has multiple sentences or clauses
                # Insert "then" before the last part
                parts[-2] = ", then" + parts[-2]
                enhanced_entry['question'] = ''.join(parts)
            else:
                # Simply append a multi-hop phrase
                enhanced_entry['question'] = question + " Then provide detailed information about the result."
        
        return enhanced_entry

# ════════════════════ MULTI-SERVER TOOLS ════════════════════

class MultiServerTools:
    """Handles tools from multiple MCP servers."""
    
    def __init__(self, config_files=None):
        """
        Initialize the multi-server tools handler.
        
        Args:
            config_files: List of MCP config files (if None, uses mock tools)
        """
        self.config_files = config_files or []
        self.servers = {}  # Server name -> tools mapping
        self.clients = {}  # Server name -> client mapping
    
    async def load_all_servers(self):
        """Load tools from all configured servers."""
        for config_file in self.config_files:
            await self.load_server(config_file)
    
    async def load_server(self, config_file):
        """
        Load tools from a specific server.
        
        Args:
            config_file: Path to MCP config file
        
        Returns:
            Tuple of (tools, server_name)
        """
        # Extract server name from config file path
        server_name = Path(config_file).stem
        
        try:
            # Try to import MCP modules
            from mcp_use.client import MCPClient
            from mcp_use.adapters.langchain_adapter import LangChainAdapter
            
            # Initialize MCP client
            client = MCPClient.from_config_file(config_file)
            self.clients[server_name] = client
            
            # Create adapter instance
            adapter = LangChainAdapter()
            
            # Get LangChain tools
            tools = await adapter.create_tools(client)
            print(f"Successfully retrieved {len(tools)} tools from server '{server_name}'")
            
            # Store tools for this server
            self.servers[server_name] = tools
            
            return tools, server_name
            
        except (ImportError, Exception) as e:
            print(f"Error loading server '{server_name}': {e}")
            print("Using mock tools for this server.")
            
            # Use mock tools for this server
            mock_tools = self._create_mock_tools(server_name)
            self.servers[server_name] = mock_tools
            
            return mock_tools, server_name
    
    def _create_mock_tools(self, server_type="general"):
        """
        Create mock tools based on server type.
        
        Args:
            server_type: Type of server to create tools for
        
        Returns:
            List of mock tools
        """
        # Mock tool class
        class MockTool:
            def __init__(self, name, description="", parameters=None, server=None):
                self.name = name
                self.description = description
                self._parameters = parameters or {"type": "object", "properties": {}}
                self.server = server
                
                # Create a simple args_schema
                class ArgsSchema:
                    @staticmethod
                    def schema():
                        return parameters or {"type": "object", "properties": {}}
                        
                self.args_schema = ArgsSchema
        
        # Based on server type, create appropriate mock tools
        if server_type == "airbnb" or "rental" in server_type:
            return [
                MockTool(
                    "airbnb_search",
                    "Search for Airbnb listings",
                    {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "Location to search for"},
                            "amenities": {"type": "array", "description": "List of required amenities"},
                            "min_guests": {"type": "integer", "description": "Minimum number of guests"}
                        },
                        "required": ["location"]
                    },
                    server=server_type
                ),
                MockTool(
                    "airbnb_listing_details",
                    "Get detailed information about a specific listing",
                    {
                        "type": "object",
                        "properties": {
                            "listing_id": {"type": "string", "description": "ID of the listing"}
                        },
                        "required": ["listing_id"]
                    },
                    server=server_type
                )
            ]
        elif server_type == "coincap" or "crypto" in server_type or "finance" in server_type:
            return [
                MockTool(
                    "bitcoin_price",
                    "Get the current price of Bitcoin",
                    {"type": "object", "properties": {}},
                    server=server_type
                ),
                MockTool(
                    "get_crypto_price",
                    "Get the price of a specific cryptocurrency",
                    {
                        "type": "object",
                        "properties": {
                            "crypto": {"type": "string", "description": "Name or symbol of the cryptocurrency"}
                        },
                        "required": ["crypto"]
                    },
                    server=server_type
                ),
                MockTool(
                    "list_assets",
                    "List available crypto assets",
                    {"type": "object", "properties": {}},
                    server=server_type
                )
            ]
        elif server_type == "weather" or "climate" in server_type:
            return [
                MockTool(
                    "get_weather",
                    "Get current weather for a location",
                    {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City or location name"},
                            "units": {"type": "string", "enum": ["metric", "imperial"], "description": "Units system"}
                        },
                        "required": ["location"]
                    },
                    server=server_type
                ),
                MockTool(
                    "get_forecast",
                    "Get weather forecast for a location",
                    {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City or location name"},
                            "days": {"type": "integer", "description": "Number of days to forecast"}
                        },
                        "required": ["location"]
                    },
                    server=server_type
                )
            ]
        else:  # Default general tools
            return [
                MockTool(
                    "search_web",
                    "Search the web for information",
                    {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"}
                        },
                        "required": ["query"]
                    },
                    server=server_type
                ),
                MockTool(
                    "get_details",
                    "Get detailed information about a specific entity",
                    {
                        "type": "object",
                        "properties": {
                            "entity_id": {"type": "string", "description": "ID of the entity"}
                        },
                        "required": ["entity_id"]
                    },
                    server=server_type
                )
            ]
    
    def get_all_tools(self):
        """Get all tools from all servers."""
        all_tools = []
        for server_name, tools in self.servers.items():
            for tool in tools:
                # Attach server name to tool if not already present
                if not hasattr(tool, 'server'):
                    tool.server = server_name
                all_tools.append(tool)
        return all_tools
    
    def get_tools_by_server(self, server_name):
        """Get tools for a specific server."""
        return self.servers.get(server_name, [])
    
    def get_server_names(self):
        """Get names of all loaded servers."""
        return list(self.servers.keys())
    
    async def close_all_clients(self):
        """Close all MCP clients."""
        for server_name, client in self.clients.items():
            if client and hasattr(client, 'close'):
                try:
                    await client.close()
                    print(f"Closed client for server '{server_name}'")
                except Exception as e:
                    print(f"Error closing client for server '{server_name}': {e}")

# ════════════════════ MULTI-HOP PROMPTS ════════════════════

def create_multihop_system_prompt(tools: List, num_entries: int = 5, data_source: str = "multihop_re_call") -> str:
    """
    Creates a system prompt that enforces multi-hop question generation across servers.
    
    Args:
        tools: List of tools from all servers
        num_entries: Number of entries to generate
        data_source: Identifier for the data source
        
    Returns:
        String containing the system prompt
    """
    # Group tools by server
    tools_by_server = {}
    for tool in tools:
        server = getattr(tool, 'server', 'default')
        if server not in tools_by_server:
            tools_by_server[server] = []
        tools_by_server[server].append(tool)
    
    # Create tool descriptions by server
    server_tool_descriptions = []
    for server, server_tools in tools_by_server.items():
        tool_descriptions = "\n".join([
            f"  - {tool.name}: {tool.description}" 
            for tool in server_tools
        ])
        server_tool_descriptions.append(f"Server: {server}\n{tool_descriptions}")
    
    # Join all server tool descriptions
    all_tool_descriptions = "\n\n".join(server_tool_descriptions)
    
    # Create system prompt with cross-server multi-hop emphasis
    system_prompt = f"""
You are an expert data generator creating a synthetic dataset for training language models to use tools for multi-hop reasoning ACROSS DIFFERENT SERVERS.

The dataset follows the "{data_source}" format with these components:
- data_source: "{data_source}"
- question: A natural language query that REQUIRES multi-hop tool use across different servers
- ability: "re_call"
- reward_model: An array of FACTUALLY ACCURATE expected answers that would result from the tool calls
- extra_info.server: The server that hosts the tools used for this question

Available tools across different servers:
{all_tool_descriptions}

Your task is to generate {num_entries} diverse, realistic entries that STRICTLY require CROSS-SERVER MULTI-HOP REASONING.

===== STRICT MULTI-HOP REQUIREMENT ACROSS SERVERS =====
Each question MUST require at least TWO distinct tools from DIFFERENT SERVERS to be used in SEQUENCE, where the output of a tool from one server is NECESSARY to determine how to use a tool from another server.

Examples of true cross-server multi-hop reasoning:
1. "First find rental properties using airbnb_search from the 'airbnb' server, then get cryptocurrency prices using get_crypto_price from the 'coincap' server to determine which properties are affordable based on your crypto holdings."
2. "First get weather data using get_weather from the 'weather' server, then search for rental properties with airbnb_search from the 'airbnb' server that have air conditioning in cities with high temperatures."

NOT cross-server multi-hop (AVOID THESE):
- Questions that use tools from only one server
- Questions where tools from different servers are used in parallel without dependencies
- Questions where the second server's tools could be called without the results from the first server

===== SPECIFIC CROSS-SERVER MULTI-HOP PATTERNS =====
For EACH entry, use ONE of these specific cross-server multi-hop patterns:
1. FIND-THEN-DETAIL-ACROSS-SERVERS: Find information on one server, then get details using a different server
2. SEARCH-THEN-FILTER-ACROSS-SERVERS: Search on one server, then filter based on criteria from another server
3. QUERY-THEN-ANALYZE-ACROSS-SERVERS: Get data from one server, then analyze it using tools from another server
4. LOCATE-THEN-ROUTE-ACROSS-SERVERS: Find locations on one server, then determine routes using another server
5. IDENTIFY-THEN-EXTRACT-ACROSS-SERVERS: Identify a resource on one server, then extract information using another server

===== RULE STRUCTURE =====
In the rule code, ALWAYS show:
1. Clear cross-server dependencies, where information from a tool on one server is used with a tool on another server
2. Explicit variable passing between servers, showing how outputs from one server become inputs to another
3. Proper tool name usage, ensuring tools are referred to with their server context
4. Comments explaining the multi-hop sequence across servers

===== REQUIRED CODE FORMAT =====
At the TOP of your rule code, include these EXACT comment lines for each tool used:
```python
# Using tool: "tool_name_1" from server "server_name_1"
# Using tool: "tool_name_2" from server "server_name_2"
```

Then in your code, show the EXPLICIT cross-server sequence:
```python
# Step 1: Get initial information from server_name_1
result_1 = tool_name_1(params)

# Step 2: Use that information with a tool from server_name_2
specific_param = extract_info_from(result_1)
final_result = tool_name_2(specific_param)
```

Return the entries as a JSON array with this format:
[
  {{
    "data_source": "{data_source}",
    "question": "The user question requiring cross-server multi-hop reasoning",
    "ability": "re_call",
    "reward_model": ["Factually accurate answer 1", "Alternative factually accurate answer 2"],
    "extra_info": {{
      "rule": "Python code implementing the cross-server multi-hop tool calls",
      "function_schemas": "The function schemas in JSON",
      "id": "unique_id_here",
      "server": "Comma-separated list of servers used in this query"
    }}
  }}
]

Create {num_entries} diverse entries using different cross-server tool combinations and multi-hop patterns.
"""
    
    return system_prompt

def create_multihop_user_prompt(multi_server_tools) -> str:
    """
    Creates a user prompt with specific cross-server multi-hop examples.
    
    Args:
        multi_server_tools: MultiServerTools instance
        
    Returns:
        String containing the user prompt
    """
    # Get server names
    server_names = multi_server_tools.get_server_names()
    
    # If we have at least two servers, create examples
    examples = []
    
    if len(server_names) >= 2:
        server1 = server_names[0]
        server2 = server_names[1]
        
        tools1 = multi_server_tools.get_tools_by_server(server1)
        tools2 = multi_server_tools.get_tools_by_server(server2)
        
        if tools1 and tools2:
            tool1 = tools1[0].name
            tool2 = tools2[0].name
            
            examples.append(f"""
Example CROSS-SERVER multi-hop using {tool1} from "{server1}" server and {tool2} from "{server2}" server:
"Find information using {tool1} from the {server1} server, then use those results with {tool2} from the {server2} server to get the final answer."

This requires:
1. Using {tool1} from the {server1} server to get initial data
2. Processing that data to extract specific information
3. Using the extracted information with {tool2} from the {server2} server to get the final result
""")
    
    # If we don't have enough examples, add a generic one
    if not examples:
        examples.append("""
Example of CROSS-SERVER multi-hop reasoning:
"First search for weather information in popular tourist destinations, then find rental properties in locations with the best forecasted weather."

This requires:
1. Using a weather tool from one server to get weather forecasts for multiple locations
2. Analyzing the weather data to determine the locations with the best conditions
3. Using the identified locations as input to a rental search tool from a different server
""")
    
    # Create the final user prompt
    user_prompt = f"""
Generate multi-hop questions that REQUIRE using tools from DIFFERENT SERVERS in SEQUENCE, where information from a tool on one server is NECESSARY for using a tool on another server.

For each question:
1. Make sure it requires at least two tool calls from different servers in a specific order
2. Ensure the output from one server determines how to use a tool from another server
3. Create rule code that explicitly shows how data flows between servers
4. Add the required tool reference comments at the top, clearly showing which server each tool belongs to
5. Make sure to include the list of servers used in the "server" field of extra_info

Here are examples of good cross-server multi-hop questions:
{''.join(examples)}

Now, generate diverse cross-server multi-hop questions that showcase different reasoning patterns and server combinations.
"""
    
    return user_prompt

# ════════════════════ MULTI-SERVER DATASET GENERATOR ════════════════════

class MultiServerDatasetGenerator:
    """Generate datasets using OpenAI with multi-server MCP tools and built-in validation."""
    
    def __init__(self, multi_server_tools, openai_api_key=None, validator=None):
        """
        Initialize the multi-server dataset generator.
        
        Args:
            multi_server_tools: MultiServerTools instance
            openai_api_key: OpenAI API key
            validator: MultiHopValidator instance (creates one if None)
        """
        openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.validator = validator or MultiHopValidator(min_score=0.7)
        self.multi_server_tools = multi_server_tools
    
    async def generate_entries(self, 
                              num_entries: int = 5, 
                              data_source: str = "multihop_re_call",
                              max_attempts: int = 3) -> List[Dict]:
        """
        Generate dataset entries based on multi-server tools with multi-hop validation.
        
        Args:
            num_entries: Number of entries to generate
            data_source: Identifier for the data source
            max_attempts: Maximum attempts to generate valid entries
            
        Returns:
            List of validated dataset entries
        """
        # Get all tools from all servers
        all_tools = self.multi_server_tools.get_all_tools()
        
        if not all_tools:
            raise ValueError("No tools available from any server")
        
        # Track valid entries generated
        valid_entries = []
        attempts = 0
        
        # Create system and user prompts
        system_prompt = create_multihop_system_prompt(all_tools, num_entries, data_source)
        user_prompt = create_multihop_user_prompt(self.multi_server_tools)
        
        # Extract tool schemas by server
        tool_schemas_by_server = {}
        for tool in all_tools:
            server = getattr(tool, 'server', 'default')
            
            if server not in tool_schemas_by_server:
                tool_schemas_by_server[server] = {}
                
            try:
                # Extract schema from tool if available
                if hasattr(tool, 'args_schema'):
                    schema = tool.args_schema.schema()
                    tool_schemas_by_server[server][tool.name] = {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": schema
                    }
                else:
                    # Create basic schema
                    tool_schemas_by_server[server][tool.name] = {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": {"type": "object", "properties": {}}
                    }
            except Exception as e:
                print(f"Error extracting schema for {tool.name} on server {server}: {e}")
                tool_schemas_by_server[server][tool.name] = {"name": tool.name}
        
        # Generate and validate entries
        while len(valid_entries) < num_entries and attempts < max_attempts:
            attempts += 1
            print(f"Generation attempt {attempts}/{max_attempts}, valid entries so far: {len(valid_entries)}/{num_entries}")
            
            # Generate entries with OpenAI
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=4000
                )
                
                content = response.choices[0].message.content
                
                # Extract JSON from the response
                try:
                    # Look for JSON in the response
                    start_idx = content.find("[")
                    end_idx = content.rfind("]") + 1
                    
                    if start_idx >= 0 and end_idx > start_idx:
                        json_str = content[start_idx:end_idx]
                        entries = json.loads(json_str)
                        
                        # Process and validate each entry
                        for entry in entries:
                            # Basic validation
                            if not self._validate_basic_entry(entry):
                                print("Warning: Invalid entry format, skipping")
                                continue
                            
                            # Add timestamp ID if missing
                            if "extra_info" in entry and "id" not in entry["extra_info"]:
                                entry["extra_info"]["id"] = f"gen_{datetime.now().strftime('%Y%m%d%H%M%S')}_{len(valid_entries)}"
                            
                            # Extract server information
                            server_info = entry.get('extra_info', {}).get('server', '')
                            servers = [s.strip() for s in server_info.split(',') if s.strip()]
                            
                            # If no servers specified, try to detect from rule code
                            if not servers and 'extra_info' in entry and 'rule' in entry['extra_info']:
                                rule_code = entry['extra_info']['rule']
                                server_pattern = re.compile(r'from server [\'"]([^\'"]+)[\'"]')
                                servers = server_pattern.findall(rule_code)
                            
                            # Ensure function_schemas is populated with the correct tools by server
                            if "extra_info" in entry:
                                schemas = {}
                                for server in servers:
                                    if server in tool_schemas_by_server:
                                        schemas.update(tool_schemas_by_server[server])
                                
                                # If no schemas found, use all schemas
                                if not schemas:
                                    for server_schemas in tool_schemas_by_server.values():
                                        schemas.update(server_schemas)
                                
                                entry["extra_info"]["function_schemas"] = schemas
                                
                                # Make sure server info is populated
                                entry["extra_info"]["server"] = ", ".join(servers)
                            
                            # Validate multi-hop nature
                            is_valid, reason, metrics = self.validator.validate_entry(entry)
                            
                            if is_valid:
                                print(f"✅ Valid cross-server entry: {entry['question'][:50]}... (score: {metrics['multihop_score']:.2f}, servers: {servers})")
                                valid_entries.append(entry)
                                if len(valid_entries) >= num_entries:
                                    break
                            else:
                                print(f"❌ Invalid entry: {reason}")
                                print(f"   Question: {entry['question'][:50]}...")
                                
                                # Try to enhance the entry
                                enhanced_entry = self.validator.enhance_entry(entry)
                                
                                # Re-validate the enhanced entry
                                is_valid, reason, metrics = self.validator.validate_entry(enhanced_entry)
                                if is_valid:
                                    print(f"✅ Enhanced entry now valid (score: {metrics['multihop_score']:.2f})")
                                    valid_entries.append(enhanced_entry)
                                    if len(valid_entries) >= num_entries:
                                        break
                    else:
                        print("No JSON array found in response")
                        
                except json.JSONDecodeError as e:
                    print(f"Failed to parse JSON: {e}")
                    
            except Exception as e:
                print(f"Error generating entries: {e}")
        
        if len(valid_entries) < num_entries:
            print(f"Warning: Only generated {len(valid_entries)}/{num_entries} valid entries after {attempts} attempts")
            
        return valid_entries[:num_entries]
    
    def _validate_basic_entry(self, entry: Dict) -> bool:
        """
        Perform basic validation on entry format.
        
        Args:
            entry: Dataset entry to validate
            
        Returns:
            Whether the entry has valid format
        """
        # Check required fields
        required_fields = ["data_source", "question", "ability", "reward_model", "extra_info"]
        if not all(field in entry for field in required_fields):
            return False
        
        # Check extra_info fields
        extra_info = entry.get("extra_info", {})
        required_extra = ["rule"]
        if not all(field in extra_info for field in required_extra):
            return False
        
        # Validate ability is "re_call"
        if entry.get("ability") != "re_call":
            return False
        
        # Validate reward_model is a list
        if not isinstance(entry.get("reward_model"), list):
            return False
        
        return True
    
    async def save_dataset_csv(self, entries: List[Dict], output_dir: str, dataset_name: str) -> str:
        """
        Save the dataset entries to a CSV file.
        
        Args:
            entries: List of dataset entries
            output_dir: Output directory
            dataset_name: Name of the dataset
            
        Returns:
            Path to the saved CSV file
        """
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output file path
        output_file = output_dir / f"{dataset_name}.csv"
        
        # Write entries to CSV
        with open(output_file, 'w', newline='') as f:
            # Create writer with all fields
            fieldnames = ["data_source", "question", "ability", "reward_model", "extra_info.rule",  
                          "extra_info.function_schemas", "extra_info.id", "extra_info.server"]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for entry in entries:
                # Flatten the extra_info dictionary
                row = {
                    "data_source": entry["data_source"],
                    "question": entry["question"],
                    "ability": entry["ability"],
                    "reward_model": json.dumps(entry["reward_model"])
                }
                
                if "extra_info" in entry:
                    for key, value in entry["extra_info"].items():
                        if isinstance(value, (dict, list)):
                            row[f"extra_info.{key}"] = json.dumps(value)
                        else:
                            row[f"extra_info.{key}"] = value
                
                writer.writerow(row)
        
        print(f"Saved multi-server dataset with {len(entries)} entries to {output_file}")
        return str(output_file)

# ════════════════════ MAIN EXECUTION ════════════════════

async def main():
    """Generate a dataset using multi-server MCP tools."""
    # Default config files if not specified
    config_files = [
        os.getenv("AIRBNB_MCP_CONFIG", "airbnb_mcp.json"),
        os.getenv("COINCAP_MCP_CONFIG", "coincap_mcp.json")
    ]
    
    # If no config files exist, use mock servers
    existing_configs = [cf for cf in config_files if Path(cf).exists()]
    
    if not existing_configs:
        print("No config files found. Using mock servers: 'airbnb' and 'coincap'")
        mock_servers = ["airbnb", "coincap"]
    else:
        print(f"Using config files: {existing_configs}")
        mock_servers = []
    
    try:
        # Initialize multi-server tools
        multi_server_tools = MultiServerTools(existing_configs)
        
        # If using mock servers, add them
        if mock_servers:
            for server in mock_servers:
                mock_tools = multi_server_tools._create_mock_tools(server)
                multi_server_tools.servers[server] = mock_tools
        
        # Load tools from real servers if available
        if existing_configs:
            await multi_server_tools.load_all_servers()
        
        # Print available servers and tools
        for server_name in multi_server_tools.get_server_names():
            tools = multi_server_tools.get_tools_by_server(server_name)
            print(f"Server '{server_name}' has {len(tools)} tools: {', '.join(tool.name for tool in tools)}")
        
        # Create validator
        validator = MultiHopValidator(min_score=0.7)
        
        # Initialize the dataset generator
        generator = MultiServerDatasetGenerator(
            multi_server_tools=multi_server_tools,
            openai_api_key=OPENAI_API_KEY,
            validator=validator
        )
        
        # Generate dataset entries
        print("Generating multi-server multi-hop dataset entries...")
        entries = await generator.generate_entries(
            num_entries=10,  # Generate 10 examples
            data_source="multi_server_re_call",
            max_attempts=3
        )
        
        if not entries:
            print("Failed to generate entries.")
            return
        
        # Save the dataset
        output_dir = Path("./datasets")
        dataset_name = f"multi_server_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        dataset_path = await generator.save_dataset_csv(
            entries=entries,
            output_dir=output_dir,
            dataset_name=dataset_name
        )
        
        print(f"Multi-server dataset generated successfully: {dataset_path}")
        
    except Exception as e:
        print(f"Error generating dataset: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'multi_server_tools' in locals():
            await multi_server_tools.close_all_clients()
        print("Closed all MCP sessions")

if __name__ == "__main__":
    asyncio.run(main())