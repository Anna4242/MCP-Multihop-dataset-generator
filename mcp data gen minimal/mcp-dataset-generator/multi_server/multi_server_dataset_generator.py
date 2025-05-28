"""
Multi-Server Dataset Generator - Creates datasets with entries distributed across multiple MCP servers
"""
import asyncio
import json
import os
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

import openai
from langchain_openai import ChatOpenAI
from mcp_use.client import MCPClient
from mcp_use.adapters.langchain_adapter import LangChainAdapter
from dotenv import load_dotenv
from multi_hop_verifier import MultiHopVerifier, verify_dataset, generate_verification_stats

# Load environment variables
load_dotenv()

# Set OpenRouter API key and base URL
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")  # Default to OpenRouter URL
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY missing in .env or environment")

# Default path to the config file
DEFAULT_CONFIG_PATH = Path(r"D:\one drive\study\ARCEE AI INTERNSHIP\mcp data gen minimal\mcp-dataset-generator\multi_server\config.json")

def load_config(config_path: Path) -> Dict:
    """Load and parse an MCP configuration file"""
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        return {}
    
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        return config_data
    except Exception as e:
        print(f"Error reading config file: {e}")
        return {}

def list_servers(config_data: Dict) -> List[str]:
    """Extract server names from configuration data"""
    servers = []
    if "mcpServers" in config_data:
        servers = list(config_data["mcpServers"].keys())
    return servers

def select_servers(servers: List[str]) -> List[str]:
    """Allow user to select which servers to use"""
    if not servers:
        print("No servers found in configuration!")
        return []
    
    print("\nAvailable MCP Servers:")
    for i, server in enumerate(servers, 1):
        print(f"{i}. {server}")
    
    print(f"\n{len(servers) + 1}. All servers")
    
    while True:
        choice = input("\nEnter server number(s) to use (comma-separated, or 'all'): ")
        
        if choice.lower() in ('all', 'a', str(len(servers) + 1)):
            print(f"Selected all {len(servers)} servers")
            return servers
        
        try:
            # Parse comma-separated list of numbers
            selections = [int(x.strip()) for x in choice.split(',')]
            
            # Validate selections
            valid_selections = [s for s in selections if 1 <= s <= len(servers)]
            
            if not valid_selections:
                print("No valid selections. Please try again.")
                continue
            
            # Convert to server names
            selected_servers = [servers[i-1] for i in valid_selections]
            
            print(f"Selected servers: {', '.join(selected_servers)}")
            return selected_servers
            
        except ValueError:
            print("Invalid input. Please enter numbers separated by commas.")

def create_filtered_config(config_data: Dict, selected_servers: List[str]) -> Dict:
    """Create a new configuration with only the selected servers"""
    if "mcpServers" not in config_data:
        return {}
    
    filtered_config = {"mcpServers": {}}
    
    for server in selected_servers:
        if server in config_data["mcpServers"]:
            filtered_config["mcpServers"][server] = config_data["mcpServers"][server]
    
    return filtered_config

def create_server_specific_config(config_data: Dict, server_name: str) -> Dict:
    """Create a configuration with only a single server"""
    if "mcpServers" not in config_data or server_name not in config_data["mcpServers"]:
        return {}
    
    return {
        "mcpServers": {
            server_name: config_data["mcpServers"][server_name]
        }
    }

async def get_mcp_tools_for_server(config_data: Dict, server_name: str) -> Tuple[List, Any]:
    """
    Retrieves tools from a specific MCP server.
    
    Args:
        config_data: The full configuration data
        server_name: Name of the server to connect to
        
    Returns:
        Tuple of (tools, client)
    """
    # Create server-specific config
    server_config = create_server_specific_config(config_data, server_name)
    
    # Save to temporary file
    temp_config_path = Path(f"temp_{server_name}_config.json")
    with open(temp_config_path, 'w') as f:
        json.dump(server_config, f, indent=2)
    
    try:
        # Initialize MCP client for this server
        client = MCPClient.from_config_file(str(temp_config_path))
        
        # Create adapter instance
        adapter = LangChainAdapter()
        
        # Get LangChain tools
        tools = await adapter.create_tools(client)
        print(f"Successfully retrieved {len(tools)} tools from {server_name} server")
        
        # Print tool names
        if tools:
            tool_names = [tool.name for tool in tools]
            print(f"Available tools from {server_name}: {', '.join(tool_names[:5])}...")
        
        return tools, client
    except Exception as e:
        print(f"Error retrieving tools from {server_name}: {e}")
        import traceback
        traceback.print_exc()
        return [], None
    finally:
        # Clean up temporary file
        if temp_config_path.exists():
            temp_config_path.unlink()

class DatasetGenerator:
    """Generate datasets using OpenAI and MCP tools."""
    
    def __init__(self, openai_api_key=None):
        """Initialize the dataset generator."""
        openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = openai.OpenAI(
            api_key=openai_api_key,
            base_url=OPENAI_API_BASE
        )
    
    async def generate_entries(self, 
                              tools: List,
                              num_entries: int = 5, 
                              data_source: str = "syntool_re_call",
                              server_name: str = "unknown") -> List[Dict]:
        """Generate dataset entries based on MCP tools.
        
        Args:
            tools: List of LangChain tools from MCP
            num_entries: Number of entries to generate
            data_source: Identifier for the data source
            server_name: Name of the server providing the tools
            
        Returns:
            List of dataset entries
        """
        # Extract tool information
        tool_descriptions = "\n".join([
            f"- {tool.name}: {tool.description}" 
            for tool in tools
        ])
        
        # Get tool schemas
        tool_schemas = {}
        for tool in tools:
            try:
                # Extract schema from tool if available
                if hasattr(tool, 'args_schema'):
                    try:
                        # Try the newer Pydantic v2 method first
                        if hasattr(tool.args_schema, 'model_json_schema'):
                            schema = tool.args_schema.model_json_schema()
                        else:
                            # Fall back to the older method
                            schema = tool.args_schema.schema()
                            
                        tool_schemas[tool.name] = {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": schema
                        }
                    except Exception as schema_e:
                        print(f"Error extracting schema for {tool.name}: {schema_e}")
                        tool_schemas[tool.name] = {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": {"type": "object", "properties": {}}
                        }
                else:
                    # Create basic schema
                    tool_schemas[tool.name] = {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": {"type": "object", "properties": {}}
                    }
            except Exception as e:
                print(f"Error extracting schema for {tool.name}: {e}")
                tool_schemas[tool.name] = {"name": tool.name}
        
        # Create a system prompt that explains the task
        system_prompt = f"""
You are an expert data generator creating a synthetic dataset for training language models to use tools for multi-hop reasoning.

The dataset follows the "syntool_re_call" format with these components:
- data_source: "{data_source}"
- question: A natural language query that requires multi-hop tool use to answer
- ability: "re_call"
- reward_model: An array of FACTUALLY ACCURATE expected answers that would result from the tool calls

You are generating examples for the "{server_name}" MCP server, which provides these tools:
{tool_descriptions}

Your task is to generate {num_entries} diverse, realistic entries that specifically require MULTI-HOP REASONING.
Multi-hop reasoning means the question requires:
1. Using MULTIPLE tools in SEQUENCE (e.g., search for something, then get details about a result)
2. COMBINING information from different tool calls
3. Making FOLLOW-UP tool calls based on initial results

CRITICALLY IMPORTANT: The reward_model must contain FACTUALLY ACCURATE information only!
- Only include verifiably true information in the reward_model
- Use widely accepted facts and information that would be returned by actual tool calls
- Avoid speculative, made-up, or potentially false information
- Use established facts, numbers, and statistics that you are confident are accurate
- If uncertain about a fact, use more general statements that are definitely true
- Structure the rewards as if they were produced by a knowledgeable system

For each entry:
1. Create a realistic user question that REQUIRES multi-hop reasoning across multiple tool calls
2. Create realistic rule code that implements the multi-step tool functionality in Python
3. Create reward_model values that would be FACTUALLY ACCURATE answers
4. Include the function schemas

Return the entries as a JSON array with this format:
[
  {{
    "data_source": "{data_source}",
    "question": "The user question requiring multi-hop reasoning",
    "ability": "re_call",
    "reward_model": ["Factually accurate answer 1", "Alternative factually accurate answer 2"],
    "extra_info": {{
      "rule": "Python code implementing the multi-hop tool calls",
      "function_schemas": "The function schemas in JSON",
      "env": "Environmental configuration required for execution",
      "id": "unique_id_here",
      "server": "{server_name}"
    }}
  }}
]

In the rule code, make sure to CLEARLY show:
1. The sequence of tool calls
2. How information from one tool call is used to inform the next
3. How the final answer is composed from multiple tool calls

IMPORTANT: Create diverse scenarios that use different combinations of the available tools. Make sure to ONLY use tools from the list provided, as these are the only ones available on this specific MCP server.
"""

        # Generate entries with OpenAI
        try:
            # Set up the parameters for the API call
            params = {
                "model": "openai/gpt-4o",  # Updated model name with provider prefix for OpenRouter
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Generate {num_entries} diverse dataset entries specifically for the {server_name} MCP server using only the tools listed. Make sure they require multi-hop reasoning."}
                ],
                "temperature": 0.7,
                "max_tokens": 4000
            }
            
            # Try to call the API with the most compatible approach
            try:
                # Try without any headers first (most compatible)
                response = self.client.chat.completions.create(**params)
            except Exception as e:
                print(f"Warning: Standard API call failed, trying alternative: {e}")
                # Try with direct API call as a fallback (older OpenAI client versions)
                try:
                    response = openai.ChatCompletion.create(
                        api_key=self.client.api_key,
                        api_base=OPENAI_API_BASE,
                        **params
                    )
                except Exception as e2:
                    print(f"Error: Both API call methods failed: {e2}")
                    raise
            
            # Extract content from response
            if hasattr(response, 'choices') and hasattr(response.choices[0], 'message'):
                content = response.choices[0].message.content
            elif isinstance(response, dict) and 'choices' in response:
                # Handle older response format
                content = response['choices'][0]['message']['content']
            else:
                print(f"Unexpected response format: {response}")
                return []
            
            # Extract JSON from the response
            try:
                # Look for JSON in the response
                start_idx = content.find("[")
                end_idx = content.rfind("]") + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = content[start_idx:end_idx]
                    entries = json.loads(json_str)
                    
                    # Validate entries and add environment config
                    valid_entries = []
                    for entry in entries:
                        if self._validate_entry(entry):
                            # Add timestamp ID and server name if missing
                            if "extra_info" in entry:
                                if "id" not in entry["extra_info"]:
                                    entry["extra_info"]["id"] = f"{server_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{len(valid_entries)}"
                                if "server" not in entry["extra_info"]:
                                    entry["extra_info"]["server"] = server_name
                                
                                # Ensure env is populated with tool schemas
                                if "env" not in entry["extra_info"]:
                                    entry["extra_info"]["env"] = self._create_env_config(entry, tool_schemas)
                            
                            valid_entries.append(entry)
                        else:
                            print(f"Warning: Invalid entry found from {server_name}, skipping")
                    
                    print(f"Generated {len(valid_entries)} valid entries for {server_name}")
                    return valid_entries
                else:
                    print(f"No JSON array found in response from {server_name}")
                    print(f"Content from {server_name}: {content[:200]}...")
                    return []
                    
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON from {server_name}: {e}")
                print(f"Content from {server_name}: {content[:200]}...")
                return []
                
        except Exception as e:
            print(f"Error generating entries for {server_name}: {e}")
            return []
    
    def _validate_entry(self, entry: Dict) -> bool:
        """Validate a dataset entry."""
        # Check required fields
        required_fields = ["data_source", "question", "ability", "reward_model", "extra_info"]
        if not all(field in entry for field in required_fields):
            return False
        
        # Check extra_info fields
        extra_info = entry.get("extra_info", {})
        required_extra = ["rule", "function_schemas"]
        if not all(field in extra_info for field in required_extra):
            return False
        
        # Validate ability is "re_call"
        if entry.get("ability") != "re_call":
            return False
        
        # Validate reward_model is a list
        if not isinstance(entry.get("reward_model"), list):
            return False
        
        return True
    
    def _create_env_config(self, entry: Dict, tool_schemas: Dict) -> Dict:
        """Create an environment configuration containing all tools used in the rule."""
        env_config = {"tools": {}}
        
        # Extract tool names from rule code
        rule_code = entry.get("extra_info", {}).get("rule", "")
        
        # Add all tools from schemas to ensure completeness
        for tool_name, schema in tool_schemas.items():
            env_config["tools"][tool_name] = schema
        
        return env_config
    
    async def save_dataset_csv(self, entries: List[Dict], output_dir: str, dataset_name: str) -> str:
        """Save the dataset entries to a CSV file."""
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output file path
        output_file = output_dir / f"{dataset_name}.csv"
        
        # Write entries to CSV
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            # Create writer with all fields
            fieldnames = ["data_source", "question", "ability", "reward_model", "extra_info.rule", "extra_info.env",  
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
        
        print(f"Saved dataset with {len(entries)} entries to {output_file}")

        return str(output_file)

async def generate_multi_server_dataset():
    """Generate a dataset with entries distributed across multiple MCP servers."""
    # Allow specifying config path as command-line argument
    import sys
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH
    
    print(f"Loading MCP server configuration from: {config_path}")
    
    # Load the configuration
    config_data = load_config(config_path)
    
    if not config_data:
        print("Failed to load configuration. Exiting.")
        return
    
    # List available servers
    servers = list_servers(config_data)
    
    if not servers:
        print("No MCP servers found in configuration. Exiting.")
        return
    
    # Allow user to select servers
    selected_servers = select_servers(servers)
    
    if not selected_servers:
        print("No servers selected. Exiting.")
        return
        
    # Get total number of entries to generate
    while True:
        try:
            total_entries = int(input("\nTotal number of entries to generate across all servers: "))
            if total_entries <= 0:
                print("Please enter a positive number.")
                continue
            break
        except ValueError:
            print("Please enter a valid number.")
    
    # Calculate entries per server (distribute evenly)
    server_count = len(selected_servers)
    base_entries_per_server = total_entries // server_count
    remainder = total_entries % server_count
    
    # Distribute entries ensuring the total equals the requested amount
    entries_distribution = {}
    for i, server in enumerate(selected_servers):
        # Add one extra entry to the first 'remainder' servers
        entries_distribution[server] = base_entries_per_server + (1 if i < remainder else 0)
    
    print("\nEntries distribution:")
    for server, count in entries_distribution.items():
        print(f"  {server}: {count} entries")
    
    # Initialize the dataset generator
    generator = DatasetGenerator(openai_api_key=OPENAI_API_KEY)
    
    # Connect to each server and generate entries
    all_entries = []
    clients = []  # Keep track of clients to close them later
    
    for server_name in selected_servers:
        print(f"\n{'='*50}")
        print(f"Processing server: {server_name}")
        print(f"{'='*50}")
        
        # Get tools from this server
        tools, client = await get_mcp_tools_for_server(config_data, server_name)
        if client:
            clients.append(client)
        
        if not tools:
            print(f"No tools available from {server_name}. Skipping.")
            continue
        
        # Generate entries for this server
        entries_count = entries_distribution[server_name]
        print(f"Generating {entries_count} entries for {server_name}...")
        
        entries = await generator.generate_entries(
            tools=tools,
            num_entries=entries_count,
            data_source="multi_server_mcp",
            server_name=server_name
        )
        
        if entries:
            all_entries.extend(entries)
            print(f"Added {len(entries)} entries from {server_name}")
        else:
            print(f"Failed to generate entries for {server_name}")
    
    # Close all clients
    for client in clients:
        if hasattr(client, 'close'):
            await client.close()
    
    # Save the combined dataset
    if all_entries:
        output_dir = Path("./datasets")
        dataset_name = f"multi_server_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        dataset_path = await generator.save_dataset_csv(
            entries=all_entries,
            output_dir=output_dir,
            dataset_name=dataset_name
        )
        
        print(f"\nMulti-server dataset generated successfully: {dataset_path}")
        print(f"Total entries: {len(all_entries)} from {len(selected_servers)} servers")
    else:
        print("\nFailed to generate any entries from the selected servers.")

if __name__ == "__main__":
    asyncio.run(generate_multi_server_dataset())