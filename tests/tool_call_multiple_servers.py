#!/usr/bin/env python3
"""
Interactive MCP Server Selector and Tool Explorer
"""
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from mcp_use.client import MCPClient
from mcp_use.adapters.langchain_adapter import LangChainAdapter
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY not found in environment")

# Default path to the config file
DEFAULT_CONFIG_PATH = Path(r"D:\one drive\study\ARCEE AI INTERNSHIP\mcp data gen minimal\mcp-dataset-generator\tests\config.json")

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

async def test_servers(config_data: Dict):
    """Test connecting to the selected MCP servers"""
    # Save filtered config to a temporary file
    temp_config_path = Path("temp_config.json")
    with open(temp_config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    print(f"\nTesting connection to {len(config_data.get('mcpServers', {}))} selected servers...")
    
    try:
        # Initialize MCP client with the config
        client = MCPClient.from_config_file(str(temp_config_path))
        
        # Create adapter instance
        adapter = LangChainAdapter()
        
        # Get LangChain tools
        tools = await adapter.create_tools(client)
        
        print(f"Successfully retrieved {len(tools)} tools from selected MCP servers")
        
        # Print tool names and descriptions
        for i, tool in enumerate(tools, 1):
            print(f"Tool {i}: {tool.name} - {tool.description}")
        
        # Try using LLM with tools if OpenAI key is available
        if OPENAI_API_KEY:
            use_llm = input("\nTest tools with LLM? (y/n): ").lower() == 'y'
            
            if use_llm:
                llm = ChatOpenAI(model="gpt-4o")
                llm_with_tools = llm.bind_tools(tools)
                
                query = input("\nEnter a query for the LLM: ") or "What tools do you have available?"
                
                print("\nSending query to LLM...")
                result = await llm_with_tools.ainvoke(query)
                print("\nLLM Response:")
                print(result.content)
        
    except Exception as e:
        print(f"Error testing MCP servers: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close client if needed
        if 'client' in locals() and hasattr(client, 'close'):
            await client.close()
        
        # Clean up temporary file
        if temp_config_path.exists():
            temp_config_path.unlink()
        
        print("\nTest completed")

async def main():
    """Main function to list servers and allow selection"""
    # Allow specifying config path as command-line argument
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
    
    # Create filtered configuration
    filtered_config = create_filtered_config(config_data, selected_servers)
    
    # Test the selected servers
    await test_servers(filtered_config)

if __name__ == "__main__":
    asyncio.run(main())