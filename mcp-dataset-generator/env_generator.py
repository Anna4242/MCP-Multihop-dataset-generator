"""MCP Environment Generator - Creates standardized env entries for training data"""
import asyncio
import os
import json
import shutil
from typing import Dict, List, Any
from langchain_openai import ChatOpenAI
from mcp_use.client import MCPClient
from mcp_use.adapters.langchain_adapter import LangChainAdapter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY missing in .env or environment")

async def get_mcp_tools(config_file="browser_mcp.json"):
    """
    Retrieves tools from an MCP server using LangChain adapter.
    
    Args:
        config_file: Path to the MCP server configuration file
    
    Returns:
        List of LangChain tools, MCPClient instance, and server name
    """
    # Extract server name from config file
    server_name = os.path.splitext(os.path.basename(config_file))[0]
    
    # Initialize MCP client
    client = MCPClient.from_config_file(config_file)
    
    # Create adapter instance
    adapter = LangChainAdapter()
    
    # Get LangChain tools
    try:
        tools = await adapter.create_tools(client)
        print(f"Successfully retrieved {len(tools)} tools from {server_name} MCP server")
        
        # Print tool names and descriptions
        for i, tool in enumerate(tools):
            print(f"Tool {i+1}: {tool.name} - {tool.description}")
        
        return tools, client, server_name
    except Exception as e:
        print(f"Error retrieving tools: {e}")
        import traceback
        traceback.print_exc()
        return [], client, server_name

async def generate_env_entry(tools, use_real_tools=True):
    """
    Generate a standardized env entry for the extra info column using GPT-4o.
    
    Args:
        tools: List of tools from the MCP server
        use_real_tools: Whether to use real tools or generate mock tools
    
    Returns:
        Dictionary containing env and func_schemas for the extra info column
    """
    try:
        # Initialize LLM
        llm = ChatOpenAI(model="gpt-4o")
        
        # Extract tool information for the prompt
        tool_info = []
        for tool in tools:
            tool_dict = {
                "name": tool.name,
                "description": tool.description
            }
            
            # Extract parameters if available
            if hasattr(tool, "args_schema"):
                try:
                    schema = tool.args_schema.schema()
                    tool_dict["parameters"] = schema.get("properties", {})
                except Exception as e:
                    print(f"Error extracting parameters for {tool.name}: {e}")
            
            tool_info.append(tool_dict)
        
        # Create the prompt for generating the env
        prompt = f"""Generate a standardized Python environment for a Model Context Protocol (MCP) server to be used in the "env" field of an extra info column. 

{'Available MCP tools (USE THESE EXACT TOOLS in your environment):' if use_real_tools else 'Available MCP tools (use these as examples but create a general-purpose env):'}
{json.dumps(tool_info, indent=2)}

Requirements:
1. Define a MCPTool dataclass with name, description, and parameters
2. {'Include EXACTLY the tools listed above with their exact parameters' if use_real_tools else 'Include 3-5 mock MCP tools with realistic parameters'}
3. Implement async functions for tool discovery and execution:
   - get_available_tools(workflow_request: Optional[str] = None)
   - execute_tool(tool_name: str, parameters: Dict[str, Any])
4. Include mock return values for each tool that match the expected return type
5. Make code realistic but concise (under 100 lines)
6. Include proper imports, type hints, and structured logging

Also generate the corresponding func_schemas JSON for these functions.

Return a valid JSON with two fields:
- "env": The Python code as a string
- "func_schemas": The JSON schema for the functions as a string
"""

        # Get the LLM's response
        response = await llm.ainvoke(prompt)
        
        # Parse the response to extract the env and func_schemas
        content = response.content
        
        # Find the JSON in the response
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = content[json_start:json_end]
            try:
                result = json.loads(json_str)
                return result
            except json.JSONDecodeError:
                # If direct JSON parsing fails, try to extract env and func_schemas separately
                env_start = content.find('```python')
                env_end = content.find('```', env_start + 10)
                
                if env_start >= 0 and env_end > env_start:
                    env_code = content[env_start + 10:env_end].strip()
                    
                    # Find func_schemas
                    schema_start = content.find('```json', env_end)
                    schema_end = content.find('```', schema_start + 8)
                    
                    if schema_start >= 0 and schema_end > schema_start:
                        func_schemas = content[schema_start + 8:schema_end].strip()
                        
                        return {
                            "env": env_code,
                            "func_schemas": func_schemas
                        }
        
        # If JSON extraction fails, return the whole response
        print("Warning: Could not parse structured response. Returning raw content.")
        return {
            "env": "# Error extracting structured environment\n" + content,
            "func_schemas": "[]"
        }
        
    except Exception as e:
        print(f"Error generating env entry: {e}")
        import traceback
        traceback.print_exc()
        return {
            "env": "# Error generating environment",
            "func_schemas": "[]"
        }

async def main():
    """Generate an env entry for the extra info column"""
    # Create the extra_info folder if it doesn't exist
    extra_info_dir = "extra_info"
    os.makedirs(extra_info_dir, exist_ok=True)
    
    # Get tools from MCP server
    config_file = "browser_mcp.json"  # Default config file
    tools, client, server_name = await get_mcp_tools(config_file)
    
    if not tools:
        print("No tools available to generate env entry.")
        return
    
    try:
        # Generate env entry with real tools
        print("\nGenerating environment entry with real tools...")
        env_entry = await generate_env_entry(tools, use_real_tools=True)
        
        # Save to file in the extra_info directory with the server name as the filename
        output_file = os.path.join(extra_info_dir, f"{server_name}.json")
        with open(output_file, "w") as f:
            json.dump(env_entry, f, indent=2)
        
        print(f"\nEnvironment entry generated and saved to {output_file}")
        print("\nSample environment code:")
        print(env_entry["env"][:500] + "..." if len(env_entry["env"]) > 500 else env_entry["env"])
    
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close MCP client if needed
        print("Finished processing")

if __name__ == "__main__":
    asyncio.run(main())