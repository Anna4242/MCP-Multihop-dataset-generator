"""
Dataset generator using OpenAI with MCP tools.
"""
import asyncio
import json
import os
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

import openai
from langchain_openai import ChatOpenAI
from mcp_use.client import MCPClient
from mcp_use.adapters.langchain_adapter import LangChainAdapter
from dotenv import load_dotenv

load_dotenv()                       # reads the .env file into os.environ

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY missing in .env or environment")

class DatasetGenerator:
    """Generate datasets using OpenAI and MCP tools."""
    
    def __init__(self, openai_api_key=None):
        """Initialize the dataset generator."""
        openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = openai.OpenAI(api_key=openai_api_key)
    
    async def generate_entries(self, 
                              tools: List,
                              num_entries: int = 5, 
                              data_source: str = "syntool_re_call") -> List[Dict]:
        """Generate dataset entries based on MCP tools.
        
        Args:
            tools: List of LangChain tools from MCP
            num_entries: Number of entries to generate
            data_source: Identifier for the data source
            
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
                    schema = tool.args_schema.schema()
                    tool_schemas[tool.name] = {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": schema
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


You have access to these tools:
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
      "id": "unique_id_here"
    }}
  }}
]

In the rule code, make sure to CLEARLY show:
1. The sequence of tool calls
2. How information from one tool call is used to inform the next
3. How the final answer is composed from multiple tool calls

In the env configuration, include ALL tools that are used in the rule and their full schemas.

Examples of multi-hop questions with factually accurate rewards:
- Question: "Find the tallest building in Chicago and tell me when it was completed"
  Factual reward: "The Willis Tower (formerly Sears Tower) is the tallest building in Chicago, standing at 1,450 feet (442 meters). It was completed in 1973 and held the title of tallest building in the world until 1998."

- Question: "What's the current population of Japan and how has it changed over the last decade?"
  Factual reward: "Japan's current population is approximately 125.7 million people. Over the last decade, Japan's population has been declining due to low birth rates and minimal immigration. The population has decreased by about 1.5 million people since 2010, representing one of the fastest rates of population decline in the world."

- Question: "Find the director of The Godfather and list two other famous movies they directed"
  Factual reward: "The Godfather was directed by Francis Ford Coppola. Two other famous movies he directed are Apocalypse Now (1979) and The Godfather Part II (1974), with the latter winning him the Academy Award for Best Director."

IMPORTANT: Create diverse scenarios that use different combinations of the available tools and ensure all facts in the reward_model are accurate.
"""

        # Generate entries with OpenAI
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Generate {num_entries} diverse dataset entries that would use the available MCP tools. Make sure they are realistic and require different types of tool interactions."}
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
                    
                    # Validate entries and add environment config
                    valid_entries = []
                    for entry in entries:
                        if self._validate_entry(entry):
                            # Add timestamp ID if missing
                            if "extra_info" in entry and "id" not in entry["extra_info"]:
                                entry["extra_info"]["id"] = f"gen_{datetime.now().strftime('%Y%m%d%H%M%S')}_{len(valid_entries)}"
                            
                            # Ensure env is populated with tool schemas
                            if "extra_info" in entry and "env" not in entry["extra_info"]:
                                entry["extra_info"]["env"] = self._create_env_config(entry, tool_schemas)
                            
                            valid_entries.append(entry)
                        else:
                            print("Warning: Invalid entry found, skipping")
                    
                    print(f"Generated {len(valid_entries)} valid entries")
                    return valid_entries
                else:
                    print("No JSON array found in response")
                    print(f"Content: {content[:200]}...")
                    return []
                    
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON: {e}")
                print(f"Content: {content[:200]}...")
                return []
                
        except Exception as e:
            print(f"Error generating entries: {e}")
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
        with open(output_file, 'w', newline='') as f:
            # Create writer with all fields
            fieldnames = ["data_source", "question", "ability", "reward_model", "extra_info.rule", "extra_info.env",  
                          "extra_info.function_schemas", "extra_info.id"]
            
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
        #print("Schema keys going into CSV:", list(entries[0]["extra_info"]["function_schemas"].keys())[:6], "…")

        return str(output_file)

async def get_mcp_tools(config_file="browser_mcp.json"):
    """
    Retrieves tools from an MCP server using LangChain adapter.
    
    Args:
        config_file: Path to the MCP server configuration file
    
    Returns:
        Tuple of (tools, client)
    """
    # Initialize MCP client
    client = MCPClient.from_config_file(config_file)
    
    # Create adapter instance
    adapter = LangChainAdapter()
    
    # Get LangChain tools
    try:
        tools = await adapter.create_tools(client)
        print(f"Successfully retrieved {len(tools)} tools from MCP server")
        
        # Print tool names
        tool_names = [tool.name for tool in tools]
        print(f"Available tools: {', '.join(tool_names)}")
        
        return tools, client
    except Exception as e:
        print(f"Error retrieving tools: {e}")
        import traceback
        traceback.print_exc()
        return [], client

async def main():
    """Generate a dataset using MCP tools and OpenAI."""
    # Get tools from MCP server
    print("Getting tools from MCP server...")
    tools, client = await get_mcp_tools()
    
    if not tools:
        print("No tools available. Exiting.")
        return
    
    try:
        # Initialize the dataset generator
        generator = DatasetGenerator(openai_api_key=OPENAI_API_KEY)
        
        # Generate dataset entries
        print("Generating dataset entries...")
        entries = await generator.generate_entries(
            tools=tools,
            num_entries=5,  # Generate 5 examples
            data_source="syntool_re_call"
        )
        
        if not entries:
            print("Failed to generate entries.")
            return
        
        # Save the dataset
        output_dir = Path("./datasets")
        dataset_name = f"mcp_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        dataset_path = await generator.save_dataset_csv(
            entries=entries,
            output_dir=output_dir,
            dataset_name=dataset_name
        )
        
        print(f"Dataset generated successfully: {dataset_path}")
        
    except Exception as e:
        print(f"Error generating dataset: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Closed all MCP sessions")

if __name__ == "__main__":
    asyncio.run(main())