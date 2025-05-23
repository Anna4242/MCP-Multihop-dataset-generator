# llm_server_simulation.py
"""
LLM-powered server simulation for MCP environment.
Uses OpenAI API to generate realistic server responses.
"""

import json
import openai
from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class ServerContext:
    """Context for maintaining server state across calls."""
    server_type: str
    conversation_history: List[Dict[str, Any]]
    persistent_state: Dict[str, Any]

class LLMServerSimulator:
    """Simulates various servers using LLM responses."""
    
    def __init__(self, client: openai.OpenAI):
        self.client = client
        self.contexts: Dict[str, ServerContext] = {}
        
    def _get_server_prompt(self, server_type: str) -> str:
        """Get system prompt for different server types."""
        
        prompts = {
            "filesystem": """You are simulating a realistic filesystem server. Generate believable file and directory structures.

Rules:
- list_dir: Return realistic directory contents (mix of files and subdirectories)
- read_file: Generate relevant file content based on filename and context
- write_file: Acknowledge file creation and remember the content
- Maintain consistency - if you list a file, it should be readable
- Use realistic filenames and content for the current task context
- Include common files like README, config files, logs, etc.

Format responses as JSON with appropriate structure.""",

            "database": """You are simulating a PostgreSQL database server with realistic data.

Rules:
- list_tables: Return common table names (users, orders, products, etc.)
- run_query: Generate realistic query results based on the SQL
- Maintain data consistency across queries
- Use realistic data types and values
- Include appropriate columns for each table type
- Generate meaningful sample data that relates to common business scenarios

Format responses as JSON with table schemas and data.""",

            "github": """You are simulating a GitHub API server with realistic repository data.

Rules:
- clone_repo: Acknowledge successful clone of realistic repositories
- list_pull_requests: Generate believable PR lists with titles, authors, status
- get_pull_request: Provide detailed PR information with diffs and discussions
- get_commit_diff: Show realistic code changes
- Use real-looking usernames, commit messages, and code changes
- Maintain consistency within repository context

Format responses as JSON matching GitHub API structure.""",

            "search": """You are simulating a web search engine with relevant results.

Rules:
- web_search: Generate realistic search results relevant to the query
- Include diverse sources (articles, documentation, tutorials, forums)
- Provide meaningful titles, URLs, and snippets
- Tailor results to the search context and intent
- Include recent and authoritative sources
- Generate realistic URLs and metadata

Format responses as JSON with search result structure."""
        }
        
        return prompts.get(server_type, "Simulate a generic server.")
    
    def simulate_tool_call(self, server_type: str, tool_name: str, args: Dict[str, Any], context: str = "") -> Dict[str, Any]:
        """Simulate a tool call using LLM."""
        
        # Get or create server context
        if server_type not in self.contexts:
            self.contexts[server_type] = ServerContext(
                server_type=server_type,
                conversation_history=[],
                persistent_state={}
            )
        
        server_context = self.contexts[server_type]
        
        # Build prompt
        system_prompt = self._get_server_prompt(server_type)
        
        # Create user message
        user_message = f"""
Tool Call: {tool_name}
Arguments: {json.dumps(args)}
Context: {context}

Previous calls in this session:
{json.dumps(server_context.conversation_history[-3:], indent=2)}

Generate a realistic response for this {server_type} server tool call. 
Respond with JSON only, no explanation.
"""
        
        try:
            response = self.client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3,  # Some creativity but mostly consistent
                max_tokens=800
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Try to parse JSON response
            if response_text.startswith('{') or response_text.startswith('['):
                result = json.loads(response_text)
            else:
                # Extract JSON from response
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                if start >= 0 and end > start:
                    result = json.loads(response_text[start:end])
                else:
                    result = {"result": response_text}
            
            # Update context
            server_context.conversation_history.append({
                "tool": tool_name,
                "args": args,
                "result": result
            })
            
            return result
            
        except Exception as e:
            return {"error": f"Server simulation failed: {str(e)}"}

# Example usage functions for each server type
def simulate_filesystem_server(simulator: LLMServerSimulator, tool_name: str, args: Dict[str, Any], context: str = "") -> Dict[str, Any]:
    """Simulate filesystem server calls."""
    return simulator.simulate_tool_call("filesystem", tool_name, args, context)

def simulate_database_server(simulator: LLMServerSimulator, tool_name: str, args: Dict[str, Any], context: str = "") -> Dict[str, Any]:
    """Simulate database server calls.""" 
    return simulator.simulate_tool_call("database", tool_name, args, context)

def simulate_github_server(simulator: LLMServerSimulator, tool_name: str, args: Dict[str, Any], context: str = "") -> Dict[str, Any]:
    """Simulate GitHub server calls."""
    return simulator.simulate_tool_call("github", tool_name, args, context)

def simulate_search_server(simulator: LLMServerSimulator, tool_name: str, args: Dict[str, Any], context: str = "") -> Dict[str, Any]:
    """Simulate search server calls."""
    return simulator.simulate_tool_call("search", tool_name, args, context)

# Integration with existing MCP environment
class LLMPoweredToolExecutor:
    """Tool executor that uses LLM simulation instead of real servers."""
    
    def __init__(self, openai_client: openai.OpenAI):
        self.simulator = LLMServerSimulator(openai_client)
        self.available_tools = [
            "list_dir", "read_file", "write_file",  # Filesystem
            "list_tables", "run_query",             # Database
            "clone_repo", "list_pull_requests", "get_pull_request", "get_commit_diff",  # GitHub
            "web_search",                           # Search
            "answer"                                # System
        ]
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tools."""
        return self.available_tools.copy()
    
    def execute_tool(self, tool_name: str, args: Dict[str, Any], context: str = "") -> Dict[str, Any]:
        """Execute a tool using LLM simulation."""
        
        if tool_name == "answer":
            return {"type": "answer", "content": args.get("answer", "")}
        
        # Map tools to server types
        tool_to_server = {
            "list_dir": "filesystem",
            "read_file": "filesystem", 
            "write_file": "filesystem",
            "list_tables": "database",
            "run_query": "database",
            "clone_repo": "github",
            "list_pull_requests": "github",
            "get_pull_request": "github",
            "get_commit_diff": "github",
            "web_search": "search"
        }
        
        server_type = tool_to_server.get(tool_name)
        if not server_type:
            return {"error": f"Unknown tool: {tool_name}"}
        
        return self.simulator.simulate_tool_call(server_type, tool_name, args, context)

# Test the LLM server simulation
def test_llm_server_simulation():
    """Test the LLM server simulation."""
    import openai
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    client = openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
    )
    
    executor = LLMPoweredToolExecutor(client)
    
    print("🧪 Testing LLM Server Simulation")
    print("=" * 40)
    
    # Test filesystem
    print("\n📁 Testing Filesystem Simulation:")
    result = executor.execute_tool("list_dir", {"path": "."}, "Exploring project directory")
    print(f"list_dir result: {json.dumps(result, indent=2)}")
    
    result = executor.execute_tool("read_file", {"filename": "README.md"}, "Reading project documentation")
    print(f"read_file result: {json.dumps(result, indent=2)}")
    
    # Test database
    print("\n🗃️ Testing Database Simulation:")
    result = executor.execute_tool("list_tables", {}, "Exploring database schema")
    print(f"list_tables result: {json.dumps(result, indent=2)}")
    
    result = executor.execute_tool("run_query", {"sql": "SELECT * FROM users LIMIT 3"}, "Getting user data")
    print(f"run_query result: {json.dumps(result, indent=2)}")
    
    # Test GitHub
    print("\n🔧 Testing GitHub Simulation:")
    result = executor.execute_tool("list_pull_requests", {"repo": "microsoft/vscode"}, "Analyzing repository activity")
    print(f"list_pull_requests result: {json.dumps(result, indent=2)}")
    
    # Test Search
    print("\n🔍 Testing Search Simulation:")
    result = executor.execute_tool("web_search", {"query": "MCP Model Context Protocol"}, "Researching MCP")
    print(f"web_search result: {json.dumps(result, indent=2)}")

if __name__ == "__main__":
    test_llm_server_simulation()