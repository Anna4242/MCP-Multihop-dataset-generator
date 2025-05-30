import asyncio
import os
import json
import sys
from io import StringIO
from contextlib import redirect_stdout
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from mcp_use import MCPAgent, MCPClient

class ExecutionTracker:
    """Tracks execution using stdout capture."""
    
    def __init__(self, output_dir="execution_logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logs = []
        
        # Track tool usage patterns
        self.tool_sequence = []
    
    def add_log(self, output):
        """Add captured output to logs."""
        self.logs.append({
            "timestamp": datetime.now().isoformat(),
            "output": output
        })
        
        # Try to extract tool calls from the output
        for line in output.split('\n'):
            if "Used tool '" in line:
                # Extract tool name
                parts = line.split("'")
                if len(parts) >= 2:
                    tool_name = parts[1]
                    self.tool_sequence.append(tool_name)
    
    def save_to_file(self, query, result):
        """Save logs to a file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"multihop_{timestamp}.json"
        
        data = {
            "query": query,
            "result": result,
            "tool_sequence": self.tool_sequence,
            "logs": self.logs
        }
        
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
        
        # Also save a summary file with just the query and tool sequence for easier analysis
        summary_file = self.output_dir / f"summary_{timestamp}.txt"
        with open(summary_file, "w") as f:
            f.write(f"Query: {query}\n\n")
            f.write("Tool Sequence:\n")
            for i, tool in enumerate(self.tool_sequence, 1):
                f.write(f"{i}. {tool}\n")
            f.write("\nResult:\n")
            f.write(result)
        
        return output_file, summary_file

async def main():
    # Load environment variables
    load_dotenv()
    
    # Create configuration dictionary
    config = {
      "mcpServers": {
        "playwright": {
          "command": "npx",
          "args": ["@playwright/mcp@latest"],
          "env": {
            "DISPLAY": ":1"
          }
        }
      }
    }
    
    # Create tracker
    tracker = ExecutionTracker()
    
    # Create MCPClient from configuration dictionary
    client = MCPClient.from_dict(config)
    
    # Create LLM
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.2
    )
    
    # Create agent with the client and verbose output enabled
    agent = MCPAgent(llm=llm, client=client, max_steps=30, verbose=True)
    
    # Choose a multi-hop query from the examples
    queries = [
        
        "Find the most cited research paper on artificial intelligence published in 2023 according to Google Scholar, then check if it's available for free download on arXiv.",
       
    ]
    
    # Select a query (change the index to try different queries)
    query_index = 0  # Change this to select a different query
    query = queries[query_index]
    
    try:
        print(f"\nExecuting multi-hop query: {query}\n")
        
        # Capture stdout while running the query
        # But also print to the console in real-time
        original_stdout = sys.stdout
        
        class TeeIO(StringIO):
            def write(self, s):
                original_stdout.write(s)
                return super().write(s)
        
        buffer = TeeIO()
        sys.stdout = buffer
        
        try:
            result = await agent.run(query)
        finally:
            sys.stdout = original_stdout
        
        # Get the captured output
        captured_output = buffer.getvalue()
        tracker.add_log(captured_output)
        
        # Save to files
        output_file, summary_file = tracker.save_to_file(query, result)
        
        print("\n" + "=" * 70)
        print(f"Multi-hop query execution completed")
        print(f"Full log saved to: {output_file}")
        print(f"Summary saved to: {summary_file}")
        print("=" * 70)
        
        print(f"\nTool Sequence:")
        for i, tool in enumerate(tracker.tool_sequence, 1):
            print(f"{i}. {tool}")
        
        print("\n" + "=" * 70)
        print(f"Result: {result}")
        
    finally:
        # Close client session
        if hasattr(client, 'close_all_sessions'):
            await client.close_all_sessions()

if __name__ == "__main__":
    asyncio.run(main())