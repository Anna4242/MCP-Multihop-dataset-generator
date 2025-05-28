#!/usr/bin/env python3
"""
mcp_pipeline.py
--------------
Pipeline script that connects query generation to execution using the class-based implementations.
1. Generates cross-server multi-hop queries using MCPQueryGenerator
2. Uses the generated queries as input for MCPQueryExecutor
3. Provides a unified interface for the entire workflow

Usage:
    python mcp_pipeline.py [--config CONFIG_PATH] [--num_queries NUM] [--execute]
"""

import asyncio
import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, List

# Import the dot-env module for environment variables
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default paths for configuration and output directories
DEFAULT_CONFIG_PATH = Path(
    r"D:\one drive\study\ARCEE AI INTERNSHIP\mcp data gen minimal\mcp-dataset-generator\mcp_pipeline\mcp_servers.json"
)
OUTPUT_DIR = Path("queries")
RESULTS_DIR = Path("execution_results")

# Load environment variables at the module level to ensure availability
load_dotenv()

# Print environment variable status for debugging
print("\nEnvironment variables status:")
if os.environ.get("OPENAI_API_KEY"):
    print("✅ OPENAI_API_KEY is set")
else:
    print("❌ OPENAI_API_KEY is not set")

if os.environ.get("MCP_INPUT_github_token"):
    print("✅ MCP_INPUT_github_token is set")
else:
    print("⚠️ MCP_INPUT_github_token is not set (might be needed for some GitHub operations)")
print()

# Import refactored generator and executor classes
try:
    # Try importing with different naming conventions
    try:
        from MCPQueryGenerator import MCPQueryGenerator
        from MCPQueryExecutor import MCPQueryExecutor
        modules_imported = True
        print("Successfully imported modules using MCPQueryGenerator/MCPQueryExecutor")
    except ImportError:
        # Try alternative names
        from MCPQueryGenerator import  MCPQueryGenerator
        from MCPQueryExecutor import MCPQueryExecutor
        modules_imported = True
        print("Successfully imported modules using mcp_query_generator/mcp_query_executor")
except ImportError:
    modules_imported = False
    logger.warning("Could not import generator or executor modules.")
    logger.warning("Make sure MCPQueryGenerator.py and MCPQueryExecutor.py (or mcp_query_generator.py and mcp_query_executor.py) are in your Python path.")
    print("⚠️ Module import error: Check that the generator and executor class files are in the correct location")


class MCPPipeline:
    """
    Pipeline class to coordinate query generation and execution.
    """
    
    def __init__(
        self,
        config_path: Union[str, Path] = DEFAULT_CONFIG_PATH,
        num_queries: int = 5,
        execute: bool = True,
        interactive: bool = True,
        input_queries: Optional[Path] = None,
        save_results: bool = True,
        model_name: str = "gpt-4o",
        temperature: float = 0,
        max_agent_steps: int = 10,
        api_key: Optional[str] = None
    ):
        """
        Initialize the MCP pipeline.
        
        Args:
            config_path: Path to MCP server configuration file
            num_queries: Number of queries to generate
            execute: Whether to execute queries after generation
            interactive: Whether to select servers interactively
            input_queries: Path to existing queries file (skips generation)
            save_results: Whether to save execution results
            model_name: LLM model to use for execution
            temperature: LLM temperature setting
            max_agent_steps: Maximum steps for the agent
            api_key: OpenAI API key (overrides environment variable)
        """
        self.config_path = Path(config_path)
        self.num_queries = num_queries
        self.execute = execute
        self.interactive = interactive
        self.input_queries = Path(input_queries) if input_queries else None
        self.save_results = save_results
        self.model_name = model_name
        self.temperature = temperature
        self.max_agent_steps = max_agent_steps
        
        # Get API key from parameters or environment
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
        # Initialize modules if available
        if modules_imported:
            if self.api_key:
                self.generator = MCPQueryGenerator(
                    config_path=self.config_path,
                    api_key=self.api_key
                )
                self.executor = None  # Will initialize later when needed
            else:
                self.generator = None
                self.executor = None
    
    async def run(self) -> int:
        """
        Run the pipeline.
        
        Returns:
            Exit code (0 for success, non-zero for errors)
        """
        try:
            # Check if OpenAI API key is set
            if not self.api_key:
                logger.error("OpenAI API key not found (neither passed as parameter nor in environment)")
                print("❌ Error: OpenAI API key not found. Please set OPENAI_API_KEY environment variable or pass --api_key")
                return 1
            
            # Check if GitHub token is set for MCP
            github_token = os.environ.get("MCP_INPUT_github_token")
            if not github_token:
                logger.warning("GitHub token (MCP_INPUT_github_token) not found in environment")
                print("⚠️ Warning: GitHub token not found in environment variables")
            
            # Step 1: Generate queries or use provided input file
            queries_path = self.input_queries
            if not queries_path:
                if not modules_imported or not self.generator:
                    logger.error("Cannot generate queries: modules not imported or generator not initialized")
                    return 1
                    
                print(f"🚀 Generating {self.num_queries} cross-server multi-hop queries...")
                queries_path = await self.generator.generate_queries(
                    num_queries=self.num_queries,
                    interactive=self.interactive
                )
            
            if not queries_path or not Path(queries_path).exists():
                logger.error("No queries file available. Exiting.")
                return 1
            
            print(f"Using queries from: {queries_path}")
            
            # Step 2: Execute queries if requested
            if self.execute:
                if not modules_imported:
                    logger.error("Cannot execute queries: modules not imported")
                    return 1
                
                print(f"🔍 Executing queries from {queries_path}")
                
                try:
                    # Initialize executor
                    print("Initializing executor...")
                    # Check if MCPQueryExecutor accepts api_key parameter
                    try:
                        import inspect
                        executor_params = inspect.signature(MCPQueryExecutor.__init__).parameters
                        accepts_api_key = 'api_key' in executor_params
                    except:
                        accepts_api_key = False
                    
                    if accepts_api_key:
                        self.executor = MCPQueryExecutor(
                            config_path=self.config_path,
                            queries_path=queries_path,
                            save_results=self.save_results,
                            output_dir=RESULTS_DIR,
                            model_name=self.model_name,
                            temperature=self.temperature,
                            max_agent_steps=self.max_agent_steps,
                            api_key=self.api_key
                        )
                    else:
                        # Initialize without api_key parameter
                        print("MCPQueryExecutor doesn't accept api_key parameter, initializing without it")
                        self.executor = MCPQueryExecutor(
                            config_path=self.config_path,
                            queries_path=queries_path,
                            save_results=self.save_results,
                            output_dir=RESULTS_DIR,
                            model_name=self.model_name,
                            temperature=self.temperature,
                            max_agent_steps=self.max_agent_steps
                        )
                    
                    # Execute queries
                    print("Starting query execution...")
                    
                    # Fall back to direct execution if the executor module isn't working correctly
                    if not hasattr(self.executor, 'execute_queries'):
                        print("Warning: Using fallback executor implementation")
                        # Direct execution fallback
                        try:
                            from langchain_openai import ChatOpenAI
                            from mcp_use import MCPAgent, MCPClient
                            
                            # Load queries
                            with open(queries_path, 'r', encoding='utf-8') as f:
                                queries_data = json.load(f)
                            
                            # Set up client
                            print("Initializing MCP client...")
                            client = MCPClient.from_config_file(str(self.config_path))
                            
                            # Set up agent
                            print("Building LLM agent...")
                            llm = ChatOpenAI(model=self.model_name, temperature=self.temperature, api_key=self.api_key)
                            agent = MCPAgent(llm=llm, client=client, max_steps=self.max_agent_steps)
                            
                            # Process each query
                            results = []
                            print(f"Processing {len(queries_data)} queries...")
                            for idx, q in enumerate(queries_data, 1):
                                prompt = q.get("query") or q.get("question") or ""
                                tag = q.get("id", f"query_{idx}")
                                
                                print(f"\nProcessing query {idx}/{len(queries_data)}: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
                                
                                rec = {"id": tag, "prompt": prompt, "answer": None, "error": None}
                                try:
                                    answer = await asyncio.wait_for(agent.run(prompt), timeout=120)
                                    rec["answer"] = answer
                                    print(f"   → {answer[:120]}{'...' if len(answer) > 120 else ''}")
                                except Exception as exc:
                                    rec["error"] = str(exc)
                                    print(f"   ✗ ERROR: {exc}")
                                
                                results.append(rec)
                            
                            # Save results
                            if results:
                                RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                                out_file = RESULTS_DIR / f"answers_{ts}.json"
                                out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
                                print(f"\n💾 Results saved to {out_file.resolve()}")
                            
                            # Clean up
                            if hasattr(client, 'close'):
                                await client.close()
                                
                            return 0
                            
                        except Exception as fallback_error:
                            logger.error(f"Fallback execution failed: {fallback_error}")
                            import traceback
                            traceback.print_exc()
                            return 1
                    
                    # Regular execution path
                    exit_code = await self.executor.execute_queries()
                    print(f"Execution completed with exit code: {exit_code}")
                    return exit_code
                except Exception as exec_error:
                    logger.error(f"Error during execution: {exec_error}")
                    import traceback
                    traceback.print_exc()
                    return 1
            else:
                logger.info(f"Queries generated at: {queries_path}")
                logger.info("Execution skipped as requested.")
                return 0
        
        except KeyboardInterrupt:
            logger.warning("Operation cancelled by user")
            return 130
        except Exception as e:
            logger.critical(f"Unhandled exception: {e}")
            import traceback
            traceback.print_exc()
            return 1


async def run_pipeline(args) -> int:
    """Run the pipeline with command-line arguments."""
    pipeline = MCPPipeline(
        config_path=args.config,
        num_queries=args.num_queries,
        execute=args.execute,
        interactive=not args.non_interactive,
        input_queries=args.input_queries,
        save_results=not args.no_save,
        model_name=args.model,
        temperature=args.temperature,
        max_agent_steps=args.max_steps,
        api_key=args.api_key
    )
    
    return await pipeline.run()


def main():
    """Command-line entry point"""
    parser = argparse.ArgumentParser(description="MCP Query Generation and Execution Pipeline")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                        help="Path to MCP server configuration file")
    parser.add_argument("--num_queries", type=int, default=5,
                        help="Number of queries to generate")
    parser.add_argument("--execute", action="store_true",
                        help="Execute queries after generation")
    parser.add_argument("--non-interactive", action="store_true",
                        help="Run in non-interactive mode (use all servers)")
    parser.add_argument("--input_queries", type=Path, default=None,
                        help="Path to existing queries file (skips generation)")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not save execution results")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="LLM model to use (default: gpt-4o)")
    parser.add_argument("--temperature", type=float, default=0,
                        help="LLM temperature (default: 0)")
    parser.add_argument("--max_steps", type=int, default=30,
                        help="Maximum agent steps (default: 30)")
    parser.add_argument("--api_key", type=str, default=None,
                        help="OpenAI API key (overrides environment variable)")
    
    args = parser.parse_args()
    
    # Print out the arguments for debugging
    print("\nCommand-line arguments:")
    print(f"  execute: {args.execute}")
    print(f"  input_queries: {args.input_queries}")
    print(f"  config: {args.config}")
    print(f"  model: {args.model}")
    print()
    
    # Force execution if user explicitly wants it
    if "--execute" in sys.argv and not args.execute:
        print("⚠️ Warning: --execute flag detected in command line but not parsed correctly")
        print("Forcing execution to true")
        args.execute = True
    
    # Create new event loop (avoid deprecation warning)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        exit_code = loop.run_until_complete(run_pipeline(args))
        
        # Clean up
        pending = asyncio.all_tasks(loop)
        if pending:
            print(f"Cancelling {len(pending)} pending tasks...")
            for task in pending:
                task.cancel()
            
            # Allow cancelled tasks to complete with a timeout
            # Remove the 'loop' parameter which is deprecated in newer Python versions
            loop.run_until_complete(
                asyncio.wait(pending, timeout=10)
            )
        
        return exit_code
    finally:
        loop.close()


if __name__ == "__main__":
    # First check for module imports to give clear error message
    if not modules_imported:
        print("\n❌ ERROR: Could not import required modules.")
        print("This script requires MCPQueryGenerator.py and MCPQueryExecutor.py")
        print("Make sure they are in your Python path or current directory.")
        print("\nPlease check:")
        print("1. That you've saved both class files in the correct location")
        print("2. That the file names match exactly (case-sensitive)")
        print("3. That the class names inside match what the pipeline is trying to import\n")
        sys.exit(1)
    
    # Check if key environment variables are set
    if "OPENAI_API_KEY" not in os.environ and "--api_key" not in sys.argv:
        print("\n⚠️ WARNING: OPENAI_API_KEY environment variable not set")
        print("You will need to provide an API key using the --api_key parameter\n")
    
    # Automatic execute mode if using an existing file
    if any("--input_queries" in arg for arg in sys.argv) and not any("--execute" in arg for arg in sys.argv):
        print("\n⚠️ Note: You provided input_queries but didn't specify --execute.")
        choice = input("Would you like to execute these queries? (y/n): ")
        if choice.lower() in ('y', 'yes'):
            sys.argv.append("--execute")
            print("Added --execute flag to command line arguments")
        
    # Run the pipeline
    sys.exit(main())