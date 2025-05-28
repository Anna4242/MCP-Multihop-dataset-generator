#!/usr/bin/env python3
"""
batch_mcp_pipeline.py
--------------------
Automated batch pipeline that:
1. Uses ALL available servers automatically (no user selection)
2. Generates 10 queries 10 times (100 total queries)
3. Executes all queries
4. Concatenates all outputs into one final dataset

Usage:
    python batch_mcp_pipeline.py [--config CONFIG_PATH] [--api_key API_KEY]
"""

import asyncio
import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import traceback

# Import the dot-env module for environment variables
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_CONFIG_PATH = Path(
    r"D:\one drive\study\ARCEE AI INTERNSHIP\mcp data gen minimal\mcp-dataset-generator\mcp_pipeline\mcp_servers.json"
)
OUTPUT_DIR = Path(__file__).parent / "batch_queries"
RESULTS_DIR = Path(__file__).parent / "batch_results"
FINAL_DATASET_DIR = Path(__file__).parent / "final_dataset"

# Load environment variables
load_dotenv()

# Print environment status
print("\n🔧 Environment Check:")
if os.environ.get("OPENAI_API_KEY"):
    print("✅ OPENAI_API_KEY is set")
else:
    print("❌ OPENAI_API_KEY is not set")

if os.environ.get("MCP_INPUT_github_token"):
    print("✅ MCP_INPUT_github_token is set")
else:
    print("⚠️ MCP_INPUT_github_token is not set")
print()

# Import modules
try:
    from MCPQueryGenerator import MCPQueryGenerator
    from MCPQueryExecutor import MCPQueryExecutor
    modules_imported = True
    print("✅ Successfully imported MCP modules")
except ImportError as e:
    modules_imported = False
    print(f"❌ Module import error: {e}")


class BatchMCPPipeline:
    """
    Automated batch pipeline for generating and executing large datasets of MCP queries.
    """
    
    def __init__(
        self,
        config_path: Path = DEFAULT_CONFIG_PATH,
        queries_per_batch: int = 10,
        num_batches: int = 10,
        model_name: str = "gpt-4o",
        temperature: float = 0,
        max_agent_steps: int = 30,
        api_key: str = None
    ):
        """
        Initialize the batch pipeline.
        
        Args:
            config_path: Path to MCP server configuration
            queries_per_batch: Number of queries per batch (default: 10)
            num_batches: Number of batches to run (default: 10)
            model_name: LLM model to use
            temperature: LLM temperature
            max_agent_steps: Max steps for agent
            api_key: OpenAI API key
        """
        self.config_path = Path(config_path)
        self.queries_per_batch = queries_per_batch
        self.num_batches = num_batches
        self.total_queries = queries_per_batch * num_batches
        self.model_name = model_name
        self.temperature = temperature
        self.max_agent_steps = max_agent_steps
        
        # Get API key
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY or use --api_key")
        
        # Initialize components
        if modules_imported:
            self.generator = MCPQueryGenerator(
                config_path=self.config_path,
                api_key=self.api_key
            )
        else:
            raise ImportError("Required MCP modules not available")
        
        # Create output directories
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        FINAL_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    
    async def generate_all_queries(self) -> List[Path]:
        """
        Generate queries for all batches.
        
        Returns:
            List of paths to generated query files
        """
        print(f"\n🔄 Generating {self.total_queries} queries ({self.num_batches} batches of {self.queries_per_batch})")
        
        query_files = []
        
        for batch_num in range(1, self.num_batches + 1):
            print(f"\n📝 Generating batch {batch_num}/{self.num_batches}...")
            
            try:
                # Generate queries for this batch - using interactive=False for all servers
                queries_path = await self.generator.generate_queries(
                    num_queries=self.queries_per_batch,
                    interactive=False  # This should use all servers automatically
                )
                
                if queries_path and Path(queries_path).exists():
                    # Rename file to include batch number
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    new_name = OUTPUT_DIR / f"batch_{batch_num:02d}_{timestamp}_queries.json"
                    
                    # Move file to batch directory
                    import shutil
                    shutil.move(str(queries_path), str(new_name))
                    
                    query_files.append(new_name)
                    print(f"✅ Batch {batch_num} saved: {new_name.name}")
                else:
                    print(f"❌ Failed to generate batch {batch_num}")
                    
            except Exception as e:
                print(f"❌ Error generating batch {batch_num}: {e}")
                logger.error(f"Batch {batch_num} generation failed: {e}")
                continue
        
        print(f"\n📊 Successfully generated {len(query_files)} batches")
        return query_files
    
    async def execute_all_queries(self, query_files: List[Path]) -> List[Path]:
        """
        Execute all query batches.
        
        Args:
            query_files: List of query file paths
            
        Returns:
            List of result file paths
        """
        print(f"\n🚀 Executing {len(query_files)} query batches...")
        
        result_files = []
        
        for i, query_file in enumerate(query_files, 1):
            print(f"\n⚡ Executing batch {i}/{len(query_files)}: {query_file.name}")
            
            try:
                # Initialize executor for this batch
                executor = MCPQueryExecutor(
                    config_path=self.config_path,
                    queries_path=query_file,
                    save_results=True,
                    output_dir=RESULTS_DIR,
                    model_name=self.model_name,
                    temperature=self.temperature,
                    max_agent_steps=self.max_agent_steps,
                    api_key=self.api_key
                )
                
                # Execute queries
                exit_code = await executor.execute_queries()
                
                if exit_code == 0:
                    # Find the most recent result file
                    result_files_pattern = list(RESULTS_DIR.glob("answers_*.json"))
                    if result_files_pattern:
                        latest_result = max(result_files_pattern, key=lambda p: p.stat().st_mtime)
                        
                        # Rename to include batch info
                        batch_result_name = RESULTS_DIR / f"batch_{i:02d}_{latest_result.name}"
                        latest_result.rename(batch_result_name)
                        
                        result_files.append(batch_result_name)
                        print(f"✅ Batch {i} executed: {batch_result_name.name}")
                    else:
                        print(f"⚠️ Batch {i} executed but no result file found")
                else:
                    print(f"❌ Batch {i} execution failed with exit code {exit_code}")
                
                # Ensure proper cleanup after each batch
                try:
                    await executor.cleanup()
                    print(f"🧹 Cleaned up batch {i} resources")
                except Exception as cleanup_error:
                    print(f"⚠️ Warning: Cleanup for batch {i} had issues: {cleanup_error}")
                    
            except Exception as e:
                print(f"❌ Error executing batch {i}: {e}")
                logger.error(f"Batch {i} execution failed: {e}")
                # Try to cleanup even if there was an error
                try:
                    await executor.cleanup()
                except:
                    pass
                continue
        
        print(f"\n📊 Successfully executed {len(result_files)} batches")
        return result_files
    
    def concatenate_datasets(self, result_files: List[Path]) -> Path:
        """
        Concatenate all result files into one final dataset.
        
        Args:
            result_files: List of result file paths
            
        Returns:
            Path to final concatenated dataset
        """
        print(f"\n🔗 Concatenating {len(result_files)} result files...")
        
        all_results = []
        total_queries = 0
        successful_queries = 0
        failed_queries = 0
        
        # Load and combine all results
        for result_file in result_files:
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    batch_results = json.load(f)
                
                # Add batch info to each result
                batch_name = result_file.stem
                for result in batch_results:
                    result['batch'] = batch_name
                    result['batch_file'] = str(result_file.name)
                    
                    # Count successes/failures
                    total_queries += 1
                    if result.get('error'):
                        failed_queries += 1
                    else:
                        successful_queries += 1
                
                all_results.extend(batch_results)
                print(f"  ➕ Added {len(batch_results)} results from {result_file.name}")
                
            except Exception as e:
                print(f"  ❌ Error reading {result_file.name}: {e}")
                continue
        
        # Create final dataset
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_dataset_path = FINAL_DATASET_DIR / f"mcp_dataset_{timestamp}.json"
        
        # Add metadata
        dataset = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "total_queries": total_queries,
                "successful_queries": successful_queries,
                "failed_queries": failed_queries,
                "success_rate": f"{(successful_queries/total_queries*100):.1f}%" if total_queries > 0 else "0%",
                "batches_processed": len(result_files),
                "queries_per_batch": self.queries_per_batch,
                "model_used": self.model_name,
                "temperature": self.temperature,
                "max_agent_steps": self.max_agent_steps
            },
            "queries": all_results
        }
        
        # Save final dataset
        with open(final_dataset_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        print(f"\n🎉 Final dataset created: {final_dataset_path}")
        print(f"📊 Dataset Statistics:")
        print(f"   Total queries: {total_queries}")
        print(f"   Successful: {successful_queries}")
        print(f"   Failed: {failed_queries}")
        print(f"   Success rate: {(successful_queries/total_queries*100):.1f}%" if total_queries > 0 else "0%")
        
        return final_dataset_path
    
    async def run_full_pipeline(self) -> Path:
        """
        Run the complete pipeline: generate -> execute -> concatenate.
        
        Returns:
            Path to final dataset file
        """
        start_time = datetime.now()
        print(f"\n🚀 Starting Batch MCP Pipeline")
        print(f"⚙️ Configuration:")
        print(f"   Queries per batch: {self.queries_per_batch}")
        print(f"   Number of batches: {self.num_batches}")
        print(f"   Total queries: {self.total_queries}")
        print(f"   Model: {self.model_name}")
        print(f"   Temperature: {self.temperature}")
        print(f"   Max steps: {self.max_agent_steps}")
        
        try:
            # Step 1: Generate all queries
            query_files = await self.generate_all_queries()
            if not query_files:
                raise Exception("No query files were generated successfully")
            
            # Step 2: Execute all queries
            result_files = await self.execute_all_queries(query_files)
            if not result_files:
                raise Exception("No result files were generated successfully")
            
            # Step 3: Concatenate into final dataset
            final_dataset = self.concatenate_datasets(result_files)
            
            # Summary
            end_time = datetime.now()
            duration = end_time - start_time
            print(f"\n✅ Pipeline completed successfully!")
            print(f"⏱️ Total time: {duration}")
            print(f"📁 Final dataset: {final_dataset}")
            
            return final_dataset
            
        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            logger.error(f"Pipeline execution failed: {e}")
            traceback.print_exc()
            raise


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Automated Batch MCP Pipeline")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                        help="Path to MCP server configuration file")
    parser.add_argument("--queries_per_batch", type=int, default=10,
                        help="Number of queries per batch (default: 10)")
    parser.add_argument("--num_batches", type=int, default=10,
                        help="Number of batches to run (default: 10)")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="LLM model to use (default: gpt-4o)")
    parser.add_argument("--temperature", type=float, default=0,
                        help="LLM temperature (default: 0)")
    parser.add_argument("--max_steps", type=int, default=30,
                        help="Maximum agent steps (default: 30)")
    parser.add_argument("--api_key", type=str, default=None,
                        help="OpenAI API key (overrides environment variable)")
    
    args = parser.parse_args()
    
    # Check modules
    if not modules_imported:
        print("\n❌ ERROR: Required MCP modules not available")
        return 1
    
    try:
        # Initialize and run pipeline
        pipeline = BatchMCPPipeline(
            config_path=args.config,
            queries_per_batch=args.queries_per_batch,
            num_batches=args.num_batches,
            model_name=args.model,
            temperature=args.temperature,
            max_agent_steps=args.max_steps,
            api_key=args.api_key
        )
        
        final_dataset = await pipeline.run_full_pipeline()
        print(f"\n🎉 SUCCESS: Dataset available at {final_dataset}")
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline cancelled by user")
        return 130
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    # Run the pipeline
    sys.exit(asyncio.run(main()))