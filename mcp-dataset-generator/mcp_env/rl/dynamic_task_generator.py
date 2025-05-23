# mcp_env/rl/dynamic_task_generator.py

import random
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
from datetime import datetime

try:
    import openai
    from dotenv import load_dotenv
    # Try to load environment variables
    load_dotenv()
except ImportError:
    print("Warning: openai or dotenv modules not available. LLM task generation will be disabled.")
    openai = None

# Handle both direct execution and module import
try:
    from .mcp_tool_executor import DynamicMCPToolExecutor
except ImportError:
    try:
        # For direct script execution
        import sys
        from pathlib import Path
        
        # Add the parent directory to the path
        current_dir = Path(__file__).resolve().parent
        parent_dir = current_dir.parent.parent
        sys.path.append(str(parent_dir))
        
        from mcp_env.rl.mcp_tool_executor import DynamicMCPToolExecutor
    except ImportError:
        print("Error: Cannot import DynamicMCPToolExecutor. Make sure you're in the correct directory.")
        DynamicMCPToolExecutor = None  # Set to None for testing

class MultiHopVerifier:
    """
    Verifier that analyzes tasks to determine if they require multi-hop reasoning.
    Adapted from the query generator to work with tasks.
    """
    
    def __init__(self, cross_server_mode=True):
        # Set cross-server mode - more lenient verification for cross-server tasks
        self.cross_server_mode = cross_server_mode
        
        # Words indicating sequential operations
        self.sequence_indicators = {
            "then", "after", "next", "followed by", "subsequently", 
            "following", "once", "first", "second", "third", "finally",
            "lastly", "before", "prior to", "earlier"
        }
        
        # Words indicating multiple parts or aspects
        self.conjunction_indicators = {
            "and", "also", "additionally", "moreover", "furthermore", 
            "besides", "plus", "along with", "as well as", "with"
        }
        
        # Words indicating comparison or relationship
        self.comparison_indicators = {
            "compare", "comparison", "versus", "vs", "difference", 
            "similarities", "better", "worse", "more", "less"
        }
        
        # Information-gathering action verbs
        self.info_gathering_verbs = {
            "find", "search", "locate", "identify", "discover",
            "check", "determine", "calculate", "analyze", "look up",
            "show", "list", "display", "tell", "give"
        }
        
        # Action verbs that suggest tool operations
        self.action_verbs = {
            "get", "retrieve", "fetch", "obtain", "extract",
            "download", "open", "navigate", "visit", "browse",
            "search", "read", "write", "save", "edit", "modify",
            "create", "delete", "remove", "update", "calculate"
        }
    
    def is_multi_hop_task(self, task: str) -> Tuple[bool, str, List[str]]:
        """
        Determine if a task requires multi-hop reasoning by analyzing its structure.
        
        Args:
            task: The task text to analyze
            
        Returns:
            Tuple of (is_multi_hop, explanation, potential_sub_tasks)
        """
        # Preprocess task
        task = task.strip()
        task_lower = task.lower()
        
        # Check for obvious patterns
        has_sequence_indicators = self._contains_words_from_set(task_lower, self.sequence_indicators)
        has_multiple_questions = task.count("?") > 1
        has_multiple_conjunctions = len(re.findall(r'\band\b|\balso\b|\bwith\b', task_lower)) > 0
        has_comparison = self._contains_words_from_set(task_lower, self.comparison_indicators)
        
        # Check for multiple action verbs
        action_verbs_present = self._find_words_from_set(task_lower, self.action_verbs)
        info_verbs_present = self._find_words_from_set(task_lower, self.info_gathering_verbs)
        all_verbs_present = action_verbs_present.union(info_verbs_present)
        has_multiple_verbs = len(all_verbs_present) >= 2
        
        # Check for sentences that could be independent sub-tasks
        sentences = self._split_into_sentences(task)
        has_multiple_sentences = len(sentences) > 1
        
        # Attempt to extract potential sub-tasks
        potential_sub_tasks = self._extract_potential_sub_tasks(task)
        has_extractable_sub_tasks = len(potential_sub_tasks) > 1
        
        # Check for specific multi-hop patterns
        has_specific_location = any(phrase in task_lower for phrase in ["specific", "from a", "in a", "in the"])
        has_navigation_pattern = ("repository" in task_lower or "repo" in task_lower or "github" in task_lower) and \
                               ("file" in task_lower or "readme" in task_lower or "content" in task_lower)
        
        # Special checks for cross-server mode
        if self.cross_server_mode:
            # Look for patterns that suggest cross-server operations
            if self._has_cross_server_indicators(task_lower):
                has_multiple_verbs = True  # Consider it as having multiple operations
            
            # File operations + web operations suggest multi-hop
            file_ops = self._contains_words_from_set(task_lower, {"save", "download", "file", "document", "folder", "upload"})
            web_ops = self._contains_words_from_set(task_lower, {"search", "find", "browse", "web", "internet", "online"})
            
            if file_ops and web_ops:
                has_multiple_verbs = True
        
        # Decision logic for explicit linguistic markers
        explicit_factors = []
        if has_sequence_indicators:
            explicit_factors.append("contains sequencing words indicating ordered steps")
        if has_multiple_questions:
            explicit_factors.append("contains multiple questions that require separate answers")
        if has_multiple_conjunctions and has_multiple_verbs:
            explicit_factors.append("contains multiple conjunctions connecting different requirements")
        if has_comparison:
            explicit_factors.append("involves comparison which typically requires gathering multiple pieces of information")
        if has_multiple_verbs and has_extractable_sub_tasks:
            explicit_factors.append(f"contains action verbs ({', '.join(list(all_verbs_present)[:3])}) suggesting different operations")
        if has_multiple_sentences and has_extractable_sub_tasks:
            explicit_factors.append("can be broken down into distinct sub-tasks")
        if has_navigation_pattern:
            explicit_factors.append("requires navigating to a specific location before accessing content")
        if has_specific_location and ("file" in task_lower or "content" in task_lower):
            explicit_factors.append("requires finding a specific file or content within a location")
        
        # Combine factors
        factors = explicit_factors
        
        # Make final determination
        is_multi_hop = (has_sequence_indicators or 
                        has_multiple_questions or 
                        (has_multiple_conjunctions and has_multiple_verbs) or 
                        (has_multiple_verbs and has_extractable_sub_tasks) or
                        has_navigation_pattern or
                        (has_specific_location and ("file" in task_lower or "content" in task_lower)))
        
        # Generate explanation
        if is_multi_hop:
            if factors:
                explanation = "This appears to be a multi-hop task because it " + "; it ".join(factors) + "."
            else:
                explanation = "This task likely requires multiple steps to complete, based on its complexity."
        else:
            explanation = "This appears to be a single-hop task as it doesn't show clear indications of requiring sequential steps."
        
        # Generate sub-tasks if not already extracted
        if is_multi_hop and not has_extractable_sub_tasks:
            potential_sub_tasks = self._generate_implicit_sub_tasks(task, all_verbs_present)
        
        return is_multi_hop, explanation, potential_sub_tasks
    
    def _has_cross_server_indicators(self, text: str) -> bool:
        """Check for phrases that suggest operations across different server types."""
        indicators = {
            # File + Web combinations
            "save the results", "download and save", "find and save", "search and download",
            "save to my", "email the results", "add to my", "save it to",
            
            # Calendar + Web combinations
            "schedule a", "book a", "add to calendar", "check my calendar", 
            
            # Generic cross-domain indicators
            "based on my", "using my", "from my", "to my", "in my",
            
            # GitHub/Repository specific indicators
            "github repository", "from github", "in the repository", "repo",
            "readme file", "readme.md", "specific file", "from a specific",
            "latest content", "recent updates", "latest updates", "recent changes"
        }
        
        return self._contains_words_from_set(text, indicators)
    
    def _contains_words_from_set(self, text: str, word_set: Set[str]) -> bool:
        """Check if text contains any words from the provided set."""
        # Ensure we're matching whole words
        for word in word_set:
            pattern = r'\b' + re.escape(word) + r'\b'
            if re.search(pattern, text):
                return True
        return False
    
    def _find_words_from_set(self, text: str, word_set: Set[str]) -> Set[str]:
        """Find all words from the set that appear in the text."""
        found_words = set()
        for word in word_set:
            pattern = r'\b' + re.escape(word) + r'\b'
            if re.search(pattern, text):
                found_words.add(word)
        return found_words
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting - can be improved with NLP libraries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _extract_potential_sub_tasks(self, task: str) -> List[str]:
        """
        Attempt to break down a task into potential sub-tasks.
        This is a heuristic approach that looks for common patterns.
        """
        sub_tasks = []
        
        # Method 1: Split by sequence indicators
        for indicator in ["then", "after that", "next", "followed by", "finally"]:
            if indicator in task.lower():
                parts = re.split(rf'\b{re.escape(indicator)}\b', task, flags=re.IGNORECASE)
                if len(parts) > 1:
                    # Clean up and add parts
                    for i, part in enumerate(parts):
                        part = part.strip()
                        if part:
                            if i > 0:  # This is a subsequent step
                                sub_tasks.append(f"{indicator} {part}")
                            else:  # This is the first step
                                sub_tasks.append(part)
                    return sub_tasks  # Return if we found a clear sequence
        
        # Method 2: Split by "and" if it seems to connect actions
        and_parts = re.split(r'\band\b', task, flags=re.IGNORECASE)
        if len(and_parts) > 1:
            # Check if these parts look like separate actions
            valid_parts = []
            for part in and_parts:
                part = part.strip()
                if part and any(verb in part.lower() for verb in self.action_verbs.union(self.info_gathering_verbs)):
                    valid_parts.append(part)
            
            if len(valid_parts) > 1:
                return valid_parts
        
        # Method 3: Split by semicolons and commas (if they appear to separate actions)
        if ";" in task:
            semicolon_parts = [p.strip() for p in task.split(";") if p.strip()]
            if len(semicolon_parts) > 1:
                return semicolon_parts
        
        # If no clear sub-tasks found, return original as single task
        if not sub_tasks:
            sub_tasks = [task]
        
        return sub_tasks
    
    def _generate_implicit_sub_tasks(self, task: str, verbs_present: Set[str] = None) -> List[str]:
        """
        Generate potential sub-tasks for implicit multi-hop tasks.
        Uses verb detection to create a logical sequence.
        """
        if not verbs_present:
            verbs_present = set()
            
        # Analyze what types of operations are needed
        needs_search = any(v in task.lower() for v in ["search", "find", "look", "locate"])
        needs_reading = any(v in task.lower() for v in ["read", "review", "analyze", "examine"])
        needs_writing = any(v in task.lower() for v in ["write", "create", "save", "generate"])
        needs_comparison = any(v in task.lower() for v in ["compare", "contrast", "versus", "better"])
        
        # Build appropriate sub-tasks based on needed operations
        sub_tasks = []
        
        if needs_search:
            sub_tasks.append("Search for the required information")
            
        if needs_reading:
            if needs_search:
                sub_tasks.append("Read and analyze the search results")
            else:
                sub_tasks.append("Read and analyze the relevant documents")
                
        if needs_comparison:
            sub_tasks.append("Compare the different options or information")
            
        if needs_writing:
            sub_tasks.append("Create or save the final output")
            
        # If we couldn't determine specific sub-tasks, use a generic template
        if not sub_tasks:
            sub_tasks = [
                "Gather the necessary information",
                "Process the information as needed",
                "Present the results in the requested format"
            ]
            
        return sub_tasks


class DynamicTaskGenerator:
    """Generates tasks dynamically based on available tools."""
    
    def __init__(self, tool_executor: DynamicMCPToolExecutor):
        self.tool_executor = tool_executor
        self.verifier = MultiHopVerifier()
        
        # Initialize LLM support if possible
        self.use_llm = openai is not None
        self.client = None
        
        if self.use_llm:
            try:
                api_key = os.environ.get("OPENAI_API_KEY")
                api_base = os.environ.get("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
                
                if api_key:
                    self.client = openai.OpenAI(
                        api_key=api_key,
                        base_url=api_base
                    )
                    print("✅ LLM-based task generation is enabled")
                else:
                    print("⚠️ OpenAI API key not found. LLM task generation will be disabled.")
                    self.use_llm = False
            except Exception as e:
                print(f"⚠️ Failed to initialize OpenAI client: {e}")
                self.use_llm = False
    
    def generate_task(self, task_type: str = None, multi_hop: bool = False) -> str:
        """
        Generate a task using LLM or return a fallback message.
        
        Args:
            task_type: Type of task to generate (ignored in this version)
            multi_hop: Whether to generate a multi-hop task
            
        Returns:
            A task string
        """
        if self.use_llm:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                tasks = loop.run_until_complete(self.generate_llm_tasks(num_tasks=1, multi_hop=multi_hop))
                if tasks:
                    return tasks[0].get("task", "Use the available tools to complete a task")
            except Exception as e:
                print(f"Error generating task: {e}")
        
        # Fallback if LLM not available
        return "Use the available tools to complete a task"
    
    def get_available_task_types(self) -> List[str]:
        """Get available task types based on available servers."""
        # Return server names as task types
        servers = list(self.tool_executor.get_tools_by_server().keys())
        if 'system' in servers:
            servers.remove('system')
        
        # Add generic types
        task_types = ["general", "multi_hop"] + servers
        return task_types
    
    async def generate_subtasks_with_llm(self, task: str) -> List[str]:
        """
        Generate subtasks for a multi-hop task using LLM.
        
        Args:
            task: The multi-hop task to break down
            
        Returns:
            List of subtasks
        """
        if not self.use_llm or not self.client:
            print("⚠️ LLM subtask generation is not available.")
            return []
        
        # Create a prompt for subtask generation
        prompt = f"""
Break down the following task into specific, sequential subtasks that an AI assistant would need to perform:

Task: "{task}"

Requirements:
1. Each subtask should be a concrete action that can be performed with a single tool
2. Subtasks should be ordered logically
3. Be specific about what needs to be done in each step
4. Consider navigation, searching, reading, and processing steps

Return the subtasks as a JSON array of strings, for example:
["First subtask", "Second subtask", "Third subtask"]
"""
        
        try:
            response = self.client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at breaking down complex tasks into simple, sequential steps."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            content = response.choices[0].message.content
            
            # Extract JSON array from response
            start_idx = content.find("[")
            end_idx = content.rfind("]") + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                subtasks = json.loads(json_str)
                return subtasks
            else:
                print("⚠️ No JSON array found in LLM response for subtasks.")
                return []
                
        except Exception as e:
            print(f"⚠️ Error generating subtasks with LLM: {e}")
            return []
    
    async def generate_llm_tasks(self, num_tasks: int = 5, multi_hop: bool = True) -> List[Dict]:
        """
        Generate tasks using LLM based on available tools.
        
        Args:
            num_tasks: Number of tasks to generate
            multi_hop: Whether to generate multi-hop tasks
            
        Returns:
            List of generated tasks with metadata
        """
        if not self.use_llm or not self.client:
            print("⚠️ LLM task generation is not available.")
            return []
        
        # Create a list of all available tools with their descriptions
        tools_with_servers = []
        for server_name, tools in self.tool_executor.get_tools_by_server().items():
            if server_name == "system":
                continue
                
            for tool_name in tools:
                tool_info = self.tool_executor.get_tool_info(tool_name)
                if tool_info:
                    tools_with_servers.append({
                        "server": server_name,
                        "name": tool_name,
                        "description": tool_info.description
                    })
        
        # Group tools by server for the prompt
        server_tool_descriptions = {}
        for tool in tools_with_servers:
            server = tool["server"]
            if server not in server_tool_descriptions:
                server_tool_descriptions[server] = []
            server_tool_descriptions[server].append(f"- {tool['name']}: {tool['description']}")
        
        # Create the system prompt
        system_prompt = f"""
You are an expert task generator creating {'MULTI-HOP' if multi_hop else 'SINGLE-HOP'} tasks for an AI assistant that uses tools from MCP (Model Context Protocol) servers.

{'## MULTI-HOP TASK REQUIREMENTS:' if multi_hop else '## SINGLE-HOP TASK REQUIREMENTS:'}
{'''
1. Each task MUST require MULTIPLE SEQUENTIAL TOOL CALLS to complete
2. Tasks should naturally flow between different tools in a logical sequence
3. The need to use multiple tools should be IMPLICIT in the task, not explicitly stated
4. A good multi-hop task will feel like a single cohesive task but require sequential operations
''' if multi_hop else '''
1. Each task should be completable with a SINGLE TOOL CALL
2. Tasks should be focused on one specific operation
3. The task should clearly indicate which type of tool is needed
4. Avoid tasks that would require gathering and processing information across multiple steps
'''}

## AVAILABLE TOOLS BY SERVER:
{os.linesep.join([f"SERVER: {server}\\n" + os.linesep.join(descriptions) for server, descriptions in server_tool_descriptions.items()])}

Generate {num_tasks} diverse, realistic tasks that a user might ask an AI assistant to help with.
Each task should be clear, specific, and accomplishable using the available tools.

Return the tasks as a JSON array with this format:
[
  {{
    "task": "The {'multi-hop' if multi_hop else 'single-hop'} task for the AI assistant",
    "expected_tools": ["tool1", "tool2", ...],
    "expected_servers": ["server1", "server2", ...],
    "tool_sequence": "{'Explanation of the sequence of tools needed' if multi_hop else 'The specific tool needed and why'}",
    "rationale": "Explanation of why this task {'requires multiple steps' if multi_hop else 'is focused on a single operation'}"
  }}
]
"""
        
        try:
            # Call the OpenAI API
            response = self.client.chat.completions.create(
                model="openai/gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Generate {num_tasks} diverse {'multi-hop' if multi_hop else 'single-hop'} tasks based on the available tools."}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            
            # Extract JSON from the response
            start_idx = content.find("[")
            end_idx = content.rfind("]") + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                generated_tasks = json.loads(json_str)
                
                # Verify each task with the verifier
                verified_tasks = []
                for i, entry in enumerate(generated_tasks):
                    task = entry.get("task", "")
                    is_multi_hop, explanation, sub_tasks = self.verifier.is_multi_hop_task(task)
                    
                    # If multi-hop and no clear subtasks, generate them with LLM
                    if is_multi_hop and len(sub_tasks) <= 1:
                        llm_subtasks = await self.generate_subtasks_with_llm(task)
                        if llm_subtasks:
                            sub_tasks = llm_subtasks
                    
                    # Add verification information
                    entry["verification"] = {
                        "is_multi_hop": is_multi_hop,
                        "explanation": explanation,
                        "sub_tasks": sub_tasks
                    }
                    
                    # Add ID
                    entry["id"] = f"llm_task_{datetime.now().strftime('%Y%m%d%H%M%S')}_{i}"
                    
                    verified_tasks.append(entry)
                
                return verified_tasks
            else:
                print(f"⚠️ No JSON array found in LLM response.")
                return []
        
        except Exception as e:
            print(f"⚠️ Error generating tasks with LLM: {e}")
            return []
    
    def save_tasks_to_file(self, tasks: List[Dict], output_dir: str = "tasks") -> Optional[str]:
        """Save generated tasks to a JSON file."""
        if not tasks:
            print("No tasks to save")
            return None
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create output file path with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_path / f"generated_tasks_{timestamp}.json"
        
        # Write tasks to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(tasks, f, indent=2)
        
        print(f"Saved {len(tasks)} tasks to {output_file}")
        return str(output_file)


# Test function
def test_dynamic_task_generator():
    """Test the enhanced dynamic task generator."""
    from mcp_env.rl.mcp_tool_executor import DynamicMCPToolExecutor
    
    print("🧪 TESTING ENHANCED TASK GENERATOR")
    print("=" * 50)
    
    # Create tool executor
    tool_executor = DynamicMCPToolExecutor()
    
    # Create task generator
    task_generator = DynamicTaskGenerator(tool_executor)
    
    # Test task verification with sample tasks
    sample_tasks = [
        "Find information about Python programming",
        "Search for machine learning tutorials and save the results",
        "Read documents and create a summary report",
        "Calculate the sum of numbers",
        "Determine the latest content updates in a README file from a specific GitHub repository"
    ]
    
    print("\nTesting task verification:")
    for task in sample_tasks:
        is_multi_hop, explanation, sub_tasks = task_generator.verifier.is_multi_hop_task(task)
        
        print(f"\nTask: {task}")
        print(f"Multi-hop: {is_multi_hop}")
        print(f"Explanation: {explanation}")
        print(f"Sub-tasks: {sub_tasks}")
        
        # If multi-hop and LLM is available, generate better subtasks
        if is_multi_hop and task_generator.use_llm and len(sub_tasks) <= 1:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                llm_subtasks = loop.run_until_complete(task_generator.generate_subtasks_with_llm(task))
                if llm_subtasks:
                    print(f"LLM-generated subtasks: {llm_subtasks}")
            except Exception as e:
                print(f"Error generating LLM subtasks: {e}")
    
    # Test LLM-based task generation if available
    if task_generator.use_llm:
        import asyncio
        
        print("\nTesting LLM-based task generation:")
        try:
            # Run in an event loop
            loop = asyncio.get_event_loop()
            llm_tasks = loop.run_until_complete(task_generator.generate_llm_tasks(num_tasks=2))
            
            for task in llm_tasks:
                print(f"\nLLM Task: {task['task']}")
                print(f"Multi-hop: {task['verification']['is_multi_hop']}")
                print(f"Expected tools: {task.get('expected_tools', [])}")
                print(f"Sub-tasks: {task['verification']['sub_tasks']}")
        except Exception as e:
            print(f"Error testing LLM generation: {e}")
    else:
        print("\nLLM-based generation not available. Skipping test.")
    
    print("\n✅ Task generator test completed")
    
    return task_generator


if __name__ == "__main__":
    test_dynamic_task_generator()