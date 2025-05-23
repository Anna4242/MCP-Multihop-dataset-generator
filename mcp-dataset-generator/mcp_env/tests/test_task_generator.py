# mcp_env/tests/test_task_generator.py
"""
Test script for the enhanced Dynamic Task Generator.
This tests the multi-hop task generation capabilities.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from mcp_env.rl.dynamic_task_generator import DynamicTaskGenerator
    from mcp_env.rl.mcp_tool_executor import DynamicMCPToolExecutor
    print("✅ Successfully imported DynamicTaskGenerator")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)

def test_task_generator():
    """Test the enhanced task generator capabilities."""
    print("\n🧪 TESTING ENHANCED TASK GENERATOR")
    print("=" * 60)
    
    # Create tool executor
    tool_executor = DynamicMCPToolExecutor()
    print(f"📊 Tool executor created with {len(tool_executor.get_available_tools())} available tools")
    
    # Create task generator
    task_generator = DynamicTaskGenerator(tool_executor)
    
    # Test available task types
    task_types = task_generator.get_available_task_types()
    print(f"\n📋 Available task types: {task_types}")
    
    # Test standard task generation
    print("\n🔍 Testing standard task generation:")
    for _ in range(3):
        task = task_generator.generate_task()
        print(f"  • {task}")
    
    # Test multi-hop task generation
    print("\n🔍 Testing multi-hop task generation:")
    for _ in range(5):
        task = task_generator.generate_task(multi_hop=True)
        print(f"  • {task}")
    
    # Test specific task types
    if 'multi_tool' in task_types:
        print("\n🔍 Testing multi-tool tasks:")
        for _ in range(2):
            task = task_generator.generate_task(task_type='multi_tool')
            print(f"  • {task}")
    
    # Find multi-hop specific task types
    multi_hop_types = [t for t in task_types if 'multi_hop' in t or t in task_generator.multi_hop_templates]
    
    if multi_hop_types:
        print(f"\n🔍 Testing specific multi-hop task types:")
        for task_type in multi_hop_types[:3]:  # Test up to 3 types
            print(f"\n  Task type: {task_type}")
            task = task_generator.generate_task(task_type=task_type)
            print(f"  • {task}")
    
    # Test task verification
    print("\n🔍 Testing multi-hop detection:")
    test_tasks = [
        "Search for information about Python programming",  # Simple task
        "Find information about AI and create a summary document",  # Multi-hop
        "Query the database, analyze the results, and generate a report",  # Clear multi-hop
        "List the files in the directory"  # Simple task
    ]
    
    for test_task in test_tasks:
        if hasattr(task_generator, 'verifier'):
            # Updated to handle three return values
            is_multi_hop, explanation, sub_tasks = task_generator.verifier.is_multi_hop_task(test_task)
            print(f"\n  Task: {test_task}")
            print(f"  Multi-hop: {'✅' if is_multi_hop else '❌'}")
            print(f"  Explanation: {explanation}")
            print(f"  Sub-tasks: {sub_tasks}")
        else:
            print(f"\n  Task: {test_task}")
            print(f"  Verifier not available")
    
    print("\n✅ Task generator test completed successfully!")
    return task_generator

def test_with_llm_agent():
    """Test the task generator with the LLM agent."""
    try:
        from mcp_env.rl.dynamic_mcp_environment import DynamicMCPEnvironment
        from mcp_env.rl.llm_agent import LLMAgent, MCPLLMRunner
        print("\n🧪 TESTING TASK GENERATOR WITH LLM AGENT")
        print("=" * 60)
    except ImportError as e:
        print(f"❌ Cannot test with LLM agent: {e}")
        return
    
    # Create environment and agent
    env = DynamicMCPEnvironment(max_steps=8)
    agent = LLMAgent(model="openai/gpt-4o-mini", temperature=0.1)
    runner = MCPLLMRunner(env, agent)
    
    # Test with multi-hop tasks
    test_cases = [
        {"task_type": "general", "description": "Standard task", "multi_hop": False},
        {"task_type": "general", "description": "Multi-hop task", "multi_hop": True},
    ]
    
    # Run test cases
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🎯 Test Case {i}: {test_case['description']}")
        print("-" * 30)
        
        # Generate a task first to see what will be tested
        task_type = test_case.get("task_type", "general")
        multi_hop = test_case.get("multi_hop", False)
        
        task = env.task_generator.generate_task(task_type=task_type, multi_hop=multi_hop)
        print(f"Generated task: {task}")
        
        # Override the task in the test case
        test_case["task"] = task
        
        # Run the episode
        result = runner.run_episode(
            task=test_case["task"],
            max_steps=6
        )
        
        print(f"\nResults:")
        print(f"  Success: {'✅' if result['success'] else '❌'}")
        print(f"  Steps: {result['steps_taken']}")
        print(f"  Reward: {result['total_reward']:.3f}")

if __name__ == "__main__":
    # Test basic task generator functionality
    task_generator = test_task_generator()
    
    # Ask if user wants to test with LLM agent (requires API key)
    if input("\nDo you want to test with the LLM agent? (y/n): ").lower() == 'y':
        test_with_llm_agent()
    else:
        print("Skipping LLM agent test.")