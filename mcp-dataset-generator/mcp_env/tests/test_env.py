# mcp_env/tests/test_env.py
"""
Test script for the Dynamic MCP Environment.
This file should be run from the project root directory.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from mcp_env.rl import DynamicMCPEnvironment
    print("✅ Successfully imported DynamicMCPEnvironment")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)

def test_environment():
    """Test the dynamic MCP environment."""
    print("\n🧪 TESTING DYNAMIC MCP ENVIRONMENT")
    print("=" * 60)
    
    try:
        # Create environment - it will auto-discover servers
        env = DynamicMCPEnvironment(max_steps=8)
        
        # Get environment info
        env_info = env.get_environment_info()
        print(f"\n📋 Environment Info:")
        print(f"   Total tools: {env_info['total_tools']}")
        print(f"   Available tools: {env_info['available_tools']}")
        
        # Test with different task types
        available_task_types = env_info['task_types']
        
        for task_type in available_task_types[:3]:  # Test first 3 task types
            print(f"\n🎯 Testing {task_type} task:")
            
            obs = env.reset(task_type=task_type)
            print(f"   Task generated successfully")
            
            # Try one action to verify everything works
            available_tools = env.tool_executor.get_available_tools()
            if available_tools:
                # Pick the first non-answer tool
                test_tool = next((t for t in available_tools if t != "answer"), "answer")
                
                if test_tool != "answer":
                    # Create a simple test action
                    import json
                    if "search" in test_tool:
                        action = json.dumps({"tool": test_tool, "args": {"query": "test"}})
                    elif "list" in test_tool:
                        action = json.dumps({"tool": test_tool, "args": {"path": "/"}})
                    else:
                        action = json.dumps({"tool": test_tool, "args": {}})
                else:
                    action = json.dumps({"tool": "answer", "args": {"answer": "Test completed"}})
                
                obs, reward, done, info = env.step(action)
                print(f"   Action executed: {test_tool}, Reward: {reward:.3f}")
                
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_environment()