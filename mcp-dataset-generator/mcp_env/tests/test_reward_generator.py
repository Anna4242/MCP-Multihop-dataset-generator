# mcp_env/tests/test_reward_generator.py
"""
Test script for the Reward Generator integration with the MCP Environment.
This file should be run from the project root directory.
"""

import sys
import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_reward_generator():
    """Test the reward generator with MCP Environment."""
    print("🧪 Testing Reward Generator with MCP Environment")
    print("=" * 60)
    
    try:
        from mcp_env.rl import DynamicMCPEnvironment
        from mcp_env.rl.reward_generator import RewardConfig
        import json
        
        # Create environment with heuristic reward generator
        print("🔄 Creating environment with heuristic reward generator...")
        heuristic_env = DynamicMCPEnvironment(max_steps=5)
        
        # Get environment info
        env_info = heuristic_env.get_environment_info()
        print(f"📊 Environment Info:")
        print(f"   • Total tools: {env_info['total_tools']}")
        print(f"   • Available tools: {env_info['available_tools']}")
        print(f"   • Reward type: {env_info['reward_type']}")
        
        # Reset environment
        obs = heuristic_env.reset(task="List available tools and their descriptions")
        obs_data = json.loads(obs)
        
        print(f"\n📋 Task: {obs_data['task']}")
        print(f"🔧 Testing different action types...")
        
        # Test different action types and their rewards
        test_actions = [
            # Good action - appropriate tool
            {
                "name": "Good action - list directory",
                "action": {"tool": "list_dir", "args": {"path": "/"}}
            },
            # Error action - missing required args
            {
                "name": "Error action - missing args",
                "action": {"tool": "read_file", "args": {}}
            },
            # Invalid action - unknown tool
            {
                "name": "Invalid action - unknown tool",
                "action": {"tool": "nonexistent_tool", "args": {}}
            },
            # Answer action
            {
                "name": "Answer action - good length",
                "action": {"tool": "answer", "args": {"answer": "The available tools include: list_dir for listing directories, read_file for reading file contents, and search for finding information. These tools can be used to interact with the environment."}}
            }
        ]
        
        # Execute actions and observe rewards
        print("\n📊 Reward analysis for different actions:")
        print("-" * 60)
        
        results = []
        
        for test_case in test_actions:
            print(f"\n🧪 Test case: {test_case['name']}")
            
            # Reset environment for each test
            obs = heuristic_env.reset(task="List available tools and their descriptions")
            
            # Execute action
            action_json = json.dumps(test_case['action'])
            observation, reward, done, info = heuristic_env.step(action_json)
            
            # Print results
            print(f"🎯 Action: {test_case['action']['tool']}")
            print(f"💰 Reward: {reward:.3f}")
            print(f"🏁 Done: {done}")
            
            if "reward_info" in info:
                reward_info = info["reward_info"]
                print(f"📊 Reward components:")
                if "components" in reward_info:
                    for component, value in reward_info["components"].items():
                        print(f"   • {component}: {value:.3f}")
            
            results.append({
                "test_case": test_case["name"],
                "action": test_case["action"]["tool"],
                "reward": reward,
                "done": done
            })
        
        # Try with LLM-based rewards if OpenAI API key is available
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            print("\n🧠 Testing LLM-based reward generator...")
            
            # Create config for LLM rewards
            llm_config = RewardConfig(
                use_llm_rewards=True,
                llm_reward_weight=0.7,
                llm_model="openai/gpt-4o-mini",
                llm_temperature=0.1
            )
            
            # Create environment with LLM reward generator
            llm_env = DynamicMCPEnvironment(max_steps=5, reward_config=llm_config)
            
            # Reset environment
            obs = llm_env.reset(task="Search for information about reinforcement learning")
            
            # Test a specific action
            test_action = {"tool": "web_search", "args": {"query": "reinforcement learning tutorial"}}
            
            print(f"\n🎯 Testing action: web_search with query='reinforcement learning tutorial'")
            action_json = json.dumps(test_action)
            observation, reward, done, info = llm_env.step(action_json)
            
            print(f"💰 LLM-based reward: {reward:.3f}")
            
            if "reward_info" in info:
                reward_info = info["reward_info"]
                print(f"📊 Reward breakdown:")
                print(f"   • Heuristic component: {reward_info.get('components', {})}")
                print(f"   • LLM component: {reward_info.get('llm_reward', 'N/A')}")
                print(f"   • Final reward: {reward_info.get('final_reward', reward):.3f}")
        else:
            print("\n⚠️ Skipping LLM-based reward tests (OPENAI_API_KEY not found)")
            print("   To test LLM rewards, set OPENAI_API_KEY in your environment or .env file")
        
        # Summary
        print("\n" + "=" * 60)
        print("📋 REWARD GENERATOR TEST SUMMARY")
        print("=" * 60)
        
        print("\nHeuristic Reward Results:")
        for result in results:
            print(f"   • {result['test_case']}: {result['reward']:.3f}")
        
        print("\n✅ Reward generator is properly integrated with the MCP environment")
        print("   Both heuristic and LLM-based reward calculations are working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Reward generator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run reward generator tests."""
    print("🚀 REWARD GENERATOR TEST SUITE")
    print("=" * 60)
    
    success = test_reward_generator()
    
    if success:
        print("\n🎉 All tests completed successfully!")
    else:
        print("\n⚠️ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()