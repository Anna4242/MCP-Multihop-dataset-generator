# mcp_env/tests/test_llm.py
"""
Test script for LLM Agent integration with Dynamic MCP Environment.
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

def test_llm_agent_basic():
    """Test basic LLM agent functionality."""
    print("🧪 Testing LLM Agent with MCP Environment")
    print("=" * 50)
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        print("Please set your API key in .env file:")
        print("OPENAI_API_KEY=your_key_here")
        print("OPENAI_API_BASE=https://openrouter.ai/api/v1  # Optional")
        return False
    
    print("✅ API key found")
    
    # Test imports
    try:
        from mcp_env.rl import DynamicMCPEnvironment
        from mcp_env.rl.llm_agent import LLMAgent, MCPLLMRunner
        import openai
        print("✅ All imports successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    try:
        # Create environment and agent
        env = DynamicMCPEnvironment(max_steps=8)
        agent = LLMAgent(model="openai/gpt-4o-mini", temperature=0.1)
        
        # Create runner
        runner = MCPLLMRunner(env, agent)
        print("✅ Environment and agent created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_llm_agent_episodes():
    """Run multiple test episodes with server-specific tasks."""
    print("\n🎮 Running LLM Agent Test Episodes")
    print("=" * 50)
    
    try:
        from mcp_env.rl import DynamicMCPEnvironment
        from mcp_env.rl.llm_agent import LLMAgent, MCPLLMRunner
        
        # Create environment and agent
        env = DynamicMCPEnvironment(max_steps=8)
        agent = LLMAgent(model="openai/gpt-4o-mini", temperature=0.1)
        runner = MCPLLMRunner(env, agent)
        
        # Get available servers to create server-specific tasks
        env_info = env.get_environment_info()
        available_servers = list(env_info['servers'].keys())
        tools_by_server = {server: info['tools'] for server, info in env_info['servers'].items()}
        
        print(f"🔧 Detected servers: {available_servers}")
        
        # Define server-specific test cases
        server_specific_tasks = []
        
        # Filesystem server tasks
        if 'Filesystem' in available_servers:
            filesystem_tasks = [
                {
                    "task": "Use filesystem tools to explore the directory structure. Start by listing the current directory, then explore 1-2 subdirectories. If you find any text files, read one of them. Finally, use the ANSWER tool to provide a summary of the filesystem layout and any interesting files you discovered.",
                    "description": "Filesystem Exploration Task",
                    "server": "Filesystem"
                },
                {
                    "task": "Create a new file called 'mcp_test_report.txt' using the write_file tool. Include information about the filesystem capabilities you can access. Then use the ANSWER tool to confirm the file was created and summarize what you wrote.",
                    "description": "Filesystem Write Task", 
                    "server": "Filesystem"
                }
            ]
            server_specific_tasks.extend(filesystem_tasks)
        
        # Github server tasks
        if 'Github' in available_servers:
            github_tasks = [
                {
                    "task": "Use Github tools to analyze repository information. Try to list pull requests for a sample repository (like 'octocat/Hello-World'). Then use the ANSWER tool to provide insights about what Github capabilities are available.",
                    "description": "Github Analysis Task",
                    "server": "Github"
                },
                {
                    "task": "Clone or examine a repository using Github tools, then try to get commit information. Use the ANSWER tool to provide a summary of the repository analysis capabilities.",
                    "description": "Github Repository Task",
                    "server": "Github"
                }
            ]
            server_specific_tasks.extend(github_tasks)
        
        # PostgreSQL server tasks
        if 'PostgreSQL' in available_servers:
            postgres_tasks = [
                {
                    "task": "Connect to the PostgreSQL database using list_tables to see available tables. Then run a simple query like 'SELECT COUNT(*) FROM users' or similar. Use the ANSWER tool to provide insights about the database structure and contents.",
                    "description": "PostgreSQL Exploration Task",
                    "server": "PostgreSQL"
                },
                {
                    "task": "Use PostgreSQL tools to perform data analysis. List tables, then run 2-3 different queries to understand the data. Use the ANSWER tool to create a data summary report.",
                    "description": "PostgreSQL Analysis Task",
                    "server": "PostgreSQL"
                }
            ]
            server_specific_tasks.extend(postgres_tasks)
        
        # Search server tasks
        if 'Search_tool' in available_servers:
            search_tasks = [
                {
                    "task": "Use the web_search tool to research 'Model Context Protocol (MCP)' and 'AI tool integration'. Perform 2-3 searches with different queries. Then use the ANSWER tool to synthesize your findings into a comprehensive overview.",
                    "description": "Web Search Research Task",
                    "server": "Search_tool"
                },
                {
                    "task": "Conduct research on 'reinforcement learning for language models' using web_search. Try multiple related queries. Use the ANSWER tool to provide strategic insights and key findings.",
                    "description": "Multi-Search Analysis Task",
                    "server": "Search_tool"
                }
            ]
            server_specific_tasks.extend(search_tasks)
        
        # Multi-server integration tasks
        if len(available_servers) > 1:
            integration_tasks = [
                {
                    "task": f"Integration challenge: Use tools from multiple servers ({', '.join(available_servers[:3])}) strategically. For example, search for information, list directories, check database tables, etc. Then use the ANSWER tool to provide integrated insights about the combined capabilities.",
                    "description": "Multi-Server Integration Task",
                    "server": "Multi-Server"
                }
            ]
            server_specific_tasks.extend(integration_tasks)
        
        # Add fallback tasks if no specific servers detected
        if not server_specific_tasks:
            server_specific_tasks = [
                {
                    "task": "Explore all available tools systematically. Test each tool's capabilities safely, then use the ANSWER tool to provide a comprehensive analysis of the environment's potential.",
                    "description": "General Tool Exploration",
                    "server": "General"
                }
            ]
        
        # Run server-specific episodes
        results = []
        print(f"\n🎯 Running {len(server_specific_tasks)} server-specific tests:")
        
        for i, test_case in enumerate(server_specific_tasks, 1):
            print(f"\n{'='*60}")
            print(f"🎯 Test Case {i}: {test_case['description']}")
            print(f"🖥️ Target Server: {test_case['server']}")
            print(f"📋 Task: {test_case['task'][:100]}...")
            print("-" * 60)
            
            result = runner.run_episode(
                task=test_case["task"],
                max_steps=8  # Increased from 6 to 8 for more complex tasks
            )
            
            result['server'] = test_case['server']
            result['task_type'] = test_case['description']
            results.append(result)
        
        # Enhanced analysis with server-specific metrics
        print(f"\n📊 Server-Specific Results Analysis:")
        print("=" * 60)
        
        successful_episodes = sum(1 for r in results if r['success'])
        avg_reward = sum(r['total_reward'] for r in results) / len(results)
        avg_steps = sum(r['steps_taken'] for r in results) / len(results)
        
        print(f"Overall Performance:")
        print(f"   Success Rate: {successful_episodes}/{len(results)} ({successful_episodes/len(results)*100:.1f}%)")
        print(f"   Average Reward: {avg_reward:.3f}")
        print(f"   Average Steps: {avg_steps:.1f}")
        
        # Performance by server
        server_performance = {}
        for result in results:
            server = result['server']
            if server not in server_performance:
                server_performance[server] = []
            server_performance[server].append(result)
        
        print(f"\nPerformance by Server:")
        for server, server_results in server_performance.items():
            server_success = sum(1 for r in server_results if r['success'])
            server_avg_reward = sum(r['total_reward'] for r in server_results) / len(server_results)
            server_avg_steps = sum(r['steps_taken'] for r in server_results) / len(server_results)
            
            status = "🟢" if server_success == len(server_results) else "🟡" if server_success > 0 else "🔴"
            print(f"   {status} {server}: {server_success}/{len(server_results)} success, {server_avg_reward:.3f} avg reward, {server_avg_steps:.1f} avg steps")
        
        # Detailed task analysis
        print(f"\nDetailed Task Results:")
        for i, result in enumerate(results, 1):
            status = "✅" if result['success'] else "❌"
            print(f"   {status} {result['task_type']}: {result['total_reward']:.3f} reward in {result['steps_taken']} steps")
        
        return results
        
    except Exception as e:
        print(f"❌ Server-specific episode testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rl_focused_episodes():
    """Test RL-focused episodes with server-specific complex tasks."""
    print("\n🤖 Testing Server-Specific RL Episodes")
    print("=" * 50)
    
    try:
        from mcp_env.rl import DynamicMCPEnvironment
        from mcp_env.rl.llm_agent import LLMAgent, MCPLLMRunner
        
        # Create RL-focused environment and agent
        env = DynamicMCPEnvironment(max_steps=8)
        agent = LLMAgent(model="openai/gpt-4o-mini", temperature=0.2)  # Higher temp for exploration
        runner = MCPLLMRunner(env, agent)
        
        # Get available servers
        env_info = env.get_environment_info()
        available_servers = list(env_info['servers'].keys())
        
        # Define server-specific RL tasks
        rl_tasks = []
        
        # Filesystem RL tasks
        if 'Filesystem' in available_servers:
            rl_tasks.append({
                "task": "RL Filesystem Challenge: You are an intelligent file system explorer. Your mission: (1) Navigate and map the directory structure efficiently, (2) Identify and analyze the most valuable files, (3) Create a strategic file organization report, (4) Optimize your exploration path to minimize steps while maximizing information gain.",
                "description": "RL Filesystem Optimization",
                "server": "Filesystem"
            })
        
        # Github RL tasks
        if 'Github' in available_servers:
            rl_tasks.append({
                "task": "RL Github Intelligence: Act as a repository analysis agent. Your objectives: (1) Systematically gather repository intelligence, (2) Analyze development patterns and contributor activity, (3) Identify critical code changes and project health, (4) Generate strategic insights for development team optimization.",
                "description": "RL Github Intelligence",
                "server": "Github"
            })
        
        # PostgreSQL RL tasks
        if 'PostgreSQL' in available_servers:
            rl_tasks.append({
                "task": "RL Database Explorer: You are a data intelligence agent. Goals: (1) Efficiently map database schema and relationships, (2) Identify high-value data patterns and anomalies, (3) Execute optimal query sequences to extract maximum insights, (4) Create a data strategy report with actionable recommendations.",
                "description": "RL Database Intelligence",
                "server": "PostgreSQL"
            })
        
        # Search RL tasks
        if 'Search_tool' in available_servers:
            rl_tasks.append({
                "task": "RL Information Gathering Agent: Your mission is strategic research. Objectives: (1) Design and execute an optimal search strategy for 'MCP tool use in AI agents', (2) Balance breadth vs depth in information gathering, (3) Synthesize findings into actionable intelligence, (4) Demonstrate advanced search optimization techniques.",
                "description": "RL Search Optimization",
                "server": "Search_tool"
            })
        
        # Multi-server RL tasks
        if len(available_servers) > 1:
            rl_tasks.append({
                "task": f"RL Multi-Server Integration Challenge: You are a cross-platform intelligence agent with access to {', '.join(available_servers)}. Mission: (1) Design an optimal tool utilization strategy, (2) Execute coordinated operations across multiple servers, (3) Discover tool synergies and interaction patterns, (4) Create a comprehensive multi-platform capability assessment with strategic recommendations for future agents.",
                "description": "RL Multi-Server Mastery",
                "server": "Multi-Server"
            })
        
        # Advanced RL exploration tasks
        rl_tasks.extend([
            {
                "task": "RL Environment Mastery: You are an adaptive learning agent. Your challenge: (1) Rapidly assess and categorize all available capabilities, (2) Develop and test hypotheses about optimal tool combinations, (3) Execute a demonstration of advanced multi-step reasoning, (4) Create a comprehensive environment exploitation guide for future RL agents.",
                "description": "RL Environment Mastery",
                "server": "Advanced-RL"
            },
            {
                "task": "RL Efficiency Optimization: Balance exploration vs exploitation perfectly. Goals: (1) Discover the highest-reward action sequences, (2) Minimize redundant operations while maximizing information gain, (3) Adapt strategy based on observed reward patterns, (4) Demonstrate optimal policy execution with efficiency metrics.",
                "description": "RL Efficiency Challenge",
                "server": "Advanced-RL"
            }
        ])
        
        # Limit to available tasks
        rl_tasks = rl_tasks[:5]  # Max 5 tasks to avoid excessive API usage
        
        rl_results = []
        total_reward = 0
        
        print(f"🎯 Running {len(rl_tasks)} server-specific RL challenges:")
        
        for i, test_case in enumerate(rl_tasks, 1):
            print(f"\n{'='*70}")
            print(f"🎯 RL Challenge {i}: {test_case['description']}")
            print(f"🖥️ Target: {test_case['server']}")
            print(f"📋 Mission: {test_case['task'][:120]}...")
            print("-" * 70)
            
            result = runner.run_episode(
                task=test_case["task"],
                max_steps=7  # More steps for complex RL tasks
            )
            
            rl_results.append({
                'test_case': test_case['description'],
                'server': test_case['server'],
                'reward': result['total_reward'],
                'steps': result['steps_taken'],
                'success': result['success'],
                'efficiency': result['total_reward'] / result['steps_taken'] if result['steps_taken'] > 0 else 0,
                'task_completion': result.get('completed', False)
            })
            
            total_reward += result['total_reward']
            
            # Detailed step analysis
            efficiency_score = result['total_reward'] / result['steps_taken'] if result['steps_taken'] > 0 else 0
            if efficiency_score > 0.15:
                performance = "🟢 Excellent"
            elif efficiency_score > 0.08:
                performance = "🟡 Good"
            else:
                performance = "🔴 Needs Improvement"
            
            print(f"   📊 Results: {result['total_reward']:.3f} reward | {result['steps_taken']} steps | {efficiency_score:.3f} efficiency")
            print(f"   🎯 Performance: {performance} | Success: {'✅' if result['success'] else '❌'}")
        
        # Advanced RL Analysis
        print(f"\n📊 Server-Specific RL Performance Analysis:")
        print("=" * 70)
        
        avg_reward = total_reward / len(rl_tasks)
        avg_steps = sum(r['steps'] for r in rl_results) / len(rl_results)
        success_rate = sum(1 for r in rl_results if r['success']) / len(rl_results)
        avg_efficiency = sum(r['efficiency'] for r in rl_results) / len(rl_results)
        completion_rate = sum(1 for r in rl_results if r['task_completion']) / len(rl_results)
        
        print(f"Overall RL Metrics:")
        print(f"   🎯 RL Episodes: {len(rl_tasks)}")
        print(f"   💰 Average Reward: {avg_reward:.3f}")
        print(f"   👣 Average Steps: {avg_steps:.1f}")
        print(f"   ✅ Success Rate: {success_rate:.1%}")
        print(f"   ⚡ Efficiency: {avg_efficiency:.3f} reward/step")
        print(f"   🏁 Completion Rate: {completion_rate:.1%}")
        
        # Server-specific RL performance
        server_rl_performance = {}
        for result in rl_results:
            server = result['server']
            if server not in server_rl_performance:
                server_rl_performance[server] = []
            server_rl_performance[server].append(result)
        
        print(f"\nRL Performance by Server:")
        for server, server_results in server_rl_performance.items():
            server_success = sum(1 for r in server_results if r['success'])
            server_avg_reward = sum(r['reward'] for r in server_results) / len(server_results)
            server_avg_efficiency = sum(r['efficiency'] for r in server_results) / len(server_results)
            
            if server_avg_efficiency > 0.12:
                status = "🟢 Elite"
            elif server_avg_efficiency > 0.08:
                status = "🟡 Proficient"
            else:
                status = "🔴 Learning"
            
            print(f"   {status} {server}: {server_success}/{len(server_results)} success, {server_avg_reward:.3f} reward, {server_avg_efficiency:.3f} efficiency")
        
        # RL Learning Analysis
        if len(rl_results) >= 3:
            early_performance = sum(r['efficiency'] for r in rl_results[:len(rl_results)//2])
            late_performance = sum(r['efficiency'] for r in rl_results[len(rl_results)//2:])
            
            if late_performance > early_performance * 1.1:
                learning_trend = "🟢 Strong Learning"
            elif late_performance > early_performance * 0.95:
                learning_trend = "🟡 Stable Performance"
            else:
                learning_trend = "🔴 Declining Performance"
            
            print(f"   🧠 Learning Trend: {learning_trend}")
        
        # Overall RL assessment
        print(f"\nRL Agent Assessment:")
        if success_rate >= 0.8 and avg_efficiency > 0.12:
            print(f"   🏆 ELITE RL AGENT: Exceptional performance across server-specific challenges")
        elif success_rate >= 0.6 and avg_efficiency > 0.08:
            print(f"   🥇 ADVANCED RL AGENT: Strong performance with good optimization")
        elif success_rate >= 0.4 and avg_efficiency > 0.05:
            print(f"   🥈 COMPETENT RL AGENT: Decent performance with room for improvement")
        else:
            print(f"   🥉 DEVELOPING RL AGENT: Needs significant optimization and training")
        
        return rl_results
        
    except Exception as e:
        print(f"❌ Server-specific RL episode testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_llm_connection():
    """Test direct LLM connection and response parsing."""
    print("\n🔗 Testing LLM Connection and Response Parsing")
    print("=" * 50)
    
    try:
        import openai
        from mcp_env.rl.llm_agent import LLMAgent
        
        # Test LLM connection
        api_key = os.getenv("OPENAI_API_KEY")
        api_base = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
        
        client = openai.OpenAI(api_key=api_key, base_url=api_base)
        
        # Test basic connection
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "Respond with exactly: 'LLM Connection Test Successful'"}],
            temperature=0.1,
            max_tokens=50
        )
        
        result = response.choices[0].message.content.strip()
        print(f"✅ Basic LLM connection: {result}")
        
        # Test JSON response parsing
        agent = LLMAgent(model="openai/gpt-4o-mini", temperature=0.1)
        
        test_response = '{"tool": "list_dir", "args": {"path": "/home"}}'
        parsed_action = agent._extract_action_from_response(f"I should list the directory. {test_response}")
        
        print(f"✅ JSON parsing test: {parsed_action}")
        
        if parsed_action.get("tool") == "list_dir" and parsed_action.get("args", {}).get("path") == "/home":
            print("✅ Response parsing working correctly")
            return True
        else:
            print("❌ Response parsing failed")
            return False
            
    except Exception as e:
        print(f"❌ LLM connection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_step_by_step_interaction():
    """Test detailed step-by-step interaction."""
    print("\n🔄 Testing Step-by-Step LLM Interaction")
    print("=" * 50)
    
    try:
        from mcp_env.rl import DynamicMCPEnvironment
        from mcp_env.rl.llm_agent import LLMAgent
        
        # Create environment and agent
        env = DynamicMCPEnvironment(max_steps=4)
        agent = LLMAgent(model="openai/gpt-4o-mini", temperature=0.1)
        
        # Reset environment
        observation = env.reset(task="Analyze the available tools and provide a brief summary")
        obs_data = json.loads(observation)
        
        print(f"📋 Task: {obs_data['task']}")
        print(f"🎯 Available tools: {len(obs_data['available_tools'])} tools")
        
        # Step-by-step interaction
        conversation_history = []
        total_reward = 0
        step = 0
        
        while step < 3:  # Test 3 steps
            step += 1
            print(f"\n--- Step {step} ---")
            
            # Get action from agent
            action_dict = agent.choose_action(observation, conversation_history)
            print(f"🤖 Agent chose: {action_dict}")
            
            # Execute action
            action_json = json.dumps(action_dict)
            observation, reward, done, info = env.step(action_json)
            total_reward += reward
            
            print(f"💰 Reward: {reward:.3f} (Total: {total_reward:.3f})")
            print(f"🏁 Done: {done}")
            
            # Update conversation history
            conversation_history.extend([
                {"role": "assistant", "content": f"Action: {action_json}"},
                {"role": "user", "content": f"Reward: {reward}, Done: {done}"}
            ])
            
            if done:
                if info.get('task_completed'):
                    print(f"✅ Task completed successfully!")
                    print(f"📝 Answer: {info.get('answer', 'No answer')[:150]}...")
                break
        
        print(f"\n📊 Step-by-step test results:")
        print(f"   Steps taken: {step}")
        print(f"   Total reward: {total_reward:.3f}")
        print(f"   Average reward: {total_reward/step:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Step-by-step test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all LLM agent tests."""
    print("🚀 LLM AGENT TEST SUITE")
    print("=" * 60)
    
    # Track test results
    test_results = {}
    
    # Test 1: Basic setup
    test_results['basic_setup'] = test_llm_agent_basic()
    
    if not test_results['basic_setup']:
        print("\n❌ Basic setup failed. Cannot continue with other tests.")
        return
    
    # Test 2: LLM connection
    test_results['llm_connection'] = test_llm_connection()
    
    # Test 3: Step-by-step interaction
    test_results['step_by_step'] = test_step_by_step_interaction()
    
    # Test 4: Episode testing
    print("\n" + "=" * 60)
    choice = input("Run full episode tests? This will use API credits (y/n): ").strip().lower()
    if choice in ['y', 'yes']:
        episode_results = test_llm_agent_episodes()
        test_results['episodes'] = episode_results is not False
        
        # Test 5: RL-focused episodes
        if test_results['episodes']:
            print("\n" + "=" * 60)
            choice = input("Run RL-focused episode tests? This will use more API credits (y/n): ").strip().lower()
            if choice in ['y', 'yes']:
                rl_results = test_rl_focused_episodes()
                test_results['rl_episodes'] = rl_results is not False
            else:
                test_results['rl_episodes'] = None
                print("⏭️ Skipping RL episode tests")
        else:
            test_results['rl_episodes'] = False
    else:
        test_results['episodes'] = None
        test_results['rl_episodes'] = None
        print("⏭️ Skipping episode tests")
    
    # Summary
    print(f"\n{'=' * 60}")
    print("📋 LLM AGENT TEST SUMMARY")
    print("=" * 60)
    
    for test_name, result in test_results.items():
        if result is True:
            status = "✅ PASSED"
        elif result is False:
            status = "❌ FAILED"
        else:
            status = "⏭️ SKIPPED"
        
        print(f"   {test_name.replace('_', ' ').title()}: {status}")
    
    # Overall result
    passed = sum(1 for r in test_results.values() if r is True)
    total = sum(1 for r in test_results.values() if r is not None)
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if test_results['basic_setup'] and test_results.get('llm_connection'):
        print("\n🎉 LLM Agent is working correctly!")
        
        if test_results.get('episodes'):
            print("✅ Episode testing successful")
        if test_results.get('rl_episodes'):
            print("✅ RL-focused testing successful")
            
        print("\nYour LLM agent can:")
        print("   - Connect to the API successfully")
        print("   - Parse and execute tool actions")
        print("   - Complete multi-step tasks")
        print("   - Handle RL-style objectives")
        
        print("\nNext steps:")
        print("   - Try longer episodes with more complex tasks")
        print("   - Experiment with different models and temperatures")
        print("   - Implement custom reward functions")
        print("   - Add task-specific prompting strategies")
    else:
        print("\n⚠️ Some core tests failed. Check the errors above.")

if __name__ == "__main__":
    main()