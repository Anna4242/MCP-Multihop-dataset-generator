# mcp_env/examples/rl_training_with_generator.py
"""
Example of training an LLM agent using reinforcement learning with the MCP environment.
Uses the task generator for generating tasks dynamically based on available tools.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Create output directory for collected data
OUTPUT_DIR = Path("collected_data")
OUTPUT_DIR.mkdir(exist_ok=True)

def collect_training_data_with_task_generator():
    """Collect training data using the task generator with detailed reward information."""
    from mcp_env.rl import DynamicMCPEnvironment, RewardConfig
    from mcp_env.rl.llm_agent import LLMAgent, MCPLLMRunner
    
    print("🚀 Collecting MCP Training Data with Dynamic Task Generation")
    print("=" * 70)
    
    # Create reward configuration
    reward_config = RewardConfig(
        # Adjust reward values as needed
        step_penalty=-0.01,
        error_penalty=-0.2,
        max_steps_penalty=-0.1,
        invalid_action_penalty=-0.3,
        answer_base_reward=0.5,
        answer_quality_bonus=0.3,
        efficiency_bonus_factor=0.15,
        process_base_reward=0.1,
        good_tool_selection_bonus=0.2,
        good_args_bonus=0.1,
        good_query_bonus=0.2,
        
        # Set to True to use LLM-based rewards if you have an OpenAI API key
        use_llm_rewards=os.getenv("OPENAI_API_KEY") is not None,
        llm_reward_weight=0.7
    )
    
    # Create environment with the custom reward config
    env = DynamicMCPEnvironment(
        max_steps=8,
        reward_config=reward_config
    )
    
    # Create LLM agent
    agent = LLMAgent(
        model="openai/gpt-4o-mini",  # Use your preferred model
        temperature=0.2  # Higher temperature for exploration
    )
    
    # Create runner
    runner = MCPLLMRunner(env, agent)
    
    # Get available task types
    env_info = env.get_environment_info()
    available_task_types = env_info["task_types"]
    
    print(f"📋 Available task types: {available_task_types}")
    print(f"🔧 Available tools: {env_info['available_tools']}")
    print(f"💰 Reward type: {'LLM-based' if reward_config.use_llm_rewards else 'Heuristic'}")
    
    # Collected data
    collected_data = []
    
    # Number of episodes to run
    num_episodes = 10  # Adjust as needed
    
    # Run episodes with all available task types
    for task_type in available_task_types:
        if len(collected_data) >= num_episodes:
            break
            
        # Run multiple episodes per task type
        episodes_per_type = max(1, num_episodes // len(available_task_types))
        
        for episode_idx in range(episodes_per_type):
            episode_count = len(collected_data) // episodes_per_type
            
            if episode_count >= num_episodes:
                break
                
            print(f"\n{'='*70}")
            print(f"🎯 Episode {episode_count+1}: Task type '{task_type}' (Run {episode_idx+1})")
            
            # Store the original conversation history and rewards
            episode_data = []
            
            # Create a fresh copy of the conversation history
            conversation_history = []
            
            # Run the episode
            observation = env.reset(task_type=task_type)
            obs_data = json.loads(observation)
            task = obs_data["task"]
            
            print(f"📋 Task: {task}")
            
            done = False
            step_count = 0
            total_reward = 0
            
            while not done and step_count < 10:
                step_count += 1
                print(f"\n--- Step {step_count} ---")
                
                # Get action from LLM
                action_dict = agent.choose_action(observation, conversation_history)
                action_json = json.dumps(action_dict)
                
                print(f"🎯 Action: {action_dict['tool']} with args {action_dict['args']}")
                
                # Execute action in environment
                next_observation, reward, done, info = env.step(action_json)
                total_reward += reward
                
                print(f"💰 Reward: {reward:.3f} (Total: {total_reward:.3f})")
                
                # Store step data
                step_data = {
                    "episode": episode_count,
                    "task": task,
                    "task_type": task_type,
                    "step": step_count,
                    "observation": observation,
                    "action": action_json,
                    "reward": reward,
                    "reward_components": info.get("reward_info", {}).get("components", {}),
                    "llm_reward": info.get("reward_info", {}).get("llm_reward", None),
                    "done": done,
                    "next_observation": next_observation
                }
                
                episode_data.append(step_data)
                
                # Update for next step
                observation = next_observation
                
                # Add to conversation history
                conversation_history.extend([
                    {"role": "assistant", "content": f"Action: {action_json}"},
                    {"role": "user", "content": f"Result: {json.dumps(info.get('result', {}))}"}
                ])
            
            # Add episode summary
            for step in episode_data:
                step["total_steps"] = step_count
                step["total_reward"] = total_reward
                step["success"] = info.get("task_completed", False) if done else False
                collected_data.append(step)
            
            print(f"\n✅ Episode complete: {step_count} steps, {total_reward:.3f} total reward")
            if done and info.get("task_completed", False):
                print(f"🎉 Task completed successfully!")
                print(f"📝 Answer: {info.get('answer', 'No answer provided')[:150]}...")
            
            # If we have enough data, break
            if len(collected_data) >= num_episodes:
                break
    
    # If we still need more episodes, run with general task type
    if len(collected_data) < num_episodes:
        remaining_episodes = num_episodes - len(collected_data)
        print(f"\nRunning {remaining_episodes} more episodes with 'general' task type...")
        
        for i in range(remaining_episodes):
            episode_count = len(collected_data) // episodes_per_type
            
            print(f"\n{'='*70}")
            print(f"🎯 Episode {episode_count+1}: Task type 'general' (Extra run {i+1})")
            
            # Store the original conversation history and rewards
            episode_data = []
            
            # Create a fresh copy of the conversation history
            conversation_history = []
            
            # Run the episode
            observation = env.reset(task_type="general")
            obs_data = json.loads(observation)
            task = obs_data["task"]
            
            print(f"📋 Task: {task}")
            
            done = False
            step_count = 0
            total_reward = 0
            
            while not done and step_count < 10:
                step_count += 1
                print(f"\n--- Step {step_count} ---")
                
                # Get action from LLM
                action_dict = agent.choose_action(observation, conversation_history)
                action_json = json.dumps(action_dict)
                
                print(f"🎯 Action: {action_dict['tool']} with args {action_dict['args']}")
                
                # Execute action in environment
                next_observation, reward, done, info = env.step(action_json)
                total_reward += reward
                
                print(f"💰 Reward: {reward:.3f} (Total: {total_reward:.3f})")
                
                # Store step data
                step_data = {
                    "episode": episode_count,
                    "task": task,
                    "task_type": "general",
                    "step": step_count,
                    "observation": observation,
                    "action": action_json,
                    "reward": reward,
                    "reward_components": info.get("reward_info", {}).get("components", {}),
                    "llm_reward": info.get("reward_info", {}).get("llm_reward", None),
                    "done": done,
                    "next_observation": next_observation
                }
                
                episode_data.append(step_data)
                
                # Update for next step
                observation = next_observation
                
                # Add to conversation history
                conversation_history.extend([
                    {"role": "assistant", "content": f"Action: {action_json}"},
                    {"role": "user", "content": f"Result: {json.dumps(info.get('result', {}))}"}
                ])
            
            # Add episode summary
            for step in episode_data:
                step["total_steps"] = step_count
                step["total_reward"] = total_reward
                step["success"] = info.get("task_completed", False) if done else False
                collected_data.append(step)
            
            print(f"\n✅ Episode complete: {step_count} steps, {total_reward:.3f} total reward")
            if done and info.get("task_completed", False):
                print(f"🎉 Task completed successfully!")
                print(f"📝 Answer: {info.get('answer', 'No answer provided')[:150]}...")
    
    # Save collected data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"mcp_training_data_{timestamp}.csv"
    
    # Convert to DataFrame
    df = pd.DataFrame([
        {
            "episode": item["episode"],
            "task": item["task"],
            "task_type": item["task_type"],
            "step": item["step"],
            "total_steps": item["total_steps"],
            "action": json.loads(item["action"])["tool"],
            "action_args": json.dumps(json.loads(item["action"])["args"]),
            "reward": item["reward"],
            "reward_components": json.dumps(item["reward_components"]),
            "llm_reward": item["llm_reward"],
            "total_reward": item["total_reward"],
            "success": item["success"],
            "done": item["done"]
        }
        for item in collected_data
    ])
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    print(f"\n💾 Saved {len(collected_data)} data points to {output_file}")
    
    # Save full JSON data with all details
    json_output_file = OUTPUT_DIR / f"mcp_training_data_{timestamp}.json"
    with open(json_output_file, "w") as f:
        json.dump(collected_data, f, indent=2)
    
    print(f"💾 Saved full JSON data to {json_output_file}")
    
    # Generate summary
    episode_count = df["episode"].nunique()
    print(f"\n📊 Training Data Summary:")
    print(f"   • Episodes: {episode_count}")
    print(f"   • Total steps: {len(collected_data)}")
    print(f"   • Average steps per episode: {len(collected_data)/episode_count:.1f}")
    success_rate = df.groupby("episode")["success"].max().mean()
    print(f"   • Success rate: {success_rate:.1%}")
    
    # Tool usage statistics
    tool_usage = df["action"].value_counts()
    print(f"\n🔧 Tool Usage:")
    for tool, count in tool_usage.items():
        print(f"   • {tool}: {count} times ({count/len(df)*100:.1f}%)")
    
    # Reward statistics
    print(f"\n💰 Reward Statistics:")
    print(f"   • Average reward per step: {df['reward'].mean():.3f}")
    print(f"   • Average total reward: {df['total_reward'].mean():.3f}")
    print(f"   • Min reward: {df['reward'].min():.3f}")
    print(f"   • Max reward: {df['reward'].max():.3f}")
    
    return output_file, json_output_file, collected_data

def analyze_training_data(csv_file: Path):
    """Analyze the collected training data."""
    print("\n📈 ANALYZING TRAINING DATA")
    print("=" * 70)
    
    # Load data
    df = pd.read_csv(csv_file)
    
    # Episode success rate
    episodes = df["episode"].unique()
    episode_success = df.groupby("episode")["success"].max()
    success_rate = episode_success.mean()
    
    print(f"📊 Overall Statistics:")
    print(f"   • Episodes: {len(episodes)}")
    print(f"   • Success rate: {success_rate:.1%}")
    print(f"   • Average steps per episode: {df.groupby('episode').size().mean():.1f}")
    
    # Reward correlation with success
    successful_rewards = df[df["success"]]["reward"].mean()
    unsuccessful_rewards = df[~df["success"]]["reward"].mean()
    
    print(f"\n💰 Reward Analysis:")
    print(f"   • Average reward for successful episodes: {successful_rewards:.3f}")
    print(f"   • Average reward for unsuccessful episodes: {unsuccessful_rewards:.3f}")
    
    # Action distribution
    action_counts = df["action"].value_counts()
    print(f"\n🎯 Action Distribution:")
    for action, count in action_counts.items():
        print(f"   • {action}: {count} ({count/len(df)*100:.1f}%)")
    
    # Success rate by task type
    task_type_success = df.groupby(["task_type", "episode"])["success"].max().reset_index()
    task_type_success_rate = task_type_success.groupby("task_type")["success"].mean()
    
    print(f"\n📋 Task Type Analysis:")
    for task_type, success_rate in task_type_success_rate.items():
        print(f"   • {task_type}: {success_rate:.1%} success rate")
    
    # Reward components analysis
    if "reward_components" in df.columns:
        print(f"\n💰 Reward Component Analysis:")
        # Parse reward components from JSON strings
        component_dfs = []
        
        for _, row in df.iterrows():
            try:
                components = json.loads(row["reward_components"])
                if components:
                    component_df = pd.DataFrame([{"episode": row["episode"], "step": row["step"], "component": k, "value": v} 
                                              for k, v in components.items()])
                    component_dfs.append(component_df)
            except:
                pass
        
        if component_dfs:
            components_df = pd.concat(component_dfs)
            component_stats = components_df.groupby("component")["value"].agg(["mean", "count"])
            
            for component, stats in component_stats.iterrows():
                print(f"   • {component}: {stats['mean']:.3f} avg ({stats['count']} occurrences)")
    
    # Task-specific analysis
    common_tasks = df["task"].value_counts().head(5)
    print(f"\n📝 Most Common Tasks:")
    for task, count in common_tasks.items():
        print(f"   • \"{task[:80]}...\" ({count} episodes)")
    
    print("\n✅ Analysis complete")

def main():
    """Run the RL training example with task generator."""
    print("🚀 MCP RL TRAINING WITH TASK GENERATOR")
    print("=" * 70)
    
    # Check for required dependencies
    try:
        import pandas as pd
    except ImportError:
        print("⚠️ This example requires pandas. Install with: pip install pandas")
        return
    
    # Check for OpenAI API key for LLM-based rewards
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Warning: OPENAI_API_KEY not found. Using only heuristic rewards.")
        print("   For LLM-based rewards, set OPENAI_API_KEY in your environment or .env file.")
    
    # Collect training data
    csv_file, json_file, _ = collect_training_data_with_task_generator()
    
    # Analyze the collected data
    analyze_training_data(csv_file)
    
    print("\n" + "=" * 70)
    print("🎉 Example completed successfully!")
    print(f"   • CSV data saved to: {csv_file}")
    print(f"   • JSON data saved to: {json_file}")
    print(f"   • Use this data to train your RL model")
    
    print("\nNext steps:")

    print(" Implement your RL algorithm (DQN, PPO, etc.) using this data")
    

if __name__ == "__main__":
    main()