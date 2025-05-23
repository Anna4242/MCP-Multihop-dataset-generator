# mcp_env/rl/llm_agent.py
"""
LLM Agent that can interact with the Dynamic MCP Environment.
Uses OpenAI API (or OpenRouter) to generate actions based on observations.
"""

import os
import json
import re
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

# Set OpenAI API key and base URL
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY missing in .env or environment")

class LLMAgent:
    """LLM-powered agent for the MCP environment."""
    
    def __init__(
        self, 
        model: str = "openai/gpt-4o-mini",  # Default to a good OpenRouter model
        temperature: float = 0.1,
        max_tokens: int = 1000,
        api_base: str = None
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Configure OpenAI client
        self.client = openai.OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=api_base or OPENAI_API_BASE
        )
        
        # System prompt for the agent
        self.system_prompt = self._create_system_prompt()
        
    def _create_system_prompt(self) -> str:
        """Create the system prompt for the LLM agent."""
        return """You are an AI assistant that helps complete tasks using available tools in an MCP (Model Context Protocol) environment.

Your job is to:
1. Analyze the given task and current situation
2. Choose the most appropriate tool to use next
3. Provide the correct arguments for that tool
4. Work step-by-step toward completing the task

IMPORTANT RULES:
- You must respond with a valid JSON object containing "tool" and "args" fields
- Available tools will be provided in the observation
- Use the "answer" tool when you have completed the task
- Be efficient - don't use unnecessary steps
- If you need to search/read/query, do it strategically

RESPONSE FORMAT:
{"tool": "tool_name", "args": {"param1": "value1", "param2": "value2"}}

EXAMPLE RESPONSES:
- To search: {"tool": "web_search", "args": {"query": "machine learning basics"}}
- To list files: {"tool": "list_dir", "args": {"path": "/home/user"}}
- To read a file: {"tool": "read_file", "args": {"filename": "example.txt"}}
- To finish: {"tool": "answer", "args": {"answer": "Task completed successfully. Here's what I found..."}}

Always explain your reasoning briefly before providing the JSON response."""

    def choose_action(self, observation: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """
        Choose an action based on the current observation.
        
        Args:
            observation: Current environment observation (JSON string)
            conversation_history: Optional conversation history
            
        Returns:
            Action dictionary with "tool" and "args" keys
        """
        try:
            # Parse observation
            obs_data = json.loads(observation)
            
            # Build conversation messages
            messages = [{"role": "system", "content": self.system_prompt}]
            
            # Add conversation history if provided
            if conversation_history:
                for msg in conversation_history[-6:]:  # Keep last 6 messages to avoid token limit
                    messages.append(msg)
            
            # Add current observation
            user_message = self._format_observation(obs_data)
            messages.append({"role": "user", "content": user_message})
            
            # Get LLM response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Parse the response
            response_text = response.choices[0].message.content.strip()
            action = self._extract_action_from_response(response_text)
            
            return action
            
        except Exception as e:
            print(f"⚠️ Error in LLM action selection: {e}")
            # Fallback to answer action
            return {"tool": "answer", "args": {"answer": f"Error occurred: {str(e)}"}}
    
    def _format_observation(self, obs_data: Dict) -> str:
        """Format observation data for the LLM."""
        task = obs_data.get("task", "No task specified")
        step = obs_data.get("step", "0/0")
        available_tools = obs_data.get("available_tools", [])
        tool_info = obs_data.get("tool_info", {})
        conversation = obs_data.get("conversation", [])
        
        # Get recent conversation
        recent_conversation = conversation[-4:] if len(conversation) > 4 else conversation
        
        prompt = f"""
CURRENT TASK: {task}
STEP: {step}

AVAILABLE TOOLS: {', '.join(available_tools)}

TOOL DESCRIPTIONS:
"""
        
        for tool_name, description in tool_info.items():
            prompt += f"- {tool_name}: {description}\n"
        
        if recent_conversation:
            prompt += f"\nRECENT CONVERSATION:\n"
            for msg in recent_conversation:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                prompt += f"{role.upper()}: {content}\n"
        
        prompt += f"\nWhat tool should you use next to progress toward completing the task? Provide your reasoning and then the JSON action."
        
        return prompt
    
    def _extract_action_from_response(self, response_text: str) -> Dict[str, Any]:
        """Extract JSON action from LLM response."""
        # Try to find JSON in the response
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, response_text)
        
        for match in matches:
            try:
                action = json.loads(match)
                if "tool" in action and "args" in action:
                    return action
            except json.JSONDecodeError:
                continue
        
        # If no valid JSON found, try to extract tool name and create simple action
        tool_match = re.search(r'"tool":\s*"([^"]+)"', response_text)
        if tool_match:
            tool_name = tool_match.group(1)
            return {"tool": tool_name, "args": {}}
        
        # Fallback
        return {"tool": "answer", "args": {"answer": "Unable to determine next action"}}

class MCPLLMRunner:
    """Runner that combines MCP Environment with LLM Agent."""
    
    def __init__(self, environment, agent: LLMAgent):
        self.env = environment
        self.agent = agent
        self.conversation_history = []
    
    def run_episode(self, task: str = None, task_type: str = None, max_steps: int = None) -> Dict[str, Any]:
        """
        Run a complete episode with the LLM agent.
        
        Args:
            task: Specific task to perform
            task_type: Type of task to generate
            max_steps: Maximum steps (uses env default if None)
            
        Returns:
            Episode results including final reward, steps taken, and success status
        """
        print(f"🎮 Starting LLM episode...")
        
        # Reset environment
        observation = self.env.reset(task=task, task_type=task_type)
        self.conversation_history = []
        
        # Parse initial observation to get task
        obs_data = json.loads(observation)
        current_task = obs_data.get("task", "Unknown task")
        print(f"📋 Task: {current_task}")
        
        total_reward = 0
        step_count = 0
        done = False
        
        while not done and (max_steps is None or step_count < max_steps):
            step_count += 1
            print(f"\n--- Step {step_count} ---")
            
            # Get action from LLM
            print("🤖 LLM choosing action...")
            action_dict = self.agent.choose_action(observation, self.conversation_history)
            
            # Convert to JSON string for environment
            action_json = json.dumps(action_dict)
            print(f"🎯 Action: {action_dict['tool']} with args {action_dict['args']}")
            
            # Execute action in environment
            observation, reward, done, info = self.env.step(action_json)
            total_reward += reward
            
            print(f"💰 Reward: {reward:.3f} (Total: {total_reward:.3f})")
            
            # Add to conversation history
            self.conversation_history.extend([
                {"role": "assistant", "content": f"Action: {action_json}"},
                {"role": "user", "content": f"Result: Reward={reward}, Done={done}, Info={info}"}
            ])
            
            # Check if episode ended
            if done:
                if info.get("task_completed"):
                    print(f"✅ Task completed successfully!")
                    print(f"📝 Final answer: {info.get('answer', 'No answer provided')}")
                elif "error" in info:
                    print(f"❌ Episode ended with error: {info['error']}")
                else:
                    print(f"⏰ Episode ended: {info.get('reason', 'Unknown reason')}")
        
        results = {
            "task": current_task,
            "total_reward": total_reward,
            "steps_taken": step_count,
            "completed": info.get("task_completed", False) if done else False,
            "final_info": info if done else {},
            "success": total_reward > 0 and info.get("task_completed", False) if done else False
        }
        
        print(f"\n🏁 Episode Summary:")
        print(f"   Steps: {results['steps_taken']}")
        print(f"   Total Reward: {results['total_reward']:.3f}")
        print(f"   Success: {results['success']}")
        
        return results

def test_llm_agent():
    """Test the LLM agent with the MCP environment."""
    try:
        from .dynamic_mcp_environment import DynamicMCPEnvironment
    except ImportError:
        # Handle direct execution
        from dynamic_mcp_environment import DynamicMCPEnvironment
    
    print("🧪 Testing LLM Agent with MCP Environment")
    print("=" * 50)
    
    # Create environment and agent
    env = DynamicMCPEnvironment(max_steps=8)
    agent = LLMAgent(model="openai/gpt-4o-mini", temperature=0.1)
    
    # Create runner
    runner = MCPLLMRunner(env, agent)
    
    # Run a few test episodes
    test_cases = [
        {"task_type": "general", "description": "General task"},
        {"task_type": "filesystem", "description": "File system task"},
        {"task": "List available tools and explain what each one does", "description": "Custom task"}
    ]
    
    results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🎯 Test Case {i}: {test_case['description']}")
        print("-" * 30)
        
        result = runner.run_episode(
            task=test_case.get("task"),
            task_type=test_case.get("task_type"),
            max_steps=5
        )
        results.append(result)
    
    # Print summary
    print(f"\n📊 Overall Results:")
    print("-" * 30)
    successful_episodes = sum(1 for r in results if r['success'])
    avg_reward = sum(r['total_reward'] for r in results) / len(results)
    avg_steps = sum(r['steps_taken'] for r in results) / len(results)
    
    print(f"Success Rate: {successful_episodes}/{len(results)} ({successful_episodes/len(results)*100:.1f}%)")
    print(f"Average Reward: {avg_reward:.3f}")
    print(f"Average Steps: {avg_steps:.1f}")
    
    return results

if __name__ == "__main__":
    test_llm_agent()