# MCP Dataset Generator

A reinforcement learning environment for training LLM agents to use tools via the Model Context Protocol (MCP). This system dynamically discovers available MCP servers, generates diverse tasks, and provides both heuristic and LLM-based reward mechanisms.

## 🚀 Overview

The MCP Dataset Generator creates a dynamic RL environment where LLM agents learn to use various tools (filesystem, database, GitHub, web search) to complete tasks. It features:

- **Dynamic tool discovery** from available MCP servers
- **Multi Hop task generation** 
- **Hybrid reward system** (heuristic + LLM-based evaluation) inspired by [process reward models](https://arxiv.org/html/2504.04736v1)


## 📈 Use Cases

1. **RL Training Data Generation**: Collect diverse tool-use trajectories
2. **Multi-hop Reasoning**: Train on complex, sequential tasks


## 📁 Project Structure

```

├── mcp_env/
│   ├── rl/                      # Core RL components
│   │   ├── dynamic_mcp_environment.py    # Main RL environment
│   │   ├── dynamic_task_generator.py     # Task generation system
│   │   ├── reward_generator.py           # Reward calculation
│   │   ├── llm_agent.py                  # LLM-based agent
│   │   └── mcp_tool_executor.py          # Tool discovery & execution
│   ├── servers/                 # MCP server implementations
│   │   ├── Filesystem/          # File operations server
│   │   ├── Github/              # GitHub API server
│   │   ├── PostgreSQL/          # Database server
│   │   └── Search_tool/         # Web search server
│   └── examples/
│       └── rl_training_example.py        # Training script
└── collected_data/              # Generated training data
```

## 🛠️ Components

### 1. Dynamic MCP Environment (dynamic_mcp_environment.py)

The core RL environment that:

- Discovers available MCP servers and tools
- Manages episode state and tool execution
- Tracks conversation history
- Provides OpenAI Gym-like interface (reset(), step())



### 2. Dynamic Task Generator (dynamic_task_generator.py)

Generates diverse tasks based on available tools:

- LLM-based generation: Uses GPT-4 to create contextual tasks
- Multi-hop task support: Creates tasks requiring sequential tool use
- Task verification: Analyzes tasks to identify multi-hop patterns
- Subtask generation: Breaks complex tasks into steps



### 3. Reward Generator (reward_generator.py)

Sophisticated reward system with two modes:

**Heuristic Rewards:**

- Step penalty: -0.01 per step (encourages efficiency)
- Error penalty: -0.3 for tool errors
- Tool selection bonus: +0.2 for appropriate tools
- Answer quality bonus: +0.3 for good responses
- Process evaluation: Rewards logical tool sequences

**LLM-based Rewards:**

- Uses GPT-4 to evaluate action quality (0.0-1.0 scale)
- Considers task context and progress


### 4. LLM Agent (llm_agent.py)

An intelligent agent that:

- Uses GPT-4 to select appropriate tools
- Parses observations and generates valid actions




### 5. MCP Servers

**Filesystem Server**
- `list_dir(path)`: List directory contents
- `read_file(path)`: Read file content
- `write_file(path, content)`: Create/update files
- Sandboxed to prevent system access

**GitHub Server**
- `list_pull_requests(repo)`: Get PR list
- `get_pull_request(repo, number)`: Get PR details
- `clone_repo(repo_url)`: Simulate repo cloning
- `get_commit_diff(repo, commit_id)`: Get commit changes

**PostgreSQL Server**
- `list_tables()`: Show database schema
- `run_query(sql)`: Execute SELECT queries
- Read-only mode for safety
- Uses SQLite for simulation

**Search Tool Server**
- `web_search(query, noisy)`: Simulate web searches
- Returns realistic search results
- Optional noise for training robustness

## 🚀 Getting Started

### Prerequisites
```bash
pip install openai python-dotenv pandas fastmcp
```

### Environment Setup
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_api_key_here
OPENAI_API_BASE=https://openrouter.ai/api/v1  # Optional: for OpenRouter
```

### Running the Training Script
```bash
# Navigate to project root
cd mcp-dataset-generator

# Run the training example
python mcp_env/examples/rl_training_example.py
```

This will:
- Initialize the environment and discover available servers
- Generate diverse tasks using LLM
- Run episodes with the LLM agent
- Collect training data with detailed rewards
- Save results to collected_data/ as CSV and JSON



## 📊 Training Data Format

The collected data includes:

- Episode information: task, task type, success status
- Step details: observations, actions, rewards
- Reward breakdown: individual components (heuristic + LLM)
- Performance metrics: total steps, efficiency


## 🔧 Configuration

### Customize Rewards
Modify RewardConfig in the training script:
```python
reward_config = RewardConfig(
    step_penalty=-0.01,
    error_penalty=-0.2,
    answer_base_reward=0.5,
    use_llm_rewards=True,
    llm_reward_weight=0.7
)
```

### Adjust Environment
```python
env = DynamicMCPEnvironment(
    max_steps=10,  # Maximum steps per episode
    reward_config=reward_config
)
```

### Configure Agent
```python
agent = LLMAgent(
    model="openai/gpt-4o-mini",
    temperature=0.2  # Higher for exploration
)
```

## 🤝 Contributing

1. Add new MCP servers in `mcp_env/servers/`
2. Enhance task generation templates....
