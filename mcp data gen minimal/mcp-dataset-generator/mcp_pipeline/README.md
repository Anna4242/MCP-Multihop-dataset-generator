# MCP Pipeline for Seed Data Generation

This directory contains the core components for generating and executing multi-hop cross-server queries for the MCP (Multi-hop Cross-server Pipeline) dataset generation.

## Components

- `pipeline.py`: Main pipeline script that coordinates query generation and execution
- `MCPQueryGenerator.py`: Generates cross-server multi-hop queries
- `MCPQueryExecutor.py`: Executes the generated queries
- `mcp_servers.json`: Configuration file for MCP servers

## Setup

1. Ensure you have the required environment variables:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `MCP_INPUT_github_token`: GitHub token for MCP operations (optional)



## Usage

Run the pipeline with default settings:
```bash
py -m pipeline --execute
```

### Command Line Options

- `--config CONFIG_PATH`: Path to MCP server configuration file
- `--num_queries NUM`: Number of queries to generate (default: 5)
- `--execute`: Execute queries after generation
- `--interactive`: Select servers interactively
- `--input_queries PATH`: Use existing queries file (skips generation)
- `--model_name NAME`: LLM model to use (default: "gpt-4")
- `--temperature TEMP`: LLM temperature setting (default: 0)
- `--max_agent_steps STEPS`: Maximum steps for the agent (default: 10)

## Output

The pipeline generates:
1. Query files in the `queries/` directory
2. Execution results in the `execution_results/` directory

## Configuration

The `mcp_servers.json` file defines the available MCP servers and their configurations. Each server entry includes:
- Server name
- API endpoints
- Required authentication
- Available operations
