#!/usr/bin/env python3
"""
query_dag.py - Tool Sequence DAG Generation for Multi-Hop Queries
Provides functionality to create and visualize directed acyclic graphs of tool calls.
"""
import json
from typing import Dict, List, Optional, Tuple, Union, Set, Any


class QueryDAG:
    """
    Represents a Directed Acyclic Graph (DAG) of tool calls for a multi-hop query.
    """
    
    def __init__(self, query: str):
        """Initialize the DAG with a query."""
        self.query = query
        self.nodes = []  # List of (tool_name, operation_description, node_id)
        self.edges = []  # List of (from_node_id, to_node_id, edge_description)
    
    def add_node(self, tool_name: str, operation: str) -> int:
        """
        Add a node to the DAG representing a tool call.
        
        Args:
            tool_name: Name of the tool
            operation: Description of the operation
            
        Returns:
            Node ID
        """
        node_id = len(self.nodes)
        self.nodes.append((tool_name, operation, node_id))
        return node_id
    
    def add_edge(self, from_node: int, to_node: int, description: str = ""):
        """
        Add an edge between two nodes.
        
        Args:
            from_node: ID of the source node
            to_node: ID of the target node
            description: Description of the relationship
        """
        self.edges.append((from_node, to_node, description))
    
    def to_dict(self) -> Dict:
        """
        Convert the DAG to a dictionary representation.
        
        Returns:
            Dictionary representation of the DAG
        """
        return {
            "query": self.query,
            "nodes": [{"id": n[2], "tool": n[0], "operation": n[1]} for n in self.nodes],
            "edges": [{"from": e[0], "to": e[1], "description": e[2]} for e in self.edges]
        }
    
    def to_json(self) -> str:
        """
        Convert the DAG to a JSON string.
        
        Returns:
            JSON representation of the DAG
        """
        return json.dumps(self.to_dict(), indent=2)
    
    @staticmethod
    def generate_mermaid(dag_dict: Dict) -> str:
        """
        Generate a Mermaid.js diagram from a DAG dictionary.
        
        Args:
            dag_dict: Dictionary representation of the DAG
            
        Returns:
            Mermaid.js diagram code
        """
        # Start the graph
        mermaid = "graph TD;\n"
        
        # Add nodes
        for node in dag_dict["nodes"]:
            node_id = f"N{node['id']}"
            label = f"{node['tool']}: {node['operation']}"
            mermaid += f'    {node_id}["{label}"];\n'
        
        # Add edges
        for edge in dag_dict["edges"]:
            from_id = f"N{edge['from']}"
            to_id = f"N{edge['to']}"
            if edge["description"]:
                mermaid += f'    {from_id} -->|"{edge["description"]}"| {to_id};\n'
            else:
                mermaid += f'    {from_id} --> {to_id};\n'
        
        return mermaid


def process_tool_sequence(query: str, tool_sequence: List[Dict], tool_map: Dict) -> Tuple[Dict, str, List[str]]:
    """
    Process a tool sequence to create a DAG.
    
    Args:
        query: The original query
        tool_sequence: List of step dictionaries with tool information
        tool_map: Mapping of tool names to tool objects
        
    Returns:
        Tuple of (dag_dict, mermaid_diagram, sub_queries)
    """
    # Create a DAG from the tool sequence
    dag = QueryDAG(query)
    
    # Validate tool names and create a mapping of step to node ID
    step_to_node = {}
    valid_tools = True
    sub_queries = []
    
    # First pass: create nodes
    for step_info in tool_sequence:
        step_num = step_info.get("step", 0)
        tool_name = step_info.get("tool", "")
        operation = step_info.get("operation", "")
        
        # Add to sub-queries
        sub_queries.append(f"{operation} using {tool_name}")
        
        # Validate that the tool exists
        if tool_name not in tool_map:
            print(f"Warning: Tool '{tool_name}' not found in available tools.")
            valid_tools = False
            break
        
        # Add the node to the DAG
        node_id = dag.add_node(tool_name, operation)
        step_to_node[step_num] = node_id
    
    # Second pass: create edges
    if valid_tools:
        for step_info in tool_sequence:
            step_num = step_info.get("step", 0)
            depends_on = step_info.get("depends_on", [])
            
            current_node = step_to_node[step_num]
            
            # Add edges from dependencies to this node
            for dep_step in depends_on:
                if dep_step in step_to_node:
                    dep_node = step_to_node[dep_step]
                    dag.add_edge(dep_node, current_node)
    
    # Generate the DAG dictionary and Mermaid diagram
    dag_dict = dag.to_dict()
    mermaid_diagram = QueryDAG.generate_mermaid(dag_dict)
    
    return dag_dict, mermaid_diagram, sub_queries


def enhance_query_with_dag(query_entry: Dict, tool_map: Dict) -> Dict:
    """
    Enhance a query entry with DAG information.
    
    Args:
        query_entry: The query entry to enhance
        tool_map: Mapping of tool names to tool objects
        
    Returns:
        Enhanced query entry
    """
    query = query_entry.get("query", "")
    tool_sequence = query_entry.get("tool_sequence", [])
    
    # Process the tool sequence to create the DAG
    dag_dict, mermaid_diagram, sub_queries = process_tool_sequence(query, tool_sequence, tool_map)
    
    # Add DAG and verification information
    query_entry["dag"] = dag_dict
    query_entry["mermaid_diagram"] = mermaid_diagram
    query_entry["verification"] = {
        "is_multi_hop": len(tool_sequence) > 1,
        "explanation": "This is a multi-hop query because it requires multiple distinct tool operations.",
        "potential_sub_queries": sub_queries
    }
    
    return query_entry


def save_queries_with_diagrams(queries: List[Dict], output_dir: str, filename: str) -> str:
    """
    Save the generated queries to a JSON file along with Mermaid diagrams.
    
    Args:
        queries: List of query entries
        output_dir: Directory to save the files
        filename: Name of the JSON file
        
    Returns:
        Path to the saved JSON file
    """
    from pathlib import Path
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output file path
    output_file = output_dir / filename
    
    # Write entries to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(queries, f, indent=2)
    
    print(f"Saved {len(queries)} queries to {output_file}")

    # Also save the Mermaid diagrams separately
    diagrams_dir = output_dir / "mermaid_diagrams"
    diagrams_dir.mkdir(exist_ok=True)
    
    for i, query in enumerate(queries):
        if "mermaid_diagram" in query:
            diagram_file = diagrams_dir / f"{query['id']}_diagram.md"
            with open(diagram_file, 'w', encoding='utf-8') as f:
                f.write("```mermaid\n")
                f.write(query["mermaid_diagram"])
                f.write("\n```")
    
    return str(output_file)


if __name__ == "__main__":
    # Example usage
    example_query = {
        "query": "Restaurants near St. Andrew station, Toronto serving nachos",
        "tool_sequence": [
            {
                "step": 1,
                "tool": "geo_locate",
                "operation": "Find the location of St. Andrew station in Toronto",
                "input": "St. Andrew station, Toronto",
                "depends_on": []
            },
            {
                "step": 2,
                "tool": "search_nearby",
                "operation": "Search for restaurants near the station",
                "input": "restaurants near [location from step 1]",
                "depends_on": [1]
            },
            {
                "step": 3,
                "tool": "filter_results",
                "operation": "Filter restaurants that serve nachos",
                "input": "filter [results from step 2] for nachos",
                "depends_on": [2]
            }
        ]
    }
    
    # Mock tool map
    mock_tools = {
        "geo_locate": {"name": "geo_locate", "description": "Find locations"},
        "search_nearby": {"name": "search_nearby", "description": "Search for places near a location"},
        "filter_results": {"name": "filter_results", "description": "Filter search results"}
    }
    
    # Process the example
    result = enhance_query_with_dag(example_query, mock_tools)
    
    # Print the result
    print(json.dumps(result, indent=2))
    print("\nMermaid Diagram:")
    print("```mermaid")
    print(result["mermaid_diagram"])
    print("```")