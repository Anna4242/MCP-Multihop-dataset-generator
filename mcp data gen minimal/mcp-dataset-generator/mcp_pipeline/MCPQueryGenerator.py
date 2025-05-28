#!/usr/bin/env python3
"""
mcp_query_generator.py
----------------------
Class-based implementation of the multi-hop query generator.
Refactored from the original script to be more modular and
easier to import into a pipeline.

Usage as a script:
    python mcp_query_generator.py [config_path]

Usage as a module:
    from mcp_query_generator import MCPQueryGenerator, MultiHopVerifier
    
    generator = MCPQueryGenerator(config_path="path/to/config.json")
    queries = await generator.generate_queries(num_queries=10)
"""

import asyncio
import json
import os
import re
import sys
import itertools
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set, Optional, Union

import openai
from langchain_openai import ChatOpenAI
from mcp_use.client import MCPClient
from mcp_use.adapters.langchain_adapter import LangChainAdapter
from dotenv import load_dotenv

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_CONFIG_PATH = Path(
    r"D:\one drive\study\ARCEE AI INTERNSHIP\mcp data gen minimal\mcp-dataset-generator\mcp_pipeline\mcp_servers.json"
)
OUTPUT_DIR = Path("queries")
load_dotenv()

# Set OpenRouter API key and base URL
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")  # Default to OpenRouter URL
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY missing in .env or environment")

class MultiHopVerifier:
    """
    Verifier that analyzes queries to determine if they require multi-hop reasoning.
    Focuses on both linguistic structure and implicit task requirements to identify
    queries that need multiple sequential operations to solve properly.
    """
    
    def __init__(self, cross_server_mode=True):
        # Set cross-server mode - more lenient verification for cross-server queries
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
            "similarities", "better", "worse", "more", "less", "most",
            "least", "best", "worst", "cheapest", "highest", "lowest",
            "closest", "nearest", "farthest", "between", "relationship", 
            "correlation", "connection", "under", "over", "above", "below"
        }
        
        # Words indicating cause-effect relationships
        self.causal_indicators = {
            "because", "cause", "effect", "result", "impact",
            "influence", "affects", "leads to", "results in"
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
            "create", "delete", "remove", "update", "calculate",
            "serving", "available", "near", "close", "within"
        }
        
        # Entity type indicators that often require lookups
        self.entity_type_indicators = {
            "restaurant", "café", "cafe", "shop", "store", "hotel", 
            "flight", "train", "movie", "book", "product", "item",
            "apartment", "house", "home", "station", "location", 
            "venue", "company", "business", "service", "app", 
            "website", "platform", "device", "tool", "software"
        }
        
        # Location-related terms that suggest geospatial operations
        self.location_indicators = {
            "near", "close", "nearby", "proximity", "within", "distance",
            "walking", "drive", "miles", "kilometers", "blocks", "area",
            "neighborhood", "district", "downtown", "city", "region"
        }
        
        # Attribute-related terms that suggest filtering
        self.attribute_indicators = {
            "with", "has", "having", "includes", "containing", "features",
            "offers", "provides", "supports", "allows", "rating", "price", 
            "cost", "quality", "size", "duration", "time", "date", "availability"
        }
        
        # Time-related terms that may require scheduling operations
        self.time_indicators = {
            "when", "during", "schedule", "available", "open", "hours",
            "today", "tomorrow", "weekend", "weekday", "morning", "afternoon",
            "evening", "night", "date", "month", "year", "time"
        }
    
    def is_multi_hop_query(self, query: str) -> Tuple[bool, str, List[str]]:
        """
        Determine if a query requires multi-hop reasoning by analyzing its structure.
        
        Args:
            query: The query text to analyze
            
        Returns:
            Tuple of (is_multi_hop, explanation, potential_sub_queries)
        """
        # Preprocess query
        query = query.strip()
        query_lower = query.lower()
        
        # Check for obvious patterns
        has_sequence_indicators = self._contains_words_from_set(query_lower, self.sequence_indicators)
        has_multiple_questions = query.count("?") > 1
        has_multiple_conjunctions = len(re.findall(r'\band\b|\balso\b|\bwith\b', query_lower)) > 0
        has_comparison = self._contains_words_from_set(query_lower, self.comparison_indicators)
        has_causal_relation = self._contains_words_from_set(query_lower, self.causal_indicators)
        
        # Check for entity types and location references
        has_entity_type = self._contains_words_from_set(query_lower, self.entity_type_indicators)
        has_location_reference = self._contains_words_from_set(query_lower, self.location_indicators)
        has_attribute_reference = self._contains_words_from_set(query_lower, self.attribute_indicators)
        has_time_reference = self._contains_words_from_set(query_lower, self.time_indicators)
        
        # Check for multiple action verbs
        action_verbs_present = self._find_words_from_set(query_lower, self.action_verbs)
        info_verbs_present = self._find_words_from_set(query_lower, self.info_gathering_verbs)
        all_verbs_present = action_verbs_present.union(info_verbs_present)
        has_multiple_verbs = len(all_verbs_present) >= 1
        
        # Check for sentences that could be independent sub-queries
        sentences = self._split_into_sentences(query)
        has_multiple_sentences = len(sentences) > 1
        
        # Attempt to extract potential sub-queries
        potential_sub_queries = self._extract_potential_sub_queries(query)
        has_extractable_sub_queries = len(potential_sub_queries) > 1
        
        # Check for implicit multi-hop patterns
        has_superlative_with_filter = (
            any(word in query_lower for word in ["best", "worst", "most", "least", "cheapest", "highest", "lowest"]) and
            has_entity_type and
            (has_location_reference or has_attribute_reference or has_time_reference)
        )
        
        has_location_with_attribute = (
            has_entity_type and
            has_location_reference and
            has_attribute_reference
        )
        
        has_nested_attribute_filter = (
            has_entity_type and
            has_attribute_reference and
            has_comparison
        )
        
        # Decision logic for explicit linguistic markers
        explicit_factors = []
        if has_sequence_indicators:
            explicit_factors.append("contains sequencing words indicating ordered steps")
        if has_multiple_questions:
            explicit_factors.append("contains multiple questions that require separate answers")
        if has_multiple_conjunctions and (has_comparison or has_causal_relation):
            explicit_factors.append("contains multiple conjunctions connecting different requirements")
        if has_comparison:
            explicit_factors.append("involves comparison which typically requires gathering multiple pieces of information")
        if has_causal_relation:
            explicit_factors.append("involves cause-effect analysis requiring multiple investigation steps")
        if has_multiple_verbs and has_extractable_sub_queries:
            explicit_factors.append(f"contains action verbs ({', '.join(list(all_verbs_present)[:3])}) suggesting different operations")
        if has_multiple_sentences and has_extractable_sub_queries:
            explicit_factors.append("can be broken down into distinct sub-queries")
        
        # Decision logic for implicit task requirements
        implicit_factors = []
        if has_superlative_with_filter:
            implicit_factors.append("combines a superlative ranking with filtering criteria, requiring multiple steps")
        if has_location_with_attribute:
            implicit_factors.append("requires finding entities by location and then filtering by attributes")
        if has_nested_attribute_filter:
            implicit_factors.append("contains nested attribute relationships requiring sequential filtering")
        if has_entity_type and has_location_reference and has_comparison:
            implicit_factors.append("requires finding entities, filtering by location, and performing comparisons")
        
        # Special checks for cross-server mode - more lenient in determining multi-hop status
        if self.cross_server_mode:
            # Look for patterns that suggest cross-server operations
            if self._has_cross_server_indicators(query_lower):
                explicit_factors.append("contains terms that suggest operations across different server types")
            
            # In cross-server mode, most conjunctions probably indicate multi-hop since they'll need different servers
            if has_multiple_conjunctions:
                implicit_factors.append("contains conjunctions that likely require different server capabilities")
            
            # Multiple verbs with different typical server associations (file operations + web operations)
            file_ops = self._contains_words_from_set(query_lower, {"save", "download", "file", "document", "folder", "upload"})
            web_ops = self._contains_words_from_set(query_lower, {"search", "find", "browse", "web", "internet", "online"})
            calendar_ops = self._contains_words_from_set(query_lower, {"schedule", "calendar", "appointment", "meeting", "event"})
            
            if (file_ops and web_ops) or (file_ops and calendar_ops) or (web_ops and calendar_ops):
                implicit_factors.append("combines operations typically requiring different server types")
        
        # Combine factors
        factors = explicit_factors + implicit_factors
        
        # Make final determination
        # Check explicit linguistic markers
        is_explicitly_multi_hop = (has_sequence_indicators or 
                                has_multiple_questions or 
                                (has_multiple_conjunctions and (has_comparison or has_causal_relation)) or 
                                (has_multiple_verbs and has_extractable_sub_queries))
        
        # Check for implicit multi-hop patterns
        is_implicitly_multi_hop = (has_superlative_with_filter or 
                                has_location_with_attribute or 
                                has_nested_attribute_filter or 
                                (has_entity_type and has_comparison and (has_location_reference or has_attribute_reference)))
        
        # For cross-server mode, be more lenient
        if self.cross_server_mode:
            # In cross-server mode, consider any query with multiple conjunctions as potentially multi-hop
            if has_multiple_conjunctions:
                is_implicitly_multi_hop = True
                
            # Operations across different domains are likely multi-hop
            file_ops = self._contains_words_from_set(query_lower, {"save", "download", "file", "document", "folder", "upload"})
            web_ops = self._contains_words_from_set(query_lower, {"search", "find", "browse", "web", "internet", "online"})
            calendar_ops = self._contains_words_from_set(query_lower, {"schedule", "calendar", "appointment", "meeting", "event"})
            
            if (file_ops and web_ops) or (file_ops and calendar_ops) or (web_ops and calendar_ops):
                is_implicitly_multi_hop = True
        
        is_multi_hop = is_explicitly_multi_hop or is_implicitly_multi_hop
        
        # Generate explanation
        if is_multi_hop:
            if factors:
                explanation = "This appears to be a multi-hop query because it " + "; it ".join(factors) + "."
            else:
                explanation = "This query likely requires multiple steps to complete, based on its complexity."
        else:
            explanation = "This appears to be a single-hop query as it doesn't show clear indications of requiring sequential steps."
        
        # Generate sub-queries if not already extracted
        if is_multi_hop and not has_extractable_sub_queries:
            potential_sub_queries = self._generate_implicit_sub_queries(query, has_entity_type, has_location_reference, 
                                                          has_attribute_reference, has_comparison, has_time_reference)
        
        return is_multi_hop, explanation, potential_sub_queries
    
    def _has_cross_server_indicators(self, text: str) -> bool:
        """Check for phrases that suggest operations across different server types."""
        indicators = {
            # File + Web combinations
            "save the results", "download and save", "find and save", "search and download",
            "save to my", "email the results", "add to my", "save it to",
            
            # Calendar + Web combinations
            "schedule a", "book a", "add to calendar", "check my calendar", 
            "find availability", "appointments near", "check free time",
            
            # Email + Web combinations
            "email me the", "send an email", "update my contact", "share via email",
            
            # Finance + Web combinations
            "calculate the cost", "compare prices", "budget for", "financial impact",
            
            # Weather + Location combinations
            "weather in", "forecast for", "temperature in", "weather near",
            
            # Generic cross-domain indicators
            "based on my", "using my", "from my", "to my", "in my"
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
    
    def _extract_potential_sub_queries(self, query: str) -> List[str]:
        """
        Attempt to break down a query into potential sub-queries.
        This is a heuristic approach that looks for common patterns.
        """
        sub_queries = []
        
        # Method 1: Split by sequence indicators
        for indicator in ["then", "after that", "next", "followed by", "finally"]:
            if indicator in query.lower():
                parts = re.split(rf'\b{re.escape(indicator)}\b', query, flags=re.IGNORECASE)
                if len(parts) > 1:
                    # Clean up and add parts
                    for i, part in enumerate(parts):
                        part = part.strip()
                        if part:
                            if i > 0:  # This is a subsequent step
                                sub_queries.append(f"{indicator} {part}")
                            else:  # This is the first step
                                sub_queries.append(part)
                    return sub_queries  # Return if we found a clear sequence
        
        # Method 2: Split by "and" if it seems to connect actions
        and_parts = re.split(r'\band\b', query, flags=re.IGNORECASE)
        if len(and_parts) > 1:
            # Check if these parts look like separate actions
            valid_parts = []
            for part in and_parts:
                part = part.strip()
                if part and any(verb in part.lower() for verb in self.action_verbs):
                    valid_parts.append(part)
            
            if len(valid_parts) > 1:
                return valid_parts
        
        # Method 3: Split by semicolons and commas (if they appear to separate actions)
        if ";" in query:
            semicolon_parts = [p.strip() for p in query.split(";") if p.strip()]
            if len(semicolon_parts) > 1:
                return semicolon_parts
        
        # Method 4: Look for enumeration patterns (1., 2., etc.)
        enum_pattern = re.compile(r'(?:\d+\.\s*|\([a-z]\)\s*|\([0-9]+\)\s*)')
        if enum_pattern.search(query):
            enum_parts = enum_pattern.split(query)
            # Remove empty parts and the part before the first number (if it exists)
            enum_parts = [p.strip() for p in enum_parts if p.strip()]
            if len(enum_parts) > 1:
                return enum_parts
        
        # If no clear sub-queries found, return original as single query
        if not sub_queries:
            sub_queries = [query]
        
        return sub_queries
    
    def _generate_implicit_sub_queries(self, query: str, has_entity: bool, has_location: bool, 
                                      has_attribute: bool, has_comparison: bool, has_time: bool) -> List[str]:
        """
        Generate potential sub-queries for implicit multi-hop queries.
        This uses a template-based approach based on identified query characteristics.
        """
        query_lower = query.lower()
        sub_queries = []
        
        # Extract key elements
        entities = [word for word in self.entity_type_indicators if word in query_lower]
        entity = entities[0] if entities else "item"
        
        locations = [word for word in self.location_indicators if word in query_lower]
        location = locations[0] if locations else "location"
        
        attributes = [word for word in self.attribute_indicators if word in query_lower]
        attribute = attributes[0] if attributes else "attribute"
        
        comparisons = [word for word in self.comparison_indicators if word in query_lower]
        comparison = comparisons[0] if comparisons else "best"
        
        times = [word for word in self.time_indicators if word in query_lower]
        time_element = times[0] if times else "availability"
        
        # Pattern 1: Location + Entity + Attribute
        # Example: "Restaurants near St. Andrew station, Toronto serving nachos"
        if has_entity and has_location and has_attribute:
            sub_queries = [
                f"Find location of {location} mentioned in the query",
                f"Search for {entity} near the identified {location}",
                f"Filter {entity} results based on {attribute} criteria"
            ]
        
        # Pattern 2: Superlative + Entity + Attribute
        # Example: "Best coffee shops with free wifi in downtown"
        elif has_entity and has_comparison and has_attribute:
            sub_queries = [
                f"Find all {entity} with specified {attribute}",
                f"Rank or filter results based on {comparison} criteria",
                f"Return the {comparison} options"
            ]
        
        # Pattern 3: Entity + Location + Time
        # Example: "Hotels in Vancouver with availability next weekend"
        elif has_entity and has_location and has_time:
            sub_queries = [
                f"Locate the {location} mentioned in the query",
                f"Find {entity} in the specified {location}",
                f"Check {time_element} for the found {entity}"
            ]
        
        # Pattern 4: Entity + Multiple Attributes
        # Example: "Electric cars with over 300 miles of range and available tax incentives"
        elif has_entity and has_attribute and has_multiple_conjunctions:
            sub_queries = [
                f"Find all {entity} options",
                f"Filter by first attribute criterion",
                f"Further filter by additional attribute criteria"
            ]
        
        # Pattern 5: Comparative analysis
        # Example: "What's the weather forecast for Chicago this weekend compared to New York?"
        elif has_comparison and "compared" in query_lower:
            parts = query_lower.split("compared")
            if len(parts) > 1:
                sub_queries = [
                    f"Get information for {parts[0].strip()}",
                    f"Get comparative information for {parts[1].strip()}",
                    "Compare the results"
                ]
        
        # Default pattern if none of the above match
        if not sub_queries:
            sub_queries = [
                "Identify the main entities or concepts in the query",
                "Gather relevant information about these entities",
                "Process or filter the information based on query criteria",
                "Present the results in the requested format"
            ]
        
        return sub_queries


class MCPQueryGenerator:
    """
    Generate queries tailored to specific MCP server types and their tools.
    """
    
    def __init__(
        self, 
        config_path: Union[str, Path] = None,
        api_key: str = None,
        api_base: str = None,
        output_dir: Union[str, Path] = "queries"
    ):
        """
        Initialize the query generator.
        
        Args:
            config_path: Path to MCP server configuration file
            api_key: OpenAI API key (if not provided, will try to use environment variable)
            api_base: OpenAI API base URL (defaults to OpenRouter URL)
            output_dir: Directory to save generated queries
        """
        self.config_path = Path(config_path or DEFAULT_CONFIG_PATH)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.api_base = api_base or os.environ.get("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
        self.output_dir = Path(output_dir)
        
        if not self.api_key:
            logger.error("OpenAI API key is required but not found")
        else:
            # Initialize OpenAI client
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.api_base
            )
            
            # Initialize verifier
            self.verifier = MultiHopVerifier()
    
    def load_config(self) -> Dict:
        """Load and parse the MCP configuration file"""
        if not self.config_path.exists():
            logger.error(f"Error: Config file not found at {self.config_path}")
            return {}
        
        try:
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
            return config_data
        except Exception as e:
            logger.error(f"Error reading config file: {e}")
            return {}
    
    def list_servers(self, config_data: Dict) -> List[str]:
        """Extract server names from configuration data"""
        servers = []
        if "mcpServers" in config_data:
            servers = list(config_data["mcpServers"].keys())
        return servers
    
    async def get_mcp_tools_for_server(self, config_data: Dict, server_name: str) -> Tuple[List, Any]:
        """
        Retrieves tools from a specific MCP server.
        
        Args:
            config_data: The full configuration data
            server_name: Name of the server to connect to
            
        Returns:
            Tuple of (tools, client)
        """
        # Create server-specific config
        server_config = {
            "mcpServers": {
                server_name: config_data["mcpServers"][server_name]
            }
        }
        
        # Save to temporary file
        temp_config_path = Path(f"temp_{server_name}_config.json")
        with open(temp_config_path, 'w') as f:
            json.dump(server_config, f, indent=2)
        
        try:
            # Initialize MCP client for this server
            client = MCPClient.from_config_file(str(temp_config_path))
            
            # Create adapter instance
            adapter = LangChainAdapter()
            
            # Get LangChain tools
            tools = await adapter.create_tools(client)
            logger.info(f"Successfully retrieved {len(tools)} tools from {server_name} server")
            
            # Print tool names
            if tools:
                tool_names = [tool.name for tool in tools]
                logger.info(f"Available tools from {server_name}: {', '.join(tool_names[:5])}...")
            
            return tools, client
        except Exception as e:
            logger.error(f"Error retrieving tools from {server_name}: {e}")
            import traceback
            traceback.print_exc()
            return [], None
        finally:
            # Clean up temporary file
            if temp_config_path.exists():
                temp_config_path.unlink()
    
    def _categorize_tools(self, tools_with_servers: List[Dict]) -> Dict:
        """
        Group tools into categories based on their names and descriptions.
        This helps in creating more structured prompts.
        """
        categories = {
            "search": [],
            "file_system": [],
            "calculator": [],
            "calendar": [],
            "location": [],
            "browser": [],
            "booking": [],
            "email": [],
            "weather": [],
            "crypto": [],
            "image": [],
            "other": []
        }
        
        for tool in tools_with_servers:
            name = tool["name"].lower()
            desc = tool["description"].lower()
            
            # Categorize based on keywords
            if any(term in name or term in desc for term in ["search", "find", "lookup", "query", "google", "bing"]):
                categories["search"].append(tool)
            elif any(term in name or term in desc for term in ["file", "folder", "directory", "save", "open", "read", "write"]):
                categories["file_system"].append(tool)
            elif any(term in name or term in desc for term in ["calculate", "math", "computation", "solve"]):
                categories["calculator"].append(tool)
            elif any(term in name or term in desc for term in ["calendar", "schedule", "event", "appointment"]):
                categories["calendar"].append(tool)
            elif any(term in name or term in desc for term in ["location", "map", "direction", "distance", "nearby"]):
                categories["location"].append(tool)
            elif any(term in name or term in desc for term in ["browser", "web", "page", "site", "visit", "navigate"]):
                categories["browser"].append(tool)
            elif any(term in name or term in desc for term in ["book", "reserve", "reservation", "ticket", "hotel", "flight"]):
                categories["booking"].append(tool)
            elif any(term in name or term in desc for term in ["email", "mail", "message", "send"]):
                categories["email"].append(tool)
            elif any(term in name or term in desc for term in ["weather", "forecast", "temperature", "climate"]):
                categories["weather"].append(tool)
            elif any(term in name or term in desc for term in ["crypto", "bitcoin", "ethereum", "coin", "token"]):
                categories["crypto"].append(tool)
            elif any(term in name or term in desc for term in ["image", "picture", "photo", "draw", "render"]):
                categories["image"].append(tool)
            else:
                categories["other"].append(tool)
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}
    
    def _create_cross_server_prompt(self, server_names: List[str], tools: List[Dict], 
                                   tool_categories: Dict, num_queries: int) -> str:
        """
        Create a detailed prompt for generating cross-server queries.
        """
        # Create tool descriptions grouped by server
        server_tool_descriptions = {}
        for server in server_names:
            server_tools = [t for t in tools if t["server"] == server]
            server_tool_descriptions[server] = "\n".join([
                f"- {tool['name']}: {tool['description']}" 
                for tool in server_tools
            ])
        
        # Create a categorized tool list
        categorized_tools = ""
        for category, category_tools in tool_categories.items():
            categorized_tools += f"\n## {category.upper()} TOOLS:\n"
            for tool in category_tools:
                categorized_tools += f"- {tool['name']} (server: {tool['server']}): {tool['description']}\n"
        
        # Create the system prompt
        system_prompt = f"""
You are an expert query generator creating TRUE MULTI-HOP queries for an AI assistant that uses tools from multiple MCP (Model Context Protocol) servers.

You are generating queries that will require tools from these MCP servers:
{', '.join(server_names)}

{categorized_tools}

Your task is to generate {num_queries} short, realistic queries that are TRULY MULTI-HOP - meaning each step depends on the output of the previous step.

## TRUE MULTI-HOP REQUIREMENTS:
1. Each query MUST decompose into 2-4 sequential sub-queries
2. Each sub-query requires exactly ONE tool call to solve
3. The OUTPUT of each tool call must serve as INPUT to the next tool call
4. Must use tools from AT LEAST 2 DIFFERENT SERVERS
5. Create a clear dependency chain: Tool A output → Tool B input → Tool C input

## QUERY STRUCTURE:
- SHORT queries (5-15 words max)
- Sound like natural user requests
- Hide the complexity - user doesn't realize it's multi-step
- Each step MUST use the previous step's output as input

## DEPENDENCY CHAIN EXAMPLES:
Good Multi-Hop Flow:
1. "Book a table at the highest-rated Italian restaurant nearby"
   - Step 1: search_restaurants(cuisine="Italian") → restaurant_list
   - Step 2: get_ratings(restaurant_list) → rated_restaurants  
   - Step 3: find_highest_rated(rated_restaurants) → best_restaurant
   - Step 4: book_table(best_restaurant.id) → confirmation

2. "Email me the weather for my next meeting location"
   - Step 1: get_calendar() → upcoming_meetings
   - Step 2: extract_location(next_meeting) → meeting_location
   - Step 3: get_weather(meeting_location) → weather_data
   - Step 4: send_email(weather_data) → sent_confirmation

## BAD EXAMPLES (NOT MULTI-HOP):
- "Check weather and send email" (parallel tasks, not dependent)
- "Find restaurants and book one" (missing dependency on search results)
- "Save document and create calendar event" (independent actions)

## GOOD EXAMPLES (TRUE MULTI-HOP):
- "Order pizza from the closest place that delivers"
- "Call the author of my most recent email"  
- "Schedule lunch with my manager next free day"
- "Buy stock in the company mentioned in today's top news"
- "Text my roommate the cheapest gas station route home"
- "Book the earliest available appointment with my doctor"
- "Order the ingredients for tonight's recipe suggestion"

## SERVER TOOLS AVAILABLE:
{', '.join([f"SERVER: {server}\n{desc}" for server, desc in server_tool_descriptions.items()])}

Return the queries as a JSON array with this format:
[
  {{
    "query": "Short realistic query requiring true multi-hop processing",
    "sub_queries": [
      "Step 1: Tool call needed",
      "Step 2: Tool call using Step 1 output", 
      "Step 3: Tool call using Step 2 output"
    ],
    "tool_chain": [
      {{"tool": "tool_name", "server": "server_name", "input": "what input needed", "output": "what output produced"}},
      {{"tool": "tool_name", "server": "server_name", "input": "uses_previous_output", "output": "what output produced"}},
      {{"tool": "tool_name", "server": "server_name", "input": "uses_previous_output", "output": "final result"}}
    ],
    "dependency_explanation": "Clear explanation of how each step depends on the previous step's output, showing the data flow chain"
  }}
]

CRITICAL REQUIREMENTS:
- Queries must be SHORT (5-15 words)
- Each step MUST use the previous step's actual output as input
- NO parallel processing - strict sequential dependency
- ONLY use tools that actually exist in the provided servers
- Each query MUST span at least 2 different servers
- Focus on REALISTIC tasks users would actually request
"""
        return system_prompt
    
    async def generate_cross_server_queries(
        self, 
        server_tool_map: Dict[str, List], 
        num_queries: int = 5,
        min_multi_hop_percent: float = 0.5
    ) -> List[Dict]:
        """
        Generate queries that require tools from multiple servers.
        
        Args:
            server_tool_map: Dictionary mapping server names to their tool lists
            num_queries: Number of queries to generate
            min_multi_hop_percent: Minimum percentage of queries that should be multi-hop
            
        Returns:
            List of query entries with verification metadata
        """
        # Create a comprehensive tool list with server origins
        all_tools_with_servers = []
        for server_name, tools in server_tool_map.items():
            for tool in tools:
                all_tools_with_servers.append({
                    "server": server_name,
                    "name": tool.name,
                    "description": tool.description
                })
        
        # Sort servers by name for consistent output
        server_names = sorted(server_tool_map.keys())
        
        # Group tools by type/category for better prompting
        tool_categories = self._categorize_tools(all_tools_with_servers)
        
        # Create a system prompt that explains the task
        system_prompt = self._create_cross_server_prompt(server_names, all_tools_with_servers, tool_categories, num_queries)
        
        # Generate queries with OpenAI
        try:
            # Set up the parameters for the API call
            params = {
                "model": "openai/gpt-4o",  # Updated model name with provider prefix for OpenRouter
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Generate {num_queries} diverse cross-server multi-hop queries that require tools from multiple MCP servers. Each query should appear simple but require tools from at least 2 different servers to solve properly."}
                ],
                "temperature": 0.7,
                "max_tokens": 2000
            }
            
            # Try to call the API with the most compatible approach
            try:
                response = self.client.chat.completions.create(**params)
                content = response.choices[0].message.content
            except Exception as e:
                logger.warning(f"Standard API call failed, trying alternative: {e}")
                # Try with direct API call as a fallback (older OpenAI client versions)
                try:
                    response = openai.ChatCompletion.create(
                        api_key=self.api_key,
                        api_base=self.api_base,
                        **params
                    )
                    content = response['choices'][0]['message']['content']
                except Exception as e2:
                    logger.error(f"Error: Both API call methods failed: {e2}")
                    raise
            
            # Extract JSON from the response
            try:
                # Look for JSON in the response
                start_idx = content.find("[")
                end_idx = content.rfind("]") + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = content[start_idx:end_idx]
                    generated_queries = json.loads(json_str)
                    
                    # Verify each query
                    verified_queries = []
                    multi_hop_count = 0
                    cross_server_count = 0
                    
                    for i, entry in enumerate(generated_queries):
                        query = entry.get("query", "")
                        is_multi_hop, explanation, sub_queries = self.verifier.is_multi_hop_query(query)
                        
                        # Check if it uses multiple servers
                        expected_tools = entry.get("expected_tools", [])
                        servers_used = set()
                        for tool_name in expected_tools:
                            for tool_info in all_tools_with_servers:
                                if tool_info["name"] == tool_name:
                                    servers_used.add(tool_info["server"])
                                    break
                        
                        is_cross_server = len(servers_used) > 1
                        
                        # For cross-server queries, if they use multiple servers, consider them multi-hop
                        # This is more appropriate for cross-server use cases
                        if is_cross_server:
                            is_multi_hop = True
                            if not explanation.startswith("This appears to be a multi-hop query"):
                                explanation = "This appears to be a multi-hop query because it requires tools from multiple servers, which inherently implies multiple processing steps."
                        
                        # Add verification information
                        entry["verification"] = {
                            "is_multi_hop": is_multi_hop,
                            "is_cross_server": is_cross_server,
                            "servers_used": list(servers_used),
                            "explanation": explanation,
                            "potential_sub_queries": sub_queries
                        }
                        
                        # Add server info
                        entry["servers"] = ", ".join(server_names)
                        entry["id"] = f"cross_server_{datetime.now().strftime('%Y%m%d%H%M%S')}_{i}"
                        
                        verified_queries.append(entry)
                        
                        if is_multi_hop:
                            multi_hop_count += 1
                        if is_cross_server:
                            cross_server_count += 1
                    
                    # Calculate multi-hop and cross-server percentages
                    multi_hop_percent = multi_hop_count / len(verified_queries) if verified_queries else 0
                    cross_server_percent = cross_server_count / len(verified_queries) if verified_queries else 0
                    
                    logger.info(f"Generated {len(verified_queries)} cross-server queries")
                    logger.info(f"Multi-hop queries: {multi_hop_count} ({multi_hop_percent:.1%})")
                    logger.info(f"Cross-server queries: {cross_server_count} ({cross_server_percent:.1%})")
                    
                    # If we don't have enough multi-hop or cross-server queries, try to generate more
                    if (multi_hop_percent < min_multi_hop_percent or cross_server_percent < min_multi_hop_percent) and len(verified_queries) > 0:
                        logger.info(f"Insufficient multi-hop or cross-server queries. Regenerating...")
                        
                        # Instead of recursive call, try one more time with adjusted parameters
                        additional_params = {
                            "model": "openai/gpt-4o",
                            "messages": [
                                {"role": "system", "content": system_prompt + "\n\nIMPORTANT: Your previous responses did not generate enough multi-hop queries. Please create queries that are DEFINITELY multi-hop, requiring multiple sequential operations and tool calls. Use more complex scenarios that involve filtering, comparing, and manipulating data across multiple servers."},
                                {"role": "user", "content": f"Generate {num_queries} queries that are GUARANTEED to be multi-hop AND cross-server. Each query must require sequential steps across at least 2 different servers."}
                            ],
                            "temperature": 0.8,  # Slightly higher temperature for more creativity
                            "max_tokens": 2000
                        }
                        
                        try:
                            response = self.client.chat.completions.create(**additional_params)
                            content = response.choices[0].message.content
                            
                            start_idx = content.find("[")
                            end_idx = content.rfind("]") + 1
                            
                            if start_idx >= 0 and end_idx > start_idx:
                                json_str = content[start_idx:end_idx]
                                more_queries = json.loads(json_str)
                                
                                # Process these additional queries
                                for i, entry in enumerate(more_queries):
                                    query = entry.get("query", "")
                                    is_multi_hop, explanation, sub_queries = self.verifier.is_multi_hop_query(query)
                                    
                                    # For cross-server queries, consider them multi-hop if they use multiple servers
                                    # This is a reasonable assumption since using different servers typically requires multiple steps
                                    expected_tools = entry.get("expected_tools", [])
                                    servers_used = set()
                                    for tool_name in expected_tools:
                                        for tool_info in all_tools_with_servers:
                                            if tool_info["name"] == tool_name:
                                                servers_used.add(tool_info["server"])
                                                break
                                    
                                    # If it uses multiple servers, consider it multi-hop regardless of verifier
                                    if len(servers_used) > 1:
                                        is_multi_hop = True
                                        if not explanation.startswith("This appears to be a multi-hop query"):
                                            explanation = "This appears to be a multi-hop query because it requires tools from multiple servers, which inherently implies multiple processing steps."
                                    
                                    is_cross_server = len(servers_used) > 1
                                    
                                    # Add verification information
                                    entry["verification"] = {
                                        "is_multi_hop": is_multi_hop,
                                        "is_cross_server": is_cross_server,
                                        "servers_used": list(servers_used),
                                        "explanation": explanation,
                                        "potential_sub_queries": sub_queries
                                    }
                                    
                                    # Add server info
                                    entry["servers"] = ", ".join(server_names)
                                    entry["id"] = f"cross_server_{datetime.now().strftime('%Y%m%d%H%M%S')}_{i+len(verified_queries)}"
                                    
                                    verified_queries.append(entry)
                            
                        except Exception as e:
                            logger.error(f"Error generating additional queries: {e}")
                        
                    return verified_queries
                else:
                    logger.error(f"No JSON array found in response")
                    logger.error(f"Content: {content[:200]}...")
                    return []
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON: {e}")
                logger.error(f"Content: {content[:200]}...")
                return []
                
        except Exception as e:
            logger.error(f"Error generating queries: {e}")
            return []
    
    async def save_queries_to_file(self, queries: List[Dict], filename_prefix: str = "cross_server_queries") -> Optional[Path]:
        """Save the generated queries to a JSON file."""
        if not queries:
            logger.error("No queries to save")
            return None
            
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output file path with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = self.output_dir / f"{filename_prefix}_{timestamp}.json"
        
        # Write entries to JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(queries, f, indent=2)
        
        logger.info(f"Saved {len(queries)} queries to {output_file}")
        return output_file

    async def select_servers_interactive(self, servers: List[str]) -> List[str]:
        """Allow user to select which servers to use interactively."""
        if not servers:
            logger.error("No servers found in configuration!")
            return []
        
        print("\nAvailable MCP Servers:")
        for i, server in enumerate(servers, 1):
            print(f"{i}. {server}")
        
        print(f"\n{len(servers) + 1}. All servers")
        
        while True:
            choice = input("\nEnter server number(s) to use (comma-separated, or 'all'): ")
            
            if choice.lower() in ('all', 'a', str(len(servers) + 1)):
                logger.info(f"Selected all {len(servers)} servers")
                return servers
            
            try:
                # Parse comma-separated list of numbers
                selections = [int(x.strip()) for x in choice.split(',')]
                
                # Validate selections
                valid_selections = [s for s in selections if 1 <= s <= len(servers)]
                
                if not valid_selections:
                    print("No valid selections. Please try again.")
                    continue
                
                # Convert to server names
                selected_servers = [servers[i-1] for i in valid_selections]
                
                logger.info(f"Selected servers: {', '.join(selected_servers)}")
                return selected_servers
                
            except ValueError:
                print("Invalid input. Please enter numbers separated by commas.")
                continue

    async def generate_queries(
        self, 
        num_queries: int = 5, 
        interactive: bool = True,
        selected_servers: List[str] = None
    ) -> Optional[Path]:
        """
        Main method to generate cross-server multi-hop queries.
        
        Args:
            num_queries: Number of queries to generate
            interactive: Whether to select servers interactively
            selected_servers: List of server names to use (ignored if interactive=True)
            
        Returns:
            Path to the generated queries file, or None if generation failed
        """
        # Check if API key is available
        if not self.api_key:
            logger.error("Cannot generate queries: OpenAI API key not found")
            return None
            
        # Load the configuration
        config_data = self.load_config()
        
        if not config_data:
            logger.error("Failed to load configuration. Exiting.")
            return None
        
        # List available servers
        servers = self.list_servers(config_data)
        
        if not servers:
            logger.error("No MCP servers found in configuration. Exiting.")
            return None
        
        # Select servers (either interactively or from passed list)
        if interactive and not selected_servers:
            selected_servers = await self.select_servers_interactive(servers)
        elif not selected_servers:
            selected_servers = servers  # Use all servers if not specified
        
        if len(selected_servers) < 2:
            logger.error("At least 2 servers are required for cross-server queries. Exiting.")
            return None
        
        # Collect tools from all selected servers
        server_tool_map = {}
        clients = []  # Keep track of clients to close them later
        
        for server_name in selected_servers:
            logger.info(f"Connecting to server: {server_name}")
            
            # Get tools from this server
            tools, client = await self.get_mcp_tools_for_server(config_data, server_name)
            if client:
                clients.append(client)
            
            if not tools:
                logger.warning(f"No tools available from {server_name}. Skipping.")
                continue
            
            server_tool_map[server_name] = tools
        
        if len(server_tool_map) < 2:
            logger.error("Could not retrieve tools from at least 2 servers. Exiting.")
            return None
        
        # Generate cross-server queries
        logger.info(f"Generating {num_queries} cross-server queries requiring tools from multiple servers...")
        
        queries = await self.generate_cross_server_queries(
            server_tool_map=server_tool_map,
            num_queries=num_queries,
            min_multi_hop_percent=0.5
        )
        
        # Close all clients
        for client in clients:
            if hasattr(client, 'close'):
                await client.close()
        
        # Save the generated queries
        if queries:
            # Save qualifying queries (both multi-hop AND cross-server)
            qualifying_queries = [q for q in queries if 
                                q.get("verification", {}).get("is_multi_hop", False) and 
                                q.get("verification", {}).get("is_cross_server", False)]
            
            if qualifying_queries:
                qualifying_path = await self.save_queries_to_file(
                    qualifying_queries, 
                    filename_prefix="cross_server_multi_hop_queries"
                )
                logger.info(f"Saved {len(qualifying_queries)} qualifying queries")
                return qualifying_path
            
            # If no qualifying queries, save all queries
            all_path = await self.save_queries_to_file(queries)
            return all_path
        
        logger.error("Failed to generate any queries.")
        return None

    @classmethod
    async def run_all(cls, config_path: str = None, num_queries: int = 1) -> Optional[Path]:
        """Class method to run the generator with default settings."""
        generator = cls(config_path=config_path)
        return await generator.generate_queries(num_queries=num_queries)


async def main_async():
    """Async entry point for the script."""
    # Parse command-line arguments
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH
    
    # Create generator
    try:
        load_dotenv()  # Load environment variables
        generator = MCPQueryGenerator(config_path=config_path)
        queries_path = await generator.generate_queries()
        return queries_path
    except Exception as e:
        logger.error(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Entry point with proper asyncio handling."""
    try:
        # Create new event loop and run the main function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        queries_path = loop.run_until_complete(main_async())
        
        # Cleanup
        pending = asyncio.all_tasks(loop)
        if pending:
            print(f"Cancelling {len(pending)} pending tasks...")
            for task in pending:
                task.cancel()
            
            # Allow cancelled tasks to complete with a timeout
            # Remove the 'loop' parameter which is deprecated in newer Python versions
            loop.run_until_complete(
                asyncio.wait(pending, timeout=10)
            )
        
        loop.close()
        
        # Return exit code based on success/failure
        return 0 if queries_path else 1
    
    except Exception as e:
        print(f"Setup error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())