#!/usr/bin/env python3
"""
MCP Multi-Hop Query Generator and Verifier (Cross-Server Version)
Generates queries that require tools from multiple MCP server types and verifies multi-hop reasoning requirements
"""
import asyncio
import json
import os
import re
import sys
import itertools
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Set, Optional

import openai
from langchain_openai import ChatOpenAI
from mcp_use.client import MCPClient
from mcp_use.adapters.langchain_adapter import LangChainAdapter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set OpenRouter API key and base URL
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")  # Default to OpenRouter URL
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY missing in .env or environment")

# Default path to the config file
DEFAULT_CONFIG_PATH = Path(r"D:\one drive\study\ARCEE AI INTERNSHIP\mcp data gen minimal\mcp-dataset-generator\tests\mcp_servers.json")

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


class QueryGenerator:
    """
    Generate queries tailored to specific MCP server types and their tools.
    """
    
    def __init__(self, openai_api_key=None, openai_api_base=None):
        """Initialize the query generator."""
        self.api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.api_base = openai_api_base or os.environ.get("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.api_base
        )
        
        # Initialize verifier
        self.verifier = MultiHopVerifier()
    
    async def generate_cross_server_queries(self, 
                                            server_tool_map: Dict[str, List], 
                                            num_queries: int = 5,
                                            min_multi_hop_percent: float = 0.5) -> List[Dict]:
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
                print(f"Warning: Standard API call failed, trying alternative: {e}")
                # Try with direct API call as a fallback (older OpenAI client versions)
                try:
                    response = openai.ChatCompletion.create(
                        api_key=self.api_key,
                        api_base=self.api_base,
                        **params
                    )
                    content = response['choices'][0]['message']['content']
                except Exception as e2:
                    print(f"Error: Both API call methods failed: {e2}")
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
                    
                    print(f"Generated {len(verified_queries)} cross-server queries")
                    print(f"Multi-hop queries: {multi_hop_count} ({multi_hop_percent:.1%})")
                    print(f"Cross-server queries: {cross_server_count} ({cross_server_percent:.1%})")
                    
    # If we don't have enough multi-hop or cross-server queries, try to generate more
                    if (multi_hop_percent < min_multi_hop_percent or cross_server_percent < min_multi_hop_percent) and len(verified_queries) > 0:
                        print(f"Insufficient multi-hop or cross-server queries. Regenerating...")
                        
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
                            print(f"Error generating additional queries: {e}")
                        
                    return verified_queries
                else:
                    print(f"No JSON array found in response")
                    print(f"Content: {content[:200]}...")
                    return []
                    
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON: {e}")
                print(f"Content: {content[:200]}...")
                return []
                
        except Exception as e:
            print(f"Error generating queries: {e}")
            return []
    
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
You are an expert query generator creating CROSS-SERVER MULTI-HOP queries for an AI assistant that uses tools from multiple MCP (Model Context Protocol) servers.

You are generating queries that will require tools from these MCP servers:
{', '.join(server_names)}

{categorized_tools}

Your task is to generate {num_queries} complex, realistic queries that will require an AI assistant to use tools from MULTIPLE DIFFERENT SERVERS to complete the task properly.

## CROSS-SERVER REQUIREMENTS:
1. Each query MUST require tools from AT LEAST 2 DIFFERENT SERVERS to solve properly
2. Each query should naturally flow between different servers' tools in a logical sequence
3. The need to use multiple servers should be IMPLICIT in the task, not explicitly stated
4. A good cross-server query will feel like a single cohesive task to the user but require different server capabilities

## INHERENT COMPLEXITY REQUIREMENTS:
1. Create queries that NATURALLY require multiple tool calls across different servers
2. The complexity should come from the TASK ITSELF, not from artificial language complexity
3. Queries should require gathering, processing, and synthesizing information from multiple sources
4. DO NOT use explicit sequencing words like "first," "then," "next," etc.
5. Each logical step should REQUIRE a DIFFERENT TOOL, preferably from DIFFERENT SERVERS

## QUERY CHARACTERISTICS:
1. Single, natural requests that real users would ask but require multiple tools to answer
2. Concise phrasing with implicit multi-step requirements
3. Queries should sound conversational and straightforward while requiring complex processing
4. Should reference capabilities across different servers in a natural way

## EXAMPLES OF GOOD CROSS-SERVER QUERIES:
- "Find the cheapest flight to Paris next month and save the details to my travel plans folder"
- "What's the weather forecast for Chicago this weekend, and email me a summary with photos of top indoor attractions"
- "Check my calendar for free time next week and book me a haircut appointment nearby"
- "Show me cryptocurrency prices for the coins mentioned in the latest financial newsletter in my email"
- "Find images of modern kitchen designs, create a mood board and schedule time with an interior designer"
- "Find a popular vegan recipe online and add the ingredients to my shopping list document"
- "Check for nearby coffee shops with wifi and create a route map to the highest-rated one"
- "Calculate how much I would save by switching to solar power based on my last electricity bill in my documents folder"
- "Find the latest research on quantum computing and create a summary document with the key findings"
- "Check my calendar and suggest flight options that fit around my upcoming meetings"

## SERVER TOOLS AVAILABLE:
{', '.join([f"SERVER: {server}\n{desc}" for server, desc in server_tool_descriptions.items()])}

Return the queries as a JSON array with this format:
[
  {{
    "query": "The cross-server query that requires tools from multiple MCP servers",
    "expected_sub_queries": ["Logical step 1 requiring server A's tool X", "Logical step 2 requiring server B's tool Y", "Logical step 3 requiring server C's tool Z", ...],
    "expected_tools": ["tool1", "tool2", "tool3", ...],
    "expected_servers": ["server1", "server2", ...],
    "tool_sequence": "Detailed explanation of how tools must be used in sequence across different servers: server1's tool1 to get initial data, server2's tool2 to process it in a specific way, server3's tool3 to perform further targeted operations...",
    "rationale": "Explanation of why this query inherently requires multiple servers, focusing on the logical dependencies between steps and server capabilities"
  }}
]

IMPORTANT:
- Each query MUST require tools from at least 2 DIFFERENT SERVERS
- Create queries that SOUND SIMPLE but require multiple server tools to solve properly
- Make queries that feel natural and conversational like real user questions
- ONLY reference tools that are actually available from the provided servers
- Ensure the expected_tools and expected_servers lists specifically name the actual tools/servers needed
"""
        return system_prompt
    
    def _infer_server_type(self, server_name: str) -> str:
        """
        Infer the server type from its name for better context in the prompt.
        """
        server_lower = server_name.lower()
        
        # Map server names to likely types
        if any(term in server_lower for term in ["playwright", "browser", "web"]):
            return "web browser automation"
        elif any(term in server_lower for term in ["search", "google", "bing"]):
            return "web search"
        elif any(term in server_lower for term in ["file", "filesystem"]):
            return "file system access"
        elif any(term in server_lower for term in ["math", "calc", "calculator"]):
            return "mathematical calculation"
        elif any(term in server_lower for term in ["time", "date", "clock"]):
            return "time and date information"
        elif any(term in server_lower for term in ["airbnb", "hotel", "booking"]):
            return "accommodation booking"
        elif any(term in server_lower for term in ["coincap", "coin", "crypto", "finance"]):
            return "cryptocurrency information"
        elif any(term in server_lower for term in ["pollinations", "image", "art"]):
            return "image generation"
        else:
            return "specialized tool"


def load_config(config_path: Path) -> Dict:
    """Load and parse an MCP configuration file"""
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        return {}
    
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        return config_data
    except Exception as e:
        print(f"Error reading config file: {e}")
        return {}


def list_servers(config_data: Dict) -> List[str]:
    """Extract server names from configuration data"""
    servers = []
    if "mcpServers" in config_data:
        servers = list(config_data["mcpServers"].keys())
    return servers


def select_servers(servers: List[str]) -> List[str]:
    """Allow user to select which servers to use"""
    if not servers:
        print("No servers found in configuration!")
        return []
    
    print("\nAvailable MCP Servers:")
    for i, server in enumerate(servers, 1):
        print(f"{i}. {server}")
    
    print(f"\n{len(servers) + 1}. All servers")
    
    while True:
        choice = input("\nEnter server number(s) to use (comma-separated, or 'all'): ")
        
        if choice.lower() in ('all', 'a', str(len(servers) + 1)):
            print(f"Selected all {len(servers)} servers")
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
            
            print(f"Selected servers: {', '.join(selected_servers)}")
            return selected_servers
            
        except ValueError:
            print("Invalid input. Please enter numbers separated by commas.")


def create_server_specific_config(config_data: Dict, server_name: str) -> Dict:
    """Create a configuration with only a single server"""
    if "mcpServers" not in config_data or server_name not in config_data["mcpServers"]:
        return {}
    
    return {
        "mcpServers": {
            server_name: config_data["mcpServers"][server_name]
        }
    }


async def get_mcp_tools_for_server(config_data: Dict, server_name: str) -> Tuple[List, Any]:
    """
    Retrieves tools from a specific MCP server.
    
    Args:
        config_data: The full configuration data
        server_name: Name of the server to connect to
        
    Returns:
        Tuple of (tools, client)
    """
    # Create server-specific config
    server_config = create_server_specific_config(config_data, server_name)
    
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
        print(f"Successfully retrieved {len(tools)} tools from {server_name} server")
        
        # Print tool names
        if tools:
            tool_names = [tool.name for tool in tools]
            print(f"Available tools from {server_name}: {', '.join(tool_names[:5])}...")
        
        return tools, client
    except Exception as e:
        print(f"Error retrieving tools from {server_name}: {e}")
        import traceback
        traceback.print_exc()
        return [], None
    finally:
        # Clean up temporary file
        if temp_config_path.exists():
            temp_config_path.unlink()


async def save_queries_to_file(queries: List[Dict], output_dir: str, filename: str) -> str:
    """Save the generated queries to a JSON file."""
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output file path
    output_file = output_dir / filename
    
    # Write entries to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(queries, f, indent=2)
    
    print(f"Saved {len(queries)} queries to {output_file}")

    return str(output_file)


async def generate_cross_server_queries():
    """Generate multi-hop queries that require tools from multiple MCP servers."""
    # Allow specifying config path as command-line argument
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH
    
    print(f"Loading MCP server configuration from: {config_path}")
    
    # Load the configuration
    config_data = load_config(config_path)
    
    if not config_data:
        print("Failed to load configuration. Exiting.")
        return
    
    # List available servers
    servers = list_servers(config_data)
    
    if not servers:
        print("No MCP servers found in configuration. Exiting.")
        return
    
    # Allow user to select servers
    selected_servers = select_servers(servers)
    
    if len(selected_servers) < 2:
        print("At least 2 servers are required for cross-server queries. Exiting.")
        return
    
    # Get number of queries to generate
    while True:
        try:
            num_queries = int(input("\nNumber of cross-server queries to generate: "))
            if num_queries <= 0:
                print("Please enter a positive number.")
                continue
            break
        except ValueError:
            print("Please enter a valid number.")
    
    # Collect tools from all selected servers
    server_tool_map = {}
    clients = []  # Keep track of clients to close them later
    
    for server_name in selected_servers:
        print(f"\nConnecting to server: {server_name}")
        
        # Get tools from this server
        tools, client = await get_mcp_tools_for_server(config_data, server_name)
        if client:
            clients.append(client)
        
        if not tools:
            print(f"No tools available from {server_name}. Skipping.")
            continue
        
        server_tool_map[server_name] = tools
    
    if len(server_tool_map) < 2:
        print("Could not retrieve tools from at least 2 servers. Exiting.")
        return
    
    # Initialize the query generator
    generator = QueryGenerator(openai_api_key=OPENAI_API_KEY, openai_api_base=OPENAI_API_BASE)
    
    # Generate cross-server queries
    print(f"\nGenerating {num_queries} cross-server queries requiring tools from multiple servers...")
    
    queries = await generator.generate_cross_server_queries(
        server_tool_map=server_tool_map,
        num_queries=num_queries,
        min_multi_hop_percent=0.5  # Reduced from 0.7 to 0.5
    )
    
    # Close all clients
    for client in clients:
        if hasattr(client, 'close'):
            await client.close()
    
    # Save the generated queries
    if queries:
        output_dir = "queries"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"cross_server_queries_{timestamp}.json"
        
        output_file = await save_queries_to_file(
            queries=queries,
            output_dir=output_dir,
            filename=filename
        )
        
        # Print final summary
        total_queries = len(queries)
        multi_hop_queries = sum(1 for q in queries if q.get("verification", {}).get("is_multi_hop", False))
        cross_server_queries = sum(1 for q in queries if q.get("verification", {}).get("is_cross_server", False))
        qualifying_queries = sum(1 for q in queries if q.get("verification", {}).get("is_multi_hop", False) and 
                                q.get("verification", {}).get("is_cross_server", False))
        
        print(f"\n{'='*50}")
        print(f"Generation Summary")
        print(f"{'='*50}")
        print(f"Total queries generated: {total_queries}")
        print(f"Multi-hop queries: {multi_hop_queries} ({multi_hop_queries/total_queries:.1%})")
        print(f"Cross-server queries: {cross_server_queries} ({cross_server_queries/total_queries:.1%})")
        print(f"Qualifying queries (both multi-hop AND cross-server): {qualifying_queries} ({qualifying_queries/total_queries:.1%})")
        print(f"Queries saved to: {output_file}")
        
        # Save qualifying queries separately
        if qualifying_queries > 0:
            qualifying_only = [q for q in queries if q.get("verification", {}).get("is_multi_hop", False) and 
                              q.get("verification", {}).get("is_cross_server", False)]
            qualifying_filename = f"cross_server_multi_hop_queries_{timestamp}.json"
            
            qualifying_file = await save_queries_to_file(
                queries=qualifying_only,
                output_dir=output_dir,
                filename=qualifying_filename
            )
            
            print(f"Qualifying cross-server multi-hop queries saved to: {qualifying_file}")
    else:
        print("\nFailed to generate any cross-server queries.")


if __name__ == "__main__":
    asyncio.run(generate_cross_server_queries())