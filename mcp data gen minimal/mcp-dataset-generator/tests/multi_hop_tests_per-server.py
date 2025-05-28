#!/usr/bin/env python3
"""
MCP Multi-Hop Query Generator and Verifier
Generates queries specific to MCP server types and verifies multi-hop reasoning requirements
"""
import asyncio
import json
import os
import re
import sys
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
    
    def __init__(self):
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
    
    async def generate_server_specific_queries(self, 
                                              server_name: str, 
                                              tools: List, 
                                              num_queries: int = 5,
                                              min_multi_hop_percent: float = 0.8) -> List[Dict]:
        """
        Generate queries specifically tailored to a server type and its tools.
        
        Args:
            server_name: Name of the MCP server
            tools: List of tools available for this server
            num_queries: Number of queries to generate
            min_multi_hop_percent: Minimum percentage of queries that should be multi-hop
            
        Returns:
            List of query entries with verification metadata
        """
        # Extract tool information
        tool_descriptions = "\n".join([
            f"- {tool.name}: {tool.description}" 
            for tool in tools
        ])
        
        # Determine server type from name for better context
        server_type = self._infer_server_type(server_name)
        
        # Create a system prompt that explains the task
        # Create a system prompt that explains the task
        system_prompt = f"""
You are an expert query generator creating INHERENTLY COMPLEX queries for an AI assistant that uses tools from an MCP (Model Context Protocol) server.

You are generating queries for the "{server_name}" MCP server which appears to be a {server_type} server.
This server provides these tools:
{tool_descriptions}

Your task is to generate {num_queries} complex, realistic queries that will require an AI assistant to break down into multiple operations without being explicitly told to do so.

## INHERENT COMPLEXITY REQUIREMENTS:
1. Create queries that NATURALLY require multiple tool calls to complete properly
2. The complexity should come from the TASK ITSELF, not from artificial language complexity
3. Queries should require gathering, processing, and synthesizing information from multiple sources
4. The need for multiple steps should be IMPLICIT in what the user is asking for
5. DO NOT use explicit sequencing words like "first," "then," "next," etc.
6. Each logical step should REQUIRE a DIFFERENT TOOL to solve

## QUERY CHARACTERISTICS:
1. Single, natural requests that real users would ask but require multiple tools to answer
2. Concise phrasing with implicit multi-step requirements
3. Queries should sound conversational and straightforward while requiring complex processing
4. Should require at least 2-4 different tools to complete properly

## EXAMPLES OF NATURALLY COMPLEX QUERIES:
- "What is the cheapest pizza closest to Airbnb HQ?"
- "Restaurants near St. Andrew station, Toronto serving nachos"
- "Best coffee shops within walking distance of Central Park with free wifi"
- "Hotels in Vancouver with availability next weekend under $200"
- "What's the weather forecast for Chicago this weekend compared to New York?"
- "Apartments for rent near University of Toronto with in-unit laundry"
- "Vegan restaurants with outdoor seating in downtown Seattle"
- "Which flights from Toronto to Miami next month have the best price-to-duration ratio?"
- "Tech companies with job openings for software engineers in Austin"
- "Movie theaters showing the new Marvel film this weekend within 5 miles"
- "Italian restaurants in Boston's North End with 4+ star ratings and reservations available Friday night"
- "Best-selling fiction books this month with ratings above 4.5 stars"
- "Women's running shoes with the most positive reviews for plantar fasciitis"
- "Electric cars with over 300 miles of range and available tax incentives"
- "Dog-friendly hiking trails within 30 minutes of Portland with moderate difficulty"
- "Stocks in the healthcare sector with the highest dividend yields"
- "Concerts in Los Angeles next month with tickets under $75"
- "Best-rated sushi restaurants that deliver to the Financial District"
- "Online programming courses for beginners with the highest completion rates"
- "Used Toyota SUVs within 50 miles with less than 50,000 miles and under $25,000"

## SERVER AND TOOL CONTEXT:
1. Include context about the server and its tools in a natural way
2. This context should feel like part of the query, not a separate instruction

Return the queries as a JSON array with this format:
[
  {{
    "query": "The inherently complex query that naturally requires multiple operations",
    "expected_sub_queries": ["Logical step 1 requiring tool X", "Logical step 2 requiring tool Y", "Logical step 3 requiring tool Z", ...],
    "expected_tools": ["tool1", "tool2", "tool3", ...],
    "tool_sequence": "Detailed explanation of how tools must be used in sequence: tool1 to get initial data, tool2 to process it in a specific way, tool3 to perform further targeted operations...",
    "rationale": "Explanation of why this query inherently requires multiple operations with different tools, focusing on the logical dependencies between steps",
    "server_info": "Description of how the server type and specific tools are needed for this particular query"
  }}
]

IMPORTANT:
- Create queries that SOUND SIMPLE but require multiple tools to solve properly
- Each logical step must naturally require a DIFFERENT tool to complete
- Make queries that feel natural and conversational like real user questions
- ONLY reference tools from the list provided for this specific MCP server
- Ensure the query is something a real user would actually ask
- Make sure the expected_sub_queries represent the distinct tool operations needed
"""

        # Generate queries with OpenAI
        try:
            # Set up the parameters for the API call
            params = {
                "model": "openai/gpt-4o",  # Updated model name with provider prefix for OpenRouter
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Generate {num_queries} diverse multi-hop queries specifically for the {server_name} MCP server using only the tools listed."}
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
                    
                    for i, entry in enumerate(generated_queries):
                        query = entry.get("query", "")
                        is_multi_hop, explanation, sub_queries = self.verifier.is_multi_hop_query(query)
                        
                        # Add verification information
                        entry["verification"] = {
                            "is_multi_hop": is_multi_hop,
                            "explanation": explanation,
                            "potential_sub_queries": sub_queries
                        }
                        
                        # Add server info
                        entry["server"] = server_name
                        entry["id"] = f"{server_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{i}"
                        
                        verified_queries.append(entry)
                        
                        if is_multi_hop:
                            multi_hop_count += 1
                    
                    # Calculate multi-hop percentage
                    multi_hop_percent = multi_hop_count / len(verified_queries) if verified_queries else 0
                    
                    print(f"Generated {len(verified_queries)} queries for {server_name}")
                    print(f"Multi-hop queries: {multi_hop_count} ({multi_hop_percent:.1%})")
                    
                    # If we don't have enough multi-hop queries, try to generate more
                    if multi_hop_percent < min_multi_hop_percent and len(verified_queries) > 0:
                        print(f"Insufficient multi-hop queries. Regenerating to achieve at least {min_multi_hop_percent:.1%} multi-hop...")
                        # Recursive call to generate more queries
                        more_queries = await self.generate_server_specific_queries(
                            server_name=server_name,
                            tools=tools,
                            num_queries=max(5, int(num_queries * (min_multi_hop_percent - multi_hop_percent) / min_multi_hop_percent) + 1),
                            min_multi_hop_percent=min_multi_hop_percent
                        )
                        
                        # Add only the multi-hop queries
                        multi_hop_queries = [q for q in more_queries if q.get("verification", {}).get("is_multi_hop", False)]
                        verified_queries.extend(multi_hop_queries)
                        
                    return verified_queries
                else:
                    print(f"No JSON array found in response from {server_name}")
                    print(f"Content: {content[:200]}...")
                    return []
                    
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON: {e}")
                print(f"Content: {content[:200]}...")
                return []
                
        except Exception as e:
            print(f"Error generating queries: {e}")
            return []
    
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


async def generate_multi_hop_queries():
    """Generate multi-hop queries for selected MCP servers."""
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
    
    if not selected_servers:
        print("No servers selected. Exiting.")
        return
        
    # Get number of queries to generate per server
    while True:
        try:
            queries_per_server = int(input("\nNumber of queries to generate per server: "))
            if queries_per_server <= 0:
                print("Please enter a positive number.")
                continue
            break
        except ValueError:
            print("Please enter a valid number.")
    
    # Initialize the query generator
    generator = QueryGenerator(openai_api_key=OPENAI_API_KEY, openai_api_base=OPENAI_API_BASE)
    
    # Generate queries for each server
    all_queries = []
    clients = []  # Keep track of clients to close them later
    
    for server_name in selected_servers:
        print(f"\n{'='*50}")
        print(f"Processing server: {server_name}")
        print(f"{'='*50}")
        
        # Get tools from this server
        tools, client = await get_mcp_tools_for_server(config_data, server_name)
        if client:
            clients.append(client)
        
        if not tools:
            print(f"No tools available from {server_name}. Skipping.")
            continue
        
        # Generate queries for this server
        print(f"Generating {queries_per_server} queries for {server_name}...")
        
        queries = await generator.generate_server_specific_queries(
            server_name=server_name,
            tools=tools,
            num_queries=queries_per_server,
            min_multi_hop_percent=0.5  # At least 80% should be multi-hop
        )
        
        # Add to overall collection
        all_queries.extend(queries)
        
        # Print a summary for this server
        multi_hop_count = sum(1 for q in queries if q.get("verification", {}).get("is_multi_hop", False))
        print(f"Generated {len(queries)} queries for {server_name}")
        print(f"Multi-hop queries: {multi_hop_count} ({multi_hop_count/len(queries):.1%})")
    
    # Close all clients
    for client in clients:
        if hasattr(client, 'close'):
            await client.close()
    
    # Save the combined queries
    if all_queries:
        output_dir = "queries"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"multi_hop_queries_{timestamp}.json"
        
        output_file = await save_queries_to_file(
            queries=all_queries,
            output_dir=output_dir,
            filename=filename
        )
        
        # Print final summary
        total_queries = len(all_queries)
        multi_hop_queries = sum(1 for q in all_queries if q.get("verification", {}).get("is_multi_hop", False))
        
        print(f"\n{'='*50}")
        print(f"Generation Summary")
        print(f"{'='*50}")
        print(f"Total queries generated: {total_queries}")
        print(f"Multi-hop queries: {multi_hop_queries} ({multi_hop_queries/total_queries:.1%})")
        print(f"Queries saved to: {output_file}")
        
        # Save multi-hop queries separately
        if multi_hop_queries > 0:
            multi_hop_only = [q for q in all_queries if q.get("verification", {}).get("is_multi_hop", False)]
            multi_hop_filename = f"multi_hop_queries_verified_{timestamp}.json"
            
            multi_hop_file = await save_queries_to_file(
                queries=multi_hop_only,
                output_dir=output_dir,
                filename=multi_hop_filename
            )
            
            print(f"Verified multi-hop queries saved to: {multi_hop_file}")
    else:
        print("\nFailed to generate any queries from the selected servers.")


if __name__ == "__main__":
    asyncio.run(generate_multi_hop_queries())