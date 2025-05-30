"""
multi_hop_verification.py - Verification module for multi-hop queries
Focuses on linguistic analysis to determine if queries require multiple steps
"""
import re
from typing import Dict, List, Any, Tuple, Set

class MultiHopVerifier:
    """
    Verifier that analyzes queries to determine if they require multi-hop reasoning.
    Focuses on linguistic structure to identify if queries can be broken down
    into atomic sub-queries that must be executed in sequence.
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
            "besides", "plus", "along with", "as well as"
        }
        
        # Words indicating comparison or relationship
        self.comparison_indicators = {
            "compare", "comparison", "versus", "vs", "difference", 
            "similarities", "better", "worse", "more", "less", 
            "between", "relationship", "correlation", "connection"
        }
        
        # Words indicating cause-effect relationships
        self.causal_indicators = {
            "because", "cause", "effect", "result", "impact",
            "influence", "affects", "leads to", "results in"
        }
        
        # Information-gathering action verbs
        self.info_gathering_verbs = {
            "find", "search", "locate", "identify", "discover",
            "check", "determine", "calculate", "analyze", "look up"
        }
        
        # Action verbs that suggest tool operations
        self.action_verbs = {
            "get", "retrieve", "fetch", "obtain", "extract",
            "download", "open", "navigate", "visit", "browse",
            "search", "read", "write", "save", "edit", "modify",
            "create", "delete", "remove", "update", "calculate"
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
        has_multiple_conjunctions = len(re.findall(r'\band\b|\balso\b', query_lower)) > 1
        has_comparison = self._contains_words_from_set(query_lower, self.comparison_indicators)
        has_causal_relation = self._contains_words_from_set(query_lower, self.causal_indicators)
        
        # Check for multiple action verbs
        action_verbs_present = self._find_words_from_set(query_lower, self.action_verbs)
        info_verbs_present = self._find_words_from_set(query_lower, self.info_gathering_verbs)
        all_verbs_present = action_verbs_present.union(info_verbs_present)
        has_multiple_verbs = len(all_verbs_present) >= 2
        
        # Check for sentences that could be independent sub-queries
        sentences = self._split_into_sentences(query)
        has_multiple_sentences = len(sentences) > 1
        
        # Attempt to extract potential sub-queries
        potential_sub_queries = self._extract_potential_sub_queries(query)
        has_extractable_sub_queries = len(potential_sub_queries) > 1
        
        # Decision logic
        factors = []
        if has_sequence_indicators:
            factors.append("contains sequencing words indicating ordered steps")
        if has_multiple_questions:
            factors.append("contains multiple questions that require separate answers")
        if has_multiple_conjunctions:
            factors.append("contains multiple conjunctions connecting different requirements")
        if has_comparison:
            factors.append("involves comparison which typically requires gathering multiple pieces of information")
        if has_causal_relation:
            factors.append("involves cause-effect analysis requiring multiple investigation steps")
        if has_multiple_verbs:
            factors.append(f"contains multiple action verbs ({', '.join(list(all_verbs_present)[:3])}) suggesting different operations")
        if has_multiple_sentences and has_extractable_sub_queries:
            factors.append("can be broken down into distinct sub-queries")
        
        # Make final determination
        is_multi_hop = (has_sequence_indicators or 
                        has_multiple_questions or 
                        (has_multiple_conjunctions and (has_comparison or has_causal_relation)) or 
                        (has_multiple_verbs and has_extractable_sub_queries))
        
        # Generate explanation
        if is_multi_hop:
            if factors:
                explanation = "This appears to be a multi-hop query because it " + "; it ".join(factors) + "."
            else:
                explanation = "This query likely requires multiple steps to complete, based on its complexity."
        else:
            explanation = "This appears to be a single-hop query as it doesn't show clear indications of requiring sequential steps."
        
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

def verify_dataset_queries(entries: List[Dict]) -> Dict:
    """
    Verify all queries in a dataset and return statistics.
    
    Args:
        entries: List of dataset entries
        
    Returns:
        Dictionary with verification statistics
    """
    verifier = MultiHopVerifier()
    multi_hop_count = 0
    verification_results = []
    
    for i, entry in enumerate(entries):
        query = entry.get("question", "")
        is_multi_hop, explanation, sub_queries = verifier.is_multi_hop_query(query)
        
        # Record the result
        verification_results.append({
            "entry_id": i,
            "query": query,
            "is_multi_hop": is_multi_hop,
            "explanation": explanation,
            "potential_sub_queries": sub_queries
        })
        
        if is_multi_hop:
            multi_hop_count += 1
    
    # Generate statistics
    stats = {
        "total_entries": len(entries),
        "multi_hop_entries": multi_hop_count,
        "multi_hop_percentage": round(multi_hop_count / len(entries) * 100, 2) if entries else 0,
        "verification_results": verification_results
    }
    
    return stats

# Utility function to demonstrate the verifier
def test_verifier(queries: List[str]) -> None:
    """
    Test the verifier with a list of queries.
    
    Args:
        queries: List of queries to test
    """
    verifier = MultiHopVerifier()
    
    print("\n" + "="*80)
    print("MULTI-HOP QUERY VERIFICATION TEST")
    print("="*80)
    
    for i, query in enumerate(queries, 1):
        is_multi_hop, explanation, sub_queries = verifier.is_multi_hop_query(query)
        
        print(f"\nQuery {i}: {query}")
        print(f"Multi-hop: {'YES' if is_multi_hop else 'NO'}")
        print(f"Explanation: {explanation}")
        
        if len(sub_queries) > 1:
            print("Potential sub-queries:")
            for j, sub in enumerate(sub_queries, 1):
                print(f"  {j}. {sub}")
        
        print("-"*80)

# Example usage
if __name__ == "__main__":
    test_queries = [
        "What is the weather in New York?",
        "Find the population of Tokyo and compare it with the population of Shanghai.",
        "Search for the best Italian restaurant in Chicago, then check if they have reservations available for tomorrow evening.",
        "What's the stock price of Apple and how has it changed over the past month?",
        "Find the tallest building in Dubai and tell me when it was completed.",
        "Look up the recipe for chocolate chip cookies and calculate how many calories are in each cookie."
    ]
    
    test_verifier(test_queries)