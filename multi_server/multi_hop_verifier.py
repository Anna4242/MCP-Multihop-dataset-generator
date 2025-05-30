"""
multi_hop_verification.py - Verification module for multi-hop reasoning in MCP tool sequences
"""
import re
from typing import Dict, List, Any, Tuple
from datetime import datetime

class MultiHopVerifier:
    """Verifies if a tool sequence demonstrates true multi-hop reasoning."""
    
    def __init__(self, min_tools=2, min_unique_tools=2):
        """
        Initialize the verifier with configuration parameters.
        
        Args:
            min_tools: Minimum number of total tool calls required (default: 2)
            min_unique_tools: Minimum number of unique tools required (default: 2)
        """
        self.min_tools = min_tools
        self.min_unique_tools = min_unique_tools
        
    def verify(self, entry: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if an entry demonstrates multi-hop reasoning.
        
        Args:
            entry: Dictionary containing query and tool sequence
            
        Returns:
            Tuple of (is_valid, results_dict)
        """
        # Initialize verification results
        results = {
            "is_multi_hop": False,
            "checks": {
                "sufficient_tools": False,
                "tool_diversity": False,
                "sequential_dependency": False,
                "information_flow": False
            },
            "explanation": [],
            "improvements": []
        }
        
        # Extract necessary information
        query = entry.get("question", "")
        
        # Extract tool sequence from the rule code
        rule_code = entry.get("extra_info", {}).get("rule", "")
        tool_sequence = self._extract_tool_sequence(rule_code)
        
        if not tool_sequence:
            results["explanation"].append("Could not extract tool sequence from rule code")
            return False, results
        
        # Check 1: Sufficient number of tools
        if len(tool_sequence) >= self.min_tools:
            results["checks"]["sufficient_tools"] = True
        else:
            results["explanation"].append(
                f"Only {len(tool_sequence)} tools used (minimum required: {self.min_tools})"
            )
            results["improvements"].append(
                "Add more tool calls to create a proper multi-step process"
            )
        
        # Check 2: Tool diversity
        unique_tools = set(step["tool"] for step in tool_sequence)
        if len(unique_tools) >= self.min_unique_tools:
            results["checks"]["tool_diversity"] = True
        else:
            results["explanation"].append(
                f"Only {len(unique_tools)} unique tools used (minimum required: {self.min_unique_tools})"
            )
            results["improvements"].append(
                "Use a more diverse set of tools"
            )
        
        # Check 3: Sequential dependency
        dependency_score = self._check_sequential_dependency(tool_sequence)
        if dependency_score >= 0.5:  # At least half of the steps show dependency
            results["checks"]["sequential_dependency"] = True
        else:
            results["explanation"].append(
                "Tool calls don't show clear sequential dependency"
            )
            results["improvements"].append(
                "Ensure each tool uses information obtained from previous tools"
            )
        
        # Check 4: Information flow
        info_flow_score = self._check_information_flow(rule_code)
        if info_flow_score >= 0.5:
            results["checks"]["information_flow"] = True
        else:
            results["explanation"].append(
                "Insufficient evidence that information flows between tool calls"
            )
            results["improvements"].append(
                "Make explicit how data from one tool is used in subsequent tools"
            )
        
        # Final determination - require at least 3 out of 4 checks to pass
        passing_checks = sum(1 for check in results["checks"].values() if check)
        results["is_multi_hop"] = passing_checks >= 3
        
        if results["is_multi_hop"]:
            results["explanation"].append(
                "Entry demonstrates genuine multi-hop reasoning with proper tool sequencing"
            )
        
        return results["is_multi_hop"], results
    
    def _extract_tool_sequence(self, rule_code: str) -> List[Dict]:
        """Extract the tool sequence from the rule code."""
        # Look for tool calls in the code
        # This is a simplified approach - adjust to match your actual code patterns
        tool_pattern = r'(?:await|result\s*=\s*await)\s+(\w+)\s*\(([^)]*)\)'
        tool_calls = re.findall(tool_pattern, rule_code)
        
        sequence = []
        for i, (tool, params) in enumerate(tool_calls, 1):
            # Extract parameters if possible
            param_dict = {}
            if params.strip():
                try:
                    # Try to parse simple key-value pairs
                    for param in params.split(','):
                        if '=' in param:
                            k, v = param.split('=', 1)
                            param_dict[k.strip()] = v.strip().strip('"\'')
                except Exception:
                    # If parsing fails, just store the raw params
                    param_dict = {"raw_params": params.strip()}
            
            sequence.append({
                "step": i,
                "tool": tool,
                "parameters": param_dict
            })
        
        return sequence
    
    def _check_sequential_dependency(self, tool_sequence: List[Dict]) -> float:
        """
        Check for sequential dependency between tool calls.
        Returns a score between 0 and 1.
        """
        if len(tool_sequence) < 2:
            return 0
        
        # Count potential dependencies
        dependency_indicators = 0
        possible_dependencies = len(tool_sequence) - 1
        
        for i in range(1, len(tool_sequence)):
            current = tool_sequence[i]
            previous = tool_sequence[i-1]
            
            # Check for typical dependency patterns
            if previous["tool"].endswith("_search") and (
                current["tool"].endswith("_details") or 
                current["tool"].endswith("_info")
            ):
                # Search followed by details lookup
                dependency_indicators += 1
            
            elif previous["tool"].endswith("_list") and current["tool"].endswith("_get"):
                # List followed by get
                dependency_indicators += 1
            
            elif previous["tool"] == "browser_snapshot" and current["tool"] in [
                "browser_click", "browser_type"
            ]:
                # Snapshot followed by interaction
                dependency_indicators += 1
            
            # Add more patterns specific to your tools
        
        return dependency_indicators / possible_dependencies
    
    def _check_information_flow(self, rule_code: str) -> float:
        """
        Check for variable usage that indicates information flow.
        Returns a score between 0 and 1.
        """
        # Look for patterns where output from one call is used as input to another
        # This is a simplified approach - adjust to match your actual code patterns
        
        # Check for variable assignments from tool calls
        assignments = re.findall(r'(\w+)\s*=\s*await\s+\w+\s*\([^)]*\)', rule_code)
        
        # Check for those variables being used in subsequent calls
        usage_count = 0
        for var in assignments:
            # Look for the variable being used in a later tool call
            if re.search(rf'await\s+\w+\s*\([^)]*{var}[^)]*\)', rule_code):
                usage_count += 1
        
        # Calculate score based on how many variables are reused
        if not assignments:
            return 0
        
        return usage_count / len(assignments)


def generate_verification_stats(entries: List[Dict]) -> Dict:
    """Generate statistics about the verification results."""
    total = len(entries)
    verified = sum(1 for e in entries if e.get("extra_info", {}).get("verification", {}).get("is_valid_multi_hop", False))
    
    # Count which checks failed most often
    failed_checks = {
        "sufficient_tools": 0,
        "tool_diversity": 0,
        "sequential_dependency": 0,
        "information_flow": 0
    }
    
    for entry in entries:
        verification = entry.get("extra_info", {}).get("verification", {})
        checks = verification.get("checks", {})
        
        for check, passed in checks.items():
            if not passed:
                failed_checks[check] = failed_checks.get(check, 0) + 1
    
    # Calculate most common reasons for verification failure
    most_common_failures = sorted(
        [(check, count) for check, count in failed_checks.items()],
        key=lambda x: x[1],
        reverse=True
    )
    
    stats = {
        "total_entries": total,
        "verified_entries": verified,
        "verification_rate": round(verified / total * 100, 2) if total > 0 else 0,
        "most_common_failures": most_common_failures
    }
    
    return stats