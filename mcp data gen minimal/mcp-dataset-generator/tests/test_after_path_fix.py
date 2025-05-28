# tests/test_after_path_fix.py
"""
Test servers after path fixes to ensure they still work.
"""

import sys
from pathlib import Path

SERVERS_DIR = Path("D:/one drive/study/ARCEE AI INTERNSHIP/mcp data gen minimal/mcp-dataset-generator/mcp_env/servers")

def test_all_servers_after_fix():
    """Test all servers after path fixes."""
    print("🧪 TESTING SERVERS AFTER PATH FIXES")
    print("=" * 50)
    
    servers = [
        ("Filesystem", "fs_server"),
        ("Github", "github_server"), 
        ("PostgreSQL", "pg_server"),
        ("Search_tool", "search_server")
    ]
    
    results = []
    
    for server_name, module_name in servers:
        print(f"\n📋 Testing {server_name}...")
        
        server_dir = SERVERS_DIR / server_name
        sys.path.insert(0, str(server_dir))
        
        try:
            # Import server
            server = __import__(module_name)
            
            # Check MCP instance
            if hasattr(server, 'mcp'):
                print(f"  ✅ {server.mcp.name}")
                
                # Check tools
                if hasattr(server.mcp, 'tools'):
                    tools = [tool.name for tool in server.mcp.tools]
                    print(f"  🔧 Tools: {tools}")
                
                results.append(True)
            else:
                print(f"  ❌ No MCP instance")
                results.append(False)
            
            # Cleanup
            if module_name in sys.modules:
                del sys.modules[module_name]
            sys.path.remove(str(server_dir))
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            results.append(False)
            if str(server_dir) in sys.path:
                sys.path.remove(str(server_dir))
    
    # Summary
    success_count = sum(results)
    total_count = len(results)
    
    print(f"\n" + "=" * 50)
    if success_count == total_count:
        print(f"🎉 SUCCESS! All {total_count} servers working after path fixes!")
        print("Your servers are now more robust and portable.")
    else:
        print(f"⚠️  {success_count}/{total_count} servers working")
        print("Some servers may need manual attention.")
    
    return success_count == total_count

if __name__ == "__main__":
    test_all_servers_after_fix()
