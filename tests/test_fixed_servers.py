# tests/test_fixed_servers.py
"""
Test your servers after applying fixture and path fixes.
"""

import sys
from pathlib import Path

SERVERS_DIR = Path("D:/one drive/study/ARCEE AI INTERNSHIP/mcp data gen minimal/mcp-dataset-generator/mcp_env/servers")

def test_all_servers_comprehensive():
    """Comprehensive test of all servers."""
    print("🧪 COMPREHENSIVE SERVER TEST (POST-FIX)")
    print("=" * 60)
    
    results = []
    
    # Test Filesystem
    print("\n📁 Testing Filesystem Server...")
    fs_result = test_filesystem_server()
    results.append(("Filesystem", fs_result))
    
    # Test Github  
    print("\n🐙 Testing Github Server...")
    github_result = test_github_server()
    results.append(("Github", github_result))
    
    # Test PostgreSQL
    print("\n🐘 Testing PostgreSQL Server...")
    pg_result = test_postgresql_server()
    results.append(("PostgreSQL", pg_result))
    
    # Test Search
    print("\n🔍 Testing Search Server...")
    search_result = test_search_server()
    results.append(("Search", search_result))
    
    # Summary
    print(f"\n" + "=" * 60)
    success_count = sum(1 for _, success in results if success)
    total_count = len(results)
    
    print(f"📊 FINAL RESULTS: {success_count}/{total_count} servers working")
    
    for server_name, success in results:
        status = "✅" if success else "❌"
        print(f"  {status} {server_name}")
    
    if success_count == total_count:
        print("\n🎉 SUCCESS! All servers are now working correctly!")
        print("\nYour MCP server environment is ready for:")
        print("  • RL training")
        print("  • Integration testing") 
        print("  • Production use")
    else:
        print(f"\n⚠️  {total_count - success_count} servers still have issues")
    
    return success_count == total_count

def test_filesystem_server():
    """Test filesystem server."""
    try:
        fs_dir = SERVERS_DIR / "Filesystem"
        sys.path.insert(0, str(fs_dir))
        import fs_server as server
        
        print(f"  ✅ Imported: {server.mcp.name}")
        
        if hasattr(server.mcp, 'tools'):
            tools = [tool.name for tool in server.mcp.tools]
            print(f"  🔧 Tools: {tools}")
            
            # Test a tool if available
            if hasattr(server, 'list_dir'):
                result = server.list_dir("/")
                print(f"  📋 list_dir('/'): {result}")
        
        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False
    finally:
        # Cleanup
        modules_to_remove = [name for name in sys.modules.keys() if 'fs_server' in name]
        for module in modules_to_remove:
            del sys.modules[module]
        if str(fs_dir) in sys.path:
            sys.path.remove(str(fs_dir))

def test_github_server():
    """Test github server."""
    try:
        github_dir = SERVERS_DIR / "Github"
        sys.path.insert(0, str(github_dir))
        import github_server as server
        
        print(f"  ✅ Imported: {server.mcp.name}")
        
        if hasattr(server.mcp, 'tools'):
            tools = [tool.name for tool in server.mcp.tools]
            print(f"  🔧 Tools: {tools}")
        
        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False
    finally:
        # Cleanup
        modules_to_remove = [name for name in sys.modules.keys() if 'github_server' in name]
        for module in modules_to_remove:
            del sys.modules[module]
        if str(github_dir) in sys.path:
            sys.path.remove(str(github_dir))

def test_postgresql_server():
    """Test postgresql server."""
    try:
        pg_dir = SERVERS_DIR / "PostgreSQL"
        sys.path.insert(0, str(pg_dir))
        import pg_server as server
        
        print(f"  ✅ Imported: {server.mcp.name}")
        
        if hasattr(server.mcp, 'tools'):
            tools = [tool.name for tool in server.mcp.tools]
            print(f"  🔧 Tools: {tools}")
        
        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False
    finally:
        # Cleanup
        modules_to_remove = [name for name in sys.modules.keys() if 'pg_server' in name]
        for module in modules_to_remove:
            del sys.modules[module]
        if str(pg_dir) in sys.path:
            sys.path.remove(str(pg_dir))

def test_search_server():
    """Test search server."""
    try:
        search_dir = SERVERS_DIR / "Search_tool"
        sys.path.insert(0, str(search_dir))
        import search_server as server
        
        print(f"  ✅ Imported: {server.mcp.name}")
        
        if hasattr(server.mcp, 'tools'):
            tools = [tool.name for tool in server.mcp.tools]
            print(f"  🔧 Tools: {tools}")
        
        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False
    finally:
        # Cleanup
        modules_to_remove = [name for name in sys.modules.keys() if 'search_server' in name]
        for module in modules_to_remove:
            del sys.modules[module]
        if str(search_dir) in sys.path:
            sys.path.remove(str(search_dir))

if __name__ == "__main__":
    test_all_servers_comprehensive()
