# fake_fs_server.py
from fastmcp import FastMCP
import os

mcp = FastMCP("Fake Filesystem")
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "fixtures", "fs"))  # base sandbox path

def _safe_path(path: str) -> str:
    # Resolve path against BASE_DIR to prevent escape
    full_path = os.path.abspath(os.path.join(BASE_DIR, path.lstrip("/\\")))
    if not full_path.startswith(BASE_DIR):
        raise FileNotFoundError(f"Path '{path}' is outside of allowed sandbox.")
    return full_path

@mcp.tool()
def list_dir(path: str = "/") -> list:
    """
    List contents of a directory in the sandbox.
    :param path: Directory path (relative to sandbox root).
    :return: List of filenames (and directories) in the path.
    """
    dir_path = _safe_path(path)
    if not os.path.isdir(dir_path):
        return []  # not a directory or doesn't exist
    entries = []
    for name in os.listdir(dir_path):
        # Append "/" to directories for clarity
        full_path = os.path.join(dir_path, name)
        if os.path.isdir(full_path):
            entries.append(name + "/")
        else:
            entries.append(name)
    return entries

@mcp.tool()
def read_file(path: str) -> str:
    """
    Read the content of a file.
    :param path: File path (relative to sandbox root).
    :return: The file's text content, or an error message.
    """
    file_path = _safe_path(path)
    if not os.path.isfile(file_path):
        return f"[Error] File not found: {path}"
    try:
        with open(file_path, "r") as f:
            content = f.read()
            # Optionally truncate if very large to avoid huge output
            if len(content) > 5000:  # example cutoff
                content = content[:5000] + "\n[TRUNCATED]"
            return content
    except Exception as e:
        return f"[Error] Could not read file: {e}"

@mcp.tool()
def write_file(path: str, content: str) -> str:
    """
    Write content to a file. Overwrites if file exists.
    :param path: File path (relative to sandbox root).
    :param content: Text content to write.
    :return: Confirmation message.
    """
    file_path = _safe_path(path)
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    try:
        with open(file_path, "w") as f:
            f.write(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"[Error] Could not write file: {e}"
