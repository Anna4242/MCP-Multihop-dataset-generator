# fake_pg_server.py
from fastmcp import FastMCP
import sqlite3, os

mcp = FastMCP("Fake PostgreSQL")

# Connect to the SQLite database (read-only mode if possible)
conn = sqlite3.connect(os.path.join(os.path.dirname(__file__), "fixtures", "fakedb.sqlite"), check_same_thread=False)
conn.row_factory = sqlite3.Row  # to get dict-like row if needed

@mcp.tool()
def run_query(sql: str) -> list:
    """
    Execute an SQL query and return the result set.
    :param sql: SQL query string.
    :return: List of result rows (each row is dict or tuple). Or a message for non-select.
    """
    sql_stripped = sql.strip().lower()
    try:
        cur = conn.cursor()
        # Simple check: if not a select, block or execute carefully
        if not sql_stripped.startswith("select"):
            # For safety, do not allow modifications in this fake environment
            return ["[Error] Only SELECT queries are allowed in read-only mode."]
        cur.execute(sql)
        rows = cur.fetchall()
        # Convert to list of dicts for readability
        columns = [col[0] for col in cur.description]  # column names
        result = []
        for row in rows:
            # sqlite3.Row supports mapping interface
            record = {col: row[col] for col in columns}
            result.append(record)
        return result
    except Exception as e:
        return [f"[Error] Query failed: {e}"]

@mcp.tool()
def list_tables() -> list:
    """
    List all tables in the database (with their schema).
    """
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cur.fetchall()]
    schemas = []
    for tbl in tables:
        # Get CREATE TABLE statement
        cur.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{tbl}'")
        create_sql = cur.fetchone()[0]
        schemas.append(create_sql)
    return schemas
