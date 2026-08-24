# environment/db_engine.py
# SQLite Database Engine with Security Safeguards
# Implements: Read-Only Authorizer, OOM Protection, Timeout Wrapper

import sqlite3
import time
from enum import Enum
from typing import Tuple, Optional
from pathlib import Path


# Query execution timeout in seconds
QUERY_TIMEOUT = 2.0

# Maximum rows to fetch (OOM protection)
MAX_FETCH_ROWS = 50

# How many SQLite VM instructions between timeout checks. Small enough that a
# runaway query is interrupted promptly, large enough that the callback
# overhead stays negligible on normal queries.
PROGRESS_INSTRUCTIONS = 1000

# SQLite authorizer action codes the agent is permitted to perform.
# This is an ALLOWLIST: every other action (INSERT, UPDATE, DELETE, DROP,
# CREATE, ALTER, ATTACH, PRAGMA, transactions, ...) is denied by SQLite itself.
#
# Why an authorizer rather than a regex denylist:
#   - A regex over query text cannot see what the statement actually does. It
#     both misses mutations (REPLACE, CREATE ... AS SELECT, ATTACH) and fires
#     on innocent SELECTs that merely mention a keyword in a string literal.
#   - PRAGMA query_only=ON is not sufficient on its own: an agent can simply
#     issue "PRAGMA query_only=OFF" and then mutate.
#   - The authorizer is consulted by SQLite for every operation in every
#     statement, so it cannot be talked around with cleverly worded SQL.
ALLOWED_ACTIONS = frozenset({
    sqlite3.SQLITE_SELECT,    # a SELECT statement
    sqlite3.SQLITE_READ,      # reading a column
    sqlite3.SQLITE_FUNCTION,  # scalar/aggregate functions: COUNT, SUM, ROUND, ...
})


class QueryStatus(str, Enum):
    """Outcome of a query execution attempt."""

    OK = "ok"          # executed successfully
    ERROR = "error"    # syntax error, unknown column, timeout, ...
    DENIED = "denied"  # blocked by the read-only authorizer


class DatabaseEngine:
    """
    SQLite Database Engine with security safeguards.

    Features:
    - In-memory SQLite database (:memory: mode)
    - Read-Only Authorizer: SQLite-enforced allowlist of read actions only
    - OOM Protection: cursor.fetchmany(50), never fetchall()
    - Timeout Wrapper: 2.0-second budget enforced via a progress handler,
      which (unlike signal.SIGALRM) works in worker threads and on Windows
    - Stringified errors: Never raises Python exceptions to caller
    """

    def __init__(self):
        """Initialize the database engine with an in-memory SQLite database."""
        self.connection: Optional[sqlite3.Connection] = None
        self.cursor: Optional[sqlite3.Cursor] = None
        self._schema_cache: Optional[str] = None
        # Lifted only by execute_privileged(), for environment-internal queries.
        # No SQL an agent can write reaches this flag.
        self._privileged: bool = False

    def _authorizer(self, action: int, arg1, arg2, db_name, trigger_name) -> int:
        """
        SQLite authorizer callback enforcing a read-only environment.

        Consulted by SQLite for every operation of every statement. Returns
        SQLITE_OK for permitted read actions and SQLITE_DENY for everything
        else, unless the engine has temporarily granted itself privilege.

        NOTE: privilege is a Python-side flag rather than an authorizer swap,
        because sqlite3.set_authorizer(None) does not reliably clear an
        installed authorizer on older Python versions (verified on 3.9).
        """
        if self._privileged:
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_OK if action in ALLOWED_ACTIONS else sqlite3.SQLITE_DENY

    def initialize(self) -> str:
        """
        Initialize a clean in-memory SQLite database and load mock data.

        Returns:
            str: Success message or error string
        """
        try:
            # Close existing connection if any
            self.close()

            # Create new in-memory database
            self.connection = sqlite3.connect(
                ':memory:',
                timeout=QUERY_TIMEOUT,
                check_same_thread=False
            )
            self.cursor = self.connection.cursor()

            # Load mock data from SQL file
            mock_data_path = Path(__file__).parent.parent / 'data' / 'mock_data.sql'

            if not mock_data_path.exists():
                return f"Error: Mock data file not found at {mock_data_path}"

            with open(mock_data_path, 'r') as f:
                sql_script = f.read()
            self.cursor.executescript(sql_script)
            self.connection.commit()

            # Cache schema info while writes are still permitted
            self._schema_cache = self._get_schema_info()

            # Seal the database: from here on the agent may only read.
            self.connection.set_authorizer(self._authorizer)

            return "Database initialized successfully"

        except Exception as e:
            return f"Error initializing database: {str(e)}"

    def _get_schema_info(self) -> str:
        """
        Get database schema information for the agent.

        Returns:
            str: Formatted schema information
        """
        if not self.cursor:
            return "Error: Database not initialized"

        try:
            # Get all table names
            self.cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = [row[0] for row in self.cursor.fetchmany(MAX_FETCH_ROWS)]

            schema_parts = ["DATABASE SCHEMA:", "=" * 50]

            for table in tables:
                schema_parts.append(f"\nTable: {table}")
                schema_parts.append("-" * 30)

                # Get column info using PRAGMA
                self.cursor.execute(f"PRAGMA table_info({table})")
                columns = self.cursor.fetchmany(MAX_FETCH_ROWS)

                for col in columns:
                    col_id, name, col_type, not_null, default, pk = col
                    pk_marker = " [PRIMARY KEY]" if pk else ""
                    null_marker = " NOT NULL" if not_null else ""
                    schema_parts.append(f"  - {name}: {col_type}{null_marker}{pk_marker}")

            return "\n".join(schema_parts)

        except Exception as e:
            return f"Error getting schema: {str(e)}"

    def get_schema(self) -> str:
        """
        Get cached schema information.

        Returns:
            str: Schema information string
        """
        if self._schema_cache:
            return self._schema_cache
        return self._get_schema_info()

    def execute_privileged_column(self, query: str) -> list:
        """
        Execute a trusted query and return its first column as a list.

        Used only by the environment itself to resolve a task's answer domain.
        Never reachable from an agent action.

        Args:
            query: SQL query string (trusted, not agent-supplied)

        Returns:
            list: First column of every returned row
        """
        if not self.connection or not self.cursor:
            return []
        self._privileged = True
        try:
            self.cursor.execute(query)
            return [row[0] for row in self.cursor.fetchall()]
        finally:
            self._privileged = False

    def execute_privileged(self, query: str):
        """
        Execute a trusted query with the authorizer temporarily lifted.

        Used only by the environment itself to resolve task ground truth from
        each task's reference SQL. Never reachable from an agent action.

        Args:
            query: SQL query string (trusted, not agent-supplied)

        Returns:
            The first column of the first row, or None if there are no rows.
        """
        if not self.connection or not self.cursor:
            return None
        self._privileged = True
        try:
            self.cursor.execute(query)
            row = self.cursor.fetchone()
            return row[0] if row else None
        finally:
            self._privileged = False

    def _install_timeout(self, seconds: float) -> None:
        """Arm a wall-clock budget for the next query."""
        deadline = time.monotonic() + seconds

        def _guard() -> int:
            # Returning non-zero aborts the running statement.
            return 1 if time.monotonic() > deadline else 0

        self.connection.set_progress_handler(_guard, PROGRESS_INSTRUCTIONS)

    def _clear_timeout(self) -> None:
        """Disarm the query budget."""
        if self.connection:
            self.connection.set_progress_handler(None, 0)

    def execute_query(self, query: str) -> Tuple[str, QueryStatus]:
        """
        Execute a SQL query with all safety measures.

        Args:
            query: SQL query string

        Returns:
            Tuple[str, QueryStatus]: (result_string, status)
        """
        if not self.connection or not self.cursor:
            return "Error: Database not initialized", QueryStatus.ERROR

        # Strip and validate query
        query = query.strip()
        if not query:
            return "Error: Empty query provided", QueryStatus.ERROR

        try:
            self._install_timeout(QUERY_TIMEOUT)
            self.cursor.execute(query)

            # OOM PROTECTION: Use fetchmany(50), NEVER fetchall()
            rows = self.cursor.fetchmany(MAX_FETCH_ROWS)

            if not rows:
                # Check if it was a query that doesn't return rows
                if self.cursor.description is None:
                    return "Query executed successfully (no results)", QueryStatus.OK
                return "Query returned no results", QueryStatus.OK

            # Get column names
            columns = [desc[0] for desc in self.cursor.description]

            # Format results
            result_lines = []
            result_lines.append("| " + " | ".join(columns) + " |")
            result_lines.append("|" + "|".join(["---"] * len(columns)) + "|")

            for row in rows:
                formatted_row = [str(val) if val is not None else "NULL" for val in row]
                result_lines.append("| " + " | ".join(formatted_row) + " |")

            result = "\n".join(result_lines)

            # Check if results were truncated
            if self.cursor.fetchmany(1):
                result += (
                    f"\n\n[TRUNCATED] Results limited to {MAX_FETCH_ROWS} rows. "
                    f"More rows exist."
                )

            return result, QueryStatus.OK

        except sqlite3.OperationalError as e:
            # Syntax errors, unknown columns, and interrupted (timed-out) queries.
            # NOTE: OperationalError subclasses DatabaseError, so this must be
            # caught BEFORE the DatabaseError branch below.
            if "interrupted" in str(e).lower():
                return (
                    f"Error: Query execution exceeded {QUERY_TIMEOUT} seconds timeout",
                    QueryStatus.ERROR,
                )
            return f"SQLite Error: {str(e)}", QueryStatus.ERROR

        except sqlite3.DatabaseError as e:
            # The authorizer refused an operation -> "not authorized"
            if "not authorized" in str(e).lower():
                return (
                    "DESTRUCTIVE_ACTION_BLOCKED: This environment is read-only. "
                    "Only SELECT queries are permitted.",
                    QueryStatus.DENIED,
                )
            return f"SQLite Error: {str(e)}", QueryStatus.ERROR

        except Exception as e:
            return f"Error: {str(e)}", QueryStatus.ERROR

        finally:
            self._clear_timeout()

    def close(self):
        """Close the database connection."""
        if self.cursor:
            self.cursor.close()
            self.cursor = None
        if self.connection:
            self.connection.close()
            self.connection = None
        self._schema_cache = None

    def __del__(self):
        """Destructor to ensure connection is closed."""
        try:
            self.close()
        except Exception:
            # Interpreter shutdown can invalidate sqlite3 internals; never
            # raise from a destructor.
            pass
