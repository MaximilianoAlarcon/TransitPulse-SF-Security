from pathlib import Path
import psycopg2
from typing import Any
from psycopg2 import sql

class DatabaseConfigError(ValueError):
    pass


REQUIRED_DB_KEYS = ["host", "database", "user", "password", "port"]


def validate_db_config(db_config: dict) -> None:
    missing = [key for key in REQUIRED_DB_KEYS if not db_config.get(key)]
    if missing:
        raise DatabaseConfigError(
            f"Missing database environment variables: {', '.join(missing)}"
        )

def _serialize_cell(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    return value

def get_db_connection(db_config: dict):
    validate_db_config(db_config)
    return psycopg2.connect(
        host=db_config["host"],
        dbname=db_config["database"],
        user=db_config["user"],
        password=db_config["password"],
        port=db_config["port"],
    )


def execute_sql_file(db_config: dict, sql_path: Path) -> None:
    sql_path = Path(sql_path)
    if not sql_path.exists():
        raise FileNotFoundError(f"SQL file not found: {sql_path}")

    sql_content = sql_path.read_text(encoding="utf-8")

    with get_db_connection(db_config) as conn:
        with conn.cursor() as cur:
            cur.execute(sql_content)
        conn.commit()

def execute_query(db_config: dict, query: str) -> dict[str, Any]:
    cleaned_query = (query or "").strip()
    if not cleaned_query:
        raise ValueError("Query cannot be empty")

    with get_db_connection(db_config) as conn:
        with conn.cursor() as cur:
            cur.execute(cleaned_query)

            if cur.description:
                columns = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
                conn.commit()
                return {
                    "has_result_set": True,
                    "columns": columns,
                    "rows": [[_serialize_cell(cell) for cell in row] for row in rows],
                    "row_count": len(rows),
                    "status_message": cur.statusmessage,
                }

            affected_rows = cur.rowcount
            status_message = cur.statusmessage
            conn.commit()
            return {
                "has_result_set": False,
                "affected_rows": affected_rows,
                "status_message": status_message,
            }