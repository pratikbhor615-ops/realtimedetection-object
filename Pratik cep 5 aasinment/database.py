"""
Database Management Module for Real-Time Object Detection Platform.
Handles MySQL connections, schema initialization, real-time event logging,
and retrieval of detection logs. Includes fallback handling for offline testing.
"""

import os
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "vision_platform")

FALLBACK_SQLITE_PATH = "vision_platform_fallback.db"

# Track live database connection state
_mysql_available: Optional[bool] = None


def get_mysql_connection(select_db: bool = True):
    """
    Establish a connection to the MySQL server using mysql-connector-python.
    """
    import mysql.connector

    config = {
        "host": DB_HOST,
        "port": DB_PORT,
        "user": DB_USER,
        "password": DB_PASSWORD,
        "connect_timeout": 3,
    }
    if select_db:
        config["database"] = DB_NAME

    return mysql.connector.connect(**config)


def init_database() -> Dict[str, Any]:
    """
    Initialize the database and detection_logs table.
    Creates vision_platform database and detection_logs table if they do not exist.
    """
    global _mysql_available
    try:
        import mysql.connector
        # Connect to MySQL server without selecting database first
        conn = get_mysql_connection(select_db=False)
        cursor = conn.cursor()

        # Create database
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
        cursor.execute(f"USE `{DB_NAME}`;")

        # Create detection_logs table per assignment requirements:
        # log_id (Primary Key), timestamp, object_class, confidence, bbox_x, bbox_y, bbox_w, bbox_h
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS detection_logs (
            log_id INT AUTO_INCREMENT PRIMARY KEY,
            timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            object_class VARCHAR(50) NOT NULL,
            confidence FLOAT NOT NULL,
            bbox_x INT NOT NULL,
            bbox_y INT NOT NULL,
            bbox_w INT NOT NULL,
            bbox_h INT NOT NULL,
            INDEX idx_timestamp (timestamp),
            INDEX idx_object_class (object_class)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
        cursor.execute(create_table_sql)
        conn.commit()
        cursor.close()
        conn.close()

        _mysql_available = True
        return {
            "success": True,
            "engine": "MySQL",
            "message": f"Successfully connected to MySQL database '{DB_NAME}' on {DB_HOST}:{DB_PORT}.",
        }

    except Exception as e:
        _mysql_available = False
        # Initialize fallback SQLite for reliable local testing when MySQL service is offline
        _init_sqlite_fallback()
        return {
            "success": False,
            "engine": "SQLite (Fallback)",
            "message": f"MySQL connection to {DB_HOST}:{DB_PORT} failed ({str(e)}). Using local SQLite fallback.",
            "error": str(e),
        }


def _init_sqlite_fallback():
    """Initializes a local fallback SQLite table matching the MySQL schema."""
    conn = sqlite3.connect(FALLBACK_SQLITE_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detection_logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            object_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            bbox_x INTEGER NOT NULL,
            bbox_y INTEGER NOT NULL,
            bbox_w INTEGER NOT NULL,
            bbox_h INTEGER NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def check_connection() -> Tuple[bool, str]:
    """
    Quickly verify if MySQL is reachable.
    """
    try:
        conn = get_mysql_connection(select_db=True)
        conn.close()
        return True, f"Connected to MySQL ({DB_HOST}:{DB_PORT}/{DB_NAME})"
    except Exception as e:
        return False, f"MySQL Unreachable ({str(e)}). Running in SQLite Fallback mode."


def log_detection(
    object_class: str,
    confidence: float,
    bbox_x: int,
    bbox_y: int,
    bbox_w: int,
    bbox_h: int,
    timestamp: Optional[datetime] = None,
) -> bool:
    """
    Insert a single object detection event into the database.
    """
    ts = timestamp or datetime.now()
    ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")

    # Try MySQL first
    try:
        conn = get_mysql_connection(select_db=True)
        cursor = conn.cursor()
        query = """
            INSERT INTO detection_logs (timestamp, object_class, confidence, bbox_x, bbox_y, bbox_w, bbox_h)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(
            query,
            (ts_str, object_class, float(confidence), int(bbox_x), int(bbox_y), int(bbox_w), int(bbox_h)),
        )
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception:
        # Fallback to SQLite
        try:
            _init_sqlite_fallback()
            conn = sqlite3.connect(FALLBACK_SQLITE_PATH)
            cursor = conn.cursor()
            query = """
                INSERT INTO detection_logs (timestamp, object_class, confidence, bbox_x, bbox_y, bbox_w, bbox_h)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            cursor.execute(
                query,
                (ts_str, object_class, float(confidence), int(bbox_x), int(bbox_y), int(bbox_w), int(bbox_h)),
            )
            conn.commit()
            conn.close()
            return True
        except Exception as sqle:
            print(f"Error logging to fallback database: {sqle}")
            return False


def log_detections_batch(detections: List[Dict[str, Any]]) -> int:
    """
    Insert a batch of object detections in a single database transaction.
    Returns the count of successfully logged rows.
    """
    if not detections:
        return 0

    rows_to_insert = []
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for d in detections:
        ts_str = d.get("timestamp", now_str)
        if isinstance(ts_str, datetime):
            ts_str = ts_str.strftime("%Y-%m-%d %H:%M:%S")

        rows_to_insert.append((
            ts_str,
            str(d["object_class"]),
            float(d["confidence"]),
            int(d["bbox_x"]),
            int(d["bbox_y"]),
            int(d["bbox_w"]),
            int(d["bbox_h"]),
        ))

    # Try MySQL
    try:
        conn = get_mysql_connection(select_db=True)
        cursor = conn.cursor()
        query = """
            INSERT INTO detection_logs (timestamp, object_class, confidence, bbox_x, bbox_y, bbox_w, bbox_h)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        cursor.executemany(query, rows_to_insert)
        conn.commit()
        count = cursor.rowcount
        cursor.close()
        conn.close()
        return count
    except Exception:
        # Fallback to SQLite
        try:
            _init_sqlite_fallback()
            conn = sqlite3.connect(FALLBACK_SQLITE_PATH)
            cursor = conn.cursor()
            query = """
                INSERT INTO detection_logs (timestamp, object_class, confidence, bbox_x, bbox_y, bbox_w, bbox_h)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            cursor.executemany(query, rows_to_insert)
            conn.commit()
            count = cursor.rowcount
            conn.close()
            return count
        except Exception as sqle:
            print(f"Error logging batch to fallback: {sqle}")
            return 0


def fetch_recent_logs(limit: int = 50, object_class: Optional[str] = None) -> pd.DataFrame:
    """
    Retrieve the most recent detection logs as a pandas DataFrame.
    """
    columns = ["log_id", "timestamp", "object_class", "confidence", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]

    # Try MySQL
    try:
        conn = get_mysql_connection(select_db=True)
        cursor = conn.cursor(dictionary=True)
        if object_class and object_class != "All":
            query = """
                SELECT log_id, timestamp, object_class, confidence, bbox_x, bbox_y, bbox_w, bbox_h
                FROM detection_logs
                WHERE object_class = %s
                ORDER BY log_id DESC
                LIMIT %s
            """
            cursor.execute(query, (object_class, limit))
        else:
            query = """
                SELECT log_id, timestamp, object_class, confidence, bbox_x, bbox_y, bbox_w, bbox_h
                FROM detection_logs
                ORDER BY log_id DESC
                LIMIT %s
            """
            cursor.execute(query, (limit,))

        records = cursor.fetchall()
        cursor.close()
        conn.close()
        if records:
            df = pd.DataFrame(records)
            df["confidence"] = df["confidence"].round(4)
            return df
        return pd.DataFrame(columns=columns)
    except Exception:
        # Fallback to SQLite
        try:
            _init_sqlite_fallback()
            conn = sqlite3.connect(FALLBACK_SQLITE_PATH)
            if object_class and object_class != "All":
                query = """
                    SELECT log_id, timestamp, object_class, confidence, bbox_x, bbox_y, bbox_w, bbox_h
                    FROM detection_logs
                    WHERE object_class = ?
                    ORDER BY log_id DESC
                    LIMIT ?
                """
                df = pd.read_sql_query(query, conn, params=(object_class, limit))
            else:
                query = """
                    SELECT log_id, timestamp, object_class, confidence, bbox_x, bbox_y, bbox_w, bbox_h
                    FROM detection_logs
                    ORDER BY log_id DESC
                    LIMIT ?
                """
                df = pd.read_sql_query(query, conn, params=(limit,))
            conn.close()
            if not df.empty:
                df["confidence"] = df["confidence"].round(4)
            return df
        except Exception:
            return pd.DataFrame(columns=columns)


def fetch_metrics() -> Dict[str, Any]:
    """
    Calculate summary statistics of logged detections for the dashboard.
    """
    total_logs = 0
    class_counts: Dict[str, int] = {}
    avg_conf = 0.0

    try:
        conn = get_mysql_connection(select_db=True)
        cursor = conn.cursor(dictionary=True)

        cursor.execute("SELECT COUNT(*) AS total, AVG(confidence) AS avg_conf FROM detection_logs")
        row = cursor.fetchone()
        if row:
            total_logs = row.get("total") or 0
            avg_conf = float(row.get("avg_conf") or 0.0)

        cursor.execute("""
            SELECT object_class, COUNT(*) AS count
            FROM detection_logs
            GROUP BY object_class
            ORDER BY count DESC
        """)
        for r in cursor.fetchall():
            class_counts[r["object_class"]] = r["count"]

        cursor.close()
        conn.close()
    except Exception:
        try:
            _init_sqlite_fallback()
            conn = sqlite3.connect(FALLBACK_SQLITE_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*), AVG(confidence) FROM detection_logs")
            row = cursor.fetchone()
            if row:
                total_logs = row[0] or 0
                avg_conf = float(row[1] or 0.0)

            cursor.execute("SELECT object_class, COUNT(*) FROM detection_logs GROUP BY object_class ORDER BY COUNT(*) DESC")
            for r in cursor.fetchall():
                class_counts[r[0]] = r[1]
            conn.close()
        except Exception:
            pass

    return {
        "total_logs": total_logs,
        "avg_confidence": round(avg_conf, 3),
        "class_counts": class_counts,
    }


def clear_all_logs() -> bool:
    """Clear all records from detection_logs."""
    try:
        conn = get_mysql_connection(select_db=True)
        cursor = conn.cursor()
        cursor.execute("TRUNCATE TABLE detection_logs")
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception:
        try:
            conn = sqlite3.connect(FALLBACK_SQLITE_PATH)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM detection_logs")
            conn.commit()
            conn.close()
            return True
        except Exception:
            return False
