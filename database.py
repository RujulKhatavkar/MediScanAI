"""
MediScanAI - Prescription History (SQLite)
Stores every scanned prescription with extracted drug info,
interactions found, and chat history.
"""

import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path

DB_PATH = os.environ.get("MEDISCANAI_DB", "mediscanai_history.db")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    with get_connection() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS prescriptions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            scanned_at  TEXT    NOT NULL,
            image_name  TEXT,
            ocr_text    TEXT,
            drug_info   TEXT,   -- JSON
            drug_names  TEXT,   -- comma-separated for quick display
            notes       TEXT
        );

        CREATE TABLE IF NOT EXISTS interactions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            prescription_id INTEGER NOT NULL REFERENCES prescriptions(id),
            drug_a          TEXT NOT NULL,
            drug_b          TEXT NOT NULL,
            severity        TEXT NOT NULL,
            description     TEXT,
            recommendation  TEXT,
            sources         TEXT    -- JSON list
        );

        CREATE TABLE IF NOT EXISTS chat_messages (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            prescription_id INTEGER NOT NULL REFERENCES prescriptions(id),
            role            TEXT NOT NULL,
            content         TEXT NOT NULL,
            sent_at         TEXT NOT NULL
        );
        """)


def save_prescription(
    image_name: str,
    ocr_text: str,
    drug_info: dict,
    notes: str = "",
) -> int:
    """Save a scanned prescription. Returns the new prescription ID."""
    init_db()
    drug_names = ", ".join(drug_info.get("drug_names") or []) if drug_info else ""
    with get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO prescriptions
               (scanned_at, image_name, ocr_text, drug_info, drug_names, notes)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                datetime.now().isoformat(timespec="seconds"),
                image_name or "unknown",
                ocr_text or "",
                json.dumps(drug_info or {}),
                drug_names,
                notes,
            ),
        )
        return cur.lastrowid


def save_interactions(prescription_id: int, interactions: list) -> None:
    """Save interaction check results for a prescription."""
    if not interactions:
        return
    init_db()
    rows = [
        (
            prescription_id,
            i.drug_a,
            i.drug_b,
            i.severity,
            i.description,
            i.recommendation,
            json.dumps(i.sources),
        )
        for i in interactions
    ]
    with get_connection() as conn:
        conn.executemany(
            """INSERT INTO interactions
               (prescription_id, drug_a, drug_b, severity, description, recommendation, sources)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )


def save_chat_message(prescription_id: int, role: str, content: str) -> None:
    """Append a chat message to the history for a prescription."""
    init_db()
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO chat_messages (prescription_id, role, content, sent_at)
               VALUES (?, ?, ?, ?)""",
            (prescription_id, role, content, datetime.now().isoformat(timespec="seconds")),
        )


def get_all_prescriptions() -> list[dict]:
    """Return all prescriptions, newest first."""
    init_db()
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT id, scanned_at, image_name, drug_names, notes
               FROM prescriptions ORDER BY id DESC"""
        ).fetchall()
    return [dict(r) for r in rows]


def get_prescription(prescription_id: int) -> dict | None:
    """Return full details for one prescription including interactions and chat."""
    init_db()
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM prescriptions WHERE id = ?", (prescription_id,)
        ).fetchone()
        if not row:
            return None

        record = dict(row)
        record["drug_info"] = json.loads(record["drug_info"] or "{}")

        record["interactions"] = [
            dict(r) for r in conn.execute(
                "SELECT * FROM interactions WHERE prescription_id = ? ORDER BY CASE severity WHEN 'high' THEN 0 WHEN 'moderate' THEN 1 WHEN 'low' THEN 2 ELSE 3 END",
                (prescription_id,),
            ).fetchall()
        ]

        record["chat"] = [
            dict(r) for r in conn.execute(
                "SELECT role, content, sent_at FROM chat_messages WHERE prescription_id = ? ORDER BY id",
                (prescription_id,),
            ).fetchall()
        ]

    return record


def delete_prescription(prescription_id: int) -> None:
    """Delete a prescription and all associated records."""
    init_db()
    with get_connection() as conn:
        conn.execute("DELETE FROM chat_messages WHERE prescription_id = ?", (prescription_id,))
        conn.execute("DELETE FROM interactions WHERE prescription_id = ?", (prescription_id,))
        conn.execute("DELETE FROM prescriptions WHERE id = ?", (prescription_id,))


def get_stats() -> dict:
    """Return summary statistics for the dashboard."""
    init_db()
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM prescriptions").fetchone()[0]
        total_interactions = conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
        high_risk = conn.execute(
            "SELECT COUNT(*) FROM interactions WHERE severity = 'high'"
        ).fetchone()[0]
        unique_drugs = conn.execute(
            "SELECT COUNT(DISTINCT drug_a) FROM interactions"
        ).fetchone()[0]
    return {
        "total_prescriptions": total,
        "total_interactions_checked": total_interactions,
        "high_risk_interactions": high_risk,
        "unique_drugs_seen": unique_drugs,
    }
