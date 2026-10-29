import logging
from contextlib import contextmanager
from typing import Any, Optional, List
from datetime import date, time as dtime
import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from app.config import settings

logger = logging.getLogger(__name__)

# Connection pool (min 1, max 10)
_pool: Optional[ConnectionPool] = None

def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            settings.database_url,
            min_size=1,
            max_size=10,
            kwargs={"row_factory": dict_row},
        )
    return _pool

SCHEMA = """
CREATE TABLE IF NOT EXISTS schedule (
    id SERIAL PRIMARY KEY,
    time_slot TIME NOT NULL,
    enabled BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS drafts (
    id SERIAL PRIMARY KEY,
    text TEXT NOT NULL,
    image_path TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    published_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS posts (
    id SERIAL PRIMARY KEY,
    draft_id INTEGER REFERENCES drafts(id),
    telegram_message_id BIGINT,
    text TEXT,
    topic TEXT,
    published_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS extra_posts (
    id SERIAL PRIMARY KEY,
    run_at TIMESTAMPTZ NOT NULL,
    text TEXT,
    topic TEXT,
    image_paths TEXT[],
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Track which schedule slots were already published on a given date (MSK)
CREATE TABLE IF NOT EXISTS published_slots (
    slot_date DATE NOT NULL,
    time_slot TIME NOT NULL,
    published_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (slot_date, time_slot)
);

-- Simple key-value store for pending manager confirms (survives restarts)
CREATE TABLE IF NOT EXISTS pending_confirms (
    key TEXT PRIMARY KEY,
    data JSONB NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
"""

@contextmanager
def db():
    pool = get_pool()
    with pool.connection() as conn:
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

def init_db():
    with db() as conn:
        conn.execute(SCHEMA)
        # Ensure topic column
        conn.execute("ALTER TABLE posts ADD COLUMN IF NOT EXISTS topic TEXT")
        # Default schedule
        cur = conn.execute("SELECT COUNT(*) AS cnt FROM schedule")
        if cur.fetchone()["cnt"] == 0:
            conn.execute("INSERT INTO schedule (time_slot) VALUES ('10:00'), ('18:00')")
            logger.info("Inserted default schedule 10:00 and 18:00")
        logger.info("Database schema initialized")

def get_schedule():
    with db() as conn:
        cur = conn.execute("SELECT time_slot FROM schedule WHERE enabled = TRUE ORDER BY time_slot")
        return [row["time_slot"] for row in cur.fetchall()]

def set_schedule(times: list):
    with db() as conn:
        conn.execute("DELETE FROM schedule")
        for t in times:
            conn.execute("INSERT INTO schedule (time_slot) VALUES (%s)", (t,))

def save_draft(text: str, image_path: Optional[str] = None) -> int:
    with db() as conn:
        cur = conn.execute(
            "INSERT INTO drafts (text, image_path, status) VALUES (%s, %s, 'pending') RETURNING id",
            (text, image_path)
        )
        return cur.fetchone()["id"]

def get_pending_draft() -> Optional[dict]:
    with db() as conn:
        cur = conn.execute(
            "SELECT * FROM drafts WHERE status = 'pending' ORDER BY created_at DESC LIMIT 1"
        )
        return cur.fetchone()

def update_draft_status(draft_id: int, status: str):
    with db() as conn:
        conn.execute(
            "UPDATE drafts SET status = %s, published_at = CASE WHEN %s = 'published' THEN NOW() ELSE published_at END WHERE id = %s",
            (status, status, draft_id)
        )

def save_post(draft_id: int, message_id: int, text: str, topic: str = None):
    with db() as conn:
        conn.execute(
            "INSERT INTO posts (draft_id, telegram_message_id, text, topic) VALUES (%s, %s, %s, %s)",
            (draft_id, message_id, text, topic)
        )

def get_recent_posts(limit: int = 12) -> list:
    with db() as conn:
        cur = conn.execute(
            "SELECT COALESCE(topic, LEFT(text, 40)) AS topic FROM posts ORDER BY published_at DESC LIMIT %s",
            (limit,)
        )
        return [row["topic"] for row in cur.fetchall() if row["topic"]]

def get_pending_extras():
    with db() as conn:
        cur = conn.execute(
            "SELECT * FROM extra_posts WHERE status = 'pending' AND run_at <= NOW() ORDER BY run_at"
        )
        return cur.fetchall()

def get_upcoming_extras():
    with db() as conn:
        cur = conn.execute(
            "SELECT * FROM extra_posts WHERE status = 'pending' AND run_at > NOW() ORDER BY run_at"
        )
        return cur.fetchall()

def save_extra_post(run_at, text=None, topic=None, image_paths=None):
    with db() as conn:
        cur = conn.execute(
            """INSERT INTO extra_posts (run_at, text, topic, image_paths)
               VALUES (%s, %s, %s, %s) RETURNING id""",
            (run_at, text, topic, image_paths or [])
        )
        return cur.fetchone()["id"]

def cancel_extra(extra_id: int):
    with db() as conn:
        conn.execute("UPDATE extra_posts SET status = 'cancelled' WHERE id = %s", (extra_id,))

def mark_extra_published(extra_id: int):
    with db() as conn:
        conn.execute("UPDATE extra_posts SET status = 'published' WHERE id = %s", (extra_id,))

# --- Published slots tracking (survives restarts) ---
def is_slot_published(slot_date: date, slot: dtime) -> bool:
    with db() as conn:
        cur = conn.execute(
            "SELECT 1 FROM published_slots WHERE slot_date = %s AND time_slot = %s",
            (slot_date, slot)
        )
        return cur.fetchone() is not None

def mark_slot_published(slot_date: date, slot: dtime):
    with db() as conn:
        conn.execute(
            """INSERT INTO published_slots (slot_date, time_slot)
               VALUES (%s, %s) ON CONFLICT DO NOTHING""",
            (slot_date, slot)
        )

# --- Pending confirms (simple JSON store) ---
def get_pending_confirm(key: str = "manager") -> Optional[dict]:
    with db() as conn:
        cur = conn.execute("SELECT data FROM pending_confirms WHERE key = %s", (key,))
        row = cur.fetchone()
        return row["data"] if row else None

def set_pending_confirm(data: Optional[dict], key: str = "manager"):
    with db() as conn:
        if data is None:
            conn.execute("DELETE FROM pending_confirms WHERE key = %s", (key,))
        else:
            conn.execute(
                """INSERT INTO pending_confirms (key, data, updated_at)
                   VALUES (%s, %s, NOW())
                   ON CONFLICT (key) DO UPDATE SET data = EXCLUDED.data, updated_at = NOW()""",
                (key, data)
            )
