import logging
from contextlib import contextmanager
from typing import Any, Optional
import psycopg
from psycopg.rows import dict_row
from app.config import settings

logger = logging.getLogger(__name__)

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
    image_paths TEXT[],  -- array of local paths
    status TEXT NOT NULL DEFAULT 'pending',  -- pending / published / cancelled
    created_at TIMESTAMPTZ DEFAULT NOW()
);
"""

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

def get_conn():
    return psycopg.connect(settings.database_url, row_factory=dict_row)

@contextmanager
def db():
    conn = get_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db():
    with db() as conn:
        conn.execute(SCHEMA)
        # Insert default schedule only if table is completely empty
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
        # Ensure topic column exists (safe for existing DBs)
        conn.execute("ALTER TABLE posts ADD COLUMN IF NOT EXISTS topic TEXT")
        conn.execute(
            "INSERT INTO posts (draft_id, telegram_message_id, text, topic) VALUES (%s, %s, %s, %s)",
            (draft_id, message_id, text, topic)
        )

def get_recent_posts(limit: int = 12) -> list:
    """Return short topics of the last N published posts (newest first)."""
    with db() as conn:
        conn.execute("ALTER TABLE posts ADD COLUMN IF NOT EXISTS topic TEXT")
        cur = conn.execute(
            "SELECT COALESCE(topic, LEFT(text, 40)) AS topic FROM posts ORDER BY published_at DESC LIMIT %s",
            (limit,)
        )
        return [row["topic"] for row in cur.fetchall() if row["topic"]]
