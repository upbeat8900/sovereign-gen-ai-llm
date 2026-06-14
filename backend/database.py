import os
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple


DATA_DIR = Path(os.getenv("CHAT_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
DB_PATH = DATA_DIR / "chat.db"

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful local assistant. Use the conversation memory bank "
    "when it is relevant, and be clear when you are unsure."
)

DEFAULT_MULTI_AGENT_PROMPT = """The conversation is inspired by public themes associated with well-known motivational, personal development, and spiritual teachers, but you must not claim to literally be any real person. You are an educational and reflective simulation.

Your job is to help the user think deeply, honestly, and practically about their life, goals, fears, relationships, work, meaning, habits, and inner growth.

You may:
- respond directly to the user;
- build on what another character said;
- respectfully disagree or add nuance;
- ask another character for their view;
- ask the user a thoughtful question;
- summarize tensions or patterns emerging in the conversation.

Do not dominate the conversation. Keep responses concise, warm, and conversational unless the user asks for depth.

Avoid repetition loops. Each message you add must be meaningfully different from what you or others already said — not a rephrase, echo, or summary of the same point. If the idea is already on the table, advance the discussion with a new angle, example, question, tension, or practical step instead of circling back.

When speaking, use your assigned character voice, but avoid catchphrases, exact imitation, or claiming private knowledge of any real person.

The transcript below labels each speaker as [Name]: …, including your own earlier lines as [{character_name}]: …. Use that history — together with your assigned character — to avoid repeating or lightly rephrasing anything already said. Those bracket labels appear only in the transcript, not in your reply.

Your reply must contain ONLY what your assigned character would say — first person, in character, never a group summary or speaking for others. Do not prefix your reply with [Name]: or bracketed speaker labels; each message is tagged separately. When addressing another participant, use only their first name."""


def _default_base_url() -> str:
    return os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


def _default_model() -> str:
    return os.getenv("OLLAMA_MODEL", "qwen3.5:9b")


def generate_memory_title(content: str) -> str:
    first_line = next((line.strip() for line in content.splitlines() if line.strip()), "")
    text = re.sub(r"[#*_`>\[\]()]|https?://\S+", " ", first_line or content)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()[:9]
    title = " ".join(words).strip(" .,:;-")
    if not title:
        return "Untitled memory"
    if len(title) > 72:
        return f"{title[:72].rstrip()}..."
    return title


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 30000")
    connection.execute("PRAGMA journal_mode = WAL")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_db() -> None:
    with get_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                sort_order INTEGER NOT NULL DEFAULT 0,
                llm_model_id INTEGER,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant')),
                content TEXT NOT NULL,
                llm_provider TEXT,
                llm_model TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                source_message_id INTEGER,
                llm_provider TEXT,
                llm_model TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                archived_at TEXT,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
                FOREIGN KEY (source_message_id) REFERENCES messages(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS llm_config (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                provider TEXT NOT NULL,
                base_url TEXT NOT NULL,
                model TEXT NOT NULL,
                api_key TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS llm_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider TEXT NOT NULL,
                base_url TEXT NOT NULL,
                model TEXT NOT NULL,
                api_key TEXT,
                is_active INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS speech_config (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                whisper_model TEXT NOT NULL DEFAULT 'base.en',
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS app_config (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                default_prompt TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS llm_generation_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                sample_count INTEGER NOT NULL DEFAULT 0,
                total_generation_ms INTEGER NOT NULL DEFAULT 0,
                total_context_chars INTEGER NOT NULL DEFAULT 0,
                total_output_chars INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(provider, model)
            );

            CREATE TABLE IF NOT EXISTS conversation_participants (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                llm_model_id INTEGER NOT NULL,
                personality TEXT NOT NULL DEFAULT '',
                sort_order INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
                FOREIGN KEY (llm_model_id) REFERENCES llm_models(id)
            );

            CREATE TABLE IF NOT EXISTS agent_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                personality TEXT NOT NULL DEFAULT '',
                llm_model_id INTEGER NOT NULL,
                tts_voice_uri TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (llm_model_id) REFERENCES llm_models(id)
            );
            """
        )
        _ensure_column(connection, "conversations", "sort_order", "INTEGER")
        _ensure_column(connection, "conversations", "llm_model_id", "INTEGER")
        _backfill_conversation_order(connection)
        _ensure_column(connection, "messages", "llm_provider", "TEXT")
        _ensure_column(connection, "messages", "llm_model", "TEXT")
        _ensure_column(connection, "messages", "image_media_type", "TEXT")
        _ensure_column(connection, "messages", "image_data", "TEXT")
        _ensure_column(connection, "messages", "generation_ms", "INTEGER")
        _ensure_column(connection, "messages", "include_history", "INTEGER")
        _ensure_column(connection, "messages", "participant_id", "INTEGER")
        _ensure_column(connection, "conversation_participants", "name", "TEXT")
        _ensure_column(connection, "conversation_participants", "tts_voice_uri", "TEXT")
        _ensure_column(connection, "conversation_participants", "agent_profile_id", "INTEGER")
        _ensure_column(connection, "conversation_participants", "tts_speech_rate", "REAL")
        _ensure_column(connection, "agent_profiles", "tts_speech_rate", "REAL")
        _ensure_column(connection, "app_config", "multi_agent_prompt", "TEXT")
        _ensure_column(connection, "memories", "title", "TEXT")
        _ensure_column(connection, "memories", "llm_provider", "TEXT")
        _ensure_column(connection, "memories", "llm_model", "TEXT")
        _ensure_column(connection, "memories", "archived_at", "TEXT")
        _ensure_column(connection, "llm_models", "comments", "TEXT")
        _ensure_column(connection, "llm_models", "tts_voice_uri", "TEXT")
        _ensure_column(connection, "speech_config", "elevenlabs_api_key", "TEXT")
        _backfill_memory_titles(connection)
        _ensure_column(connection, "memories", "title_generated_at", "TEXT")
        _backfill_memory_title_generated_at(connection)
        connection.execute(
            """
            INSERT OR IGNORE INTO llm_config (id, provider, base_url, model, api_key)
            VALUES (1, 'ollama', ?, ?, NULL)
            """,
            (_default_base_url(), _default_model()),
        )
        _seed_llm_models(connection)
        _backfill_conversation_models(connection)
        connection.execute(
            """
            INSERT OR IGNORE INTO speech_config (id, whisper_model)
            VALUES (1, ?)
            """,
            (os.getenv("WHISPER_MODEL", "base.en"),),
        )
        connection.execute(
            """
            INSERT OR IGNORE INTO app_config (id, default_prompt)
            VALUES (1, ?)
            """,
            (DEFAULT_SYSTEM_PROMPT,),
        )
        _backfill_generation_stats(connection)


def _ensure_column(connection: sqlite3.Connection, table: str, column: str, column_type: str) -> None:
    existing = {row["name"] for row in connection.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")


def _backfill_memory_titles(connection: sqlite3.Connection) -> None:
    rows = connection.execute(
        "SELECT id, content FROM memories WHERE title IS NULL OR TRIM(title) = ''"
    ).fetchall()
    for row in rows:
        connection.execute(
            "UPDATE memories SET title = ? WHERE id = ?",
            (generate_memory_title(row["content"]), row["id"]),
        )


def _backfill_memory_title_generated_at(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        UPDATE memories
        SET title_generated_at = created_at
        WHERE title_generated_at IS NULL
        """
    )


def _backfill_conversation_order(connection: sqlite3.Connection) -> None:
    rows = connection.execute(
        """
        SELECT id FROM conversations
        WHERE sort_order IS NULL OR sort_order = 0
        ORDER BY updated_at DESC, id DESC
        """
    ).fetchall()
    for index, row in enumerate(rows):
        connection.execute("UPDATE conversations SET sort_order = ? WHERE id = ?", (index, row["id"]))


def _backfill_conversation_models(connection: sqlite3.Connection) -> None:
    active_model = connection.execute(
        "SELECT id FROM llm_models WHERE is_active = 1 ORDER BY id LIMIT 1"
    ).fetchone()
    if active_model:
        connection.execute(
            "UPDATE conversations SET llm_model_id = ? WHERE llm_model_id IS NULL",
            (active_model["id"],),
        )


def _seed_llm_models(connection: sqlite3.Connection) -> None:
    count = connection.execute("SELECT COUNT(*) AS count FROM llm_models").fetchone()["count"]
    if count == 0:
        legacy = connection.execute("SELECT * FROM llm_config WHERE id = 1").fetchone()
        if legacy:
            connection.execute(
                """
                INSERT INTO llm_models (provider, base_url, model, api_key, is_active)
                VALUES (?, ?, ?, ?, 1)
                """,
                (legacy["provider"], legacy["base_url"], legacy["model"], legacy["api_key"]),
            )
        else:
            connection.execute(
                """
                INSERT INTO llm_models (provider, base_url, model, api_key, is_active)
                VALUES ('ollama', ?, ?, NULL, 1)
                """,
                (_default_base_url(), _default_model()),
            )
        return

    active_count = connection.execute(
        "SELECT COUNT(*) AS count FROM llm_models WHERE is_active = 1"
    ).fetchone()["count"]
    if active_count == 0:
        first_model = connection.execute("SELECT id FROM llm_models ORDER BY id LIMIT 1").fetchone()
        connection.execute("UPDATE llm_models SET is_active = 1 WHERE id = ?", (first_model["id"],))


def row_to_dict(row: Optional[sqlite3.Row]) -> Optional[dict]:
    if row is None:
        return None
    return dict(row)


def _backfill_generation_stats(connection: sqlite3.Connection) -> None:
    existing = connection.execute("SELECT COUNT(*) AS count FROM llm_generation_stats").fetchone()["count"]
    if existing:
        return

    rows = connection.execute(
        """
        SELECT llm_provider, llm_model, generation_ms, LENGTH(content) AS output_chars
        FROM messages
        WHERE role = 'assistant'
          AND generation_ms IS NOT NULL
          AND llm_provider IS NOT NULL
          AND llm_model IS NOT NULL
        """
    ).fetchall()
    aggregates: Dict[Tuple[str, str], Dict[str, int]] = {}
    for row in rows:
        key = (row["llm_provider"], row["llm_model"])
        output_chars = max(1, int(row["output_chars"] or 1))
        context_chars = output_chars * 2
        bucket = aggregates.setdefault(
            key,
            {"sample_count": 0, "total_generation_ms": 0, "total_context_chars": 0, "total_output_chars": 0},
        )
        bucket["sample_count"] += 1
        bucket["total_generation_ms"] += int(row["generation_ms"])
        bucket["total_context_chars"] += context_chars
        bucket["total_output_chars"] += output_chars

    for (provider, model), bucket in aggregates.items():
        connection.execute(
            """
            INSERT INTO llm_generation_stats (
                provider, model, sample_count, total_generation_ms, total_context_chars, total_output_chars
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                provider,
                model,
                bucket["sample_count"],
                bucket["total_generation_ms"],
                bucket["total_context_chars"],
                bucket["total_output_chars"],
            ),
        )
