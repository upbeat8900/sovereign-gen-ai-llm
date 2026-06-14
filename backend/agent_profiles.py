from typing import Callable, List, Optional

from fastapi import HTTPException


def _public_agent_profile(row: dict) -> dict:
    return {
        "id": row["id"],
        "name": row["name"],
        "personality": row.get("personality") or "",
        "llm_model_id": row["llm_model_id"],
        "tts_voice_uri": (row.get("tts_voice_uri") or "").strip() or None,
        "tts_speech_rate": row.get("tts_speech_rate"),
        "llm_provider": row.get("llm_provider"),
        "llm_model": row.get("llm_model"),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def load_agent_profiles(connection) -> List[dict]:
    rows = connection.execute(
        """
        SELECT
            ap.id,
            ap.name,
            ap.personality,
            ap.llm_model_id,
            ap.tts_voice_uri,
            ap.tts_speech_rate,
            ap.created_at,
            ap.updated_at,
            lm.provider AS llm_provider,
            lm.model AS llm_model
        FROM agent_profiles ap
        JOIN llm_models lm ON lm.id = ap.llm_model_id
        ORDER BY ap.name COLLATE NOCASE ASC, ap.id ASC
        """
    ).fetchall()
    return [_public_agent_profile(dict(row)) for row in rows]


def agent_profile_or_404(connection, profile_id: int) -> dict:
    row = connection.execute(
        """
        SELECT
            ap.id,
            ap.name,
            ap.personality,
            ap.llm_model_id,
            ap.tts_voice_uri,
            ap.tts_speech_rate,
            ap.created_at,
            ap.updated_at,
            lm.provider AS llm_provider,
            lm.model AS llm_model
        FROM agent_profiles ap
        JOIN llm_models lm ON lm.id = ap.llm_model_id
        WHERE ap.id = ?
        """,
        (profile_id,),
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Agent profile not found")
    return _public_agent_profile(dict(row))


def create_agent_profile(connection, payload, llm_model_or_404: Callable) -> dict:
    llm_model_or_404(connection, payload.llm_model_id)
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Agent name is required")
    cursor = connection.execute(
        """
        INSERT INTO agent_profiles (name, personality, llm_model_id, tts_voice_uri, tts_speech_rate)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            name,
            (payload.personality or "").strip(),
            payload.llm_model_id,
            (payload.tts_voice_uri or "").strip() or None,
            payload.tts_speech_rate,
        ),
    )
    return agent_profile_or_404(connection, cursor.lastrowid)


def update_agent_profile(connection, profile_id: int, payload, llm_model_or_404: Callable) -> dict:
    agent_profile_or_404(connection, profile_id)
    llm_model_or_404(connection, payload.llm_model_id)
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Agent name is required")
    connection.execute(
        """
        UPDATE agent_profiles
        SET name = ?, personality = ?, llm_model_id = ?, tts_voice_uri = ?, tts_speech_rate = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (
            name,
            (payload.personality or "").strip(),
            payload.llm_model_id,
            (payload.tts_voice_uri or "").strip() or None,
            payload.tts_speech_rate,
            profile_id,
        ),
    )
    return agent_profile_or_404(connection, profile_id)


def delete_agent_profile(connection, profile_id: int) -> None:
    agent_profile_or_404(connection, profile_id)
    connection.execute("DELETE FROM agent_profiles WHERE id = ?", (profile_id,))
