import base64
import json
import os
import time
from pathlib import Path
from typing import List, Optional, Union

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .database import (
    DEFAULT_SYSTEM_PROMPT,
    generate_memory_title as fallback_memory_title,
    get_connection,
    init_db,
    row_to_dict,
)
from .agent_profiles import (
    agent_profile_or_404,
    create_agent_profile,
    delete_agent_profile,
    load_agent_profiles,
    update_agent_profile,
)
from .llm import chat as llm_chat
from .multi_agent import (
    load_participants,
    public_participant,
    replace_participants,
    run_multi_agent_turns,
    clamp_discussion_rounds,
    participant_display_name,
    previous_discussion_speaker,
    build_discussion_response_instruction,
)
from .stt import WHISPER_MODEL_OPTIONS, set_whisper_model, transcribe_audio_bytes
from .models import (
    AgentProfileCreate,
    AgentProfileRead,
    AgentProfileUpdate,
    Conversation,
    ConversationCreate,
    ConversationDelete,
    ConversationDetail,
    ConversationModelUpdate,
    ConversationParticipantRead,
    ConversationParticipantsUpdate,
    ConversationReorder,
    ConversationTitleUpdate,
    CrossConversationMemoryMerge,
    LlmContextPreview,
    LlmConfigRead,
    LlmConfigUpdate,
    LlmModelRead,
    Memory,
    MemoryGroup,
    MemoryIntegrate,
    MemoryMerge,
    MemoryMove,
    MessageCreate,
    MessageResponse,
    RememberCreate,
    PromptConfigRead,
    PromptConfigUpdate,
    SpeechConfigRead,
    SpeechConfigUpdate,
)


app = FastAPI(title="Sovereign Gen AI LLM")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_IMAGE_MEDIA_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024
DEFAULT_IMAGE_PROMPT = "What can you tell me about this image?"


@app.on_event("startup")
def startup() -> None:
    init_db()
    with get_connection() as connection:
        row = connection.execute("SELECT whisper_model FROM speech_config WHERE id = 1").fetchone()
        if row:
            set_whisper_model(row["whisper_model"])


def _conversation_or_404(connection, conversation_id: int) -> dict:
    conversation = row_to_dict(
        connection.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
    )
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


def _conversation_participant_count(connection, conversation_id: int) -> int:
    row = connection.execute(
        "SELECT COUNT(*) AS count FROM conversation_participants WHERE conversation_id = ?",
        (conversation_id,),
    ).fetchone()
    return int(row["count"])


def _public_conversation(connection, conversation_id: int) -> dict:
    conversation = _conversation_or_404(connection, conversation_id)
    conversation["participant_count"] = _conversation_participant_count(connection, conversation_id)
    return conversation


def _all_active_memories(
    connection,
    sort: str = "created_at",
    order: str = "asc",
    *,
    exclude_conversation_id: int = None,
) -> List[dict]:
    if sort not in {"created_at", "content", "title", "llm_model"}:
        raise HTTPException(status_code=400, detail="Unsupported memory sort field")
    if order not in {"asc", "desc"}:
        raise HTTPException(status_code=400, detail="Unsupported memory sort order")

    if exclude_conversation_id is None:
        rows = connection.execute(
            f"""
            SELECT * FROM memories
            WHERE archived_at IS NULL
            ORDER BY {sort} {order.upper()}
            """
        ).fetchall()
    else:
        rows = connection.execute(
            f"""
            SELECT * FROM memories
            WHERE archived_at IS NULL AND conversation_id != ?
            ORDER BY {sort} {order.upper()}
            """,
            (exclude_conversation_id,),
        ).fetchall()
    return [_public_memory(dict(row)) for row in rows]


def _public_memory(memory: dict) -> dict:
    public = dict(memory)
    public["title_pending"] = not public.get("title_generated_at")
    public.pop("title_generated_at", None)
    return public


def _active_memories(connection, conversation_id: int, sort: str = "created_at", order: str = "desc") -> List[dict]:
    if sort not in {"created_at", "content", "title", "llm_model"}:
        raise HTTPException(status_code=400, detail="Unsupported memory sort field")
    if order not in {"asc", "desc"}:
        raise HTTPException(status_code=400, detail="Unsupported memory sort order")

    rows = connection.execute(
        f"""
        SELECT * FROM memories
        WHERE conversation_id = ? AND archived_at IS NULL
        ORDER BY {sort} {order.upper()}
        """,
        (conversation_id,),
    ).fetchall()
    return [_public_memory(dict(row)) for row in rows]


def _insert_message(
    connection,
    conversation_id: int,
    role: str,
    content: str,
    llm_provider: str = None,
    llm_model: str = None,
    image_data: str = None,
    image_media_type: str = None,
    generation_ms: int = None,
    include_history: int = None,
    participant_id: int = None,
) -> dict:
    cursor = connection.execute(
        """
        INSERT INTO messages (
            conversation_id, role, content, llm_provider, llm_model, image_data, image_media_type,
            generation_ms, include_history, participant_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            conversation_id,
            role,
            content,
            llm_provider,
            llm_model,
            image_data,
            image_media_type,
            generation_ms,
            include_history,
            participant_id,
        ),
    )
    connection.execute(
        "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (conversation_id,),
    )
    return row_to_dict(connection.execute("SELECT * FROM messages WHERE id = ?", (cursor.lastrowid,)).fetchone())


def _clean_generated_memory_title(title: str, content: str) -> str:
    cleaned = next((line.strip(" -\t") for line in title.splitlines() if line.strip()), "")
    cleaned = cleaned.strip("\"'`")
    if cleaned.lower().startswith("title:"):
        cleaned = cleaned.split(":", 1)[1].strip()
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        return fallback_memory_title(content)
    return cleaned[:72].rstrip(" .,:;-") or fallback_memory_title(content)


def _generate_memory_title(connection, content: str, conversation_id: int = None) -> str:
    fallback_title = fallback_memory_title(content)
    try:
        config = (
            _conversation_llm_model(connection, _conversation_or_404(connection, conversation_id))
            if conversation_id is not None
            else _active_llm_model(connection)
        )
        generated_title = llm_chat(
            provider=config["provider"],
            base_url=config["base_url"],
            model=config["model"],
            api_key=config["api_key"],
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Create concise, descriptive titles for saved chat memories. "
                        "Return only the title, with no quotes, bullets, or explanation."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Write a specific title of 3 to 8 words for this memory:\n\n"
                        f"{content[:4000]}"
                    ),
                },
            ],
        )
        return _clean_generated_memory_title(generated_title, content)
    except Exception:
        return fallback_title


def _finalize_memory_title(memory_id: int, content: str, conversation_id: int) -> None:
    with get_connection() as connection:
        title = _generate_memory_title(connection, content, conversation_id)
        connection.execute(
            """
            UPDATE memories
            SET title = ?, title_generated_at = CURRENT_TIMESTAMP
            WHERE id = ? AND archived_at IS NULL
            """,
            (title, memory_id),
        )


def _schedule_memory_title_generation(
    background_tasks: BackgroundTasks,
    memory: dict,
    conversation_id: int,
) -> dict:
    if memory.get("title_pending"):
        background_tasks.add_task(
            _finalize_memory_title,
            memory["id"],
            memory["content"],
            conversation_id,
        )
    return memory


def _generate_integrated_memory(connection, rows: List[dict], target_conversation_id: int = None) -> str:
    fallback = "\n\n".join(row["content"] for row in rows).strip()
    try:
        config = (
            _conversation_llm_model(connection, _conversation_or_404(connection, target_conversation_id))
            if target_conversation_id is not None
            else _active_llm_model(connection)
        )
        memory_text = "\n\n".join(
            f"{index}. {row['title']}\n{row['content']}"
            for index, row in enumerate(rows, start=1)
        )
        integrated = llm_chat(
            provider=config["provider"],
            base_url=config["base_url"],
            model=config["model"],
            api_key=config["api_key"],
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Integrate saved chat memories into one concise, durable memory. "
                        "Preserve important facts and conclusions, remove duplication, and return only the integrated memory text."
                    ),
                },
                {"role": "user", "content": f"Integrate these memories:\n\n{memory_text[:6000]}"},
            ],
        ).strip()
        return integrated or fallback
    except Exception:
        return fallback


def _oldest_conversation_id_for_memories(connection, rows: List[dict]) -> int:
    conversation_ids = sorted({row["conversation_id"] for row in rows})
    placeholders = ",".join("?" for _ in conversation_ids)
    conversation = connection.execute(
        f"""
        SELECT id FROM conversations
        WHERE id IN ({placeholders})
        ORDER BY created_at ASC, id ASC
        LIMIT 1
        """,
        conversation_ids,
    ).fetchone()
    return conversation["id"]


def _memory_model_from_sources(connection, source_message_id=None, fallback_rows=None, conversation_id: int = None) -> tuple:
    if source_message_id is not None:
        source = row_to_dict(
            connection.execute(
                "SELECT llm_provider, llm_model FROM messages WHERE id = ?",
                (source_message_id,),
            ).fetchone()
        )
        if source and source.get("llm_model"):
            return source.get("llm_provider"), source.get("llm_model")

    rows = fallback_rows or []
    models = {
        (row["llm_provider"], row["llm_model"])
        for row in rows
        if row["llm_model"]
    }
    if len(models) == 1:
        return next(iter(models))
    if len(models) > 1:
        return "mixed", "Multiple models"

    try:
        config = (
            _conversation_llm_model(connection, _conversation_or_404(connection, conversation_id))
            if conversation_id is not None
            else _active_llm_model(connection)
        )
        return config["provider"], config["model"]
    except HTTPException:
        return None, None


def _insert_memory(
    connection,
    conversation_id: int,
    content: str,
    source_message_id=None,
    llm_provider: str = None,
    llm_model: str = None,
    fallback_model_rows=None,
) -> dict:
    if not llm_model:
        llm_provider, llm_model = _memory_model_from_sources(connection, source_message_id, fallback_model_rows, conversation_id)
    title = fallback_memory_title(content)
    cursor = connection.execute(
        """
        INSERT INTO memories (
            conversation_id, title, content, source_message_id, llm_provider, llm_model, title_generated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, NULL)
        """,
        (conversation_id, title, content, source_message_id, llm_provider, llm_model),
    )
    connection.execute(
        "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (conversation_id,),
    )
    memory = connection.execute("SELECT * FROM memories WHERE id = ?", (cursor.lastrowid,)).fetchone()
    return _public_memory(dict(memory))


def _latest_assistant_message(connection, conversation_id: int) -> dict:
    message = row_to_dict(
        connection.execute(
            """
            SELECT * FROM messages
            WHERE conversation_id = ? AND role = 'assistant'
            ORDER BY id DESC
            LIMIT 1
            """,
            (conversation_id,),
        ).fetchone()
    )
    if not message:
        raise HTTPException(status_code=400, detail="No assistant message to remember yet")
    return message


def _message_or_404(connection, conversation_id: int, message_id: int) -> dict:
    message = row_to_dict(
        connection.execute(
            "SELECT * FROM messages WHERE id = ? AND conversation_id = ?",
            (message_id, conversation_id),
        ).fetchone()
    )
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
    return message


def _active_llm_model(connection) -> dict:
    config = row_to_dict(
        connection.execute(
            """
            SELECT * FROM llm_models
            WHERE is_active = 1
            ORDER BY id
            LIMIT 1
            """
        ).fetchone()
    )
    if not config:
        raise HTTPException(status_code=500, detail="Active LLM model is missing")
    return config


def _llm_model_or_404(connection, model_id: int) -> dict:
    model = row_to_dict(
        connection.execute("SELECT * FROM llm_models WHERE id = ?", (model_id,)).fetchone()
    )
    if not model:
        raise HTTPException(status_code=404, detail="LLM model not found")
    return model


def _conversation_llm_model(connection, conversation: dict) -> dict:
    model_id = conversation.get("llm_model_id")
    if model_id is not None:
        model = row_to_dict(
            connection.execute("SELECT * FROM llm_models WHERE id = ?", (model_id,)).fetchone()
        )
        if model:
            return model
    return _active_llm_model(connection)


def _mask_api_key(api_key: str | None) -> str | None:
    if not api_key:
        return None
    if len(api_key) <= 6:
        return "*" * len(api_key)
    return f"{api_key[:3]}{'*' * (len(api_key) - 6)}{api_key[-3:]}"


def _public_model(connection, config: dict) -> dict:
    comments = (config.get("comments") or "").strip() or None
    tts_voice_uri = (config.get("tts_voice_uri") or "").strip() or None
    return {
        "id": config["id"],
        "provider": config["provider"],
        "base_url": config["base_url"],
        "model": config["model"],
        "comments": comments,
        "tts_voice_uri": tts_voice_uri,
        "has_api_key": bool(config["api_key"]),
        "api_key_preview": _mask_api_key(config.get("api_key")),
        "is_active": bool(config["is_active"]),
        "updated_at": config["updated_at"],
        **_model_generation_timing(connection, config["provider"], config["model"]),
    }


def _public_config(connection) -> dict:
    rows = connection.execute("SELECT * FROM llm_models ORDER BY is_active DESC, id ASC").fetchall()
    models = [_public_model(connection, dict(row)) for row in rows]
    active_model = next((model for model in models if model["is_active"]), models[0] if models else None)
    if not active_model:
        raise HTTPException(status_code=500, detail="LLM config is missing")
    return {"models": models, "active_model": active_model}


def _normalize_image_payload(image_data: str, image_media_type: str) -> tuple[str, str]:
    media_type = image_media_type.strip().lower()
    if media_type not in ALLOWED_IMAGE_MEDIA_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported image type. Use JPEG, PNG, GIF, or WebP.")

    payload = image_data.strip()
    if payload.startswith("data:"):
        header, _, encoded = payload.partition(",")
        if encoded:
            payload = encoded
        if not media_type and ";" in header:
            media_type = header[5:].split(";", 1)[0].strip().lower()

    try:
        raw = base64.b64decode(payload, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image data") from exc

    if not raw:
        raise HTTPException(status_code=400, detail="Invalid image data")
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="Image must be 10 MB or smaller")

    return base64.b64encode(raw).decode("ascii"), media_type


def _format_message_for_llm(row: dict, provider: str, *, speaker_label: str = None) -> dict:
    role = row["role"]
    content = row["content"]
    if speaker_label and role == "assistant":
        content = f"[{speaker_label}]: {content}"
    image_data = row.get("image_data")
    image_media_type = row.get("image_media_type")

    if role != "user" or not image_data:
        return {"role": role, "content": content}

    prompt = content.strip() or DEFAULT_IMAGE_PROMPT
    normalized_provider = provider.lower().strip().replace("_", "-")
    if normalized_provider == "ollama":
        return {"role": "user", "content": prompt, "images": [image_data]}

    return {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{image_media_type or 'image/jpeg'};base64,{image_data}"},
            },
        ],
    }


def _history_limit_from_include_history(include_history: Union[bool, int]) -> int:
    if include_history is False or include_history == 0:
        return 0
    if include_history is True:
        return 40
    if isinstance(include_history, int) and include_history > 0:
        return min(include_history, 40)
    return 40


def _include_history_enabled(include_history: Union[bool, int]) -> bool:
    return _history_limit_from_include_history(include_history) > 0


def _stored_include_history(include_history: Union[bool, int]) -> int:
    if include_history is False or include_history == 0:
        return 0
    if include_history is True:
        return -1
    if isinstance(include_history, int):
        return include_history
    return -1


def _answer_length_constraint(answer_length: int) -> str | None:
    level = max(1, min(5, int(answer_length)))
    if level >= 5:
        return None
    instructions = {
        1: "Reply in exactly 1 sentence. Do not use more than one sentence.",
        2: "Reply in about 2–3 sentences.",
        3: "Reply in about 3–4 sentences.",
        4: "Reply in about 4–5 sentences.",
    }
    return instructions[level]


def _answer_length_priority_message(answer_length: int) -> str | None:
    constraint = _answer_length_constraint(answer_length)
    if not constraint:
        return None
    return (
        "HIGHEST PRIORITY — response length:\n"
        f"{constraint}\n"
        "This length requirement overrides your usual verbosity, personality, and other instructions. "
        "Stay within this limit even when the topic feels like it deserves more detail."
    )


def _memories_excluding_history_duplicates(memories: List[dict], history_rows: List[dict]) -> List[dict]:
    history_ids = {row["id"] for row in history_rows if row.get("id") is not None}
    history_contents = {
        (row.get("content") or "").strip()
        for row in history_rows
        if (row.get("content") or "").strip()
    }
    included = []
    for memory in memories:
        source_id = memory.get("source_message_id")
        if source_id is not None and source_id in history_ids:
            continue
        content = (memory.get("content") or "").strip()
        if content and content in history_contents:
            continue
        included.append(memory)
    return included


def _participant_speaker_label(connection, participant_id: int, participants_by_id: dict) -> str:
    if participant_id in participants_by_id:
        return participant_display_name(participants_by_id[participant_id])
    participant = row_to_dict(
        connection.execute(
            """
            SELECT cp.id, cp.personality, cp.name, cp.tts_voice_uri, lm.model AS llm_model, lm.comments AS llm_comments
            FROM conversation_participants cp
            JOIN llm_models lm ON lm.id = cp.llm_model_id
            WHERE cp.id = ?
            """,
            (participant_id,),
        ).fetchone()
    )
    if participant:
        participants_by_id[participant_id] = participant
        return participant_display_name(participant)
    return "Assistant"


def _build_llm_messages(
    connection,
    conversation_id: int,
    provider: str,
    *,
    include_history: Union[bool, int] = True,
    include_memories: bool = True,
    include_all_memories: bool = False,
    pending_user: dict = None,
    participant: dict = None,
    answer_length: int = 3,
) -> tuple[List[dict], int, int]:
    memories = (
        _active_memories(connection, conversation_id, sort="created_at", order="asc")
        if include_memories
        else []
    )
    all_memories = (
        _all_active_memories(
            connection,
            sort="created_at",
            order="asc",
            exclude_conversation_id=conversation_id,
        )
        if include_all_memories
        else []
    )
    history_limit = _history_limit_from_include_history(include_history)
    if history_limit > 0:
        history = connection.execute(
            """
            SELECT id, role, content, image_data, image_media_type, participant_id FROM messages
            WHERE conversation_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (conversation_id, history_limit),
        ).fetchall()
        history_rows = [dict(row) for row in reversed(history)]
    else:
        history_rows = []
        if not pending_user:
            history = connection.execute(
                """
                SELECT id, role, content, image_data, image_media_type, participant_id FROM messages
                WHERE conversation_id = ? AND role = 'user'
                ORDER BY id DESC
                LIMIT 1
                """,
                (conversation_id,),
            ).fetchall()
            history_rows = [dict(row) for row in reversed(history)]

    if pending_user:
        history_rows.append(pending_user)

    included_memories = _memories_excluding_history_duplicates(memories, history_rows)
    included_all_memories = _memories_excluding_history_duplicates(all_memories, history_rows)

    prompt_config = _prompt_config_or_default(connection)
    messages = [
        {
            "role": "system",
            "content": prompt_config["default_prompt"],
        }
    ]
    length_priority = _answer_length_priority_message(answer_length)
    if length_priority:
        messages.append({"role": "system", "content": length_priority})
    personality = (participant.get("personality") or "").strip() if participant else ""
    if personality:
        messages.append(
            {
                "role": "system",
                "content": f"Your perspective and role:\n{personality}",
            }
        )
    if included_memories:
        memory_text = "\n".join(f"- {memory['content']}" for memory in included_memories)
        messages.append({"role": "system", "content": f"Conversation memory bank:\n{memory_text}"})
    if included_all_memories:
        all_memory_text = "\n".join(f"- {memory['content']}" for memory in included_all_memories)
        messages.append({"role": "user", "content": f"User memories:\n{all_memory_text}"})

    participants_by_id = {}
    all_participants = []
    if participant:
        all_participants = load_participants(connection, conversation_id)
        for item in all_participants:
            participants_by_id[item["id"]] = item

    for row in history_rows:
        speaker_label = None
        if row.get("role") == "assistant" and row.get("participant_id"):
            speaker_label = _participant_speaker_label(connection, row["participant_id"], participants_by_id)
        messages.append(_format_message_for_llm(row, provider, speaker_label=speaker_label))

    if participant and len(all_participants) > 1:
        def resolve_participant_name(participant_id: int) -> str:
            return _participant_speaker_label(connection, participant_id, participants_by_id)

        previous_speaker_label, _previous_row = previous_discussion_speaker(
            history_rows,
            resolve_participant_name=resolve_participant_name,
        )
        messages.append(
            {
                "role": "system",
                "content": build_discussion_response_instruction(
                    participant,
                    all_participants,
                    previous_speaker_label,
                ),
            }
        )

    length_constraint = _answer_length_constraint(answer_length)
    if length_constraint:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Final reminder — highest priority before you reply:\n"
                    f"{length_constraint}"
                ),
            }
        )

    return messages, len(included_memories), len(included_all_memories)


def _extract_llm_message_text(message: dict) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return " ".join(parts).strip()
    return str(content)


def _extract_llm_message_image_bytes(message: dict) -> int:
    total = 0
    images = message.get("images")
    if isinstance(images, list):
        for image in images:
            if isinstance(image, str):
                total += int(len(image) * 3 / 4)
    content = message.get("content", "")
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            url = part.get("image_url", {}).get("url", "")
            if isinstance(url, str) and "base64," in url:
                total += int(len(url.split("base64,", 1)[1]) * 3 / 4)
    return total


def _label_llm_message(message: dict) -> str:
    role = message.get("role", "")
    content = _extract_llm_message_text(message)
    if role == "system":
        if content.startswith("Conversation memory bank:"):
            return "Memory bank"
        return "System prompt"
    if role == "user":
        if content.startswith("User memories:"):
            return "User memories"
        return "User message"
    if role == "assistant":
        return "Assistant reply"
    return role.title() or "Message"


def _summarize_llm_context(
    llm_messages: List[dict],
    *,
    provider: str,
    model: str,
    include_history: Union[bool, int],
    include_memories: bool,
    include_all_memories: bool,
    memory_count: int,
    all_memory_count: int,
) -> dict:
    items = []
    total_chars = 0
    image_count = 0
    history_message_count = 0
    images_resent_from_history = False
    user_indices = [index for index, message in enumerate(llm_messages) if message.get("role") == "user"]
    last_user_index = user_indices[-1] if user_indices else -1

    for index, message in enumerate(llm_messages):
        text = _extract_llm_message_text(message)
        has_image = bool(message.get("images")) or any(
            isinstance(part, dict) and part.get("type") == "image_url"
            for part in (message.get("content") if isinstance(message.get("content"), list) else [])
        )
        image_bytes = _extract_llm_message_image_bytes(message)
        char_estimate = len(json.dumps(message, ensure_ascii=False))
        preview = text.strip()
        if len(preview) > 220:
            preview = preview[:220].rstrip() + "..."
        if not preview and has_image:
            preview = "(image attachment)"

        role = message.get("role", "")
        if role in {"user", "assistant"}:
            history_message_count += 1
        if _include_history_enabled(include_history) and has_image and role == "user" and index != last_user_index:
            images_resent_from_history = True

        if has_image:
            image_count += 1

        total_chars += char_estimate
        items.append(
            {
                "role": role,
                "label": _label_llm_message(message),
                "content_preview": preview,
                "has_image": has_image,
                "image_bytes": image_bytes,
                "char_estimate": char_estimate,
            }
        )

    return {
        "provider": provider,
        "model": model,
        "include_history": include_history,
        "include_memories": include_memories,
        "include_all_memories": include_all_memories,
        "memory_count": memory_count,
        "all_memory_count": all_memory_count,
        "items": items,
        "total_chars": total_chars,
        "approx_tokens": max(1, total_chars // 4),
        "image_count": image_count,
        "history_message_count": history_message_count,
        "images_resent_from_history": images_resent_from_history and _include_history_enabled(include_history),
    }


def _get_generation_stats(connection, provider: str, model: str) -> Optional[dict]:
    return row_to_dict(
        connection.execute(
            """
            SELECT sample_count, total_generation_ms, total_context_chars, total_output_chars
            FROM llm_generation_stats
            WHERE provider = ? AND model = ?
            """,
            (provider, model),
        ).fetchone()
    )


def _seconds_per_char(stats: Optional[dict]) -> Optional[float]:
    if not stats or stats["sample_count"] <= 0:
        return None
    total_chars = int(stats["total_context_chars"]) + int(stats["total_output_chars"])
    if total_chars <= 0:
        return None
    return (int(stats["total_generation_ms"]) / 1000.0) / total_chars


def _estimate_generation_seconds(stats: Optional[dict], context_chars: int) -> Optional[float]:
    rate = _seconds_per_char(stats)
    if rate is None:
        return None
    avg_output_chars = max(1, int(stats["total_output_chars"]) // int(stats["sample_count"]))
    char_estimate = max(1, int(context_chars)) + avg_output_chars
    return max(1.0, rate * char_estimate)


def _generation_timing_metadata(connection, provider: str, model: str, context_chars: int) -> dict:
    stats = _get_generation_stats(connection, provider, model)
    rate = _seconds_per_char(stats)
    return {
        "generation_estimate_sec": _estimate_generation_seconds(stats, context_chars),
        "seconds_per_char": rate,
        "generation_sample_count": int(stats["sample_count"]) if stats else 0,
    }


def _model_generation_timing(connection, provider: str, model: str) -> dict:
    stats = _get_generation_stats(connection, provider, model)
    rate = _seconds_per_char(stats)
    if not stats or stats["sample_count"] <= 0:
        return {
            "generation_sample_count": 0,
            "seconds_per_char": None,
            "avg_generation_sec": None,
            "reference_generation_estimate_sec": None,
        }
    sample_count = int(stats["sample_count"])
    avg_generation_sec = round(int(stats["total_generation_ms"]) / sample_count / 1000.0, 2)
    avg_context_chars = int(int(stats["total_context_chars"]) / sample_count)
    return {
        "generation_sample_count": sample_count,
        "seconds_per_char": rate,
        "avg_generation_sec": avg_generation_sec,
        "reference_generation_estimate_sec": _estimate_generation_seconds(stats, avg_context_chars),
    }


def _reset_generation_stats(connection, provider: str, model: str) -> None:
    connection.execute(
        "DELETE FROM llm_generation_stats WHERE provider = ? AND model = ?",
        (provider, model),
    )


def _update_generation_stats(
    connection,
    provider: str,
    model: str,
    generation_ms: int,
    context_chars: int,
    output_chars: int,
) -> None:
    context_chars = max(1, int(context_chars))
    output_chars = max(1, int(output_chars))
    connection.execute(
        """
        INSERT INTO llm_generation_stats (
            provider, model, sample_count, total_generation_ms, total_context_chars, total_output_chars
        )
        VALUES (?, ?, 1, ?, ?, ?)
        ON CONFLICT(provider, model) DO UPDATE SET
            sample_count = sample_count + 1,
            total_generation_ms = total_generation_ms + excluded.total_generation_ms,
            total_context_chars = total_context_chars + excluded.total_context_chars,
            total_output_chars = total_output_chars + excluded.total_output_chars,
            updated_at = CURRENT_TIMESTAMP
        """,
        (provider, model, generation_ms, context_chars, output_chars),
    )


def _attach_generation_estimate(connection, summary: dict) -> dict:
    summary.update(
        _generation_timing_metadata(
            connection,
            summary["provider"],
            summary["model"],
            summary["total_chars"],
        )
    )
    return summary


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/agent-profiles", response_model=List[AgentProfileRead])
def list_agent_profiles() -> List[dict]:
    with get_connection() as connection:
        return load_agent_profiles(connection)


@app.post("/api/agent-profiles", response_model=AgentProfileRead)
def create_agent_profile_route(payload: AgentProfileCreate) -> dict:
    with get_connection() as connection:
        return create_agent_profile(connection, payload, _llm_model_or_404)


@app.put("/api/agent-profiles/{profile_id}", response_model=AgentProfileRead)
def update_agent_profile_route(profile_id: int, payload: AgentProfileUpdate) -> dict:
    with get_connection() as connection:
        return update_agent_profile(connection, profile_id, payload, _llm_model_or_404)


@app.delete("/api/agent-profiles/{profile_id}")
def delete_agent_profile_route(profile_id: int) -> dict:
    with get_connection() as connection:
        delete_agent_profile(connection, profile_id)
        return {"ok": True}


@app.post("/api/stt/transcribe")
async def transcribe_speech(file: UploadFile = File(...)) -> dict:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio file")
    if len(data) > 25 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Audio file is too large")

    try:
        text = transcribe_audio_bytes(data, file.filename or "audio.webm")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Speech transcription failed: {exc}") from exc

    return {"text": text}


def _prompt_config_or_default(connection) -> dict:
    row = row_to_dict(connection.execute("SELECT * FROM app_config WHERE id = 1").fetchone())
    if not row:
        connection.execute(
            "INSERT INTO app_config (id, default_prompt) VALUES (1, ?)",
            (DEFAULT_SYSTEM_PROMPT,),
        )
        row = row_to_dict(connection.execute("SELECT * FROM app_config WHERE id = 1").fetchone())
    return row


@app.get("/api/config/prompt", response_model=PromptConfigRead)
def get_prompt_config() -> dict:
    with get_connection() as connection:
        config = _prompt_config_or_default(connection)
        return {
            "default_prompt": config["default_prompt"],
            "default_prompt_baseline": DEFAULT_SYSTEM_PROMPT,
            "updated_at": config["updated_at"],
        }


@app.put("/api/config/prompt", response_model=PromptConfigRead)
def update_prompt_config(payload: PromptConfigUpdate) -> dict:
    prompt = payload.default_prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Default prompt cannot be empty")

    with get_connection() as connection:
        _prompt_config_or_default(connection)
        connection.execute(
            """
            UPDATE app_config
            SET default_prompt = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = 1
            """,
            (prompt,),
        )
        config = row_to_dict(connection.execute("SELECT * FROM app_config WHERE id = 1").fetchone())
        return {
            "default_prompt": config["default_prompt"],
            "default_prompt_baseline": DEFAULT_SYSTEM_PROMPT,
            "updated_at": config["updated_at"],
        }


def _speech_config_or_default(connection) -> dict:
    row = row_to_dict(connection.execute("SELECT * FROM speech_config WHERE id = 1").fetchone())
    if not row:
        default_model = os.getenv("WHISPER_MODEL", "base.en")
        connection.execute(
            "INSERT INTO speech_config (id, whisper_model) VALUES (1, ?)",
            (default_model,),
        )
        row = row_to_dict(connection.execute("SELECT * FROM speech_config WHERE id = 1").fetchone())
    return row


@app.get("/api/config/speech", response_model=SpeechConfigRead)
def get_speech_config() -> dict:
    with get_connection() as connection:
        config = _speech_config_or_default(connection)
        return {
            "whisper_model": config["whisper_model"],
            "whisper_model_options": WHISPER_MODEL_OPTIONS,
            "updated_at": config["updated_at"],
        }


@app.put("/api/config/speech", response_model=SpeechConfigRead)
def update_speech_config(payload: SpeechConfigUpdate) -> dict:
    model_name = payload.whisper_model.strip()
    if model_name not in WHISPER_MODEL_OPTIONS:
        raise HTTPException(status_code=400, detail="Unsupported Whisper model")

    with get_connection() as connection:
        _speech_config_or_default(connection)
        connection.execute(
            """
            UPDATE speech_config
            SET whisper_model = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = 1
            """,
            (model_name,),
        )
        set_whisper_model(model_name)
        config = row_to_dict(connection.execute("SELECT * FROM speech_config WHERE id = 1").fetchone())
        return {
            "whisper_model": config["whisper_model"],
            "whisper_model_options": WHISPER_MODEL_OPTIONS,
            "updated_at": config["updated_at"],
        }


@app.get("/api/conversations", response_model=List[Conversation])
def list_conversations() -> List[dict]:
    with get_connection() as connection:
        rows = connection.execute("SELECT * FROM conversations ORDER BY sort_order ASC, updated_at DESC, id DESC").fetchall()
        result = []
        for row in rows:
            conversation = dict(row)
            count = connection.execute(
                "SELECT COUNT(*) AS count FROM conversation_participants WHERE conversation_id = ?",
                (conversation["id"],),
            ).fetchone()["count"]
            conversation["participant_count"] = int(count)
            result.append(conversation)
        return result


@app.post("/api/conversations", response_model=Conversation)
def create_conversation(payload: ConversationCreate) -> dict:
    title = (payload.title or "New Conversation").strip() or "New Conversation"
    with get_connection() as connection:
        active_model = _active_llm_model(connection)
        next_sort_order = connection.execute(
            "SELECT COALESCE(MIN(sort_order), 0) - 1 AS sort_order FROM conversations"
        ).fetchone()["sort_order"]
        cursor = connection.execute(
            "INSERT INTO conversations (title, sort_order, llm_model_id) VALUES (?, ?, ?)",
            (title, next_sort_order, active_model["id"]),
        )
        conversation_id = cursor.lastrowid
        if payload.participants:
            replace_participants(connection, conversation_id, payload.participants, _llm_model_or_404)
        return _public_conversation(connection, conversation_id)


@app.put("/api/conversations/{conversation_id}/title", response_model=Conversation)
def update_conversation_title(conversation_id: int, payload: ConversationTitleUpdate) -> dict:
    title = payload.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title cannot be empty")

    with get_connection() as connection:
        _conversation_or_404(connection, conversation_id)
        connection.execute(
            """
            UPDATE conversations
            SET title = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (title, conversation_id),
        )
        return _public_conversation(connection, conversation_id)


@app.put("/api/conversations/{conversation_id}/model", response_model=Conversation)
def update_conversation_model(conversation_id: int, payload: ConversationModelUpdate) -> dict:
    with get_connection() as connection:
        _conversation_or_404(connection, conversation_id)
        _llm_model_or_404(connection, payload.llm_model_id)
        connection.execute(
            """
            UPDATE conversations
            SET llm_model_id = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (payload.llm_model_id, conversation_id),
        )
        return _public_conversation(connection, conversation_id)


@app.put("/api/conversations/reorder", response_model=List[Conversation])
def reorder_conversations(payload: ConversationReorder) -> List[dict]:
    conversation_ids = list(dict.fromkeys(payload.conversation_ids))
    with get_connection() as connection:
        existing_ids = {
            row["id"]
            for row in connection.execute("SELECT id FROM conversations").fetchall()
        }
        if set(conversation_ids) != existing_ids:
            raise HTTPException(status_code=400, detail="Reorder payload must include every conversation exactly once")

        for sort_order, conversation_id in enumerate(conversation_ids):
            connection.execute(
                "UPDATE conversations SET sort_order = ? WHERE id = ?",
                (sort_order, conversation_id),
            )

        rows = connection.execute("SELECT * FROM conversations ORDER BY sort_order ASC, updated_at DESC, id DESC").fetchall()
        return [dict(row) for row in rows]


@app.post("/api/conversations/{conversation_id}/delete")
def delete_conversation(conversation_id: int, payload: ConversationDelete) -> dict:
    with get_connection() as connection:
        _conversation_or_404(connection, conversation_id)
        memory_action = payload.memory_action.strip().lower()

        if memory_action == "move":
            if payload.target_conversation_id is None:
                raise HTTPException(status_code=400, detail="Choose a destination conversation for memories")
            if payload.target_conversation_id == conversation_id:
                raise HTTPException(status_code=400, detail="Destination conversation must be different")
            _conversation_or_404(connection, payload.target_conversation_id)
            connection.execute(
                """
                UPDATE memories
                SET conversation_id = ?, source_message_id = NULL
                WHERE conversation_id = ?
                """,
                (payload.target_conversation_id, conversation_id),
            )
            connection.execute(
                "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (payload.target_conversation_id,),
            )
        elif memory_action != "delete":
            raise HTTPException(status_code=400, detail="memory_action must be 'delete' or 'move'")

        connection.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        return {"ok": True}


@app.get("/api/conversations/{conversation_id}", response_model=ConversationDetail)
def get_conversation(conversation_id: int) -> dict:
    with get_connection() as connection:
        conversation = _conversation_or_404(connection, conversation_id)
        messages = [dict(row) for row in connection.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id ASC",
            (conversation_id,),
        ).fetchall()]
        memories = _active_memories(connection, conversation_id)
        participants = [public_participant(row) for row in load_participants(connection, conversation_id)]
        return {"conversation": conversation, "messages": messages, "memories": memories, "participants": participants}


@app.put("/api/conversations/{conversation_id}/participants", response_model=List[ConversationParticipantRead])
def update_conversation_participants(conversation_id: int, payload: ConversationParticipantsUpdate) -> List[dict]:
    with get_connection() as connection:
        _conversation_or_404(connection, conversation_id)
        rows = replace_participants(connection, conversation_id, payload.participants, _llm_model_or_404)
        return [public_participant(row) for row in rows]


@app.post("/api/conversations/{conversation_id}/llm-context-preview", response_model=LlmContextPreview)
def preview_llm_context(conversation_id: int, payload: MessageCreate) -> dict:
    content = payload.content.strip()
    image_data = None
    image_media_type = None
    if payload.image_data:
        image_data, image_media_type = _normalize_image_payload(payload.image_data, payload.image_media_type or "")

    pending_user = {
        "role": "user",
        "content": content,
        "image_data": image_data,
        "image_media_type": image_media_type,
    }

    with get_connection() as connection:
        conversation = _conversation_or_404(connection, conversation_id)
        participants = load_participants(connection, conversation_id)
        multi_agent_note = None
        if participants:
            config = _llm_model_or_404(connection, participants[0]["llm_model_id"])
            multi_agent_note = (
                f"Multi-agent preview for first participant ({config['model']}). "
                "Other participants use the same shared context with different models and personalities."
            )
        else:
            config = _conversation_llm_model(connection, conversation)
        llm_messages, included_memory_count, included_all_memory_count = _build_llm_messages(
            connection,
            conversation_id,
            config["provider"],
            include_history=payload.include_history,
            include_memories=payload.include_memories,
            include_all_memories=payload.include_all_memories,
            pending_user=pending_user,
            participant=participants[0] if participants else None,
            answer_length=payload.answer_length,
        )
        summary = _attach_generation_estimate(
            connection,
            _summarize_llm_context(
                llm_messages,
                provider=config["provider"],
                model=config["model"],
                include_history=payload.include_history,
                include_memories=payload.include_memories,
                include_all_memories=payload.include_all_memories,
                memory_count=included_memory_count,
                all_memory_count=included_all_memory_count,
            ),
        )
        if multi_agent_note:
            summary["multi_agent_note"] = multi_agent_note
        return summary


@app.post("/api/conversations/{conversation_id}/messages", response_model=MessageResponse)
def send_message(conversation_id: int, payload: MessageCreate, background_tasks: BackgroundTasks) -> dict:
    content = payload.content.strip()
    normalized = content.lower()
    image_data = None
    image_media_type = None
    if payload.image_data:
        image_data, image_media_type = _normalize_image_payload(payload.image_data, payload.image_media_type or "")

    with get_connection() as connection:
        conversation = _conversation_or_404(connection, conversation_id)

        if not image_data and normalized == "remember this message":
            source = _latest_assistant_message(connection, conversation_id)
            memory = _insert_memory(connection, conversation_id, source["content"], source["id"])
            _schedule_memory_title_generation(background_tasks, memory, conversation_id)
            return {"memory": memory}

        if not image_data and normalized.startswith("remember:"):
            memory_content = content.split(":", 1)[1].strip()
            if not memory_content:
                raise HTTPException(status_code=400, detail="Provide text after remember:")
            memory = _insert_memory(connection, conversation_id, memory_content)
            _schedule_memory_title_generation(background_tasks, memory, conversation_id)
            return {"memory": memory}

        message_content = content
        user_message = _insert_message(
            connection,
            conversation_id,
            "user",
            message_content,
            image_data=image_data,
            image_media_type=image_media_type,
        )
        if conversation["title"] == "New Conversation":
            title_source = content or "Image message"
            title = title_source[:60] + ("..." if len(title_source) > 60 else "")
            connection.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conversation_id))

        participants = load_participants(connection, conversation_id)
        if participants:
            discussion_rounds = clamp_discussion_rounds(payload.discussion_rounds)
            assistant_messages = run_multi_agent_turns(
                connection,
                conversation_id,
                participants,
                discussion_rounds=discussion_rounds,
                include_history=payload.include_history,
                include_memories=payload.include_memories,
                include_all_memories=payload.include_all_memories,
                answer_length=payload.answer_length,
                build_llm_messages=_build_llm_messages,
                insert_message=_insert_message,
                update_generation_stats=_update_generation_stats,
                summarize_llm_context=_summarize_llm_context,
                llm_model_or_404=_llm_model_or_404,
                stored_include_history=_stored_include_history,
                commit_connection=lambda conn: conn.commit(),
            )
            last_message = assistant_messages[-1] if assistant_messages else None
            return {
                "user_message": user_message,
                "assistant_message": last_message,
                "assistant_messages": assistant_messages,
            }

        config = _conversation_llm_model(connection, conversation)
        llm_messages, included_memory_count, included_all_memory_count = _build_llm_messages(
            connection,
            conversation_id,
            config["provider"],
            include_history=payload.include_history,
            include_memories=payload.include_memories,
            include_all_memories=payload.include_all_memories,
            answer_length=payload.answer_length,
        )
        context_summary = _summarize_llm_context(
            llm_messages,
            provider=config["provider"],
            model=config["model"],
            include_history=payload.include_history,
            include_memories=payload.include_memories,
            include_all_memories=payload.include_all_memories,
            memory_count=included_memory_count,
            all_memory_count=included_all_memory_count,
        )
        started_at = time.perf_counter()
        try:
            assistant_content = llm_chat(
                provider=config["provider"],
                base_url=config["base_url"],
                model=config["model"],
                api_key=config["api_key"],
                messages=llm_messages,
            )
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        generation_ms = max(0, int((time.perf_counter() - started_at) * 1000))
        _update_generation_stats(
            connection,
            config["provider"],
            config["model"],
            generation_ms,
            context_summary["total_chars"],
            len(assistant_content),
        )

        assistant_message = _insert_message(
            connection,
            conversation_id,
            "assistant",
            assistant_content,
            llm_provider=config["provider"],
            llm_model=config["model"],
            generation_ms=generation_ms,
            include_history=_stored_include_history(payload.include_history),
        )
        return {"user_message": user_message, "assistant_message": assistant_message}


@app.delete("/api/conversations/{conversation_id}/messages/{message_id}")
def delete_message(conversation_id: int, message_id: int) -> dict:
    with get_connection() as connection:
        _conversation_or_404(connection, conversation_id)
        _message_or_404(connection, conversation_id, message_id)
        connection.execute("DELETE FROM messages WHERE id = ?", (message_id,))
        connection.execute(
            "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (conversation_id,),
        )
        return {"ok": True}


@app.post("/api/conversations/{conversation_id}/remember", response_model=Memory)
def remember_message(conversation_id: int, payload: RememberCreate, background_tasks: BackgroundTasks) -> dict:
    with get_connection() as connection:
        _conversation_or_404(connection, conversation_id)
        if payload.message_id is not None:
            source = _message_or_404(connection, conversation_id, payload.message_id)
            content = (payload.content or "").strip() or source["content"]
            memory = _insert_memory(connection, conversation_id, content, source["id"])
        else:
            content = (payload.content or "").strip()
            if content:
                memory = _insert_memory(connection, conversation_id, content)
            else:
                source = _latest_assistant_message(connection, conversation_id)
                memory = _insert_memory(connection, conversation_id, source["content"], source["id"])
    return _schedule_memory_title_generation(background_tasks, memory, conversation_id)


@app.get("/api/conversations/{conversation_id}/memories", response_model=List[Memory])
def list_memories(
    conversation_id: int,
    sort: str = Query("created_at", pattern="^(created_at|content|title|llm_model)$"),
    order: str = Query("desc", pattern="^(asc|desc)$"),
) -> List[dict]:
    with get_connection() as connection:
        _conversation_or_404(connection, conversation_id)
        return _active_memories(connection, conversation_id, sort=sort, order=order)


@app.get("/api/memories", response_model=List[MemoryGroup])
def list_memory_groups(
    sort: str = Query("created_at", pattern="^(created_at|content|title|llm_model)$"),
    order: str = Query("desc", pattern="^(asc|desc)$"),
) -> List[dict]:
    with get_connection() as connection:
        conversations = connection.execute(
            "SELECT * FROM conversations ORDER BY sort_order ASC, updated_at DESC, id DESC"
        ).fetchall()
        return [
            {
                "conversation": dict(conversation),
                "memories": _active_memories(connection, conversation["id"], sort=sort, order=order),
            }
            for conversation in conversations
        ]


@app.put("/api/memories/{memory_id}/move", response_model=Memory)
def move_memory(memory_id: int, payload: MemoryMove) -> dict:
    with get_connection() as connection:
        _conversation_or_404(connection, payload.target_conversation_id)
        memory = row_to_dict(
            connection.execute(
                "SELECT * FROM memories WHERE id = ? AND archived_at IS NULL",
                (memory_id,),
            ).fetchone()
        )
        if not memory:
            raise HTTPException(status_code=404, detail="Memory not found")

        connection.execute(
            """
            UPDATE memories
            SET conversation_id = ?, source_message_id = NULL
            WHERE id = ?
            """,
            (payload.target_conversation_id, memory_id),
        )
        connection.execute(
            "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id IN (?, ?)",
            (memory["conversation_id"], payload.target_conversation_id),
        )
        return _public_memory(dict(connection.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()))


@app.delete("/api/conversations/{conversation_id}/memories/{memory_id}")
def delete_memory(conversation_id: int, memory_id: int) -> dict:
    with get_connection() as connection:
        _conversation_or_404(connection, conversation_id)
        cursor = connection.execute(
            "DELETE FROM memories WHERE id = ? AND conversation_id = ?",
            (memory_id, conversation_id),
        )
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Memory not found")
        return {"ok": True}


@app.post("/api/memories/merge", response_model=Memory)
def merge_memories_across_conversations(payload: CrossConversationMemoryMerge, background_tasks: BackgroundTasks) -> dict:
    memory_ids = sorted(set(payload.memory_ids))
    placeholders = ",".join("?" for _ in memory_ids)
    with get_connection() as connection:
        _conversation_or_404(connection, payload.target_conversation_id)
        rows = connection.execute(
            f"""
            SELECT * FROM memories
            WHERE archived_at IS NULL AND id IN ({placeholders})
            """,
            memory_ids,
        ).fetchall()
        if len(rows) != len(memory_ids):
            raise HTTPException(status_code=400, detail="One or more memories were not found")

        merged = _insert_memory(
            connection,
            payload.target_conversation_id,
            payload.content.strip(),
            fallback_model_rows=rows,
        )
        connection.execute(
            f"""
            UPDATE memories
            SET archived_at = CURRENT_TIMESTAMP
            WHERE id IN ({placeholders})
            """,
            memory_ids,
        )
        touched_conversation_ids = {row["conversation_id"] for row in rows}
        touched_conversation_ids.add(payload.target_conversation_id)
        touched_placeholders = ",".join("?" for _ in touched_conversation_ids)
        connection.execute(
            f"UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id IN ({touched_placeholders})",
            list(touched_conversation_ids),
        )
        return _schedule_memory_title_generation(background_tasks, merged, payload.target_conversation_id)


@app.post("/api/memories/integrate", response_model=Memory)
def integrate_memories(payload: MemoryIntegrate, background_tasks: BackgroundTasks) -> dict:
    memory_ids = sorted(set(payload.memory_ids))
    if len(memory_ids) < 2:
        raise HTTPException(status_code=400, detail="Select at least two memories to integrate")
    placeholders = ",".join("?" for _ in memory_ids)
    with get_connection() as connection:
        rows = connection.execute(
            f"""
            SELECT * FROM memories
            WHERE archived_at IS NULL AND id IN ({placeholders})
            ORDER BY created_at ASC, id ASC
            """,
            memory_ids,
        ).fetchall()
        if len(rows) != len(memory_ids):
            raise HTTPException(status_code=400, detail="One or more memories were not found")

        target_conversation_id = _oldest_conversation_id_for_memories(connection, rows)
        integrated_content = _generate_integrated_memory(connection, rows, target_conversation_id)
        integrated = _insert_memory(
            connection,
            target_conversation_id,
            integrated_content,
            fallback_model_rows=rows,
        )
        touched_conversation_ids = {row["conversation_id"] for row in rows}
        touched_conversation_ids.add(target_conversation_id)
        touched_placeholders = ",".join("?" for _ in touched_conversation_ids)
        connection.execute(
            f"UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id IN ({touched_placeholders})",
            list(touched_conversation_ids),
        )
        return _schedule_memory_title_generation(background_tasks, integrated, target_conversation_id)


@app.post("/api/conversations/{conversation_id}/memories/merge", response_model=Memory)
def merge_memories(conversation_id: int, payload: MemoryMerge, background_tasks: BackgroundTasks) -> dict:
    memory_ids = sorted(set(payload.memory_ids))
    placeholders = ",".join("?" for _ in memory_ids)
    with get_connection() as connection:
        _conversation_or_404(connection, conversation_id)
        rows = connection.execute(
            f"""
            SELECT * FROM memories
            WHERE conversation_id = ? AND archived_at IS NULL AND id IN ({placeholders})
            """,
            [conversation_id, *memory_ids],
        ).fetchall()
        if len(rows) != len(memory_ids):
            raise HTTPException(status_code=400, detail="One or more memories were not found")

        merged = _insert_memory(connection, conversation_id, payload.content.strip(), fallback_model_rows=rows)
        connection.execute(
            f"""
            UPDATE memories
            SET archived_at = CURRENT_TIMESTAMP
            WHERE conversation_id = ? AND id IN ({placeholders})
            """,
            [conversation_id, *memory_ids],
        )
        return _schedule_memory_title_generation(background_tasks, merged, conversation_id)


@app.get("/api/config/llm", response_model=LlmConfigRead)
def get_llm_config() -> dict:
    with get_connection() as connection:
        return _public_config(connection)


@app.delete("/api/config/llm/models/{model_id}/generation-stats", response_model=LlmModelRead)
def reset_model_generation_stats(model_id: int) -> dict:
    with get_connection() as connection:
        model = _llm_model_or_404(connection, model_id)
        _reset_generation_stats(connection, model["provider"], model["model"])
        return _public_model(connection, model)


@app.put("/api/config/llm", response_model=LlmConfigRead)
def update_llm_config(payload: LlmConfigUpdate) -> dict:
    with get_connection() as connection:
        existing = {
            row["id"]: dict(row)
            for row in connection.execute("SELECT * FROM llm_models").fetchall()
        }
        active_payload_index = next(
            (index for index, model in enumerate(payload.models) if model.is_active),
            0,
        )
        saved_ids = []
        active_id = None

        for index, model_payload in enumerate(payload.models):
            provider = model_payload.provider.strip().lower() or "ollama"
            base_url = model_payload.base_url.strip().rstrip("/")
            model = model_payload.model.strip()
            comments = (model_payload.comments or "").strip() or None
            tts_voice_uri = (model_payload.tts_voice_uri or "").strip() or None
            if not base_url or not model:
                raise HTTPException(status_code=400, detail="Model rows need an address and model name")

            current = existing.get(model_payload.id) if model_payload.id is not None else None
            if model_payload.clear_api_key:
                api_key = None
            elif model_payload.api_key is not None:
                api_key = model_payload.api_key.strip() or None
            else:
                api_key = current["api_key"] if current else None

            if current:
                connection.execute(
                    """
                    UPDATE llm_models
                    SET provider = ?, base_url = ?, model = ?, comments = ?, api_key = ?, tts_voice_uri = ?,
                        is_active = 0, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (provider, base_url, model, comments, api_key, tts_voice_uri, model_payload.id),
                )
                model_id = model_payload.id
            else:
                cursor = connection.execute(
                    """
                    INSERT INTO llm_models (provider, base_url, model, comments, api_key, tts_voice_uri, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, 0)
                    """,
                    (provider, base_url, model, comments, api_key, tts_voice_uri),
                )
                model_id = cursor.lastrowid

            saved_ids.append(model_id)
            if index == active_payload_index:
                active_id = model_id

        if saved_ids:
            placeholders = ",".join("?" for _ in saved_ids)
            connection.execute(f"DELETE FROM llm_models WHERE id NOT IN ({placeholders})", saved_ids)
            connection.execute("UPDATE llm_models SET is_active = 0")
            connection.execute("UPDATE llm_models SET is_active = 1, updated_at = CURRENT_TIMESTAMP WHERE id = ?", (active_id,))
            connection.execute(
                f"""
                UPDATE conversations
                SET llm_model_id = ?
                WHERE llm_model_id IS NULL OR llm_model_id NOT IN ({placeholders})
                """,
                [active_id, *saved_ids],
            )

        return _public_config(connection)


frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
