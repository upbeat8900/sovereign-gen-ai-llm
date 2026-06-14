import base64
import json
import os
import threading
import time
from pathlib import Path
from typing import List, Optional, Union

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .database import (
    DEFAULT_AGENTIC_MAX_ITERATIONS,
    DEFAULT_DIRECTOR_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_MULTI_AGENT_PROMPT,
    generate_memory_title as fallback_memory_title,
    get_connection,
    init_db,
    row_to_dict,
)
from .agentic import run_agentic_turn
from .documents import (
    archive_document,
    create_document,
    document_or_404,
    load_documents,
    normalize_upload_to_markdown,
    sanitize_filename,
    update_document,
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
    is_multi_agent_discussion,
    participant_display_name,
    participant_address_name,
    previous_discussion_speaker,
    build_discussion_response_instruction,
    build_in_character_final_reminder,
    build_participant_character_instruction,
    build_single_character_instruction,
)
from .scrape import scrape_website_content
from .stt import WHISPER_MODEL_OPTIONS, get_whisper_model, set_whisper_model, transcribe_audio_bytes
from .tts import list_elevenlabs_voices, synthesize_elevenlabs_speech
from .memory_map import create_memory_map, get_viz_spec_detail, list_conversation_viz_specs, update_viz_client_state
from .models import (
    AgentProfileCreate,
    AgentPersonalityDraftCreate,
    AgentPersonalityDraftRead,
    AgentProfileRead,
    AgentProfileUpdate,
    AgenticControlRequest,
    Conversation,
    ConversationAgenticSetupUpdate,
    ConversationCreate,
    ConversationDelete,
    ConversationDetail,
    ConversationModelUpdate,
    ConversationParticipantRead,
    ConversationParticipantsUpdate,
    ConversationReorder,
    ConversationTitleUpdate,
    CrossConversationMemoryMerge,
    Document,
    DocumentCreate,
    DocumentUpdate,
    DocumentUploadResponse,
    DocumentWebsiteUploadRequest,
    ElevenLabsSynthesizeRequest,
    ElevenLabsVoiceRead,
    LlmContextPreview,
    LlmConfigRead,
    LlmConfigUpdate,
    LlmModelRead,
    Memory,
    MemoryGroup,
    MemoryIntegrate,
    MemoryMapCreate,
    MemoryMapResponse,
    MemoryMerge,
    MemoryMove,
    MessageCreate,
    MessageResponse,
    RememberCreate,
    PromptConfigRead,
    PromptConfigUpdate,
    SpeechConfigRead,
    SpeechConfigUpdate,
    UsageStatsRead,
    VizSpecDetail,
    VizClientStateUpdate,
    VizSpecSummary,
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
    threading.Thread(target=get_whisper_model, name="whisper-warmup", daemon=True).start()


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
    conversation["mode"] = (conversation.get("mode") or "single").strip() or "single"
    if conversation["mode"] == "agentic" and not conversation.get("agentic_status"):
        conversation["agentic_status"] = "idle"
    return conversation


def _public_message(message: dict) -> dict:
    public = dict(message)
    metadata = public.pop("metadata_json", None)
    public["metadata"] = json.loads(metadata) if metadata else None
    return public


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
    message_kind: str = None,
    metadata_json: str = None,
    parent_message_id: int = None,
) -> dict:
    cursor = connection.execute(
        """
        INSERT INTO messages (
            conversation_id, role, content, llm_provider, llm_model, image_data, image_media_type,
            generation_ms, include_history, participant_id, message_kind, metadata_json, parent_message_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            message_kind,
            metadata_json,
            parent_message_id,
        ),
    )
    connection.execute(
        "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (conversation_id,),
    )
    return _public_message(
        dict(connection.execute("SELECT * FROM messages WHERE id = ?", (cursor.lastrowid,)).fetchone())
    )


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
    messages = []
    if not participant:
        messages.append(
            {
                "role": "system",
                "content": prompt_config["default_prompt"],
            }
        )
    length_priority = _answer_length_priority_message(answer_length)
    if length_priority:
        messages.append({"role": "system", "content": length_priority})

    participants_by_id = {}
    all_participants = []
    if participant:
        all_participants = load_participants(connection, conversation_id)
        for item in all_participants:
            participants_by_id[item["id"]] = item
        if is_multi_agent_discussion(all_participants):
            character_instruction = build_participant_character_instruction(
                participant,
                all_participants,
                prompt_config.get("multi_agent_prompt") or DEFAULT_MULTI_AGENT_PROMPT,
            )
        else:
            character_instruction = build_single_character_instruction(
                participant,
                prompt_config["default_prompt"],
            )
        messages.append({"role": "system", "content": character_instruction})
    if included_memories:
        memory_text = "\n".join(f"- {memory['content']}" for memory in included_memories)
        messages.append({"role": "system", "content": f"Conversation memory bank:\n{memory_text}"})
    if included_all_memories:
        all_memory_text = "\n".join(f"- {memory['content']}" for memory in included_all_memories)
        messages.append({"role": "user", "content": f"User memories:\n{all_memory_text}"})

    for row in history_rows:
        speaker_label = None
        if row.get("role") == "assistant" and row.get("participant_id"):
            speaker_label = _participant_speaker_label(connection, row["participant_id"], participants_by_id)
        messages.append(_format_message_for_llm(row, provider, speaker_label=speaker_label))

    if participant and len(all_participants) > 1:
        def resolve_participant_name(participant_id: int) -> str:
            return _participant_speaker_label(connection, participant_id, participants_by_id)

        previous_speaker_label, previous_row = previous_discussion_speaker(
            history_rows,
            resolve_participant_name=resolve_participant_name,
        )
        previous_speaker_address = previous_speaker_label
        if previous_row and previous_row.get("participant_id"):
            previous_participant = participants_by_id.get(previous_row["participant_id"])
            if previous_participant:
                previous_speaker_address = participant_address_name(previous_participant)
        messages.append(
            {
                "role": "system",
                "content": build_discussion_response_instruction(
                    participant,
                    all_participants,
                    previous_speaker_label,
                    previous_speaker_address,
                ),
            }
        )

    if participant:
        current_name = participant_display_name(participant)
        messages.append(
            {
                "role": "system",
                "content": build_in_character_final_reminder(current_name),
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


@app.post("/api/agent-profiles/draft-personality", response_model=AgentPersonalityDraftRead)
def draft_agent_personality(payload: AgentPersonalityDraftCreate) -> dict:
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Agent name is required")

    seed_personality = (payload.seed_personality or "").strip()
    messages = [
        {
            "role": "system",
            "content": (
                "You draft reusable AI agent personality and perspective instructions. "
                "Return only the personality text, written in second person for the agent. "
                "Do not include a title, markdown fence, or explanation."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Agent name: {name}\n\n"
                f"User notes or rough personality draft: {seed_personality or '(none)'}\n\n"
                "Create a polished personality / perspective for this saved agent. "
                "Make it specific, practical, and usable in multi-agent discussions. "
                "Keep it to 2-4 concise paragraphs."
            ),
        },
    ]

    with get_connection() as connection:
        config = _active_llm_model(connection)
        context_chars = sum(len(str(message.get("content", ""))) for message in messages)
        started_at = time.perf_counter()
        try:
            personality = llm_chat(
                provider=config["provider"],
                base_url=config["base_url"],
                model=config["model"],
                api_key=config["api_key"],
                messages=messages,
            ).strip()
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        generation_ms = max(0, int((time.perf_counter() - started_at) * 1000))
        if personality:
            _update_generation_stats(
                connection,
                config["provider"],
                config["model"],
                generation_ms,
                context_chars,
                len(personality),
            )

    if not personality:
        raise HTTPException(status_code=502, detail="The default LLM returned an empty draft")
    return {"personality": personality}


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
            "INSERT INTO app_config (id, default_prompt, multi_agent_prompt, director_prompt) VALUES (1, ?, ?, ?)",
            (DEFAULT_SYSTEM_PROMPT, DEFAULT_MULTI_AGENT_PROMPT, DEFAULT_DIRECTOR_PROMPT),
        )
        row = row_to_dict(connection.execute("SELECT * FROM app_config WHERE id = 1").fetchone())
    else:
        if not (row.get("multi_agent_prompt") or "").strip():
            connection.execute(
                "UPDATE app_config SET multi_agent_prompt = ? WHERE id = 1",
                (DEFAULT_MULTI_AGENT_PROMPT,),
            )
        if not (row.get("director_prompt") or "").strip():
            connection.execute(
                "UPDATE app_config SET director_prompt = ? WHERE id = 1",
                (DEFAULT_DIRECTOR_PROMPT,),
            )
        row = row_to_dict(connection.execute("SELECT * FROM app_config WHERE id = 1").fetchone())
    return row


def _public_prompt_config(config: dict) -> dict:
    return {
        "default_prompt": config["default_prompt"],
        "default_prompt_baseline": DEFAULT_SYSTEM_PROMPT,
        "multi_agent_prompt": (config.get("multi_agent_prompt") or "").strip() or DEFAULT_MULTI_AGENT_PROMPT,
        "multi_agent_prompt_baseline": DEFAULT_MULTI_AGENT_PROMPT,
        "director_prompt": (config.get("director_prompt") or "").strip() or DEFAULT_DIRECTOR_PROMPT,
        "director_prompt_baseline": DEFAULT_DIRECTOR_PROMPT,
        "updated_at": config["updated_at"],
    }


@app.get("/api/config/prompt", response_model=PromptConfigRead)
def get_prompt_config() -> dict:
    with get_connection() as connection:
        config = _prompt_config_or_default(connection)
        return _public_prompt_config(config)


@app.put("/api/config/prompt", response_model=PromptConfigRead)
def update_prompt_config(payload: PromptConfigUpdate) -> dict:
    prompt = payload.default_prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Default prompt cannot be empty")
    multi_agent_prompt = payload.multi_agent_prompt.strip()
    if not multi_agent_prompt:
        raise HTTPException(status_code=400, detail="Multi-agent prompt cannot be empty")
    director_prompt = payload.director_prompt.strip()
    if not director_prompt:
        raise HTTPException(status_code=400, detail="Director prompt cannot be empty")

    with get_connection() as connection:
        _prompt_config_or_default(connection)
        connection.execute(
            """
            UPDATE app_config
            SET default_prompt = ?, multi_agent_prompt = ?, director_prompt = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = 1
            """,
            (prompt, multi_agent_prompt, director_prompt),
        )
        config = row_to_dict(connection.execute("SELECT * FROM app_config WHERE id = 1").fetchone())
        return _public_prompt_config(config)


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


def _public_speech_config(config: dict) -> dict:
    api_key = (config.get("elevenlabs_api_key") or "").strip() or None
    return {
        "whisper_model": config["whisper_model"],
        "whisper_model_options": WHISPER_MODEL_OPTIONS,
        "has_elevenlabs_api_key": bool(api_key),
        "elevenlabs_api_key_preview": _mask_api_key(api_key),
        "updated_at": config["updated_at"],
    }


def _elevenlabs_api_key_or_400(connection) -> str:
    config = _speech_config_or_default(connection)
    api_key = (config.get("elevenlabs_api_key") or "").strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="ElevenLabs API key is not configured")
    return api_key


@app.get("/api/config/speech", response_model=SpeechConfigRead)
def get_speech_config() -> dict:
    with get_connection() as connection:
        config = _speech_config_or_default(connection)
        return _public_speech_config(config)


@app.put("/api/config/speech", response_model=SpeechConfigRead)
def update_speech_config(payload: SpeechConfigUpdate) -> dict:
    with get_connection() as connection:
        current = _speech_config_or_default(connection)
        model_name = current["whisper_model"]
        if payload.whisper_model is not None:
            model_name = payload.whisper_model.strip()
            if model_name not in WHISPER_MODEL_OPTIONS:
                raise HTTPException(status_code=400, detail="Unsupported Whisper model")

        if payload.clear_elevenlabs_api_key:
            api_key = None
        elif payload.elevenlabs_api_key is not None:
            api_key = payload.elevenlabs_api_key.strip() or None
        else:
            api_key = (current.get("elevenlabs_api_key") or "").strip() or None

        connection.execute(
            """
            UPDATE speech_config
            SET whisper_model = ?, elevenlabs_api_key = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = 1
            """,
            (model_name, api_key),
        )
        set_whisper_model(model_name)
        config = row_to_dict(connection.execute("SELECT * FROM speech_config WHERE id = 1").fetchone())
        return _public_speech_config(config)


@app.get("/api/tts/elevenlabs/voices", response_model=List[ElevenLabsVoiceRead])
def get_elevenlabs_voices() -> List[dict]:
    with get_connection() as connection:
        api_key = _elevenlabs_api_key_or_400(connection)
    try:
        return list_elevenlabs_voices(api_key)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/tts/elevenlabs/synthesize")
def synthesize_elevenlabs_audio(payload: ElevenLabsSynthesizeRequest) -> Response:
    with get_connection() as connection:
        api_key = _elevenlabs_api_key_or_400(connection)
    try:
        audio = synthesize_elevenlabs_speech(
            api_key=api_key,
            voice_id=payload.voice_id.strip(),
            text=payload.text,
            speech_rate=payload.speech_rate,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return Response(content=audio, media_type="audio/mpeg")


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
    mode = payload.mode or "single"
    with get_connection() as connection:
        active_model = _active_llm_model(connection)
        next_sort_order = connection.execute(
            "SELECT COALESCE(MIN(sort_order), 0) - 1 AS sort_order FROM conversations"
        ).fetchone()["sort_order"]
        cursor = connection.execute(
            """
            INSERT INTO conversations (
                title, sort_order, llm_model_id, mode, agentic_goal, agentic_success_criteria,
                agentic_scrape_url, agentic_scrape_depth, agentic_report_format, agentic_status, agentic_max_iterations
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                title,
                next_sort_order,
                active_model["id"],
                mode,
                (payload.agentic_goal or "").strip() or None,
                (payload.agentic_success_criteria or "").strip() or None,
                (payload.agentic_scrape_url or "").strip() or None,
                payload.agentic_scrape_depth or 1 if mode == "agentic" else None,
                (payload.agentic_report_format or "").strip() or None,
                "idle" if mode == "agentic" else None,
                payload.agentic_max_iterations or DEFAULT_AGENTIC_MAX_ITERATIONS if mode == "agentic" else None,
            ),
        )
        conversation_id = cursor.lastrowid
        if payload.participants:
            replace_participants(
                connection,
                conversation_id,
                payload.participants,
                _llm_model_or_404,
                mode=mode if mode in {"discussion", "agentic"} else "discussion",
            )
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


@app.post("/api/conversations/{conversation_id}/agentic-control", response_model=Conversation)
def control_agentic_conversation(conversation_id: int, payload: AgenticControlRequest) -> dict:
    with get_connection() as connection:
        conversation = _conversation_or_404(connection, conversation_id)
        if (conversation.get("mode") or "single") != "agentic":
            raise HTTPException(status_code=400, detail="Agentic controls are only available for agentic conversations")
        status = (conversation.get("agentic_status") or "idle").strip()
        if status not in {"running", "stop_requested", "wrap_requested"}:
            raise HTTPException(status_code=400, detail="No agentic process is currently running")
        next_status = "stop_requested" if payload.action == "stop" else "wrap_requested"
        connection.execute(
            """
            UPDATE conversations
            SET agentic_status = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (next_status, conversation_id),
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
        conversation = _public_conversation(connection, conversation_id)
        messages = [
            _public_message(dict(row))
            for row in connection.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id ASC",
                (conversation_id,),
            ).fetchall()
        ]
        memories = _active_memories(connection, conversation_id)
        participants = [public_participant(row) for row in load_participants(connection, conversation_id)]
        documents = load_documents(connection, conversation_id)
        return {
            "conversation": conversation,
            "messages": messages,
            "memories": memories,
            "participants": participants,
            "documents": documents,
        }


def _conversation_detail_payload(connection, conversation_id: int) -> dict:
    conversation = _public_conversation(connection, conversation_id)
    messages = [
        _public_message(dict(row))
        for row in connection.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id ASC",
            (conversation_id,),
        ).fetchall()
    ]
    memories = _active_memories(connection, conversation_id)
    participants = [public_participant(row) for row in load_participants(connection, conversation_id)]
    documents = load_documents(connection, conversation_id)
    return {
        "conversation": conversation,
        "messages": messages,
        "memories": memories,
        "participants": participants,
        "documents": documents,
    }


@app.put("/api/conversations/{conversation_id}/participants", response_model=List[ConversationParticipantRead])
def update_conversation_participants(conversation_id: int, payload: ConversationParticipantsUpdate) -> List[dict]:
    with get_connection() as connection:
        conversation = _conversation_or_404(connection, conversation_id)
        mode = payload.mode or conversation.get("mode") or "discussion"
        rows = replace_participants(
            connection,
            conversation_id,
            payload.participants,
            _llm_model_or_404,
            mode=mode if mode in {"discussion", "agentic"} else "discussion",
        )
        return [public_participant(row) for row in rows]


@app.put("/api/conversations/{conversation_id}/agentic-setup", response_model=ConversationDetail)
def update_agentic_conversation_setup(conversation_id: int, payload: ConversationAgenticSetupUpdate) -> dict:
    with get_connection() as connection:
        conversation = _conversation_or_404(connection, conversation_id)
        if (conversation.get("mode") or "single") != "agentic":
            raise HTTPException(status_code=400, detail="Agentic setup is only available for agentic conversations")

        replace_participants(
            connection,
            conversation_id,
            payload.participants,
            _llm_model_or_404,
            mode="agentic",
        )
        connection.execute(
            """
            UPDATE conversations
            SET
                agentic_goal = ?,
                agentic_success_criteria = ?,
                agentic_scrape_url = ?,
                agentic_scrape_depth = ?,
                agentic_report_format = ?,
                agentic_max_iterations = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                payload.agentic_goal.strip(),
                payload.agentic_success_criteria.strip(),
                (payload.agentic_scrape_url or "").strip() or None,
                payload.agentic_scrape_depth,
                (payload.agentic_report_format or "").strip() or None,
                payload.agentic_max_iterations,
                conversation_id,
            ),
        )
        return _conversation_detail_payload(connection, conversation_id)


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
        conversation_mode = (conversation.get("mode") or "single").strip() or "single"
        multi_agent_note = None
        if conversation_mode == "discussion" or is_multi_agent_discussion(participants):
            preview_model_id = payload.override_llm_model_id or participants[0]["llm_model_id"]
            config = _llm_model_or_404(connection, preview_model_id)
            if payload.override_llm_model_id:
                multi_agent_note = (
                    f"Multi-agent preview using composer model override ({config['model']}). "
                    "All agents use this model until the override is cleared."
                )
            else:
                multi_agent_note = (
                    f"Multi-agent preview for first participant ({config['model']}). "
                    "Other participants use the same shared context with different models and personalities."
                )
        elif participants:
            config = _llm_model_or_404(connection, participants[0]["llm_model_id"])
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
        conversation_mode = (conversation.get("mode") or "single").strip() or "single"

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

        if payload.agentic_start and conversation_mode != "agentic":
            raise HTTPException(status_code=400, detail="agentic_start is only available for agentic conversations")

        message_content = content
        user_message = None
        if not payload.agentic_start:
            user_message = _insert_message(
                connection,
                conversation_id,
                "user",
                message_content,
                image_data=image_data,
                image_media_type=image_media_type,
            )

        if payload.agentic_ad_hoc:
            if conversation_mode != "agentic":
                raise HTTPException(status_code=400, detail="Ad-hoc messages are only available for agentic conversations")
            status = (conversation.get("agentic_status") or "idle").strip()
            if status not in {"running", "wrap_requested", "stop_requested"}:
                raise HTTPException(status_code=400, detail="No agentic process is currently running")
            return {"user_message": user_message}

        if conversation["title"] == "New Conversation" and not payload.agentic_start:
            title_source = content or "Image message"
            title = title_source[:60] + ("..." if len(title_source) > 60 else "")
            connection.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conversation_id))

        participants = load_participants(connection, conversation_id)

        if conversation_mode == "agentic":
            status = (conversation.get("agentic_status") or "idle").strip()
            if status in {"running", "wrap_requested", "stop_requested"}:
                raise HTTPException(status_code=400, detail="Agentic process is already running")
            prompt_config = _prompt_config_or_default(connection)
            assistant_messages = run_agentic_turn(
                connection,
                conversation_id,
                conversation,
                participants,
                user_message=message_content,
                include_memories=payload.include_memories,
                include_all_memories=payload.include_all_memories,
                answer_length=payload.answer_length,
                insert_message=_insert_message,
                update_generation_stats=_update_generation_stats,
                summarize_llm_context=_summarize_llm_context,
                llm_model_or_404=_llm_model_or_404,
                active_memories_fn=_active_memories,
                all_memories_fn=_all_active_memories,
                insert_memory_fn=_insert_memory,
                director_prompt_template=(prompt_config.get("director_prompt") or DEFAULT_DIRECTOR_PROMPT),
                commit_connection=lambda conn: conn.commit(),
            )
            last_message = assistant_messages[-1] if assistant_messages else None
            return {
                "user_message": user_message,
                "assistant_message": last_message,
                "assistant_messages": assistant_messages,
            }

        if conversation_mode == "discussion" or is_multi_agent_discussion(participants):
            if payload.override_llm_model_id is not None:
                _llm_model_or_404(connection, payload.override_llm_model_id)
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
                override_llm_model_id=payload.override_llm_model_id,
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

        participant = participants[0] if participants else None
        if participant:
            config = _llm_model_or_404(connection, participant["llm_model_id"])
        else:
            config = _conversation_llm_model(connection, conversation)
        llm_messages, included_memory_count, included_all_memory_count = _build_llm_messages(
            connection,
            conversation_id,
            config["provider"],
            include_history=payload.include_history,
            include_memories=payload.include_memories,
            include_all_memories=payload.include_all_memories,
            participant=participant,
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
            participant_id=participant["id"] if participant else None,
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


@app.post("/api/conversations/{conversation_id}/memory-map", response_model=MemoryMapResponse)
def generate_memory_map(conversation_id: int, payload: MemoryMapCreate) -> MemoryMapResponse:
    with get_connection() as connection:
        return create_memory_map(connection, conversation_id, payload)


@app.get("/api/viz-specs/{viz_id}", response_model=VizSpecDetail)
def read_viz_spec(viz_id: str) -> VizSpecDetail:
    with get_connection() as connection:
        return get_viz_spec_detail(connection, viz_id)


@app.get("/api/conversations/{conversation_id}/viz-specs", response_model=List[VizSpecSummary])
def read_conversation_viz_specs(conversation_id: int) -> List[VizSpecSummary]:
    with get_connection() as connection:
        return list_conversation_viz_specs(connection, conversation_id)


@app.put("/api/viz-specs/{viz_id}/client-state", response_model=VizSpecDetail)
def save_viz_client_state(viz_id: str, payload: VizClientStateUpdate) -> VizSpecDetail:
    with get_connection() as connection:
        return update_viz_client_state(connection, viz_id, payload)


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


def _render_document_html(document: dict) -> str:
    import html

    title = html.escape(document["title"])
    body = html.escape(document["content_markdown"])
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <style>
    body {{ font-family: Georgia, 'Times New Roman', serif; max-width: 820px; margin: 40px auto; line-height: 1.6; color: #1f2937; padding: 0 24px; }}
    pre {{ white-space: pre-wrap; background: #f8fafc; padding: 16px; border-radius: 8px; }}
    h1 {{ font-size: 2rem; margin-bottom: 0.5rem; }}
    .meta {{ color: #64748b; font-size: 0.9rem; margin-bottom: 2rem; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class="meta">{html.escape(document.get('kind') or 'document')} · updated {html.escape(document.get('updated_at') or '')}</div>
  <pre>{body}</pre>
</body>
</html>"""


@app.get("/api/conversations/{conversation_id}/documents", response_model=List[Document])
def list_conversation_documents(conversation_id: int) -> List[dict]:
    with get_connection() as connection:
        _conversation_or_404(connection, conversation_id)
        return load_documents(connection, conversation_id)


@app.post("/api/conversations/{conversation_id}/documents", response_model=Document)
def create_conversation_document(conversation_id: int, payload: DocumentCreate) -> dict:
    with get_connection() as connection:
        _conversation_or_404(connection, conversation_id)
        return create_document(
            connection,
            conversation_id,
            title=payload.title,
            content_markdown=payload.content_markdown,
            kind=payload.kind,
        )


@app.post("/api/conversations/{conversation_id}/documents/upload", response_model=DocumentUploadResponse)
async def upload_conversation_document(
    conversation_id: int,
    file: UploadFile = File(...),
) -> dict:
    with get_connection() as connection:
        _conversation_or_404(connection, conversation_id)
        raw_bytes = await file.read()
        markdown, media_type, metadata = normalize_upload_to_markdown(
            file.filename or "document.txt",
            file.content_type or "",
            raw_bytes,
        )
        document = create_document(
            connection,
            conversation_id,
            title=sanitize_filename(file.filename or "Uploaded document"),
            content_markdown=markdown,
            kind="uploaded",
            source_filename=sanitize_filename(file.filename or "document.txt"),
            source_media_type=media_type,
            metadata=metadata,
        )
        return {"document": document}


@app.post("/api/conversations/{conversation_id}/documents/upload-website", response_model=DocumentUploadResponse)
def upload_conversation_document_from_website(
    conversation_id: int,
    payload: DocumentWebsiteUploadRequest,
) -> dict:
    with get_connection() as connection:
        _conversation_or_404(connection, conversation_id)

    try:
        scrape_result = scrape_website_content(payload.url, depth=payload.depth)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not scrape website: {exc}") from exc

    content = (scrape_result.get("content_markdown") or "").strip()
    if not content or content == "No readable content extracted.":
        raise HTTPException(status_code=400, detail="No readable website content was extracted")

    structures = scrape_result.get("structure") or []
    scraped_title = ""
    if structures and isinstance(structures[0], dict):
        scraped_title = (structures[0].get("title") or "").strip()
    title = (payload.title or "").strip() or scraped_title or scrape_result["url"]
    metadata = {
        "source": "website",
        "url": scrape_result["url"],
        "depth": scrape_result["depth"],
        "pages_scraped": scrape_result["pages_scraped"],
        "page_urls": scrape_result["page_urls"],
        "rendered_pages": scrape_result.get("rendered_pages", 0),
        "render_errors": scrape_result.get("render_errors", []),
        "extracted_chars": scrape_result["extracted_chars"],
    }

    with get_connection() as connection:
        document = create_document(
            connection,
            conversation_id,
            title=title,
            content_markdown=content,
            kind="uploaded",
            source_filename=scrape_result["url"],
            source_media_type="text/markdown",
            metadata=metadata,
        )
        return {"document": document}


@app.get("/api/conversations/{conversation_id}/documents/{document_id}", response_model=Document)
def get_conversation_document(conversation_id: int, document_id: int) -> dict:
    with get_connection() as connection:
        return document_or_404(connection, conversation_id, document_id)


@app.put("/api/conversations/{conversation_id}/documents/{document_id}", response_model=Document)
def update_conversation_document(conversation_id: int, document_id: int, payload: DocumentUpdate) -> dict:
    with get_connection() as connection:
        return update_document(
            connection,
            conversation_id,
            document_id,
            title=payload.title,
            content_markdown=payload.content_markdown,
        )


@app.delete("/api/conversations/{conversation_id}/documents/{document_id}")
def delete_conversation_document(conversation_id: int, document_id: int) -> dict:
    with get_connection() as connection:
        archive_document(connection, conversation_id, document_id)
        return {"ok": True}


@app.get("/api/conversations/{conversation_id}/documents/{document_id}/pdf")
def download_conversation_document_pdf(conversation_id: int, document_id: int) -> Response:
    with get_connection() as connection:
        document = document_or_404(connection, conversation_id, document_id)
    html = _render_document_html(document)
    filename = sanitize_filename(document["title"]) + ".html"
    return Response(
        content=html,
        media_type="text/html",
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


def _usage_date_filter(days: Optional[int]) -> tuple[str, list]:
    if days is not None and days > 0:
        return "AND created_at >= datetime('now', ?)", [f"-{int(days)} days"]
    return "", []


def _usage_agent_label(row: dict) -> str:
    if row.get("participant_id"):
        participant = {
            "name": row.get("participant_name") or "",
            "llm_model": row.get("participant_llm_model") or row.get("llm_model"),
            "llm_comments": row.get("llm_comments") or "",
        }
        return participant_display_name(participant)
    profile_name = (row.get("profile_name") or "").strip()
    if profile_name:
        return profile_name
    model = (row.get("llm_model") or "").strip()
    if model:
        return f"Assistant ({model})"
    return "Assistant"


def _load_usage_stats(connection, days: Optional[int] = None) -> dict:
    from datetime import date, timedelta

    message_date_filter, message_date_params = _usage_date_filter(days)
    memory_date_filter, memory_date_params = _usage_date_filter(days)

    conversation_count = connection.execute("SELECT COUNT(*) AS count FROM conversations").fetchone()["count"]
    message_summary = connection.execute(
        f"""
        SELECT
            COUNT(*) AS total_messages,
            SUM(CASE WHEN role = 'user' THEN 1 ELSE 0 END) AS user_messages,
            SUM(CASE WHEN role = 'assistant' THEN 1 ELSE 0 END) AS assistant_messages,
            MIN(created_at) AS first_activity,
            MAX(created_at) AS last_activity
        FROM messages
        WHERE role IN ('user', 'assistant')
        {message_date_filter}
        """,
        message_date_params,
    ).fetchone()
    memory_count = connection.execute(
        f"""
        SELECT COUNT(*) AS count
        FROM memories
        WHERE archived_at IS NULL
        {memory_date_filter}
        """,
        memory_date_params,
    ).fetchone()["count"]

    daily_rows = connection.execute(
        f"""
        SELECT
            DATE(created_at) AS bucket_date,
            SUM(CASE WHEN role = 'user' THEN 1 ELSE 0 END) AS user_messages,
            SUM(CASE WHEN role = 'assistant' THEN 1 ELSE 0 END) AS assistant_messages,
            COUNT(*) AS total_messages
        FROM messages
        WHERE role IN ('user', 'assistant')
        {message_date_filter}
        GROUP BY DATE(created_at)
        ORDER BY bucket_date ASC
        """,
        message_date_params,
    ).fetchall()

    daily_model_rows = connection.execute(
        f"""
        SELECT
            DATE(created_at) AS bucket_date,
            COALESCE(NULLIF(TRIM(llm_provider), ''), 'unknown') AS provider,
            COALESCE(NULLIF(TRIM(llm_model), ''), 'unknown') AS model,
            COUNT(*) AS message_count
        FROM messages
        WHERE role = 'assistant'
        {message_date_filter}
        GROUP BY bucket_date, provider, model
        ORDER BY bucket_date ASC, message_count DESC, model ASC
        """,
        message_date_params,
    ).fetchall()

    daily_model_lookup: dict[str, list[dict]] = {}
    for row in daily_model_rows:
        provider = row["provider"]
        model = row["model"]
        daily_model_lookup.setdefault(row["bucket_date"], []).append(
            {
                "provider": provider,
                "model": model,
                "label": f"{provider} / {model}" if provider != "unknown" else model,
                "message_count": int(row["message_count"]),
            }
        )

    daily_lookup = {
        row["bucket_date"]: {
            "date": row["bucket_date"],
            "user_messages": int(row["user_messages"]),
            "assistant_messages": int(row["assistant_messages"]),
            "total_messages": int(row["total_messages"]),
            "model_requests": daily_model_lookup.get(row["bucket_date"], []),
        }
        for row in daily_rows
    }

    daily: list[dict] = []
    if daily_lookup or days is not None:
        today = date.today()
        if days is not None:
            cursor = today - timedelta(days=max(int(days), 1) - 1)
            end = today
        else:
            cursor = date.fromisoformat(min(daily_lookup.keys()))
            end = date.fromisoformat(max(daily_lookup.keys()))
        while cursor <= end:
            key = cursor.isoformat()
            daily.append(
                daily_lookup.get(
                    key,
                    {
                        "date": key,
                        "user_messages": 0,
                        "assistant_messages": 0,
                        "total_messages": 0,
                        "model_requests": [],
                    },
                )
            )
            cursor += timedelta(days=1)

    model_rows = connection.execute(
        f"""
        SELECT
            COALESCE(NULLIF(TRIM(llm_provider), ''), 'unknown') AS provider,
            COALESCE(NULLIF(TRIM(llm_model), ''), 'unknown') AS model,
            COUNT(*) AS message_count
        FROM messages
        WHERE role = 'assistant'
        {message_date_filter}
        GROUP BY provider, model
        ORDER BY message_count DESC, model ASC
        """,
        message_date_params,
    ).fetchall()

    agent_rows = connection.execute(
        f"""
        SELECT
            m.participant_id,
            m.llm_model,
            cp.name AS participant_name,
            lm.model AS participant_llm_model,
            lm.comments AS llm_comments,
            ap.name AS profile_name,
            COUNT(*) AS message_count
        FROM messages m
        LEFT JOIN conversation_participants cp ON cp.id = m.participant_id
        LEFT JOIN llm_models lm ON lm.id = cp.llm_model_id
        LEFT JOIN agent_profiles ap ON ap.id = cp.agent_profile_id
        WHERE m.role = 'assistant'
        {message_date_filter.replace('created_at', 'm.created_at') if message_date_filter else ''}
        GROUP BY
            m.participant_id,
            m.llm_model,
            cp.name,
            lm.model,
            lm.comments,
            ap.name
        """,
        message_date_params,
    ).fetchall()

    agent_totals: dict[str, dict] = {}
    for row in agent_rows:
        label = _usage_agent_label(dict(row))
        model = (row["llm_model"] or row["participant_llm_model"] or "").strip() or None
        bucket = agent_totals.setdefault(label, {"agent_name": label, "llm_model": model, "message_count": 0})
        if not bucket["llm_model"] and model:
            bucket["llm_model"] = model
        bucket["message_count"] += int(row["message_count"])

    by_agent = sorted(agent_totals.values(), key=lambda item: (-item["message_count"], item["agent_name"].lower()))

    return {
        "days": days,
        "summary": {
            "conversation_count": int(conversation_count),
            "total_messages": int(message_summary["total_messages"] or 0),
            "user_messages": int(message_summary["user_messages"] or 0),
            "assistant_messages": int(message_summary["assistant_messages"] or 0),
            "memory_count": int(memory_count or 0),
            "first_activity": message_summary["first_activity"],
            "last_activity": message_summary["last_activity"],
        },
        "daily": daily,
        "by_model": [
            {
                "provider": row["provider"],
                "model": row["model"],
                "label": f"{row['provider']} / {row['model']}" if row["provider"] != "unknown" else row["model"],
                "message_count": int(row["message_count"]),
            }
            for row in model_rows
        ],
        "by_agent": by_agent,
    }


@app.get("/api/usage/stats", response_model=UsageStatsRead)
def get_usage_stats(days: Optional[int] = Query(default=None, ge=1, le=3650)) -> dict:
    with get_connection() as connection:
        return _load_usage_stats(connection, days)


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
frontend_index = frontend_dist / "index.html"


@app.get("/viz/{_rest:path}")
def serve_viz_spa(_rest: str) -> FileResponse:
    if not frontend_index.exists():
        raise HTTPException(status_code=404, detail="Frontend build not found")
    return FileResponse(frontend_index)


if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
