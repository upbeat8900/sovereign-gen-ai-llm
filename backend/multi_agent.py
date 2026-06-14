import re
import time
from typing import Callable, List, Optional, Union

from fastapi import HTTPException

from .database import DEFAULT_MULTI_AGENT_PROMPT
from .llm import chat as llm_chat


MAX_DISCUSSION_ROUNDS = 10


def clamp_discussion_rounds(value: int) -> int:
    return max(1, min(MAX_DISCUSSION_ROUNDS, int(value)))


def validate_participant_payloads(
    participants: list,
    connection,
    llm_model_or_404: Callable,
    *,
    mode: str = "discussion",
) -> None:
    if not participants:
        raise HTTPException(status_code=400, detail="Provide at least one participant")
    if mode == "discussion" and len(participants) > 3:
        raise HTTPException(status_code=400, detail="Provide 1 to 3 discussion participants")
    for participant in participants:
        llm_model_or_404(connection, participant.llm_model_id)


def load_participants(connection, conversation_id: int) -> List[dict]:
    rows = connection.execute(
        """
        SELECT
            cp.id,
            cp.conversation_id,
            cp.llm_model_id,
            cp.personality,
            cp.name,
            cp.tts_voice_uri,
            cp.tts_speech_rate,
            cp.agent_profile_id,
            cp.sort_order,
            lm.provider AS llm_provider,
            lm.model AS llm_model,
            lm.comments AS llm_comments
        FROM conversation_participants cp
        JOIN llm_models lm ON lm.id = cp.llm_model_id
        WHERE cp.conversation_id = ?
        ORDER BY cp.sort_order ASC, cp.id ASC
        """,
        (conversation_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def public_participant(row: dict) -> dict:
    return {
        "id": row["id"],
        "conversation_id": row["conversation_id"],
        "llm_model_id": row["llm_model_id"],
        "personality": row.get("personality") or "",
        "name": row.get("name") or "",
        "sort_order": row["sort_order"],
        "tts_voice_uri": (row.get("tts_voice_uri") or "").strip() or None,
        "tts_speech_rate": row.get("tts_speech_rate"),
        "agent_profile_id": row.get("agent_profile_id"),
        "llm_provider": row.get("llm_provider"),
        "llm_model": row.get("llm_model"),
        "llm_comments": (row.get("llm_comments") or "").strip() or None,
    }


def replace_participants(
    connection,
    conversation_id: int,
    participants: list,
    llm_model_or_404: Callable,
    *,
    mode: str = "discussion",
) -> List[dict]:
    validate_participant_payloads(participants, connection, llm_model_or_404, mode=mode)
    connection.execute(
        "DELETE FROM conversation_participants WHERE conversation_id = ?",
        (conversation_id,),
    )
    for index, participant in enumerate(participants):
        connection.execute(
            """
            INSERT INTO conversation_participants (
                conversation_id, llm_model_id, personality, name, tts_voice_uri, tts_speech_rate,
                agent_profile_id, sort_order
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                participant.llm_model_id,
                (participant.personality or "").strip(),
                (participant.name or "").strip(),
                (participant.tts_voice_uri or "").strip() or None,
                participant.tts_speech_rate,
                participant.agent_profile_id,
                index,
            ),
        )
    return load_participants(connection, conversation_id)


def participant_display_name(participant: dict) -> str:
    name = (participant.get("name") or "").strip()
    if name:
        return name
    comments = (participant.get("llm_comments") or "").strip()
    if comments:
        return comments
    return participant.get("llm_model") or "Assistant"


def participant_address_name(participant: dict) -> str:
    full_name = participant_display_name(participant).strip()
    if not full_name:
        return "Assistant"
    return full_name.split()[0]


def strip_leading_speaker_label(content: str) -> str:
    return re.sub(r"^\[[^\]]+\]:\s*", "", (content or "").strip(), count=1)


def previous_discussion_speaker(
    history_rows: List[dict],
    *,
    resolve_participant_name: Callable[[int], str],
) -> tuple[str, dict | None]:
    for row in reversed(history_rows):
        role = row.get("role")
        if role == "user":
            return "the user", row
        if role == "assistant":
            participant_id = row.get("participant_id")
            if participant_id:
                return resolve_participant_name(participant_id), row
            return "the previous speaker", row
    return "the user", None


def build_multi_agent_roundtable_prompt(
    participant: dict,
    all_participants: List[dict],
    template: str = DEFAULT_MULTI_AGENT_PROMPT,
) -> str:
    name = participant_display_name(participant)
    prompt_template = (template or "").strip() or DEFAULT_MULTI_AGENT_PROMPT
    try:
        rendered = prompt_template.format(character_name=name, cast_line="")
    except (KeyError, ValueError):
        rendered = DEFAULT_MULTI_AGENT_PROMPT.format(character_name=name)
    return rendered.strip()


def is_multi_agent_discussion(participants: List[dict]) -> bool:
    return len(participants) > 1


def build_single_character_instruction(participant: dict, default_prompt: str) -> str:
    name = participant_display_name(participant)
    personality = (participant.get("personality") or "").strip()
    lines = [
        (default_prompt or "").strip() or "You are a helpful assistant.",
        "",
        f'You are speaking as "{name}". Stay in character — first person, from their point of view.',
        "Do not break character, mention being an AI, or add meta-commentary.",
    ]
    if personality:
        lines.append(f"Character and perspective:\n{personality}")
    else:
        lines.append(f"Embody {name} consistently in how you think, feel, and speak.")
    return "\n".join(lines)


def build_participant_character_instruction(
    participant: dict,
    all_participants: List[dict],
    multi_agent_prompt: str = DEFAULT_MULTI_AGENT_PROMPT,
) -> str:
    name = participant_display_name(participant)
    personality = (participant.get("personality") or "").strip()
    lines = [
        build_multi_agent_roundtable_prompt(participant, all_participants, multi_agent_prompt),
        "",
        f'Your assigned character is "{name}".',
        "Every word of your reply must come only from this character, in first person, from their point of view.",
        "Do not speak as, quote at length, or impersonate any other participant or the user.",
        "When addressing another participant, use only their first name — not their full name.",
        "Do not prefix your reply with [Name]: or any bracketed speaker label; your message is tagged separately.",
        "Do not break character, mention being an AI, or add meta-commentary about the discussion format.",
    ]
    if personality:
        lines.append(f"Character and perspective:\n{personality}")
    else:
        lines.append(f"Embody {name} consistently in how you think, feel, and speak.")
    return "\n".join(lines)


def build_discussion_response_instruction(
    current_participant: dict,
    all_participants: List[dict],
    previous_speaker_label: str,
    previous_speaker_address: str,
) -> str:
    current_name = participant_display_name(current_participant)
    novelty_rules = (
        "Add something genuinely new — do not repeat, paraphrase, or agree by restating points "
        f"already made in the transcript (including your own earlier [{current_name}]: lines)."
    )
    no_label_rule = (
        "Do not prefix your reply with [Name]: or any bracketed speaker label; your message is tagged separately."
    )

    if previous_speaker_label == "the user":
        return (
            f'As "{current_name}", respond to the user\'s latest message in your character voice. '
            f"{novelty_rules} {no_label_rule} The other participants will respond afterward in theirs."
        )

    return (
        f'As "{current_name}", respond directly to {previous_speaker_label}\'s most recent message above. '
        f'When you address them, use only their first name (for example, "{previous_speaker_address}, …"). '
        f"{no_label_rule} React to their specific points from your character's point of view. {novelty_rules}"
    )


def build_in_character_final_reminder(current_name: str) -> str:
    return (
        f'Before you reply: write only what "{current_name}" would say — '
        "first person, in character, from their point of view. No other voices. "
        f"Check the transcript for your earlier [{current_name}]: lines and add new substance only. "
        "Do not start with [Name]: or bracketed labels. When addressing someone, use their first name only."
    )


def run_multi_agent_turns(
    connection,
    conversation_id: int,
    participants: List[dict],
    *,
    discussion_rounds: int,
    include_history: Union[bool, int],
    include_memories: bool,
    include_all_memories: bool,
    answer_length: int,
    override_llm_model_id: Optional[int] = None,
    build_llm_messages: Callable,
    insert_message: Callable,
    update_generation_stats: Callable,
    summarize_llm_context: Callable,
    llm_model_or_404: Callable,
    stored_include_history: Callable,
    commit_connection: Callable,
) -> List[dict]:
    rounds = clamp_discussion_rounds(discussion_rounds)
    assistant_messages: List[dict] = []

    for _round_index in range(rounds):
        for participant in participants:
            model_id = override_llm_model_id or participant["llm_model_id"]
            model_config = llm_model_or_404(connection, model_id)
            llm_messages, included_memory_count, included_all_memory_count = build_llm_messages(
                connection,
                conversation_id,
                model_config["provider"],
                include_history=include_history,
                include_memories=include_memories,
                include_all_memories=include_all_memories,
                participant=participant,
                answer_length=answer_length,
            )
            context_summary = summarize_llm_context(
                llm_messages,
                provider=model_config["provider"],
                model=model_config["model"],
                include_history=include_history,
                include_memories=include_memories,
                include_all_memories=include_all_memories,
                memory_count=included_memory_count,
                all_memory_count=included_all_memory_count,
            )
            started_at = time.perf_counter()
            try:
                assistant_content = llm_chat(
                    provider=model_config["provider"],
                    base_url=model_config["base_url"],
                    model=model_config["model"],
                    api_key=model_config["api_key"],
                    messages=llm_messages,
                )
            except (RuntimeError, ValueError) as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc

            assistant_content = strip_leading_speaker_label(assistant_content)

            generation_ms = max(0, int((time.perf_counter() - started_at) * 1000))
            update_generation_stats(
                connection,
                model_config["provider"],
                model_config["model"],
                generation_ms,
                context_summary["total_chars"],
                len(assistant_content),
            )
            assistant_message = insert_message(
                connection,
                conversation_id,
                "assistant",
                assistant_content,
                llm_provider=model_config["provider"],
                llm_model=model_config["model"],
                generation_ms=generation_ms,
                include_history=stored_include_history(include_history),
                participant_id=participant["id"],
            )
            assistant_messages.append(assistant_message)
            commit_connection(connection)

    return assistant_messages
