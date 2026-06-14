import time
from typing import Callable, List, Optional, Union

from fastapi import HTTPException

from .llm import chat as llm_chat


MAX_DISCUSSION_ROUNDS = 10


def clamp_discussion_rounds(value: int) -> int:
    return max(1, min(MAX_DISCUSSION_ROUNDS, int(value)))


def validate_participant_payloads(participants: list, connection, llm_model_or_404: Callable) -> None:
    if not participants or len(participants) > 3:
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


def replace_participants(connection, conversation_id: int, participants: list, llm_model_or_404: Callable) -> List[dict]:
    validate_participant_payloads(participants, connection, llm_model_or_404)
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


def build_multi_agent_roundtable_prompt(participant: dict, all_participants: List[dict]) -> str:
    name = participant_display_name(participant)
    others = [
        participant_display_name(item)
        for item in all_participants
        if item["id"] != participant["id"]
    ]
    if not others:
        cast_line = "You are participating in a fictional roundtable conversation with the user."
    elif len(others) == 1:
        cast_line = (
            "You are participating in a fictional roundtable conversation with the user "
            f"and one other mentor-like character ({others[0]})."
        )
    else:
        cast_line = (
            "You are participating in a fictional roundtable conversation with the user "
            f"and {len(others)} other mentor-like characters ({', '.join(others)})."
        )

    return (
        f"{cast_line}\n\n"
        "The conversation is inspired by public themes associated with well-known motivational, "
        "personal development, and spiritual teachers, but you must not claim to literally be any real "
        "person. You are an educational and reflective simulation.\n\n"
        "Your job is to help the user think deeply, honestly, and practically about their life, goals, "
        "fears, relationships, work, meaning, habits, and inner growth.\n\n"
        "You may:\n"
        "- respond directly to the user;\n"
        "- build on what another character said;\n"
        "- respectfully disagree or add nuance;\n"
        "- ask another character for their view;\n"
        "- ask the user a thoughtful question;\n"
        "- summarize tensions or patterns emerging in the conversation.\n\n"
        "Do not dominate the conversation. Keep responses concise, warm, and conversational unless "
        "the user asks for depth.\n\n"
        "When speaking, use your assigned character voice, but avoid catchphrases, exact imitation, "
        "or claiming private knowledge of any real person.\n\n"
        "Messages from other participants appear labeled as [Name]: … in the transcript below.\n"
        "Your reply must contain ONLY what your assigned character would say — first person, in character, "
        "never a group summary or speaking for others."
    )


def build_participant_character_instruction(participant: dict, all_participants: List[dict]) -> str:
    name = participant_display_name(participant)
    personality = (participant.get("personality") or "").strip()
    lines = [
        build_multi_agent_roundtable_prompt(participant, all_participants),
        "",
        f'Your assigned character is "{name}".',
        "Every word of your reply must come only from this character, in first person, from their point of view.",
        "Do not speak as, quote at length, or impersonate any other participant or the user.",
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
) -> str:
    current_name = participant_display_name(current_participant)

    if previous_speaker_label == "the user":
        return (
            f'As "{current_name}", respond to the user\'s latest message in your character voice. '
            "The other participants will respond afterward in theirs."
        )

    return (
        f'As "{current_name}", respond directly to {previous_speaker_label}\'s most recent message above. '
        f'Start by addressing them by name (for example, "{previous_speaker_label}, …"). '
        "React to their specific points from your character's point of view."
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
