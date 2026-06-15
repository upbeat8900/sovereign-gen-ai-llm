import json
import re
import time
from typing import Callable, Dict, List, Optional, Tuple

from fastapi import HTTPException

from .database import DEFAULT_AGENTIC_MAX_ITERATIONS, DEFAULT_DIRECTOR_PROMPT
from .documents import create_document, document_list_summary, load_documents, search_documents
from .llm import chat as llm_chat
from .multi_agent import participant_display_name
from .scrape import scrape_website_content

MAX_SCRAPE_CALLS = 5
MAX_TOOL_CALLS = 30

SUB_AGENT_PROMPT = """You are a specialist on a team led by a Director. Your job is to answer only the delegated task below.

Use the conversation memory bank when relevant. Focus on the assigned task, state assumptions and uncertainties, and say whether your answer satisfies the assignment.
If essential details are missing, ask the Director a concise clarification question. Do not ask the user questions.
If the Director is responding to a clarification question you asked, use that clarification directly in your answer.
If the Director asks you to revise or continue after an earlier answer, focus on what is materially different in the new instructions.

Do not discuss orchestration, other agents, or the full conversation history.

Professional role and perspective:
{personality_block}

Delegated task:
{task}

Expected output:
{expected_output}

Relevant evidence:
{evidence}

When full source documents are included above, use them directly. Do not ask the Director to resend material that is already provided.
"""


class AgenticState(dict):
    """Runtime state for the director loop."""
    iteration: int
    done: bool
    last_action: dict
    last_result: str
    success_score: int
    remaining_gaps: List[str]
    final_answer: str
    parse_repair_attempted: bool
    tool_calls: int
    scrape_calls: int
    referenced_memory_ids: List[int]
    referenced_document_ids: List[int]
    message_ids: List[int]


def _extract_json_object(text: str) -> dict:
    content = (text or "").strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, flags=re.DOTALL)
    if fence_match:
        content = fence_match.group(1)
    else:
        brace_match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if brace_match:
            content = brace_match.group(0)
    return json.loads(content)


def parse_director_action(raw: str) -> dict:
    action = _extract_json_object(raw)
    if "action" not in action:
        raise ValueError("Director action missing 'action'")
    return action


def build_sub_agent_prompt(participant: dict, task: str, expected_output: str, evidence: str) -> str:
    personality = (participant.get("personality") or "").strip()
    name = participant_display_name(participant)
    personality_block = personality or f"You are {name}, a focused specialist."
    return SUB_AGENT_PROMPT.format(
        personality_block=personality_block,
        task=task.strip(),
        expected_output=(expected_output or "A clear, actionable answer.").strip(),
        evidence=(evidence or "(none provided)").strip(),
    )


def build_director_prompt(
    *,
    template: str,
    goal: str,
    success_criteria: str,
    report_format: str,
    agent_list: str,
    document_list: str,
    scrape_url: str,
) -> str:
    prompt_template = (template or "").strip() or DEFAULT_DIRECTOR_PROMPT
    values = {
        "goal": goal or "(not set)",
        "success_criteria": success_criteria or "(not set)",
        "report_format": report_format or "(director chooses best format)",
        "agent_list": agent_list or "(none attached)",
        "document_list": document_list or "(none)",
        "scrape_url": scrape_url or "(none configured)",
    }
    try:
        return prompt_template.format(**values)
    except (KeyError, ValueError):
        return DEFAULT_DIRECTOR_PROMPT.format(**values)


def _runtime_action_guidance(participants: List[dict], documents: List[dict], scrape_url: Optional[str]) -> str:
    participant_ids = [str(participant["id"]) for participant in participants]
    document_ids = [str(document["id"]) for document in documents]
    lines = [
        "Runtime action constraints:",
        f"- Valid participant_id values for call_agent: {', '.join(participant_ids) if participant_ids else '(none)'}",
        "- The specialist roster above includes each agent's id, name, and role/perspective.",
        "- If you use call_agent, choose the best specialist by role/perspective and put that exact id in participant_id.",
        "- call_agent arguments MUST include one valid participant_id, task, and expected_output.",
        "- If no specialist clearly fits, use consult_agents to ask every attached agent for a brief opinion, then delegate or synthesize.",
        "- consult_agents arguments MUST include question and expected_output.",
        "- When a delegated task depends on conversation documents, include their ids in call_agent.document_ids.",
        "- Specialists automatically receive the full text of every document listed in document_ids or referenced in the task.",
        "- generate_report must store the actual user-facing deliverable (draft, memo, plan, copy, etc.), not process notes.",
        "- Put the full deliverable markdown in generate_report.content, compiled from specialist outputs before generating.",
        "- You may use generate_report to create a new Markdown document even when no input documents exist.",
    ]
    if document_ids:
        lines.append(f"- Valid document_id values for read_document: {', '.join(document_ids)}")
        lines.append("- Use read_document only with one of the valid document_id values above.")
    else:
        lines.append("- No documents are currently available. Do not use search_documents or read_document.")
    if scrape_url:
        lines.append("- scrape_website is available for the configured website.")
    else:
        lines.append("- No scrape URL is configured. Do not use scrape_website.")
    lines.append("- If no tool is available or needed, call a valid specialist, generate_report, or complete.")
    return "\n".join(lines)


def _format_memories(memories: List[dict], query: Optional[str] = None) -> Tuple[str, List[int]]:
    if query and query.strip():
        needle = query.strip().lower()
        filtered = [
            memory
            for memory in memories
            if needle in f"{memory.get('title', '')}\n{memory.get('content', '')}".lower()
        ]
    else:
        filtered = memories
    if not filtered:
        return "No matching memories found.", []
    lines = []
    ids = []
    for memory in filtered[:20]:
        ids.append(memory["id"])
        lines.append(f"[{memory['id']}] {memory.get('title', 'Memory')}: {memory.get('content', '')[:800]}")
    return "\n".join(lines), ids


def _looks_like_clarification_request(text: str) -> bool:
    content = (text or "").strip().lower()
    if not content or "?" not in content:
        return False
    clarification_markers = [
        "clarify",
        "clarification",
        "could you",
        "can you",
        "please provide",
        "need more",
        "before i",
        "what should",
        "which",
        "do you want",
        "would you like",
        "should i",
    ]
    return any(marker in content for marker in clarification_markers)


def _document_context(documents: List[dict]) -> str:
    if not documents:
        return "(none)"
    lines = []
    for document in documents[:8]:
        lines.append(f"[{document['id']}] {document['title']} ({document['kind']}): {document['content_markdown'][:1200]}")
    return "\n\n".join(lines)


_DOCUMENT_ID_PATTERNS = (
    re.compile(r"\[(\d+)\]"),
    re.compile(r"(?:document|doc)\s*#?\s*(\d+)", re.IGNORECASE),
)


def _normalize_document_ids(values: List[int], valid_ids: set[int]) -> List[int]:
    seen: set[int] = set()
    ordered: List[int] = []
    for value in values:
        if value in valid_ids and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _action_int_list(arguments: dict, key: str) -> List[int]:
    value = arguments.get(key)
    if value is None or value == "":
        return []
    if isinstance(value, bool):
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, list):
        ids: List[int] = []
        for item in value:
            try:
                ids.append(int(item))
            except (TypeError, ValueError):
                continue
        return ids
    if isinstance(value, str):
        ids = []
        for part in re.split(r"[,;\s]+", value.strip()):
            if not part:
                continue
            try:
                ids.append(int(part))
            except ValueError:
                continue
        return ids
    return []


def _document_ids_from_titles(text: str, documents: List[dict]) -> List[int]:
    content = (text or "").lower()
    matches: List[int] = []
    for document in documents:
        title = (document.get("title") or "").strip()
        if len(title) < 4:
            continue
        if title.lower() in content:
            matches.append(document["id"])
    return matches


def _extract_document_ids_from_text(text: str, valid_ids: set[int]) -> List[int]:
    found: List[int] = []
    for pattern in _DOCUMENT_ID_PATTERNS:
        for match in pattern.finditer(text or ""):
            try:
                doc_id = int(match.group(1))
            except (TypeError, ValueError):
                continue
            if doc_id in valid_ids:
                found.append(doc_id)
    return found


def _resolve_delegation_document_ids(
    *,
    task: str,
    expected_output: str,
    rationale: str,
    arguments: dict,
    documents: List[dict],
) -> List[int]:
    valid_ids = {document["id"] for document in documents}
    combined_text = "\n".join(part for part in (task, expected_output, rationale) if (part or "").strip())
    return _normalize_document_ids(
        _action_int_list(arguments, "document_ids")
        + _extract_document_ids_from_text(combined_text, valid_ids)
        + _document_ids_from_titles(combined_text, documents),
        valid_ids,
    )


def _format_full_documents(documents: List[dict], document_ids: List[int]) -> str:
    if not document_ids:
        return ""
    by_id = {document["id"]: document for document in documents}
    blocks: List[str] = []
    for document_id in document_ids:
        document = by_id.get(document_id)
        if not document:
            continue
        kind = document.get("kind") or "uploaded"
        blocks.append(
            f"## [{document_id}] {document['title']} ({kind})\n\n"
            f"{(document.get('content_markdown') or '').strip()}"
        )
    return "\n\n".join(blocks)


def _build_delegation_evidence(
    *,
    last_result: Optional[str],
    task: str,
    expected_output: str,
    rationale: str,
    arguments: dict,
    documents: List[dict],
) -> Tuple[str, List[int]]:
    document_ids = _resolve_delegation_document_ids(
        task=task,
        expected_output=expected_output,
        rationale=rationale,
        arguments=arguments,
        documents=documents,
    )
    parts: List[str] = []
    document_text = _format_full_documents(documents, document_ids)
    if document_text:
        parts.append(f"Full source documents for this task:\n\n{document_text}")
    if (last_result or "").strip():
        parts.append(f"Additional context from the Director:\n{last_result.strip()}")
    evidence = "\n\n".join(parts) if parts else "(none)"
    return evidence, document_ids


def _scrape_document_title(scrape_result: dict, query: Optional[str] = None) -> str:
    structures = scrape_result.get("structure") or []
    scraped_title = ""
    if structures and isinstance(structures[0], dict):
        scraped_title = (structures[0].get("title") or "").strip()
    base_title = scraped_title or scrape_result.get("url") or "Scraped website"
    if query and query.strip():
        return f"{base_title} - {query.strip()}"
    return base_title


def _scrape_document_metadata(scrape_result: dict, query: Optional[str] = None) -> dict:
    return {
        "source": "agentic_scrape_url",
        "url": scrape_result["url"],
        "query": (query or "").strip() or None,
        "depth": scrape_result.get("depth"),
        "pages_scraped": scrape_result.get("pages_scraped"),
        "page_urls": scrape_result.get("page_urls"),
        "rendered_pages": scrape_result.get("rendered_pages", 0),
        "render_errors": scrape_result.get("render_errors", []),
        "structure": scrape_result.get("structure"),
        "extracted_chars": scrape_result.get("extracted_chars"),
    }


def _matching_scrape_document(documents: List[dict], url: str, depth: int, query: Optional[str] = None) -> Optional[dict]:
    normalized_query = (query or "").strip() or None
    for document in documents:
        metadata = document.get("metadata") or {}
        if metadata.get("source") != "agentic_scrape_url":
            continue
        if metadata.get("url") != url:
            continue
        if metadata.get("depth") != depth:
            continue
        if (metadata.get("query") or None) == normalized_query:
            return document
    return None


def _create_scrape_document(
    connection,
    conversation_id: int,
    scrape_result: dict,
    query: Optional[str] = None,
) -> dict:
    content = (scrape_result.get("content_markdown") or scrape_result.get("summary") or "").strip()
    if not content:
        content = "No readable content extracted."
    return create_document(
        connection,
        conversation_id,
        title=_scrape_document_title(scrape_result, query),
        content_markdown=content,
        kind="uploaded",
        source_filename=scrape_result["url"],
        source_media_type="text/markdown",
        metadata=_scrape_document_metadata(scrape_result, query),
    )


def _steps_context(steps: List[dict]) -> str:
    lines = []
    for step in steps[-20:]:
        kind = step.get("message_kind") or step.get("role") or "step"
        content = (step.get("content") or "").strip()
        if content:
            lines.append(f"[{kind}] {content[:1000]}")
    return "\n\n".join(lines) or "(none)"


def _step_metadata(step: dict) -> dict:
    metadata = step.get("metadata")
    if isinstance(metadata, dict):
        return metadata
    metadata_json = step.get("metadata_json")
    if metadata_json:
        try:
            parsed = json.loads(metadata_json)
            return parsed if isinstance(parsed, dict) else {}
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
    return {}


def _normalize_task_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _is_repeated_delegation(participant_id: int, task: str, prior_steps: List[dict]) -> bool:
    current_task = _normalize_task_text(task)
    if not current_task:
        return False
    for step in reversed(prior_steps):
        if step.get("message_kind") != "director_delegation":
            continue
        metadata = _step_metadata(step)
        if metadata.get("participant_id") != participant_id:
            continue
        prior_task = _normalize_task_text((metadata.get("arguments") or {}).get("task") or step.get("content") or "")
        if prior_task == current_task or current_task in prior_task or prior_task in current_task:
            return True
    return False


def _fallback_success_score(content: str) -> int:
    text = (content or "").strip()
    if not text or text.startswith("Sub-agent call failed:"):
        return 0
    if _looks_like_clarification_request(text):
        return 5
    score = 20
    if len(text) >= 120:
        score += 15
    if len(text) >= 600:
        score += 10
    lower = text.lower()
    deliverable_terms = ["draft", "post", "blog", "report", "plan", "recommendation", "cta", "call to action"]
    if any(term in lower for term in deliverable_terms):
        score += 15
    if any(term in lower for term in ["assumption", "uncertain", "gap", "need", "missing"]):
        score -= 5
    return max(0, min(75, score))


def _fallback_evaluation(agent_content: str, synthesis: str) -> dict:
    score = _fallback_success_score(agent_content)
    if score == 0:
        reason = "0% because the specialist response did not provide usable content for the task."
    elif score <= 10:
        reason = f"{score}% because the specialist primarily asked for clarification rather than producing the requested deliverable."
    elif score < 50:
        reason = (
            f"{score}% because the specialist response contains partial usable material, but the Director could not parse "
            "a structured evaluation and remaining gaps still need validation against the success criteria."
        )
    else:
        reason = (
            f"{score}% because the specialist response appears to include a substantive draft or deliverable, but the Director "
            "could not parse a structured evaluation to confirm all success criteria."
        )
    return {
        "success_score": score,
        "remaining_gaps": ["Structured evaluation could not be parsed; Director synthesis needs review."],
        "assessment": f"{reason}\n\nDirector synthesis:\n{synthesis[:2000]}",
        "ready_to_complete": False,
    }


def _director_synthesize_agent_response(
    *,
    director_config: dict,
    system_prompt: str,
    goal: str,
    success_criteria: str,
    task: str,
    agent_content: str,
    prior_steps: List[dict],
) -> str:
    prompt = (
        "A specialist has replied. As Director, extract and interpret the useful information before evaluating. "
        "Do not ask the user questions. Do not repeat the same delegation unless new evidence is required. "
        "Return a concise synthesis with: usable findings, implications for the goal, gaps, and recommended next step.\n\n"
        f"Goal:\n{goal or '(not set)'}\n\n"
        f"Success criteria:\n{success_criteria or '(not set)'}\n\n"
        f"Delegated task:\n{task}\n\n"
        f"Specialist reply:\n{agent_content[:6000]}\n\n"
        f"Recent process context:\n{_steps_context(prior_steps)}"
    )
    return llm_chat(
        provider=director_config["provider"],
        base_url=director_config["base_url"],
        model=director_config["model"],
        api_key=director_config["api_key"],
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
        temperature=0.2,
    ).strip()


def _director_clarification_answer(
    *,
    director_config: dict,
    system_prompt: str,
    goal: str,
    success_criteria: str,
    agent_question: str,
    documents: List[dict],
    prior_steps: List[dict],
    website_context: str,
) -> str:
    prompt = (
        "A specialist agent asked the Director for clarification. Answer as the Director using the user goal, "
        "success criteria, prior steps, available input documents, and website evidence. Do not ask the user. "
        "Infer the best answer when information is incomplete, and state assumptions briefly.\n\n"
        f"Goal:\n{goal or '(not set)'}\n\n"
        f"Success criteria:\n{success_criteria or '(not set)'}\n\n"
        f"Specialist clarification request:\n{agent_question[:3000]}\n\n"
        f"Input documents:\n{_document_context(documents)}\n\n"
        f"Website evidence:\n{website_context or '(none)'}\n\n"
        f"Recent process context:\n{_steps_context(prior_steps)}\n\n"
        "Return a concise clarification answer the specialist can act on."
    )
    return llm_chat(
        provider=director_config["provider"],
        base_url=director_config["base_url"],
        model=director_config["model"],
        api_key=director_config["api_key"],
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
        temperature=0.2,
    ).strip()


def _agent_list_text(participants: List[dict]) -> str:
    if not participants:
        return "(none)"
    lines = []
    for participant in participants:
        name = participant_display_name(participant)
        role = (participant.get("personality") or participant.get("llm_comments") or "").strip()
        model = participant.get("llm_model") or "unknown model"
        role_text = role[:600] if role else "No role description provided."
        lines.append(f"- id={participant['id']} | name={name} | model={model} | role/perspective: {role_text}")
    return "\n".join(lines)


def _persist_step(
    insert_message: Callable,
    connection,
    conversation_id: int,
    *,
    message_kind: str,
    content: str,
    metadata: Optional[dict] = None,
    participant_id: Optional[int] = None,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    generation_ms: Optional[int] = None,
    parent_message_id: Optional[int] = None,
) -> dict:
    return insert_message(
        connection,
        conversation_id,
        "assistant",
        content,
        llm_provider=llm_provider,
        llm_model=llm_model,
        generation_ms=generation_ms,
        participant_id=participant_id,
        message_kind=message_kind,
        metadata_json=json.dumps(metadata) if metadata else None,
        parent_message_id=parent_message_id,
    )


def _set_agentic_progress(connection, conversation_id: int, progress: Optional[dict]) -> None:
    connection.execute(
        "UPDATE conversations SET agentic_progress_json = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (json.dumps(progress) if progress else None, conversation_id),
    )


def _action_int(arguments: dict, key: str) -> int:
    value = arguments.get(key)
    if value is None or value == "":
        raise ValueError(f"Missing required argument: {key}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer argument for {key}: {value!r}") from exc


def _action_arguments(action: dict) -> dict:
    arguments = action.get("arguments") or {}
    return arguments if isinstance(arguments, dict) else {}


def _participant_role_text(participant: dict) -> str:
    return (participant.get("personality") or participant.get("llm_comments") or "").strip()


def _coerce_valid_participant_id(value, valid_ids: set[int]) -> Optional[int]:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        participant_id = int(value)
    except (TypeError, ValueError):
        return None
    return participant_id if participant_id in valid_ids else None


def _extract_participant_id_from_action(action: dict, participants: List[dict]) -> Optional[int]:
    valid_ids = {participant["id"] for participant in participants}
    arguments = _action_arguments(action)
    for key in ("participant_id", "participant", "agent_id", "agent"):
        participant_id = _coerce_valid_participant_id(arguments.get(key), valid_ids)
        if participant_id is not None:
            return participant_id
        name_value = str(arguments.get(key) or "").strip()
        if name_value:
            for participant in participants:
                if participant_display_name(participant).lower() == name_value.lower():
                    return participant["id"]
    for key in ("participant_id", "participant"):
        participant_id = _coerce_valid_participant_id(action.get(key), valid_ids)
        if participant_id is not None:
            return participant_id
    return None


def _delegation_context_text(action: dict, goal: str) -> str:
    arguments = _action_arguments(action)
    return " ".join(
        str(value or "")
        for value in [
            goal,
            arguments.get("participant"),
            arguments.get("agent"),
            arguments.get("agent_name"),
            arguments.get("role"),
            arguments.get("task"),
            arguments.get("expected_output"),
            arguments.get("question"),
            action.get("rationale"),
            action.get("expected_result"),
        ]
    )


def _tokenize_for_match(text: str) -> set[str]:
    stop_words = {
        "agent",
        "assistant",
        "specialist",
        "expert",
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "into",
        "about",
        "task",
        "goal",
        "output",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(token) >= 3 and token not in stop_words
    }


def _score_participant_for_delegation(participant: dict, context: str) -> int:
    context_lower = context.lower()
    context_tokens = _tokenize_for_match(context)
    name = participant_display_name(participant)
    role = _participant_role_text(participant)
    participant_text = f"{name} {role}"
    participant_tokens = _tokenize_for_match(participant_text)
    score = len(context_tokens & participant_tokens)
    if name and name.lower() in context_lower:
        score += 5
    if role and role.lower() in context_lower:
        score += 3
    return score


def _rank_participants_for_delegation(action: dict, participants: List[dict], goal: str) -> List[tuple[int, int, str]]:
    context = _delegation_context_text(action, goal)
    ranked = [
        (_score_participant_for_delegation(participant, context), participant["id"], participant_display_name(participant))
        for participant in participants
    ]
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return ranked


def _resolve_participant_id_from_context(action: dict, participants: List[dict], goal: str = "") -> Optional[int]:
    ranked = _rank_participants_for_delegation(action, participants, goal)
    if not ranked:
        return None
    top_score, top_id, _top_name = ranked[0]
    second_score = ranked[1][0] if len(ranked) > 1 else -1
    if top_score > 0 and top_score > second_score:
        return top_id
    return None


CONSULT_AGENT_PROMPT = """You are a specialist on a team led by a Director. The Director is gathering brief opinions from every specialist before choosing who should do the work.

Professional role and perspective:
{personality_block}

Goal context:
{goal}

Success criteria:
{success_criteria}

Director question:
{question}

Expected from your opinion:
{expected_output}

Reply in 2-4 sentences. Recommend whether you are the best fit and why. Do not ask questions.
"""


def _repair_director_action(action: dict, participants: List[dict], goal: str) -> Tuple[dict, List[str]]:
    repaired = dict(action)
    arguments = dict(_action_arguments(action))
    repairs: List[str] = []
    action_name = (repaired.get("action") or "").strip()

    if action_name == "call_agent":
        extracted_participant_id = _extract_participant_id_from_action(repaired, participants)
        if extracted_participant_id and not arguments.get("participant_id"):
            arguments["participant_id"] = extracted_participant_id
            repairs.append(f"Recovered participant_id {extracted_participant_id} from the action payload.")
        if not arguments.get("participant_id") and len(participants) == 1:
            arguments["participant_id"] = participants[0]["id"]
            repairs.append(f"Defaulted participant_id to the only attached agent ({participants[0]['id']}).")
        elif not arguments.get("participant_id"):
            resolved_participant_id = _resolve_participant_id_from_context(repaired, participants, goal)
            if resolved_participant_id:
                arguments["participant_id"] = resolved_participant_id
                repairs.append(f"Resolved participant_id to agent {resolved_participant_id} from the requested role/task.")
            else:
                ranked = _rank_participants_for_delegation(repaired, participants, goal)
                if ranked:
                    top_score, top_id, top_name = ranked[0]
                    second_score = ranked[1][0] if len(ranked) > 1 else -1
                    if top_score > second_score:
                        arguments["participant_id"] = top_id
                        repairs.append(
                            f"Selected best-matching specialist {top_name} (id={top_id}) from role/title fit."
                        )
                    elif len(participants) > 1:
                        question = (
                            (arguments.get("task") or "").strip()
                            or (repaired.get("expected_result") or "").strip()
                            or (repaired.get("rationale") or "").strip()
                            or goal.strip()
                            or "Which specialist should lead this work?"
                        )
                        repaired["action"] = "consult_agents"
                        action_name = "consult_agents"
                        arguments = {
                            "question": question,
                            "expected_output": (
                                (arguments.get("expected_output") or "").strip()
                                or "Brief opinion on fit and recommended approach."
                            ),
                        }
                        repairs.append(
                            "No clear specialist match; consulting all attached agents before delegating."
                        )
        if action_name == "call_agent":
            if not (arguments.get("task") or "").strip():
                task = (
                    (arguments.get("expected_output") or "").strip()
                    or (repaired.get("expected_result") or "").strip()
                    or (repaired.get("rationale") or "").strip()
                    or goal.strip()
                    or "Contribute specialist analysis toward the conversation goal."
                )
                arguments["task"] = task
                repairs.append("Filled missing task from the Director context.")
            if not (arguments.get("expected_output") or "").strip():
                arguments["expected_output"] = "A concise specialist answer that advances the goal and notes uncertainties."
                repairs.append("Filled missing expected_output.")
    elif action_name == "consult_agents":
        if not (arguments.get("question") or "").strip():
            arguments["question"] = (
                (repaired.get("expected_result") or "").strip()
                or (repaired.get("rationale") or "").strip()
                or goal.strip()
                or "Which specialist perspective best advances the goal?"
            )
            repairs.append("Filled missing consult_agents.question from the Director context.")
        if not (arguments.get("expected_output") or "").strip():
            arguments["expected_output"] = "Brief opinion on fit, approach, and who should lead."
            repairs.append("Filled missing consult_agents.expected_output.")

    if repairs and not (repaired.get("rationale") or "").strip():
        repaired["rationale"] = repairs[0]

    repaired["arguments"] = arguments
    return repaired, repairs


def _director_messages(
    *,
    system_prompt: str,
    user_message: str,
    prior_steps: List[dict],
    last_result: Optional[str],
) -> List[dict]:
    messages = [{"role": "system", "content": system_prompt}]
    for step in prior_steps[-30:]:
        role = step.get("role") or "assistant"
        kind = step.get("message_kind") or "chat"
        content = step.get("content") or ""
        if role == "user":
            messages.append({"role": "user", "content": f"[user clarification] {content}"})
        else:
            messages.append({"role": "assistant", "content": f"[{kind}] {content}"})
    if user_message:
        messages.append({"role": "user", "content": user_message})
    if last_result:
        messages.append({"role": "user", "content": f"Latest action result:\n{last_result}"})
    messages.append(
        {
            "role": "user",
            "content": (
                "Choose your next JSON action. Evaluate progress against the success criteria before deciding. "
                "Do not ask the user questions; use assumptions, tools, or specialists instead."
            ),
        }
    )
    return messages


def _load_agentic_steps(connection, conversation_id: int) -> List[dict]:
    rows = connection.execute(
        """
        SELECT id, role, message_kind, content, metadata_json, participant_id
        FROM messages
        WHERE conversation_id = ?
        ORDER BY id ASC
        """,
        (conversation_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def _extract_agent_deliverables(steps: List[dict]) -> List[str]:
    deliverables: List[str] = []
    for step in steps:
        if step.get("message_kind") != "agent_reply":
            continue
        content = (step.get("content") or "").strip()
        if not content or content.lower().startswith("sub-agent call failed"):
            continue
        deliverables.append(content)
    return deliverables


def _extract_latest_synthesis(steps: List[dict]) -> str:
    for step in reversed(steps):
        if step.get("message_kind") == "director_synthesis":
            return (step.get("content") or "").strip()
    return ""


def _resolve_deliverable_body(
    *,
    steps: List[dict],
    explicit_content: Optional[str] = None,
    final_answer: Optional[str] = None,
) -> str:
    explicit = (explicit_content or "").strip()
    if explicit:
        return explicit

    agent_deliverables = _extract_agent_deliverables(steps)
    if len(agent_deliverables) == 1:
        return agent_deliverables[0]
    if len(agent_deliverables) > 1:
        return "\n\n".join(
            f"### Specialist contribution {index}\n\n{content}"
            for index, content in enumerate(agent_deliverables, start=1)
        )

    synthesis = _extract_latest_synthesis(steps)
    if synthesis:
        return synthesis

    final = (final_answer or "").strip()
    if final:
        return final

    return "(No deliverable content was captured from specialist outputs.)"


def _generate_report_markdown(
    *,
    title: str,
    format_request: Optional[str],
    goal: str,
    success_criteria: str,
    deliverable_body: str,
    director_summary: str,
    documents: List[dict],
    memories_used: List[int],
    documents_used: List[int],
) -> str:
    format_line = format_request or "Task-appropriate deliverable"
    provenance_docs = [doc for doc in documents if doc["id"] in documents_used]
    lines = [
        f"# {title}",
        "",
        deliverable_body.strip(),
        "",
        "---",
        "",
        f"**Format:** {format_line}",
        "",
        "## Goal",
        goal or "(not set)",
        "",
        "## Success criteria",
        success_criteria or "(not set)",
        "",
    ]
    summary = (director_summary or "").strip()
    if summary and summary != deliverable_body.strip():
        lines.extend(["## Director summary", summary, ""])
    lines.extend(["## Sources and provenance", ""])
    if memories_used:
        lines.append(f"- Conversation/global memories referenced: {', '.join(str(item) for item in memories_used)}")
    if provenance_docs:
        lines.append("- Documents referenced:")
        for doc in provenance_docs:
            lines.append(f"  - [{doc['id']}] {doc['title']}")
    if not memories_used and not provenance_docs:
        lines.append("- No explicit memory or document references recorded.")
    return "\n".join(lines)


def _generate_process_report_markdown(
    *,
    goal: str,
    success_criteria: str,
    steps: List[dict],
    memories_used: List[int],
    documents_used: List[int],
) -> str:
    lines = [
        "# Process and rationale",
        "",
        "## Goal",
        goal or "(not set)",
        "",
        "## Success criteria",
        success_criteria or "(not set)",
        "",
        "## Director process",
    ]
    for step in steps:
        kind = step.get("message_kind") or "step"
        content = (step.get("content") or "").strip()
        if not content:
            continue
        if kind.startswith("director_") or kind == "agent_reply":
            lines.extend([f"### {kind.replace('_', ' ').title()}", content, ""])
    lines.extend(["## References", ""])
    if memories_used:
        lines.append(f"- Memories referenced: {', '.join(str(item) for item in memories_used)}")
    if documents_used:
        lines.append(f"- Documents referenced: {', '.join(str(item) for item in documents_used)}")
    if not memories_used and not documents_used:
        lines.append("- No explicit memory or document references recorded.")
    return "\n".join(lines)


def _deliverable_title(title: str) -> str:
    cleaned = (title or "Task report").strip() or "Task report"
    if cleaned.lower().startswith("deliverable"):
        return cleaned
    return f"Deliverable - {cleaned}"


def _create_deliverable_documents(
    connection,
    conversation_id: int,
    *,
    title: str,
    format_request: Optional[str],
    goal: str,
    success_criteria: str,
    deliverable_body: str,
    director_summary: str,
    steps: List[dict],
    state: AgenticState,
    insert_message: Callable,
    director_config: dict,
    assistant_messages: List[dict],
    prior_steps: List[dict],
    parent_message_id: Optional[int] = None,
) -> Optional[dict]:
    body = (deliverable_body or "").strip()
    if not body or body.startswith("(No deliverable content was captured"):
        return None

    markdown = _generate_report_markdown(
        title=title,
        format_request=format_request,
        goal=goal,
        success_criteria=success_criteria,
        deliverable_body=body,
        director_summary=director_summary,
        documents=load_documents(connection, conversation_id),
        memories_used=list(dict.fromkeys(state.get("referenced_memory_ids") or [])),
        documents_used=list(dict.fromkeys(state.get("referenced_document_ids") or [])),
    )
    report_metadata = {
        "format_request": format_request,
        "goal": goal,
        "success_criteria": success_criteria,
        "message_ids": [step.get("id") for step in steps if step.get("id")],
        "memory_ids": list(dict.fromkeys(state.get("referenced_memory_ids") or [])),
        "document_ids": list(dict.fromkeys(state.get("referenced_document_ids") or [])),
        "upserted": True,
        "document_role": "output",
    }
    document = _upsert_generated_document(
        connection,
        conversation_id,
        kind="generated_report",
        title=title,
        content_markdown=markdown,
        metadata=report_metadata,
    )
    process_markdown = _generate_process_report_markdown(
        goal=goal,
        success_criteria=success_criteria,
        steps=steps,
        memories_used=list(dict.fromkeys(state.get("referenced_memory_ids") or [])),
        documents_used=list(dict.fromkeys(state.get("referenced_document_ids") or [])),
    )
    process_document = _upsert_generated_document(
        connection,
        conversation_id,
        kind="generated_process",
        title="Process and rationale",
        content_markdown=process_markdown,
        metadata={**report_metadata, "document_role": "process"},
    )
    doc_message = _persist_step(
        insert_message,
        connection,
        conversation_id,
        message_kind="director_document",
        content=(
            f"Generated deliverable: {document['title']}\n"
            f"Generated process/rationale document: {process_document['title']}"
        ),
        metadata={
            "document_id": document["id"],
            "process_document_id": process_document["id"],
            "format_request": format_request,
        },
        llm_provider=director_config["provider"],
        llm_model=director_config["model"],
        parent_message_id=parent_message_id,
    )
    assistant_messages.append(doc_message)
    prior_steps.append(doc_message)
    state["report_generated"] = True
    state["last_result"] = f"Deliverable stored as document #{document['id']}."
    return document


def _upsert_generated_document(
    connection,
    conversation_id: int,
    *,
    kind: str,
    title: str,
    content_markdown: str,
    metadata: dict,
) -> dict:
    existing_reports = [document for document in load_documents(connection, conversation_id) if document["kind"] == kind]
    if not existing_reports:
        return create_document(
            connection,
            conversation_id,
            title=title,
            content_markdown=content_markdown,
            kind=kind,
            metadata=metadata,
        )

    keeper = existing_reports[0]
    content = (content_markdown or "").strip()
    if not content:
        raise HTTPException(status_code=400, detail="Document content cannot be empty")
    connection.execute(
        """
        UPDATE documents
        SET title = ?, content_markdown = ?, metadata_json = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ? AND conversation_id = ?
        """,
        (
            (title or "Generated document").strip() or "Generated document",
            content,
            json.dumps(metadata) if metadata else None,
            keeper["id"],
            conversation_id,
        ),
    )
    for extra in existing_reports[1:]:
        connection.execute(
            """
            UPDATE documents
            SET archived_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
            WHERE id = ? AND conversation_id = ?
            """,
            (extra["id"], conversation_id),
        )
    connection.execute(
        "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (conversation_id,),
    )
    return next(document for document in load_documents(connection, conversation_id) if document["id"] == keeper["id"])


def _agentic_control_status(connection, conversation_id: int) -> str:
    row = connection.execute(
        "SELECT agentic_status FROM conversations WHERE id = ?",
        (conversation_id,),
    ).fetchone()
    return (row["agentic_status"] if row else "") or ""


def _director_wrap_up_answer(
    *,
    director_config: dict,
    system_prompt: str,
    user_message: str,
    prior_steps: List[dict],
    last_result: str,
) -> str:
    messages = _director_messages(
        system_prompt=system_prompt,
        user_message=user_message,
        prior_steps=prior_steps,
        last_result=last_result,
    )
    messages.append(
        {
            "role": "user",
            "content": (
                "The user requested wrap-up. Stop delegating and produce the best final answer now "
                "using only the evidence and agent outputs collected so far. State any remaining gaps briefly."
            ),
        }
    )
    return llm_chat(
        provider=director_config["provider"],
        base_url=director_config["base_url"],
        model=director_config["model"],
        api_key=director_config["api_key"],
        messages=messages,
        temperature=0.2,
    ).strip()


def run_agentic_turn(
    connection,
    conversation_id: int,
    conversation: dict,
    participants: List[dict],
    *,
    user_message: str,
    include_memories: bool,
    include_all_memories: bool,
    answer_length: int,
    insert_message: Callable,
    update_generation_stats: Callable,
    summarize_llm_context: Callable,
    llm_model_or_404: Callable,
    active_memories_fn: Callable,
    all_memories_fn: Callable,
    director_prompt_template: str,
    commit_connection: Callable,
) -> List[dict]:
    del answer_length
    if not participants:
        raise HTTPException(status_code=400, detail="Agentic conversations require at least one attached agent")

    goal = (conversation.get("agentic_goal") or "").strip()
    success_criteria = (conversation.get("agentic_success_criteria") or "").strip()
    scrape_url = (conversation.get("agentic_scrape_url") or "").strip() or None
    scrape_depth = max(1, min(3, int(conversation.get("agentic_scrape_depth") or 1)))
    report_format = (conversation.get("agentic_report_format") or "").strip()
    max_iterations = int(conversation.get("agentic_max_iterations") or DEFAULT_AGENTIC_MAX_ITERATIONS)
    current_documents = load_documents(connection, conversation_id)

    def scrape_progress(progress: dict) -> None:
        _set_agentic_progress(
            connection,
            conversation_id,
            {
                **progress,
                "tool": "scrape_website",
                "depth": scrape_depth,
                "max_pages": 25,
                "timestamp": time.time(),
            },
        )
        commit_connection(connection)

    initial_scrape_error = ""
    if scrape_url and not _matching_scrape_document(current_documents, scrape_url, scrape_depth):
        try:
            scrape_result = scrape_website_content(scrape_url, depth=scrape_depth, progress_callback=scrape_progress)
            _create_scrape_document(connection, conversation_id, scrape_result)
            current_documents = load_documents(connection, conversation_id)
        except Exception as exc:
            initial_scrape_error = f"Initial website scrape failed: {exc}"

    director_config = llm_model_or_404(connection, conversation["llm_model_id"])
    system_prompt = build_director_prompt(
        template=director_prompt_template,
        goal=goal,
        success_criteria=success_criteria,
        report_format=report_format,
        agent_list=_agent_list_text(participants),
        document_list=document_list_summary(current_documents),
        scrape_url=scrape_url or "",
    )
    system_prompt = f"{system_prompt}\n\n{_runtime_action_guidance(participants, current_documents, scrape_url)}"

    connection.execute(
        "UPDATE conversations SET agentic_status = 'running', updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (conversation_id,),
    )
    _set_agentic_progress(connection, conversation_id, None)
    commit_connection(connection)

    assistant_messages: List[dict] = []
    prior_steps = _load_agentic_steps(connection, conversation_id)

    state: AgenticState = {
        "iteration": 0,
        "done": False,
        "last_result": "",
        "success_score": 0,
        "remaining_gaps": [],
        "final_answer": "",
        "parse_repair_attempted": False,
        "tool_calls": 0,
        "scrape_calls": 0,
        "report_generated": any(step.get("message_kind") == "director_document" for step in prior_steps),
        "referenced_memory_ids": [],
        "referenced_document_ids": [],
        "message_ids": [],
    }
    if initial_scrape_error:
        state["last_result"] = initial_scrape_error

    participant_by_id = {participant["id"]: participant for participant in participants}
    final_status = "stopped"

    while not state["done"] and state["iteration"] < max_iterations:
        control_status = _agentic_control_status(connection, conversation_id)
        if control_status == "stop_requested":
            stop_message = _persist_step(
                insert_message,
                connection,
                conversation_id,
                message_kind="director_error",
                content="Director process interrupted by the user.",
                metadata={"iteration": state["iteration"], "interrupted": True},
                llm_provider=director_config["provider"],
                llm_model=director_config["model"],
            )
            assistant_messages.append(stop_message)
            prior_steps.append(stop_message)
            state["done"] = True
            final_status = "stopped"
            commit_connection(connection)
            break
        if control_status == "wrap_requested":
            try:
                final_answer = _director_wrap_up_answer(
                    director_config=director_config,
                    system_prompt=system_prompt,
                    user_message=user_message,
                    prior_steps=prior_steps,
                    last_result=state.get("last_result") or "",
                )
            except Exception as exc:
                final_answer = f"Director wrap-up requested, but synthesis failed: {exc}"
            final_message = _persist_step(
                insert_message,
                connection,
                conversation_id,
                message_kind="director_final",
                content=final_answer or "Director wrapped up with the available information.",
                metadata={
                    "wrapped_up": True,
                    "success_score": state.get("success_score", 0),
                    "remaining_gaps": state.get("remaining_gaps") or [],
                },
                llm_provider=director_config["provider"],
                llm_model=director_config["model"],
            )
            assistant_messages.append(final_message)
            prior_steps.append(final_message)
            state["done"] = True
            final_status = "completed"
            commit_connection(connection)
            break

        state["iteration"] += 1
        prior_steps = _load_agentic_steps(connection, conversation_id)
        started_at = time.perf_counter()
        director_messages = _director_messages(
            system_prompt=system_prompt,
            user_message=user_message,
            prior_steps=prior_steps,
            last_result=state.get("last_result"),
        )
        try:
            raw_action = llm_chat(
                provider=director_config["provider"],
                base_url=director_config["base_url"],
                model=director_config["model"],
                api_key=director_config["api_key"],
                messages=director_messages,
                temperature=0.2,
            )
        except (RuntimeError, ValueError) as exc:
            error_message = _persist_step(
                insert_message,
                connection,
                conversation_id,
                message_kind="director_error",
                content=f"Director LLM call failed: {exc}",
                metadata={"iteration": state["iteration"]},
                llm_provider=director_config["provider"],
                llm_model=director_config["model"],
            )
            assistant_messages.append(error_message)
            commit_connection(connection)
            break

        generation_ms = max(0, int((time.perf_counter() - started_at) * 1000))
        try:
            action = parse_director_action(raw_action)
        except (ValueError, json.JSONDecodeError):
            if not state["parse_repair_attempted"]:
                state["parse_repair_attempted"] = True
                state["last_result"] = "Your previous response was not valid JSON. Return one JSON object only."
                eval_message = _persist_step(
                    insert_message,
                    connection,
                    conversation_id,
                    message_kind="director_evaluation",
                    content="Director action could not be parsed. Requesting repair.",
                    metadata={"iteration": state["iteration"], "raw": raw_action[:1000]},
                    llm_provider=director_config["provider"],
                    llm_model=director_config["model"],
                    generation_ms=generation_ms,
                )
                assistant_messages.append(eval_message)
                prior_steps.append(eval_message)
                commit_connection(connection)
                continue
            error_message = _persist_step(
                insert_message,
                connection,
                conversation_id,
                message_kind="director_error",
                content="Director action remained invalid after repair attempt.",
                metadata={"iteration": state["iteration"], "raw": raw_action[:1000]},
                llm_provider=director_config["provider"],
                llm_model=director_config["model"],
                generation_ms=generation_ms,
            )
            assistant_messages.append(error_message)
            commit_connection(connection)
            break
        action, action_repairs = _repair_director_action(action, participants, goal)

        rationale_message = _persist_step(
            insert_message,
            connection,
            conversation_id,
            message_kind="director_rationale",
            content=(action.get("rationale") or "Proceeding with selected action.").strip(),
            metadata={
                "iteration": state["iteration"],
                "action": action.get("action"),
                "arguments": action.get("arguments") or {},
                "expected_result": action.get("expected_result"),
                "criteria_addressed": action.get("criteria_addressed") or [],
                "repairs": action_repairs,
            },
            llm_provider=director_config["provider"],
            llm_model=director_config["model"],
            generation_ms=generation_ms,
            parent_message_id=None,
        )
        assistant_messages.append(rationale_message)
        prior_steps.append(rationale_message)
        commit_connection(connection)

        action_name = (action.get("action") or "").strip()
        arguments = _action_arguments(action)
        result_summary = ""
        metadata = {"iteration": state["iteration"], "action": action_name, "arguments": arguments}

        if action_name in {
            "search_conversation_memories",
            "search_all_memories",
            "search_documents",
            "read_document",
            "scrape_website",
        }:
            state["tool_calls"] += 1
            if state["tool_calls"] > MAX_TOOL_CALLS:
                state["done"] = True
                state["final_answer"] = "Tool budget exhausted before completion."
                break

        if action_name == "search_conversation_memories":
            memories = active_memories_fn(connection, conversation_id) if include_memories else []
            result_summary, memory_ids = _format_memories(memories, arguments.get("query"))
            state["referenced_memory_ids"].extend(memory_ids)
        elif action_name == "search_all_memories":
            memories = all_memories_fn(connection, exclude_conversation_id=conversation_id) if include_all_memories else []
            result_summary, memory_ids = _format_memories(memories, arguments.get("query"))
            state["referenced_memory_ids"].extend(memory_ids)
        elif action_name == "search_documents":
            docs = search_documents(current_documents, arguments.get("query"))
            if not docs:
                result_summary = (
                    "No documents are available for this conversation."
                    if not current_documents
                    else "No matching documents found."
                )
            else:
                result_summary = "\n".join(
                    f"[{doc['id']}] {doc['title']} ({doc['kind']}): {doc['content_markdown'][:400]}"
                    for doc in docs[:10]
                )
                state["referenced_document_ids"].extend(doc["id"] for doc in docs)
        elif action_name == "read_document":
            current_documents = load_documents(connection, conversation_id)
            if not current_documents:
                result_summary = "No documents are available for this conversation. Choose another action."
                tool_message = _persist_step(
                    insert_message,
                    connection,
                    conversation_id,
                    message_kind="director_tool",
                    content=result_summary,
                    metadata=metadata,
                    llm_provider=director_config["provider"],
                    llm_model=director_config["model"],
                    parent_message_id=rationale_message["id"],
                )
                assistant_messages.append(tool_message)
                prior_steps.append(tool_message)
                state["last_result"] = result_summary
                commit_connection(connection)
                continue
            try:
                document_id = _action_int(arguments, "document_id")
            except ValueError as exc:
                result_summary = f"{exc}. Valid document IDs: {', '.join(str(doc['id']) for doc in current_documents)}"
                tool_message = _persist_step(
                    insert_message,
                    connection,
                    conversation_id,
                    message_kind="director_tool",
                    content=result_summary,
                    metadata=metadata,
                    llm_provider=director_config["provider"],
                    llm_model=director_config["model"],
                    parent_message_id=rationale_message["id"],
                )
                assistant_messages.append(tool_message)
                prior_steps.append(tool_message)
                state["last_result"] = result_summary
                commit_connection(connection)
                continue
            doc = next((item for item in current_documents if item["id"] == document_id), None)
            if not doc:
                result_summary = f"Document {document_id} not found."
            else:
                result_summary = f"# {doc['title']}\n\n{doc['content_markdown'][:6000]}"
                state["referenced_document_ids"].append(document_id)
        elif action_name == "scrape_website":
            if not scrape_url:
                result_summary = "No scrape URL configured for this conversation."
            else:
                state["scrape_calls"] += 1
                if state["scrape_calls"] > MAX_SCRAPE_CALLS:
                    result_summary = "Scrape budget exhausted."
                else:
                    try:
                        query = arguments.get("query")
                        scrape_result = scrape_website_content(
                            scrape_url,
                            query,
                            depth=scrape_depth,
                            progress_callback=scrape_progress,
                        )
                        current_documents = load_documents(connection, conversation_id)
                        document = _matching_scrape_document(current_documents, scrape_result["url"], scrape_depth, query)
                        if not document:
                            document = _create_scrape_document(connection, conversation_id, scrape_result, query)
                            current_documents = load_documents(connection, conversation_id)
                        state["referenced_document_ids"].append(document["id"])
                        result_summary = (
                            f"Scraped {scrape_result['url']} with query '{scrape_result.get('query')}'. "
                            f"Depth {scrape_result.get('depth')} across {scrape_result.get('pages_scraped')} page(s). "
                            f"Stored document #{document['id']}.\n\n{scrape_result['summary'][:2000]}"
                        )
                        metadata["document_id"] = document["id"]
                        metadata["scrape_url"] = scrape_result["url"]
                        metadata["scrape_depth"] = scrape_result.get("depth")
                        metadata["pages_scraped"] = scrape_result.get("pages_scraped")
                        metadata["page_urls"] = scrape_result.get("page_urls")
                        metadata["structure"] = scrape_result.get("structure")
                        metadata["extracted_chars"] = scrape_result.get("extracted_chars")
                    except Exception as exc:
                        result_summary = f"Scrape failed: {exc}"
            tool_message = _persist_step(
                insert_message,
                connection,
                conversation_id,
                message_kind="director_tool",
                content=result_summary,
                metadata=metadata,
                llm_provider=director_config["provider"],
                llm_model=director_config["model"],
                parent_message_id=rationale_message["id"],
            )
            assistant_messages.append(tool_message)
            prior_steps.append(tool_message)
            state["last_result"] = result_summary
            commit_connection(connection)
            continue
        elif action_name == "consult_agents":
            question = (arguments.get("question") or goal or "").strip() or "Which specialist should lead this work?"
            expected_output = (arguments.get("expected_output") or "Brief opinion on fit and approach.").strip()
            opinion_sections: List[str] = []
            for participant in participants:
                personality = _participant_role_text(participant)
                name = participant_display_name(participant)
                personality_block = personality or f"You are {name}, a focused specialist."
                consult_prompt = CONSULT_AGENT_PROMPT.format(
                    personality_block=personality_block,
                    goal=goal or "(not set)",
                    success_criteria=success_criteria or "(not set)",
                    question=question,
                    expected_output=expected_output,
                )
                model_config = llm_model_or_404(connection, participant["llm_model_id"])
                consult_started = time.perf_counter()
                try:
                    opinion_content = llm_chat(
                        provider=model_config["provider"],
                        base_url=model_config["base_url"],
                        model=model_config["model"],
                        api_key=model_config["api_key"],
                        messages=[
                            {"role": "system", "content": consult_prompt},
                            {"role": "user", "content": question},
                        ],
                        temperature=0.3,
                    ).strip()
                except (RuntimeError, ValueError) as exc:
                    opinion_content = f"Opinion unavailable: {exc}"

                consult_generation_ms = max(0, int((time.perf_counter() - consult_started) * 1000))
                consult_message = _persist_step(
                    insert_message,
                    connection,
                    conversation_id,
                    message_kind="agent_reply",
                    content=opinion_content,
                    metadata={
                        "participant_id": participant["id"],
                        "consultation": True,
                        "question": question,
                    },
                    participant_id=participant["id"],
                    llm_provider=model_config["provider"],
                    llm_model=model_config["model"],
                    generation_ms=consult_generation_ms,
                    parent_message_id=rationale_message["id"],
                )
                assistant_messages.append(consult_message)
                prior_steps.append(consult_message)
                opinion_sections.append(
                    f"### {name} (id={participant['id']})\n{opinion_content}"
                )
                commit_connection(connection)

            combined_opinions = "\n\n".join(opinion_sections)
            consult_synthesis_prompt = (
                "All attached specialists have shared brief opinions. As Director, compare their fit against the goal "
                "and success criteria, choose the best specialist to delegate to next (by id), or synthesize directly "
                "if one response is sufficient. Do not ask the user questions.\n\n"
                f"Goal:\n{goal or '(not set)'}\n\n"
                f"Success criteria:\n{success_criteria or '(not set)'}\n\n"
                f"Consultation question:\n{question}\n\n"
                f"Specialist opinions:\n{combined_opinions[:8000]}"
            )
            try:
                consultation_summary = llm_chat(
                    provider=director_config["provider"],
                    base_url=director_config["base_url"],
                    model=director_config["model"],
                    api_key=director_config["api_key"],
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": consult_synthesis_prompt},
                    ],
                    temperature=0.2,
                ).strip()
            except (RuntimeError, ValueError) as exc:
                consultation_summary = (
                    "Specialist opinions collected, but Director synthesis failed. "
                    f"Review the opinions and choose the best participant_id for call_agent.\n\n{combined_opinions[:4000]}\n\n"
                    f"Synthesis error: {exc}"
                )

            consultation_message = _persist_step(
                insert_message,
                connection,
                conversation_id,
                message_kind="director_synthesis",
                content=consultation_summary,
                metadata={"consultation": True, "question": question},
                llm_provider=director_config["provider"],
                llm_model=director_config["model"],
                parent_message_id=rationale_message["id"],
            )
            assistant_messages.append(consultation_message)
            prior_steps.append(consultation_message)
            state["last_result"] = (
                f"Consulted {len(participants)} specialist(s).\n\n{consultation_summary}\n\n"
                "Next step: delegate with call_agent using the chosen participant_id, or generate_report/complete if ready."
            )
            commit_connection(connection)
            continue
        elif action_name == "call_agent":
            try:
                participant_id = _action_int(arguments, "participant_id")
            except ValueError:
                resolved_participant_id = _resolve_participant_id_from_context(action, participants, goal)
                if resolved_participant_id is None:
                    ranked = _rank_participants_for_delegation(action, participants, goal)
                    if ranked:
                        top_score, top_id, top_name = ranked[0]
                        second_score = ranked[1][0] if len(ranked) > 1 else -1
                        if top_score > second_score:
                            resolved_participant_id = top_id
                if resolved_participant_id is not None:
                    participant_id = resolved_participant_id
                    arguments["participant_id"] = participant_id
                    metadata["auto_selected_participant_id"] = participant_id
                else:
                    valid_ids = ", ".join(str(participant["id"]) for participant in participants)
                    result_summary = (
                        f"Missing required argument: participant_id. Valid participant IDs: {valid_ids or '(none)'}. "
                        "Choose the best-matching specialist id from the roster, or use consult_agents when unsure."
                    )
                    tool_message = _persist_step(
                        insert_message,
                        connection,
                        conversation_id,
                        message_kind="director_tool",
                        content=result_summary,
                        metadata=metadata,
                        llm_provider=director_config["provider"],
                        llm_model=director_config["model"],
                        parent_message_id=rationale_message["id"],
                    )
                    assistant_messages.append(tool_message)
                    prior_steps.append(tool_message)
                    state["last_result"] = result_summary
                    commit_connection(connection)
                    continue
            participant = participant_by_id.get(participant_id)
            if not participant:
                result_summary = f"Unknown participant id {participant_id}."
                tool_message = _persist_step(
                    insert_message,
                    connection,
                    conversation_id,
                    message_kind="director_error",
                    content=result_summary,
                    metadata=metadata,
                    parent_message_id=rationale_message["id"],
                )
                assistant_messages.append(tool_message)
                prior_steps.append(tool_message)
                state["last_result"] = result_summary
                commit_connection(connection)
                continue

            task = (arguments.get("task") or "").strip()
            expected_output = (arguments.get("expected_output") or "A clear specialist answer.").strip()
            if _is_repeated_delegation(participant_id, task, prior_steps):
                result_summary = (
                    "Repeated delegation blocked. The Director already asked this specialist the same task; "
                    "it must synthesize the prior answer, choose a different next action, or complete."
                )
                tool_message = _persist_step(
                    insert_message,
                    connection,
                    conversation_id,
                    message_kind="director_tool",
                    content=result_summary,
                    metadata={**metadata, "participant_id": participant_id, "blocked_repeat": True},
                    llm_provider=director_config["provider"],
                    llm_model=director_config["model"],
                    parent_message_id=rationale_message["id"],
                )
                assistant_messages.append(tool_message)
                prior_steps.append(tool_message)
                state["last_result"] = result_summary
                commit_connection(connection)
                continue
            current_documents = load_documents(connection, conversation_id)
            delegation_evidence, delegation_document_ids = _build_delegation_evidence(
                last_result=state.get("last_result"),
                task=task,
                expected_output=expected_output,
                rationale=(action.get("rationale") or "").strip(),
                arguments=arguments,
                documents=current_documents,
            )
            delegation_sections = [
                f"Request to {participant_display_name(participant)}",
                f"## Instructions\n{task}",
                f"## Expected output\n{expected_output}",
            ]
            if delegation_document_ids:
                document_titles = []
                by_id = {document["id"]: document for document in current_documents}
                for document_id in delegation_document_ids:
                    document = by_id.get(document_id)
                    if document:
                        kind = document.get("kind") or "uploaded"
                        document_titles.append(f"- [{document_id}] {document['title']} ({kind})")
                if document_titles:
                    delegation_sections.append("## Source documents\n" + "\n".join(document_titles))
            delegation_message = _persist_step(
                insert_message,
                connection,
                conversation_id,
                message_kind="director_delegation",
                content="\n\n".join(delegation_sections),
                metadata={
                    **metadata,
                    "participant_id": participant_id,
                    "expected_output": expected_output,
                    "document_ids": delegation_document_ids,
                },
                llm_provider=director_config["provider"],
                llm_model=director_config["model"],
                parent_message_id=rationale_message["id"],
            )
            assistant_messages.append(delegation_message)
            prior_steps.append(delegation_message)
            if delegation_document_ids:
                state["referenced_document_ids"].extend(delegation_document_ids)
            commit_connection(connection)

            prompt = build_sub_agent_prompt(
                participant,
                task,
                expected_output,
                delegation_evidence,
            )
            memories = active_memories_fn(connection, conversation_id) if include_memories else []
            if memories:
                memory_text, memory_ids = _format_memories(memories)
                state["referenced_memory_ids"].extend(memory_ids)
                prompt += f"\n\nConversation memory bank:\n{memory_text}"

            model_config = llm_model_or_404(connection, participant["llm_model_id"])
            sub_started = time.perf_counter()
            try:
                agent_content = llm_chat(
                    provider=model_config["provider"],
                    base_url=model_config["base_url"],
                    model=model_config["model"],
                    api_key=model_config["api_key"],
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": task},
                    ],
                )
            except (RuntimeError, ValueError) as exc:
                agent_content = f"Sub-agent call failed: {exc}"

            sub_generation_ms = max(0, int((time.perf_counter() - sub_started) * 1000))
            agent_message = _persist_step(
                insert_message,
                connection,
                conversation_id,
                message_kind="agent_reply",
                content=agent_content.strip(),
                metadata={"participant_id": participant_id, "task": task},
                participant_id=participant_id,
                llm_provider=model_config["provider"],
                llm_model=model_config["model"],
                generation_ms=sub_generation_ms,
                parent_message_id=delegation_message["id"],
            )
            assistant_messages.append(agent_message)
            prior_steps.append(agent_message)
            state["last_result"] = agent_content.strip()
            commit_connection(connection)

            if _looks_like_clarification_request(agent_content) and not agent_content.startswith("Sub-agent call failed:"):
                website_context = ""
                clarification_metadata = {
                    "participant_id": participant_id,
                    "task": task,
                    "agent_message_id": agent_message["id"],
                }
                if scrape_url and state["scrape_calls"] < MAX_SCRAPE_CALLS:
                    state["scrape_calls"] += 1
                    try:
                        query = agent_content.strip()
                        scrape_result = scrape_website_content(
                            scrape_url,
                            query,
                            depth=scrape_depth,
                            progress_callback=scrape_progress,
                        )
                        website_context = scrape_result["summary"]
                        current_documents = load_documents(connection, conversation_id)
                        document = _matching_scrape_document(current_documents, scrape_result["url"], scrape_depth, query)
                        if not document:
                            document = _create_scrape_document(connection, conversation_id, scrape_result, query)
                            current_documents = load_documents(connection, conversation_id)
                        state["referenced_document_ids"].append(document["id"])
                        clarification_metadata.update(
                            {
                                "document_id": document["id"],
                                "scrape_url": scrape_result["url"],
                                "scrape_depth": scrape_result.get("depth"),
                                "pages_scraped": scrape_result.get("pages_scraped"),
                                "extracted_chars": scrape_result.get("extracted_chars"),
                            }
                        )
                    except Exception as exc:
                        website_context = f"Website lookup failed: {exc}"
                        clarification_metadata["scrape_error"] = str(exc)

                try:
                    clarification_answer = _director_clarification_answer(
                        director_config=director_config,
                        system_prompt=system_prompt,
                        goal=goal,
                        success_criteria=success_criteria,
                        agent_question=agent_content.strip(),
                        documents=current_documents,
                        prior_steps=prior_steps,
                        website_context=website_context,
                    )
                except (RuntimeError, ValueError) as exc:
                    clarification_answer = (
                        "Proceed using the original user goal and success criteria. "
                        f"Make reasonable assumptions where details are missing. Director clarification failed: {exc}"
                    )
                    clarification_metadata["clarification_error"] = str(exc)

                clarification_message = _persist_step(
                    insert_message,
                    connection,
                    conversation_id,
                    message_kind="director_clarification",
                    content=clarification_answer,
                    metadata=clarification_metadata,
                    llm_provider=director_config["provider"],
                    llm_model=director_config["model"],
                    parent_message_id=agent_message["id"],
                )
                assistant_messages.append(clarification_message)
                prior_steps.append(clarification_message)
                state["last_result"] = clarification_answer
                commit_connection(connection)

                clarified_prompt = (
                    f"{prompt}\n\nDirector clarification:\n{clarification_answer}\n\n"
                    "Continue the delegated task now. Do not ask more questions unless the task is impossible; "
                    "use the Director clarification and state any assumptions."
                )
                clarified_started = time.perf_counter()
                try:
                    clarified_agent_content = llm_chat(
                        provider=model_config["provider"],
                        base_url=model_config["base_url"],
                        model=model_config["model"],
                        api_key=model_config["api_key"],
                        messages=[
                            {"role": "system", "content": clarified_prompt},
                            {"role": "user", "content": task},
                            {"role": "assistant", "content": agent_content.strip()},
                            {"role": "user", "content": f"Director clarification:\n{clarification_answer}"},
                        ],
                    )
                except (RuntimeError, ValueError) as exc:
                    clarified_agent_content = f"Sub-agent call failed after Director clarification: {exc}"

                clarified_generation_ms = max(0, int((time.perf_counter() - clarified_started) * 1000))
                agent_content = clarified_agent_content.strip()
                agent_message = _persist_step(
                    insert_message,
                    connection,
                    conversation_id,
                    message_kind="agent_reply",
                    content=agent_content,
                    metadata={
                        "participant_id": participant_id,
                        "task": task,
                        "clarification_message_id": clarification_message["id"],
                    },
                    participant_id=participant_id,
                    llm_provider=model_config["provider"],
                    llm_model=model_config["model"],
                    generation_ms=clarified_generation_ms,
                    parent_message_id=clarification_message["id"],
                )
                assistant_messages.append(agent_message)
                prior_steps.append(agent_message)
                state["last_result"] = agent_content
                commit_connection(connection)

            try:
                synthesis = _director_synthesize_agent_response(
                    director_config=director_config,
                    system_prompt=system_prompt,
                    goal=goal,
                    success_criteria=success_criteria,
                    task=task,
                    agent_content=agent_content.strip(),
                    prior_steps=prior_steps,
                )
            except (RuntimeError, ValueError) as exc:
                synthesis = (
                    "Director synthesis unavailable. Treating the specialist reply as partial evidence. "
                    f"Synthesis failed: {exc}\n\nSpecialist reply:\n{agent_content.strip()[:3000]}"
                )
            synthesis_message = _persist_step(
                insert_message,
                connection,
                conversation_id,
                message_kind="director_synthesis",
                content=synthesis,
                metadata={"participant_id": participant_id, "task": task, "agent_message_id": agent_message["id"]},
                llm_provider=director_config["provider"],
                llm_model=director_config["model"],
                parent_message_id=agent_message["id"],
            )
            assistant_messages.append(synthesis_message)
            prior_steps.append(synthesis_message)
            state["last_result"] = synthesis
            commit_connection(connection)

            eval_started = time.perf_counter()
            eval_messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        "Evaluate the Director synthesis of the latest specialist response against the goal and success criteria. "
                        "Use partial credit for useful extracted information even if the deliverable is incomplete. "
                        "Do not recommend repeating the same task to the same specialist unless new evidence changes the task. "
                        "The assessment MUST explain why the numeric percentage was assigned. "
                        "Return JSON: {\"success_score\": 0-100 percentage, \"remaining_gaps\": [\"...\"], "
                        "\"assessment\": \"...\", \"ready_to_complete\": true/false}"
                    ),
                },
                {"role": "assistant", "content": synthesis},
            ]
            try:
                eval_raw = llm_chat(
                    provider=director_config["provider"],
                    base_url=director_config["base_url"],
                    model=director_config["model"],
                    api_key=director_config["api_key"],
                    messages=eval_messages,
                    temperature=0.1,
                )
                evaluation = _extract_json_object(eval_raw)
            except Exception:
                evaluation = _fallback_evaluation(agent_content, synthesis)
            eval_generation_ms = max(0, int((time.perf_counter() - eval_started) * 1000))
            try:
                state["success_score"] = max(0, min(100, int(evaluation.get("success_score") or 0)))
            except (TypeError, ValueError):
                state["success_score"] = _fallback_success_score(agent_content)
            state["remaining_gaps"] = list(evaluation.get("remaining_gaps") or [])
            assessment = (evaluation.get("assessment") or "").strip()
            if not assessment:
                assessment = (
                    f"Success score is {state['success_score']}% based on the Director synthesis. "
                    "The evaluator did not provide a detailed explanation."
                )
            evaluation_message = _persist_step(
                insert_message,
                connection,
                conversation_id,
                message_kind="director_evaluation",
                content=assessment,
                metadata={
                    "success_score": state["success_score"],
                    "remaining_gaps": state["remaining_gaps"],
                    "ready_to_complete": bool(evaluation.get("ready_to_complete")),
                    "participant_id": participant_id,
                },
                llm_provider=director_config["provider"],
                llm_model=director_config["model"],
                generation_ms=eval_generation_ms,
                parent_message_id=agent_message["id"],
            )
            assistant_messages.append(evaluation_message)
            prior_steps.append(evaluation_message)
            state["last_result"] = evaluation_message["content"]
            if evaluation.get("ready_to_complete"):
                state["done"] = True
                state["final_answer"] = evaluation_message["content"]
                final_status = "completed"
            commit_connection(connection)
            continue
        elif action_name == "generate_report":
            if state.get("report_generated"):
                state["done"] = True
                state["final_answer"] = (
                    "The output and process/rationale documents have already been generated for this task. "
                    "Completing instead of generating another report draft."
                )
                final_message = _persist_step(
                    insert_message,
                    connection,
                    conversation_id,
                    message_kind="director_final",
                    content=state["final_answer"],
                    metadata={
                        "success_score": state.get("success_score", 0),
                        "remaining_gaps": state.get("remaining_gaps") or [],
                        "duplicate_report_blocked": True,
                    },
                    llm_provider=director_config["provider"],
                    llm_model=director_config["model"],
                    parent_message_id=rationale_message["id"],
                )
                assistant_messages.append(final_message)
                prior_steps.append(final_message)
                final_status = "completed"
                commit_connection(connection)
                continue
            title = _deliverable_title((arguments.get("title") or "Task report").strip() or "Task report")
            format_request = (arguments.get("format_request") or report_format or "").strip() or None
            deliverable_body = _resolve_deliverable_body(
                steps=prior_steps,
                explicit_content=arguments.get("content"),
                final_answer=state.get("final_answer") or state.get("last_result"),
            )
            document = _create_deliverable_documents(
                connection,
                conversation_id,
                title=title,
                format_request=format_request,
                goal=goal,
                success_criteria=success_criteria,
                deliverable_body=deliverable_body,
                director_summary=state.get("final_answer") or "",
                steps=prior_steps,
                state=state,
                insert_message=insert_message,
                director_config=director_config,
                assistant_messages=assistant_messages,
                prior_steps=prior_steps,
                parent_message_id=rationale_message["id"],
            )
            if not document:
                result_summary = (
                    "Could not generate deliverable document because no specialist output was available yet. "
                    "Delegate to a specialist first, then call generate_report with the deliverable in content."
                )
                tool_message = _persist_step(
                    insert_message,
                    connection,
                    conversation_id,
                    message_kind="director_tool",
                    content=result_summary,
                    metadata=metadata,
                    llm_provider=director_config["provider"],
                    llm_model=director_config["model"],
                    parent_message_id=rationale_message["id"],
                )
                assistant_messages.append(tool_message)
                prior_steps.append(tool_message)
                state["last_result"] = result_summary
                commit_connection(connection)
                continue
            commit_connection(connection)
            continue
        elif action_name == "complete":
            state["done"] = True
            state["final_answer"] = (arguments.get("final_answer") or "").strip()
            try:
                state["success_score"] = int(arguments.get("success_score") or state.get("success_score") or 0)
            except (TypeError, ValueError):
                state["success_score"] = int(state.get("success_score") or 0)
            state["remaining_gaps"] = list(arguments.get("remaining_gaps") or state.get("remaining_gaps") or [])
            final_message = _persist_step(
                insert_message,
                connection,
                conversation_id,
                message_kind="director_final",
                content=state["final_answer"] or (arguments.get("success_assessment") or "Task complete.").strip(),
                metadata={
                    "success_score": state["success_score"],
                    "remaining_gaps": state["remaining_gaps"],
                    "success_assessment": arguments.get("success_assessment"),
                },
                llm_provider=director_config["provider"],
                llm_model=director_config["model"],
                parent_message_id=rationale_message["id"],
            )
            assistant_messages.append(final_message)
            prior_steps.append(final_message)
            final_status = "completed"
            commit_connection(connection)
            continue
        else:
            result_summary = f"Unsupported director action: {action_name}"

        if action_name in {
            "search_conversation_memories",
            "search_all_memories",
            "search_documents",
            "read_document",
        }:
            tool_message = _persist_step(
                insert_message,
                connection,
                conversation_id,
                message_kind="director_tool",
                content=result_summary,
                metadata=metadata,
                llm_provider=director_config["provider"],
                llm_model=director_config["model"],
                parent_message_id=rationale_message["id"],
            )
            assistant_messages.append(tool_message)
            prior_steps.append(tool_message)
            state["last_result"] = result_summary
            commit_connection(connection)

    if not state["done"]:
        stop_message = _persist_step(
            insert_message,
            connection,
            conversation_id,
            message_kind="director_final",
            content=state.get("final_answer")
            or "Director stopped before success criteria were fully met.",
            metadata={
                "success_score": state.get("success_score", 0),
                "remaining_gaps": state.get("remaining_gaps") or ["Iteration budget exhausted."],
                "stopped": True,
            },
            llm_provider=director_config["provider"],
            llm_model=director_config["model"],
        )
        assistant_messages.append(stop_message)
        commit_connection(connection)

    if not state.get("report_generated"):
        latest_steps = _load_agentic_steps(connection, conversation_id)
        auto_title = _deliverable_title((report_format or goal[:80] or "Task deliverable").strip())
        deliverable_body = _resolve_deliverable_body(
            steps=latest_steps,
            final_answer=state.get("final_answer") or state.get("last_result"),
        )
        _create_deliverable_documents(
            connection,
            conversation_id,
            title=auto_title,
            format_request=report_format or None,
            goal=goal,
            success_criteria=success_criteria,
            deliverable_body=deliverable_body,
            director_summary=state.get("final_answer") or "",
            steps=latest_steps,
            state=state,
            insert_message=insert_message,
            director_config=director_config,
            assistant_messages=assistant_messages,
            prior_steps=latest_steps,
        )
        commit_connection(connection)

    if state.get("done") and final_status != "stopped":
        final_status = "completed"
    _set_agentic_progress(connection, conversation_id, None)
    connection.execute(
        "UPDATE conversations SET agentic_status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (final_status, conversation_id),
    )
    commit_connection(connection)
    return assistant_messages
