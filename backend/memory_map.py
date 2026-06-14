import json
import re
import uuid
from typing import List, Literal

from fastapi import HTTPException
from pydantic import TypeAdapter, ValidationError

from .database import row_to_dict
from .llm import chat as llm_chat
from .models import (
    Memory,
    MemoryMapCreate,
    MemoryMapGraphSpec,
    MemoryMapNodeSpec,
    MemoryMapResponse,
    MemoryMapSpec,
    MemoryMapWordSpec,
    MemoryMapWordcloudSpec,
    VizClientState,
    VizClientStateUpdate,
    VizSpecDetail,
    VizSpecSummary,
)

MAX_MEMORIES_PER_MAP = 30
MEMORY_MAP_SPEC_ADAPTER = TypeAdapter(MemoryMapSpec)

VIZ_HINT_INSTRUCTIONS = {
    "graph": 'Use type "graph" with nodes and edges showing relationships between concepts.',
    "mindmap": 'Use type "mindmap" with a hierarchical root node and children branches.',
    "kanban": 'Use type "kanban" with columns grouping related items (e.g. priority tiers, status, themes).',
    "wordcloud": (
        'Use type "wordcloud" with short keyword-style words and weights (1-10) for emphasis — '
        "ideal for priorities, tags, or brief text fragments."
    ),
    "auto": (
        'Pick the best type: "wordcloud" for short keywords or brief text fragments, "graph" for cross-links, '
        '"mindmap" for a single theme with branches, or "kanban" for grouping by status, priority, or category.'
    ),
}


def _public_memory(memory: dict) -> dict:
    public = dict(memory)
    public["title_pending"] = not public.get("title_generated_at")
    public.pop("title_generated_at", None)
    return public


def _conversation_or_404(connection, conversation_id: int) -> dict:
    conversation = row_to_dict(
        connection.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
    )
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


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


def _conversation_llm_model(connection, conversation: dict) -> dict:
    model_id = conversation.get("llm_model_id")
    if model_id is not None:
        model = row_to_dict(connection.execute("SELECT * FROM llm_models WHERE id = ?", (model_id,)).fetchone())
        if model:
            return model
    return _active_llm_model(connection)


def _active_memories(connection, conversation_id: int) -> List[dict]:
    rows = connection.execute(
        """
        SELECT * FROM memories
        WHERE conversation_id = ? AND archived_at IS NULL
        ORDER BY created_at ASC
        """,
        (conversation_id,),
    ).fetchall()
    return [_public_memory(dict(row)) for row in rows]


def _all_active_memories(connection, *, exclude_conversation_id: int) -> List[dict]:
    rows = connection.execute(
        """
        SELECT * FROM memories
        WHERE archived_at IS NULL AND conversation_id != ?
        ORDER BY created_at ASC
        """,
        (exclude_conversation_id,),
    ).fetchall()
    return [_public_memory(dict(row)) for row in rows]


def _extract_json_object(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response")
    return json.loads(stripped[start : end + 1])


def _collect_memories_for_map(
    connection,
    conversation_id: int,
    *,
    include_memories: bool,
    include_all_memories: bool,
) -> List[dict]:
    memories: List[dict] = []
    if include_memories:
        memories.extend(_active_memories(connection, conversation_id))
    if include_all_memories:
        memories.extend(_all_active_memories(connection, exclude_conversation_id=conversation_id))
    seen_ids: set[int] = set()
    unique: List[dict] = []
    for memory in memories:
        memory_id = memory["id"]
        if memory_id in seen_ids:
            continue
        seen_ids.add(memory_id)
        unique.append(memory)
    return unique[:MAX_MEMORIES_PER_MAP]


def _memory_word_count(memory: dict) -> int:
    text = f"{memory.get('title') or ''} {memory.get('content') or ''}".strip()
    return len(text.split())


def _memories_suit_wordcloud(memories: List[dict]) -> bool:
    if not memories:
        return False
    word_counts = [_memory_word_count(memory) for memory in memories]
    average_words = sum(word_counts) / len(word_counts)
    short_memories = sum(1 for count in word_counts if count <= 12)
    return average_words <= 14 or short_memories >= max(1, len(memories) // 2)


def _fallback_wordcloud_spec(memories: List[dict], title: str = "Memory word map") -> MemoryMapWordcloudSpec:
    words: List[MemoryMapWordSpec] = []
    for index, memory in enumerate(memories):
        label = (memory.get("title") or memory.get("content") or f"Memory {memory['id']}").strip()
        if len(label) > 48:
            label = f"{label[:48].rstrip()}..."
        weight = max(1, min(10, 10 - index))
        words.append(
            MemoryMapWordSpec(
                id=f"word-{memory['id']}",
                text=label,
                weight=weight,
                memory_ids=[memory["id"]],
            )
        )
    return MemoryMapWordcloudSpec(type="wordcloud", title=title, words=words)


def _fallback_graph_spec(memories: List[dict], title: str = "Memory map") -> MemoryMapGraphSpec:
    nodes = [
        MemoryMapNodeSpec(
            id=f"memory-{memory['id']}",
            label=memory.get("title") or f"Memory {memory['id']}",
            detail=(memory.get("content") or "")[:240] or None,
            memory_ids=[memory["id"]],
        )
        for memory in memories
    ]
    return MemoryMapGraphSpec(type="graph", title=title, nodes=nodes, edges=[])


def _build_memory_map_prompt(
    memories: List[dict],
    connection,
    viz_hint: Literal["graph", "mindmap", "kanban", "wordcloud", "auto"],
) -> str:
    conversation_titles = {
        row["id"]: row["title"]
        for row in connection.execute("SELECT id, title FROM conversations").fetchall()
    }
    lines = []
    for memory in memories:
        conversation_title = conversation_titles.get(memory["conversation_id"], "Unknown")
        content = (memory.get("content") or "").strip()
        if len(content) > 800:
            content = f"{content[:800].rstrip()}..."
        lines.append(
            json.dumps(
                {
                    "id": memory["id"],
                    "title": memory.get("title") or f"Memory {memory['id']}",
                    "content": content,
                    "conversation_title": conversation_title,
                },
                ensure_ascii=False,
            )
        )
    memory_block = "\n".join(lines)
    hint = VIZ_HINT_INSTRUCTIONS[viz_hint]
    return (
        "Create an interactive visualization spec from these saved chat memories.\n"
        f"{hint}\n\n"
        "Return ONLY valid JSON matching exactly one of these shapes:\n"
        '{"type":"graph","title":"...","nodes":[{"id":"n1","label":"...","detail":"...","memory_ids":[1]}],'
        '"edges":[{"source":"n1","target":"n2","label":"..."}]}\n'
        '{"type":"mindmap","title":"...","root":{"id":"root","label":"...","detail":"...","memory_ids":[],"children":[{"id":"m1","label":"...","memory_ids":[1]}]}}\n'
        '{"type":"kanban","title":"...","columns":[{"id":"c1","title":"...","cards":[{"id":"card1","title":"...",'
        '"body":"...","memory_ids":[1]}]}]}\n'
        '{"type":"wordcloud","title":"...","words":[{"id":"w1","text":"Health","weight":8,"memory_ids":[1]}]}\n\n'
        "Rules:\n"
        "- Every node/card/word should reference relevant memory_ids from the input.\n"
        "- Use short labels; put longer text in detail/body fields.\n"
        "- For wordcloud, use concise words or short phrases (1-4 words each) with weight 1-10.\n"
        "- For mindmap, put each memory on its own child branch; do not attach all memory_ids to the root.\n"
        "- Do not invent memory ids.\n"
        "- Do not include x/y coordinates.\n\n"
        f"Memories:\n{memory_block}"
    )


def _parse_memory_map_spec(raw: str) -> MemoryMapSpec:
    payload = _extract_json_object(raw)
    return MEMORY_MAP_SPEC_ADAPTER.validate_python(payload)


def _fallback_spec(memories: List[dict], viz_hint: Literal["graph", "mindmap", "kanban", "wordcloud", "auto"]) -> MemoryMapSpec:
    if viz_hint == "wordcloud" or (viz_hint == "auto" and _memories_suit_wordcloud(memories)):
        return _fallback_wordcloud_spec(memories)
    return _fallback_graph_spec(memories)


def _generate_memory_map_spec(
    connection,
    conversation_id: int,
    memories: List[dict],
    viz_hint: Literal["graph", "mindmap", "kanban", "wordcloud", "auto"],
) -> MemoryMapSpec:
    fallback = _fallback_spec(memories, viz_hint)
    try:
        conversation = _conversation_or_404(connection, conversation_id)
        config = _conversation_llm_model(connection, conversation)
        prompt = _build_memory_map_prompt(memories, connection, viz_hint)
        raw = llm_chat(
            provider=config["provider"],
            base_url=config["base_url"],
            model=config["model"],
            api_key=config["api_key"],
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You produce strict JSON visualization specs for concept maps from saved memories. "
                        "Return JSON only with no markdown or commentary."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        ).strip()
        try:
            return _parse_memory_map_spec(raw)
        except (ValueError, ValidationError, json.JSONDecodeError):
            repair = llm_chat(
                provider=config["provider"],
                base_url=config["base_url"],
                model=config["model"],
                api_key=config["api_key"],
                temperature=0.0,
                messages=[
                    {"role": "system", "content": "Fix the JSON visualization spec. Return JSON only."},
                    {"role": "user", "content": f"Fix this JSON:\n\n{raw[:12000]}"},
                ],
            ).strip()
            return _parse_memory_map_spec(repair)
    except HTTPException:
        raise
    except Exception:
        return fallback


def create_memory_map(connection, conversation_id: int, payload: MemoryMapCreate) -> MemoryMapResponse:
    _conversation_or_404(connection, conversation_id)
    if not payload.include_memories and not payload.include_all_memories:
        raise HTTPException(status_code=400, detail="Enable at least one memory scope")

    memories = _collect_memories_for_map(
        connection,
        conversation_id,
        include_memories=payload.include_memories,
        include_all_memories=payload.include_all_memories,
    )
    if not memories:
        raise HTTPException(status_code=400, detail="No saved memories match the selected scope")

    spec = _generate_memory_map_spec(connection, conversation_id, memories, payload.viz_hint)
    viz_id = str(uuid.uuid4())
    memory_ids = [memory["id"] for memory in memories]
    connection.execute(
        """
        INSERT INTO viz_specs (id, conversation_id, spec_json, memory_ids_json)
        VALUES (?, ?, ?, ?)
        """,
        (viz_id, conversation_id, spec.model_dump_json(), json.dumps(memory_ids)),
    )
    return MemoryMapResponse(viz_id=viz_id, spec=spec)


def _spec_title(spec: MemoryMapSpec) -> str:
    return (spec.title or "Memory map").strip() or "Memory map"


def list_conversation_viz_specs(connection, conversation_id: int) -> List[VizSpecSummary]:
    _conversation_or_404(connection, conversation_id)
    rows = connection.execute(
        """
        SELECT id, conversation_id, spec_json, memory_ids_json, created_at, updated_at
        FROM viz_specs
        WHERE conversation_id = ?
        ORDER BY COALESCE(updated_at, created_at) DESC, created_at DESC
        """,
        (conversation_id,),
    ).fetchall()
    summaries: List[VizSpecSummary] = []
    for row in rows:
        spec = MEMORY_MAP_SPEC_ADAPTER.validate_json(row["spec_json"])
        memory_ids = json.loads(row["memory_ids_json"] or "[]")
        summaries.append(
            VizSpecSummary(
                viz_id=row["id"],
                conversation_id=row["conversation_id"],
                title=_spec_title(spec),
                spec_type=spec.type,
                memory_count=len(memory_ids),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
        )
    return summaries


def update_viz_client_state(connection, viz_id: str, payload: VizClientStateUpdate) -> VizSpecDetail:
    row = connection.execute(
        "SELECT id FROM viz_specs WHERE id = ?",
        (viz_id,),
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Visualization not found")

    client_state = VizClientState(active_view=payload.active_view, specs=payload.specs)
    connection.execute(
        """
        UPDATE viz_specs
        SET client_state_json = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (client_state.model_dump_json(), viz_id),
    )
    return get_viz_spec_detail(connection, viz_id)


def get_viz_spec_detail(connection, viz_id: str) -> VizSpecDetail:
    row = connection.execute(
        "SELECT * FROM viz_specs WHERE id = ?",
        (viz_id,),
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Visualization not found")

    spec = MEMORY_MAP_SPEC_ADAPTER.validate_json(row["spec_json"])
    memory_ids = json.loads(row["memory_ids_json"] or "[]")
    memories: List[Memory] = []
    if memory_ids:
        placeholders = ",".join("?" for _ in memory_ids)
        rows = connection.execute(
            f"""
            SELECT * FROM memories
            WHERE id IN ({placeholders})
            ORDER BY created_at ASC, id ASC
            """,
            memory_ids,
        ).fetchall()
        memories = [Memory(**_public_memory(dict(memory_row))) for memory_row in rows]

    client_state = None
    if row["client_state_json"]:
        client_state = VizClientState.model_validate_json(row["client_state_json"])

    return VizSpecDetail(
        viz_id=row["id"],
        conversation_id=row["conversation_id"],
        spec=spec,
        memories=memories,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        client_state=client_state,
    )
