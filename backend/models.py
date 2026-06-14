from typing import Annotated, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

TTS_SPEECH_RATE_MIN = 0.5
TTS_SPEECH_RATE_MAX = 2.5


def _clamp_tts_speech_rate(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return max(TTS_SPEECH_RATE_MIN, min(TTS_SPEECH_RATE_MAX, value))


class ConversationParticipantUpdate(BaseModel):
    llm_model_id: int
    personality: str = ""
    name: str = ""
    tts_voice_uri: Optional[str] = None
    tts_speech_rate: Optional[float] = None
    agent_profile_id: Optional[int] = None

    @field_validator("tts_speech_rate")
    @classmethod
    def validate_tts_speech_rate(cls, value: Optional[float]) -> Optional[float]:
        return _clamp_tts_speech_rate(value)


class ConversationParticipantRead(BaseModel):
    id: int
    conversation_id: int
    llm_model_id: int
    personality: str
    name: str
    sort_order: int
    tts_voice_uri: Optional[str] = None
    tts_speech_rate: Optional[float] = None
    agent_profile_id: Optional[int] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    llm_comments: Optional[str] = None


class AgentProfileCreate(BaseModel):
    name: str = Field(min_length=1)
    personality: str = ""
    llm_model_id: int
    tts_voice_uri: Optional[str] = None
    tts_speech_rate: Optional[float] = None

    @field_validator("tts_speech_rate")
    @classmethod
    def validate_tts_speech_rate(cls, value: Optional[float]) -> Optional[float]:
        return _clamp_tts_speech_rate(value)


class AgentProfileUpdate(BaseModel):
    name: str = Field(min_length=1)
    personality: str = ""
    llm_model_id: int
    tts_voice_uri: Optional[str] = None
    tts_speech_rate: Optional[float] = None

    @field_validator("tts_speech_rate")
    @classmethod
    def validate_tts_speech_rate(cls, value: Optional[float]) -> Optional[float]:
        return _clamp_tts_speech_rate(value)


class AgentProfileRead(BaseModel):
    id: int
    name: str
    personality: str
    llm_model_id: int
    tts_voice_uri: Optional[str] = None
    tts_speech_rate: Optional[float] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    created_at: str
    updated_at: str


class AgentPersonalityDraftCreate(BaseModel):
    name: str = Field(min_length=1)
    seed_personality: str = ""


class AgentPersonalityDraftRead(BaseModel):
    personality: str


class ConversationParticipantsUpdate(BaseModel):
    participants: List[ConversationParticipantUpdate] = Field(min_length=1)
    mode: Optional[Literal["discussion", "agentic"]] = None


class ConversationAgenticSetupUpdate(BaseModel):
    participants: List[ConversationParticipantUpdate] = Field(min_length=1)
    agentic_goal: str = Field(min_length=1)
    agentic_success_criteria: str = Field(min_length=1)
    agentic_scrape_url: Optional[str] = None
    agentic_scrape_depth: int = Field(default=1, ge=1, le=3)
    agentic_report_format: Optional[str] = None
    agentic_max_iterations: int = Field(default=20, ge=1, le=50)

    @model_validator(mode="after")
    def validate_agentic_setup(self):
        if not self.agentic_goal.strip():
            raise ValueError("Agentic conversations require a goal")
        if not self.agentic_success_criteria.strip():
            raise ValueError("Agentic conversations require success criteria")
        return self


class ConversationCreate(BaseModel):
    title: Optional[str] = None
    mode: Literal["single", "discussion", "agentic"] = "single"
    participants: Optional[List[ConversationParticipantUpdate]] = None
    agentic_goal: Optional[str] = None
    agentic_success_criteria: Optional[str] = None
    agentic_scrape_url: Optional[str] = None
    agentic_scrape_depth: Optional[int] = Field(default=1, ge=1, le=3)
    agentic_report_format: Optional[str] = None
    agentic_max_iterations: Optional[int] = None

    @model_validator(mode="after")
    def validate_agentic_fields(self):
        if self.mode == "agentic":
            if not (self.agentic_goal or "").strip():
                raise ValueError("Agentic conversations require a goal")
            if not (self.agentic_success_criteria or "").strip():
                raise ValueError("Agentic conversations require success criteria")
            if not self.participants:
                raise ValueError("Agentic conversations require at least one agent")
        return self


class ConversationTitleUpdate(BaseModel):
    title: str = Field(min_length=1)


class ConversationDelete(BaseModel):
    memory_action: str = "delete"
    target_conversation_id: Optional[int] = None


class ConversationReorder(BaseModel):
    conversation_ids: List[int] = Field(min_length=1)


class ConversationModelUpdate(BaseModel):
    llm_model_id: int


class AgenticControlRequest(BaseModel):
    action: Literal["stop", "wrap"]


class Conversation(BaseModel):
    id: int
    title: str
    sort_order: int
    llm_model_id: Optional[int] = None
    participant_count: int = 0
    mode: str = "single"
    agentic_goal: Optional[str] = None
    agentic_success_criteria: Optional[str] = None
    agentic_scrape_url: Optional[str] = None
    agentic_scrape_depth: Optional[int] = None
    agentic_report_format: Optional[str] = None
    agentic_status: Optional[str] = None
    agentic_max_iterations: Optional[int] = None
    created_at: str
    updated_at: str


class Message(BaseModel):
    id: int
    conversation_id: int
    role: str
    content: str
    image_media_type: Optional[str] = None
    image_data: Optional[str] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    generation_ms: Optional[int] = None
    include_history: Optional[Union[bool, int]] = None
    participant_id: Optional[int] = None
    message_kind: Optional[str] = None
    metadata: Optional[dict] = None
    parent_message_id: Optional[int] = None
    created_at: str


class Memory(BaseModel):
    id: int
    conversation_id: int
    title: str
    content: str
    source_message_id: Optional[int] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    created_at: str
    archived_at: Optional[str] = None
    title_pending: bool = False


class MemoryGroup(BaseModel):
    conversation: Conversation
    memories: List[Memory]


class MemoryMove(BaseModel):
    target_conversation_id: int


class ConversationDetail(BaseModel):
    conversation: Conversation
    messages: List[Message]
    memories: List[Memory]
    participants: List[ConversationParticipantRead] = []
    documents: List["Document"] = []


class Document(BaseModel):
    id: int
    conversation_id: int
    title: str
    kind: str
    content_markdown: str
    source_filename: Optional[str] = None
    source_media_type: Optional[str] = None
    metadata: Optional[dict] = None
    created_at: str
    updated_at: str


class DocumentCreate(BaseModel):
    title: str = Field(min_length=1)
    content_markdown: str = Field(min_length=1)
    kind: Literal["uploaded", "generated_report", "generated_process"] = "uploaded"


class DocumentWebsiteUploadRequest(BaseModel):
    url: str = Field(min_length=1)
    title: Optional[str] = None
    depth: int = Field(default=1, ge=1, le=3)


class DocumentUpdate(BaseModel):
    title: Optional[str] = None
    content_markdown: Optional[str] = None


class DocumentUploadResponse(BaseModel):
    document: Document


class DocumentGenerateRequest(BaseModel):
    title: str = Field(min_length=1)
    format_request: Optional[str] = None
    include_provenance: bool = True


class MessageCreate(BaseModel):
    content: str = ""
    image_data: Optional[str] = None
    image_media_type: Optional[str] = None
    agentic_start: bool = False
    agentic_ad_hoc: bool = False
    include_history: Union[bool, int] = True
    include_memories: bool = True
    include_all_memories: bool = False
    discussion_rounds: int = 1
    answer_length: int = 3
    override_llm_model_id: Optional[int] = None

    @field_validator("answer_length")
    @classmethod
    def validate_answer_length(cls, value: int) -> int:
        return max(1, min(5, int(value)))

    @model_validator(mode="after")
    def validate_content_or_image(self):
        content = self.content.strip()
        image_data = (self.image_data or "").strip()
        if not content and not image_data and not self.agentic_start:
            raise ValueError("Provide text, an image, or both")
        if self.agentic_ad_hoc and not content:
            raise ValueError("Ad-hoc agentic messages require text")
        if image_data and not (self.image_media_type or "").strip():
            raise ValueError("image_media_type is required when image_data is provided")
        return self


class MessageResponse(BaseModel):
    user_message: Optional[Message] = None
    assistant_message: Optional[Message] = None
    assistant_messages: Optional[List[Message]] = None
    memory: Optional[Memory] = None


class LlmContextItem(BaseModel):
    role: str
    label: str
    content_preview: str
    has_image: bool = False
    image_bytes: int = 0
    char_estimate: int = 0


class LlmContextPreview(BaseModel):
    provider: str
    model: str
    include_history: Union[bool, int]
    include_memories: bool
    include_all_memories: bool
    memory_count: int
    all_memory_count: int
    items: List[LlmContextItem]
    total_chars: int
    approx_tokens: int
    image_count: int
    history_message_count: int
    images_resent_from_history: bool
    generation_estimate_sec: Optional[float] = None
    seconds_per_char: Optional[float] = None
    generation_sample_count: int = 0
    multi_agent_note: Optional[str] = None


class RememberCreate(BaseModel):
    content: Optional[str] = None
    message_id: Optional[int] = None


class MemoryMerge(BaseModel):
    memory_ids: List[int] = Field(min_length=2)
    content: str = Field(min_length=1)


class MemoryIntegrate(BaseModel):
    memory_ids: List[int] = Field(min_length=2)


class CrossConversationMemoryMerge(MemoryMerge):
    target_conversation_id: int


class LlmModelRead(BaseModel):
    id: int
    provider: str
    base_url: str
    model: str
    comments: Optional[str] = None
    has_api_key: bool
    api_key_preview: Optional[str] = None
    is_active: bool
    updated_at: str
    generation_sample_count: int = 0
    seconds_per_char: Optional[float] = None
    avg_generation_sec: Optional[float] = None
    reference_generation_estimate_sec: Optional[float] = None
    tts_voice_uri: Optional[str] = None


class LlmModelUpdate(BaseModel):
    id: Optional[int] = None
    provider: str = "ollama"
    base_url: str = Field(min_length=1)
    model: str = Field(min_length=1)
    comments: Optional[str] = None
    api_key: Optional[str] = None
    clear_api_key: bool = False
    is_active: bool = False
    tts_voice_uri: Optional[str] = None


class LlmConfigRead(BaseModel):
    models: List[LlmModelRead]
    active_model: LlmModelRead


class LlmConfigUpdate(BaseModel):
    models: List[LlmModelUpdate] = Field(min_length=1)


class SpeechConfigRead(BaseModel):
    whisper_model: str
    whisper_model_options: List[str]
    has_elevenlabs_api_key: bool
    elevenlabs_api_key_preview: Optional[str] = None
    updated_at: str


class SpeechConfigUpdate(BaseModel):
    whisper_model: Optional[str] = None
    elevenlabs_api_key: Optional[str] = None
    clear_elevenlabs_api_key: bool = False


class ElevenLabsVoiceRead(BaseModel):
    voice_id: str
    name: str
    gender: Optional[str] = None
    age: Optional[str] = None
    characteristics: Optional[str] = None
    label: str


class ElevenLabsSynthesizeRequest(BaseModel):
    text: str = Field(min_length=1)
    voice_id: str = Field(min_length=1)
    speech_rate: Optional[float] = None


class PromptConfigRead(BaseModel):
    default_prompt: str
    default_prompt_baseline: str
    multi_agent_prompt: str
    multi_agent_prompt_baseline: str
    director_prompt: str
    director_prompt_baseline: str
    updated_at: str


class PromptConfigUpdate(BaseModel):
    default_prompt: str = Field(min_length=1)
    multi_agent_prompt: str = Field(min_length=1)
    director_prompt: str = Field(min_length=1)


class MemoryMapNodeSpec(BaseModel):
    id: str
    label: str
    detail: Optional[str] = None
    memory_ids: List[int] = Field(default_factory=list)


class MemoryMapEdgeSpec(BaseModel):
    source: str
    target: str
    label: Optional[str] = None


class MemoryMapTreeNodeSpec(BaseModel):
    id: str
    label: str
    detail: Optional[str] = None
    memory_ids: List[int] = Field(default_factory=list)
    children: List["MemoryMapTreeNodeSpec"] = Field(default_factory=list)


class MemoryMapKanbanCardSpec(BaseModel):
    id: str
    title: str
    body: Optional[str] = None
    memory_ids: List[int] = Field(default_factory=list)


class MemoryMapKanbanColumnSpec(BaseModel):
    id: str
    title: str
    cards: List[MemoryMapKanbanCardSpec] = Field(default_factory=list)


class MemoryMapGraphSpec(BaseModel):
    type: Literal["graph"]
    title: str
    nodes: List[MemoryMapNodeSpec]
    edges: List[MemoryMapEdgeSpec] = Field(default_factory=list)


class MemoryMapMindmapSpec(BaseModel):
    type: Literal["mindmap"]
    title: str
    root: MemoryMapTreeNodeSpec


class MemoryMapKanbanSpec(BaseModel):
    type: Literal["kanban"]
    title: str
    columns: List[MemoryMapKanbanColumnSpec]


class MemoryMapWordSpec(BaseModel):
    id: str
    text: str
    weight: int = Field(ge=1, le=10)
    memory_ids: List[int] = Field(default_factory=list)


class MemoryMapWordcloudSpec(BaseModel):
    type: Literal["wordcloud"]
    title: str
    words: List[MemoryMapWordSpec]


MemoryMapSpec = Annotated[
    Union[MemoryMapGraphSpec, MemoryMapMindmapSpec, MemoryMapKanbanSpec, MemoryMapWordcloudSpec],
    Field(discriminator="type"),
]


class MemoryMapCreate(BaseModel):
    include_memories: bool = True
    include_all_memories: bool = False
    viz_hint: Literal["graph", "mindmap", "kanban", "wordcloud", "auto"] = "auto"


class MemoryMapResponse(BaseModel):
    viz_id: str
    spec: MemoryMapGraphSpec | MemoryMapMindmapSpec | MemoryMapKanbanSpec | MemoryMapWordcloudSpec


class VizSpecSummary(BaseModel):
    viz_id: str
    conversation_id: int
    title: str
    spec_type: str
    memory_count: int
    created_at: str
    updated_at: Optional[str] = None


class VizClientState(BaseModel):
    active_view: Literal["graph", "mindmap", "kanban", "wordcloud"]
    specs: dict


class VizClientStateUpdate(BaseModel):
    active_view: Literal["graph", "mindmap", "kanban", "wordcloud"]
    specs: dict


class VizSpecDetail(BaseModel):
    viz_id: str
    conversation_id: int
    spec: MemoryMapGraphSpec | MemoryMapMindmapSpec | MemoryMapKanbanSpec | MemoryMapWordcloudSpec
    memories: List[Memory]
    created_at: str
    updated_at: Optional[str] = None
    client_state: Optional[VizClientState] = None


class UsageSummary(BaseModel):
    conversation_count: int
    total_messages: int
    user_messages: int
    assistant_messages: int
    memory_count: int
    first_activity: Optional[str] = None
    last_activity: Optional[str] = None


class UsageModelBucket(BaseModel):
    provider: str
    model: str
    label: str
    message_count: int


class UsageDailyBucket(BaseModel):
    date: str
    user_messages: int
    assistant_messages: int
    total_messages: int
    model_requests: List[UsageModelBucket] = Field(default_factory=list)


class UsageAgentBucket(BaseModel):
    agent_name: str
    llm_model: Optional[str] = None
    message_count: int


class UsageStatsRead(BaseModel):
    days: Optional[int] = None
    summary: UsageSummary
    daily: List[UsageDailyBucket]
    by_model: List[UsageModelBucket]
    by_agent: List[UsageAgentBucket]


MemoryMapTreeNodeSpec.model_rebuild()
ConversationDetail.model_rebuild()
