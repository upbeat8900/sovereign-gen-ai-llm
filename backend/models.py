from typing import List, Optional, Union

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


class ConversationParticipantsUpdate(BaseModel):
    participants: List[ConversationParticipantUpdate] = Field(min_length=1, max_length=3)


class ConversationCreate(BaseModel):
    title: Optional[str] = None
    participants: Optional[List[ConversationParticipantUpdate]] = None


class ConversationTitleUpdate(BaseModel):
    title: str = Field(min_length=1)


class ConversationDelete(BaseModel):
    memory_action: str = "delete"
    target_conversation_id: Optional[int] = None


class ConversationReorder(BaseModel):
    conversation_ids: List[int] = Field(min_length=1)


class ConversationModelUpdate(BaseModel):
    llm_model_id: int


class Conversation(BaseModel):
    id: int
    title: str
    sort_order: int
    llm_model_id: Optional[int] = None
    participant_count: int = 0
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


class MessageCreate(BaseModel):
    content: str = ""
    image_data: Optional[str] = None
    image_media_type: Optional[str] = None
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
        if not content and not image_data:
            raise ValueError("Provide text, an image, or both")
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
    updated_at: str


class PromptConfigUpdate(BaseModel):
    default_prompt: str = Field(min_length=1)
    multi_agent_prompt: str = Field(min_length=1)
