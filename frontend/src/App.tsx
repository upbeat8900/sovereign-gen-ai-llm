import { ChangeEvent, ClipboardEvent, FormEvent, KeyboardEvent, MouseEvent as ReactMouseEvent, useEffect, useState, useMemo, useRef } from "react";
import "katex/dist/katex.min.css";
import rehypeKatex from "rehype-katex";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import {
  Download,
  ArrowDown,
  ArrowDownAZ,
  ArrowUp,
  ArrowUpDown,
  Bell,
  Bot,
  ChevronLeft,
  ChevronRight,
  Clock,
  Brain,
  Check,
  FileDown,
  GripVertical,
  ImagePlus,
  Loader2,
  MessageCircle,
  Mic,
  Pencil,
  Play,
  Plus,
  RotateCcw,
  Save,
  Settings,
  Sparkles,
  Square,
  Trash2,
  X,
} from "lucide-react";


type Page = "chat" | "memories" | "settings";
type SettingsTab = "models" | "agents" | "prompt" | "speech";
type ConversationSortMode = "custom" | "recent" | "alphabetical";
const MARKDOWN_REMARK_PLUGINS = [remarkGfm, remarkMath];
const MARKDOWN_REHYPE_PLUGINS = [rehypeKatex];

type Conversation = {
  id: number;
  title: string;
  sort_order: number;
  llm_model_id?: number | null;
  participant_count?: number;
  created_at: string;
  updated_at: string;
};

type ConversationParticipant = {
  id: number;
  conversation_id: number;
  llm_model_id: number;
  personality: string;
  name: string;
  sort_order: number;
  tts_voice_uri?: string | null;
  tts_speech_rate?: number | null;
  agent_profile_id?: number | null;
  llm_provider?: string | null;
  llm_model?: string | null;
  llm_comments?: string | null;
};

type AgentProfile = {
  id: number;
  name: string;
  personality: string;
  llm_model_id: number;
  tts_voice_uri?: string | null;
  tts_speech_rate?: number | null;
  llm_provider?: string | null;
  llm_model?: string | null;
  created_at: string;
  updated_at: string;
};

type ParticipantDraft = {
  llm_model_id: number;
  personality: string;
  name: string;
  tts_voice_uri?: string;
  tts_speech_rate?: number | null;
  agent_profile_id?: number | null;
};

type ConversationParticipantPayload = {
  llm_model_id: number;
  personality: string;
  name: string;
  tts_voice_uri: string | null;
  tts_speech_rate: number | null;
  agent_profile_id: number | null;
};

type Message = {
  id: number;
  conversation_id: number;
  role: "system" | "user" | "assistant";
  content: string;
  image_media_type?: string | null;
  image_data?: string | null;
  llm_provider?: string | null;
  llm_model?: string | null;
  generation_ms?: number | null;
  include_history?: boolean | number | null;
  participant_id?: number | null;
  created_at: string;
};

type PendingImage = {
  base64: string;
  mediaType: string;
  previewUrl: string;
  name?: string;
};

type Memory = {
  id: number;
  conversation_id: number;
  title: string;
  content: string;
  source_message_id?: number | null;
  llm_provider?: string | null;
  llm_model?: string | null;
  created_at: string;
  archived_at?: string | null;
  title_pending?: boolean;
};

type ConversationDetail = {
  conversation: Conversation;
  messages: Message[];
  memories: Memory[];
  participants: ConversationParticipant[];
};

type MemoryGroup = {
  conversation: Conversation;
  memories: Memory[];
};

type LlmModel = {
  id?: number;
  provider: string;
  base_url: string;
  model: string;
  comments?: string;
  has_api_key?: boolean;
  api_key_preview?: string | null;
  is_active: boolean;
  updated_at?: string;
  api_key?: string;
  clear_api_key?: boolean;
  generation_sample_count?: number;
  seconds_per_char?: number | null;
  avg_generation_sec?: number | null;
  reference_generation_estimate_sec?: number | null;
  tts_voice_uri?: string | null;
};

type LlmConfig = {
  models: LlmModel[];
  active_model: LlmModel;
};

type SpeechConfig = {
  whisper_model: string;
  whisper_model_options: string[];
  has_elevenlabs_api_key: boolean;
  elevenlabs_api_key_preview?: string | null;
  updated_at: string;
};

type PromptConfig = {
  default_prompt: string;
  default_prompt_baseline: string;
  multi_agent_prompt: string;
  multi_agent_prompt_baseline: string;
  updated_at: string;
};

type TtsVoiceOption = {
  uri: string;
  label: string;
};

type ElevenLabsVoice = {
  voice_id: string;
  name: string;
  gender?: string | null;
  age?: string | null;
  characteristics?: string | null;
  label: string;
};

type ElevenLabsVoiceFilters = {
  name: string;
  gender: string;
  age: string;
  characteristics: string;
};

const EMPTY_ELEVENLABS_VOICE_FILTERS: ElevenLabsVoiceFilters = {
  name: "",
  gender: "",
  age: "",
  characteristics: "",
};

type ElevenLabsVoiceSortKey = keyof ElevenLabsVoiceFilters;

type ElevenLabsVoiceSort = {
  key: ElevenLabsVoiceSortKey;
  direction: "asc" | "desc";
};

function sortElevenLabsVoices(
  voices: ElevenLabsVoice[],
  sort: ElevenLabsVoiceSort | null,
): ElevenLabsVoice[] {
  if (!sort) return voices;
  const factor = sort.direction === "asc" ? 1 : -1;
  return [...voices].sort((left, right) => {
    const leftValue = (left[sort.key] ?? "").trim().toLowerCase();
    const rightValue = (right[sort.key] ?? "").trim().toLowerCase();
    return leftValue.localeCompare(rightValue) * factor;
  });
}

function matchesElevenLabsVoiceFilter(value: string | null | undefined, filter: string): boolean {
  const query = filter.trim().toLowerCase();
  if (!query) return true;
  return (value ?? "").toLowerCase().includes(query);
}

function filterElevenLabsVoices(voices: ElevenLabsVoice[], filters: ElevenLabsVoiceFilters): ElevenLabsVoice[] {
  return voices.filter(
    (voice) =>
      matchesElevenLabsVoiceFilter(voice.name, filters.name) &&
      matchesElevenLabsVoiceFilter(voice.gender, filters.gender) &&
      matchesElevenLabsVoiceFilter(voice.age, filters.age) &&
      matchesElevenLabsVoiceFilter(voice.characteristics, filters.characteristics),
  );
}

type SelectedClip = {
  content: string;
  messageId: number;
  x: number;
  y: number;
};

type MessageContextMenu = {
  message: Message;
  x: number;
  y: number;
};

type MemoryExport = {
  conversation: Conversation;
  memories: Memory[];
  scope: "all" | "selected";
};

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";
const ELEVENLABS_VOICE_PREFIX = "elevenlabs:";
const ELEVENLABS_CATALOG_STORAGE_KEY = "elevenlabs_voice_catalog";
const TTS_VOICE_STORAGE_KEY = "tts_voice_uri";
const TTS_RATE_STORAGE_KEY = "tts_speech_rate";
const CURRENT_MESSAGE_ONLY_STORAGE_KEY = "chat_current_message_only";
const HISTORY_MESSAGE_LIMIT_STORAGE_KEY = "chat_history_message_limit";
const INCLUDE_MEMORIES_STORAGE_KEY = "chat_include_memories";
const INCLUDE_ALL_MEMORIES_STORAGE_KEY = "chat_include_all_memories";
const CONVERSATION_SORT_STORAGE_KEY = "chat_conversation_sort";
const DISCUSSION_ROUNDS_STORAGE_KEY = "chat_discussion_rounds";
const MULTI_AGENT_MODEL_OVERRIDE_KEY = "chat_multi_agent_model_override";
const ANSWER_LENGTH_STORAGE_KEY = "chat_answer_length";
const ANSWER_LENGTH_MIN = 1;
const ANSWER_LENGTH_MAX = 5;
const ANSWER_LENGTH_LABELS = ["1 sent.", "2–3 sent.", "3–4 sent.", "4–5 sent.", "Auto"];
const HISTORY_FULL_SENTINEL = -1;
const LLM_HISTORY_DB_CAP = 40;

function llmContextOverhead(
  memoryCount: number,
  includeMemories: boolean,
  allMemoryCount: number,
  includeAllMemories: boolean,
) {
  let overhead = 1;
  if (includeMemories && memoryCount > 0) {
    overhead += 1;
  }
  if (includeAllMemories && allMemoryCount > 0) {
    overhead += 1;
  }
  return overhead;
}

function historySliderMin(overhead: number) {
  return overhead + 1;
}

function historySliderMax(overhead: number, dbMessageCount: number) {
  const cappedDb = Math.min(dbMessageCount, LLM_HISTORY_DB_CAP);
  return overhead + cappedDb + 1;
}

function dbLimitFromHistorySliderTotal(total: number, overhead: number, dbMessageCount: number) {
  const cappedDb = Math.min(dbMessageCount, LLM_HISTORY_DB_CAP);
  return Math.max(0, Math.min(cappedDb, total - overhead - 1));
}

function historyLimitToIncludeHistory(sliderTotal: number, overhead: number, dbMessageCount: number): boolean | number {
  const dbLimit = dbLimitFromHistorySliderTotal(sliderTotal, overhead, dbMessageCount);
  if (dbLimit <= 0) {
    return false;
  }
  const cappedDb = Math.min(dbMessageCount, LLM_HISTORY_DB_CAP);
  if (cappedDb <= 0) {
    return false;
  }
  if (dbLimit >= cappedDb) {
    return true;
  }
  return dbLimit;
}

function readStoredHistoryMessageLimit(): number {
  const stored = localStorage.getItem(HISTORY_MESSAGE_LIMIT_STORAGE_KEY);
  if (stored !== null) {
    const parsed = Number(stored);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  if (localStorage.getItem(CURRENT_MESSAGE_ONLY_STORAGE_KEY) === "true") {
    return 0;
  }
  return HISTORY_FULL_SENTINEL;
}

function migrateStoredHistoryContextTotal(
  stored: number,
  overhead: number,
  min: number,
  max: number,
  dbMessageCount: number,
) {
  if (stored < 0) {
    return max;
  }
  if (stored >= min && stored <= max) {
    return stored;
  }
  const cappedDb = Math.min(dbMessageCount, LLM_HISTORY_DB_CAP);
  if (stored === 0) {
    return min;
  }
  if (stored <= cappedDb) {
    return Math.min(max, overhead + stored + 1);
  }
  return Math.min(max, Math.max(min, stored));
}

function persistHistoryMessageLimit(limit: number) {
  localStorage.setItem(HISTORY_MESSAGE_LIMIT_STORAGE_KEY, String(limit));
  localStorage.removeItem(CURRENT_MESSAGE_ONLY_STORAGE_KEY);
}

function readStoredIncludeMemories(): boolean {
  return localStorage.getItem(INCLUDE_MEMORIES_STORAGE_KEY) !== "false";
}

function persistIncludeMemories(value: boolean) {
  localStorage.setItem(INCLUDE_MEMORIES_STORAGE_KEY, String(value));
}

function readStoredIncludeAllMemories(): boolean {
  return localStorage.getItem(INCLUDE_ALL_MEMORIES_STORAGE_KEY) === "true";
}

function persistIncludeAllMemories(value: boolean) {
  localStorage.setItem(INCLUDE_ALL_MEMORIES_STORAGE_KEY, String(value));
}

function readStoredConversationSortMode(): ConversationSortMode {
  const stored = localStorage.getItem(CONVERSATION_SORT_STORAGE_KEY);
  if (stored === "custom" || stored === "recent" || stored === "alphabetical") {
    return stored;
  }
  return "custom";
}

function persistConversationSortMode(mode: ConversationSortMode) {
  localStorage.setItem(CONVERSATION_SORT_STORAGE_KEY, mode);
}

function sortConversations(conversations: Conversation[], mode: ConversationSortMode) {
  const list = [...conversations];
  if (mode === "recent") {
    return list.sort(
      (left, right) =>
        new Date(right.updated_at).getTime() - new Date(left.updated_at).getTime() || right.id - left.id,
    );
  }
  if (mode === "alphabetical") {
    return list.sort(
      (left, right) =>
        left.title.localeCompare(right.title, undefined, { sensitivity: "base" }) || left.id - right.id,
    );
  }
  return list.sort(
    (left, right) =>
      left.sort_order - right.sort_order ||
      new Date(right.updated_at).getTime() - new Date(left.updated_at).getTime() ||
      right.id - left.id,
  );
}

function clampHistoryContextTotal(next: number, min: number, max: number) {
  return Math.max(min, Math.min(max, next));
}

function storedHistoryLimitFromContextTotal(next: number, min: number, max: number) {
  const clamped = clampHistoryContextTotal(next, min, max);
  return clamped >= max ? HISTORY_FULL_SENTINEL : clamped;
}

function resolveHistoryContextTotal(stored: number, min: number, max: number) {
  if (stored < 0 || stored > max) {
    return max;
  }
  if (stored < min) {
    return min;
  }
  return stored;
}

function historyLimitTitle(total: number, min: number, max: number) {
  if (total <= min) {
    return "This message only";
  }
  if (total >= max) {
    return "Full conversation history";
  }
  return `${total} messages in request`;
}
const TTS_RATE_MIN = 0.5;
const TTS_RATE_MAX = 2.5;
const TTS_SAMPLE_TEXT = "Hello, this is a sample of how the assistant will sound when reading replies.";
const TRANSCRIBING_PLACEHOLDER = "Transcribing…";
const ALLOWED_IMAGE_TYPES = new Set(["image/jpeg", "image/png", "image/gif", "image/webp"]);
const MAX_IMAGE_BYTES = 10 * 1024 * 1024;
const CONVERSATION_PANE_WIDTH = 330;
const CONVERSATION_PANE_AUTO_COLLAPSE_MULTIPLIER = 3;
const WHISPER_MODEL_LABELS: Record<string, string> = {
  "tiny.en": "Tiny English — fastest",
  "base.en": "Base English — balanced",
  "small.en": "Small English — more accurate",
  "medium.en": "Medium English — high accuracy",
  "large-v3": "Large v3 — multilingual",
  tiny: "Tiny — multilingual",
  base: "Base — multilingual",
  small: "Small — multilingual",
  medium: "Medium — multilingual",
  "large-v3-turbo": "Large v3 Turbo — multilingual",
};
const PROVIDER_DEFAULTS = {
  ollama: {
    base_url: "http://host.docker.internal:11434",
    model: "qwen3.5:9b",
    addressLabel: "Ollama address",
    keyPlaceholder: "No key needed for local Ollama",
  },
  openai: {
    base_url: "https://api.openai.com/v1",
    model: "gpt-4o-mini",
    addressLabel: "OpenAI API base URL",
    keyPlaceholder: "Required for OpenAI API",
  },
  "openai-compatible": {
    base_url: "https://api.example.com/v1",
    model: "model-name",
    addressLabel: "API base URL",
    keyPlaceholder: "Optional, depending on the provider",
  },
} as const;

type ProviderKey = keyof typeof PROVIDER_DEFAULTS;

function buildFallbackMemoryTitle(content: string) {
  const firstLine = content.split("\n").map((line) => line.trim()).find(Boolean) ?? content;
  const text = firstLine.replace(/[#*_`>\[\]()]|https?:\/\/\S+/g, " ").replace(/\s+/g, " ").trim();
  const words = text.split(" ").slice(0, 9).join(" ").trim().replace(/[ .,:;-]+$/, "");
  if (!words) return "Untitled memory";
  return words.length > 72 ? `${words.slice(0, 72).trim()}...` : words;
}

function buildOptimisticMemory(message: Message, conversationId: number): Memory {
  const content = message.content.trim();
  return {
    id: -message.id,
    conversation_id: conversationId,
    title: buildFallbackMemoryTitle(content),
    content,
    source_message_id: message.id,
    llm_provider: message.llm_provider,
    llm_model: message.llm_model,
    created_at: new Date().toISOString(),
    title_pending: true,
  };
}

function sleep(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function providerDefaults(provider: string) {
  return PROVIDER_DEFAULTS[(provider as ProviderKey) in PROVIDER_DEFAULTS ? (provider as ProviderKey) : "ollama"];
}

function createAddModelDraft(isFirstModel: boolean): LlmModel {
  const defaults = providerDefaults("ollama");
  return {
    provider: "ollama",
    base_url: defaults.base_url,
    model: defaults.model,
    comments: "",
    tts_voice_uri: "",
    is_active: isFirstModel,
    api_key: "",
    clear_api_key: false,
  };
}

function maskApiKey(value: string): string {
  if (!value) return "";
  if (value.length <= 6) return "*".repeat(value.length);
  return `${value.slice(0, 3)}${"*".repeat(value.length - 6)}${value.slice(-3)}`;
}

type MaskedApiKeyInputProps = {
  value: string;
  preview?: string | null;
  hasApiKey?: boolean;
  placeholder?: string;
  onChange: (value: string) => void;
};

function MaskedApiKeyInput({ value, preview, hasApiKey, placeholder, onChange }: MaskedApiKeyInputProps) {
  const [focused, setFocused] = useState(false);
  const displayValue = focused
    ? value
    : value
      ? maskApiKey(value)
      : hasApiKey && preview
        ? preview
        : "";

  return (
    <input
      type="text"
      value={displayValue}
      placeholder={placeholder}
      autoComplete="off"
      spellCheck={false}
      onFocus={() => setFocused(true)}
      onBlur={() => setFocused(false)}
      onChange={(event) => onChange(event.target.value)}
    />
  );
}

function readBlobAsDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error ?? new Error("Could not read image"));
    reader.readAsDataURL(blob);
  });
}

function parseImageDataUrl(dataUrl: string) {
  const [header, base64 = ""] = dataUrl.split(",", 2);
  const mediaType = header.match(/^data:(.*?);base64$/)?.[1] ?? "image/png";
  return { base64, mediaType };
}

function messageImageSrc(message: Message) {
  if (!message.image_data || !message.image_media_type) return null;
  return `data:${message.image_media_type};base64,${message.image_data}`;
}

function parseSearchTerms(value: string) {
  return value
    .trim()
    .toLowerCase()
    .split(/\s+/)
    .filter(Boolean);
}

function escapeRegExp(value: string) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function highlightMatches(value: string, terms: string[]) {
  if (!terms.length) return value;
  const pattern = new RegExp(`(${terms.map(escapeRegExp).join("|")})`, "gi");
  return value.split(pattern).map((part, index) =>
    terms.includes(part.toLowerCase()) ? (
      <mark key={`${part}-${index}`} className="search-highlight">
        {part}
      </mark>
    ) : (
      part
    ),
  );
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers ?? {}),
    },
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail ?? `Request failed: ${response.status}`);
  }

  return response.json() as Promise<T>;
}

function isElevenLabsVoiceUri(uri?: string | null): boolean {
  return Boolean(uri?.startsWith(ELEVENLABS_VOICE_PREFIX));
}

function parseElevenLabsVoiceId(uri: string): string {
  return uri.slice(ELEVENLABS_VOICE_PREFIX.length);
}

function toElevenLabsVoiceUri(voiceId: string): string {
  return `${ELEVENLABS_VOICE_PREFIX}${voiceId}`;
}

function elevenlabsVoicesToOptions(voices: ElevenLabsVoice[]): TtsVoiceOption[] {
  return voices.map((voice) => ({
    uri: toElevenLabsVoiceUri(voice.voice_id),
    label: voice.label,
  }));
}

function mergeTtsVoiceOptions(browserOptions: TtsVoiceOption[], elevenLabsVoices: ElevenLabsVoice[]): TtsVoiceOption[] {
  return [...browserOptions, ...elevenlabsVoicesToOptions(elevenLabsVoices)];
}

function readStoredElevenlabsCatalog(): ElevenLabsVoice[] {
  try {
    const raw = localStorage.getItem(ELEVENLABS_CATALOG_STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (voice): voice is ElevenLabsVoice =>
        Boolean(voice) &&
        typeof voice === "object" &&
        typeof (voice as ElevenLabsVoice).voice_id === "string" &&
        typeof (voice as ElevenLabsVoice).name === "string",
    );
  } catch {
    return [];
  }
}

function persistElevenlabsCatalog(voices: ElevenLabsVoice[]) {
  localStorage.setItem(ELEVENLABS_CATALOG_STORAGE_KEY, JSON.stringify(voices));
}

function clearStoredElevenlabsCatalog() {
  localStorage.removeItem(ELEVENLABS_CATALOG_STORAGE_KEY);
}

async function requestElevenLabsAudio(text: string, voiceId: string, speechRate?: number): Promise<Blob> {
  const response = await fetch(`${API_BASE}/api/tts/elevenlabs/synthesize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text,
      voice_id: voiceId,
      speech_rate: speechRate,
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail ?? `Synthesis failed: ${response.status}`);
  }

  return response.blob();
}

function isAbortError(err: unknown): boolean {
  return err instanceof DOMException && err.name === "AbortError";
}

async function transcribeAudio(blob: Blob): Promise<string> {
  const formData = new FormData();
  formData.append("file", blob, "recording.webm");
  const response = await fetch(`${API_BASE}/api/stt/transcribe`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail ?? `Transcription failed: ${response.status}`);
  }

  const data = (await response.json()) as { text: string };
  return data.text;
}

function stripMarkdownForSpeech(value: string) {
  return value
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/`[^`]+`/g, " ")
    .replace(/!\[([^\]]*)\]\([^)]+\)/g, "$1")
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/[#*_>~|-]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function stripSpeakerPrefixForSpeech(value: string) {
  return value.replace(/^\[[^\]]+\]:\s*/i, "").trim();
}

function prepareAssistantTextForSpeech(value: string) {
  return stripSpeakerPrefixForSpeech(stripMarkdownForSpeech(value));
}

function getSelectedTextInMessage(messageId: number) {
  const selection = window.getSelection();
  if (!selection || selection.rangeCount === 0) return null;

  const content = selection.toString().trim();
  if (!content) return null;

  const range = selection.getRangeAt(0);
  const container = document.getElementById(`message-${messageId}`);
  if (!container || !container.contains(range.commonAncestorContainer)) {
    return null;
  }

  const rect = range.getBoundingClientRect();
  if (!rect.width && !rect.height) return null;

  return content;
}

function createOptimisticTimestamp() {
  return new Date().toISOString().slice(0, 19);
}

function persistedMessageCount(messages: Message[] | undefined) {
  return messages?.filter((message) => message.id >= 0).length ?? 0;
}

function sortTtsVoices(voices: SpeechSynthesisVoice[]) {
  return [...voices].sort((left, right) => {
    const leftEnglish = left.lang.toLowerCase().startsWith("en");
    const rightEnglish = right.lang.toLowerCase().startsWith("en");
    if (leftEnglish !== rightEnglish) {
      return leftEnglish ? -1 : 1;
    }
    if (left.default !== right.default) {
      return left.default ? -1 : 1;
    }
    return `${left.name} ${left.lang}`.localeCompare(`${right.name} ${right.lang}`);
  });
}

function buildTtsVoiceOptions(voices: SpeechSynthesisVoice[]): TtsVoiceOption[] {
  return sortTtsVoices(voices).map((voice) => ({
    uri: voice.voiceURI,
    label: `${voice.name} (${voice.lang})${voice.default ? " — default" : ""}`,
  }));
}

function resolveTtsVoice(voiceUri: string) {
  if (!window.speechSynthesis || !voiceUri) return null;
  return window.speechSynthesis.getVoices().find((voice) => voice.voiceURI === voiceUri) ?? null;
}

function readStoredTtsSpeechRate(): number {
  const raw = localStorage.getItem(TTS_RATE_STORAGE_KEY);
  if (raw === null || raw === "") return 1;
  const stored = Number(raw);
  if (!Number.isFinite(stored)) return 1;
  return Math.min(TTS_RATE_MAX, Math.max(TTS_RATE_MIN, stored));
}

function persistTtsSpeechRate(rate: number): number {
  const clamped = Math.min(TTS_RATE_MAX, Math.max(TTS_RATE_MIN, rate));
  localStorage.setItem(TTS_RATE_STORAGE_KEY, String(clamped));
  return clamped;
}

function clampTtsSpeechRate(rate: number): number {
  return Math.min(TTS_RATE_MAX, Math.max(TTS_RATE_MIN, rate));
}

function resolveAgentProfileForParticipant(
  participant: ConversationParticipant | ParticipantDraft,
  agentProfiles: AgentProfile[],
): AgentProfile | undefined {
  const profileId = participant.agent_profile_id;
  if (profileId == null) return undefined;
  return agentProfiles.find((profile) => profile.id === profileId);
}

function resolveTtsSpeechRateForParticipant(
  participant: ConversationParticipant | ParticipantDraft,
  agentProfiles: AgentProfile[] = [],
): number {
  if (participant.tts_speech_rate != null && Number.isFinite(participant.tts_speech_rate)) {
    return clampTtsSpeechRate(participant.tts_speech_rate);
  }
  const profile = resolveAgentProfileForParticipant(participant, agentProfiles);
  if (profile?.tts_speech_rate != null && Number.isFinite(profile.tts_speech_rate)) {
    return clampTtsSpeechRate(profile.tts_speech_rate);
  }
  return readStoredTtsSpeechRate();
}

function resolveTtsSpeechRateForMessage(
  message: Message,
  participants: ConversationParticipant[],
  agentProfiles: AgentProfile[] = [],
): number {
  if (message.participant_id != null) {
    const participant = participants.find((item) => item.id === message.participant_id);
    if (participant) {
      return resolveTtsSpeechRateForParticipant(participant, agentProfiles);
    }
  }
  return readStoredTtsSpeechRate();
}

function formatDate(value: string) {
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(`${value}Z`));
}

type LlmContextItem = {
  role: string;
  label: string;
  content_preview: string;
  has_image: boolean;
  image_bytes: number;
  char_estimate: number;
};

type LlmContextPreview = {
  provider: string;
  model: string;
  include_history: boolean | number;
  include_memories: boolean;
  include_all_memories: boolean;
  memory_count: number;
  all_memory_count: number;
  items: LlmContextItem[];
  total_chars: number;
  approx_tokens: number;
  image_count: number;
  history_message_count: number;
  images_resent_from_history: boolean;
  generation_estimate_sec?: number | null;
  seconds_per_char?: number | null;
  generation_sample_count?: number;
  multi_agent_note?: string | null;
};

function readStoredDiscussionRounds(): number {
  const stored = localStorage.getItem(DISCUSSION_ROUNDS_STORAGE_KEY);
  if (stored !== null) {
    const parsed = Number(stored);
    if (Number.isFinite(parsed) && parsed >= 1 && parsed <= 10) {
      return Math.round(parsed);
    }
  }
  return 1;
}

function readStoredAnswerLength(): number {
  const stored = localStorage.getItem(ANSWER_LENGTH_STORAGE_KEY);
  if (stored !== null) {
    const parsed = Number(stored);
    if (Number.isFinite(parsed) && parsed >= ANSWER_LENGTH_MIN && parsed <= ANSWER_LENGTH_MAX) {
      return Math.round(parsed);
    }
  }
  return 3;
}

function answerLengthLabel(level: number) {
  return ANSWER_LENGTH_LABELS[Math.max(0, Math.min(ANSWER_LENGTH_LABELS.length - 1, level - 1))];
}

function multiAgentModelOverrideStorageKey(conversationId: number) {
  return `${MULTI_AGENT_MODEL_OVERRIDE_KEY}_${conversationId}`;
}

function readStoredMultiAgentModelOverride(conversationId: number): number | null {
  const stored = localStorage.getItem(multiAgentModelOverrideStorageKey(conversationId));
  if (stored === null || stored === "") return null;
  const parsed = Number(stored);
  return Number.isFinite(parsed) && parsed > 0 ? Math.round(parsed) : null;
}

function assistantMessageHasVisibleContent(message: Message) {
  return Boolean(message.content?.trim());
}

function filterVisibleChatMessages(
  messages: Message[],
  options: {
    isMultiAgent: boolean;
    isGenerating: boolean;
    allowSinglePlaceholder: boolean;
  },
) {
  let keptSinglePlaceholder = false;
  return messages.filter((message) => {
    if (message.role !== "assistant") {
      return true;
    }
    if (assistantMessageHasVisibleContent(message)) {
      return true;
    }
    if (options.isMultiAgent) {
      return false;
    }
    if (!options.isGenerating || !options.allowSinglePlaceholder || message.id >= 0) {
      return false;
    }
    if (keptSinglePlaceholder) {
      return false;
    }
    keptSinglePlaceholder = true;
    return true;
  });
}

function createParticipantDraft(models: LlmModel[], index = 0): ParticipantDraft {
  const model = models[index]?.id != null ? models[index] : models[0];
  return {
    llm_model_id: model?.id ?? 0,
    personality: "",
    name: "",
    tts_voice_uri: "",
    agent_profile_id: null,
  };
}

function participantDraftFromProfile(profile: AgentProfile): ParticipantDraft {
  return {
    agent_profile_id: profile.id,
    name: profile.name,
    llm_model_id: profile.llm_model_id,
    personality: profile.personality,
    tts_voice_uri: profile.tts_voice_uri ?? "",
    tts_speech_rate: profile.tts_speech_rate ?? null,
  };
}

function participantDisplayName(participant: ConversationParticipant | ParticipantDraft, models: LlmModel[]) {
  if (participant.name?.trim()) {
    return participant.name.trim();
  }
  if ("llm_model" in participant && participant.llm_model) {
    const comments = participant.llm_comments?.trim();
    if (comments) return comments;
    return participant.llm_model;
  }
  const model = models.find((item) => item.id === participant.llm_model_id);
  if (model?.comments?.trim()) return model.comments.trim();
  return model?.model ?? "Agent";
}

function messageSpeakerLabel(message: Message, participants: ConversationParticipant[], models: LlmModel[]) {
  if (message.role !== "assistant") return "You";
  if (message.participant_id != null) {
    const participant = participants.find((item) => item.id === message.participant_id);
    if (participant) return participantDisplayName(participant, models);
  }
  if (message.llm_model) return message.llm_model;
  return "Assistant";
}

function assistantUsedCurrentMessageOnly(message: Message) {
  return message.include_history === false || message.include_history === 0;
}

function formatGenerationDuration(ms: number) {
  if (ms < 1000) {
    return `${Math.round(ms)} ms`;
  }
  if (ms < 60_000) {
    const seconds = ms / 1000;
    return seconds < 10 ? `${seconds.toFixed(1)} s` : `${Math.round(seconds)} s`;
  }
  const minutes = Math.floor(ms / 60_000);
  const seconds = Math.round((ms % 60_000) / 1000);
  return seconds > 0 ? `${minutes}m ${seconds}s` : `${minutes}m`;
}

function formatDurationSeconds(seconds: number) {
  if (seconds < 60) {
    return seconds < 10 ? `${seconds.toFixed(1)} s` : `${Math.round(seconds)} s`;
  }
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.round(seconds % 60);
  return remainder > 0 ? `${minutes}m ${remainder}s` : `${minutes}m`;
}

function generationProgressPercent(elapsedSec: number, estimateSec: number | null | undefined) {
  if (!estimateSec || estimateSec <= 0) {
    return null;
  }
  return Math.min(95, Math.max(2, (elapsedSec / estimateSec) * 100));
}

type GenerationJob = {
  conversationId: number;
  conversationTitle: string;
  modelName: string;
  contextPreview: LlmContextPreview | null;
  startedAt: number;
  isMultiAgent?: boolean;
  totalSteps?: number;
  completedSteps?: number;
  currentParticipantName?: string;
  initialMessageCount?: number;
  speakReplies?: boolean;
  autoPlayReplies?: boolean;
  participants?: ConversationParticipant[];
  rounds?: number;
  singleModelName?: string;
  overrideLlmModelId?: number | null;
};

type GenerationAgentStep = {
  key: string;
  name: string;
  model: string;
  round: number | null;
  status: "done" | "active" | "pending";
};

function buildGenerationAgentSteps(job: GenerationJob, models: LlmModel[]): GenerationAgentStep[] {
  const completed = job.completedSteps ?? 0;

  if (!job.isMultiAgent) {
    const name = job.singleModelName ?? job.modelName;
    return [
      {
        key: "single",
        name,
        model: name,
        round: null,
        status: completed >= 1 ? "done" : "active",
      },
    ];
  }

  const participants = job.participants ?? [];
  const rounds = job.rounds ?? 1;
  const overrideModel =
    job.overrideLlmModelId != null ? models.find((item) => item.id === job.overrideLlmModelId) : null;
  const steps: GenerationAgentStep[] = [];
  let stepIndex = 0;

  for (let round = 0; round < rounds; round += 1) {
    for (const participant of participants) {
      let status: GenerationAgentStep["status"] = "pending";
      if (stepIndex < completed) {
        status = "done";
      } else if (stepIndex === completed) {
        status = "active";
      }

      const model = models.find((item) => item.id === participant.llm_model_id);
      steps.push({
        key: `${participant.id}-round-${round}`,
        name: participantDisplayName(participant, models),
        model: overrideModel?.model ?? participant.llm_model ?? model?.model ?? "",
        round: rounds > 1 ? round + 1 : null,
        status,
      });
      stepIndex += 1;
    }
  }

  return steps;
}

type CompletedGenerationAlert = {
  conversationId: number;
  title: string;
  error?: string;
};

type AssistantSpeechItem = {
  conversationId: number;
  messageId: number;
  text: string;
  voiceUri: string;
  speechRate: number;
};

function assistantSpeechItemKey(item: AssistantSpeechItem): string {
  return `${item.messageId}:${item.voiceUri}:${item.speechRate}`;
}

function participantModelLabel(participant: ParticipantDraft, models: LlmModel[]) {
  const model = models.find((item) => item.id === participant.llm_model_id);
  return model ? `${model.model} (${model.provider})` : "No model selected";
}

function AgentProfileFields({
  draft,
  onChange,
  models,
  agentProfiles,
  ttsVoiceOptions,
  onPlayVoiceSample,
  expandedPersonality = false,
}: {
  draft: ParticipantDraft;
  onChange: (patch: Partial<ParticipantDraft>) => void;
  models: LlmModel[];
  agentProfiles: AgentProfile[];
  ttsVoiceOptions: TtsVoiceOption[];
  onPlayVoiceSample: (voiceUri?: string, speechRate?: number) => void;
  expandedPersonality?: boolean;
}) {
  return (
    <div className="agent-profile-form">
      <label>
        Name
        <input
          value={draft.name}
          placeholder="e.g. Optimist, Devil's advocate"
          onChange={(event) => onChange({ name: event.target.value })}
        />
      </label>
      <label>
        Model
        <select
          value={draft.llm_model_id}
          onChange={(event) => onChange({ llm_model_id: Number(event.target.value) })}
        >
          {models.map((model) => (
            <option key={model.id} value={model.id}>
              {model.model} ({model.provider})
            </option>
          ))}
        </select>
      </label>
      <div className="participant-voice-row">
        <label>
          Playback voice
          <select
            value={draft.tts_voice_uri ?? ""}
            onChange={(event) => onChange({ tts_voice_uri: event.target.value })}
            disabled={!ttsVoiceOptions.length}
          >
            <option value="">Default voice (model or Speech settings)</option>
            {ttsVoiceOptions.map((voice) => (
              <option key={voice.uri} value={voice.uri}>
                {voice.label}
              </option>
            ))}
          </select>
        </label>
        <button
          type="button"
          className="speech-sample-button"
          onClick={() =>
            onPlayVoiceSample(draft.tts_voice_uri, resolveTtsSpeechRateForParticipant(draft, agentProfiles))
          }
          disabled={!ttsVoiceOptions.length}
        >
          <Play size={16} />
          Sample
        </button>
      </div>
      <div className="participant-speech-rate-row">
        <label className="participant-speech-rate">
          <span>Speech speed</span>
          <span className="participant-speech-rate-controls">
            <input
              type="range"
              min={TTS_RATE_MIN}
              max={TTS_RATE_MAX}
              step={0.05}
              value={draft.tts_speech_rate ?? readStoredTtsSpeechRate()}
              disabled={draft.tts_speech_rate == null}
              onInput={(event) => onChange({ tts_speech_rate: Number(event.currentTarget.value) })}
            />
            <span className="participant-speech-rate-value">
              {draft.tts_speech_rate == null
                ? `Global (${readStoredTtsSpeechRate().toFixed(1)}×)`
                : `${draft.tts_speech_rate.toFixed(1)}×`}
            </span>
          </span>
        </label>
        <label className="participant-global-rate-toggle">
          <input
            type="checkbox"
            checked={draft.tts_speech_rate == null}
            onChange={(event) =>
              onChange({
                tts_speech_rate: event.target.checked ? null : readStoredTtsSpeechRate(),
              })
            }
          />
          Use global speed
        </label>
      </div>
      <label className={expandedPersonality ? "agent-personality-field" : undefined}>
        Personality / perspective
        <textarea
          className={expandedPersonality ? "agent-personality-textarea" : undefined}
          rows={expandedPersonality ? 16 : 4}
          value={draft.personality}
          placeholder={`Optional vision for ${participantDisplayName(draft, models)}`}
          onChange={(event) => onChange({ personality: event.target.value })}
        />
      </label>
    </div>
  );
}

function ParticipantEditor({
  participants,
  onChange,
  models,
  agentProfiles,
  ttsVoiceOptions,
  onSaveProfile,
  onUpdateProfile,
  onPlayVoiceSample,
  savingProfileIndex,
}: {
  participants: ParticipantDraft[];
  onChange: (next: ParticipantDraft[]) => void;
  models: LlmModel[];
  agentProfiles: AgentProfile[];
  ttsVoiceOptions: TtsVoiceOption[];
  onSaveProfile: (index: number) => void | Promise<void>;
  onUpdateProfile: (index: number) => void | Promise<void>;
  onPlayVoiceSample: (voiceUri?: string, speechRate?: number) => void;
  savingProfileIndex?: number | null;
}) {
  const [expandedIndex, setExpandedIndex] = useState<number | null>(0);

  useEffect(() => {
    setExpandedIndex((current) => {
      if (current == null) return current;
      return Math.min(current, Math.max(0, participants.length - 1));
    });
  }, [participants.length]);

  function updateParticipant(index: number, patch: Partial<ParticipantDraft>) {
    onChange(participants.map((participant, i) => (i === index ? { ...participant, ...patch } : participant)));
  }

  function applySavedAgent(index: number, profileId: number | null) {
    if (!profileId) {
      updateParticipant(index, { agent_profile_id: null });
      return;
    }
    const profile = agentProfiles.find((item) => item.id === profileId);
    if (profile) {
      onChange(participants.map((participant, i) => (i === index ? participantDraftFromProfile(profile) : participant)));
      setExpandedIndex(index);
    }
  }

  function addParticipant() {
    if (participants.length >= 3) return;
    onChange([...participants, createParticipantDraft(models, participants.length)]);
    setExpandedIndex(participants.length);
  }

  function removeParticipant(index: number) {
    if (participants.length <= 1) return;
    onChange(participants.filter((_, i) => i !== index));
    setExpandedIndex((current) => {
      if (current == null) return current;
      if (current === index) return Math.max(0, index - 1);
      if (current > index) return current - 1;
      return current;
    });
  }

  function toggleParticipant(index: number) {
    setExpandedIndex((current) => (current === index ? null : index));
  }

  return (
    <div className="participant-editor">
      <div className="participant-editor-scroll">
        {participants.map((participant, index) => {
          const isExpanded = expandedIndex === index;
          const displayName = participant.name.trim() || `Agent ${index + 1}`;
          const modelLabel = participantModelLabel(participant, models);
          const personalityPreview = participant.personality.trim();
          return (
            <section
              className={`participant-editor-row ${isExpanded ? "is-expanded" : "is-collapsed"}`}
              key={index}
            >
              <div className="participant-editor-row-header">
                <button
                  type="button"
                  className="participant-editor-toggle"
                  aria-expanded={isExpanded}
                  onClick={() => toggleParticipant(index)}
                >
                  <ChevronRight size={18} className="participant-editor-chevron" />
                  <span className="participant-editor-toggle-text">
                    <strong>{displayName}</strong>
                    <span className="participant-editor-summary">
                      {modelLabel}
                      {personalityPreview
                        ? ` · ${personalityPreview.slice(0, 48)}${personalityPreview.length > 48 ? "…" : ""}`
                        : ""}
                    </span>
                  </span>
                </button>
                {participants.length > 1 && (
                  <button
                    type="button"
                    className="secondary-button participant-remove"
                    onClick={() => removeParticipant(index)}
                  >
                    Remove
                  </button>
                )}
              </div>
              {isExpanded && (
                <div className="participant-editor-body">
                  <label>
                    Saved agent
                    <select
                      value={participant.agent_profile_id ?? ""}
                      onChange={(event) => {
                        const value = event.target.value;
                        applySavedAgent(index, value ? Number(value) : null);
                      }}
                    >
                      <option value="">Custom agent</option>
                      {agentProfiles.map((profile) => (
                        <option key={profile.id} value={profile.id}>
                          {profile.name}
                        </option>
                      ))}
                    </select>
                  </label>
                  <AgentProfileFields
                    draft={participant}
                    onChange={(patch) => updateParticipant(index, patch)}
                    models={models}
                    agentProfiles={agentProfiles}
                    ttsVoiceOptions={ttsVoiceOptions}
                    onPlayVoiceSample={onPlayVoiceSample}
                  />
                  <div className="participant-library-actions">
                    {participant.agent_profile_id ? (
                      <button
                        type="button"
                        className="secondary-button"
                        disabled={savingProfileIndex === index || !participant.name.trim()}
                        onClick={() => void onUpdateProfile(index)}
                      >
                        {savingProfileIndex === index ? (
                          <>
                            <Loader2 size={16} className="spin" />
                            Updating...
                          </>
                        ) : (
                          <>
                            <Save size={16} />
                            Update library
                          </>
                        )}
                      </button>
                    ) : (
                      <button
                        type="button"
                        className="secondary-button"
                        disabled={savingProfileIndex === index || !participant.name.trim()}
                        onClick={() => void onSaveProfile(index)}
                      >
                        {savingProfileIndex === index ? (
                          <>
                            <Loader2 size={16} className="spin" />
                            Saving...
                          </>
                        ) : (
                          <>
                            <Save size={16} />
                            Save to library
                          </>
                        )}
                      </button>
                    )}
                  </div>
                </div>
              )}
            </section>
          );
        })}
      </div>
      {participants.length < 3 && (
        <button type="button" className="secondary-button participant-add" onClick={addParticipant}>
          <Plus size={16} />
          Add agent
        </button>
      )}
    </div>
  );
}

export default function App() {
  const [page, setPage] = useState<Page>("chat");
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<number | null>(null);
  const [draggedConversationId, setDraggedConversationId] = useState<number | null>(null);
  const [dragOverConversationId, setDragOverConversationId] = useState<number | null>(null);
  const [editingConversationId, setEditingConversationId] = useState<number | null>(null);
  const [editingConversationTitle, setEditingConversationTitle] = useState("");
  const [detail, setDetail] = useState<ConversationDetail | null>(null);
  const [isConversationModelModalOpen, setIsConversationModelModalOpen] = useState(false);
  const [selectingModelId, setSelectingModelId] = useState<number | null>(null);
  const [input, setInput] = useState("");
  const [pendingImage, setPendingImage] = useState<PendingImage | null>(null);
  const [newTitle, setNewTitle] = useState("");
  const [isNewConversationModalOpen, setIsNewConversationModalOpen] = useState(false);
  const [newConversationMode, setNewConversationMode] = useState<"single" | "multi">("single");
  const [newConversationParticipants, setNewConversationParticipants] = useState<ParticipantDraft[]>([]);
  const [isParticipantsModalOpen, setIsParticipantsModalOpen] = useState(false);
  const [participantsDraft, setParticipantsDraft] = useState<ParticipantDraft[]>([]);
  const [savingParticipants, setSavingParticipants] = useState(false);
  const [agentProfiles, setAgentProfiles] = useState<AgentProfile[]>([]);
  const [savingAgentProfileIndex, setSavingAgentProfileIndex] = useState<number | null>(null);
  const [editingAgentProfile, setEditingAgentProfile] = useState<AgentProfile | null>(null);
  const [isCreatingAgentProfile, setIsCreatingAgentProfile] = useState(false);
  const [agentProfileEditDraft, setAgentProfileEditDraft] = useState<ParticipantDraft | null>(null);
  const [savingAgentProfileEdit, setSavingAgentProfileEdit] = useState(false);
  const [participantEditorTarget, setParticipantEditorTarget] = useState<"new" | "edit">("new");
  const [discussionRounds, setDiscussionRounds] = useState(readStoredDiscussionRounds);
  const [multiAgentModelOverrideId, setMultiAgentModelOverrideId] = useState<number | null>(null);
  const [answerLength, setAnswerLength] = useState(readStoredAnswerLength);
  const [deleteConversationTarget, setDeleteConversationTarget] = useState<Conversation | null>(null);
  const [deleteMessageCount, setDeleteMessageCount] = useState(0);
  const [deleteMemoryCount, setDeleteMemoryCount] = useState(0);
  const [deleteMemoryAction, setDeleteMemoryAction] = useState<"delete" | "move">("delete");
  const [deleteTargetId, setDeleteTargetId] = useState<number | "">("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [settingsSaveSuccessMessage, setSettingsSaveSuccessMessage] = useState<string | null>(null);
  const [generationJobs, setGenerationJobs] = useState<Record<number, GenerationJob>>({});
  const [progressModalConversationId, setProgressModalConversationId] = useState<number | null>(null);
  const [completedGenerationAlert, setCompletedGenerationAlert] = useState<CompletedGenerationAlert | null>(null);
  const [unreviewedReplyConversationIds, setUnreviewedReplyConversationIds] = useState<number[]>([]);
  const [generationClock, setGenerationClock] = useState(() => Date.now());
  const [historyMessageLimit, setHistoryMessageLimit] = useState(readStoredHistoryMessageLimit);
  const [includeMemories, setIncludeMemories] = useState(readStoredIncludeMemories);
  const [includeAllMemories, setIncludeAllMemories] = useState(readStoredIncludeAllMemories);
  const [memorySearch, setMemorySearch] = useState("");
  const [memorySort, setMemorySort] = useState<"created_at" | "title" | "llm_model">("created_at");
  const [memoryModelFilter, setMemoryModelFilter] = useState("");
  const [memoryOrder, setMemoryOrder] = useState<"asc" | "desc">("desc");
  const [selectedMemoryIds, setSelectedMemoryIds] = useState<number[]>([]);
  const [expandedMemoryIds, setExpandedMemoryIds] = useState<number[]>([]);
  const [memoryGroups, setMemoryGroups] = useState<MemoryGroup[]>([]);
  const [expandedMemoryGroupIds, setExpandedMemoryGroupIds] = useState<number[]>([]);
  const [draggedMemoryId, setDraggedMemoryId] = useState<number | null>(null);
  const [dragOverMemoryGroupId, setDragOverMemoryGroupId] = useState<number | null>(null);
  const [mergeTargetConversationId, setMergeTargetConversationId] = useState<number | "">("");
  const [printMemoryExport, setPrintMemoryExport] = useState<MemoryExport | null>(null);
  const [mergeText, setMergeText] = useState("");
  const [config, setConfig] = useState<LlmConfig | null>(null);
  const [configDraft, setConfigDraft] = useState<LlmModel[]>([]);
  const [settingsTab, setSettingsTab] = useState<SettingsTab>("models");
  const [isAddModelModalOpen, setIsAddModelModalOpen] = useState(false);
  const [addModelDraft, setAddModelDraft] = useState<LlmModel>(() => createAddModelDraft(true));
  const [resettingTimingModelId, setResettingTimingModelId] = useState<number | null>(null);
  const [speechConfig, setSpeechConfig] = useState<SpeechConfig | null>(null);
  const [whisperModelDraft, setWhisperModelDraft] = useState("base.en");
  const [elevenlabsApiKeyDraft, setElevenlabsApiKeyDraft] = useState("");
  const [clearElevenlabsApiKeyDraft, setClearElevenlabsApiKeyDraft] = useState(false);
  const [elevenlabsCatalogVoices, setElevenlabsCatalogVoices] = useState<ElevenLabsVoice[]>(readStoredElevenlabsCatalog);
  const [elevenlabsVoiceFilters, setElevenlabsVoiceFilters] = useState<ElevenLabsVoiceFilters>(
    EMPTY_ELEVENLABS_VOICE_FILTERS,
  );
  const [elevenlabsVoiceSort, setElevenlabsVoiceSort] = useState<ElevenLabsVoiceSort | null>(null);
  const [isImportingElevenlabsVoices, setIsImportingElevenlabsVoices] = useState(false);
  const [promptConfig, setPromptConfig] = useState<PromptConfig | null>(null);
  const [defaultPromptDraft, setDefaultPromptDraft] = useState("");
  const [defaultMultiAgentPromptDraft, setDefaultMultiAgentPromptDraft] = useState("");
  const [ttsVoiceOptions, setTtsVoiceOptions] = useState<TtsVoiceOption[]>([]);
  const [selectedTtsVoiceUri, setSelectedTtsVoiceUri] = useState(
    () => localStorage.getItem(TTS_VOICE_STORAGE_KEY) ?? "",
  );
  const [ttsSpeechRate, setTtsSpeechRate] = useState(readStoredTtsSpeechRate);
  const [selectedClip, setSelectedClip] = useState<SelectedClip | null>(null);
  const [messageContextMenu, setMessageContextMenu] = useState<MessageContextMenu | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [speakingMessageId, setSpeakingMessageId] = useState<number | null>(null);
  const [conversationPaneCollapsed, setConversationPaneCollapsed] = useState(false);
  const [conversationSortMode, setConversationSortMode] = useState(readStoredConversationSortMode);
  const skipConversationTitleSaveRef = useRef(false);
  const isRecordingRef = useRef(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const mediaChunksRef = useRef<Blob[]>([]);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const speakingMessageIdRef = useRef<number | null>(null);
  const speakingPlaybackRef = useRef<{
    id: number | "sample";
    content: string;
    voiceUri?: string;
    speechRate?: number;
  } | null>(null);
  const audioPlaybackRef = useRef<HTMLAudioElement | null>(null);
  const audioBlobUrlRef = useRef<string | null>(null);
  const playbackSessionRef = useRef(0);
  const assistantSpeechQueueRef = useRef<AssistantSpeechItem[]>([]);
  const assistantSpeechProcessingRef = useRef(false);
  const ttsPrefetchRef = useRef<{ key: string; promise: Promise<Blob> } | null>(null);
  const elevenlabsCatalogHydratedRef = useRef(false);
  const spokenMessageIdsByConversationRef = useRef<Record<number, Set<number>>>({});
  const speakRepliesByConversationRef = useRef<Record<number, boolean>>({});
  const speakReplyStartIndexByConversationRef = useRef<Record<number, number>>({});
  const generationAbortByConversationRef = useRef(new Map<number, AbortController>());
  const activeIdRef = useRef<number | null>(null);
  const generationJobsRef = useRef<Record<number, GenerationJob>>({});
  const dismissedProgressModalOnPlaybackRef = useRef<Set<number>>(new Set());
  const editingConversationIdRef = useRef<number | null>(null);
  const speechSelectionRef = useRef<{ messageId: number; text: string } | null>(null);
  const imageInputRef = useRef<HTMLInputElement | null>(null);
  const pendingLatestScrollRef = useRef(false);
  const expandConversationMemoriesRef = useRef<number | null>(null);

  const allMemories = useMemo(() => memoryGroups.flatMap((group) => group.memories), [memoryGroups]);
  const displayedElevenlabsCatalogVoices = useMemo(() => {
    const filtered = filterElevenLabsVoices(elevenlabsCatalogVoices, elevenlabsVoiceFilters);
    return sortElevenLabsVoices(filtered, elevenlabsVoiceSort);
  }, [elevenlabsCatalogVoices, elevenlabsVoiceFilters, elevenlabsVoiceSort]);
  const hasActiveElevenlabsVoiceFilters = useMemo(
    () => Object.values(elevenlabsVoiceFilters).some((value) => value.trim()),
    [elevenlabsVoiceFilters],
  );
  const otherConversationMemories = useMemo(
    () => allMemories.filter((memory) => memory.conversation_id !== activeId),
    [allMemories, activeId],
  );
  const selectedMemories = useMemo(
    () => allMemories.filter((memory) => selectedMemoryIds.includes(memory.id)),
    [allMemories, selectedMemoryIds],
  );
  const oldestSelectedConversation = useMemo(() => {
    const selectedConversationIds = new Set(selectedMemories.map((memory) => memory.conversation_id));
    return conversations
      .filter((conversation) => selectedConversationIds.has(conversation.id))
      .sort((left, right) => new Date(left.created_at).getTime() - new Date(right.created_at).getTime() || left.id - right.id)[0];
  }, [conversations, selectedMemories]);
  const memorySearchTerms = useMemo(() => parseSearchTerms(memorySearch), [memorySearch]);
  const sortMemoryList = (memories: Memory[]) =>
    [...memories].sort((left, right) => {
        const direction = memoryOrder === "asc" ? 1 : -1;
        if (memorySort === "title") {
          return left.title.localeCompare(right.title) * direction;
        }
        if (memorySort === "llm_model") {
          return (left.llm_model ?? "").localeCompare(right.llm_model ?? "") * direction;
        }
        return (new Date(left.created_at).getTime() - new Date(right.created_at).getTime()) * direction;
      });
  const memoryModelOptions = useMemo(
    () =>
      Array.from(new Set(allMemories.map((memory) => memory.llm_model).filter((model): model is string => Boolean(model)))).sort(),
    [allMemories],
  );
  const visibleMemoryGroups = useMemo(
    () =>
      memoryGroups
        .map((group) => {
          const filteredByModel = memoryModelFilter
            ? group.memories.filter((memory) => memory.llm_model === memoryModelFilter)
            : group.memories;
          const sortedGroupMemories = sortMemoryList(filteredByModel);
          if (!memorySearchTerms.length) {
            return { ...group, memories: sortedGroupMemories };
          }

          const visibleIndexes = new Set<number>();
          sortedGroupMemories.forEach((memory, index) => {
            const text = `${memory.title} ${memory.content} ${memory.llm_model ?? ""}`.toLowerCase();
            if (memorySearchTerms.every((term) => text.includes(term))) {
              visibleIndexes.add(index);
              if (index > 0) visibleIndexes.add(index - 1);
              if (index < sortedGroupMemories.length - 1) visibleIndexes.add(index + 1);
            }
          });

          return {
            ...group,
            memories: sortedGroupMemories.filter((_, index) => visibleIndexes.has(index)),
          };
        })
        .filter((group) => !memorySearchTerms.length || group.memories.length > 0),
    [memoryGroups, memoryModelFilter, memoryOrder, memorySearchTerms, memorySort],
  );
  const rememberedMessageIds = useMemo(
    () => new Set(detail?.memories.map((memory) => memory.source_message_id).filter((id): id is number => typeof id === "number") ?? []),
    [detail?.memories],
  );
  const userMessageIds = useMemo(
    () => detail?.messages.filter((message) => message.role === "user").map((message) => message.id) ?? [],
    [detail?.messages],
  );
  const latestConversation = useMemo(() => {
    if (!conversations.length) return null;
    return conversations.reduce((latest, conversation) =>
      conversation.updated_at > latest.updated_at ? conversation : latest,
    );
  }, [conversations]);
  const displayedConversations = useMemo(
    () => sortConversations(conversations, conversationSortMode),
    [conversations, conversationSortMode],
  );
  const activeConversation = useMemo(() => {
    if (activeId == null) return null;
    const fromList = conversations.find((conversation) => conversation.id === activeId);
    if (fromList) return fromList;
    if (detail?.conversation.id === activeId) return detail.conversation;
    return null;
  }, [activeId, conversations, detail?.conversation]);
  const deleteDestinationConversations = useMemo(
    () => conversations.filter((conversation) => conversation.id !== deleteConversationTarget?.id),
    [conversations, deleteConversationTarget?.id],
  );
  const conversationModel = useMemo(() => {
    const modelId = detail?.conversation.llm_model_id;
    return config?.models.find((model) => model.id === modelId) ?? config?.active_model ?? null;
  }, [config, detail?.conversation.llm_model_id]);
  const activeModelName = conversationModel?.model ?? config?.active_model.model ?? "qwen3.5:9b";
  const activeParticipants = detail?.participants ?? [];
  const isMultiAgentConversation = activeParticipants.length > 0;
  const multiAgentOverrideModel = useMemo(() => {
    if (multiAgentModelOverrideId == null) return null;
    return config?.models.find((model) => model.id === multiAgentModelOverrideId) ?? null;
  }, [config?.models, multiAgentModelOverrideId]);
  const activeConversationGenerating = activeId != null && activeId in generationJobs;
  const activeGenerationJob = activeId != null ? generationJobs[activeId] : undefined;
  const progressModalJob =
    progressModalConversationId != null ? generationJobs[progressModalConversationId] ?? null : null;
  const progressModalElapsedSec = progressModalJob
    ? Math.floor((generationClock - progressModalJob.startedAt) / 1000)
    : 0;
  const progressModalAgentSteps =
    progressModalJob && config ? buildGenerationAgentSteps(progressModalJob, config.models) : [];
  const chatMessages = useMemo(() => {
    const base = detail?.messages ?? [];
    const job = activeId != null ? generationJobs[activeId] : undefined;
    let messages = base;

    if (job && activeId && !job.isMultiAgent) {
      const last = messages[messages.length - 1];
      const hasPendingPlaceholder =
        last?.role === "assistant" && !last.content && last.id < 0;
      if (last?.role === "user" && !hasPendingPlaceholder) {
        messages = [
          ...base,
          {
            id: -job.startedAt,
            conversation_id: activeId,
            role: "assistant" as const,
            content: "",
            llm_model: job.modelName,
            created_at: new Date().toISOString(),
          },
        ];
      }
    }

    return filterVisibleChatMessages(messages, {
      isMultiAgent: isMultiAgentConversation,
      isGenerating: Boolean(job),
      allowSinglePlaceholder: Boolean(job && !job.isMultiAgent),
    });
  }, [detail?.messages, activeId, generationJobs, isMultiAgentConversation]);
  const historyContextOverhead = llmContextOverhead(
    detail?.memories.length ?? 0,
    includeMemories,
    otherConversationMemories.length,
    includeAllMemories,
  );
  const historyDbMessageCount = detail?.messages.length ?? 0;
  const historySliderMinValue = historySliderMin(historyContextOverhead);
  const historySliderMaxValue = historySliderMax(historyContextOverhead, historyDbMessageCount);
  const composerHistoryContextTotal = resolveHistoryContextTotal(
    historyMessageLimit,
    historySliderMinValue,
    historySliderMaxValue,
  );

  useEffect(() => {
    setHistoryMessageLimit((current) => {
      const migrated = migrateStoredHistoryContextTotal(
        current,
        historyContextOverhead,
        historySliderMinValue,
        historySliderMaxValue,
        historyDbMessageCount,
      );
      const resolved = resolveHistoryContextTotal(migrated, historySliderMinValue, historySliderMaxValue);
      const stored = storedHistoryLimitFromContextTotal(resolved, historySliderMinValue, historySliderMaxValue);
      return current === stored ? current : stored;
    });
  }, [activeId, historyContextOverhead, historyDbMessageCount, historySliderMinValue, historySliderMaxValue]);

  const historyControlsDisabled = !activeId || activeConversationGenerating || isTranscribing;

  function updateHistoryMessageLimit(next: number) {
    const stored = storedHistoryLimitFromContextTotal(next, historySliderMinValue, historySliderMaxValue);
    setHistoryMessageLimit(stored);
    persistHistoryMessageLimit(stored);
  }

  function updateIncludeMemories(next: boolean) {
    setIncludeMemories(next);
    persistIncludeMemories(next);
  }

  function updateIncludeAllMemories(next: boolean) {
    setIncludeAllMemories(next);
    persistIncludeAllMemories(next);
  }

  function updateAnswerLength(next: number) {
    const clamped = Math.max(ANSWER_LENGTH_MIN, Math.min(ANSWER_LENGTH_MAX, Math.round(next)));
    setAnswerLength(clamped);
    localStorage.setItem(ANSWER_LENGTH_STORAGE_KEY, String(clamped));
  }

  function updateMultiAgentModelOverride(modelId: number | null) {
    if (activeId == null) {
      setMultiAgentModelOverrideId(modelId);
      return;
    }
    setMultiAgentModelOverrideId(modelId);
    const key = multiAgentModelOverrideStorageKey(activeId);
    if (modelId == null) {
      localStorage.removeItem(key);
      return;
    }
    localStorage.setItem(key, String(modelId));
  }

  useEffect(() => {
    if (activeId == null || !isMultiAgentConversation) {
      setMultiAgentModelOverrideId(null);
      return;
    }
    const stored = readStoredMultiAgentModelOverride(activeId);
    if (stored != null && config?.models.some((model) => model.id === stored)) {
      setMultiAgentModelOverrideId(stored);
      return;
    }
    if (stored != null) {
      localStorage.removeItem(multiAgentModelOverrideStorageKey(activeId));
    }
    setMultiAgentModelOverrideId(null);
  }, [activeId, isMultiAgentConversation, config?.models]);

  useEffect(() => {
    loadConversations();
    loadMemoryGroups();
    loadConfig();
    void loadAgentProfiles();
    void loadSpeechConfig({ autoImportVoices: true });
    setTtsSpeechRate(readStoredTtsSpeechRate());
  }, []);

  useEffect(() => {
    const collapseThreshold = CONVERSATION_PANE_WIDTH * CONVERSATION_PANE_AUTO_COLLAPSE_MULTIPLIER;

    function syncConversationPaneCollapse() {
      if (window.innerWidth < collapseThreshold) {
        setConversationPaneCollapsed(true);
      }
    }

    syncConversationPaneCollapse();
    window.addEventListener("resize", syncConversationPaneCollapse);
    return () => window.removeEventListener("resize", syncConversationPaneCollapse);
  }, []);

  function toggleConversationPane() {
    setConversationPaneCollapsed((current) => !current);
  }

  useEffect(() => {
    function handleSpeechRateStorage(event: StorageEvent) {
      if (event.key === TTS_RATE_STORAGE_KEY) {
        setTtsSpeechRate(readStoredTtsSpeechRate());
      }
    }

    window.addEventListener("storage", handleSpeechRateStorage);
    return () => window.removeEventListener("storage", handleSpeechRateStorage);
  }, []);

  useEffect(() => {
    activeIdRef.current = activeId;
  }, [activeId]);

  useEffect(() => {
    generationJobsRef.current = generationJobs;
  }, [generationJobs]);

  useEffect(() => {
    editingConversationIdRef.current = editingConversationId;
  }, [editingConversationId]);

  useEffect(() => {
    if (!Object.keys(generationJobs).length) return;
    const intervalId = window.setInterval(() => setGenerationClock(Date.now()), 1000);
    return () => window.clearInterval(intervalId);
  }, [generationJobs]);

  useEffect(() => {
    if (!Object.values(generationJobs).some((job) => job.isMultiAgent)) return;
    const pollGeneratingConversations = () => {
      Object.values(generationJobsRef.current)
        .filter((job) => job.isMultiAgent)
        .forEach((job) => {
          void loadConversation(job.conversationId);
        });
    };
    pollGeneratingConversations();
    const intervalId = window.setInterval(pollGeneratingConversations, 800);
    return () => window.clearInterval(intervalId);
  }, [generationJobs]);

  useEffect(() => {
    if (
      isNewConversationModalOpen &&
      newConversationMode === "multi" &&
      config?.models.length &&
      !newConversationParticipants.length
    ) {
      setNewConversationParticipants([createParticipantDraft(config.models)]);
    }
  }, [isNewConversationModalOpen, newConversationMode, config, newConversationParticipants.length]);

  useEffect(() => {
    if (page === "settings") {
      void loadSpeechConfig();
      void loadPromptConfig();
    }
  }, [page]);

  useEffect(() => {
    let cancelled = false;

    function syncVoices() {
      const browserOptions = window.speechSynthesis
        ? buildTtsVoiceOptions(window.speechSynthesis.getVoices())
        : [];

      if (cancelled) return;

      const options = mergeTtsVoiceOptions(browserOptions, elevenlabsCatalogVoices);
      setTtsVoiceOptions(options);
      setSelectedTtsVoiceUri((current) => {
        if (current && options.some((option) => option.uri === current)) {
          return current;
        }
        const stored = localStorage.getItem(TTS_VOICE_STORAGE_KEY);
        if (stored && options.some((option) => option.uri === stored)) {
          return stored;
        }
        const preferred = options[0]?.uri ?? "";
        if (preferred) {
          localStorage.setItem(TTS_VOICE_STORAGE_KEY, preferred);
        }
        return preferred;
      });
    }

    function handleVoicesChanged() {
      syncVoices();
    }

    syncVoices();
    const synth = window.speechSynthesis;
    if (synth) {
      synth.onvoiceschanged = handleVoicesChanged;
    }
    return () => {
      cancelled = true;
      if (synth) {
        synth.onvoiceschanged = null;
      }
    };
  }, [elevenlabsCatalogVoices]);

  useEffect(() => {
    stopSpeechPlayback();
    setEditingConversationId(null);
    setEditingConversationTitle("");
    if (activeId) {
      setUnreviewedReplyConversationIds((current) =>
        current.includes(activeId) ? current.filter((id) => id !== activeId) : current,
      );
      loadConversation(activeId);
      setIsConversationModelModalOpen(false);
      setExpandedMemoryIds([]);
      setExpandedMemoryGroupIds((current) => (current.includes(activeId) ? current : [...current, activeId]));
      setMergeTargetConversationId((current) => current || activeId);
    } else {
      setDetail(null);
    }
  }, [activeId]);

  useEffect(() => {
    if (page === "memories") {
      loadMemoryGroups();
    }
  }, [page, memorySort, memoryOrder, activeId]);

  useEffect(() => {
    if (page !== "chat" || !userMessageIds.length || pendingLatestScrollRef.current) return;
    const latestUserMessageId = userMessageIds[userMessageIds.length - 1];
    requestAnimationFrame(() => {
      document.getElementById(`message-${latestUserMessageId}`)?.scrollIntoView({
        block: "start",
        behavior: "auto",
      });
    });
  }, [page, userMessageIds]);

  useEffect(() => {
    if (!pendingLatestScrollRef.current || page !== "chat" || activeId !== latestConversation?.id) return;
    const assistantMessages = detail?.messages.filter((message) => message.role === "assistant") ?? [];
    const latestAssistantMessage = assistantMessages[assistantMessages.length - 1];
    requestAnimationFrame(() => {
      if (latestAssistantMessage) {
        document.getElementById(`message-${latestAssistantMessage.id}`)?.scrollIntoView({
          block: "end",
          behavior: "smooth",
        });
      } else {
        const messagesEl = document.querySelector(".messages");
        if (messagesEl) {
          messagesEl.scrollTop = messagesEl.scrollHeight;
        }
      }
      pendingLatestScrollRef.current = false;
    });
  }, [page, activeId, latestConversation?.id, detail?.messages]);

  useEffect(() => {
    const handleAfterPrint = () => setPrintMemoryExport(null);
    window.addEventListener("afterprint", handleAfterPrint);
    return () => window.removeEventListener("afterprint", handleAfterPrint);
  }, []);

  useEffect(() => {
    function handleDocumentMouseDown(event: MouseEvent) {
      const target = event.target as HTMLElement | null;
      if (target?.closest(".message-context-menu")) {
        return;
      }
      setMessageContextMenu(null);
      if (
        target?.closest(".selection-memory-popover") ||
        target?.closest(".message-content") ||
        target?.closest(".message-meta-actions")
      ) {
        return;
      }
      setSelectedClip(null);
    }

    document.addEventListener("mousedown", handleDocumentMouseDown);
    return () => document.removeEventListener("mousedown", handleDocumentMouseDown);
  }, []);

  useEffect(() => {
    function handleDocumentKeyDown(event: globalThis.KeyboardEvent) {
      if (event.key === "Escape") {
        setMessageContextMenu(null);
      }
    }

    document.addEventListener("keydown", handleDocumentKeyDown);
    return () => document.removeEventListener("keydown", handleDocumentKeyDown);
  }, []);

  useEffect(() => {
    return () => {
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
        mediaRecorderRef.current.stop();
      }
      mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
      window.speechSynthesis?.cancel();
      if (audioPlaybackRef.current) {
        audioPlaybackRef.current.pause();
        audioPlaybackRef.current = null;
      }
      if (audioBlobUrlRef.current) {
        URL.revokeObjectURL(audioBlobUrlRef.current);
        audioBlobUrlRef.current = null;
      }
      speakingMessageIdRef.current = null;
      speakingPlaybackRef.current = null;
    };
  }, []);

  function revokeAudioBlobUrl() {
    if (audioBlobUrlRef.current) {
      URL.revokeObjectURL(audioBlobUrlRef.current);
      audioBlobUrlRef.current = null;
    }
  }

  function stopAudioPlayback() {
    const audio = audioPlaybackRef.current;
    if (audio) {
      audio.pause();
      audio.onended = null;
      audio.onerror = null;
      audio.removeAttribute("src");
      audio.load();
      audioPlaybackRef.current = null;
    }
    revokeAudioBlobUrl();
  }

  function playElevenLabsBlob(
    blob: Blob,
    session: number,
    playbackId: number | "sample",
    onPlaybackStart?: () => void,
  ): Promise<void> {
    return new Promise((resolve) => {
      revokeAudioBlobUrl();
      const url = URL.createObjectURL(blob);
      audioBlobUrlRef.current = url;
      const audio = new Audio(url);
      audioPlaybackRef.current = audio;

      const finishPlayback = () => {
        if (playbackSessionRef.current !== session) {
          resolve();
          return;
        }
        stopAudioPlayback();
        speakingPlaybackRef.current = null;
        if (typeof playbackId === "number") {
          speakingMessageIdRef.current = null;
          setSpeakingMessageId(null);
        }
        resolve();
      };

      audio.onended = finishPlayback;
      audio.onerror = finishPlayback;
      void audio
        .play()
        .then(() => {
          onPlaybackStart?.();
        })
        .catch(finishPlayback);
    });
  }

  function stopSpeechPlayback() {
    playbackSessionRef.current += 1;
    assistantSpeechQueueRef.current = [];
    assistantSpeechProcessingRef.current = false;
    ttsPrefetchRef.current = null;
    window.speechSynthesis?.cancel();
    stopAudioPlayback();
    speakingMessageIdRef.current = null;
    speakingPlaybackRef.current = null;
    setSpeakingMessageId(null);
  }

  function startTtsPrefetchForItem(item: AssistantSpeechItem) {
    if (!isElevenLabsVoiceUri(item.voiceUri)) return;
    const key = assistantSpeechItemKey(item);
    if (ttsPrefetchRef.current?.key === key) return;
    ttsPrefetchRef.current = {
      key,
      promise: requestElevenLabsAudio(
        item.text,
        parseElevenLabsVoiceId(item.voiceUri),
        item.speechRate,
      ),
    };
  }

  async function processAssistantSpeechQueue() {
    if (assistantSpeechProcessingRef.current) return;
    assistantSpeechProcessingRef.current = true;

    try {
      while (assistantSpeechQueueRef.current.length > 0) {
        const item = assistantSpeechQueueRef.current.shift();
        if (!item) continue;
        const nextItem = assistantSpeechQueueRef.current[0] ?? null;

        const prefetchKey = assistantSpeechItemKey(item);
        const prefetchedBlob =
          ttsPrefetchRef.current?.key === prefetchKey ? ttsPrefetchRef.current.promise : undefined;
        if (ttsPrefetchRef.current?.key === prefetchKey) {
          ttsPrefetchRef.current = null;
        }

        await speakTextAsync(item.text, item.messageId, item.voiceUri, item.speechRate, {
          prefetchedBlob,
          onPlaybackStart: () => {
            maybeDismissProgressModalOnPlaybackStart(item.conversationId);
            if (nextItem) {
              startTtsPrefetchForItem(nextItem);
            }
          },
        });
      }
    } catch {
      // Individual playback errors are handled inside speakTextAsync.
    } finally {
      assistantSpeechProcessingRef.current = false;
      if (assistantSpeechQueueRef.current.length > 0) {
        void processAssistantSpeechQueue();
      }
    }
  }

  async function speakTextAsync(
    content: string,
    messageId: number,
    voiceUri?: string,
    speechRate?: number,
    options?: {
      prefetchedBlob?: Promise<Blob>;
      onPlaybackStart?: () => void;
    },
  ): Promise<void> {
    const plain = content.trim();
    if (!plain) return;

    const resolvedVoiceUri = voiceUri || resolveDefaultTtsVoiceUri();
    const resolvedSpeechRate = speechRate ?? readStoredTtsSpeechRate();

    if (isElevenLabsVoiceUri(resolvedVoiceUri)) {
      window.speechSynthesis?.cancel();
      stopAudioPlayback();
      const session = ++playbackSessionRef.current;

      speakingPlaybackRef.current = {
        id: messageId,
        content: plain,
        voiceUri: resolvedVoiceUri,
        speechRate: resolvedSpeechRate,
      };
      speakingMessageIdRef.current = messageId;
      setSpeakingMessageId(messageId);

      try {
        const blob = options?.prefetchedBlob
          ? await options.prefetchedBlob
          : await requestElevenLabsAudio(
              plain,
              parseElevenLabsVoiceId(resolvedVoiceUri),
              resolvedSpeechRate,
            );
        if (playbackSessionRef.current !== session) return;
        await playElevenLabsBlob(blob, session, messageId, options?.onPlaybackStart);
      } catch (err) {
        if (playbackSessionRef.current === session) {
          speakingPlaybackRef.current = null;
          speakingMessageIdRef.current = null;
          setSpeakingMessageId(null);
          setError(err instanceof Error ? err.message : "Could not play speech");
        }
      }
      return;
    }

    if (!window.speechSynthesis) return;

    return new Promise((resolve) => {
      window.speechSynthesis.cancel();
      const session = ++playbackSessionRef.current;
      const utterance = new SpeechSynthesisUtterance(plain);
      applyTtsVoice(utterance, resolvedVoiceUri, resolvedSpeechRate);

      const finishPlayback = () => {
        if (playbackSessionRef.current !== session) {
          resolve();
          return;
        }
        speakingPlaybackRef.current = null;
        speakingMessageIdRef.current = null;
        setSpeakingMessageId(null);
        resolve();
      };

      utterance.onstart = () => {
        options?.onPlaybackStart?.();
      };
      utterance.onend = finishPlayback;
      utterance.onerror = finishPlayback;

      speakingPlaybackRef.current = {
        id: messageId,
        content: plain,
        voiceUri: resolvedVoiceUri,
        speechRate: resolvedSpeechRate,
      };
      speakingMessageIdRef.current = messageId;
      setSpeakingMessageId(messageId);
      window.speechSynthesis.speak(utterance);
    });
  }

  function shouldAutoPlayAssistantReplies(conversationId: number) {
    if (speakRepliesByConversationRef.current[conversationId]) {
      return true;
    }
    const job = generationJobsRef.current[conversationId];
    return Boolean(job?.isMultiAgent && job.autoPlayReplies);
  }

  function beginAssistantReplyPlayback(conversationId: number, initialMessageCount: number) {
    spokenMessageIdsByConversationRef.current[conversationId] = new Set();
    speakReplyStartIndexByConversationRef.current[conversationId] = initialMessageCount;
    stopSpeechPlayback();
  }

  function enqueueAssistantSpeech(
    conversationId: number,
    message: Message,
    participants: ConversationParticipant[] = activeParticipants,
  ) {
    if (message.id <= 0 || message.role !== "assistant" || !message.content.trim()) return;

    const spoken =
      spokenMessageIdsByConversationRef.current[conversationId] ??
      (spokenMessageIdsByConversationRef.current[conversationId] = new Set());
    if (spoken.has(message.id)) return;
    spoken.add(message.id);

    const plain = prepareAssistantTextForSpeech(message.content);
    if (!plain) return;

    const item: AssistantSpeechItem = {
      conversationId,
      messageId: message.id,
      text: plain,
      voiceUri: resolveTtsVoiceUriForMessage(message, participants, config?.models ?? []),
      speechRate: resolveTtsSpeechRateForMessage(message, participants, agentProfiles),
    };
    assistantSpeechQueueRef.current.push(item);

    const queue = assistantSpeechQueueRef.current;
    if (assistantSpeechProcessingRef.current) {
      startTtsPrefetchForItem(item);
    } else if (queue.length >= 2) {
      startTtsPrefetchForItem(queue[1]);
    }

    void processAssistantSpeechQueue();
  }

  function maybeSpeakPendingAssistantReplies(
    conversationId: number,
    messages: Message[],
    participants: ConversationParticipant[] = activeParticipants,
  ) {
    if (!shouldAutoPlayAssistantReplies(conversationId)) return;
    const initialMessageCount = speakReplyStartIndexByConversationRef.current[conversationId];
    const startIndex = initialMessageCount != null ? initialMessageCount + 1 : 0;
    for (const message of messages.slice(startIndex)) {
      if (message.role !== "assistant" || !message.content.trim()) continue;
      enqueueAssistantSpeech(conversationId, message, participants);
    }
  }

  function resolveDefaultTtsVoiceUri() {
    return selectedTtsVoiceUri || localStorage.getItem(TTS_VOICE_STORAGE_KEY) || "";
  }

  function resolveTtsVoiceUriForModel(modelName: string | null | undefined) {
    if (modelName && config?.models) {
      const match = config.models.find((entry) => entry.model === modelName);
      if (match?.tts_voice_uri) {
        return match.tts_voice_uri;
      }
    }
    return resolveDefaultTtsVoiceUri();
  }

  function resolveTtsVoiceUriForParticipant(participant: ConversationParticipant | ParticipantDraft, models: LlmModel[]) {
    if (participant.tts_voice_uri?.trim()) {
      return participant.tts_voice_uri.trim();
    }
    const profile = resolveAgentProfileForParticipant(participant, agentProfiles);
    if (profile?.tts_voice_uri?.trim()) {
      return profile.tts_voice_uri.trim();
    }
    const model = models.find((entry) => entry.id === participant.llm_model_id);
    return resolveTtsVoiceUriForModel(model?.model);
  }

  function resolveTtsVoiceUriForMessage(message: Message, participants: ConversationParticipant[], models: LlmModel[]) {
    if (message.participant_id != null) {
      const participant = participants.find((entry) => entry.id === message.participant_id);
      if (participant) {
        return resolveTtsVoiceUriForParticipant(participant, models);
      }
    }
    return resolveTtsVoiceUriForModel(message.llm_model);
  }

  function speakText(content: string, playbackId: number | "sample", voiceUri?: string, speechRate?: number) {
    const plain = content.trim();
    if (!plain) return;

    const resolvedVoiceUri = voiceUri || resolveDefaultTtsVoiceUri();
    const resolvedSpeechRate = speechRate ?? readStoredTtsSpeechRate();

    if (isElevenLabsVoiceUri(resolvedVoiceUri)) {
      window.speechSynthesis?.cancel();
      stopAudioPlayback();
      const session = ++playbackSessionRef.current;

      speakingPlaybackRef.current = {
        id: playbackId,
        content: plain,
        voiceUri: resolvedVoiceUri,
        speechRate: resolvedSpeechRate,
      };
      if (typeof playbackId === "number") {
        speakingMessageIdRef.current = playbackId;
        setSpeakingMessageId(playbackId);
      } else {
        speakingMessageIdRef.current = null;
        setSpeakingMessageId(null);
      }

      void requestElevenLabsAudio(
        plain,
        parseElevenLabsVoiceId(resolvedVoiceUri),
        resolvedSpeechRate,
      )
        .then((blob) => {
          if (playbackSessionRef.current !== session) return;
          return playElevenLabsBlob(blob, session, playbackId);
        })
        .catch((err) => {
          if (playbackSessionRef.current !== session) return;
          speakingPlaybackRef.current = null;
          speakingMessageIdRef.current = null;
          setSpeakingMessageId(null);
          setError(err instanceof Error ? err.message : "Could not play speech");
        });
      return;
    }

    if (!window.speechSynthesis) return;

    window.speechSynthesis.cancel();
    const session = ++playbackSessionRef.current;
    const utterance = new SpeechSynthesisUtterance(plain);
    applyTtsVoice(utterance, resolvedVoiceUri, resolvedSpeechRate);

    const finishPlayback = () => {
      if (playbackSessionRef.current !== session) return;
      speakingPlaybackRef.current = null;
      if (typeof playbackId === "number") {
        speakingMessageIdRef.current = null;
        setSpeakingMessageId(null);
      }
    };

    utterance.onend = finishPlayback;
    utterance.onerror = finishPlayback;

    speakingPlaybackRef.current = {
      id: playbackId,
      content: plain,
      voiceUri: resolvedVoiceUri,
      speechRate: resolvedSpeechRate,
    };
    if (typeof playbackId === "number") {
      speakingMessageIdRef.current = playbackId;
      setSpeakingMessageId(playbackId);
    } else {
      speakingMessageIdRef.current = null;
      setSpeakingMessageId(null);
    }
    window.speechSynthesis.speak(utterance);
  }

  function applyTtsVoice(utterance: SpeechSynthesisUtterance, voiceUri?: string, speechRate?: number) {
    const voice = resolveTtsVoice(voiceUri || resolveDefaultTtsVoiceUri());
    if (voice) {
      utterance.voice = voice;
    }
    utterance.rate = speechRate ?? readStoredTtsSpeechRate();
  }

  function handleTtsSpeechRateChange(rate: number) {
    const clamped = persistTtsSpeechRate(rate);
    setTtsSpeechRate(clamped);

    const current = speakingPlaybackRef.current;
    if (!current) return;
    const isPlaying =
      Boolean(window.speechSynthesis?.speaking) ||
      Boolean(audioPlaybackRef.current && !audioPlaybackRef.current.paused);
    if (!isPlaying) return;
    speakText(current.content, current.id, current.voiceUri, clamped);
  }

  function playAssistantMessage(message: Message, selectedText?: string) {
    const selected = selectedText?.trim() || getSelectedTextInMessage(message.id);
    const plain = selected || prepareAssistantTextForSpeech(message.content);
    if (!plain) return;
    speakText(
      plain,
      message.id,
      resolveTtsVoiceUriForMessage(message, activeParticipants, config?.models ?? []),
      resolveTtsSpeechRateForMessage(message, activeParticipants, agentProfiles),
    );
  }

  function captureSpeechSelection(message: Message) {
    const selected = getSelectedTextInMessage(message.id);
    speechSelectionRef.current = selected ? { messageId: message.id, text: selected } : null;
  }

  function handlePlayAssistantMessage(message: Message) {
    const captured = speechSelectionRef.current;
    speechSelectionRef.current = null;
    const selected = captured?.messageId === message.id ? captured.text : undefined;
    playAssistantMessage(message, selected);
  }

  function playVoiceSample(voiceUri?: string, speechRate?: number) {
    speakText(TTS_SAMPLE_TEXT, "sample", voiceUri, speechRate ?? readStoredTtsSpeechRate());
  }

  function updateElevenlabsVoiceFilter(key: keyof ElevenLabsVoiceFilters, value: string) {
    setElevenlabsVoiceFilters((current) => ({ ...current, [key]: value }));
  }

  function toggleElevenlabsVoiceSort(key: ElevenLabsVoiceSortKey) {
    setElevenlabsVoiceSort((current) => {
      if (current?.key !== key) {
        return { key, direction: "asc" };
      }
      if (current.direction === "asc") {
        return { key, direction: "desc" };
      }
      return null;
    });
  }

  function renderElevenlabsSortIcon(key: ElevenLabsVoiceSortKey) {
    if (elevenlabsVoiceSort?.key !== key) {
      return <ArrowUpDown size={14} />;
    }
    return elevenlabsVoiceSort.direction === "asc" ? <ArrowUp size={14} /> : <ArrowDown size={14} />;
  }

  function renderElevenlabsVoiceColumnHeader(label: string, sortKey: ElevenLabsVoiceSortKey) {
    return (
      <>
        <div className="elevenlabs-column-header">
          <span>{label}</span>
          <button
            type="button"
            className={`elevenlabs-sort-button ${elevenlabsVoiceSort?.key === sortKey ? "elevenlabs-sort-button-active" : ""}`}
            onClick={() => toggleElevenlabsVoiceSort(sortKey)}
            aria-label={`Sort by ${label.toLowerCase()}`}
          >
            {renderElevenlabsSortIcon(sortKey)}
          </button>
        </div>
        <input
          type="search"
          className="elevenlabs-column-filter"
          value={elevenlabsVoiceFilters[sortKey]}
          onChange={(event) => updateElevenlabsVoiceFilter(sortKey, event.target.value)}
          placeholder="Filter"
          aria-label={`Filter by ${label.toLowerCase()}`}
        />
      </>
    );
  }

  function displayElevenlabsVoiceMeta(value?: string | null) {
    return value?.trim() || "—";
  }

  async function importElevenLabsVoices(options?: { silent?: boolean; skipApiKeyCheck?: boolean }) {
    if (!options?.skipApiKeyCheck && !speechConfig?.has_elevenlabs_api_key) {
      if (!options?.silent) {
        setError("Save an ElevenLabs API key before importing voices.");
      }
      return;
    }

    if (!options?.silent) {
      setError("");
    }
    setIsImportingElevenlabsVoices(true);
    try {
      const voices = await request<ElevenLabsVoice[]>("/api/tts/elevenlabs/voices");
      setElevenlabsCatalogVoices(voices);
      persistElevenlabsCatalog(voices);
      setElevenlabsVoiceFilters(EMPTY_ELEVENLABS_VOICE_FILTERS);
      setElevenlabsVoiceSort(null);
      if (!options?.silent) {
        showSettingsSaveSuccess(
          voices.length
            ? `Imported ${voices.length} ElevenLabs voice${voices.length === 1 ? "" : "s"}.`
            : "No ElevenLabs voices found on this account.",
        );
      }
    } catch (err) {
      if (!options?.silent) {
        setError(err instanceof Error ? err.message : "Could not import ElevenLabs voices");
      }
    } finally {
      setIsImportingElevenlabsVoices(false);
    }
  }

  function playModelVoiceSample(voiceUri: string | null | undefined, speechRate?: number) {
    playVoiceSample(voiceUri || undefined, speechRate);
  }

  function handleTtsVoiceChange(voiceUri: string) {
    setSelectedTtsVoiceUri(voiceUri);
    localStorage.setItem(TTS_VOICE_STORAGE_KEY, voiceUri);
  }

  function clearPendingImage() {
    setPendingImage(null);
    if (imageInputRef.current) {
      imageInputRef.current.value = "";
    }
  }

  async function attachImageFile(file: File) {
    if (!ALLOWED_IMAGE_TYPES.has(file.type)) {
      setError("Use JPEG, PNG, GIF, or WebP images.");
      return;
    }
    if (file.size > MAX_IMAGE_BYTES) {
      setError("Image must be 10 MB or smaller.");
      return;
    }

    try {
      const previewUrl = await readBlobAsDataUrl(file);
      const { base64, mediaType } = parseImageDataUrl(previewUrl);
      setPendingImage({
        base64,
        mediaType,
        previewUrl,
        name: file.name,
      });
      setError("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not read image");
    }
  }

  async function handleImageInputChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    await attachImageFile(file);
  }

  function handleComposerPaste(event: ClipboardEvent<HTMLTextAreaElement>) {
    const items = event.clipboardData?.items;
    if (!items) return;

    for (const item of items) {
      if (!item.type.startsWith("image/")) continue;
      const file = item.getAsFile();
      if (!file) continue;
      event.preventDefault();
      void attachImageFile(file);
      return;
    }
  }

  function showSettingsSaveSuccess(message: string) {
    setSettingsSaveSuccessMessage(message);
  }

  async function saveSpeechConfig(event: FormEvent) {
    event.preventDefault();
    setError("");
    try {
      const payload: Record<string, string | boolean> = {
        whisper_model: whisperModelDraft,
        clear_elevenlabs_api_key: clearElevenlabsApiKeyDraft,
      };
      if (elevenlabsApiKeyDraft.trim()) {
        payload.elevenlabs_api_key = elevenlabsApiKeyDraft.trim();
      }
      const saved = await request<SpeechConfig>("/api/config/speech", {
        method: "PUT",
        body: JSON.stringify(payload),
      });
      setSpeechConfig(saved);
      setWhisperModelDraft(saved.whisper_model);
      setElevenlabsApiKeyDraft("");
      setClearElevenlabsApiKeyDraft(false);
      if (!saved.has_elevenlabs_api_key) {
        setElevenlabsCatalogVoices([]);
        clearStoredElevenlabsCatalog();
        setElevenlabsVoiceFilters(EMPTY_ELEVENLABS_VOICE_FILTERS);
        setElevenlabsVoiceSort(null);
      } else {
        void importElevenLabsVoices({ silent: true });
      }
      showSettingsSaveSuccess("Speech settings saved.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not save speech settings");
    }
  }

  async function savePromptConfig(event: FormEvent) {
    event.preventDefault();
    setError("");
    try {
      const saved = await request<PromptConfig>("/api/config/prompt", {
        method: "PUT",
        body: JSON.stringify({
          default_prompt: defaultPromptDraft,
          multi_agent_prompt: defaultMultiAgentPromptDraft,
        }),
      });
      setPromptConfig(saved);
      setDefaultPromptDraft(saved.default_prompt);
      setDefaultMultiAgentPromptDraft(saved.multi_agent_prompt);
      showSettingsSaveSuccess("Default prompts saved.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not save default prompt");
    }
  }

  function resetDefaultPrompt() {
    const baseline = promptConfig?.default_prompt_baseline ?? "";
    setDefaultPromptDraft(baseline);
  }

  function resetDefaultMultiAgentPrompt() {
    const baseline = promptConfig?.multi_agent_prompt_baseline ?? "";
    setDefaultMultiAgentPromptDraft(baseline);
  }

  function mergeDetailPreservingOptimistic(
    serverDetail: ConversationDetail,
    currentDetail: ConversationDetail | null,
    conversationId: number,
  ): ConversationDetail {
    if (!currentDetail || currentDetail.conversation.id !== conversationId) {
      return serverDetail;
    }
    const job = generationJobsRef.current[conversationId];
    if (!job) {
      return serverDetail;
    }

    const optimisticMessages = currentDetail.messages.filter((message) => message.id < 0);
    if (!optimisticMessages.length) {
      return serverDetail;
    }

    const initialCount = job.initialMessageCount ?? persistedMessageCount(serverDetail.messages);
    const pendingOptimistic = optimisticMessages.filter((message) => {
      if (message.role === "user") {
        if (message.content.trim() === TRANSCRIBING_PLACEHOLDER) {
          return serverDetail.messages.length <= initialCount;
        }
        return !serverDetail.messages.some(
          (serverMessage) =>
            serverMessage.role === "user" && serverMessage.content.trim() === message.content.trim(),
        );
      }
      if (message.role === "assistant" && !message.content.trim()) {
        if (job?.isMultiAgent) {
          return false;
        }
        const lastServer = serverDetail.messages[serverDetail.messages.length - 1];
        return lastServer?.role !== "assistant" || Boolean(lastServer.content?.trim());
      }
      return false;
    });

    if (!pendingOptimistic.length) {
      return serverDetail;
    }

    return {
      ...serverDetail,
      messages: [...serverDetail.messages, ...pendingOptimistic],
    };
  }

  function updateOptimisticMessageContent(messageId: number, content: string) {
    setDetail((current) =>
      current
        ? {
            ...current,
            messages: current.messages.map((message) =>
              message.id === messageId ? { ...message, content } : message,
            ),
          }
        : current,
    );
  }

  function removeOptimisticMessage(messageId: number) {
    setDetail((current) =>
      current
        ? {
            ...current,
            messages: current.messages.filter((message) => message.id !== messageId),
          }
        : current,
    );
  }

  function appendOptimisticUserMessage(content: string) {
    if (!activeId) return null;

    const userId = -Date.now();
    const createdAt = createOptimisticTimestamp();
    const optimisticMessage: Message = {
      id: userId,
      conversation_id: activeId,
      role: "user",
      content,
      created_at: createdAt,
    };
    setDetail((current) => {
      if (!current || current.conversation.id !== activeId) {
        const conversation = conversations.find((item) => item.id === activeId);
        if (!conversation) return current;
        return {
          conversation,
          messages: [optimisticMessage],
          memories: [],
          participants: [],
        };
      }
      return {
        ...current,
        messages: [...current.messages, optimisticMessage],
      };
    });
    requestAnimationFrame(() => {
      const messagesEl = document.querySelector(".messages");
      if (messagesEl) {
        messagesEl.scrollTop = messagesEl.scrollHeight;
      }
    });
    return userId;
  }

  function appendOptimisticAssistantPlaceholder() {
    if (!activeId) return null;

    const assistantId = -Date.now() - 1;
    const createdAt = createOptimisticTimestamp();
    setDetail((current) => {
      if (!current) return current;
      const last = current.messages[current.messages.length - 1];
      if (last?.role === "assistant" && !last.content && last.id < 0) {
        return current;
      }
      return {
        ...current,
        messages: [
          ...current.messages,
          {
            id: assistantId,
            conversation_id: activeId,
            role: "assistant",
            content: "",
            llm_model: activeModelName,
            created_at: createdAt,
          },
        ],
      };
    });
    return assistantId;
  }

  function appendOptimisticExchange(content: string) {
    if (!activeId) return { userId: null, assistantId: null };

    const userId = -Date.now();
    const assistantId = userId - 1;
    const createdAt = createOptimisticTimestamp();
    setDetail((current) =>
      current && {
        ...current,
        messages: [
          ...current.messages,
          {
            id: userId,
            conversation_id: activeId,
            role: "user",
            content,
            created_at: createdAt,
          },
          {
            id: assistantId,
            conversation_id: activeId,
            role: "assistant",
            content: "",
            llm_model: activeModelName,
            created_at: createdAt,
          },
        ],
      },
    );
    requestAnimationFrame(() => {
      const messagesEl = document.querySelector(".messages");
      if (messagesEl) {
        messagesEl.scrollTop = messagesEl.scrollHeight;
      }
    });
    return { userId, assistantId };
  }

  async function loadConversations() {
    const data = await request<Conversation[]>("/api/conversations");
    setConversations(data);
    setActiveId((current) => current ?? data[0]?.id ?? null);
  }

  async function loadConversation(conversationId: number) {
    const data = await request<ConversationDetail>(`/api/conversations/${conversationId}`);
    if (activeIdRef.current === conversationId) {
      setDetail((current) => mergeDetailPreservingOptimistic(data, current, conversationId));
    }
    setConversations((current) =>
      current.map((item) =>
        item.id === conversationId
          ? { ...item, participant_count: data.participants.length, updated_at: data.conversation.updated_at }
          : item,
      ),
    );
    const job = generationJobsRef.current[conversationId];
    if (job?.isMultiAgent && job.initialMessageCount != null) {
      const newAssistants = data.messages
        .slice(job.initialMessageCount + 1)
        .filter((message) => message.role === "assistant" && message.content.trim());
      const completedSteps = newAssistants.length;
      const lastAssistant = newAssistants[newAssistants.length - 1];
      const currentParticipantName = lastAssistant
        ? messageSpeakerLabel(lastAssistant, data.participants, config?.models ?? [])
        : job.currentParticipantName;
      updateGenerationJob(conversationId, { completedSteps, currentParticipantName });
    }
    maybeSpeakPendingAssistantReplies(conversationId, data.messages, data.participants);
    return data;
  }

  function updateGenerationJob(conversationId: number, patch: Partial<GenerationJob>) {
    setGenerationJobs((current) => {
      const job = current[conversationId];
      if (!job) return current;
      return { ...current, [conversationId]: { ...job, ...patch } };
    });
  }

  function removeGenerationJob(conversationId: number) {
    setGenerationJobs((current) => {
      if (!(conversationId in current)) return current;
      const next = { ...current };
      delete next[conversationId];
      return next;
    });
    generationAbortByConversationRef.current.delete(conversationId);
    dismissedProgressModalOnPlaybackRef.current.delete(conversationId);
    setProgressModalConversationId((current) => (current === conversationId ? null : current));
  }

  function dismissProgressModal() {
    setProgressModalConversationId(null);
  }

  function maybeDismissProgressModalOnPlaybackStart(conversationId: number) {
    if (dismissedProgressModalOnPlaybackRef.current.has(conversationId)) return;
    dismissedProgressModalOnPlaybackRef.current.add(conversationId);
    setProgressModalConversationId((current) => (current === conversationId ? null : current));
  }

  function cancelGeneration(conversationId?: number) {
    const targetId = conversationId ?? progressModalConversationId ?? activeId;
    if (targetId == null) return;
    generationAbortByConversationRef.current.get(targetId)?.abort();
  }

  function dismissCompletedGenerationAlert() {
    if (!completedGenerationAlert) return;
    const { conversationId, error } = completedGenerationAlert;
    setCompletedGenerationAlert(null);
    if (!error && activeIdRef.current !== conversationId) {
      setUnreviewedReplyConversationIds((current) =>
        current.includes(conversationId) ? current : [...current, conversationId],
      );
    }
  }

  async function goToCompletedAnswer() {
    const alert = completedGenerationAlert;
    if (!alert) return;
    setCompletedGenerationAlert(null);
    pendingLatestScrollRef.current = true;
    setPage("chat");
    setMessageContextMenu(null);
    setSelectedClip(null);
    if (activeId !== alert.conversationId) {
      setActiveId(alert.conversationId);
      return;
    }
    await loadConversation(alert.conversationId);
  }

  async function loadMemoryGroups() {
    const groups = await request<MemoryGroup[]>(`/api/memories?sort=${memorySort}&order=${memoryOrder}`);
    setMemoryGroups(groups);
    const expandConversationId = expandConversationMemoriesRef.current;
    if (expandConversationId) {
      expandConversationMemoriesRef.current = null;
      setExpandedMemoryGroupIds((current) =>
        current.includes(expandConversationId) ? current : [...current, expandConversationId],
      );
      const targetGroup = groups.find((group) => group.conversation.id === expandConversationId);
      if (targetGroup?.memories.length) {
        setExpandedMemoryIds(targetGroup.memories.map((memory) => memory.id));
      }
    }
    if (activeId) {
      setExpandedMemoryGroupIds((current) => (current.includes(activeId) ? current : [...current, activeId]));
      setMergeTargetConversationId((current) => current || activeId);
      const activeGroup = groups.find((group) => group.conversation.id === activeId);
      if (activeGroup) {
        setDetail((current) => current && { ...current, memories: activeGroup.memories });
      }
    }
  }

  async function loadConfig() {
    const data = await request<LlmConfig>("/api/config/llm");
    setConfig(data);
    setConfigDraft(
      data.models.map((model) => ({
        ...model,
        api_key: "",
        clear_api_key: false,
      })),
    );
  }

  async function loadAgentProfiles() {
    const data = await request<AgentProfile[]>("/api/agent-profiles");
    setAgentProfiles(data);
  }

  function participantPayloadFromDraft(participant: ParticipantDraft): ConversationParticipantPayload {
    return {
      llm_model_id: participant.llm_model_id,
      personality: participant.personality,
      name: participant.name,
      tts_voice_uri: participant.tts_voice_uri?.trim() || null,
      tts_speech_rate: participant.tts_speech_rate ?? null,
      agent_profile_id: participant.agent_profile_id ?? null,
    };
  }

  function agentProfilePayloadFromDraft(participant: ParticipantDraft) {
    return {
      name: participant.name.trim(),
      personality: participant.personality,
      llm_model_id: participant.llm_model_id,
      tts_voice_uri: participant.tts_voice_uri?.trim() || null,
      tts_speech_rate: participant.tts_speech_rate ?? null,
    };
  }

  function updateParticipantDraftAt(index: number, patch: Partial<ParticipantDraft>) {
    if (participantEditorTarget === "new") {
      setNewConversationParticipants((current) =>
        current.map((participant, i) => (i === index ? { ...participant, ...patch } : participant)),
      );
      return;
    }
    setParticipantsDraft((current) =>
      current.map((participant, i) => (i === index ? { ...participant, ...patch } : participant)),
    );
  }

  async function saveAgentProfileFromDraft(index: number) {
    const drafts = participantEditorTarget === "new" ? newConversationParticipants : participantsDraft;
    const draft = drafts[index];
    if (!draft?.name.trim()) {
      setError("Agent name is required to save to the library");
      return;
    }
    setSavingAgentProfileIndex(index);
    setError("");
    try {
      const saved = await request<AgentProfile>("/api/agent-profiles", {
        method: "POST",
        body: JSON.stringify(agentProfilePayloadFromDraft(draft)),
      });
      setAgentProfiles((current) =>
        [...current.filter((item) => item.id !== saved.id), saved].sort((a, b) => a.name.localeCompare(b.name)),
      );
      updateParticipantDraftAt(index, { agent_profile_id: saved.id });
      setNotice(`Saved agent "${saved.name}" to library.`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not save agent profile");
    } finally {
      setSavingAgentProfileIndex(null);
    }
  }

  async function updateAgentProfileFromDraft(index: number) {
    const drafts = participantEditorTarget === "new" ? newConversationParticipants : participantsDraft;
    const draft = drafts[index];
    if (!draft?.name.trim()) {
      setError("Agent name is required");
      return;
    }
    if (!draft.agent_profile_id) {
      await saveAgentProfileFromDraft(index);
      return;
    }
    setSavingAgentProfileIndex(index);
    setError("");
    try {
      const saved = await request<AgentProfile>(`/api/agent-profiles/${draft.agent_profile_id}`, {
        method: "PUT",
        body: JSON.stringify(agentProfilePayloadFromDraft(draft)),
      });
      setAgentProfiles((current) =>
        current
          .map((item) => (item.id === saved.id ? saved : item))
          .sort((a, b) => a.name.localeCompare(b.name)),
      );
      setNotice(`Updated agent "${saved.name}" in library.`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not update agent profile");
    } finally {
      setSavingAgentProfileIndex(null);
    }
  }

  async function deleteAgentProfile(profileId: number) {
    setError("");
    try {
      await request(`/api/agent-profiles/${profileId}`, { method: "DELETE" });
      setAgentProfiles((current) => current.filter((item) => item.id !== profileId));
      setNewConversationParticipants((current) =>
        current.map((participant) =>
          participant.agent_profile_id === profileId ? { ...participant, agent_profile_id: null } : participant,
        ),
      );
      setParticipantsDraft((current) =>
        current.map((participant) =>
          participant.agent_profile_id === profileId ? { ...participant, agent_profile_id: null } : participant,
        ),
      );
      if (editingAgentProfile?.id === profileId) {
        closeAgentProfileEditor();
      }
      setNotice("Agent removed from library.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not delete agent profile");
    }
  }

  function openAgentProfileEditor(profile: AgentProfile) {
    setIsCreatingAgentProfile(false);
    setEditingAgentProfile(profile);
    setAgentProfileEditDraft(participantDraftFromProfile(profile));
    setError("");
  }

  function openCreateAgentProfile() {
    if (!config?.models.length) {
      setError("Add an LLM model before creating agents");
      return;
    }
    setIsCreatingAgentProfile(true);
    setEditingAgentProfile(null);
    setAgentProfileEditDraft(createParticipantDraft(config.models));
    setError("");
  }

  function closeAgentProfileEditor() {
    setIsCreatingAgentProfile(false);
    setEditingAgentProfile(null);
    setAgentProfileEditDraft(null);
  }

  function syncParticipantsFromSavedAgentProfile(saved: AgentProfile, previousProfile?: AgentProfile | null) {
    const previousVoice = (previousProfile?.tts_voice_uri ?? "").trim() || null;
    const previousRate = previousProfile?.tts_speech_rate ?? null;

    function shouldInheritSavedVoice(participant: ConversationParticipant | ParticipantDraft) {
      if (participant.agent_profile_id !== saved.id) return false;
      const participantVoice = (participant.tts_voice_uri ?? "").trim() || null;
      if (!participantVoice) return true;
      return participantVoice === previousVoice;
    }

    function shouldInheritSavedRate(participant: ConversationParticipant | ParticipantDraft) {
      if (participant.agent_profile_id !== saved.id) return false;
      if (participant.tts_speech_rate == null) return true;
      return participant.tts_speech_rate === previousRate;
    }

    setDetail((current) => {
      if (!current) return current;
      return {
        ...current,
        participants: current.participants.map((participant) => {
          if (!shouldInheritSavedVoice(participant) && !shouldInheritSavedRate(participant)) {
            return participant;
          }
          return {
            ...participant,
            tts_voice_uri: shouldInheritSavedVoice(participant) ? saved.tts_voice_uri : participant.tts_voice_uri,
            tts_speech_rate: shouldInheritSavedRate(participant) ? saved.tts_speech_rate : participant.tts_speech_rate,
          };
        }),
      };
    });

    const syncDraft = (participant: ParticipantDraft): ParticipantDraft => {
      if (!shouldInheritSavedVoice(participant) && !shouldInheritSavedRate(participant)) {
        return participant;
      }
      return {
        ...participant,
        tts_voice_uri: shouldInheritSavedVoice(participant)
          ? saved.tts_voice_uri ?? ""
          : participant.tts_voice_uri,
        tts_speech_rate: shouldInheritSavedRate(participant) ? saved.tts_speech_rate : participant.tts_speech_rate,
      };
    };

    setParticipantsDraft((current) => current.map(syncDraft));
    setNewConversationParticipants((current) => current.map(syncDraft));
  }

  async function saveAgentProfileEdit(event?: FormEvent) {
    event?.preventDefault();
    if (!agentProfileEditDraft?.name.trim()) {
      setError("Agent name is required");
      return;
    }
    setSavingAgentProfileEdit(true);
    setError("");
    try {
      if (isCreatingAgentProfile) {
        const saved = await request<AgentProfile>("/api/agent-profiles", {
          method: "POST",
          body: JSON.stringify(agentProfilePayloadFromDraft(agentProfileEditDraft)),
        });
        setAgentProfiles((current) =>
          [...current.filter((item) => item.id !== saved.id), saved].sort((a, b) => a.name.localeCompare(b.name)),
        );
        closeAgentProfileEditor();
        setNotice(`Created agent "${saved.name}".`);
      } else {
        if (!editingAgentProfile) return;
        const saved = await request<AgentProfile>(`/api/agent-profiles/${editingAgentProfile.id}`, {
          method: "PUT",
          body: JSON.stringify(agentProfilePayloadFromDraft(agentProfileEditDraft)),
        });
        setAgentProfiles((current) =>
          [...current.filter((item) => item.id !== saved.id), saved].sort((a, b) => a.name.localeCompare(b.name)),
        );
        syncParticipantsFromSavedAgentProfile(saved, editingAgentProfile);
        closeAgentProfileEditor();
        setNotice(`Updated agent "${saved.name}".`);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not save agent profile");
    } finally {
      setSavingAgentProfileEdit(false);
    }
  }

  async function resetModelGenerationStats(modelId: number) {
    setError("");
    setResettingTimingModelId(modelId);
    try {
      const updatedModel = await request<LlmModel>(`/api/config/llm/models/${modelId}/generation-stats`, {
        method: "DELETE",
      });
      setConfig((current) =>
        current
          ? {
              ...current,
              models: current.models.map((model) => (model.id === modelId ? { ...model, ...updatedModel } : model)),
              active_model: current.active_model.id === modelId ? { ...current.active_model, ...updatedModel } : current.active_model,
            }
          : current,
      );
      setConfigDraft((current) =>
        current.map((model) =>
          model.id === modelId
            ? {
                ...model,
                ...updatedModel,
                api_key: model.api_key ?? "",
                clear_api_key: model.clear_api_key ?? false,
              }
            : model,
        ),
      );
      showSettingsSaveSuccess("Generation timing estimate reset.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not reset generation timing");
    } finally {
      setResettingTimingModelId(null);
    }
  }

  async function loadSpeechConfig(options?: { autoImportVoices?: boolean }) {
    const data = await request<SpeechConfig>("/api/config/speech");
    setSpeechConfig(data);
    setWhisperModelDraft(data.whisper_model);
    setElevenlabsApiKeyDraft("");
    setClearElevenlabsApiKeyDraft(false);
    if (!data.has_elevenlabs_api_key) {
      setElevenlabsCatalogVoices([]);
      clearStoredElevenlabsCatalog();
      return;
    }
    if (options?.autoImportVoices && !elevenlabsCatalogHydratedRef.current) {
      elevenlabsCatalogHydratedRef.current = true;
      void importElevenLabsVoices({ silent: true, skipApiKeyCheck: true });
    }
  }

  async function loadPromptConfig() {
    const data = await request<PromptConfig>("/api/config/prompt");
    setPromptConfig(data);
    setDefaultPromptDraft(data.default_prompt);
    setDefaultMultiAgentPromptDraft(data.multi_agent_prompt);
  }

  async function createConversation(event?: FormEvent) {
    event?.preventDefault();
    setError("");
    const payload: { title?: string; participants?: ConversationParticipantPayload[] } = {
      title: newTitle || undefined,
    };
    if (newConversationMode === "multi") {
      if (newConversationParticipants.length < 1 || newConversationParticipants.length > 3) {
        setError("Select 1 to 3 discussion participants");
        return;
      }
      payload.participants = newConversationParticipants.map(participantPayloadFromDraft);
    }
    const conversation = await request<Conversation>("/api/conversations", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    setNewTitle("");
    setNewConversationMode("single");
    setNewConversationParticipants([]);
    setIsNewConversationModalOpen(false);
    setConversations((current) => [conversation, ...current]);
    setActiveId(conversation.id);
    setPage("chat");
  }

  function openNewConversationModal() {
    setError("");
    setNewTitle("");
    setNewConversationMode("single");
    if (config?.models.length) {
      setNewConversationParticipants([createParticipantDraft(config.models)]);
    } else {
      setNewConversationParticipants([]);
    }
    setParticipantEditorTarget("new");
    setIsNewConversationModalOpen(true);
  }

  function openParticipantsModal() {
    if (!detail?.participants.length) return;
    setParticipantEditorTarget("edit");
    setParticipantsDraft(
      detail.participants.map((participant) => ({
        llm_model_id: participant.llm_model_id,
        personality: participant.personality,
        name: participant.name,
        tts_voice_uri: participant.tts_voice_uri ?? "",
        tts_speech_rate: participant.tts_speech_rate ?? null,
        agent_profile_id: participant.agent_profile_id ?? null,
      })),
    );
    setIsParticipantsModalOpen(true);
  }

  async function saveParticipants(event?: FormEvent) {
    event?.preventDefault();
    if (!activeId || participantsDraft.length < 1 || participantsDraft.length > 3) {
      setError("Select 1 to 3 discussion participants");
      return;
    }
    setSavingParticipants(true);
    setError("");
    try {
      const participants = await request<ConversationParticipant[]>(
        `/api/conversations/${activeId}/participants`,
        {
          method: "PUT",
          body: JSON.stringify({ participants: participantsDraft.map(participantPayloadFromDraft) }),
        },
      );
      setDetail((current) => (current ? { ...current, participants } : current));
      setConversations((current) =>
        current.map((item) =>
          item.id === activeId ? { ...item, participant_count: participants.length } : item,
        ),
      );
      setIsParticipantsModalOpen(false);
      setNotice("Discussion participants updated.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not update participants");
    } finally {
      setSavingParticipants(false);
    }
  }

  async function changeConversationModel(modelId: number) {
    if (!activeId) return;
    if (conversationModel?.id === modelId) {
      setIsConversationModelModalOpen(false);
      return;
    }

    setError("");
    setSelectingModelId(modelId);
    try {
      const conversation = await request<Conversation>(`/api/conversations/${activeId}/model`, {
        method: "PUT",
        body: JSON.stringify({ llm_model_id: modelId }),
      });
      setDetail((current) => current && { ...current, conversation });
      setConversations((current) =>
        current.map((item) => (item.id === conversation.id ? conversation : item)),
      );
      setIsConversationModelModalOpen(false);
      setNotice("Discussion model updated.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not update discussion model");
    } finally {
      setSelectingModelId(null);
    }
  }

  function startEditingConversationTitle(conversation: Conversation) {
    skipConversationTitleSaveRef.current = false;
    setError("");
    editingConversationIdRef.current = conversation.id;
    setEditingConversationId(conversation.id);
    setEditingConversationTitle(conversation.title);
  }

  function cancelConversationTitleEdit() {
    skipConversationTitleSaveRef.current = true;
    editingConversationIdRef.current = null;
    setEditingConversationId(null);
    setEditingConversationTitle("");
  }

  async function saveConversationTitle(conversation: Conversation, nextTitle?: string) {
    if (skipConversationTitleSaveRef.current) {
      skipConversationTitleSaveRef.current = false;
      return;
    }
    if (editingConversationIdRef.current !== conversation.id) {
      return;
    }

    const title = (nextTitle ?? editingConversationTitle).trim() || "New Conversation";
    skipConversationTitleSaveRef.current = true;
    editingConversationIdRef.current = null;
    setEditingConversationId(null);
    setEditingConversationTitle("");

    if (title === conversation.title) {
      skipConversationTitleSaveRef.current = false;
      return;
    }

    setError("");
    try {
      const updatedConversation = await request<Conversation>(`/api/conversations/${conversation.id}/title`, {
        method: "PUT",
        body: JSON.stringify({ title }),
      });
      setConversations((current) =>
        current.map((item) => (item.id === updatedConversation.id ? updatedConversation : item)),
      );
      setDetail((current) =>
        current?.conversation.id === updatedConversation.id
          ? { ...current, conversation: updatedConversation }
          : current,
      );
      setMemoryGroups((current) =>
        current.map((group) =>
          group.conversation.id === updatedConversation.id ? { ...group, conversation: updatedConversation } : group,
        ),
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not update discussion title");
    } finally {
      skipConversationTitleSaveRef.current = false;
    }
  }

  function handleConversationTitleKeyDown(event: KeyboardEvent<HTMLInputElement>, conversation: Conversation) {
    if (event.key === "Enter") {
      event.preventDefault();
      void saveConversationTitle(conversation, event.currentTarget.value);
    }
    if (event.key === "Escape") {
      event.preventDefault();
      cancelConversationTitleEdit();
    }
  }

  function changeConversationSortMode(mode: ConversationSortMode) {
    setConversationSortMode(mode);
    persistConversationSortMode(mode);
  }

  async function persistConversationOrder(nextConversations: Conversation[]) {
    try {
      const saved = await request<Conversation[]>("/api/conversations/reorder", {
        method: "PUT",
        body: JSON.stringify({ conversation_ids: nextConversations.map((conversation) => conversation.id) }),
      });
      setConversations(saved);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not save conversation order");
      await loadConversations();
    }
  }

  function moveConversation(draggedId: number, targetId: number) {
    if (draggedId === targetId) return;
    const currentList = sortConversations(conversations, conversationSortMode);
    const fromIndex = currentList.findIndex((conversation) => conversation.id === draggedId);
    const toIndex = currentList.findIndex((conversation) => conversation.id === targetId);
    if (fromIndex === -1 || toIndex === -1) return;

    const nextConversations = [...currentList];
    const [moved] = nextConversations.splice(fromIndex, 1);
    nextConversations.splice(toIndex, 0, moved);
    if (conversationSortMode !== "custom") {
      changeConversationSortMode("custom");
    }
    setConversations(nextConversations);
    void persistConversationOrder(nextConversations);
  }

  async function openDeleteConversationModal(conversation: Conversation) {
    setError("");
    const destinations = conversations.filter((candidate) => candidate.id !== conversation.id);
    const cachedDetail = detail?.conversation.id === conversation.id ? detail : null;
    setDeleteConversationTarget(conversation);
    setDeleteMessageCount(cachedDetail ? cachedDetail.messages.filter((message) => message.id >= 0).length : 0);
    setDeleteMemoryCount(cachedDetail?.memories.length ?? 0);
    setDeleteMemoryAction(
      (cachedDetail?.memories.length ?? 0) > 0 && destinations.length > 0 ? "move" : "delete",
    );
    setDeleteTargetId(destinations[0]?.id ?? "");

    try {
      const conversationDetail = await request<ConversationDetail>(`/api/conversations/${conversation.id}`);
      setDeleteMessageCount(conversationDetail.messages.filter((message) => message.id >= 0).length);
      setDeleteMemoryCount(conversationDetail.memories.length);
      setDeleteMemoryAction(conversationDetail.memories.length > 0 && destinations.length > 0 ? "move" : "delete");
      setDeleteTargetId(destinations[0]?.id ?? "");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load conversation details");
    }
  }

  async function confirmDeleteConversation(event: FormEvent) {
    event.preventDefault();
    if (!deleteConversationTarget) return;

    setError("");
    try {
      await request(`/api/conversations/${deleteConversationTarget.id}/delete`, {
        method: "POST",
        body: JSON.stringify({
          memory_action: deleteMemoryAction,
          target_conversation_id: deleteMemoryAction === "move" ? deleteTargetId : undefined,
        }),
      });

      const updatedConversations = await request<Conversation[]>("/api/conversations");
      const nextActiveId =
        activeId === deleteConversationTarget.id || !updatedConversations.some((conversation) => conversation.id === activeId)
          ? updatedConversations[0]?.id ?? null
          : activeId;

      setConversations(updatedConversations);
      setActiveId(nextActiveId);
      if (nextActiveId === activeId && nextActiveId !== null) {
        await loadConversation(nextActiveId);
      }
      setDeleteConversationTarget(null);
      setDeleteMessageCount(0);
      setDeleteMemoryCount(0);
      setDeleteMemoryAction("delete");
      setDeleteTargetId("");
      setNotice("Conversation deleted.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not delete conversation");
    }
  }

  function stopMediaRecorder(): Promise<Blob | null> {
    return new Promise((resolve) => {
      const recorder = mediaRecorderRef.current;
      if (!recorder || recorder.state === "inactive") {
        resolve(null);
        return;
      }

      recorder.onstop = () => {
        const blob = new Blob(mediaChunksRef.current, { type: recorder.mimeType || "audio/webm" });
        mediaChunksRef.current = [];
        mediaRecorderRef.current = null;
        resolve(blob.size > 0 ? blob : null);
      };
      recorder.stop();
      mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    });
  }

  async function startVoiceCapture() {
    if (!activeId || activeConversationGenerating || isTranscribing || isRecordingRef.current) return;
    if (!navigator.mediaDevices?.getUserMedia) {
      setError("Microphone capture is not supported in this browser.");
      return;
    }

    stopSpeechPlayback();
    setError("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : MediaRecorder.isTypeSupported("audio/webm")
          ? "audio/webm"
          : undefined;
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      mediaStreamRef.current = stream;
      mediaChunksRef.current = [];
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          mediaChunksRef.current.push(event.data);
        }
      };
      recorder.start();
      mediaRecorderRef.current = recorder;
      isRecordingRef.current = true;
      setIsRecording(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Microphone access denied");
    }
  }

  function toggleVoiceCapture() {
    if (isRecordingRef.current) {
      void stopRecordingAndSend();
      return;
    }
    void startVoiceCapture();
  }

  async function stopRecordingAndSend() {
    if (!isRecordingRef.current) return;

    isRecordingRef.current = false;
    setIsRecording(false);
    const blob = await stopMediaRecorder();
    if (!blob) {
      setError("No audio captured.");
      return;
    }

    setIsTranscribing(true);
    setError("");
    let transcribingMessageId: number | null = null;
    try {
      transcribingMessageId = appendOptimisticUserMessage(TRANSCRIBING_PLACEHOLDER);
      const text = (await transcribeAudio(blob)).trim();
      if (!text) {
        if (transcribingMessageId != null) {
          removeOptimisticMessage(transcribingMessageId);
        }
        setError("No speech detected.");
        return;
      }
      if (transcribingMessageId != null) {
        updateOptimisticMessageContent(transcribingMessageId, text);
      } else {
        appendOptimisticUserMessage(text);
      }
      setInput(text);
      const initialMessageCount = persistedMessageCount(detail?.messages);
      await submitMessage(text, {
        speakReply: true,
        userMessageAlreadyShown: true,
        initialMessageCount,
      });
    } catch (err) {
      if (transcribingMessageId != null) {
        removeOptimisticMessage(transcribingMessageId);
      }
      setError(err instanceof Error ? err.message : "Could not transcribe speech");
      if (activeId) {
        await loadConversation(activeId);
      }
    } finally {
      setIsTranscribing(false);
    }
  }

  async function sendMessage(event: FormEvent) {
    event.preventDefault();
    if (isRecordingRef.current) {
      await stopRecordingAndSend();
      return;
    }
    await submitMessage();
  }

  async function submitMessage(
    overrideContent?: string,
    options?: {
      speakReply?: boolean;
      image?: PendingImage | null;
      userMessageAlreadyShown?: boolean;
      initialMessageCount?: number;
    },
  ) {
    const image = options?.image === undefined ? pendingImage : options.image;
    const content = (overrideContent ?? input).trim();
    const conversationId = activeId;
    if (!conversationId || conversationId in generationJobs || (!content && !image)) return;

    const conversationTitle =
      detail?.conversation.title ??
      conversations.find((conversation) => conversation.id === conversationId)?.title ??
      "Conversation";
    const modelName = isMultiAgentConversation
      ? multiAgentOverrideModel
        ? `${activeParticipants.length} agents → ${multiAgentOverrideModel.model}`
        : `${activeParticipants.length} agents`
      : activeModelName;
    const totalSteps = isMultiAgentConversation
      ? activeParticipants.length * Math.max(1, Math.min(10, discussionRounds))
      : 1;
    const roundsForThisSend = isMultiAgentConversation
      ? Math.max(1, Math.min(10, discussionRounds))
      : 1;
    const initialMessageCount =
      options?.initialMessageCount ?? persistedMessageCount(detail?.messages);
    const abortController = new AbortController();
    generationAbortByConversationRef.current.set(conversationId, abortController);
    dismissedProgressModalOnPlaybackRef.current.delete(conversationId);

    setGenerationJobs((current) => ({
      ...current,
      [conversationId]: {
        conversationId,
        conversationTitle,
        modelName,
        contextPreview: null,
        startedAt: Date.now(),
        isMultiAgent: isMultiAgentConversation,
        totalSteps,
        completedSteps: 0,
        currentParticipantName: isMultiAgentConversation
          ? participantDisplayName(activeParticipants[0], config?.models ?? [])
          : undefined,
        initialMessageCount,
        speakReplies: options?.speakReply ?? false,
        autoPlayReplies: isMultiAgentConversation,
        participants: isMultiAgentConversation ? activeParticipants : undefined,
        rounds: roundsForThisSend,
        overrideLlmModelId: isMultiAgentConversation ? multiAgentModelOverrideId : undefined,
        singleModelName: isMultiAgentConversation ? undefined : activeModelName,
      },
    }));
    setProgressModalConversationId(conversationId);
    setGenerationClock(Date.now());

    if (content || image) {
      if (isMultiAgentConversation) {
        if (!options?.userMessageAlreadyShown) {
          appendOptimisticUserMessage(content || "Image message");
        }
      } else if (options?.userMessageAlreadyShown) {
        appendOptimisticAssistantPlaceholder();
      } else {
        appendOptimisticExchange(content || "Image message");
      }
    }
    setInput("");
    clearPendingImage();
    setError("");
    setNotice("");

    if (isMultiAgentConversation || options?.speakReply) {
      beginAssistantReplyPlayback(conversationId, initialMessageCount);
    }
    if (options?.speakReply) {
      speakRepliesByConversationRef.current[conversationId] = true;
    }

    try {
      const payload: {
        content: string;
        image_data?: string;
        image_media_type?: string;
        include_history: boolean | number;
        include_memories: boolean;
        include_all_memories: boolean;
        discussion_rounds?: number;
        answer_length: number;
        override_llm_model_id?: number;
      } = {
        content,
        include_history: historyLimitToIncludeHistory(
          composerHistoryContextTotal,
          historyContextOverhead,
          historyDbMessageCount,
        ),
        include_memories: includeMemories,
        include_all_memories: includeAllMemories,
        answer_length: answerLength,
      };
      if (isMultiAgentConversation) {
        payload.discussion_rounds = roundsForThisSend;
        if (multiAgentModelOverrideId != null) {
          payload.override_llm_model_id = multiAgentModelOverrideId;
        }
      }
      if (image) {
        payload.image_data = image.base64;
        payload.image_media_type = image.mediaType;
      }

      const requestOptions = { signal: abortController.signal };

      try {
        const preview = await request<LlmContextPreview>(`/api/conversations/${conversationId}/llm-context-preview`, {
          method: "POST",
          body: JSON.stringify(payload),
          ...requestOptions,
        });
        updateGenerationJob(conversationId, { contextPreview: preview });
      } catch (err) {
        if (isAbortError(err)) throw err;
        updateGenerationJob(conversationId, { contextPreview: null });
      }

      const response = await request<{
        user_message?: Message;
        assistant_message?: Message;
        assistant_messages?: Message[];
        memory?: Memory;
      }>(`/api/conversations/${conversationId}/messages`, {
        method: "POST",
        body: JSON.stringify(payload),
        ...requestOptions,
      });

      await loadConversation(conversationId);
      removeGenerationJob(conversationId);
      await loadConversations();
      speakRepliesByConversationRef.current[conversationId] = false;

      if (isMultiAgentConversation && roundsForThisSend > 1) {
        setDiscussionRounds(1);
        localStorage.setItem(DISCUSSION_ROUNDS_STORAGE_KEY, "1");
      }

      if (activeIdRef.current === conversationId) {
        if (response.memory) {
          applyRememberedMemory(response.memory);
          setNotice("Saved to this conversation's memory bank.");
        }
      } else {
        setCompletedGenerationAlert({
          conversationId,
          title: conversationTitle,
        });
      }
    } catch (err) {
      removeGenerationJob(conversationId);
      speakRepliesByConversationRef.current[conversationId] = false;
      delete speakReplyStartIndexByConversationRef.current[conversationId];
      delete spokenMessageIdsByConversationRef.current[conversationId];
      if (isAbortError(err)) {
        if (activeIdRef.current === conversationId) {
          setNotice("Generation cancelled.");
        }
        await loadConversation(conversationId);
      } else {
        const message = err instanceof Error ? err.message : "Could not send message";
        if (activeIdRef.current === conversationId) {
          setError(message);
        } else {
          setCompletedGenerationAlert({
            conversationId,
            title: conversationTitle,
            error: message,
          });
        }
        await loadConversation(conversationId);
      }
    }
  }

  function handleComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (isRecordingRef.current && (event.key === " " || event.key === "Enter")) {
      event.preventDefault();
      void stopRecordingAndSend();
      return;
    }
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void submitMessage();
    }
  }

  async function pollMemoryTitleUpdate(memoryId: number, conversationId: number) {
    for (let attempt = 0; attempt < 20; attempt += 1) {
      await sleep(2000);
      try {
        const memories = await request<Memory[]>(`/api/conversations/${conversationId}/memories`);
        const updated = memories.find((memory) => memory.id === memoryId);
        if (!updated || !updated.title_pending) {
          if (updated) {
            setDetail((current) =>
              current?.conversation.id === conversationId
                ? {
                    ...current,
                    memories: current.memories.map((memory) => (memory.id === memoryId ? updated : memory)),
                  }
                : current,
            );
            if (page === "memories") {
              await loadMemoryGroups();
            }
          }
          return;
        }
      } catch {
        return;
      }
    }
  }

  function applyRememberedMemory(memory: Memory, optimisticId?: number) {
    setDetail((current) =>
      current
        ? {
            ...current,
            memories: [
              memory,
              ...current.memories.filter(
                (entry) =>
                  entry.id !== optimisticId &&
                  entry.source_message_id !== memory.source_message_id,
              ),
            ],
          }
        : current,
    );
    if (memory.title_pending) {
      void pollMemoryTitleUpdate(memory.id, memory.conversation_id);
    }
  }

  async function rememberMessage(message: Message) {
    if (!activeId || rememberedMessageIds.has(message.id)) return;
    const conversationId = activeId;
    const optimisticMemory = buildOptimisticMemory(message, conversationId);
    setError("");
    applyRememberedMemory(optimisticMemory);
    setNotice("Saving to memory...");

    try {
      const memory = await request<Memory>(`/api/conversations/${conversationId}/remember`, {
        method: "POST",
        body: JSON.stringify({ message_id: message.id }),
      });
      applyRememberedMemory(memory, optimisticMemory.id);
      setNotice("Message saved to memory.");
    } catch (err) {
      setDetail((current) =>
        current
          ? {
              ...current,
              memories: current.memories.filter((entry) => entry.id !== optimisticMemory.id),
            }
          : current,
      );
      setError(err instanceof Error ? err.message : "Could not save message to memory");
    }
  }

  function openMessageContextMenu(event: ReactMouseEvent<HTMLDivElement>, message: Message) {
    event.preventDefault();
    setSelectedClip(null);
    setMessageContextMenu({
      message,
      x: Math.max(8, Math.min(event.clientX, window.innerWidth - 190)),
      y: Math.max(8, Math.min(event.clientY, window.innerHeight - 100)),
    });
  }

  function memorizeContextMessage(message: Message) {
    setMessageContextMenu(null);
    void rememberMessage(message);
  }

  async function deleteMessage(message: Message) {
    setMessageContextMenu(null);
    setError("");
    try {
      await request(`/api/conversations/${message.conversation_id}/messages/${message.id}`, { method: "DELETE" });
      if (activeId) {
        await loadConversation(activeId);
      }
      await loadConversations();
      setNotice("Message deleted.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not delete message");
    }
  }

  function handleMessageSelection(message: Message) {
    if (message.role !== "assistant") return;
    const selection = window.getSelection();
    const content = getSelectedTextInMessage(message.id);
    if (!selection || !content || selection.rangeCount === 0) {
      setSelectedClip(null);
      return;
    }

    const range = selection.getRangeAt(0);
    const rect = range.getBoundingClientRect();
    if (!rect.width && !rect.height) {
      setSelectedClip(null);
      return;
    }

    setSelectedClip({
      content,
      messageId: message.id,
      x: Math.min(rect.right + 8, window.innerWidth - 170),
      y: Math.max(rect.top - 6, 8),
    });
  }

  async function rememberSelectedClip() {
    if (!activeId || !selectedClip) return;
    const conversationId = activeId;
    const optimisticMemory: Memory = {
      id: -selectedClip.messageId,
      conversation_id: conversationId,
      title: buildFallbackMemoryTitle(selectedClip.content),
      content: selectedClip.content,
      source_message_id: selectedClip.messageId,
      created_at: new Date().toISOString(),
      title_pending: true,
    };
    setError("");
    applyRememberedMemory(optimisticMemory);
    setNotice("Saving clip to memory...");

    try {
      const memory = await request<Memory>(`/api/conversations/${conversationId}/remember`, {
        method: "POST",
        body: JSON.stringify({ content: selectedClip.content, message_id: selectedClip.messageId }),
      });
      applyRememberedMemory(memory, optimisticMemory.id);
      setNotice("Selected clip saved to memory.");
      setSelectedClip(null);
      window.getSelection()?.removeAllRanges();
    } catch (err) {
      setDetail((current) =>
        current
          ? {
              ...current,
              memories: current.memories.filter((entry) => entry.id !== optimisticMemory.id),
            }
          : current,
      );
      setError(err instanceof Error ? err.message : "Could not save selected clip");
    }
  }

  async function deleteMemory(memory: Memory) {
    await request(`/api/conversations/${memory.conversation_id}/memories/${memory.id}`, { method: "DELETE" });
    setSelectedMemoryIds((current) => current.filter((id) => id !== memory.id));
    await loadMemoryGroups();
    if (activeId === memory.conversation_id) {
      await loadConversation(activeId);
    }
  }

  async function mergeMemories() {
    const targetConversationId = mergeTargetConversationId || activeId;
    if (!targetConversationId || selectedMemoryIds.length < 2) return;
    const content = mergeText.trim() || selectedMemories.map((memory) => memory.content).join("\n\n");
    if (!content) return;

    setError("");
    try {
      await request<Memory>("/api/memories/merge", {
        method: "POST",
        body: JSON.stringify({
          memory_ids: selectedMemoryIds,
          content,
          target_conversation_id: targetConversationId,
        }),
      });
      setSelectedMemoryIds([]);
      setMergeText("");
      setNotice("Merged selected memories.");
      await loadMemoryGroups();
      if (activeId) {
        await loadConversation(activeId);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not merge selected memories");
    }
  }

  async function integrateMemories() {
    if (selectedMemoryIds.length < 2) return;

    setError("");
    try {
      await request<Memory>("/api/memories/integrate", {
        method: "POST",
        body: JSON.stringify({ memory_ids: selectedMemoryIds }),
      });
      setSelectedMemoryIds([]);
      setMergeText("");
      setNotice(`Integrated selected memories in ${oldestSelectedConversation?.title ?? "the oldest discussion"}.`);
      await loadMemoryGroups();
      if (activeId) {
        await loadConversation(activeId);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not integrate selected memories");
    }
  }

  async function moveMemoryToConversation(memoryId: number, targetConversationId: number) {
    const memory = allMemories.find((item) => item.id === memoryId);
    if (!memory || memory.conversation_id === targetConversationId) return;
    await request<Memory>(`/api/memories/${memoryId}/move`, {
      method: "PUT",
      body: JSON.stringify({ target_conversation_id: targetConversationId }),
    });
    setNotice("Memory moved.");
    setExpandedMemoryGroupIds((current) =>
      current.includes(targetConversationId) ? current : [...current, targetConversationId],
    );
    await loadMemoryGroups();
    if (activeId && [memory.conversation_id, targetConversationId].includes(activeId)) {
      await loadConversation(activeId);
    }
  }

  async function saveConfig(event: FormEvent) {
    event.preventDefault();
    setError("");
    const activeIndex = configDraft.findIndex((model) => model.is_active);
    const models = configDraft.map((model, index) => {
      const payload: Record<string, string | number | boolean | undefined> = {
        id: model.id,
        provider: model.provider,
        base_url: model.base_url,
        model: model.model,
        comments: model.comments?.trim() || undefined,
        tts_voice_uri: model.tts_voice_uri?.trim() || undefined,
        is_active: activeIndex === -1 ? index === 0 : model.is_active,
        clear_api_key: Boolean(model.clear_api_key),
      };
      if (model.api_key?.trim()) {
        payload.api_key = model.api_key.trim();
      }
      return payload;
    });

    try {
      const saved = await request<LlmConfig>("/api/config/llm", {
        method: "PUT",
        body: JSON.stringify({ models }),
      });
      setConfig(saved);
      setConfigDraft(
        saved.models.map((model) => ({
          ...model,
          api_key: "",
          clear_api_key: false,
        })),
      );
      showSettingsSaveSuccess("LLM settings saved.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not save LLM settings");
    }
  }

  function openAddModelModal() {
    setAddModelDraft(createAddModelDraft(configDraft.length === 0));
    setIsAddModelModalOpen(true);
  }

  function cancelAddModelModal() {
    setIsAddModelModalOpen(false);
    setAddModelDraft(createAddModelDraft(configDraft.length === 0));
  }

  function changeAddModelProvider(provider: ProviderKey) {
    const defaults = providerDefaults(provider);
    setAddModelDraft((current) => {
      const previousDefaults = providerDefaults(current.provider);
      return {
        ...current,
        provider,
        base_url: !current.base_url || current.base_url === previousDefaults.base_url ? defaults.base_url : current.base_url,
        model: !current.model || current.model === previousDefaults.model ? defaults.model : current.model,
      };
    });
  }

  function saveAddModelModal(event: FormEvent) {
    event.preventDefault();
    const baseUrl = addModelDraft.base_url.trim();
    const modelName = addModelDraft.model.trim();
    if (!baseUrl || !modelName) {
      setError("Provider address and model name are required.");
      return;
    }

    const newModel: LlmModel = {
      ...addModelDraft,
      base_url: baseUrl,
      model: modelName,
      comments: addModelDraft.comments?.trim() || "",
      api_key: addModelDraft.api_key?.trim() || "",
    };

    setConfigDraft((current) => {
      const next = [...current, newModel];
      if (newModel.is_active) {
        return next.map((model, index) => ({
          ...model,
          is_active: index === next.length - 1,
        }));
      }
      return next;
    });
    setIsAddModelModalOpen(false);
    setAddModelDraft(createAddModelDraft(false));
    setError("");
  }

  function changeModelProvider(index: number, provider: ProviderKey) {
    const defaults = providerDefaults(provider);
    setConfigDraft((current) =>
      current.map((model, currentIndex) => {
        if (currentIndex !== index) return model;
        const previousDefaults = providerDefaults(model.provider);
        return {
          ...model,
          provider,
          base_url: !model.base_url || model.base_url === previousDefaults.base_url ? defaults.base_url : model.base_url,
          model: !model.model || model.model === previousDefaults.model ? defaults.model : model.model,
        };
      }),
    );
  }

  function updateModelDraft(index: number, updates: Partial<LlmModel>) {
    setConfigDraft((current) =>
      current.map((model, currentIndex) => (currentIndex === index ? { ...model, ...updates } : model)),
    );
  }

  function setActiveModelDraft(index: number) {
    setConfigDraft((current) =>
      current.map((model, currentIndex) => ({ ...model, is_active: currentIndex === index })),
    );
  }

  function removeModelRow(index: number) {
    setConfigDraft((current) => {
      const next = current.filter((_, currentIndex) => currentIndex !== index);
      if (next.length && !next.some((model) => model.is_active)) {
        next[0] = { ...next[0], is_active: true };
      }
      return next;
    });
  }

  function toggleMemory(memoryId: number) {
    setSelectedMemoryIds((current) =>
      current.includes(memoryId) ? current.filter((id) => id !== memoryId) : [...current, memoryId],
    );
  }

  function toggleExpandedMemory(memoryId: number) {
    setExpandedMemoryIds((current) =>
      current.includes(memoryId) ? current.filter((id) => id !== memoryId) : [...current, memoryId],
    );
  }

  function toggleExpandedMemoryGroup(conversationId: number) {
    setExpandedMemoryGroupIds((current) =>
      current.includes(conversationId) ? current.filter((id) => id !== conversationId) : [...current, conversationId],
    );
  }

  function openConversationMemories(conversationId: number) {
    setExpandedMemoryGroupIds((current) =>
      current.includes(conversationId) ? current : [...current, conversationId],
    );
    const cachedMemoryIds =
      conversationId === activeId
        ? (detail?.memories ?? []).map((memory) => memory.id)
        : (memoryGroups.find((group) => group.conversation.id === conversationId)?.memories ?? []).map(
            (memory) => memory.id,
          );
    if (cachedMemoryIds.length) {
      setExpandedMemoryIds(cachedMemoryIds);
    } else {
      expandConversationMemoriesRef.current = conversationId;
    }
    setPage("memories");
  }

  function openDiscussion(conversationId: number) {
    setActiveId(conversationId);
    setPage("chat");
    setSelectedMemoryIds([]);
    setMergeText("");
  }

  function sortMemoriesBy(sort: "created_at" | "title" | "llm_model") {
    setMemorySort((currentSort) => {
      if (currentSort === sort) {
        setMemoryOrder((currentOrder) => (currentOrder === "asc" ? "desc" : "asc"));
        return currentSort;
      }
      setMemoryOrder(sort === "created_at" ? "desc" : "asc");
      return sort;
    });
  }

  function exportDiscussionMemoriesToPdf(group: MemoryGroup) {
    const sourceGroup = memoryGroups.find((candidate) => candidate.conversation.id === group.conversation.id) ?? group;
    const selectedInGroup = sortMemoryList(
      sourceGroup.memories.filter((memory) => selectedMemoryIds.includes(memory.id)),
    );
    const memoriesToExport = selectedInGroup.length ? selectedInGroup : sortMemoryList(sourceGroup.memories);
    if (!memoriesToExport.length) return;

    setPrintMemoryExport({
      conversation: sourceGroup.conversation,
      memories: memoriesToExport,
      scope: selectedInGroup.length ? "selected" : "all",
    });
    requestAnimationFrame(() => {
      requestAnimationFrame(() => window.print());
    });
  }

  function adjacentUserMessageId(messageId: number, direction: "previous" | "next") {
    const index = userMessageIds.indexOf(messageId);
    if (index === -1) return null;
    return userMessageIds[direction === "previous" ? index - 1 : index + 1] ?? null;
  }

  function scrollToUserMessage(messageId: number, direction: "previous" | "next") {
    const targetId = adjacentUserMessageId(messageId, direction);
    if (!targetId) return;
    document.getElementById(`message-${targetId}`)?.scrollIntoView({
      block: "start",
      behavior: "smooth",
    });
  }

  function fillMergeText() {
    setMergeText(selectedMemories.map((memory) => memory.content).join("\n\n"));
  }

  async function goToLatestDiscussion() {
    if (!latestConversation) return;

    pendingLatestScrollRef.current = true;
    setPage("chat");
    setMessageContextMenu(null);
    setSelectedClip(null);

    if (activeId === latestConversation.id) {
      await loadConversation(latestConversation.id);
      return;
    }

    setActiveId(latestConversation.id);
  }

  return (
    <div className={`app-shell ${conversationPaneCollapsed ? "conversation-pane-collapsed" : ""}`}>
      <header className="top-nav">
        <div className="brand">
          <Sparkles size={22} />
          <span>Sovereign Gen AI</span>
        </div>
        <nav className="top-nav-menu" aria-label="Main navigation">
          <button
            type="button"
            className={page === "chat" ? "nav-active" : ""}
            title={latestConversation ? `Open latest: ${latestConversation.title}` : "No conversations yet"}
            disabled={!latestConversation}
            onClick={() => void goToLatestDiscussion()}
          >
            <MessageCircle size={18} />
            Latest Chat
          </button>
          <button className={page === "memories" ? "nav-active" : ""} onClick={() => setPage("memories")}>
            <Brain size={18} />
            Memories
          </button>
          <button className={page === "settings" ? "nav-active" : ""} onClick={() => setPage("settings")}>
            <Settings size={18} />
            Config
          </button>
        </nav>
        <div className="top-nav-actions">
          <button
            type="button"
            className="top-nav-new"
            aria-label="New conversation"
            title="New conversation"
            onClick={() => openNewConversationModal()}
          >
            <Plus size={18} strokeWidth={2.75} />
          </button>
        </div>
      </header>

      <main className="workspace">
        {notice && <div className="notice">{notice}</div>}
        {error && <div className="error">{error}</div>}
        {page === "chat" && (
          <section className="chat-panel">
            <header className="page-header">
              <div>
                <p className="eyebrow">Conversation</p>
                <div className="conversation-title-row">
                  {activeConversation && editingConversationId === activeConversation.id ? (
                    <input
                      className="conversation-header-title-input"
                      autoFocus
                      value={editingConversationTitle}
                      onChange={(event) => setEditingConversationTitle(event.target.value)}
                      onBlur={(event) => void saveConversationTitle(activeConversation, event.currentTarget.value)}
                      onFocus={(event) => event.currentTarget.select()}
                      onKeyDown={(event) => handleConversationTitleKeyDown(event, activeConversation)}
                      aria-label="Conversation title"
                    />
                  ) : (
                    <>
                      <h1
                        className={activeConversation ? "conversation-header-title" : undefined}
                        title={activeConversation ? "Click to rename" : undefined}
                        onClick={
                          activeConversation
                            ? () => startEditingConversationTitle(activeConversation)
                            : undefined
                        }
                      >
                        {activeConversation?.title ?? "Start a conversation"}
                      </h1>
                      {activeConversation && (
                        <button
                          type="button"
                          className="conversation-header-rename"
                          title="Rename conversation"
                          aria-label="Rename conversation"
                          onMouseDown={(event) => event.preventDefault()}
                          onClick={() => startEditingConversationTitle(activeConversation)}
                        >
                          <Pencil size={16} />
                        </button>
                      )}
                    </>
                  )}
                  {activeConversation && editingConversationId !== activeConversation.id && (
                    <>
                      {isMultiAgentConversation && (
                        <button
                          type="button"
                          className="conversation-header-participants"
                          title="Edit discussion participants"
                          onClick={openParticipantsModal}
                          disabled={activeConversationGenerating}
                        >
                          <Sparkles size={16} />
                          Participants
                        </button>
                      )}
                      <button
                        type="button"
                        className="conversation-header-memories"
                        title="View memories for this conversation"
                        aria-label="View memories for this conversation"
                        onClick={() => openConversationMemories(activeConversation.id)}
                      >
                        <Brain size={18} />
                      </button>
                      <button
                        type="button"
                        className="conversation-header-delete"
                        title="Delete conversation"
                        onClick={() => void openDeleteConversationModal(activeConversation)}
                      >
                        <Trash2 size={18} />
                      </button>
                    </>
                  )}
                </div>
              </div>
            </header>

            <div className="messages">
              {!chatMessages.length && (
                <div className="empty-state">
                  <Brain size={32} />
                  <h2>No messages yet</h2>
                  <p>Create or select a conversation, then ask something. Memories stay scoped here.</p>
                </div>
              )}
              {chatMessages.map((message) => {
                const isRemembered = rememberedMessageIds.has(message.id);
                const isThinkingPlaceholder =
                  message.role === "assistant" && message.id < 0 && !message.content && activeConversationGenerating;
                const isTranscribingPlaceholder =
                  message.role === "user" && message.content.trim() === TRANSCRIBING_PLACEHOLDER;
                const isSpeaking = speakingMessageId === message.id;
                const speakerLabel = messageSpeakerLabel(
                  message,
                  activeParticipants,
                  config?.models ?? [],
                );
                const participant =
                  message.participant_id != null
                    ? activeParticipants.find((item) => item.id === message.participant_id)
                    : undefined;
                return (
                  <article
                    id={`message-${message.id}`}
                    key={message.id}
                    className={`message ${message.role}${isSpeaking ? " message-being-read" : ""}`}
                  >
                    <div className="message-meta">
                      <strong title={participant?.personality?.trim() || undefined}>{speakerLabel}</strong>
                      <span>{formatDate(message.created_at)}</span>
                      {message.role === "assistant" &&
                        (message.llm_model ||
                          message.generation_ms != null ||
                          assistantUsedCurrentMessageOnly(message)) && (
                        <span className="message-model-meta">
                          {message.llm_model && <span className="message-model-pill">{message.llm_model}</span>}
                          {assistantUsedCurrentMessageOnly(message) && (
                            <span
                              className="message-context-pill"
                              title="Generated without prior chat history in the LLM request"
                            >
                              Current message only
                            </span>
                          )}
                          {message.generation_ms != null && (
                            <span className="message-generation-pill" title="Time to generate this reply">
                              {formatGenerationDuration(message.generation_ms)}
                            </span>
                          )}
                        </span>
                      )}
                      <div className="message-meta-actions">
                        {message.role === "assistant" && !isThinkingPlaceholder && (
                          <>
                            {isSpeaking ? (
                              <button
                                type="button"
                                className="message-speech-active"
                                title="Stop playback"
                                onClick={stopSpeechPlayback}
                              >
                                <Square size={15} />
                              </button>
                            ) : (
                              <button
                                type="button"
                                title="Play message (selection only when text is selected)"
                                onMouseDown={(event) => {
                                  event.preventDefault();
                                  captureSpeechSelection(message);
                                }}
                                onClick={() => handlePlayAssistantMessage(message)}
                              >
                                <Play size={15} />
                              </button>
                            )}
                            <label
                              className="message-speech-rate"
                              title="Playback speed"
                              onClick={(event) => event.stopPropagation()}
                            >
                              <input
                                type="range"
                                min={TTS_RATE_MIN}
                                max={TTS_RATE_MAX}
                                step={0.05}
                                value={ttsSpeechRate}
                                onInput={(event) => handleTtsSpeechRateChange(Number(event.currentTarget.value))}
                              />
                              <span>{ttsSpeechRate.toFixed(1)}×</span>
                            </label>
                          </>
                        )}
                        <button
                          className={isRemembered ? "remembered-message-button" : ""}
                          title={isRemembered ? "This message is in memory" : "Remember this message"}
                          onClick={() => rememberMessage(message)}
                          disabled={isThinkingPlaceholder || isTranscribingPlaceholder}
                        >
                          <Brain size={15} />
                        </button>
                        <button
                          type="button"
                          className="message-delete-button"
                          title="Delete message"
                          onClick={() => void deleteMessage(message)}
                          disabled={isThinkingPlaceholder || isTranscribingPlaceholder}
                        >
                          <Trash2 size={15} />
                        </button>
                      </div>
                    </div>
                    <div
                      className="message-content"
                      onContextMenu={(event) => openMessageContextMenu(event, message)}
                      onMouseUp={() => handleMessageSelection(message)}
                    >
                      {isThinkingPlaceholder || isTranscribingPlaceholder ? (
                        <div className="message-thinking">
                          {isTranscribingPlaceholder ? "Transcribing..." : "Thinking..."}
                        </div>
                      ) : (
                        <>
                          {messageImageSrc(message) && (
                            <img className="message-image" src={messageImageSrc(message) ?? undefined} alt="Uploaded attachment" />
                          )}
                          {message.content.trim() && (
                            <ReactMarkdown remarkPlugins={MARKDOWN_REMARK_PLUGINS} rehypePlugins={MARKDOWN_REHYPE_PLUGINS}>
                              {message.content}
                            </ReactMarkdown>
                          )}
                        </>
                      )}
                    </div>
                    {message.role === "user" && (
                      <div className="message-jump-controls">
                        <button
                          type="button"
                          title="Jump to previous message from you"
                          disabled={!adjacentUserMessageId(message.id, "previous")}
                          onClick={() => scrollToUserMessage(message.id, "previous")}
                        >
                          <ArrowUp size={15} />
                        </button>
                        <button
                          type="button"
                          title="Jump to next message from you"
                          disabled={!adjacentUserMessageId(message.id, "next")}
                          onClick={() => scrollToUserMessage(message.id, "next")}
                        >
                          <ArrowDown size={15} />
                        </button>
                      </div>
                    )}
                  </article>
                );
              })}
            </div>
            {messageContextMenu && (
              <div
                className="message-context-menu"
                style={{ left: messageContextMenu.x, top: messageContextMenu.y }}
                onMouseDown={(event) => event.stopPropagation()}
              >
                <button type="button" onClick={() => memorizeContextMessage(messageContextMenu.message)}>
                  <Brain size={15} />
                  Memorize
                </button>
                <button type="button" className="danger-menu-item" onClick={() => void deleteMessage(messageContextMenu.message)}>
                  <Trash2 size={15} />
                  Delete message
                </button>
              </div>
            )}

            <form className="composer" onSubmit={sendMessage}>
              {activeGenerationJob && (
                <button
                  type="button"
                  className="composer-generating-banner"
                  onClick={() => activeId != null && setProgressModalConversationId(activeId)}
                  title="View generation progress"
                >
                  <Loader2 size={16} className="spin" aria-hidden="true" />
                  <span className="composer-generating-banner-text">
                    {activeGenerationJob.isMultiAgent && activeGenerationJob.totalSteps
                      ? `Generating · ${Math.min(
                          activeGenerationJob.completedSteps ?? 0,
                          activeGenerationJob.totalSteps,
                        )}/${activeGenerationJob.totalSteps} agents`
                      : `Generating with ${activeGenerationJob.modelName}`}
                  </span>
                  <span className="composer-generating-banner-action">View</span>
                </button>
              )}
              {pendingImage && (
                <div className="composer-attachment">
                  <img src={pendingImage.previewUrl} alt={pendingImage.name ?? "Attached image"} />
                  <div className="composer-attachment-meta">
                    <strong>{pendingImage.name ?? "Attached image"}</strong>
                    <span>Paste or upload one image per message</span>
                  </div>
                  <button type="button" className="composer-attachment-remove" onClick={clearPendingImage} title="Remove image">
                    <X size={16} />
                  </button>
                </div>
              )}
              <textarea
                placeholder={
                  isRecording
                    ? "Listening… click mic again, Send, or Space when done"
                    : isTranscribing
                      ? "Transcribing speech..."
                      : activeId
                        ? "Ask something, paste an image, or type 'remember this message'..."
                        : "Create a conversation first..."
                }
                value={input}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={handleComposerKeyDown}
                onPaste={handleComposerPaste}
                disabled={!activeId || activeConversationGenerating || isTranscribing}
              />
              <div className="composer-actions">
                <div className="composer-send-cluster">
                  <div className="composer-send-row">
                    <button
                      type="button"
                      className={`composer-mic ${isRecording ? "composer-mic-recording" : ""} ${
                        !isRecording && (isTranscribing || activeConversationGenerating) ? "composer-mic-busy" : ""
                      }`}
                      title={isRecording ? "Click again to stop and send" : "Start voice input"}
                      onClick={toggleVoiceCapture}
                      disabled={!activeId || activeConversationGenerating || isTranscribing}
                    >
                      {isTranscribing ? <Loader2 size={18} className="spin" /> : <Mic size={18} />}
                    </button>
                    <input
                      ref={imageInputRef}
                      type="file"
                      accept="image/jpeg,image/png,image/gif,image/webp"
                      hidden
                      onChange={(event) => void handleImageInputChange(event)}
                    />
                    <button
                      type="button"
                      className="composer-mic"
                      title="Upload an image"
                      onClick={() => imageInputRef.current?.click()}
                      disabled={!activeId || activeConversationGenerating || isTranscribing || isRecording}
                    >
                      <ImagePlus size={18} />
                    </button>
                    <button
                      type="submit"
                      className="composer-send"
                      disabled={!activeId || activeConversationGenerating || isTranscribing || (!isRecording && !input.trim() && !pendingImage)}
                    >
                      {isTranscribing ? "Transcribing..." : activeConversationGenerating ? "Thinking..." : isRecording ? "Stop & send" : "Send"}
                    </button>
                  </div>
                  <div className="composer-model-controls">
                    {isMultiAgentConversation ? (
                      <>
                        <label
                          className={`composer-model-override ${multiAgentModelOverrideId != null ? "is-active" : ""}`}
                          title="Override all agents' models for this discussion until cleared"
                        >
                          <span>Model</span>
                          <select
                            value={multiAgentModelOverrideId ?? ""}
                            disabled={!activeId || activeConversationGenerating || isTranscribing || !config?.models.length}
                            onChange={(event) => {
                              const value = event.target.value;
                              updateMultiAgentModelOverride(value ? Number(value) : null);
                            }}
                          >
                            <option value="">Agent defaults</option>
                            {config?.models.map((model) => (
                              <option key={model.id} value={model.id}>
                                {model.model} ({model.provider})
                              </option>
                            ))}
                          </select>
                        </label>
                        <label className="discussion-rounds-input" title="Autonomous discussion rounds before you can reply again">
                          <span>Rounds</span>
                          <input
                            type="number"
                            min={1}
                            max={10}
                            value={discussionRounds}
                            disabled={!activeId || activeConversationGenerating || isTranscribing}
                            onChange={(event) => {
                              const next = Math.max(1, Math.min(10, Number(event.target.value) || 1));
                              setDiscussionRounds(next);
                              localStorage.setItem(DISCUSSION_ROUNDS_STORAGE_KEY, String(next));
                            }}
                          />
                        </label>
                      </>
                    ) : (
                      <button
                        type="button"
                        className="model-pill composer-model-pill"
                        title="Choose discussion model"
                        onClick={() => setIsConversationModelModalOpen(true)}
                        disabled={!detail || !config?.models.length}
                      >
                        <Bot size={16} />
                        <span>{conversationModel?.model ?? "qwen3.5:9b"}</span>
                      </button>
                    )}
                  </div>
                </div>
              </div>
              <div className="composer-history">
                <div className="composer-history-controls">
                <label
                  className="composer-memory-toggle"
                  title={
                    (detail?.memories.length ?? 0) > 0
                      ? "Include this conversation's saved memories in the LLM request"
                      : "No saved memories in this conversation"
                  }
                >
                  <input
                    type="checkbox"
                    checked={includeMemories}
                    onChange={(event) => updateIncludeMemories(event.target.checked)}
                    disabled={historyControlsDisabled}
                  />
                  Memories
                </label>
                <button
                  type="button"
                  className="composer-history-step"
                  title="Include one fewer prior message"
                  aria-label="Decrease history by one"
                  disabled={historyControlsDisabled || composerHistoryContextTotal <= historySliderMinValue}
                  onClick={() => updateHistoryMessageLimit(composerHistoryContextTotal - 1)}
                >
                  <ChevronLeft size={18} />
                </button>
                <input
                  type="range"
                  min={historySliderMinValue}
                  max={historySliderMaxValue}
                  step={1}
                  value={composerHistoryContextTotal}
                  title={historyLimitTitle(composerHistoryContextTotal, historySliderMinValue, historySliderMaxValue)}
                  onChange={(event) => updateHistoryMessageLimit(Number(event.target.value))}
                  onKeyDown={(event) => {
                    if (event.key === "ArrowLeft" || event.key === "ArrowDown") {
                      event.preventDefault();
                      updateHistoryMessageLimit(composerHistoryContextTotal - 1);
                    } else if (event.key === "ArrowRight" || event.key === "ArrowUp") {
                      event.preventDefault();
                      updateHistoryMessageLimit(composerHistoryContextTotal + 1);
                    }
                  }}
                  disabled={historyControlsDisabled}
                  aria-label="Messages to send in request"
                  aria-valuetext={historyLimitTitle(
                    composerHistoryContextTotal,
                    historySliderMinValue,
                    historySliderMaxValue,
                  )}
                />
                <button
                  type="button"
                  className="composer-history-step"
                  title="Include one more message in request"
                  aria-label="Increase message count by one"
                  disabled={historyControlsDisabled || composerHistoryContextTotal >= historySliderMaxValue}
                  onClick={() => updateHistoryMessageLimit(composerHistoryContextTotal + 1)}
                >
                  <ChevronRight size={18} />
                </button>
                <span className="composer-history-value" aria-hidden="true">
                  {composerHistoryContextTotal}
                </span>
                <label
                  className="composer-memory-toggle"
                  title={
                    otherConversationMemories.length > 0
                      ? "Include memories from all other conversations as user memories in the LLM request"
                      : "No saved memories in other conversations"
                  }
                >
                  <input
                    type="checkbox"
                    checked={includeAllMemories}
                    onChange={(event) => updateIncludeAllMemories(event.target.checked)}
                    disabled={historyControlsDisabled}
                  />
                  Other conversations
                </label>
                </div>
                <label
                  className="composer-answer-length"
                  title={
                    answerLength >= ANSWER_LENGTH_MAX
                      ? "Let the AI choose reply length"
                      : "How long replies should be — highest priority (single and multi-agent chats)"
                  }
                >
                  <span>Answer length</span>
                  <input
                    type="range"
                    min={ANSWER_LENGTH_MIN}
                    max={ANSWER_LENGTH_MAX}
                    step={1}
                    value={answerLength}
                    disabled={historyControlsDisabled}
                    onChange={(event) => updateAnswerLength(Number(event.currentTarget.value))}
                    aria-label="Answer length"
                    aria-valuetext={answerLengthLabel(answerLength)}
                  />
                  <span className="composer-answer-length-value">{answerLengthLabel(answerLength)}</span>
                </label>
              </div>
            </form>
          </section>
        )}

        {page === "memories" && (
          <section className="memory-page">
            <header className="page-header">
              <div>
                <p className="eyebrow">Memory bank</p>
                <h1>{detail?.conversation.title ?? "Select a conversation"}</h1>
              </div>
              <div className="memory-toolbar">
                <input
                  value={memorySearch}
                  onChange={(event) => setMemorySearch(event.target.value)}
                  placeholder="Search memories..."
                />
                <button
                  type="button"
                  onClick={() => setSelectedMemoryIds([])}
                  disabled={!selectedMemoryIds.length}
                >
                  Clear selection
                </button>
                <select value={memoryModelFilter} onChange={(event) => setMemoryModelFilter(event.target.value)}>
                  <option value="">All models</option>
                  {memoryModelOptions.map((model) => (
                    <option key={model} value={model}>
                      {model}
                    </option>
                  ))}
                </select>
                <button
                  type="button"
                  className={memorySort === "title" ? "sort-active" : ""}
                  onClick={() => sortMemoriesBy("title")}
                >
                  Title {memorySort === "title" ? (memoryOrder === "asc" ? "↑" : "↓") : ""}
                </button>
                <button
                  type="button"
                  className={memorySort === "created_at" ? "sort-active" : ""}
                  onClick={() => sortMemoriesBy("created_at")}
                >
                  Date {memorySort === "created_at" ? (memoryOrder === "asc" ? "↑" : "↓") : ""}
                </button>
                <button
                  type="button"
                  className={memorySort === "llm_model" ? "sort-active" : ""}
                  onClick={() => sortMemoriesBy("llm_model")}
                >
                  Model {memorySort === "llm_model" ? (memoryOrder === "asc" ? "↑" : "↓") : ""}
                </button>
              </div>
            </header>

            <div className={`memory-grid ${selectedMemoryIds.length < 2 ? "memory-grid-full" : ""}`}>
              <div className="memory-list">
                {visibleMemoryGroups.map((group) => {
                  const groupIsExpanded =
                    memorySearchTerms.length > 0 || expandedMemoryGroupIds.includes(group.conversation.id);
                  const sourceGroup = memoryGroups.find((candidate) => candidate.conversation.id === group.conversation.id) ?? group;
                  const selectedInGroupCount = sourceGroup.memories.filter((memory) => selectedMemoryIds.includes(memory.id)).length;
                  const exportCount = selectedInGroupCount || sourceGroup.memories.length;
                  return (
                    <section
                      key={group.conversation.id}
                      className={`memory-group ${dragOverMemoryGroupId === group.conversation.id ? "memory-group-drag-over" : ""}`}
                      onDragOver={(event) => {
                        event.preventDefault();
                        setDragOverMemoryGroupId(group.conversation.id);
                      }}
                      onDragLeave={() => setDragOverMemoryGroupId(null)}
                      onDrop={(event) => {
                        event.preventDefault();
                        const droppedMemoryId = draggedMemoryId ?? Number(event.dataTransfer.getData("text/plain"));
                        if (droppedMemoryId) {
                          void moveMemoryToConversation(droppedMemoryId, group.conversation.id);
                        }
                        setDraggedMemoryId(null);
                        setDragOverMemoryGroupId(null);
                      }}
                    >
                      <div className="memory-group-header">
                        <button
                          type="button"
                          className="memory-group-title-button"
                          onClick={() => toggleExpandedMemoryGroup(group.conversation.id)}
                        >
                          <span>{groupIsExpanded ? "▾" : "▸"}</span>
                          <strong>{group.conversation.title}</strong>
                        </button>
                        <button
                          type="button"
                          className="memory-group-open-discussion"
                          title={`Open discussion: ${group.conversation.title}`}
                          aria-label={`Open discussion: ${group.conversation.title}`}
                          onClick={() => openDiscussion(group.conversation.id)}
                        >
                          <MessageCircle size={15} />
                        </button>
                        {activeId === group.conversation.id && <span className="current-discussion-pill">Current</span>}
                        <small>{sourceGroup.memories.length} memories</small>
                        <button
                          type="button"
                          className="memory-export-button"
                          onClick={() => exportDiscussionMemoriesToPdf(group)}
                          disabled={!exportCount}
                          title={
                            selectedInGroupCount
                              ? `Export ${selectedInGroupCount} selected memories from this discussion`
                              : "Export all memories from this discussion"
                          }
                        >
                          <FileDown size={15} />
                          {selectedInGroupCount ? `Export selected (${selectedInGroupCount})` : "Export PDF"}
                        </button>
                      </div>

                      {groupIsExpanded && (
                        <div className="memory-group-items">
                          {group.memories.length === 0 && <div className="empty-card">No memories in this discussion yet.</div>}
                          {group.memories.map((memory) => {
                            const isExpanded = memorySearchTerms.length > 0 || expandedMemoryIds.includes(memory.id);
                            return (
                              <div
                                key={memory.id}
                                className="memory-row"
                                draggable
                                onDragStart={(event) => {
                                  event.dataTransfer.effectAllowed = "move";
                                  event.dataTransfer.setData("text/plain", String(memory.id));
                                  setDraggedMemoryId(memory.id);
                                }}
                                onDragEnd={() => {
                                  setDraggedMemoryId(null);
                                  setDragOverMemoryGroupId(null);
                                }}
                              >
                                <input
                                  type="checkbox"
                                  checked={selectedMemoryIds.includes(memory.id)}
                                  onChange={() => toggleMemory(memory.id)}
                                />
                                <div className="memory-body">
                                  <button
                                    type="button"
                                    className="memory-title-button"
                                    onClick={() => toggleExpandedMemory(memory.id)}
                                  >
                                    <span>{isExpanded ? "▾" : "▸"}</span>
                                    {memory.title_pending ? (
                                      <span className="memory-title-pending">
                                        <Loader2 size={13} className="spin" />
                                        {highlightMatches(memory.title, memorySearchTerms)}
                                      </span>
                                    ) : (
                                      highlightMatches(memory.title, memorySearchTerms)
                                    )}
                                  </button>
                                  <div className="memory-meta-row">
                                    <small className="memory-date">{formatDate(memory.created_at)}</small>
                                    {memory.llm_model && <span className="memory-model-pill">{memory.llm_model}</span>}
                                  </div>
                                  {isExpanded && (
                                    <div className="message-content memory-content">
                                      {memorySearchTerms.length > 0 ? (
                                        <div className="memory-search-content">
                                          {highlightMatches(memory.content, memorySearchTerms)}
                                        </div>
                                      ) : (
                                        <ReactMarkdown
                                          remarkPlugins={MARKDOWN_REMARK_PLUGINS}
                                          rehypePlugins={MARKDOWN_REHYPE_PLUGINS}
                                        >
                                          {memory.content}
                                        </ReactMarkdown>
                                      )}
                                    </div>
                                  )}
                                </div>
                                <button
                                  type="button"
                                  className="memory-delete-button"
                                  onClick={() => deleteMemory(memory)}
                                  title="Delete memory"
                                >
                                  <Trash2 size={15} />
                                </button>
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </section>
                  );
                })}
                {memoryGroups.every((group) => group.memories.length === 0) && (
                  <div className="empty-card">No memories saved yet.</div>
                )}
                {memoryGroups.some((group) => group.memories.length > 0) && visibleMemoryGroups.length === 0 && (
                  <div className="empty-card">No memories match that search.</div>
                )}
              </div>

              {selectedMemoryIds.length >= 2 && (
                <aside className="merge-card">
                  <h2>Memory actions</h2>
                  <p>{selectedMemoryIds.length} selected across discussions</p>

                  <div className="memory-action-section">
                    <h3>Integrate</h3>
                    <p>
                      Keep the source memories and create a new integrated memory in{" "}
                      <strong>{oldestSelectedConversation?.title ?? "the oldest selected discussion"}</strong>.
                    </p>
                    <button type="button" onClick={integrateMemories}>
                      <Check size={16} />
                      Integrate selected
                    </button>
                  </div>

                  <div className="memory-action-section">
                    <h3>Merge</h3>
                    <p>Archive the source memories and save one replacement.</p>
                    <label>
                      Save merged memory in
                      <select
                        value={mergeTargetConversationId}
                        onChange={(event) => setMergeTargetConversationId(Number(event.target.value))}
                      >
                        {conversations.map((conversation) => (
                          <option key={conversation.id} value={conversation.id}>
                            {conversation.title}
                          </option>
                        ))}
                      </select>
                    </label>
                    <button type="button" onClick={fillMergeText}>
                      Use selected text
                    </button>
                    <textarea
                      value={mergeText}
                      onChange={(event) => setMergeText(event.target.value)}
                      placeholder="Edit the merged memory before saving..."
                    />
                    <button type="button" onClick={mergeMemories}>
                      <Check size={16} />
                      Merge selected
                    </button>
                  </div>
                </aside>
              )}
            </div>
          </section>
        )}
        {page === "settings" && (
          <section className="settings-page">
            <header className="page-header settings-page-header">
              <div>
                <p className="eyebrow">Configuration</p>
                <h1>Settings</h1>
                <nav className="settings-tabs" role="tablist" aria-label="Settings sections">
                  <button
                    type="button"
                    role="tab"
                    id="settings-tab-models"
                    aria-selected={settingsTab === "models"}
                    aria-controls="settings-panel-models"
                    className={`settings-tab ${settingsTab === "models" ? "settings-tab-active" : ""}`}
                    onClick={() => setSettingsTab("models")}
                  >
                    <Bot size={16} />
                    Models
                  </button>
                  <button
                    type="button"
                    role="tab"
                    id="settings-tab-agents"
                    aria-selected={settingsTab === "agents"}
                    aria-controls="settings-panel-agents"
                    className={`settings-tab ${settingsTab === "agents" ? "settings-tab-active" : ""}`}
                    onClick={() => setSettingsTab("agents")}
                  >
                    <Sparkles size={16} />
                    Agents
                  </button>
                  <button
                    type="button"
                    role="tab"
                    id="settings-tab-prompt"
                    aria-selected={settingsTab === "prompt"}
                    aria-controls="settings-panel-prompt"
                    className={`settings-tab ${settingsTab === "prompt" ? "settings-tab-active" : ""}`}
                    onClick={() => setSettingsTab("prompt")}
                  >
                    <Brain size={16} />
                    Default prompt
                  </button>
                  <button
                    type="button"
                    role="tab"
                    id="settings-tab-speech"
                    aria-selected={settingsTab === "speech"}
                    aria-controls="settings-panel-speech"
                    className={`settings-tab ${settingsTab === "speech" ? "settings-tab-active" : ""}`}
                    onClick={() => setSettingsTab("speech")}
                  >
                    <Mic size={16} />
                    Speech
                  </button>
                </nav>
              </div>
            </header>
            {settingsTab === "models" && (
            <form
              className="settings-card model-settings-card"
              id="settings-panel-models"
              role="tabpanel"
              aria-labelledby="settings-tab-models"
              onSubmit={saveConfig}
            >
              <div className="model-settings-header">
                <p>Save local Ollama models or external OpenAI-compatible APIs here, then tick the one to use for new chat replies. Assign each model a playback voice to tell them apart when listening.</p>
                <button type="button" onClick={openAddModelModal}>
                  <Plus size={16} />
                  Add model
                </button>
              </div>
              <div className="model-row-list">
                {configDraft.map((model, index) => (
                  <div className="model-row" key={model.id ?? `new-${index}`}>
                    <label className="active-model-check">
                      <input
                        type="checkbox"
                        checked={model.is_active}
                        onChange={() => setActiveModelDraft(index)}
                      />
                      Use
                    </label>
                    <label>
                      Provider
                      <select
                        value={model.provider}
                        onChange={(event) => changeModelProvider(index, event.target.value as ProviderKey)}
                      >
                        <option value="ollama">Ollama local</option>
                        <option value="openai">OpenAI API</option>
                        <option value="openai-compatible">OpenAI-compatible API</option>
                      </select>
                    </label>
                    <label>
                      {providerDefaults(model.provider).addressLabel}
                      <input
                        value={model.base_url}
                        onChange={(event) => updateModelDraft(index, { base_url: event.target.value })}
                        placeholder={providerDefaults(model.provider).base_url}
                      />
                    </label>
                    <label>
                      Model
                      <input
                        value={model.model}
                        onChange={(event) => updateModelDraft(index, { model: event.target.value })}
                        placeholder={providerDefaults(model.provider).model}
                      />
                    </label>
                    <label className="model-row-comments">
                      What this model is for
                      <textarea
                        value={model.comments ?? ""}
                        onChange={(event) => updateModelDraft(index, { comments: event.target.value })}
                        placeholder="e.g. Fast general chat, coding help, or long-form writing"
                        rows={2}
                      />
                    </label>
                    <div className="model-row-voice">
                      <label>
                        Playback voice
                        <select
                          value={model.tts_voice_uri ?? ""}
                          onChange={(event) => updateModelDraft(index, { tts_voice_uri: event.target.value })}
                          disabled={!ttsVoiceOptions.length}
                        >
                          <option value="">Default voice (Speech settings)</option>
                          {ttsVoiceOptions.length === 0 && <option value="">Loading voices...</option>}
                          {ttsVoiceOptions.map((voice) => (
                            <option key={voice.uri} value={voice.uri}>
                              {voice.label}
                            </option>
                          ))}
                        </select>
                      </label>
                      <button
                        type="button"
                        className="speech-sample-button"
                        onClick={() => playModelVoiceSample(model.tts_voice_uri)}
                        disabled={!ttsVoiceOptions.length}
                      >
                        <Play size={16} />
                        Play sample
                      </button>
                    </div>
                    <div className="model-row-timing">
                      <div>
                        <p className="model-row-timing-label">Generation timing estimate</p>
                        {model.generation_sample_count && model.generation_sample_count > 0 ? (
                          <p className="model-row-timing-copy">
                            ~{formatDurationSeconds(model.reference_generation_estimate_sec ?? 0)} typical reply ·{" "}
                            {((model.seconds_per_char ?? 0) * 1000).toFixed(2)} ms/char · avg{" "}
                            {formatDurationSeconds(model.avg_generation_sec ?? 0)} · {model.generation_sample_count}{" "}
                            sample{model.generation_sample_count === 1 ? "" : "s"}
                          </p>
                        ) : (
                          <p className="model-row-timing-copy">No timing data yet. Estimates appear after chat replies.</p>
                        )}
                      </div>
                      <button
                        type="button"
                        className="secondary-button model-row-timing-reset"
                        disabled={!model.id || resettingTimingModelId === model.id || !model.generation_sample_count}
                        onClick={() => model.id && void resetModelGenerationStats(model.id)}
                      >
                        {resettingTimingModelId === model.id ? (
                          <>
                            <Loader2 size={16} className="spin" />
                            Resetting...
                          </>
                        ) : (
                          <>
                            <RotateCcw size={16} />
                            Reset timing
                          </>
                        )}
                      </button>
                    </div>
                    <label className="model-row-api-key">
                      API key
                      <MaskedApiKeyInput
                        value={model.api_key ?? ""}
                        preview={model.api_key_preview}
                        hasApiKey={model.has_api_key}
                        placeholder={
                          model.has_api_key
                            ? "Leave blank to keep current key"
                            : providerDefaults(model.provider).keyPlaceholder
                        }
                        onChange={(api_key) => updateModelDraft(index, { api_key })}
                      />
                    </label>
                    <label className="checkbox-label">
                      <input
                        type="checkbox"
                        checked={Boolean(model.clear_api_key)}
                        onChange={(event) => updateModelDraft(index, { clear_api_key: event.target.checked })}
                      />
                      Clear key
                    </label>
                    <button type="button" className="delete-model-button" onClick={() => removeModelRow(index)} disabled={configDraft.length === 1}>
                      <Trash2 size={16} />
                    </button>
                  </div>
                ))}
              </div>
              <button type="submit">
                <Save size={16} />
                Save settings
              </button>
            </form>
            )}

            {settingsTab === "agents" && (
            <section
              className="settings-card agent-library-card"
              id="settings-panel-agents"
              role="tabpanel"
              aria-labelledby="settings-tab-agents"
            >
              <div className="model-settings-header">
                <div>
                  <h2>Agent library</h2>
                  <p>Saved agents keep their name, personality, model, voice, and speech speed. Click an agent to edit it.</p>
                </div>
                <button
                  type="button"
                  className="secondary-button agent-library-create-button"
                  onClick={openCreateAgentProfile}
                  disabled={!config?.models.length}
                >
                  <Plus size={16} />
                  Create new
                </button>
              </div>
              {!agentProfiles.length && (
                <p className="hint">No saved agents yet. Use Create new or save one from a multi-agent conversation.</p>
              )}
              <div className="agent-library-list">
                {agentProfiles.map((profile) => (
                  <article className="agent-library-item" key={profile.id}>
                    <button
                      type="button"
                      className="agent-library-item-main"
                      title={`Edit ${profile.name}`}
                      onClick={() => openAgentProfileEditor(profile)}
                    >
                      <strong className="agent-library-item-name">{profile.name}</strong>
                      <p className="agent-library-item-model">
                        {profile.llm_model} ({profile.llm_provider})
                      </p>
                      <p
                        className={`agent-library-item-personality${
                          profile.personality.trim() ? "" : " is-empty"
                        }`}
                      >
                        {profile.personality.trim() || "No personality set"}
                      </p>
                      <div className="agent-library-item-meta">
                        <p className="agent-library-item-voice">
                          Voice:{" "}
                          {profile.tts_voice_uri
                            ? ttsVoiceOptions.find((voice) => voice.uri === profile.tts_voice_uri)?.label ??
                              profile.tts_voice_uri
                            : "Default"}
                        </p>
                        <p className="agent-library-item-speed">
                          Speech speed:{" "}
                          {profile.tts_speech_rate != null
                            ? `${profile.tts_speech_rate.toFixed(1)}×`
                            : "Global default"}
                        </p>
                      </div>
                    </button>
                    <button
                      type="button"
                      className="delete-model-button agent-library-item-delete"
                      title={`Delete ${profile.name}`}
                      onClick={(event) => {
                        event.stopPropagation();
                        void deleteAgentProfile(profile.id);
                      }}
                    >
                      <Trash2 size={16} />
                    </button>
                  </article>
                ))}
              </div>
            </section>
            )}

            {settingsTab === "prompt" && (
            <form
              className="settings-card prompt-settings-card"
              id="settings-panel-prompt"
              role="tabpanel"
              aria-labelledby="settings-tab-prompt"
              onSubmit={savePromptConfig}
            >
              <div>
                <h2>Default prompts</h2>
                <p>
                  These system prompts are sent at the start of each reply. Conversation memories are still added
                  automatically when relevant. Per-agent personality and name are appended separately for multi-agent
                  chats.
                </p>
              </div>

              <section className="prompt-settings-section" aria-labelledby="single-agent-prompt-heading">
                <h3 id="single-agent-prompt-heading">Single-agent chats</h3>
                <p>Used for normal one-assistant conversations.</p>
                <label>
                  System prompt
                  <textarea
                    className="prompt-settings-textarea"
                    value={defaultPromptDraft}
                    onChange={(event) => setDefaultPromptDraft(event.target.value)}
                    rows={4}
                  />
                </label>
                <div className="prompt-settings-actions">
                  <button
                    type="button"
                    className="prompt-reset-button"
                    onClick={resetDefaultPrompt}
                    disabled={!promptConfig || defaultPromptDraft === promptConfig.default_prompt_baseline}
                  >
                    <RotateCcw size={16} />
                    Reset to default
                  </button>
                </div>
              </section>

              <section className="prompt-settings-section" aria-labelledby="multi-agent-prompt-heading">
                <h3 id="multi-agent-prompt-heading">Multi-agent discussions</h3>
                <p>
                  Used instead of the single-agent prompt when a conversation has discussion participants. The placeholder{" "}
                  <code>{`{character_name}`}</code> is replaced with each agent&apos;s name; who else is in the discussion is
                  already visible in the labeled transcript below.
                </p>
                <label>
                  Multi-agent system prompt
                  <textarea
                    className="prompt-settings-textarea prompt-settings-textarea-multi"
                    value={defaultMultiAgentPromptDraft}
                    onChange={(event) => setDefaultMultiAgentPromptDraft(event.target.value)}
                    rows={22}
                  />
                </label>
                <div className="prompt-settings-actions">
                  <button
                    type="button"
                    className="prompt-reset-button"
                    onClick={resetDefaultMultiAgentPrompt}
                    disabled={
                      !promptConfig ||
                      defaultMultiAgentPromptDraft === promptConfig.multi_agent_prompt_baseline
                    }
                  >
                    <RotateCcw size={16} />
                    Reset to default
                  </button>
                </div>
              </section>

              <div className="prompt-settings-actions prompt-settings-save-row">
                <button type="submit">
                  <Save size={16} />
                  Save prompts
                </button>
              </div>
            </form>
            )}

            {settingsTab === "speech" && (
            <form
              className="settings-card speech-settings-card"
              id="settings-panel-speech"
              role="tabpanel"
              aria-labelledby="settings-tab-speech"
              onSubmit={saveSpeechConfig}
            >
              <div>
                <h2>Speech settings</h2>
                <p>
                  Set a default playback voice for reading replies aloud. Configure microphone input and ElevenLabs
                  voices below.
                </p>
              </div>

              <section className="speech-settings-section speech-default-section" aria-labelledby="speech-default-heading">
                <h3 id="speech-default-heading">Default playback voice</h3>
                <p>Fallback voice used when a model or agent does not specify its own playback voice.</p>
                <div className="speech-voice-row">
                  <label>
                    Voice
                    <select
                      value={selectedTtsVoiceUri}
                      onChange={(event) => handleTtsVoiceChange(event.target.value)}
                      disabled={!ttsVoiceOptions.length}
                    >
                      {ttsVoiceOptions.length === 0 && <option value="">Loading voices...</option>}
                      {ttsVoiceOptions.map((voice) => (
                        <option key={voice.uri} value={voice.uri}>
                          {voice.label}
                        </option>
                      ))}
                    </select>
                  </label>
                  <button
                    type="button"
                    className="speech-sample-button"
                    onClick={() => playVoiceSample()}
                    disabled={!ttsVoiceOptions.length}
                  >
                    <Play size={16} />
                    Play sample
                  </button>
                </div>
              </section>

              <div className="speech-settings-divider" role="separator" aria-hidden="true" />

              <section className="speech-settings-section" aria-labelledby="speech-input-heading">
                <h3 id="speech-input-heading">Speech-to-text</h3>
                <p>Choose the Whisper model used when you record with the microphone.</p>
                <label>
                  Whisper model
                  <select
                    value={whisperModelDraft}
                    onChange={(event) => setWhisperModelDraft(event.target.value)}
                  >
                    {(speechConfig?.whisper_model_options ?? Object.keys(WHISPER_MODEL_LABELS)).map((model) => (
                      <option key={model} value={model}>
                        {WHISPER_MODEL_LABELS[model] ?? model}
                      </option>
                    ))}
                  </select>
                </label>
              </section>

              <div className="speech-settings-divider" role="separator" aria-hidden="true" />

              <section className="speech-settings-section speech-elevenlabs-section" aria-labelledby="elevenlabs-voices-heading">
                <h3 id="elevenlabs-voices-heading">ElevenLabs voices</h3>
                <p>Connect your ElevenLabs account to import custom and cloned voices into the playback dropdowns.</p>

                <label className="speech-api-key-field">
                  API key
                  <MaskedApiKeyInput
                    value={elevenlabsApiKeyDraft}
                    preview={speechConfig?.elevenlabs_api_key_preview}
                    hasApiKey={Boolean(speechConfig?.has_elevenlabs_api_key)}
                    placeholder={
                      speechConfig?.has_elevenlabs_api_key
                        ? "Leave blank to keep current key"
                        : "Required for ElevenLabs voices"
                    }
                    onChange={setElevenlabsApiKeyDraft}
                  />
                </label>
                <label className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={clearElevenlabsApiKeyDraft}
                    onChange={(event) => setClearElevenlabsApiKeyDraft(event.target.checked)}
                  />
                  Clear ElevenLabs key
                </label>

                <div className="elevenlabs-voices-toolbar">
                  <button
                    type="button"
                    className="elevenlabs-import-button"
                    onClick={() => void importElevenLabsVoices()}
                    disabled={!speechConfig?.has_elevenlabs_api_key || isImportingElevenlabsVoices}
                  >
                    {isImportingElevenlabsVoices ? <Loader2 size={16} className="spin" /> : <Download size={16} />}
                    {isImportingElevenlabsVoices ? "Importing..." : "Import voices from account"}
                  </button>
                </div>

                {!speechConfig?.has_elevenlabs_api_key && (
                  <p className="elevenlabs-voices-hint">Save your ElevenLabs API key to import your voice catalog.</p>
                )}

                {speechConfig?.has_elevenlabs_api_key && elevenlabsCatalogVoices.length === 0 && !isImportingElevenlabsVoices && (
                  <p className="elevenlabs-voices-hint">No voices imported yet. Click import to load your account catalog.</p>
                )}

                {elevenlabsCatalogVoices.length > 0 && (
                  <div className="elevenlabs-voice-table-panel">
                    <div className="elevenlabs-voice-table-meta">
                      <p className="elevenlabs-voices-hint">
                        Showing {displayedElevenlabsCatalogVoices.length} of {elevenlabsCatalogVoices.length} voices
                      </p>
                      {hasActiveElevenlabsVoiceFilters && (
                        <button
                          type="button"
                          className="elevenlabs-clear-filters-button"
                          onClick={() => setElevenlabsVoiceFilters(EMPTY_ELEVENLABS_VOICE_FILTERS)}
                        >
                          Clear filters
                        </button>
                      )}
                    </div>
                    <div className="elevenlabs-voice-table-scroll">
                      <div className="elevenlabs-voice-list" role="table" aria-label="ElevenLabs voices">
                        <div className="elevenlabs-voice-list-header" role="row">
                          <div className="elevenlabs-voice-col elevenlabs-voice-col-name" role="columnheader">
                            {renderElevenlabsVoiceColumnHeader("Name", "name")}
                          </div>
                          <div className="elevenlabs-voice-col elevenlabs-voice-col-gender" role="columnheader">
                            {renderElevenlabsVoiceColumnHeader("Gender", "gender")}
                          </div>
                          <div className="elevenlabs-voice-col elevenlabs-voice-col-age" role="columnheader">
                            {renderElevenlabsVoiceColumnHeader("Age", "age")}
                          </div>
                          <div className="elevenlabs-voice-col elevenlabs-voice-col-characteristics" role="columnheader">
                            {renderElevenlabsVoiceColumnHeader("Characteristics", "characteristics")}
                          </div>
                          <div className="elevenlabs-voice-col elevenlabs-voice-col-sample" role="columnheader">
                            <span>Sample</span>
                          </div>
                        </div>

                        {displayedElevenlabsCatalogVoices.length === 0 ? (
                          <div className="elevenlabs-voice-list-empty" role="row">
                            No voices match the current filters.
                          </div>
                        ) : (
                          displayedElevenlabsCatalogVoices.map((voice) => (
                            <div className="elevenlabs-voice-list-row" role="row" key={voice.voice_id}>
                              <div className="elevenlabs-voice-col elevenlabs-voice-col-name" role="cell">
                                <span className="elevenlabs-voice-name">{voice.name}</span>
                              </div>
                              <div className="elevenlabs-voice-col elevenlabs-voice-col-gender elevenlabs-voice-trait-cell" role="cell">
                                {displayElevenlabsVoiceMeta(voice.gender)}
                              </div>
                              <div className="elevenlabs-voice-col elevenlabs-voice-col-age elevenlabs-voice-trait-cell" role="cell">
                                {displayElevenlabsVoiceMeta(voice.age)}
                              </div>
                              <div
                                className="elevenlabs-voice-col elevenlabs-voice-col-characteristics elevenlabs-voice-characteristics-cell"
                                role="cell"
                              >
                                {displayElevenlabsVoiceMeta(voice.characteristics)}
                              </div>
                              <div className="elevenlabs-voice-col elevenlabs-voice-col-sample elevenlabs-voice-sample-cell" role="cell">
                                <button
                                  type="button"
                                  className="speech-sample-button elevenlabs-voice-sample-button"
                                  onClick={() => playVoiceSample(toElevenLabsVoiceUri(voice.voice_id))}
                                >
                                  <Play size={16} />
                                  Play
                                </button>
                              </div>
                            </div>
                          ))
                        )}
                      </div>
                    </div>
                  </div>
                )}
              </section>

              <div className="speech-settings-actions">
                <button type="submit">
                  <Save size={16} />
                  Save speech settings
                </button>
              </div>
            </form>
            )}
          </section>
        )}
      </main>

      <aside className={`conversation-pane ${conversationPaneCollapsed ? "conversation-pane-is-collapsed" : ""}`}>
        <div className="conversation-pane-toolbar">
          {!conversationPaneCollapsed && (
            <div className="conversation-sort-controls" role="group" aria-label="Sort discussions">
              <button
                type="button"
                className={`conversation-sort-button ${conversationSortMode === "custom" ? "conversation-sort-active" : ""}`}
                title="Custom order (drag to reorder)"
                aria-label="Custom order"
                aria-pressed={conversationSortMode === "custom"}
                onClick={() => changeConversationSortMode("custom")}
              >
                <GripVertical size={16} />
              </button>
              <button
                type="button"
                className={`conversation-sort-button ${conversationSortMode === "recent" ? "conversation-sort-active" : ""}`}
                title="Most recent activity first"
                aria-label="Most recent activity first"
                aria-pressed={conversationSortMode === "recent"}
                onClick={() => changeConversationSortMode("recent")}
              >
                <Clock size={16} />
              </button>
              <button
                type="button"
                className={`conversation-sort-button ${conversationSortMode === "alphabetical" ? "conversation-sort-active" : ""}`}
                title="Alphabetical by title"
                aria-label="Alphabetical by title"
                aria-pressed={conversationSortMode === "alphabetical"}
                onClick={() => changeConversationSortMode("alphabetical")}
              >
                <ArrowDownAZ size={16} />
              </button>
            </div>
          )}
          <button
            type="button"
            className="conversation-pane-toggle"
            aria-label={conversationPaneCollapsed ? "Expand discussions" : "Collapse discussions"}
            title={conversationPaneCollapsed ? "Expand discussions" : "Collapse discussions"}
            onClick={toggleConversationPane}
          >
            {conversationPaneCollapsed ? <ChevronLeft size={18} /> : <ChevronRight size={18} />}
          </button>
        </div>

        {!conversationPaneCollapsed && (
        <div className="conversation-list">
          {displayedConversations.map((conversation) => (
            <div
              key={conversation.id}
              className={`conversation-item ${conversationSortMode === "custom" ? "conversation-item-draggable" : ""} ${
                conversation.id === activeId ? "conversation-active" : ""
              } ${conversation.id === dragOverConversationId ? "conversation-drag-over" : ""}`}
              draggable={conversationSortMode === "custom" && editingConversationId !== conversation.id}
              onDragStart={(event) => {
                setDraggedConversationId(conversation.id);
                event.dataTransfer.effectAllowed = "move";
                event.dataTransfer.setData("text/plain", String(conversation.id));
              }}
              onDragOver={(event) => {
                event.preventDefault();
                event.dataTransfer.dropEffect = "move";
                setDragOverConversationId(conversation.id);
              }}
              onDragLeave={() => setDragOverConversationId(null)}
              onDrop={(event) => {
                event.preventDefault();
                const draggedId = Number(event.dataTransfer.getData("text/plain")) || draggedConversationId;
                setDragOverConversationId(null);
                setDraggedConversationId(null);
                if (draggedId) {
                  moveConversation(draggedId, conversation.id);
                }
              }}
              onDragEnd={() => {
                setDraggedConversationId(null);
                setDragOverConversationId(null);
              }}
            >
              {editingConversationId === conversation.id ? (
                <div className="conversation-title-edit">
                  <input
                    autoFocus
                    value={editingConversationTitle}
                    onChange={(event) => setEditingConversationTitle(event.target.value)}
                    onBlur={(event) => void saveConversationTitle(conversation, event.currentTarget.value)}
                    onFocus={(event) => event.currentTarget.select()}
                    onKeyDown={(event) => handleConversationTitleKeyDown(event, conversation)}
                    aria-label="Discussion title"
                  />
                  <span>{formatDate(conversation.updated_at)}</span>
                </div>
              ) : (
                <div className="conversation-item-main">
                <button
                  type="button"
                  className="conversation-select"
                  onClick={() => {
                    setActiveId(conversation.id);
                    setPage("chat");
                    setSelectedMemoryIds([]);
                    setMergeText("");
                  }}
                >
                  <strong
                    onDoubleClick={(event) => {
                      event.stopPropagation();
                      startEditingConversationTitle(conversation);
                    }}
                  >
                    {conversation.title}
                  </strong>
                  <span className="conversation-select-meta">
                    {(conversation.participant_count ?? 0) > 0 && (
                      <span className="conversation-multi-badge" title="Multi-agent discussion">
                        Multi
                      </span>
                    )}
                    {generationJobs[conversation.id] ? (
                      <span
                        className="conversation-generating-pill"
                        title="Generating reply — click to view progress"
                        onClick={(event) => {
                          event.stopPropagation();
                          setProgressModalConversationId(conversation.id);
                        }}
                      >
                        <Loader2 size={13} className="spin" />
                        Generating
                      </span>
                    ) : unreviewedReplyConversationIds.includes(conversation.id) ? (
                      <span className="conversation-unreviewed-pill" title="Reply ready — open to review">
                        <Bell size={13} />
                        Ready
                      </span>
                    ) : null}
                    <span>{formatDate(conversation.updated_at)}</span>
                  </span>
                </button>
                <button
                  type="button"
                  className="conversation-rename-button"
                  title="Rename discussion"
                  aria-label="Rename discussion"
                  onMouseDown={(event) => event.preventDefault()}
                  onClick={(event) => {
                    event.stopPropagation();
                    startEditingConversationTitle(conversation);
                  }}
                >
                  <Pencil size={14} />
                </button>
                </div>
              )}
            </div>
          ))}
          {!conversations.length && <p className="hint">Create your first conversation.</p>}
        </div>
        )}
      </aside>

      {isNewConversationModalOpen && (
        <div className="modal-backdrop" role="presentation" onMouseDown={() => setIsNewConversationModalOpen(false)}>
          <form className="modal-card new-conversation-modal participants-modal" onSubmit={createConversation} onMouseDown={(event) => event.stopPropagation()}>
            <div>
              <p className="eyebrow">New conversation</p>
              <h2>Name this conversation</h2>
            </div>
            <input
              autoFocus
              value={newTitle}
              onChange={(event) => setNewTitle(event.target.value)}
              placeholder="Conversation name"
            />
            <div className="conversation-mode-toggle">
              <label>
                <input
                  type="radio"
                  name="conversation-mode"
                  checked={newConversationMode === "single"}
                  onChange={() => setNewConversationMode("single")}
                />
                Single model
              </label>
              <label>
                <input
                  type="radio"
                  name="conversation-mode"
                  checked={newConversationMode === "multi"}
                  onChange={() => {
                    setNewConversationMode("multi");
                    if (!newConversationParticipants.length && config?.models.length) {
                      setNewConversationParticipants([createParticipantDraft(config.models)]);
                    }
                  }}
                />
                Multi-agent discussion
              </label>
            </div>
            {newConversationMode === "multi" && config?.models.length ? (
              <ParticipantEditor
                participants={newConversationParticipants}
                onChange={setNewConversationParticipants}
                models={config.models}
                agentProfiles={agentProfiles}
                ttsVoiceOptions={ttsVoiceOptions}
                onSaveProfile={saveAgentProfileFromDraft}
                onUpdateProfile={updateAgentProfileFromDraft}
                onPlayVoiceSample={playModelVoiceSample}
                savingProfileIndex={participantEditorTarget === "new" ? savingAgentProfileIndex : null}
              />
            ) : null}
            <div className="modal-actions">
              <button type="button" className="secondary-button" onClick={() => setIsNewConversationModalOpen(false)}>
                Cancel
              </button>
              <button type="submit">
                <Plus size={16} />
                Create
              </button>
            </div>
          </form>
        </div>
      )}

      {(isCreatingAgentProfile || editingAgentProfile) && agentProfileEditDraft && config?.models.length && (
        <div className="modal-backdrop" role="presentation" onMouseDown={closeAgentProfileEditor}>
          <form
            className="modal-card agent-profile-modal"
            onSubmit={saveAgentProfileEdit}
            onMouseDown={(event) => event.stopPropagation()}
          >
            <div className="agent-profile-modal-header">
              <p className="eyebrow">Agent library</p>
              <h2>{isCreatingAgentProfile ? "New agent" : editingAgentProfile!.name}</h2>
              <p className="modal-copy">
                {isCreatingAgentProfile
                  ? "Define a saved agent for multi-agent discussions."
                  : "Changes apply when this agent is used in new or updated discussions."}
              </p>
            </div>
            <AgentProfileFields
              draft={agentProfileEditDraft}
              onChange={(patch) => setAgentProfileEditDraft((current) => (current ? { ...current, ...patch } : current))}
              models={config.models}
              agentProfiles={agentProfiles}
              ttsVoiceOptions={ttsVoiceOptions}
              onPlayVoiceSample={playModelVoiceSample}
              expandedPersonality
            />
            <div className="modal-actions">
              <button type="button" className="secondary-button" onClick={closeAgentProfileEditor}>
                Cancel
              </button>
              <button type="submit" disabled={savingAgentProfileEdit || !agentProfileEditDraft.name.trim()}>
                {savingAgentProfileEdit ? (
                  <>
                    <Loader2 size={16} className="spin" />
                    {isCreatingAgentProfile ? "Creating..." : "Saving..."}
                  </>
                ) : (
                  <>
                    <Save size={16} />
                    {isCreatingAgentProfile ? "Create" : "Save"}
                  </>
                )}
              </button>
            </div>
          </form>
        </div>
      )}

      {isParticipantsModalOpen && config?.models.length && (
        <div className="modal-backdrop" role="presentation" onMouseDown={() => setIsParticipantsModalOpen(false)}>
          <form className="modal-card new-conversation-modal participants-modal" onSubmit={saveParticipants} onMouseDown={(event) => event.stopPropagation()}>
            <div>
              <p className="eyebrow">Discussion participants</p>
              <h2>Edit agents</h2>
              <p className="modal-copy">Changes apply to the next messages. Existing replies keep their original agent metadata.</p>
            </div>
            <ParticipantEditor
              participants={participantsDraft}
              onChange={setParticipantsDraft}
              models={config.models}
              agentProfiles={agentProfiles}
              ttsVoiceOptions={ttsVoiceOptions}
              onSaveProfile={saveAgentProfileFromDraft}
              onUpdateProfile={updateAgentProfileFromDraft}
              onPlayVoiceSample={playModelVoiceSample}
              savingProfileIndex={participantEditorTarget === "edit" ? savingAgentProfileIndex : null}
            />
            <div className="modal-actions">
              <button type="button" className="secondary-button" onClick={() => setIsParticipantsModalOpen(false)}>
                Cancel
              </button>
              <button type="submit" disabled={savingParticipants}>
                {savingParticipants ? (
                  <>
                    <Loader2 size={16} className="spin" />
                    Saving...
                  </>
                ) : (
                  <>
                    <Save size={16} />
                    Save
                  </>
                )}
              </button>
            </div>
          </form>
        </div>
      )}

      {deleteConversationTarget && (
        <div className="modal-backdrop" role="presentation" onMouseDown={() => setDeleteConversationTarget(null)}>
          <form className="modal-card delete-modal" onSubmit={confirmDeleteConversation} onMouseDown={(event) => event.stopPropagation()}>
            <div>
              <p className="eyebrow">Delete conversation</p>
              <h2>{deleteConversationTarget.title}</h2>
            </div>
            <p className="modal-copy">
              {deleteMessageCount} {deleteMessageCount === 1 ? "message" : "messages"} in this conversation will be deleted.
              {deleteMemoryCount > 0 && (
                <>
                  {" "}
                  This conversation also has {deleteMemoryCount} saved {deleteMemoryCount === 1 ? "memory" : "memories"}.
                </>
              )}
            </p>

            {deleteMemoryCount > 0 && (
              <div className="delete-options">
                <label>
                  <input
                    type="radio"
                    name="memory-delete-action"
                    checked={deleteMemoryAction === "move"}
                    disabled={!deleteDestinationConversations.length}
                    onChange={() => setDeleteMemoryAction("move")}
                  />
                  Move memories to another conversation
                </label>
                {deleteMemoryAction === "move" && (
                  <select
                    value={deleteTargetId}
                    disabled={!deleteDestinationConversations.length}
                    onChange={(event) => setDeleteTargetId(Number(event.target.value))}
                  >
                    {deleteDestinationConversations.map((conversation) => (
                      <option key={conversation.id} value={conversation.id}>
                        {conversation.title}
                      </option>
                    ))}
                  </select>
                )}
                <label>
                  <input
                    type="radio"
                    name="memory-delete-action"
                    checked={deleteMemoryAction === "delete"}
                    onChange={() => setDeleteMemoryAction("delete")}
                  />
                  Delete memories too
                </label>
              </div>
            )}

            <div className="modal-actions">
              <button type="button" className="secondary-button" onClick={() => setDeleteConversationTarget(null)}>
                Cancel
              </button>
              <button type="submit" className="danger-button" disabled={deleteMemoryAction === "move" && !deleteTargetId}>
                <Trash2 size={16} />
                Delete conversation
              </button>
            </div>
          </form>
        </div>
      )}

      {isAddModelModalOpen && (
        <div
          className="modal-backdrop"
          role="dialog"
          aria-modal="true"
          aria-labelledby="add-model-modal-title"
          onMouseDown={cancelAddModelModal}
        >
          <form className="modal-card add-model-modal" onSubmit={saveAddModelModal} onMouseDown={(event) => event.stopPropagation()}>
            <div>
              <p className="eyebrow">Model settings</p>
              <h2 id="add-model-modal-title">Add model</h2>
              <p className="modal-copy">Enter the provider details below. Use Save settings on the page to store changes on the server.</p>
            </div>

            <div className="add-model-form">
              <label className="active-model-check">
                <input
                  type="checkbox"
                  checked={addModelDraft.is_active}
                  onChange={(event) => setAddModelDraft((current) => ({ ...current, is_active: event.target.checked }))}
                />
                Use as active model for new chats
              </label>
              <label>
                Provider
                <select
                  value={addModelDraft.provider}
                  onChange={(event) => changeAddModelProvider(event.target.value as ProviderKey)}
                >
                  <option value="ollama">Ollama local</option>
                  <option value="openai">OpenAI API</option>
                  <option value="openai-compatible">OpenAI-compatible API</option>
                </select>
              </label>
              <label>
                {providerDefaults(addModelDraft.provider).addressLabel}
                <input
                  value={addModelDraft.base_url}
                  onChange={(event) => setAddModelDraft((current) => ({ ...current, base_url: event.target.value }))}
                  placeholder={providerDefaults(addModelDraft.provider).base_url}
                  required
                />
              </label>
              <label>
                Model
                <input
                  value={addModelDraft.model}
                  onChange={(event) => setAddModelDraft((current) => ({ ...current, model: event.target.value }))}
                  placeholder={providerDefaults(addModelDraft.provider).model}
                  required
                />
              </label>
              <label>
                What this model is for
                <textarea
                  value={addModelDraft.comments ?? ""}
                  onChange={(event) => setAddModelDraft((current) => ({ ...current, comments: event.target.value }))}
                  placeholder="e.g. Fast general chat, coding help, or long-form writing"
                  rows={3}
                />
              </label>
              <div className="model-row-voice add-model-voice-row">
                <label>
                  Playback voice
                  <select
                    value={addModelDraft.tts_voice_uri ?? ""}
                    onChange={(event) =>
                      setAddModelDraft((current) => ({ ...current, tts_voice_uri: event.target.value }))
                    }
                    disabled={!ttsVoiceOptions.length}
                  >
                    <option value="">Default voice (Speech settings)</option>
                    {ttsVoiceOptions.length === 0 && <option value="">Loading voices...</option>}
                    {ttsVoiceOptions.map((voice) => (
                      <option key={voice.uri} value={voice.uri}>
                        {voice.label}
                      </option>
                    ))}
                  </select>
                </label>
                <button
                  type="button"
                  className="speech-sample-button"
                  onClick={() => playModelVoiceSample(addModelDraft.tts_voice_uri)}
                  disabled={!ttsVoiceOptions.length}
                >
                  <Play size={16} />
                  Play sample
                </button>
              </div>
              <label className="model-row-api-key">
                API key
                <MaskedApiKeyInput
                  value={addModelDraft.api_key ?? ""}
                  placeholder={providerDefaults(addModelDraft.provider).keyPlaceholder}
                  onChange={(api_key) => setAddModelDraft((current) => ({ ...current, api_key }))}
                />
              </label>
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={Boolean(addModelDraft.clear_api_key)}
                  onChange={(event) => setAddModelDraft((current) => ({ ...current, clear_api_key: event.target.checked }))}
                />
                Clear key
              </label>
            </div>

            <div className="modal-actions">
              <button type="button" className="secondary-button" onClick={cancelAddModelModal}>
                Cancel
              </button>
              <button type="submit">
                <Save size={16} />
                Save
              </button>
            </div>
          </form>
        </div>
      )}

      {isConversationModelModalOpen && config && (
        <div
          className="modal-backdrop"
          role="dialog"
          aria-modal="true"
          aria-labelledby="conversation-model-modal-title"
          onMouseDown={() => setIsConversationModelModalOpen(false)}
        >
          <div className="modal-card model-picker-modal" onMouseDown={(event) => event.stopPropagation()}>
            <div>
              <p className="eyebrow">Discussion model</p>
              <h2 id="conversation-model-modal-title">Choose a model</h2>
              <p className="modal-copy">Pick the model for new replies in this conversation.</p>
            </div>

            <div className="model-picker-list">
              {config.models.map((model) => {
                const isCurrent = model.id === conversationModel?.id;
                const isSelecting = selectingModelId === model.id;
                return (
                  <div className={`model-picker-item ${isCurrent ? "model-picker-item-current" : ""}`} key={model.id}>
                    <div className="model-picker-item-body">
                      <div className="model-picker-item-header">
                        <strong>{model.model}</strong>
                        {isCurrent && <span className="model-picker-current-badge">Current</span>}
                      </div>
                      <p className="model-picker-item-provider">{model.provider}</p>
                      <p className="model-picker-item-comment">
                        {model.comments?.trim()
                          ? model.comments
                          : "No notes added for this model yet. Add them in Config → Model settings."}
                      </p>
                    </div>
                    <button
                      type="button"
                      className="model-picker-select-button"
                      disabled={isCurrent || selectingModelId !== null}
                      onClick={() => void changeConversationModel(model.id!)}
                    >
                      {isSelecting ? (
                        <>
                          <Loader2 size={16} className="spin" />
                          Selecting...
                        </>
                      ) : isCurrent ? (
                        "Selected"
                      ) : (
                        "Select"
                      )}
                    </button>
                  </div>
                );
              })}
            </div>

            <div className="modal-actions">
              <button type="button" className="secondary-button" onClick={() => setIsConversationModelModalOpen(false)}>
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {settingsSaveSuccessMessage && (
        <div
          className="modal-backdrop"
          role="dialog"
          aria-modal="true"
          aria-labelledby="settings-save-success-title"
          onMouseDown={() => setSettingsSaveSuccessMessage(null)}
        >
          <div className="modal-card settings-save-modal" onMouseDown={(event) => event.stopPropagation()}>
            <div className="settings-save-modal-icon">
              <Check size={28} />
            </div>
            <div>
              <p className="eyebrow">Settings</p>
              <h2 id="settings-save-success-title">Saved successfully</h2>
            </div>
            <p className="modal-copy">{settingsSaveSuccessMessage}</p>
            <div className="modal-actions">
              <button type="button" onClick={() => setSettingsSaveSuccessMessage(null)}>
                OK
              </button>
            </div>
          </div>
        </div>
      )}

      {progressModalJob && (
        <div
          className="llm-progress-backdrop"
          role="status"
          aria-live="polite"
          aria-label="LLM generation in progress"
          onMouseDown={dismissProgressModal}
        >
          <div className="llm-progress-card" onMouseDown={(event) => event.stopPropagation()}>
            <div className="llm-progress-header">
              <div className="llm-progress-icon">
                <Brain size={36} />
              </div>
              <div>
                <p>{progressModalJob.isMultiAgent ? "Multi-agent discussion" : "Generating with"}</p>
                <strong>
                  {progressModalJob.isMultiAgent && progressModalJob.totalSteps
                    ? progressModalJob.currentParticipantName
                      ? `${progressModalJob.currentParticipantName} · ${Math.min(
                          progressModalJob.completedSteps ?? 0,
                          progressModalJob.totalSteps,
                        )}/${progressModalJob.totalSteps}`
                      : `Step ${Math.min(progressModalJob.completedSteps ?? 0, progressModalJob.totalSteps)}/${progressModalJob.totalSteps}`
                    : progressModalJob.modelName}
                </strong>
                <span className="llm-progress-provider">{progressModalJob.conversationTitle}</span>
              </div>
            </div>

            {progressModalAgentSteps.length > 0 && (
              <ol className="generation-agent-list" aria-label="Agents generating replies">
                {progressModalAgentSteps.map((step) => (
                  <li
                    key={step.key}
                    className={`generation-agent-step generation-agent-step-${step.status}`}
                  >
                    <span className="generation-agent-step-icon" aria-hidden="true">
                      {step.status === "done" ? (
                        <Check size={16} />
                      ) : step.status === "active" ? (
                        <Loader2 size={16} className="spin" />
                      ) : (
                        <span className="generation-agent-step-pending-dot" />
                      )}
                    </span>
                    <span className="generation-agent-step-body">
                      <strong>{step.name}</strong>
                      <span>
                        {step.model}
                        {step.round != null ? ` · Round ${step.round}` : ""}
                        {step.status === "active" ? " · Generating now" : ""}
                        {step.status === "done" ? " · Done" : ""}
                        {step.status === "pending" ? " · Waiting" : ""}
                      </span>
                    </span>
                  </li>
                ))}
              </ol>
            )}

            {progressModalJob.contextPreview ? (
              <>
                <p className="llm-context-memories">
                  {progressModalJob.contextPreview.include_memories && progressModalJob.contextPreview.memory_count > 0 && (
                    <>
                      Including {progressModalJob.contextPreview.memory_count}{" "}
                      {progressModalJob.contextPreview.memory_count === 1 ? "memory" : "memories"} from this conversation
                    </>
                  )}
                  {progressModalJob.contextPreview.include_memories &&
                    progressModalJob.contextPreview.memory_count > 0 &&
                    progressModalJob.contextPreview.include_all_memories &&
                    progressModalJob.contextPreview.all_memory_count > 0 &&
                    " · "}
                  {progressModalJob.contextPreview.include_all_memories && progressModalJob.contextPreview.all_memory_count > 0 && (
                    <>
                      Including {progressModalJob.contextPreview.all_memory_count} user{" "}
                      {progressModalJob.contextPreview.all_memory_count === 1 ? "memory" : "memories"} from other conversations
                    </>
                  )}
                  {!progressModalJob.contextPreview.include_memories &&
                    !(progressModalJob.contextPreview.include_all_memories && progressModalJob.contextPreview.all_memory_count > 0) &&
                    "No memories included"}
                </p>
                <dl className="llm-context-stats">
                  <div>
                    <dt>Messages</dt>
                    <dd>{progressModalJob.contextPreview.items.length}</dd>
                  </div>
                  <div>
                    <dt>Memories</dt>
                    <dd>
                      {progressModalJob.contextPreview.memory_count}
                      {progressModalJob.contextPreview.include_all_memories && progressModalJob.contextPreview.all_memory_count > 0
                        ? ` + ${progressModalJob.contextPreview.all_memory_count} other`
                        : ""}
                    </dd>
                  </div>
                  <div>
                    <dt>Images</dt>
                    <dd>{progressModalJob.contextPreview.image_count}</dd>
                  </div>
                  <div>
                    <dt>Characters</dt>
                    <dd>{progressModalJob.contextPreview.total_chars.toLocaleString()}</dd>
                  </div>
                </dl>
                {progressModalJob.contextPreview.multi_agent_note && (
                  <p className="llm-context-memories">{progressModalJob.contextPreview.multi_agent_note}</p>
                )}
                {progressModalJob.contextPreview.generation_estimate_sec != null && (
                  <div className="llm-progress-estimate">
                    <div
                      className="llm-progress-bar"
                      role="progressbar"
                      aria-valuemin={0}
                      aria-valuemax={100}
                      aria-valuenow={Math.round(
                        generationProgressPercent(
                          progressModalElapsedSec,
                          progressModalJob.contextPreview.generation_estimate_sec,
                        ) ?? 0,
                      )}
                      aria-label="Estimated generation progress"
                    >
                      <div
                        className="llm-progress-bar-fill"
                        style={{
                          width: `${generationProgressPercent(progressModalElapsedSec, progressModalJob.contextPreview.generation_estimate_sec) ?? 0}%`,
                        }}
                      />
                    </div>
                    <div className="llm-progress-timing">
                      <span>
                        Elapsed: <strong>{formatDurationSeconds(progressModalElapsedSec)}</strong>
                      </span>
                      <span>
                        Predicted:{" "}
                        <strong>{formatDurationSeconds(progressModalJob.contextPreview.generation_estimate_sec)}</strong>
                      </span>
                      <span>
                        Remaining:{" "}
                        <strong>
                          {formatDurationSeconds(
                            Math.max(0, progressModalJob.contextPreview.generation_estimate_sec - progressModalElapsedSec),
                          )}
                        </strong>
                      </span>
                    </div>
                    {progressModalJob.contextPreview.generation_sample_count != null &&
                      progressModalJob.contextPreview.generation_sample_count > 0 &&
                      progressModalJob.contextPreview.seconds_per_char != null && (
                        <p className="llm-progress-rate">
                          Based on {progressModalJob.contextPreview.generation_sample_count} previous request
                          {progressModalJob.contextPreview.generation_sample_count === 1 ? "" : "s"} (
                          {(progressModalJob.contextPreview.seconds_per_char * 1000).toFixed(2)} ms/char)
                        </p>
                      )}
                  </div>
                )}
              </>
            ) : (
              <p className="llm-context-loading">Loading request preview...</p>
            )}

            <div className="llm-progress-footer">
              {progressModalJob.contextPreview?.generation_estimate_sec == null && (
                <span className="llm-progress-timer" aria-live="polite">
                  Elapsed: {formatDurationSeconds(progressModalElapsedSec)}
                </span>
              )}
              <button
                type="button"
                className="secondary-button llm-progress-background"
                onClick={dismissProgressModal}
              >
                Close and continue
              </button>
              <button
                type="button"
                className="secondary-button llm-progress-cancel"
                onClick={() => cancelGeneration(progressModalJob.conversationId)}
              >
                Stop now
              </button>
            </div>
          </div>
        </div>
      )}

      {completedGenerationAlert && (
        <div
          className="modal-backdrop"
          role="dialog"
          aria-modal="true"
          aria-labelledby="generation-complete-title"
          onMouseDown={dismissCompletedGenerationAlert}
        >
          <div className="modal-card generation-complete-modal" onMouseDown={(event) => event.stopPropagation()}>
            <div className={`generation-complete-icon ${completedGenerationAlert.error ? "generation-complete-icon-error" : ""}`}>
              {completedGenerationAlert.error ? <X size={28} /> : <Check size={28} />}
            </div>
            <div>
              <p className="eyebrow">{completedGenerationAlert.error ? "Generation failed" : "Reply ready"}</p>
              <h2 id="generation-complete-title">{completedGenerationAlert.title}</h2>
            </div>
            <p className="modal-copy">
              {completedGenerationAlert.error
                ? completedGenerationAlert.error
                : "The assistant finished generating a reply while you were in another conversation."}
            </p>
            <div className="modal-actions">
              <button type="button" className="secondary-button" onClick={dismissCompletedGenerationAlert}>
                Dismiss
              </button>
              <button type="button" onClick={() => void goToCompletedAnswer()}>
                {completedGenerationAlert.error ? "Go to conversation" : "Go to answer"}
              </button>
            </div>
          </div>
        </div>
      )}

      {printMemoryExport && (
        <section className="memory-export">
          <header className="memory-export-header">
            <p className="eyebrow">Conversation memories</p>
            <h1>{printMemoryExport.conversation.title}</h1>
            <dl>
              <div>
                <dt>Created</dt>
                <dd>{formatDate(printMemoryExport.conversation.created_at)}</dd>
              </div>
              <div>
                <dt>Updated</dt>
                <dd>{formatDate(printMemoryExport.conversation.updated_at)}</dd>
              </div>
              <div>
                <dt>Scope</dt>
                <dd>{printMemoryExport.scope === "selected" ? "Selected memories" : "All memories"}</dd>
              </div>
              <div>
                <dt>Memories</dt>
                <dd>{printMemoryExport.memories.length}</dd>
              </div>
              <div>
                <dt>Exported</dt>
                <dd>{new Date().toLocaleString()}</dd>
              </div>
            </dl>
          </header>

          <div className="memory-export-list">
            {printMemoryExport.memories.map((memory) => (
              <article key={memory.id} className="memory-export-item">
                <h2>{memory.title}</h2>
                <small>{formatDate(memory.created_at)}</small>
                <div className="message-content memory-content">
                  <ReactMarkdown remarkPlugins={MARKDOWN_REMARK_PLUGINS} rehypePlugins={MARKDOWN_REHYPE_PLUGINS}>
                    {memory.content}
                  </ReactMarkdown>
                </div>
              </article>
            ))}
          </div>
        </section>
      )}

      {selectedClip && (
        <button
          type="button"
          className="selection-memory-popover"
          style={{ left: selectedClip.x, top: selectedClip.y }}
          onMouseDown={(event) => event.preventDefault()}
          onClick={() => void rememberSelectedClip()}
        >
          <Brain size={15} />
          Remember clip
        </button>
      )}
    </div>
  );
}
