import { ChangeEvent, ClipboardEvent, FormEvent, KeyboardEvent, MouseEvent as ReactMouseEvent, useEffect, useMemo, useRef, useState } from "react";
import "katex/dist/katex.min.css";
import rehypeKatex from "rehype-katex";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import {
  ArrowDown,
  ArrowUp,
  Bot,
  ChevronLeft,
  ChevronRight,
  Brain,
  Check,
  FileDown,
  ImagePlus,
  Loader2,
  MessageCircle,
  Mic,
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
const MARKDOWN_REMARK_PLUGINS = [remarkGfm, remarkMath];
const MARKDOWN_REHYPE_PLUGINS = [rehypeKatex];

type Conversation = {
  id: number;
  title: string;
  sort_order: number;
  llm_model_id?: number | null;
  created_at: string;
  updated_at: string;
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
};

type ConversationDetail = {
  conversation: Conversation;
  messages: Message[];
  memories: Memory[];
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
};

type LlmConfig = {
  models: LlmModel[];
  active_model: LlmModel;
};

type SpeechConfig = {
  whisper_model: string;
  whisper_model_options: string[];
  updated_at: string;
};

type PromptConfig = {
  default_prompt: string;
  default_prompt_baseline: string;
  updated_at: string;
};

type TtsVoiceOption = {
  uri: string;
  label: string;
};

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
const TTS_VOICE_STORAGE_KEY = "tts_voice_uri";
const TTS_RATE_STORAGE_KEY = "tts_speech_rate";
const CURRENT_MESSAGE_ONLY_STORAGE_KEY = "chat_current_message_only";
const HISTORY_MESSAGE_LIMIT_STORAGE_KEY = "chat_history_message_limit";
const INCLUDE_MEMORIES_STORAGE_KEY = "chat_include_memories";
const INCLUDE_ALL_MEMORIES_STORAGE_KEY = "chat_include_all_memories";
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
const TTS_RATE_MAX = 2;
const TTS_SAMPLE_TEXT = "Hello, this is a sample of how the assistant will sound when reading replies.";
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
};

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
  const [deleteConversationTarget, setDeleteConversationTarget] = useState<Conversation | null>(null);
  const [deleteMessageCount, setDeleteMessageCount] = useState(0);
  const [deleteMemoryCount, setDeleteMemoryCount] = useState(0);
  const [deleteMemoryAction, setDeleteMemoryAction] = useState<"delete" | "move">("delete");
  const [deleteTargetId, setDeleteTargetId] = useState<number | "">("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [settingsSaveSuccessMessage, setSettingsSaveSuccessMessage] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  const [historyMessageLimit, setHistoryMessageLimit] = useState(readStoredHistoryMessageLimit);
  const [includeMemories, setIncludeMemories] = useState(readStoredIncludeMemories);
  const [includeAllMemories, setIncludeAllMemories] = useState(readStoredIncludeAllMemories);
  const [llmProgressModel, setLlmProgressModel] = useState("");
  const [generationElapsedSec, setGenerationElapsedSec] = useState(0);
  const [llmContextPreview, setLlmContextPreview] = useState<LlmContextPreview | null>(null);
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
  const [isAddModelModalOpen, setIsAddModelModalOpen] = useState(false);
  const [addModelDraft, setAddModelDraft] = useState<LlmModel>(() => createAddModelDraft(true));
  const [resettingTimingModelId, setResettingTimingModelId] = useState<number | null>(null);
  const [speechConfig, setSpeechConfig] = useState<SpeechConfig | null>(null);
  const [whisperModelDraft, setWhisperModelDraft] = useState("base.en");
  const [promptConfig, setPromptConfig] = useState<PromptConfig | null>(null);
  const [defaultPromptDraft, setDefaultPromptDraft] = useState("");
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
  const skipConversationTitleSaveRef = useRef(false);
  const isRecordingRef = useRef(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const mediaChunksRef = useRef<Blob[]>([]);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const speakingMessageIdRef = useRef<number | null>(null);
  const speakingPlaybackRef = useRef<{ id: number | "sample"; content: string } | null>(null);
  const playbackSessionRef = useRef(0);
  const generationAbortRef = useRef<AbortController | null>(null);
  const generationStartedAtRef = useRef<number | null>(null);
  const speechSelectionRef = useRef<{ messageId: number; text: string } | null>(null);
  const imageInputRef = useRef<HTMLInputElement | null>(null);
  const pendingLatestScrollRef = useRef(false);

  const allMemories = useMemo(() => memoryGroups.flatMap((group) => group.memories), [memoryGroups]);
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
  const deleteDestinationConversations = useMemo(
    () => conversations.filter((conversation) => conversation.id !== deleteConversationTarget?.id),
    [conversations, deleteConversationTarget?.id],
  );
  const conversationModel = useMemo(() => {
    const modelId = detail?.conversation.llm_model_id;
    return config?.models.find((model) => model.id === modelId) ?? config?.active_model ?? null;
  }, [config, detail?.conversation.llm_model_id]);
  const activeModelName = conversationModel?.model ?? config?.active_model.model ?? "qwen3.5:9b";
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

  const historyControlsDisabled = !activeId || sending || isTranscribing;

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

  useEffect(() => {
    loadConversations();
    loadMemoryGroups();
    loadConfig();
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
    if (!sending) {
      setGenerationElapsedSec(0);
      return;
    }

    const startedAt = generationStartedAtRef.current ?? Date.now();
    const tick = () => {
      setGenerationElapsedSec(Math.floor((Date.now() - startedAt) / 1000));
    };

    tick();
    const intervalId = window.setInterval(tick, 1000);
    return () => window.clearInterval(intervalId);
  }, [sending]);

  useEffect(() => {
    if (page === "settings") {
      void loadSpeechConfig();
      void loadPromptConfig();
    }
  }, [page]);

  useEffect(() => {
    if (page !== "settings" || !window.speechSynthesis) return;

    function syncVoices() {
      const options = buildTtsVoiceOptions(window.speechSynthesis.getVoices());
      setTtsVoiceOptions(options);
      setSelectedTtsVoiceUri((current) => {
        if (current && options.some((option) => option.uri === current)) {
          return current;
        }
        const preferred = options[0]?.uri ?? "";
        if (preferred) {
          localStorage.setItem(TTS_VOICE_STORAGE_KEY, preferred);
        }
        return preferred;
      });
    }

    syncVoices();
    const synth = window.speechSynthesis;
    synth.onvoiceschanged = syncVoices;
    return () => {
      synth.onvoiceschanged = null;
    };
  }, [page]);

  useEffect(() => {
    stopSpeechPlayback();
    if (activeId) {
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
      speakingMessageIdRef.current = null;
      speakingPlaybackRef.current = null;
    };
  }, []);

  function stopSpeechPlayback() {
    playbackSessionRef.current += 1;
    window.speechSynthesis?.cancel();
    speakingMessageIdRef.current = null;
    speakingPlaybackRef.current = null;
    setSpeakingMessageId(null);
  }

  function speakText(content: string, playbackId: number | "sample") {
    if (!window.speechSynthesis) return;
    const plain = content.trim();
    if (!plain) return;

    window.speechSynthesis.cancel();
    const session = ++playbackSessionRef.current;
    const utterance = new SpeechSynthesisUtterance(plain);
    applyTtsVoice(utterance);

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

    speakingPlaybackRef.current = { id: playbackId, content: plain };
    if (typeof playbackId === "number") {
      speakingMessageIdRef.current = playbackId;
      setSpeakingMessageId(playbackId);
    } else {
      speakingMessageIdRef.current = null;
      setSpeakingMessageId(null);
    }
    window.speechSynthesis.speak(utterance);
  }

  function applyTtsVoice(utterance: SpeechSynthesisUtterance) {
    const voice = resolveTtsVoice(selectedTtsVoiceUri || localStorage.getItem(TTS_VOICE_STORAGE_KEY) || "");
    if (voice) {
      utterance.voice = voice;
    }
    utterance.rate = readStoredTtsSpeechRate();
  }

  function handleTtsSpeechRateChange(rate: number) {
    const clamped = persistTtsSpeechRate(rate);
    setTtsSpeechRate(clamped);

    const current = speakingPlaybackRef.current;
    if (!current || !window.speechSynthesis?.speaking) return;
    speakText(current.content, current.id);
  }

  function playAssistantMessage(message: Message, selectedText?: string) {
    const selected = selectedText?.trim() || getSelectedTextInMessage(message.id);
    const plain = selected || stripMarkdownForSpeech(message.content);
    if (!plain) return;
    speakText(plain, message.id);
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

  function playVoiceSample() {
    speakText(TTS_SAMPLE_TEXT, "sample");
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
      const saved = await request<SpeechConfig>("/api/config/speech", {
        method: "PUT",
        body: JSON.stringify({ whisper_model: whisperModelDraft }),
      });
      setSpeechConfig(saved);
      setWhisperModelDraft(saved.whisper_model);
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
        body: JSON.stringify({ default_prompt: defaultPromptDraft }),
      });
      setPromptConfig(saved);
      setDefaultPromptDraft(saved.default_prompt);
      showSettingsSaveSuccess("Default prompt saved.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not save default prompt");
    }
  }

  function resetDefaultPrompt() {
    const baseline = promptConfig?.default_prompt_baseline ?? "";
    setDefaultPromptDraft(baseline);
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
    setDetail(data);
  }

  async function loadMemoryGroups() {
    const groups = await request<MemoryGroup[]>(`/api/memories?sort=${memorySort}&order=${memoryOrder}`);
    setMemoryGroups(groups);
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

  async function loadSpeechConfig() {
    const data = await request<SpeechConfig>("/api/config/speech");
    setSpeechConfig(data);
    setWhisperModelDraft(data.whisper_model);
  }

  async function loadPromptConfig() {
    const data = await request<PromptConfig>("/api/config/prompt");
    setPromptConfig(data);
    setDefaultPromptDraft(data.default_prompt);
  }

  async function createConversation(event?: FormEvent) {
    event?.preventDefault();
    setError("");
    const conversation = await request<Conversation>("/api/conversations", {
      method: "POST",
      body: JSON.stringify({ title: newTitle || undefined }),
    });
    setNewTitle("");
    setIsNewConversationModalOpen(false);
    setConversations((current) => [conversation, ...current]);
    setActiveId(conversation.id);
    setPage("chat");
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
    setEditingConversationId(conversation.id);
    setEditingConversationTitle(conversation.title);
  }

  function cancelConversationTitleEdit() {
    skipConversationTitleSaveRef.current = true;
    setEditingConversationId(null);
    setEditingConversationTitle("");
  }

  async function saveConversationTitle(conversation: Conversation) {
    if (skipConversationTitleSaveRef.current) {
      skipConversationTitleSaveRef.current = false;
      return;
    }

    const title = editingConversationTitle.trim() || "New Conversation";
    setEditingConversationId(null);
    setEditingConversationTitle("");

    if (title === conversation.title) return;

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
    }
  }

  function handleConversationTitleKeyDown(event: KeyboardEvent<HTMLInputElement>, conversation: Conversation) {
    if (event.key === "Enter") {
      event.preventDefault();
      event.currentTarget.blur();
    }
    if (event.key === "Escape") {
      event.preventDefault();
      cancelConversationTitleEdit();
    }
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
    const fromIndex = conversations.findIndex((conversation) => conversation.id === draggedId);
    const toIndex = conversations.findIndex((conversation) => conversation.id === targetId);
    if (fromIndex === -1 || toIndex === -1) return;

    const nextConversations = [...conversations];
    const [moved] = nextConversations.splice(fromIndex, 1);
    nextConversations.splice(toIndex, 0, moved);
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
    if (!activeId || sending || isTranscribing || isRecordingRef.current) return;
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
    try {
      const text = (await transcribeAudio(blob)).trim();
      if (!text) {
        setError("No speech detected.");
        return;
      }
      appendOptimisticExchange(text);
      setInput("");
      await submitMessage(text, { speakReply: true });
    } catch (err) {
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

  function cancelGeneration() {
    generationAbortRef.current?.abort();
  }

  async function submitMessage(overrideContent?: string, options?: { speakReply?: boolean; image?: PendingImage | null }) {
    const image = options?.image === undefined ? pendingImage : options.image;
    const content = (overrideContent ?? input).trim();
    if (!activeId || sending || (!content && !image)) return;

    const abortController = new AbortController();
    generationAbortRef.current = abortController;
    generationStartedAtRef.current = Date.now();
    setGenerationElapsedSec(0);

    setSending(true);
    setLlmProgressModel(activeModelName);
    setLlmContextPreview(null);
    setError("");
    setNotice("");
    try {
      const payload: {
        content: string;
        image_data?: string;
        image_media_type?: string;
        include_history: boolean | number;
        include_memories: boolean;
        include_all_memories: boolean;
      } = {
        content,
        include_history: historyLimitToIncludeHistory(
          composerHistoryContextTotal,
          historyContextOverhead,
          historyDbMessageCount,
        ),
        include_memories: includeMemories,
        include_all_memories: includeAllMemories,
      };
      if (image) {
        payload.image_data = image.base64;
        payload.image_media_type = image.mediaType;
      }

      const requestOptions = { signal: abortController.signal };

      try {
        const preview = await request<LlmContextPreview>(`/api/conversations/${activeId}/llm-context-preview`, {
          method: "POST",
          body: JSON.stringify(payload),
          ...requestOptions,
        });
        setLlmContextPreview(preview);
      } catch (err) {
        if (isAbortError(err)) throw err;
        setLlmContextPreview(null);
      }

      const response = await request<{
        user_message?: Message;
        assistant_message?: Message;
        memory?: Memory;
      }>(`/api/conversations/${activeId}/messages`, {
        method: "POST",
        body: JSON.stringify(payload),
        ...requestOptions,
      });
      setInput("");
      clearPendingImage();
      if (response.memory) {
        setNotice("Saved to this conversation's memory bank.");
      }
      await loadConversation(activeId);
      await loadConversations();
      if (options?.speakReply && response.assistant_message?.content) {
        playAssistantMessage(response.assistant_message);
      }
    } catch (err) {
      if (isAbortError(err)) {
        setNotice("Generation cancelled.");
        await loadConversation(activeId);
      } else {
        setError(err instanceof Error ? err.message : "Could not send message");
        await loadConversation(activeId);
      }
    } finally {
      generationAbortRef.current = null;
      generationStartedAtRef.current = null;
      setSending(false);
      setLlmProgressModel("");
      setLlmContextPreview(null);
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

  async function rememberMessage(message: Message) {
    if (!activeId) return;
    setError("");
    setLlmProgressModel(activeModelName);
    try {
      const memory = await request<Memory>(`/api/conversations/${activeId}/remember`, {
        method: "POST",
        body: JSON.stringify({ message_id: message.id }),
      });
      setNotice("Message saved to memory.");
      setDetail((current) => current && { ...current, memories: [memory, ...current.memories] });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not save message to memory");
    } finally {
      setLlmProgressModel("");
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
    setError("");
    setLlmProgressModel(activeModelName);
    try {
      const memory = await request<Memory>(`/api/conversations/${activeId}/remember`, {
        method: "POST",
        body: JSON.stringify({ content: selectedClip.content, message_id: selectedClip.messageId }),
      });
      setNotice("Selected clip saved to memory.");
      setDetail((current) => current && { ...current, memories: [memory, ...current.memories] });
      setSelectedClip(null);
      window.getSelection()?.removeAllRanges();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not save selected clip");
    } finally {
      setLlmProgressModel("");
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
    setLlmProgressModel(activeModelName);
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
    } finally {
      setLlmProgressModel("");
    }
  }

  async function integrateMemories() {
    if (selectedMemoryIds.length < 2) return;

    setError("");
    setLlmProgressModel(activeModelName);
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
    } finally {
      setLlmProgressModel("");
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
            onClick={() => {
              setNewTitle("");
              setIsNewConversationModalOpen(true);
            }}
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
                  <h1>{detail?.conversation.title ?? "Start a conversation"}</h1>
                  {detail && activeId && (
                    <button
                      type="button"
                      className="conversation-header-delete"
                      title="Delete conversation"
                      onClick={() => void openDeleteConversationModal(detail.conversation)}
                    >
                      <Trash2 size={18} />
                    </button>
                  )}
                </div>
              </div>
            </header>

            <div className="messages">
              {!detail?.messages.length && (
                <div className="empty-state">
                  <Brain size={32} />
                  <h2>No messages yet</h2>
                  <p>Create or select a conversation, then ask something. Memories stay scoped here.</p>
                </div>
              )}
              {detail?.messages.map((message) => {
                const isRemembered = rememberedMessageIds.has(message.id);
                const isThinkingPlaceholder = message.role === "assistant" && message.id < 0 && !message.content && sending;
                const isSpeaking = speakingMessageId === message.id;
                return (
                  <article id={`message-${message.id}`} key={message.id} className={`message ${message.role}`}>
                    <div className="message-meta">
                      <strong>{message.role === "user" ? "You" : "Assistant"}</strong>
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
                            <button
                              type="button"
                              className={isSpeaking ? "message-speech-active" : ""}
                              title="Play message (selection only when text is selected)"
                              onMouseDown={(event) => {
                                event.preventDefault();
                                captureSpeechSelection(message);
                              }}
                              onClick={() => handlePlayAssistantMessage(message)}
                            >
                              <Play size={15} />
                            </button>
                            <button
                              type="button"
                              title="Stop playback"
                              disabled={!isSpeaking}
                              onClick={stopSpeechPlayback}
                            >
                              <Square size={15} />
                            </button>
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
                          disabled={isThinkingPlaceholder}
                        >
                          <Brain size={15} />
                        </button>
                        <button
                          type="button"
                          className="message-delete-button"
                          title="Delete message"
                          onClick={() => void deleteMessage(message)}
                          disabled={isThinkingPlaceholder}
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
                      {isThinkingPlaceholder ? (
                        <div className="message-thinking">Thinking...</div>
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
                disabled={!activeId || sending || isTranscribing}
              />
              <div className="composer-actions">
                <input
                  ref={imageInputRef}
                  type="file"
                  accept="image/jpeg,image/png,image/gif,image/webp"
                  hidden
                  onChange={(event) => void handleImageInputChange(event)}
                />
                <button
                  type="button"
                  className="composer-image"
                  title="Upload an image"
                  onClick={() => imageInputRef.current?.click()}
                  disabled={!activeId || sending || isTranscribing || isRecording}
                >
                  <ImagePlus size={18} />
                </button>
                <div className="composer-send-cluster">
                  <div className="composer-send-row">
                    <button
                      type="button"
                      className={`composer-mic ${isRecording ? "composer-mic-recording" : ""} ${
                        !isRecording && (isTranscribing || sending) ? "composer-mic-busy" : ""
                      }`}
                      title={isRecording ? "Click again to stop and send" : "Start voice input"}
                      onClick={toggleVoiceCapture}
                      disabled={!activeId || sending || isTranscribing}
                    >
                      {isTranscribing ? <Loader2 size={18} className="spin" /> : <Mic size={18} />}
                    </button>
                    <button
                      type="submit"
                      className="composer-send"
                      disabled={!activeId || sending || isTranscribing || (!isRecording && !input.trim() && !pendingImage)}
                    >
                      {isTranscribing ? "Transcribing..." : sending ? "Thinking..." : isRecording ? "Stop & send" : "Send"}
                    </button>
                  </div>
                  <div className="composer-model-controls">
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
                  </div>
                </div>
              </div>
              <div className="composer-history">
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
                                    {highlightMatches(memory.title, memorySearchTerms)}
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
            <header className="page-header">
              <div>
                <p className="eyebrow">LLM config</p>
                <h1>Model settings</h1>
              </div>
            </header>
            <form className="settings-card model-settings-card" onSubmit={saveConfig}>
              <div className="model-settings-header">
                <p>Save local Ollama models or external OpenAI-compatible APIs here, then tick the one to use for new chat replies.</p>
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

            <form className="settings-card prompt-settings-card" onSubmit={savePromptConfig}>
              <div>
                <h2>Default prompt</h2>
                <p>
                  This system prompt is sent at the start of every chat reply. Conversation memories are still added
                  automatically when relevant.
                </p>
              </div>

              <label>
                System prompt
                <textarea
                  className="prompt-settings-textarea"
                  value={defaultPromptDraft}
                  onChange={(event) => setDefaultPromptDraft(event.target.value)}
                  rows={5}
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
                <button type="submit">
                  <Save size={16} />
                  Save prompt
                </button>
              </div>
            </form>

            <form className="settings-card speech-settings-card" onSubmit={saveSpeechConfig}>
              <div>
                <h2>Speech settings</h2>
                <p>Choose the Whisper model for microphone input and the browser voice for assistant playback.</p>
              </div>

              <label>
                Speech-to-text model (Whisper)
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

              <div className="speech-voice-row">
                <label>
                  Assistant voice (text-to-speech)
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
                  onClick={playVoiceSample}
                  disabled={!ttsVoiceOptions.length}
                >
                  <Play size={16} />
                  Play sample
                </button>
              </div>

              <button type="submit">
                <Save size={16} />
                Save speech settings
              </button>
            </form>
          </section>
        )}
      </main>

      <aside className={`conversation-pane ${conversationPaneCollapsed ? "conversation-pane-is-collapsed" : ""}`}>
        <div className="conversation-pane-toolbar">
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
          {conversations.map((conversation) => (
            <div
              key={conversation.id}
              className={`conversation-item ${conversation.id === activeId ? "conversation-active" : ""} ${
                conversation.id === dragOverConversationId ? "conversation-drag-over" : ""
              }`}
              draggable={editingConversationId !== conversation.id}
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
                    onBlur={() => void saveConversationTitle(conversation)}
                    onFocus={(event) => event.currentTarget.select()}
                    onKeyDown={(event) => handleConversationTitleKeyDown(event, conversation)}
                    aria-label="Discussion title"
                  />
                  <span>{formatDate(conversation.updated_at)}</span>
                </div>
              ) : (
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
                  <span>{formatDate(conversation.updated_at)}</span>
                </button>
              )}
            </div>
          ))}
          {!conversations.length && <p className="hint">Create your first conversation.</p>}
        </div>
        )}
      </aside>

      {isNewConversationModalOpen && (
        <div className="modal-backdrop" role="presentation" onMouseDown={() => setIsNewConversationModalOpen(false)}>
          <form className="modal-card" onSubmit={createConversation} onMouseDown={(event) => event.stopPropagation()}>
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

      {llmProgressModel && (
        <div className="llm-progress-backdrop" role="status" aria-live="polite" aria-label="LLM generation in progress">
          <div className="llm-progress-card">
            <div className="llm-progress-header">
              <div className="llm-progress-icon">
                <Brain size={36} />
              </div>
              <div>
                <p>Generating with</p>
                <strong>{llmProgressModel}</strong>
              </div>
            </div>

            {llmContextPreview ? (
              <>
                <p className="llm-context-memories">
                  {llmContextPreview.include_memories && llmContextPreview.memory_count > 0 && (
                    <>
                      Including {llmContextPreview.memory_count}{" "}
                      {llmContextPreview.memory_count === 1 ? "memory" : "memories"} from this conversation
                    </>
                  )}
                  {llmContextPreview.include_memories &&
                    llmContextPreview.memory_count > 0 &&
                    llmContextPreview.include_all_memories &&
                    llmContextPreview.all_memory_count > 0 &&
                    " · "}
                  {llmContextPreview.include_all_memories && llmContextPreview.all_memory_count > 0 && (
                    <>
                      Including {llmContextPreview.all_memory_count} user{" "}
                      {llmContextPreview.all_memory_count === 1 ? "memory" : "memories"} from other conversations
                    </>
                  )}
                  {!llmContextPreview.include_memories &&
                    !(llmContextPreview.include_all_memories && llmContextPreview.all_memory_count > 0) &&
                    "No memories included"}
                </p>
                <dl className="llm-context-stats">
                  <div>
                    <dt>Messages</dt>
                    <dd>{llmContextPreview.items.length}</dd>
                  </div>
                  <div>
                    <dt>Memories</dt>
                    <dd>
                      {llmContextPreview.memory_count}
                      {llmContextPreview.include_all_memories && llmContextPreview.all_memory_count > 0
                        ? ` + ${llmContextPreview.all_memory_count} other`
                        : ""}
                    </dd>
                  </div>
                  <div>
                    <dt>Images</dt>
                    <dd>{llmContextPreview.image_count}</dd>
                  </div>
                  <div>
                    <dt>Characters</dt>
                    <dd>{llmContextPreview.total_chars.toLocaleString()}</dd>
                  </div>
                </dl>
                {sending && llmContextPreview.generation_estimate_sec != null && (
                  <div className="llm-progress-estimate">
                    <div
                      className="llm-progress-bar"
                      role="progressbar"
                      aria-valuemin={0}
                      aria-valuemax={100}
                      aria-valuenow={Math.round(generationProgressPercent(generationElapsedSec, llmContextPreview.generation_estimate_sec) ?? 0)}
                      aria-label="Estimated generation progress"
                    >
                      <div
                        className="llm-progress-bar-fill"
                        style={{
                          width: `${generationProgressPercent(generationElapsedSec, llmContextPreview.generation_estimate_sec) ?? 0}%`,
                        }}
                      />
                    </div>
                    <div className="llm-progress-timing">
                      <span>
                        Elapsed: <strong>{formatDurationSeconds(generationElapsedSec)}</strong>
                      </span>
                      <span>
                        Predicted:{" "}
                        <strong>{formatDurationSeconds(llmContextPreview.generation_estimate_sec)}</strong>
                      </span>
                      <span>
                        Remaining:{" "}
                        <strong>
                          {formatDurationSeconds(
                            Math.max(0, llmContextPreview.generation_estimate_sec - generationElapsedSec),
                          )}
                        </strong>
                      </span>
                    </div>
                    {llmContextPreview.generation_sample_count != null &&
                      llmContextPreview.generation_sample_count > 0 &&
                      llmContextPreview.seconds_per_char != null && (
                        <p className="llm-progress-rate">
                          Based on {llmContextPreview.generation_sample_count} previous request
                          {llmContextPreview.generation_sample_count === 1 ? "" : "s"} (
                          {(llmContextPreview.seconds_per_char * 1000).toFixed(2)} ms/char)
                        </p>
                      )}
                  </div>
                )}
              </>
            ) : (
              <p className="llm-context-loading">Loading request preview...</p>
            )}

            {sending && (
              <div className="llm-progress-footer">
                {llmContextPreview?.generation_estimate_sec == null && (
                  <span className="llm-progress-timer" aria-live="polite">
                    Elapsed: {formatDurationSeconds(generationElapsedSec)}
                  </span>
                )}
                <button type="button" className="secondary-button llm-progress-cancel" onClick={cancelGeneration}>
                  Cancel
                </button>
              </div>
            )}
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
