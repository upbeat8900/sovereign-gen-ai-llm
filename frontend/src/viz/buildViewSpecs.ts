import type {
  Memory,
  MemoryMapGraphSpec,
  MemoryMapKanbanSpec,
  MemoryMapMindmapSpec,
  MemoryMapSpec,
  MemoryMapTreeNodeSpec,
  MemoryMapWordcloudSpec,
  ViewMode,
  ViewSpecsBundle,
} from "./types";

export type { ViewMode } from "./types";

export const VIEW_MODES: ViewMode[] = ["wordcloud", "graph", "mindmap", "kanban"];

export const VIEW_LABELS: Record<ViewMode, string> = {
  wordcloud: "Words",
  graph: "Graph",
  mindmap: "Mind map",
  kanban: "Board",
};

const STOP_WORDS = new Set([
  "a",
  "an",
  "and",
  "are",
  "as",
  "at",
  "be",
  "but",
  "by",
  "for",
  "from",
  "had",
  "has",
  "have",
  "he",
  "her",
  "him",
  "his",
  "i",
  "if",
  "in",
  "into",
  "is",
  "it",
  "its",
  "let",
  "me",
  "my",
  "not",
  "of",
  "on",
  "or",
  "our",
  "she",
  "so",
  "that",
  "the",
  "their",
  "them",
  "then",
  "there",
  "they",
  "this",
  "to",
  "up",
  "was",
  "we",
  "were",
  "what",
  "when",
  "with",
  "you",
  "your",
]);

function memoryTitle(memory: Memory): string {
  return memory.title.trim() || `Memory ${memory.id}`;
}

function memorySnippet(memory: Memory, max = 240): string | null {
  const text = memory.content.trim();
  if (!text) {
    return null;
  }
  return text.length > max ? `${text.slice(0, max).trim()}…` : text;
}

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/\[([^\]]+)\]/g, " ")
    .replace(/[^a-z0-9'-]+/g, " ")
    .split(/\s+/)
    .map((token) => token.replace(/^'+|'+$/g, ""))
    .filter((token) => token.length >= 3 && !STOP_WORDS.has(token));
}

export function buildWordFrequencyList(memories: Memory[]) {
  const counts = new Map<string, { count: number; memoryIds: Set<number> }>();

  for (const memory of memories) {
    for (const token of tokenize(`${memory.title} ${memory.content}`)) {
      const bucket = counts.get(token) ?? { count: 0, memoryIds: new Set<number>() };
      bucket.count += 1;
      bucket.memoryIds.add(memory.id);
      counts.set(token, bucket);
    }
  }

  return Array.from(counts.entries())
    .sort((left, right) => right[1].count - left[1].count || left[0].localeCompare(right[0]))
    .slice(0, 48)
    .map(([text, data], index) => ({
      id: `word-${index}-${text}`,
      text,
      weight: data.count,
      memory_ids: Array.from(data.memoryIds),
    }));
}

export function wordFontSize(count: number, minCount: number, maxCount: number): number {
  const MIN_PX = 14;
  const MAX_PX = 80;
  if (maxCount <= minCount) {
    return (MIN_PX + MAX_PX) / 2;
  }
  const safeMin = Math.max(1, minCount);
  const safeMax = Math.max(safeMin + 1, maxCount);
  const logMin = Math.log(safeMin);
  const logMax = Math.log(safeMax);
  const logCount = Math.log(Math.max(1, count));
  const ratio = (logCount - logMin) / (logMax - logMin);
  return MIN_PX + ratio * (MAX_PX - MIN_PX);
}

function countTreeNodes(node: MemoryMapTreeNodeSpec): number {
  return 1 + (node.children ?? []).reduce((total, child) => total + countTreeNodes(child), 0);
}

export function buildGraphSpec(memories: Memory[], title: string): MemoryMapGraphSpec {
  return {
    type: "graph",
    title,
    nodes: memories.map((memory) => ({
      id: `memory-${memory.id}`,
      label: memoryTitle(memory),
      detail: memorySnippet(memory),
      memory_ids: [memory.id],
    })),
    edges: [],
  };
}

export function buildMindmapSpec(memories: Memory[], title: string): MemoryMapMindmapSpec {
  const theme =
    memories.length === 1
      ? memoryTitle(memories[0])
      : title.replace(/^Memory map$/i, "Core themes").trim() || "Core themes";

  return {
    type: "mindmap",
    title,
    root: {
      id: "root",
      label: theme,
      detail: `${memories.length} saved memories`,
      memory_ids: [],
      children: memories.map((memory) => ({
        id: `memory-${memory.id}`,
        label: memoryTitle(memory),
        detail: memorySnippet(memory, 120),
        memory_ids: [memory.id],
        children: [],
      })),
    },
  };
}

export function buildKanbanSpec(memories: Memory[], title: string): MemoryMapKanbanSpec {
  return {
    type: "kanban",
    title,
    columns: [
      {
        id: "memories",
        title: "Memories",
        cards: memories.map((memory) => ({
          id: `memory-${memory.id}`,
          title: memoryTitle(memory),
          body: memorySnippet(memory, 160),
          memory_ids: [memory.id],
        })),
      },
    ],
  };
}

export function buildFrequencyWordcloudSpec(memories: Memory[], title: string): MemoryMapWordcloudSpec {
  const words = buildWordFrequencyList(memories);

  if (!words.length) {
    return {
      type: "wordcloud",
      title,
      words: memories.map((memory, index) => ({
        id: `memory-${memory.id}`,
        text: memoryTitle(memory),
        weight: Math.max(1, 10 - index),
        memory_ids: [memory.id],
      })),
    };
  }

  return {
    type: "wordcloud",
    title,
    words,
  };
}

export function normalizeMindmapSpec(
  spec: MemoryMapMindmapSpec,
  memories: Memory[],
  title: string,
): MemoryMapMindmapSpec {
  const branchCount = Math.max(0, countTreeNodes(spec.root) - 1);
  const rootHasManyMemories = spec.root.memory_ids.length > 1;
  if (branchCount === 0 && (rootHasManyMemories || memories.length > 1)) {
    return buildMindmapSpec(memories, title);
  }
  return spec;
}

export function buildViewSpecs(memories: Memory[], llmSpec: MemoryMapSpec): ViewSpecsBundle {
  const title = llmSpec.title || "Memory map";

  return {
    wordcloud: buildFrequencyWordcloudSpec(memories, title),
    graph: llmSpec.type === "graph" && llmSpec.nodes.length > 0 ? llmSpec : buildGraphSpec(memories, title),
    mindmap:
      llmSpec.type === "mindmap"
        ? normalizeMindmapSpec(llmSpec, memories, title)
        : buildMindmapSpec(memories, title),
    kanban:
      llmSpec.type === "kanban" && llmSpec.columns.length > 0
        ? llmSpec
        : buildKanbanSpec(memories, title),
  };
}

export function initialViewMode(llmType: MemoryMapSpec["type"]): ViewMode {
  if (llmType === "wordcloud" || llmType === "graph" || llmType === "mindmap" || llmType === "kanban") {
    return llmType;
  }
  return "wordcloud";
}
