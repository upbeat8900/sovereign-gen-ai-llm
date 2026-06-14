export type Memory = {
  id: number;
  conversation_id: number;
  title: string;
  content: string;
  created_at: string;
};

export type MemoryMapNodeSpec = {
  id: string;
  label: string;
  detail?: string | null;
  memory_ids: number[];
};

export type MemoryMapEdgeSpec = {
  source: string;
  target: string;
  label?: string | null;
};

export type MemoryMapTreeNodeSpec = {
  id: string;
  label: string;
  detail?: string | null;
  memory_ids: number[];
  children?: MemoryMapTreeNodeSpec[];
};

export type MemoryMapKanbanCardSpec = {
  id: string;
  title: string;
  body?: string | null;
  memory_ids: number[];
};

export type MemoryMapKanbanColumnSpec = {
  id: string;
  title: string;
  cards: MemoryMapKanbanCardSpec[];
};

export type MemoryMapGraphSpec = {
  type: "graph";
  title: string;
  nodes: MemoryMapNodeSpec[];
  edges: MemoryMapEdgeSpec[];
};

export type MemoryMapMindmapSpec = {
  type: "mindmap";
  title: string;
  root: MemoryMapTreeNodeSpec;
};

export type MemoryMapKanbanSpec = {
  type: "kanban";
  title: string;
  columns: MemoryMapKanbanColumnSpec[];
};

export type MemoryMapWordSpec = {
  id: string;
  text: string;
  weight: number;
  memory_ids: number[];
};

export type MemoryMapWordcloudSpec = {
  type: "wordcloud";
  title: string;
  words: MemoryMapWordSpec[];
};

export type MemoryMapSpec =
  | MemoryMapGraphSpec
  | MemoryMapMindmapSpec
  | MemoryMapKanbanSpec
  | MemoryMapWordcloudSpec;

export type VizSpecDetail = {
  viz_id: string;
  conversation_id: number;
  spec: MemoryMapSpec;
  memories: Memory[];
  created_at: string;
  updated_at?: string | null;
  client_state?: {
    active_view: ViewMode;
    specs: ViewSpecsBundle;
  } | null;
};

export type VizSpecSummary = {
  viz_id: string;
  conversation_id: number;
  title: string;
  spec_type: string;
  memory_count: number;
  created_at: string;
  updated_at?: string | null;
};

export type ViewMode = "graph" | "mindmap" | "kanban" | "wordcloud";

export type ViewSpecsBundle = {
  wordcloud: MemoryMapWordcloudSpec;
  graph: MemoryMapGraphSpec;
  mindmap: MemoryMapMindmapSpec;
  kanban: MemoryMapKanbanSpec;
};

export type SelectedVizItem = {
  label: string;
  detail?: string | null;
  memoryIds: number[];
};
