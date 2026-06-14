import { useCallback, useEffect, useState } from "react";
import { Loader2, Save } from "lucide-react";
import GraphView from "./GraphView";
import KanbanView from "./KanbanView";
import MindmapView from "./MindmapView";
import WordcloudView from "./WordcloudView";
import {
  VIEW_LABELS,
  VIEW_MODES,
  buildViewSpecs,
  initialViewMode,
} from "./buildViewSpecs";
import type {
  MemoryMapGraphSpec,
  MemoryMapKanbanSpec,
  MemoryMapMindmapSpec,
  ViewMode,
  ViewSpecsBundle,
  VizSpecDetail,
} from "./types";
import "./viz.css";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, options);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

function loadViewState(detail: VizSpecDetail): { specs: ViewSpecsBundle; activeView: ViewMode } {
  if (detail.client_state?.specs) {
    return {
      specs: detail.client_state.specs,
      activeView: detail.client_state.active_view,
    };
  }
  return {
    specs: buildViewSpecs(detail.memories, detail.spec),
    activeView: initialViewMode(detail.spec.type),
  };
}

type VizPageProps = {
  vizId: string;
};

export default function VizPage({ vizId }: VizPageProps) {
  const [detail, setDetail] = useState<VizSpecDetail | null>(null);
  const [viewSpecs, setViewSpecs] = useState<ViewSpecsBundle | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeView, setActiveView] = useState<ViewMode>("wordcloud");
  const [saveState, setSaveState] = useState<"idle" | "saving" | "saved" | "error">("idle");

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    void request<VizSpecDetail>(`/api/viz-specs/${vizId}`)
      .then((data) => {
        if (!cancelled) {
          const { specs, activeView: initialView } = loadViewState(data);
          setDetail(data);
          setViewSpecs(specs);
          setActiveView(initialView);
          setSaveState("idle");
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Could not load visualization");
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [vizId]);

  const handleKanbanChange = useCallback((spec: MemoryMapKanbanSpec) => {
    setViewSpecs((current) => (current ? { ...current, kanban: spec } : current));
    setSaveState("idle");
  }, []);

  const handleMindmapChange = useCallback((spec: MemoryMapMindmapSpec) => {
    setViewSpecs((current) => (current ? { ...current, mindmap: spec } : current));
    setSaveState("idle");
  }, []);

  const handleGraphChange = useCallback((spec: MemoryMapGraphSpec) => {
    setViewSpecs((current) => (current ? { ...current, graph: spec } : current));
    setSaveState("idle");
  }, []);

  async function saveAdjustments() {
    if (!viewSpecs) {
      return;
    }
    setSaveState("saving");
    try {
      const updated = await request<VizSpecDetail>(`/api/viz-specs/${vizId}/client-state`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          active_view: activeView,
          specs: viewSpecs,
        }),
      });
      setDetail(updated);
      setSaveState("saved");
    } catch (err) {
      setSaveState("error");
      setError(err instanceof Error ? err.message : "Could not save visualization");
    }
  }

  if (loading) {
    return (
      <div className="viz-page">
        <div className="viz-status">
          <Loader2 size={28} className="spin" />
          <p>Loading visualization…</p>
        </div>
      </div>
    );
  }

  if (error || !detail || !viewSpecs) {
    return (
      <div className="viz-page">
        <div className="viz-status viz-status-error">
          <p>{error ?? "Visualization not found"}</p>
        </div>
      </div>
    );
  }

  const title = detail.spec.title;

  return (
    <div className="viz-page">
      <header className="viz-page-header">
        <div>
          <p className="viz-page-eyebrow">{VIEW_LABELS[activeView]}</p>
          <h1>{title}</h1>
        </div>
        <div className="viz-page-actions">
          <button
            type="button"
            className="viz-save-button"
            disabled={saveState === "saving"}
            onClick={() => void saveAdjustments()}
          >
            {saveState === "saving" ? <Loader2 size={16} className="spin" /> : <Save size={16} />}
            {saveState === "saved" ? "Saved" : "Save changes"}
          </button>
          <p className="viz-page-meta">
            {detail.memories.length} memories · {new Date(detail.updated_at ?? detail.created_at).toLocaleString()}
          </p>
        </div>
      </header>
      <nav className="viz-view-tabs" aria-label="Visualization views">
        {VIEW_MODES.map((mode) => (
          <button
            key={mode}
            type="button"
            className={activeView === mode ? "viz-view-tab is-active" : "viz-view-tab"}
            onClick={() => {
              setActiveView(mode);
              setSaveState("idle");
            }}
          >
            {VIEW_LABELS[mode]}
          </button>
        ))}
      </nav>
      <main className="viz-page-main">
        {activeView === "graph" && (
          <GraphView spec={viewSpecs.graph} memories={detail.memories} onSpecChange={handleGraphChange} />
        )}
        {activeView === "mindmap" && (
          <MindmapView spec={viewSpecs.mindmap} memories={detail.memories} onSpecChange={handleMindmapChange} />
        )}
        {activeView === "kanban" && (
          <KanbanView spec={viewSpecs.kanban} memories={detail.memories} onSpecChange={handleKanbanChange} />
        )}
        {activeView === "wordcloud" && (
          <WordcloudView spec={viewSpecs.wordcloud} memories={detail.memories} frequencyBased />
        )}
      </main>
    </div>
  );
}
