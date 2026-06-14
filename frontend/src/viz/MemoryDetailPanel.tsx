import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Memory, SelectedVizItem } from "./types";

type MemoryDetailPanelProps = {
  selection: SelectedVizItem | null;
  memories: Memory[];
  onClose: () => void;
};

export default function MemoryDetailPanel({ selection, memories, onClose }: MemoryDetailPanelProps) {
  if (!selection) {
    return null;
  }

  const linkedMemories = memories.filter((memory) => selection.memoryIds.includes(memory.id));

  return (
    <aside className="viz-detail-panel">
      <div className="viz-detail-header">
        <h2>{selection.label}</h2>
        <button type="button" className="viz-detail-close" onClick={onClose} aria-label="Close details">
          ×
        </button>
      </div>
      {selection.detail && <p className="viz-detail-summary">{selection.detail}</p>}
      {linkedMemories.length === 0 && !selection.detail && (
        <p className="viz-detail-empty">No linked memories for this item.</p>
      )}
      {linkedMemories.map((memory) => (
        <article key={memory.id} className="viz-detail-memory">
          <h3>{memory.title}</h3>
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{memory.content}</ReactMarkdown>
        </article>
      ))}
    </aside>
  );
}
