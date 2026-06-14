import { useMemo, useState } from "react";
import { buildWordFrequencyList, wordFontSize } from "./buildViewSpecs";
import MemoryDetailPanel from "./MemoryDetailPanel";
import type { Memory, MemoryMapWordcloudSpec, SelectedVizItem } from "./types";

const WORD_COLORS = [
  "#1d4ed8",
  "#7c3aed",
  "#db2777",
  "#059669",
  "#d97706",
  "#0891b2",
  "#be123c",
  "#4338ca",
];

type WordcloudViewProps = {
  spec: MemoryMapWordcloudSpec;
  memories: Memory[];
  frequencyBased?: boolean;
};

export default function WordcloudView({ spec, memories, frequencyBased = false }: WordcloudViewProps) {
  const [selection, setSelection] = useState<SelectedVizItem | null>(null);

  const words = useMemo(
    () => (frequencyBased ? buildWordFrequencyList(memories) : spec.words),
    [frequencyBased, memories, spec.words],
  );

  const counts = useMemo(() => words.map((word) => word.weight), [words]);
  const minCount = useMemo(() => Math.min(...counts, 1), [counts]);
  const maxCount = useMemo(() => Math.max(...counts, 1), [counts]);

  const positionedWords = useMemo(
    () =>
      words.map((word, index) => {
        const size = wordFontSize(word.weight, minCount, maxCount);
        const rotation = ((index * 37) % 7) - 3;
        const color = WORD_COLORS[index % WORD_COLORS.length];
        return { ...word, size, rotation, color };
      }),
    [words, minCount, maxCount],
  );

  return (
    <div className="viz-wordcloud-wrap">
      <div className="viz-wordcloud-panel">
        <p className="viz-wordcloud-hint">
          {frequencyBased
            ? "Word size reflects how often each term appears across your saved memories."
            : "Word size reflects emphasis from the generated map."}
        </p>
        <div className="viz-wordcloud-canvas" aria-label="Memory word cloud">
          {positionedWords.map((word) => (
            <button
              key={word.id}
              type="button"
              className="viz-wordcloud-word"
              style={{
                color: word.color,
                fontSize: `${word.size}px`,
                transform: `rotate(${word.rotation}deg)`,
              }}
              title={
                frequencyBased
                  ? `"${word.text}" — ${word.weight} occurrence${word.weight === 1 ? "" : "s"} in ${word.memory_ids.length} memor${word.memory_ids.length === 1 ? "y" : "ies"}`
                  : `Weight ${word.weight}`
              }
              onClick={() =>
                setSelection({
                  label: word.text,
                  detail: frequencyBased
                    ? `Appears ${word.weight} time${word.weight === 1 ? "" : "s"} across ${word.memory_ids.length} memor${word.memory_ids.length === 1 ? "y" : "ies"}.`
                    : null,
                  memoryIds: word.memory_ids,
                })
              }
            >
              {word.text}
            </button>
          ))}
        </div>
      </div>
      <MemoryDetailPanel selection={selection} memories={memories} onClose={() => setSelection(null)} />
    </div>
  );
}
