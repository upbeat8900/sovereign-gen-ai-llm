import { useMemo, useState } from "react";
import {
  DndContext,
  DragEndEvent,
  DragOverlay,
  DragStartEvent,
  KeyboardSensor,
  PointerSensor,
  closestCorners,
  useSensor,
  useSensors,
} from "@dnd-kit/core";
import {
  SortableContext,
  arrayMove,
  horizontalListSortingStrategy,
  sortableKeyboardCoordinates,
  useSortable,
  verticalListSortingStrategy,
} from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import MemoryDetailPanel from "./MemoryDetailPanel";
import type {
  Memory,
  MemoryMapKanbanCardSpec,
  MemoryMapKanbanColumnSpec,
  MemoryMapKanbanSpec,
  SelectedVizItem,
} from "./types";

type KanbanState = {
  columns: MemoryMapKanbanColumnSpec[];
};

type KanbanCardProps = {
  card: MemoryMapKanbanCardSpec;
  onSelect: (item: SelectedVizItem) => void;
};

function KanbanCard({ card, onSelect }: KanbanCardProps) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({
    id: card.id,
    data: { type: "card", card },
  });

  return (
    <article
      ref={setNodeRef}
      className={`viz-kanban-card ${isDragging ? "is-dragging" : ""}`}
      style={{
        transform: CSS.Transform.toString(transform),
        transition,
      }}
      {...attributes}
      {...listeners}
      onClick={() =>
        onSelect({
          label: card.title,
          detail: card.body,
          memoryIds: card.memory_ids,
        })
      }
      onDoubleClick={() =>
        onSelect({
          label: card.title,
          detail: card.body,
          memoryIds: card.memory_ids,
        })
      }
    >
      <strong>{card.title}</strong>
      {card.body && <p>{card.body}</p>}
    </article>
  );
}

type KanbanColumnProps = {
  column: MemoryMapKanbanColumnSpec;
  onSelect: (item: SelectedVizItem) => void;
};

function KanbanColumn({ column, onSelect }: KanbanColumnProps) {
  const { attributes, listeners, setNodeRef, transform, transition } = useSortable({
    id: column.id,
    data: { type: "column", column },
  });
  const cardIds = column.cards.map((card) => card.id);

  return (
    <section
      ref={setNodeRef}
      className="viz-kanban-column"
      style={{
        transform: CSS.Transform.toString(transform),
        transition,
      }}
    >
      <header {...attributes} {...listeners}>
        <h3>{column.title}</h3>
        <span>{column.cards.length}</span>
      </header>
      <SortableContext items={cardIds} strategy={verticalListSortingStrategy}>
        <div className="viz-kanban-column-cards">
          {column.cards.map((card) => (
            <KanbanCard key={card.id} card={card} onSelect={onSelect} />
          ))}
        </div>
      </SortableContext>
    </section>
  );
}

type KanbanViewProps = {
  spec: MemoryMapKanbanSpec;
  memories: Memory[];
  onSpecChange?: (spec: MemoryMapKanbanSpec) => void;
};

export default function KanbanView({ spec, memories, onSpecChange }: KanbanViewProps) {
  const [state, setState] = useState<KanbanState>({ columns: spec.columns });
  const [selection, setSelection] = useState<SelectedVizItem | null>(null);
  const [activeCard, setActiveCard] = useState<MemoryMapKanbanCardSpec | null>(null);

  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 6 } }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates }),
  );

  const columnIds = useMemo(() => state.columns.map((column) => column.id), [state.columns]);

  function publishSpec(columns: MemoryMapKanbanColumnSpec[]) {
    onSpecChange?.({ type: "kanban", title: spec.title, columns });
  }

  function findColumnByCardId(cardId: string) {
    return state.columns.find((column) => column.cards.some((card) => card.id === cardId));
  }

  function handleDragStart(event: DragStartEvent) {
    const card = event.active.data.current?.card as MemoryMapKanbanCardSpec | undefined;
    if (card) {
      setActiveCard(card);
    }
  }

  function handleDragEnd(event: DragEndEvent) {
    setActiveCard(null);
    const { active, over } = event;
    if (!over) {
      return;
    }

    const activeType = active.data.current?.type;
    const overType = over.data.current?.type;

    if (activeType === "card") {
      const activeCardId = String(active.id);
      const sourceColumn = findColumnByCardId(activeCardId);
      if (!sourceColumn) {
        return;
      }

      let targetColumnId = overType === "column" ? String(over.id) : findColumnByCardId(String(over.id))?.id;
      if (!targetColumnId) {
        return;
      }

      if (sourceColumn.id === targetColumnId && overType === "card") {
        const cardIds = sourceColumn.cards.map((card) => card.id);
        const oldIndex = cardIds.indexOf(activeCardId);
        const newIndex = cardIds.indexOf(String(over.id));
        if (oldIndex !== -1 && newIndex !== -1 && oldIndex !== newIndex) {
          setState((current) => {
            const columns = current.columns.map((column) =>
              column.id === sourceColumn.id
                ? { ...column, cards: arrayMove(column.cards, oldIndex, newIndex) }
                : column,
            );
            publishSpec(columns);
            return { columns };
          });
        }
        return;
      }

      if (sourceColumn.id !== targetColumnId) {
        const movingCard = sourceColumn.cards.find((card) => card.id === activeCardId);
        if (!movingCard) {
          return;
        }
        setState((current) => {
          const columns = current.columns.map((column) => {
            if (column.id === sourceColumn.id) {
              return { ...column, cards: column.cards.filter((card) => card.id !== activeCardId) };
            }
            if (column.id === targetColumnId) {
              const nextCards = [...column.cards];
              if (overType === "card") {
                const insertIndex = nextCards.findIndex((card) => card.id === String(over.id));
                nextCards.splice(insertIndex >= 0 ? insertIndex : nextCards.length, 0, movingCard);
              } else {
                nextCards.push(movingCard);
              }
              return { ...column, cards: nextCards };
            }
            return column;
          });
          publishSpec(columns);
          return { columns };
        });
      }
    }
  }

  return (
    <div className="viz-kanban-wrap">
      <DndContext
        sensors={sensors}
        collisionDetection={closestCorners}
        onDragStart={handleDragStart}
        onDragEnd={handleDragEnd}
      >
        <SortableContext items={columnIds} strategy={horizontalListSortingStrategy}>
          <div className="viz-kanban-board">
            {state.columns.map((column) => (
              <KanbanColumn key={column.id} column={column} onSelect={setSelection} />
            ))}
          </div>
        </SortableContext>
        <DragOverlay>{activeCard ? <article className="viz-kanban-card overlay">{activeCard.title}</article> : null}</DragOverlay>
      </DndContext>
      <MemoryDetailPanel selection={selection} memories={memories} onClose={() => setSelection(null)} />
    </div>
  );
}
