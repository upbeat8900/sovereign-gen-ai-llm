import { useCallback, useEffect, useRef, useState } from "react";
import {
  Background,
  BaseEdge,
  Controls,
  Edge,
  EdgeLabelRenderer,
  EdgeProps,
  Handle,
  MiniMap,
  Node,
  NodeProps,
  Position,
  ReactFlow,
  getBezierPath,
  useEdgesState,
  useNodesState,
  type Connection,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import {
  addGraphEdge,
  addGraphNode,
  cloneGraphSpec,
  createGraphNode,
  edgeFlowId,
  edgeIndexFromFlowId,
  findGraphNode,
  removeGraphEdgeAt,
  removeGraphNode,
  updateGraphEdgeAt,
  updateGraphNode,
} from "./graphSpec";
import { layoutGraphNodes, NODE_HEIGHT, NODE_WIDTH } from "./layoutElk";
import { mindmapNodeDimensions } from "./mindmapTree";
import type { Memory, MemoryMapGraphSpec, MemoryMapNodeSpec } from "./types";

type GraphEditDraft = {
  nodeId: string;
  label: string;
  detail: string;
};

type GraphNodeData = {
  label: string;
  detail?: string | null;
  memoryIds: number[];
  isActive: boolean;
  onActivate: (nodeId: string) => void;
  onDraftLabelChange: (nodeId: string, label: string) => void;
  onDraftDetailChange: (nodeId: string, detail: string) => void;
  onCommitEdit: (nodeId: string) => void;
  onCancelEdit: () => void;
  onNodeContextMenu: (nodeId: string, x: number, y: number) => void;
};

type GraphEdgeData = {
  onLabelContextMenu: (edgeId: string, x: number, y: number) => void;
};

type NodeContextMenuState = {
  nodeId: string;
  x: number;
  y: number;
};

type EdgeContextMenuState = {
  edgeId: string;
  x: number;
  y: number;
  editing: boolean;
  draftLabel: string;
};

function GraphNode({ id, data }: NodeProps<Node<GraphNodeData>>) {
  const nodeId = id;

  return (
    <div
      className={`viz-graph-node viz-mindmap-node ${data.isActive ? "viz-mindmap-node-active" : ""}`}
      onClick={(event) => {
        event.stopPropagation();
        data.onActivate(nodeId);
      }}
      onContextMenu={(event) => {
        event.preventDefault();
        event.stopPropagation();
        data.onNodeContextMenu(nodeId, event.clientX, event.clientY);
      }}
    >
      <Handle type="target" position={Position.Top} className="viz-graph-handle" />
      {data.isActive ? (
        <>
          <input
            className="viz-mindmap-node-title-input"
            value={data.label}
            onClick={(event) => event.stopPropagation()}
            onChange={(event) => data.onDraftLabelChange(nodeId, event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.preventDefault();
                data.onCommitEdit(nodeId);
              } else if (event.key === "Escape") {
                event.preventDefault();
                data.onCancelEdit();
              }
            }}
            aria-label="Node title"
          />
          <textarea
            className="viz-mindmap-node-detail-input"
            value={data.detail ?? ""}
            onClick={(event) => event.stopPropagation()}
            onChange={(event) => data.onDraftDetailChange(nodeId, event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
                event.preventDefault();
                data.onCommitEdit(nodeId);
              } else if (event.key === "Escape") {
                event.preventDefault();
                data.onCancelEdit();
              }
            }}
            placeholder="Add details... (Ctrl+Enter to save)"
            aria-label="Node details"
          />
          <span className="viz-mindmap-node-edit-hint">Enter to save</span>
        </>
      ) : (
        <>
          <strong>{data.label}</strong>
          {data.detail ? <span className="viz-mindmap-node-detail-preview">{data.detail}</span> : null}
        </>
      )}
      <Handle type="source" position={Position.Bottom} className="viz-graph-handle" />
    </div>
  );
}

function GraphEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  label,
  data,
}: EdgeProps<Edge<GraphEdgeData>>) {
  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
  });

  return (
    <>
      <BaseEdge id={id} path={edgePath} />
      <EdgeLabelRenderer>
        <div
          className={`viz-graph-edge-label-wrap nodrag nopan ${label ? "has-label" : "is-empty"}`}
          style={{
            transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)`,
          }}
          onContextMenu={(event) => {
            event.preventDefault();
            event.stopPropagation();
            data?.onLabelContextMenu(id, event.clientX, event.clientY);
          }}
        >
          {label ? <span className="viz-graph-edge-label">{label}</span> : <span className="viz-graph-edge-label-dot" />}
        </div>
      </EdgeLabelRenderer>
    </>
  );
}

const nodeTypes = { graphNode: GraphNode };
const edgeTypes = { graphEdge: GraphEdge };

function buildFlowElements(
  graphSpec: MemoryMapGraphSpec,
  activeNodeId: string | null,
  editDraft: GraphEditDraft | null,
  handlers: Pick<
    GraphNodeData,
    | "onActivate"
    | "onDraftLabelChange"
    | "onDraftDetailChange"
    | "onCommitEdit"
    | "onCancelEdit"
    | "onNodeContextMenu"
  >,
  onLabelContextMenu: (edgeId: string, x: number, y: number) => void,
): { nodes: Node<GraphNodeData>[]; edges: Edge<GraphEdgeData>[] } {
  const nodes: Node<GraphNodeData>[] = graphSpec.nodes.map((node, index) => {
    const isActive = node.id === activeNodeId;
    const draft = isActive && editDraft?.nodeId === node.id ? editDraft : null;
    const displayNode: MemoryMapNodeSpec = draft
      ? { ...node, label: draft.label, detail: draft.detail }
      : node;
    const { width, height } = mindmapNodeDimensions(
      {
        id: displayNode.id,
        label: displayNode.label,
        detail: displayNode.detail,
        memory_ids: displayNode.memory_ids,
        children: [],
      },
      isActive,
    );

    return {
      id: node.id,
      type: "graphNode",
      position: { x: (index % 4) * 240, y: Math.floor(index / 4) * 120 },
      data: {
        label: displayNode.label,
        detail: displayNode.detail,
        memoryIds: node.memory_ids,
        isActive,
        ...handlers,
      },
      style: { width, height },
    };
  });

  const edges: Edge<GraphEdgeData>[] = graphSpec.edges.map((edge, index) => ({
    id: edgeFlowId(index),
    type: "graphEdge",
    source: edge.source,
    target: edge.target,
    label: edge.label ?? undefined,
    data: { onLabelContextMenu },
  }));

  return { nodes, edges };
}

type GraphViewProps = {
  spec: MemoryMapGraphSpec;
  memories: Memory[];
  onSpecChange?: (spec: MemoryMapGraphSpec) => void;
};

export default function GraphView({ spec, memories, onSpecChange }: GraphViewProps) {
  void memories;
  const [graphSpec, setGraphSpec] = useState<MemoryMapGraphSpec>(() => cloneGraphSpec(spec));
  const [activeNodeId, setActiveNodeId] = useState<string | null>(null);
  const [editDraft, setEditDraft] = useState<GraphEditDraft | null>(null);
  const [nodeContextMenu, setNodeContextMenu] = useState<NodeContextMenuState | null>(null);
  const [edgeContextMenu, setEdgeContextMenu] = useState<EdgeContextMenuState | null>(null);
  const [nodes, setNodes, onNodesChange] = useNodesState<Node<GraphNodeData>>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge<GraphEdgeData>>([]);
  const graphSpecRef = useRef(graphSpec);
  graphSpecRef.current = graphSpec;

  useEffect(() => {
    setGraphSpec(cloneGraphSpec(spec));
    setActiveNodeId(null);
    setEditDraft(null);
    setNodeContextMenu(null);
    setEdgeContextMenu(null);
  }, [spec]);

  const publishGraph = useCallback(
    (next: MemoryMapGraphSpec) => {
      setGraphSpec(next);
      onSpecChange?.(next);
    },
    [onSpecChange],
  );

  const beginEdit = useCallback((nodeId: string) => {
    const node = findGraphNode(graphSpecRef.current, nodeId);
    if (!node) {
      return;
    }
    setActiveNodeId(nodeId);
    setEditDraft({
      nodeId,
      label: node.label,
      detail: node.detail ?? "",
    });
    setNodeContextMenu(null);
  }, []);

  const handleActivate = useCallback(
    (nodeId: string) => {
      if (activeNodeId === nodeId) {
        return;
      }
      beginEdit(nodeId);
    },
    [activeNodeId, beginEdit],
  );

  const handleNodeContextMenu = useCallback(
    (nodeId: string, x: number, y: number) => {
      beginEdit(nodeId);
      setNodeContextMenu({ nodeId, x, y });
      setEdgeContextMenu(null);
    },
    [beginEdit],
  );

  const handleDraftLabelChange = useCallback((nodeId: string, label: string) => {
    setEditDraft((current) => (current?.nodeId === nodeId ? { ...current, label } : current));
  }, []);

  const handleDraftDetailChange = useCallback((nodeId: string, detail: string) => {
    setEditDraft((current) => (current?.nodeId === nodeId ? { ...current, detail } : current));
  }, []);

  const handleCommitEdit = useCallback(
    (nodeId: string) => {
      const draft = editDraft?.nodeId === nodeId ? editDraft : null;
      if (!draft) {
        setActiveNodeId(null);
        return;
      }
      const nextSpec = updateGraphNode(graphSpecRef.current, nodeId, (node) => ({
        ...node,
        label: draft.label.trim() || node.label,
        detail: draft.detail,
      }));
      publishGraph(nextSpec);
      setActiveNodeId(null);
      setEditDraft(null);
    },
    [editDraft, publishGraph],
  );

  const handleCancelEdit = useCallback(() => {
    setActiveNodeId(null);
    setEditDraft(null);
  }, []);

  const handleAddNode = useCallback(() => {
    const node = createGraphNode();
    publishGraph(addGraphNode(graphSpecRef.current, node));
    beginEdit(node.id);
  }, [beginEdit, publishGraph]);

  const handleDeleteNode = useCallback(
    (nodeId: string) => {
      publishGraph(removeGraphNode(graphSpecRef.current, nodeId));
      if (activeNodeId === nodeId) {
        setActiveNodeId(null);
        setEditDraft(null);
      }
      setNodeContextMenu(null);
    },
    [activeNodeId, publishGraph],
  );

  const handleLabelContextMenu = useCallback((edgeId: string, x: number, y: number) => {
    const edgeIndex = edgeIndexFromFlowId(graphSpecRef.current, edgeId);
    const edge = edgeIndex >= 0 ? graphSpecRef.current.edges[edgeIndex] : null;
    setEdgeContextMenu({
      edgeId,
      x,
      y,
      editing: false,
      draftLabel: edge?.label ?? "",
    });
    setNodeContextMenu(null);
  }, []);

  const handleDeleteEdge = useCallback(
    (edgeId: string) => {
      const edgeIndex = edgeIndexFromFlowId(graphSpecRef.current, edgeId);
      if (edgeIndex < 0) {
        return;
      }
      publishGraph(removeGraphEdgeAt(graphSpecRef.current, edgeIndex));
      setEdgeContextMenu(null);
    },
    [publishGraph],
  );

  const handleCommitEdgeLabel = useCallback(
    (edgeId: string) => {
      const edgeIndex = edgeIndexFromFlowId(graphSpecRef.current, edgeId);
      if (edgeIndex < 0) {
        setEdgeContextMenu(null);
        return;
      }
      const draftLabel = edgeContextMenu?.edgeId === edgeId ? edgeContextMenu.draftLabel : "";
      const trimmed = draftLabel.trim();
      publishGraph(
        updateGraphEdgeAt(graphSpecRef.current, edgeIndex, (edge) => ({
          ...edge,
          label: trimmed || null,
        })),
      );
      setEdgeContextMenu(null);
    },
    [edgeContextMenu, publishGraph],
  );

  const handleConnect = useCallback(
    (connection: Connection) => {
      if (!connection.source || !connection.target) {
        return;
      }
      publishGraph(addGraphEdge(graphSpecRef.current, connection.source, connection.target));
    },
    [publishGraph],
  );

  useEffect(() => {
    let cancelled = false;
    const handlers = {
      onActivate: handleActivate,
      onDraftLabelChange: handleDraftLabelChange,
      onDraftDetailChange: handleDraftDetailChange,
      onCommitEdit: handleCommitEdit,
      onCancelEdit: handleCancelEdit,
      onNodeContextMenu: handleNodeContextMenu,
    };
    const { nodes: initialNodes, edges: initialEdges } = buildFlowElements(
      graphSpec,
      activeNodeId,
      editDraft,
      handlers,
      handleLabelContextMenu,
    );

    void layoutGraphNodes(initialNodes, initialEdges, "layered").then((layoutNodes) => {
      if (!cancelled) {
        setNodes(layoutNodes);
        setEdges(initialEdges);
      }
    });

    return () => {
      cancelled = true;
    };
  }, [
    graphSpec,
    activeNodeId,
    handleActivate,
    handleDraftLabelChange,
    handleDraftDetailChange,
    handleCommitEdit,
    handleCancelEdit,
    handleNodeContextMenu,
    handleLabelContextMenu,
    setNodes,
    setEdges,
  ]);

  useEffect(() => {
    if (!activeNodeId || !editDraft || editDraft.nodeId !== activeNodeId) {
      return;
    }
    const draftNode: MemoryMapNodeSpec = {
      id: editDraft.nodeId,
      label: editDraft.label,
      detail: editDraft.detail,
      memory_ids: [],
    };
    const { width, height } = mindmapNodeDimensions(
      { ...draftNode, children: [] },
      true,
    );
    setNodes((current) =>
      current.map((node) =>
        node.id === activeNodeId
          ? {
              ...node,
              data: {
                ...node.data,
                label: editDraft.label,
                detail: editDraft.detail,
              },
              style: { width, height },
            }
          : node,
      ),
    );
  }, [activeNodeId, editDraft, setNodes]);

  useEffect(() => {
    if (!nodeContextMenu && !edgeContextMenu) {
      return;
    }
    const closeMenus = () => {
      setNodeContextMenu(null);
      setEdgeContextMenu(null);
    };
    window.addEventListener("click", closeMenus);
    return () => window.removeEventListener("click", closeMenus);
  }, [nodeContextMenu, edgeContextMenu]);

  return (
    <div className="viz-canvas-wrap viz-mindmap-wrap">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        fitView
        nodesDraggable
        nodesConnectable
        onConnect={handleConnect}
        onPaneClick={() => {
          handleCancelEdit();
          setNodeContextMenu(null);
          setEdgeContextMenu(null);
        }}
        onEdgeContextMenu={(event, edge) => {
          event.preventDefault();
          handleLabelContextMenu(edge.id, event.clientX, event.clientY);
        }}
      >
        <Background gap={18} size={1} />
        <MiniMap pannable zoomable />
        <Controls />
      </ReactFlow>

      {nodeContextMenu && (
        <div
          className="viz-mindmap-context-menu"
          style={{ top: nodeContextMenu.y, left: nodeContextMenu.x }}
          onClick={(event) => event.stopPropagation()}
        >
          <button type="button" onClick={() => handleAddNode()}>
            Add node
          </button>
          <button type="button" onClick={() => handleDeleteNode(nodeContextMenu.nodeId)}>
            Delete node
          </button>
        </div>
      )}

      {edgeContextMenu && (
        <div
          className="viz-mindmap-context-menu viz-graph-edge-context-menu"
          style={{ top: edgeContextMenu.y, left: edgeContextMenu.x }}
          onClick={(event) => event.stopPropagation()}
        >
          {edgeContextMenu.editing ? (
            <>
              <input
                className="viz-graph-edge-label-input"
                value={edgeContextMenu.draftLabel}
                placeholder="Connection tag"
                onChange={(event) =>
                  setEdgeContextMenu((current) =>
                    current ? { ...current, draftLabel: event.target.value } : current,
                  )
                }
                onKeyDown={(event) => {
                  if (event.key === "Enter") {
                    event.preventDefault();
                    handleCommitEdgeLabel(edgeContextMenu.edgeId);
                  } else if (event.key === "Escape") {
                    event.preventDefault();
                    setEdgeContextMenu(null);
                  }
                }}
              />
              <button type="button" onClick={() => handleCommitEdgeLabel(edgeContextMenu.edgeId)}>
                Save tag
              </button>
            </>
          ) : (
            <>
              <button
                type="button"
                onClick={() =>
                  setEdgeContextMenu((current) => (current ? { ...current, editing: true } : current))
                }
              >
                Edit tag
              </button>
              <button type="button" onClick={() => handleDeleteEdge(edgeContextMenu.edgeId)}>
                Delete connection
              </button>
            </>
          )}
        </div>
      )}
    </div>
  );
}
