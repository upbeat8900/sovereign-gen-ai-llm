import { useCallback, useEffect, useRef, useState } from "react";
import {
  Background,
  Controls,
  Edge,
  Handle,
  MiniMap,
  Node,
  NodeProps,
  Position,
  ReactFlow,
  useEdgesState,
  useNodesState,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { layoutGraphNodes } from "./layoutElk";
import {
  addTreeChild,
  cloneTreeNode,
  createMindmapNode,
  findTreeNode,
  mindmapNodeDimensions,
  removeTreeNode,
  treeToMindmapSpec,
  updateTreeNode,
} from "./mindmapTree";
import type { Memory, MemoryMapMindmapSpec, MemoryMapTreeNodeSpec } from "./types";

type MindmapEditDraft = {
  nodeId: string;
  label: string;
  detail: string;
};

type MindmapNodeData = {
  label: string;
  detail?: string | null;
  memoryIds: number[];
  isActive: boolean;
  isRoot: boolean;
  onActivate: (nodeId: string) => void;
  onDraftLabelChange: (nodeId: string, label: string) => void;
  onDraftDetailChange: (nodeId: string, detail: string) => void;
  onCommitEdit: (nodeId: string) => void;
  onCancelEdit: () => void;
  onNodeContextMenu: (nodeId: string, x: number, y: number) => void;
};

type MindmapContextMenuState = {
  nodeId: string;
  x: number;
  y: number;
};

function MindmapNode({ id, data }: NodeProps<Node<MindmapNodeData>>) {
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
      <Handle type="target" position={Position.Left} />
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
      <Handle type="source" position={Position.Right} />
    </div>
  );
}

const nodeTypes = { mindmapNode: MindmapNode };

function flattenTree(
  root: MemoryMapTreeNodeSpec,
  activeNodeId: string | null,
  editDraft: MindmapEditDraft | null,
  handlers: Pick<
    MindmapNodeData,
    "onActivate" | "onDraftLabelChange" | "onDraftDetailChange" | "onCommitEdit" | "onCancelEdit" | "onNodeContextMenu"
  >,
): { nodes: Node<MindmapNodeData>[]; edges: Edge[] } {
  const nodes: Node<MindmapNodeData>[] = [];
  const edges: Edge[] = [];

  function walk(node: MemoryMapTreeNodeSpec, depth: number, index: number) {
    const isActive = node.id === activeNodeId;
    const draft = isActive && editDraft?.nodeId === node.id ? editDraft : null;
    const displayNode: MemoryMapTreeNodeSpec = draft
      ? { ...node, label: draft.label, detail: draft.detail }
      : node;
    const { width, height } = mindmapNodeDimensions(displayNode, isActive);
    nodes.push({
      id: node.id,
      type: "mindmapNode",
      position: { x: depth * 320, y: index * 120 },
      data: {
        label: displayNode.label,
        detail: displayNode.detail,
        memoryIds: node.memory_ids,
        isActive,
        isRoot: node.id === root.id,
        ...handlers,
      },
      style: { width, height },
    });
    node.children?.forEach((child, childIndex) => {
      edges.push({
        id: `${node.id}-${child.id}`,
        source: node.id,
        target: child.id,
      });
      walk(child, depth + 1, index + childIndex + 1);
    });
  }

  walk(root, 0, 0);
  return { nodes, edges };
}

type MindmapViewProps = {
  spec: MemoryMapMindmapSpec;
  memories: Memory[];
  onSpecChange?: (spec: MemoryMapMindmapSpec) => void;
};

export default function MindmapView({ spec, memories, onSpecChange }: MindmapViewProps) {
  void memories;
  const [root, setRoot] = useState<MemoryMapTreeNodeSpec>(() => cloneTreeNode(spec.root));
  const [activeNodeId, setActiveNodeId] = useState<string | null>(null);
  const [editDraft, setEditDraft] = useState<MindmapEditDraft | null>(null);
  const [contextMenu, setContextMenu] = useState<MindmapContextMenuState | null>(null);
  const [nodes, setNodes, onNodesChange] = useNodesState<Node<MindmapNodeData>>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const rootRef = useRef(root);
  rootRef.current = root;

  useEffect(() => {
    setRoot(cloneTreeNode(spec.root));
    setActiveNodeId(null);
    setEditDraft(null);
    setContextMenu(null);
  }, [spec]);

  const publishRoot = useCallback(
    (nextRoot: MemoryMapTreeNodeSpec) => {
      setRoot(nextRoot);
      onSpecChange?.(treeToMindmapSpec(spec.title, nextRoot));
    },
    [onSpecChange, spec.title],
  );

  const beginEdit = useCallback((nodeId: string) => {
    const node = findTreeNode(rootRef.current, nodeId);
    if (!node) {
      return;
    }
    setActiveNodeId(nodeId);
    setEditDraft({
      nodeId,
      label: node.label,
      detail: node.detail ?? "",
    });
    setContextMenu(null);
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
      setContextMenu({ nodeId, x, y });
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
      const nextRoot = updateTreeNode(rootRef.current, nodeId, (node) => ({
        ...node,
        label: draft.label.trim() || node.label,
        detail: draft.detail,
      }));
      publishRoot(nextRoot);
      setActiveNodeId(null);
      setEditDraft(null);
    },
    [editDraft, publishRoot],
  );

  const handleCancelEdit = useCallback(() => {
    setActiveNodeId(null);
    setEditDraft(null);
  }, []);

  const handleAddSubnode = useCallback(
    (parentId: string) => {
      const child = createMindmapNode();
      const nextRoot = addTreeChild(rootRef.current, parentId, child);
      publishRoot(nextRoot);
      beginEdit(child.id);
    },
    [beginEdit, publishRoot],
  );

  const handleDeleteNode = useCallback(
    (nodeId: string) => {
      if (nodeId === rootRef.current.id) {
        return;
      }
      const nextRoot = removeTreeNode(rootRef.current, nodeId);
      if (!nextRoot) {
        return;
      }
      publishRoot(nextRoot);
      if (activeNodeId === nodeId) {
        setActiveNodeId(null);
        setEditDraft(null);
      }
      setContextMenu(null);
    },
    [activeNodeId, publishRoot],
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
    const { nodes: initialNodes, edges: initialEdges } = flattenTree(root, activeNodeId, editDraft, handlers);

    void layoutGraphNodes(initialNodes, initialEdges, "mrtree").then((layoutNodes) => {
      if (!cancelled) {
        setNodes(layoutNodes);
        setEdges(initialEdges);
      }
    });

    return () => {
      cancelled = true;
    };
  }, [
    root,
    activeNodeId,
    handleActivate,
    handleDraftLabelChange,
    handleDraftDetailChange,
    handleCommitEdit,
    handleCancelEdit,
    handleNodeContextMenu,
    setNodes,
    setEdges,
  ]);

  useEffect(() => {
    if (!activeNodeId || !editDraft || editDraft.nodeId !== activeNodeId) {
      return;
    }
    const draftNode: MemoryMapTreeNodeSpec = {
      id: editDraft.nodeId,
      label: editDraft.label,
      detail: editDraft.detail,
      memory_ids: [],
      children: [],
    };
    const { width, height } = mindmapNodeDimensions(draftNode, true);
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
    if (!contextMenu) {
      return;
    }
    const closeMenu = () => setContextMenu(null);
    window.addEventListener("click", closeMenu);
    return () => window.removeEventListener("click", closeMenu);
  }, [contextMenu]);

  const contextNodeIsRoot = contextMenu?.nodeId === root.id;

  return (
    <div className="viz-canvas-wrap viz-mindmap-wrap">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        nodeTypes={nodeTypes}
        fitView
        nodesDraggable
        nodesConnectable={false}
        onPaneClick={() => {
          handleCancelEdit();
          setContextMenu(null);
        }}
      >
        <Background gap={18} size={1} />
        <MiniMap pannable zoomable />
        <Controls />
      </ReactFlow>

      {contextMenu && (
        <div
          className="viz-mindmap-context-menu"
          style={{ top: contextMenu.y, left: contextMenu.x }}
          onClick={(event) => event.stopPropagation()}
        >
          <button type="button" onClick={() => handleAddSubnode(contextMenu.nodeId)}>
            Add subnode
          </button>
          <button
            type="button"
            disabled={contextNodeIsRoot}
            title={contextNodeIsRoot ? "The root node cannot be deleted" : undefined}
            onClick={() => handleDeleteNode(contextMenu.nodeId)}
          >
            Delete node
          </button>
        </div>
      )}
    </div>
  );
}
