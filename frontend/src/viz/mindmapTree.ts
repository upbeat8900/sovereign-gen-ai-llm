import type { MemoryMapMindmapSpec, MemoryMapTreeNodeSpec } from "./types";

export function cloneTreeNode(node: MemoryMapTreeNodeSpec): MemoryMapTreeNodeSpec {
  return {
    ...node,
    children: (node.children ?? []).map(cloneTreeNode),
  };
}

export function newMindmapNodeId(): string {
  return `node-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export function createMindmapNode(partial?: Partial<MemoryMapTreeNodeSpec>): MemoryMapTreeNodeSpec {
  return {
    id: partial?.id ?? newMindmapNodeId(),
    label: partial?.label ?? "New idea",
    detail: partial?.detail ?? "",
    memory_ids: partial?.memory_ids ?? [],
    children: partial?.children ?? [],
  };
}

export function findTreeNode(node: MemoryMapTreeNodeSpec, nodeId: string): MemoryMapTreeNodeSpec | null {
  if (node.id === nodeId) {
    return node;
  }
  for (const child of node.children ?? []) {
    const found = findTreeNode(child, nodeId);
    if (found) {
      return found;
    }
  }
  return null;
}

export function updateTreeNode(
  node: MemoryMapTreeNodeSpec,
  nodeId: string,
  updater: (current: MemoryMapTreeNodeSpec) => MemoryMapTreeNodeSpec,
): MemoryMapTreeNodeSpec {
  if (node.id === nodeId) {
    return updater(node);
  }
  if (!node.children?.length) {
    return node;
  }
  return {
    ...node,
    children: node.children.map((child) => updateTreeNode(child, nodeId, updater)),
  };
}

export function addTreeChild(
  node: MemoryMapTreeNodeSpec,
  parentId: string,
  child: MemoryMapTreeNodeSpec,
): MemoryMapTreeNodeSpec {
  if (node.id === parentId) {
    return {
      ...node,
      children: [...(node.children ?? []), child],
    };
  }
  if (!node.children?.length) {
    return node;
  }
  return {
    ...node,
    children: node.children.map((current) => addTreeChild(current, parentId, child)),
  };
}

export function removeTreeNode(node: MemoryMapTreeNodeSpec, nodeId: string): MemoryMapTreeNodeSpec | null {
  if (node.id === nodeId) {
    return null;
  }
  if (!node.children?.length) {
    return node;
  }
  const children = node.children
    .map((child) => removeTreeNode(child, nodeId))
    .filter((child): child is MemoryMapTreeNodeSpec => child !== null);
  return { ...node, children };
}

export function mindmapNodeDimensions(node: MemoryMapTreeNodeSpec, isActive: boolean) {
  const baseWidth = 200;
  const baseHeight = 72;
  if (!isActive) {
    return { width: baseWidth, height: baseHeight };
  }
  const width = 320;
  const detail = node.detail ?? "";
  const wrappedLines = Math.ceil(Math.max(detail.length, 1) / 38);
  const explicitLines = detail.split("\n").length;
  const lineCount = Math.max(wrappedLines, explicitLines, 3);
  const detailHeight = Math.max(88, Math.min(320, lineCount * 20 + 16));
  return { width, height: 56 + detailHeight };
}

export function treeToMindmapSpec(title: string, root: MemoryMapTreeNodeSpec): MemoryMapMindmapSpec {
  return { type: "mindmap", title, root: cloneTreeNode(root) };
}
