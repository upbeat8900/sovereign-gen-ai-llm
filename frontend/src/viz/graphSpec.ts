import { newMindmapNodeId } from "./mindmapTree";
import type { MemoryMapEdgeSpec, MemoryMapGraphSpec, MemoryMapNodeSpec } from "./types";

export function cloneGraphSpec(spec: MemoryMapGraphSpec): MemoryMapGraphSpec {
  return {
    ...spec,
    nodes: spec.nodes.map((node) => ({ ...node })),
    edges: spec.edges.map((edge) => ({ ...edge })),
  };
}

export function createGraphNode(partial?: Partial<MemoryMapNodeSpec>): MemoryMapNodeSpec {
  return {
    id: partial?.id ?? newMindmapNodeId(),
    label: partial?.label ?? "New idea",
    detail: partial?.detail ?? "",
    memory_ids: partial?.memory_ids ?? [],
  };
}

export function findGraphNode(spec: MemoryMapGraphSpec, nodeId: string): MemoryMapNodeSpec | undefined {
  return spec.nodes.find((node) => node.id === nodeId);
}

export function updateGraphNode(
  spec: MemoryMapGraphSpec,
  nodeId: string,
  updater: (current: MemoryMapNodeSpec) => MemoryMapNodeSpec,
): MemoryMapGraphSpec {
  return {
    ...spec,
    nodes: spec.nodes.map((node) => (node.id === nodeId ? updater(node) : node)),
  };
}

export function removeGraphNode(spec: MemoryMapGraphSpec, nodeId: string): MemoryMapGraphSpec {
  return {
    ...spec,
    nodes: spec.nodes.filter((node) => node.id !== nodeId),
    edges: spec.edges.filter((edge) => edge.source !== nodeId && edge.target !== nodeId),
  };
}

export function addGraphNode(spec: MemoryMapGraphSpec, node: MemoryMapNodeSpec): MemoryMapGraphSpec {
  return {
    ...spec,
    nodes: [...spec.nodes, node],
  };
}

export function addGraphEdge(
  spec: MemoryMapGraphSpec,
  source: string,
  target: string,
  label?: string | null,
): MemoryMapGraphSpec {
  if (source === target) {
    return spec;
  }
  if (spec.edges.some((edge) => edge.source === source && edge.target === target)) {
    return spec;
  }
  return {
    ...spec,
    edges: [...spec.edges, { source, target, label: label ?? null }],
  };
}

export function removeGraphEdgeAt(spec: MemoryMapGraphSpec, index: number): MemoryMapGraphSpec {
  if (index < 0 || index >= spec.edges.length) {
    return spec;
  }
  return {
    ...spec,
    edges: spec.edges.filter((_, edgeIndex) => edgeIndex !== index),
  };
}

export function updateGraphEdgeAt(
  spec: MemoryMapGraphSpec,
  index: number,
  updater: (current: MemoryMapEdgeSpec) => MemoryMapEdgeSpec,
): MemoryMapGraphSpec {
  if (index < 0 || index >= spec.edges.length) {
    return spec;
  }
  return {
    ...spec,
    edges: spec.edges.map((edge, edgeIndex) => (edgeIndex === index ? updater(edge) : edge)),
  };
}

export function edgeFlowId(index: number): string {
  return `edge-${index}`;
}

export function edgeIndexFromFlowId(spec: MemoryMapGraphSpec, flowId: string): number {
  const match = /^edge-(\d+)$/.exec(flowId);
  if (!match) {
    return spec.edges.findIndex((edge, index) => edgeFlowId(index) === flowId);
  }
  const index = Number(match[1]);
  return index >= 0 && index < spec.edges.length ? index : -1;
}
