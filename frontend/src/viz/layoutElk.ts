import ELK from "elkjs/lib/elk.bundled.js";
import type { Edge, Node } from "@xyflow/react";

const elk = new ELK();

const NODE_WIDTH = 200;
const NODE_HEIGHT = 72;

export async function layoutGraphNodes<T extends Record<string, unknown>>(
  nodes: Node<T>[],
  edges: Edge[],
  algorithm: "layered" | "mrtree" = "layered",
): Promise<Node<T>[]> {
  if (!nodes.length) {
    return nodes;
  }

  const graph = {
    id: "root",
    layoutOptions: {
      "elk.algorithm": algorithm,
      "elk.spacing.nodeNode": "48",
      "elk.layered.spacing.nodeNodeBetweenLayers": "64",
    },
    children: nodes.map((node) => ({
      id: node.id,
      width: typeof node.style?.width === "number" ? node.style.width : NODE_WIDTH,
      height: typeof node.style?.height === "number" ? node.style.height : NODE_HEIGHT,
    })),
    edges: edges.map((edge) => ({
      id: edge.id,
      sources: [edge.source],
      targets: [edge.target],
    })),
  };

  const layout = await elk.layout(graph);
  const positions = new Map(
    (layout.children ?? []).map((child) => [child.id, { x: child.x ?? 0, y: child.y ?? 0 }]),
  );

  return nodes.map((node) => ({
    ...node,
    position: positions.get(node.id) ?? node.position,
  }));
}

export { NODE_WIDTH, NODE_HEIGHT };
