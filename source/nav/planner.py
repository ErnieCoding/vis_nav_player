"""The planner: "which way to the goal?"

The answer is the cheapest chain of nodes (photos) through the map from where we are to a
goal node, adding up edge weights along the way. Dijkstra's algorithm finds it, and networkx
has it built in.
"""

from __future__ import annotations

import networkx as nx
import math


def plan_path(graph: nx.Graph, start: int, goal_nodes: list[int]) -> list[int]:
    """Node numbers from ``start`` to the nearest goal node, both included. ``[start]`` if
    ``start`` is a goal node; ``[]`` if no goal node can be reached from ``start``."""
    best_path = []
    best_weight = math.inf
    for i in range(len(goal_nodes)):
        try:
            candidate_path = nx.shortest_path(graph, start, goal_nodes[i], weight="weight")
            candidate_weight = nx.path_weight(graph, candidate_path, "weight")

            if candidate_weight < best_weight:
                best_path = candidate_path
                best_weight = candidate_weight
        except nx.NetworkXNoPath:
            continue

    return best_path
