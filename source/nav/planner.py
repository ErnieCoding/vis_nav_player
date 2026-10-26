"""The planner: "which way to the goal?"

The answer is the cheapest chain of nodes (photos) through the map from where we are to a
goal node, adding up edge weights along the way. Dijkstra's algorithm finds it, and networkx
has it built in.
"""

from __future__ import annotations

import networkx as nx


def plan_path(graph: nx.Graph, start: int, goal_nodes: list[int]) -> list[int]:
    """Node numbers from ``start`` to the nearest goal node, both included. ``[start]`` if
    ``start`` is a goal node; ``[]`` if no goal node can be reached from ``start``."""
    # TODO: shortest path to the nearest goal node.
    #   Option A: for each goal node g, try nx.shortest_path(graph, start, g, weight="weight")
    #   and keep the path with the smallest total weight (nx.path_weight(graph, path,
    #   "weight")). A goal in another piece of the map raises nx.NetworkXNoPath: skip it.
    #   Option B (one call): nx.multi_source_dijkstra(graph, set(goal_nodes), target=start,
    #   weight="weight") returns (length, path) from the NEAREST goal to start. Reverse that
    #   path. It raises nx.NetworkXNoPath when none can be reached: return [].
    raise NotImplementedError("plan_path")
