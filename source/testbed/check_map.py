"""
Testbed: grade a built map with the answer key. Ready to use.

    uv run source/build_map.py --data data/eval-21-3          # build it first
    uv run source/testbed/check_map.py --data data/eval-21-3

Checks:
  - shortcut edges: a shortcut is wrong if its two photos are more than SHORTCUT_MAX_CELLS
    grid cells apart when walking along the corridors. Photos on two sides of a wall are close
    in a straight line but far apart on foot: for the planner, that edge is a door through the
    wall. Facing different ways at the same spot is fine (that is a turn). Target: zero wrong.
  - pieces: is the map one connected piece? If not, some places cannot be planned to.
  - goal photos: how far from the real goal, and facing how far off view 0?
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # source/

import networkx as nx
from nav.data import load_answer_key
from nav.place_map import PlaceMap, map_file

from testbed.tools import angle_deg, distance_m, load_truth, walking_cells

SHORTCUT_MAX_CELLS = 3  # 0.6 m of corridor. Two sides of a wall are 4+ cells apart on foot.


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data", required=True, help="a testbed folder, e.g. data/eval-21-3")
    args = p.parse_args()

    data_dir = Path(args.data)
    place_map = PlaceMap.load(map_file(data_dir))
    answer = load_answer_key(data_dir)
    truth = load_truth(data_dir)
    photos, graph = place_map.photos, place_map.graph

    def pose(node: int):
        return answer[photos[node].key]

    shortcuts = [(i, j) for i, j, kind in graph.edges(data="kind") if kind == "shortcut"]
    routes = [(i, j) for i, j, kind in graph.edges(data="kind") if kind == "route"]

    def apart(i: int, j: int) -> int | None:
        return walking_cells(truth["grid"], pose(i), pose(j))

    def is_wrong(i: int, j: int) -> bool:
        cells = apart(i, j)
        return cells is None or cells > SHORTCUT_MAX_CELLS

    wrong = [(i, j) for i, j in shortcuts if is_wrong(i, j)]
    print(
        f"{graph.number_of_nodes()} nodes, {len(routes)} route edges, "
        f"{len(shortcuts)} shortcut edges"
    )
    print(f"wrong shortcuts: {len(wrong)} of {len(shortcuts)}")
    for i, j in wrong[:10]:
        cells = apart(i, j)
        on_foot = "over 10" if cells is None else str(cells)
        print(
            f"  {photos[i].key} -- {photos[j].key}: {distance_m(pose(i), pose(j)):.2f} m apart in "
            f"a straight line, {on_foot} cells on foot, facing "
            f"{angle_deg(pose(i)[2], pose(j)[2]):.0f} deg apart"
        )

    pieces = sorted(nx.connected_components(graph), key=len, reverse=True)
    sizes = [len(c) for c in pieces[:5]]
    print(f"map pieces: {len(pieces)} (largest: {sizes})")

    target = truth["target"]
    print(f"goal at x={target[0]:.2f} y={target[1]:.2f}; your goal photos:")
    for node in place_map.goal_nodes:
        where = pose(node)
        print(
            f"  {photos[node].key}: {distance_m(where, target):.2f} m from the goal, "
            f"facing {angle_deg(where[2], target[2]):.0f} deg off view 0"
        )


if __name__ == "__main__":
    main()
