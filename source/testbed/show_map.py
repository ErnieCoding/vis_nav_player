"""
Testbed: draw a built map. Ready to use.

    uv run source/build_map.py --data data/eval-21-3       # build it first
    uv run source/testbed/show_map.py --data data/eval-21-3

Writes two pictures to results/:

  map_<folder>.png    the maze from above, with every photo placed where it was REALLY taken
                      (from the answer key: the map itself has no positions), route edges in
                      each drive's colour, shortcut edges in green, the goal photos, and the
                      shortest path from the start to the goal that the planner would follow
  path_<folder>.jpg   the photos along that path, in order: what the robot expects to see
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # source/

import cv2
import networkx as nx
import numpy as np
from nav.data import load_answer_key, read_image, split_goal_photo
from nav.place_map import PlaceMap, map_file
from nav.place_recognition import RECOGNIZERS

from testbed.tools import load_truth

CELL = 32  # pixels per 0.2 m grid cell
ROUTE_COLOURS = [(200, 120, 40), (40, 140, 240), (180, 90, 160), (120, 160, 0), (0, 160, 200)]
SHORTCUT = (60, 170, 60)
PATH = (40, 40, 220)
GOAL = (0, 0, 200)
START = (0, 150, 0)
WALL = (70, 70, 70)
FREE = (245, 245, 245)
TEXT = (30, 30, 30)


def to_px(x: float, y: float, cell_m: float = 0.2) -> tuple[int, int]:
    """Metres -> pixels. Cell (row, col) is centred at x = col * 0.2, y = row * 0.2."""
    return int(round(x / cell_m * CELL + CELL / 2)), int(round(y / cell_m * CELL + CELL / 2))


def arrow(image, x, y, yaw, length, colour, thickness=1):
    x0, y0 = to_px(x, y)
    x1, y1 = int(x0 + length * math.cos(yaw)), int(y0 + length * math.sin(yaw))
    cv2.arrowedLine(image, (x0, y0), (x1, y1), colour, thickness, cv2.LINE_AA, tipLength=0.35)


def planned_path(place_map: PlaceMap) -> list[int]:
    """The cheapest path from a route's first photo (the start) to a goal photo: the same thing
    plan_path computes, done here with networkx so the picture works before plan_path does."""
    graph = place_map.graph
    starts = [photo.index for photo in place_map.photos if photo.number == 0]
    best: list[int] = []
    for start in starts:
        for goal in place_map.goal_nodes:
            try:
                path = nx.shortest_path(graph, start, goal, weight="weight")
            except nx.NetworkXNoPath:
                continue
            if not best or nx.path_weight(graph, path, "weight") < nx.path_weight(
                graph, best, "weight"
            ):
                best = path
    return best


def draw_map(place_map: PlaceMap, answer: dict, truth: dict, path: list[int]) -> np.ndarray:
    grid = truth["grid"]
    rows, cols = len(grid), len(grid[0])
    legend_h = 95
    image = np.full((rows * CELL + legend_h, cols * CELL, 3), 255, np.uint8)
    for r, line in enumerate(grid):
        for c, ch in enumerate(line):
            colour = WALL if ch == "#" else FREE
            cv2.rectangle(
                image, (c * CELL, r * CELL), ((c + 1) * CELL - 1, (r + 1) * CELL - 1), colour, -1
            )

    photos, graph = place_map.photos, place_map.graph

    def where(node: int) -> tuple[int, int]:
        x, y, _ = answer[photos[node].key]
        return to_px(x, y)

    for i, j, kind in graph.edges(data="kind"):
        if kind == "route":
            colour = ROUTE_COLOURS[photos[i].route % len(ROUTE_COLOURS)]
            cv2.line(image, where(i), where(j), colour, 1, cv2.LINE_AA)
    for i, j, kind in graph.edges(data="kind"):
        if kind == "shortcut":
            cv2.line(image, where(i), where(j), SHORTCUT, 2, cv2.LINE_AA)
    for photo in photos:
        colour = ROUTE_COLOURS[photo.route % len(ROUTE_COLOURS)]
        cv2.circle(image, where(photo.index), 2, colour, -1, cv2.LINE_AA)

    for a, b in zip(path[:-1], path[1:], strict=True):
        cv2.line(image, where(a), where(b), PATH, 3, cv2.LINE_AA)

    sx, sy, syaw = truth["spawn"]
    cv2.rectangle(image, np.subtract(to_px(sx, sy), 7), np.add(to_px(sx, sy), 7), START, -1)
    for node in place_map.goal_nodes:
        x, y, yaw = answer[photos[node].key]
        cv2.circle(image, to_px(x, y), 7, GOAL, 2, cv2.LINE_AA)
        arrow(image, x, y, yaw, CELL * 0.7, GOAL, 1)
    tx, ty, tyaw = truth["target"]
    cv2.drawMarker(image, to_px(tx, ty), GOAL, cv2.MARKER_STAR, 22, 2, cv2.LINE_AA)
    arrow(image, tx, ty, tyaw, CELL * 1.2, GOAL, 2)

    shortcuts = sum(1 for *_, kind in graph.edges(data="kind") if kind == "shortcut")
    on_path = sum(
        1 for a, b in zip(path[:-1], path[1:], strict=True) if graph[a][b]["kind"] == "shortcut"
    )
    y0 = rows * CELL + 18
    font = cv2.FONT_HERSHEY_SIMPLEX
    items = [
        (f"route {r}", ROUTE_COLOURS[r % len(ROUTE_COLOURS)])
        for r in sorted({p.route for p in photos})
    ]
    items += [("shortcut", SHORTCUT), ("planned path", PATH), ("start", START), ("goal", GOAL)]
    x = 8
    for label, colour in items:
        width = 40 + 9 * len(label)
        if x + width > image.shape[1]:  # wrap to a second line
            x, y0 = 8, y0 + 24
        cv2.line(image, (x, y0 - 4), (x + 22, y0 - 4), colour, 3)
        cv2.putText(image, label, (x + 28, y0), font, 0.45, TEXT, 1, cv2.LINE_AA)
        x += width
    summary = (
        f"{len(photos)} photos, {shortcuts} shortcuts | path: {len(path) - 1} hops, "
        f"{on_path} shortcuts used | arrows: which way the camera faced"
    )
    cv2.putText(image, summary, (8, y0 + 28), font, 0.45, TEXT, 1, cv2.LINE_AA)
    return image


def draw_path_strip(place_map: PlaceMap, path: list[int], goal_front: np.ndarray) -> np.ndarray:
    """About a dozen photos along the path, in order, then the goal's front view."""
    photos, graph = place_map.photos, place_map.graph
    picks = sorted(set(np.linspace(0, len(path) - 1, 11).round().astype(int).tolist()))
    tiles = []
    for k in picks:
        node = path[k]
        tile = cv2.resize(read_image(photos[node]), (240, 180))
        label = f"hop {k}: {photos[node].key}"
        if k + 1 < len(path) and graph[node][path[k + 1]]["kind"] == "shortcut":
            label += " (next: shortcut)"
        cv2.rectangle(tile, (0, 0), (239, 16), (255, 255, 255), -1)
        cv2.putText(tile, label, (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT, 1, cv2.LINE_AA)
        tiles.append(tile)
    goal = cv2.resize(goal_front, (240, 180))
    cv2.rectangle(goal, (0, 0), (239, 16), (255, 255, 255), -1)
    cv2.putText(goal, "goal photo: front view", (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, GOAL, 1)
    tiles.append(goal)
    while len(tiles) % 4:
        tiles.append(np.full((180, 240, 3), 255, np.uint8))
    rows = [np.hstack(tiles[i : i + 4]) for i in range(0, len(tiles), 4)]
    return np.vstack(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data", required=True, help="a testbed folder, e.g. data/eval-21-3")
    p.add_argument("--recognizer", choices=RECOGNIZERS, default="vlad", help="whose map")
    args = p.parse_args()

    data_dir = Path(args.data)
    place_map = PlaceMap.load(map_file(data_dir, args.recognizer))
    answer = load_answer_key(data_dir)
    truth = load_truth(data_dir)
    path = planned_path(place_map)

    out = Path("results")
    out.mkdir(exist_ok=True)
    tag = data_dir.name if args.recognizer == "vlad" else f"{data_dir.name}_{args.recognizer}"
    map_png = out / f"map_{tag}.png"
    cv2.imwrite(str(map_png), draw_map(place_map, answer, truth, path))
    print(f"saved {map_png}")
    if path:
        strip = out / f"path_{tag}.jpg"
        cv2.imwrite(str(strip), draw_path_strip(place_map, path, split_goal_photo(data_dir)[0]))
        print(f"saved {strip}")
    else:
        print("no path from a start photo to a goal photo: the map is in several pieces")


if __name__ == "__main__":
    main()
