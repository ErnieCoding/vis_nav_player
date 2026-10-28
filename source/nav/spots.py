"""Spots: groups of recorded photos taken from the same place, and which way each one faces.

The explorer robot turns in place a lot: about half of all consecutive photos were taken from
the same spot, only rotated. Following the photo chain one photo at a time copies all of that
turning, and a photo facing another way is often out of view, so the robot searches for it.

Two photos from one spot differ by a pure rotation about the vertical axis, which the matched
keypoints pin down to a fraction of a degree (nav/geometry.py's ``rotation_fit``). That gives

  - a same-spot test: most matches are explained by a pure rotation, and
  - each spot's *heading table*: every member photo's direction relative to the others.

With it the robot can tell how far to turn to face any member photo by recognizing any other
member photo, so at a corner it turns once, straight to the way out.

Built once per map before the attempt (about 10 ms per photo) and cached next to the map.
"""

from __future__ import annotations

import pickle
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np

from nav.data import Photo
from nav.geometry import rotation_fit
from nav.matching import Matcher

SAME_SPOT_FRACTION = 0.5  # same spot when a pure rotation explains this share of the matches
MIN_MATCHES = 8  # fewer matches than this: no verdict (treated as different spots)


def same_spot_turn(matcher: Matcher, a: Photo, b: Photo) -> float | None:
    """The turn from photo a's heading to photo b's, if both were taken from the same spot;
    None if they were not (or cannot be told apart)."""
    points_a, points_b = matcher.match(matcher.photo_features(a), matcher.photo_features(b))
    if len(points_a) < MIN_MATCHES:
        return None
    turn, explained = rotation_fit(points_a, points_b)
    return turn if explained >= SAME_SPOT_FRACTION * len(points_a) else None


@dataclass
class Spots:
    spot: np.ndarray  # spot[i]: which spot node i belongs to
    heading: np.ndarray  # heading[i]: node i's direction in degrees (+ = left), relative to
    #                      the first photo of its spot; only comparable within one spot
    members: list[list[int]]  # members[s]: the nodes of spot s

    def turn_between(self, a: int, b: int) -> float:
        """Degrees to turn (+ = left) from node a's heading to node b's (same spot only)."""
        return (self.heading[b] - self.heading[a] + 180.0) % 360.0 - 180.0


def build_spots(photos: list[Photo], graph: nx.Graph, matcher: Matcher) -> Spots:
    """Join consecutive photos of a route, and shortcut ends, that pass the same-spot test;
    each connected group is a spot. Headings follow the joins from the group's first photo."""
    links: dict[int, list[tuple[int, float]]] = {i: [] for i in range(len(photos))}
    pairs = [(i, i + 1) for i in range(len(photos) - 1) if photos[i].route == photos[i + 1].route]
    pairs += [(i, j) for i, j, kind in graph.edges(data="kind") if kind == "shortcut"]
    for k, (i, j) in enumerate(pairs):
        turn = same_spot_turn(matcher, photos[i], photos[j])
        if turn is not None:
            links[i].append((j, turn))
            links[j].append((i, -turn))
        if k % 500 == 0:
            print(f"spots: checked {k} of {len(pairs)} photo pairs")

    spot = np.full(len(photos), -1)
    heading = np.zeros(len(photos))
    members: list[list[int]] = []
    for start in range(len(photos)):
        if spot[start] >= 0:
            continue
        spot[start], group, queue = len(members), [start], deque([start])
        while queue:  # breadth-first from the first photo: headings add up along the joins
            i = queue.popleft()
            for j, turn in links[i]:
                if spot[j] < 0:
                    spot[j], heading[j] = spot[start], heading[i] + turn
                    group.append(j)
                    queue.append(j)
        members.append(group)
    sizes = [len(g) for g in members]
    print(f"spots: {len(members)} spots for {len(photos)} photos (largest {max(sizes)} photos)")
    return Spots(spot, heading, members)


def spots_file(map_path: Path) -> Path:
    return Path(map_path).with_name(Path(map_path).stem + "_spots.pkl")


def load_or_build_spots(
    map_path: Path, photos: list[Photo], graph: nx.Graph, matcher: Matcher
) -> Spots:
    """The spots of this map, built the first time and cached next to the map file."""
    path = spots_file(map_path)
    stamp = (len(photos), graph.number_of_edges())
    if path.exists():
        cached_stamp, spots = pickle.loads(path.read_bytes())
        if cached_stamp == stamp:
            return spots
    spots = build_spots(photos, graph, matcher)
    path.write_bytes(pickle.dumps((stamp, spots)))
    return spots
