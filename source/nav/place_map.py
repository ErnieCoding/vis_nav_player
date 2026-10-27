"""The map: a graph of the recorded photos, built BEFORE the attempt (by build_map.py).

A *graph* is dots joined by lines:

  - a *node* (dot) is one recorded photo, which means one spot in the maze AND the direction
    the camera faced there. Node i is photos[i].
  - an *edge* (line) between two nodes means "you can drive from one to the other in a few
    ticks". There are two kinds:
      route edges     consecutive photos of one recorded drive: the explorer robot really
                      drove from one to the next (about 5 ticks apart)
      shortcut edges  photos from different drives (or far apart in one drive) that show the
                      same spot; they join the separate drives into one connected map
  - an edge's *weight* is the cost of travelling along it (roughly, in photos ~ 5 ticks).

The graph is a ``networkx.Graph``, a ready-made Python data structure (``graph.add_edge(i, j)``,
``graph.neighbors(i)``, ``nx.shortest_path(...)``). ``PlaceMap`` bundles it with everything
else the agent needs during the attempt, and saves it all to one file.
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np

from nav.data import Photo, load_photos, split_goal_photo
from nav.matching import MIN_INLIERS, Matcher
from nav.place_recognition import most_similar

import cv2

ROUTE_WEIGHT = 1.0  # one route edge: about one photo apart (~5 ticks)
SHORTCUT_WEIGHT = 1.5  # a shortcut joins two photos of the same spot seen from a bit apart


@dataclass
class PlaceMap:
    photos: list[Photo]  # photos[i] is node i
    fingerprints: np.ndarray  # (N, D): fingerprints[i] is photos[i]'s fingerprint
    graph: nx.Graph  # nodes 0..N-1; edges have "kind" ("route"/"shortcut") and "weight"
    goal_nodes: list[int]  # photos taken at or right next to the goal, best first

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path: Path) -> PlaceMap:
        return pickle.loads(Path(path).read_bytes())


def map_file(data_dir: Path, recognizer: str = "vlad") -> Path:
    """Where the map of this data folder lives: ``cache/<folder name>/place_map.pkl``
    (``data/local-7`` -> ``cache/local-7/``, ``data/<challenge id>`` -> ``cache/<id>/``).
    A map's fingerprints only work with the recognizer that made them, so each recognizer
    has its own file: ``place_map.pkl`` for VLAD, ``place_map_dino.pkl`` for DINOv2."""
    name = "place_map.pkl" if recognizer == "vlad" else f"place_map_{recognizer}.pkl"
    return Path("cache") / Path(data_dir).name / name


def add_route_edges(graph: nx.Graph, photos: list[Photo]) -> None:
    """Join each photo to the next photo of the same recorded drive."""
    for i in range(len(photos) - 1):
        if photos[i].route == photos[i + 1].route:
            graph.add_edge(i, i + 1, kind="route", weight=ROUTE_WEIGHT)


def add_shortcut_edges(
    graph: nx.Graph,
    photos: list[Photo],
    fingerprints: np.ndarray,
    matcher: Matcher,
    candidates: int = 5,
    min_gap: int = 30,
    min_inliers: int = MIN_INLIERS,
    step: int = 8,
    max_plane: float = 0.8,
) -> int:
    """Join photos of the same spot taken at different times. Returns how many were added."""
    similarity = fingerprints @ fingerprints.T

    def features(n: int):
        return matcher.photo_features(photos[n])

    def drives_agree(i: int, j: int) -> bool:
        """Check 1: do the photos `step` further along both drives also show the same spot?
        A poster is in view for only a few photos; a corridor both drives share keeps matching."""
        for a in (i - step, i + step):
            for b in (j - step, j + step):
                if not (0 <= a < len(photos) and 0 <= b < len(photos)):
                    continue  # off the end of the list
                if photos[a].route != photos[i].route or photos[b].route != photos[j].route:
                    continue  # ran into another drive
                if matcher.same_place(features(a), features(b), min_inliers):
                    return True
        return False

    def on_one_plane(i: int, j: int) -> bool:
        """Check 2: do (almost) all the RANSAC inliers lie on one flat surface? Then it is
        probably the same poster hanging in two places. True when it cannot tell."""
        points_i, points_j = matcher.match(features(i), features(j))
        if len(points_i) < 8:
            return True
        F, mask = cv2.findFundamentalMat(points_i, points_j, cv2.FM_RANSAC, 1.0, 0.99)
        if mask is None:
            return True
        inlier = mask.ravel() == 1
        points_i, points_j = points_i[inlier], points_j[inlier]  # the inliers only
        if len(points_i) < 8:
            return True
        H, on_plane = cv2.findHomography(points_i, points_j, cv2.RANSAC, 3.0)
        return on_plane is None or on_plane.mean() >= max_plane

    count_added = 0
    for i in range(len(photos)):
        sim_sorted = np.argsort(-similarity[i])

        checked = 0
        for k in range(len(sim_sorted)):
            j = int(sim_sorted[k])

            if (
                (j == i)
                or (photos[j].route == photos[i].route and abs(j - i) < min_gap)
                or graph.has_edge(i, j)
            ):
                continue

            if (
                matcher.same_place(features(i), features(j), min_inliers)
                and not on_one_plane(i, j)
                and drives_agree(i, j)
            ):
                graph.add_edge(i, j, kind="shortcut", weight=SHORTCUT_WEIGHT)
                count_added += 1

            checked += 1
            if checked == candidates:
                break

        if i % 100 == 0:
            print(f"Progress: photo {i}, new_edges {count_added}")

    return count_added


def find_goal(
    goal_views: list[np.ndarray],
    photos: list[Photo],
    fingerprints: np.ndarray,
    recognizer,
    matcher: Matcher,
    graph: nx.Graph,
    keep: int = 3,
) -> list[int]:
    """The node numbers of the photos taken at (or right next to) the goal, best first.
    ``goal_views`` are the four goal photos: front, left, back, right."""
    # TODO: find the goal.
    #   Simplest version (start here):
    #     1. f = recognizer.describe(goal_views[0])  -- the front view's fingerprint
    #     2. nav.place_recognition.most_similar(fingerprints, f, k=20) -> 20 candidate photos
    #     3. Re-rank those 20 by matcher.count_inliers(*matcher.match(
    #            matcher.features(goal_views[0]), matcher.photo_features(photos[j])))
    #     4. Return the `keep` best node numbers.
    #   Better version: do the same for all four views. They were all taken from ONE spot,
    #   so prefer candidates that are close in the graph (few edges apart) to good matches
    #   of the other views. This rules out a look-alike wall elsewhere.
    #   source/testbed/check_map.py prints how far your goal photos really are from the goal.
    candidate_set = set()
    for view in goal_views:
        nodes, _ = most_similar(fingerprints, recognizer.describe(view), k=20)
        candidate_set.update(int(j) for j in nodes)

    view_features = [matcher.features(view) for view in goal_views]
    score = {}
    for j in candidate_set:
        photo_features = matcher.photo_features(photos[j])
        counts = [matcher.count_inliers(*matcher.match(f, photo_features)) for f in view_features]
        score[j] = max(counts)

    # Graph check
    def has_support(j: int) -> bool:
        near = nx.single_source_shortest_path_length(graph, j, cutoff=2)
        return any(n != j and n in score for n in near)

    supported = [j for j in score if has_support(j)] or list(score)

    supported.sort(key=score.get, reverse=True)
    return supported[:keep]


def build_place_map(data_dir: Path, recognizer, matcher: Matcher, every: int = 1) -> PlaceMap:
    """Photos -> fingerprints -> graph -> goal. Runs before the attempt, so it can be slow."""
    t0 = time.perf_counter()
    photos = load_photos(data_dir, every)
    print(f"{len(photos)} photos from {len({p.route for p in photos})} routes in {data_dir}")

    fingerprints = recognizer.fit(photos)
    print(f"fingerprints: {fingerprints.shape} ({time.perf_counter() - t0:.0f} s)")

    graph = nx.Graph()
    graph.add_nodes_from(range(len(photos)))
    add_route_edges(graph, photos)
    print(f"route edges: {graph.number_of_edges()}")
    added = add_shortcut_edges(graph, photos, fingerprints, matcher)
    print(f"shortcut edges: {added} ({time.perf_counter() - t0:.0f} s)")
    pieces = nx.number_connected_components(graph)
    print(f"the map is in {pieces} piece(s){'' if pieces == 1 else ': some routes never join'}")

    goal_nodes = find_goal(
        split_goal_photo(data_dir), photos, fingerprints, recognizer, matcher, graph
    )
    print(f"goal photos: {[photos[n].key for n in goal_nodes]}")
    print(f"built in {time.perf_counter() - t0:.0f} s")
    return PlaceMap(photos, fingerprints, graph, goal_nodes)
