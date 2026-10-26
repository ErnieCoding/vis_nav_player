"""
Testbed: grade place recognition (the fingerprints) with the answer key. Ready to use.

For every photo of one route (the "query"), find the most similar photo among the OTHER
routes, the way the agent will for a camera frame. Then look up in the answer key whether the
two were really taken at the same spot (within 0.4 m, facing within 30 degrees). Queries whose
spot no other route ever saw are left out: they have no right answer.

    uv run source/testbed/check_place_recognition.py --data data/eval-21-3
    uv run source/testbed/check_place_recognition.py --data data/eval-21-3 --show 10

recall@1 = how often the single most similar photo was right; recall@5 = how often one of the
five most similar was. --show N saves N wrong matches side by side in results/place_recognition/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # source/

import cv2
import numpy as np
from nav.data import load_answer_key, load_photos, read_image
from nav.place_map import map_file
from nav.place_recognition import VladRecognizer

from testbed.tools import SAME_SPOT_DEG, SAME_SPOT_M


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data", required=True, help="a testbed folder, e.g. data/eval-21-3")
    p.add_argument("--every", type=int, default=1, help="keep one photo in this many")
    p.add_argument("--show", type=int, default=0, help="save this many wrong matches as images")
    args = p.parse_args()

    data_dir = Path(args.data)
    photos = load_photos(data_dir, args.every)
    answer = load_answer_key(data_dir)
    poses = np.array([answer[photo.key] for photo in photos])  # (N, 3): x, y, yaw
    routes = np.array([photo.route for photo in photos])

    # Swap in DinoRecognizer here (and in build_map.py, nav_agent.py) to compare.
    recognizer = VladRecognizer(map_file(data_dir).parent / "vlad")
    fingerprints = recognizer.fit(photos)

    # The answer key, for every pair of photos at once: same spot, different route?
    dx = poses[:, None, 0] - poses[None, :, 0]
    dy = poses[:, None, 1] - poses[None, :, 1]
    turn = np.abs((poses[:, None, 2] - poses[None, :, 2] + np.pi) % (2 * np.pi) - np.pi)
    other_route = routes[:, None] != routes[None, :]
    right = (np.hypot(dx, dy) <= SAME_SPOT_M) & (np.degrees(turn) <= SAME_SPOT_DEG) & other_route

    similarity = fingerprints @ fingerprints.T
    similarity[~other_route] = -np.inf  # only search the other routes
    top5 = np.argsort(-similarity, axis=1)[:, :5]

    answerable = right.any(axis=1)
    rows = np.arange(len(photos))
    hit1 = right[rows, top5[:, 0]]
    hit5 = right[rows[:, None], top5].any(axis=1)

    print(f"{len(photos)} photos; {answerable.sum()} have a right answer in another route")
    print(f"{'route':>5} {'queries':>8} {'recall@1':>9} {'recall@5':>9}")
    for route in sorted(set(routes)):
        mask = answerable & (routes == route)
        if mask.any():
            print(f"{route:>5} {mask.sum():>8} {hit1[mask].mean():>9.1%} {hit5[mask].mean():>9.1%}")
    print(
        f"{'all':>5} {answerable.sum():>8} {hit1[answerable].mean():>9.1%} "
        f"{hit5[answerable].mean():>9.1%}"
    )

    if args.show:
        out = Path("results") / "place_recognition"
        out.mkdir(parents=True, exist_ok=True)
        wrong = np.flatnonzero(answerable & ~hit1)
        rng = np.random.default_rng(0)
        for i in rng.choice(wrong, size=min(args.show, len(wrong)), replace=False):
            j = top5[i, 0]
            gap = float(np.hypot(*(poses[i, :2] - poses[j, :2])))
            degrees = float(np.degrees(turn[i, j]))
            pair = np.hstack([read_image(photos[i]), read_image(photos[j])])
            for row, text in enumerate(
                (
                    f"query {photos[i].key} | best match {photos[j].key}",
                    f"really {gap:.2f} m apart, facing {degrees:.0f} deg apart",
                )
            ):
                cv2.putText(
                    pair, text, (5, 15 + 15 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1
                )
            cv2.imwrite(str(out / f"wrong_{i}.jpg"), pair)
        print(f"saved {min(args.show, len(wrong))} wrong matches to {out}")


if __name__ == "__main__":
    main()
