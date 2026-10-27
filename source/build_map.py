"""
Build the map BEFORE the attempt. Ready to use; the steps it calls are in nav/place_map.py.

    photos -> fingerprints -> graph (route + shortcut edges) -> goal photos
    saved to cache/<data folder>/place_map.pkl, which nav_agent.py loads

    uv run source/build_map.py --local 7                 # local maze 7 (data/local-7)
    uv run source/build_map.py --data data/eval-21-3     # a testbed recording
    uv run source/build_map.py                           # the challenge in .env (downloads it)
    uv run source/build_map.py --local 7 --recognizer dino   # DINOv2 fingerprints instead

No clock runs and nothing is spent, so it can take as long as it needs. --force rebuilds.
"""

from __future__ import annotations

import cli
from nav.matching import Matcher
from nav.place_map import build_place_map, map_file
from nav.place_recognition import RECOGNIZERS, make_recognizer

if __name__ == "__main__":
    parser = cli.parser(__doc__)
    parser.add_argument("--data", default=None, help="exploration data folder")
    parser.add_argument("--every", type=int, default=1, help="keep one photo in this many")
    parser.add_argument("--force", action="store_true", help="rebuild even if a map exists")
    parser.add_argument("--recognizer", choices=RECOGNIZERS, default="vlad", help="fingerprints")
    args = parser.parse_args()

    data_dir = cli.exploration_data(args, args.data)
    path = map_file(data_dir, args.recognizer)
    if path.exists() and not args.force:
        raise SystemExit(f"{path} already exists (--force to rebuild)")

    recognizer = make_recognizer(args.recognizer, path.parent)
    place_map = build_place_map(data_dir, recognizer, Matcher(), args.every)
    place_map.save(path)
    print(f"saved {path}")
