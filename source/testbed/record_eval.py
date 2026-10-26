"""
Testbed: record practice mazes WITH an answer key. Ready to use.

A challenge's exploration data never says where its photos were taken. The simulator on your
machine knows, so this records the same kind of folder a challenge hands out, plus one extra
file per route, poses.json: where each photo was really taken (x, y in metres, yaw in
radians). The agent never reads it. The testbed uses it to grade the agent's guesses.

    uv run source/testbed/record_eval.py                                # 21x21 mazes 3, 7, 11
    uv run source/testbed/record_eval.py --seeds 2026,4242 --sizes 21,31,41

Writes data/eval-<size>-<seed>/ (a few seconds each).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # source/

from testbed.tools import parse_ints, record

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--seeds", default="3,7,11", help="maze seeds, comma-separated")
    p.add_argument("--sizes", default="21", help="maze sizes (cells per side): 21, 31, 41")
    args = p.parse_args()
    for size in parse_ints(args.sizes):
        for seed in parse_ints(args.seeds):
            print(record(size, seed))
