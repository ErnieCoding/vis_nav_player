"""Helpers shared by the testbed scripts. Ready to use."""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import vis_nav_sim as sim

SAME_SPOT_M = 0.4  # two photos show "the same spot" if taken this close...
SAME_SPOT_DEG = 30.0  # ...and facing within this many degrees of each other

_textures = None


def textures() -> sim.Textures:
    """The wall photos, loaded once per process (the first run ever downloads 123 MB)."""
    global _textures
    if _textures is None:
        _textures = sim.Textures.find()
    return _textures


def make_world(size: int, seed: int, motion_seed: int, bias=None) -> sim.Simulator:
    """A maze rendered the way the server renders it (4x anti-aliasing), with motion noise.
    The same ``size`` and ``seed`` always give the same maze."""
    return sim.Simulator(
        textures(),
        seed,
        rows=size,
        cols=size,
        motion_noise=sim.DEFAULT_NOISE,
        motion_bias=bias,
        motion_seed=motion_seed,
        msaa=4,
    )


def eval_dir(size: int, seed: int) -> Path:
    return Path("data") / f"eval-{size}-{seed}"


def record(size: int, seed: int) -> Path:
    """``data/eval-<size>-<seed>/``: the exploration data a challenge on this maze would hand
    out, plus the answer key (``poses.json`` per route) and ``truth.json`` (start, goal,
    shortest route). Recorded once, in a few seconds; later calls return the folder."""
    out = eval_dir(size, seed)
    if (out / "truth.json").exists():
        return out
    world = make_world(size, seed, motion_seed=seed)
    summary = world.record(out, routes=3, seed=seed, capture_every=5, student_files=False)
    truth = {
        "size": size,
        "seed": seed,
        "spawn": [float(v) for v in world.spawn],
        "target": [float(v) for v in world.target],  # x, y (metres), yaw (radians)
        "oracle_steps": int(world.shortest_path["oracle_steps"]),
        "grid": list(world.grid),
    }
    (out / "truth.json").write_text(json.dumps(truth, indent=1))
    print(f"recorded {out}: {summary['frames']} photos on {summary['routes']} routes")
    return out


def load_truth(data_dir: Path) -> dict:
    return json.loads((Path(data_dir) / "truth.json").read_text())


def angle_deg(a: float, b: float) -> float:
    """The difference between two headings in radians, as 0..180 degrees."""
    d = (a - b + math.pi) % (2 * math.pi) - math.pi
    return abs(math.degrees(d))


def distance_m(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def same_spot(a, b) -> bool:
    """Were two poses ``(x, y, yaw)`` the same spot, facing about the same way?"""
    return distance_m(a, b) <= SAME_SPOT_M and angle_deg(a[2], b[2]) <= SAME_SPOT_DEG


def walking_cells(grid: list[str], a, b, cell: float = 0.2, give_up: int = 10) -> int | None:
    """How many grid cells apart positions ``a`` and ``b`` are when walking along the
    corridors (not through walls); None if more than ``give_up``. Cell (row, col) is centred
    at x = col * cell, y = row * cell."""
    start = (int(round(a[1] / cell)), int(round(a[0] / cell)))
    goal = (int(round(b[1] / cell)), int(round(b[0] / cell)))
    seen, frontier = {start}, [start]
    for steps in range(give_up + 1):
        if goal in seen:
            return steps
        nxt = []
        for r, c in frontier:
            for n in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if n not in seen and grid[n[0]][n[1]] != "#":
                    seen.add(n)
                    nxt.append(n)
        frontier = nxt
    return None


def server_like(frame: np.ndarray, world: sim.Simulator) -> np.ndarray:
    """The frame after the JPEG round trip the server's frames go through."""
    data = np.frombuffer(world.jpeg(frame), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def rubric_points(distance: float, heading: float, spl: float) -> int:
    """The course's suggested rubric: nothing unless within 0.2 m and 20 degrees; then up to
    3 for distance, 3 for heading, 4 for SPL."""
    if distance > 0.2 or heading > 20:
        return 0
    points = 3 if distance <= 0.10 else 2 if distance <= 0.15 else 1
    points += 3 if heading <= 10 else 2 if heading <= 15 else 1
    points += (
        4 if spl >= 0.95 else 3 if spl >= 0.85 else 2 if spl >= 0.75 else 1 if spl >= 0.5 else 0
    )
    return points


def parse_ints(text: str) -> list[int]:
    return [int(s) for s in text.split(",") if s.strip()]
