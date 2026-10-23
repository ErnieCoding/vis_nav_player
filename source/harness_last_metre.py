"""
Oracle test harness for the last-metre module (testing only, never part of the agent).

For each maze and trial it uses the simulator's TRUE pose to:
  1. drive the robot to a random free cell near the goal (0..--max-cells cells away, 0.2 m each),
  2. turn it to a random heading,
  3. hand control to LastMetre and let it run until it checks in, says LOST, or runs out,
  4. record the true distance and heading at check-in, as the rubric scores them.

    uv run source/harness_last_metre.py --mazes 7 --trials 3
    uv run source/harness_last_metre.py --mazes 7,8,9,10,11 --trials 5 --max-cells 2

Results print as a table and are saved to results/last_metre_runs.csv.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import vis_nav_sim as sim
from last_metre import LastMetre
from vis_nav_sdk import Action


def wrap(a: float) -> float:
    """Angle into -pi..pi."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def points(d: float, h: float) -> int:
    """The course's suggested rubric, distance + heading blocks only (SPL is navigation's)."""
    if d > 0.2 or h > 20:
        return 0
    dist = 3 if d <= 0.10 else 2 if d <= 0.15 else 1
    head = 3 if h <= 10 else 2 if h <= 15 else 1
    return dist + head


class Oracle:
    """Drives the simulator using the truth. Allowed in tests only."""

    def __init__(self, world: sim.Simulator, jpeg: bool) -> None:
        self.world, self.jpeg = world, jpeg
        self.cs = world.cell_size
        self.free = ~world.occupancy
        self.frame = None
        self._pick_cell_mapping()
        self._calibrate_turn_sign()

    # -- frames as the agent would see them

    def look(self, frame: np.ndarray) -> np.ndarray:
        if not self.jpeg:
            return frame
        data = np.frombuffer(self.world.jpeg(frame), np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)

    def step(self, action: int, repeat: int = 1) -> np.ndarray:
        self.frame = self.world.step(int(action), max(1, min(int(repeat), sim.MAX_REPEAT)))[-1]
        return self.frame

    # -- grid <-> metres (worked out from the known spawn and target cells)

    def _pick_cell_mapping(self) -> None:
        """Cell (row, col) has its centre at x = col * cell_size, y = row * cell_size
        (checked on maze 7: target (5.0, 0.4) is cell (2, 25), spawn (0.2, 2.6) is (13, 1))."""
        for pos, cell in ((self.world.target, self.world.target_cell),
                          (self.world.spawn, self.world.spawn_cell)):
            if self.cell_of(pos[0], pos[1]) != tuple(cell):
                raise RuntimeError(f"grid mapping check failed: {pos} vs cell {cell}")

    def cell_of(self, x: float, y: float) -> tuple[int, int]:
        return int(round(y / self.cs)), int(round(x / self.cs))

    def centre(self, r: int, c: int) -> tuple[float, float]:
        return c * self.cs, r * self.cs

    def _calibrate_turn_sign(self) -> None:
        yaw0 = self.world.pose()[2]
        self.step(sim.LEFT)
        self.left_sign = 1 if wrap(self.world.pose()[2] - yaw0) > 0 else -1
        self.step(sim.RIGHT)

    # -- moving

    def turn_to(self, yaw: float, tol_deg: float = 1.8, cap: int = 250) -> None:
        for _ in range(cap):
            err = wrap(yaw - self.world.pose()[2])
            if abs(err) <= math.radians(tol_deg):
                return
            ccw = err > 0
            self.step(sim.LEFT if (ccw == (self.left_sign > 0)) else sim.RIGHT)

    def drive_to(self, x: float, y: float, cap: int = 600) -> None:
        stuck, last = 0, None
        for _ in range(cap):
            px, py, yaw = self.world.pose()
            d = math.hypot(x - px, y - py)
            if d < 0.04:
                return
            want = math.atan2(y - py, x - px)
            if abs(wrap(want - yaw)) > math.radians(8):
                self.turn_to(want)
                continue
            self.step(sim.FORWARD, max(1, min(5, int(d / 0.03))))
            if last is not None and math.hypot(px - last[0], py - last[1]) < 0.005:
                stuck += 1
                if stuck > 10:
                    return
            last = (px, py)

    def bfs(self, start: tuple[int, int]) -> dict:
        """Parent links from start over free cells (4-connected)."""
        parent, q = {start: None}, deque([start])
        while q:
            r, c = q.popleft()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (r + dr, c + dc)
                if (0 <= n[0] < self.free.shape[0] and 0 <= n[1] < self.free.shape[1]
                        and self.free[n] and n not in parent):
                    parent[n] = (r, c)
                    q.append(n)
        return parent

    def go_to_cell(self, cell: tuple[int, int]) -> None:
        here = self.cell_of(*self.world.pose()[:2])
        parent = self.bfs(here)
        if cell not in parent:
            raise RuntimeError(f"cell {cell} not reachable from {here}")
        path, n = [], cell
        while n is not None:
            path.append(n)
            n = parent[n]
        for r, c in reversed(path[:-1]):  # skip the cell we're in
            self.drive_to(*self.centre(r, c))

    def cells_near_goal(self, max_cells: int) -> list[tuple[tuple[int, int], int]]:
        """Free cells within max_cells steps of the goal cell, with their step count."""
        goal = tuple(self.world.target_cell)
        parent = self.bfs(goal)
        out = []
        for cell in parent:
            k, n = 0, cell
            while parent[n] is not None:
                n, k = parent[n], k + 1
            if k <= max_cells:
                out.append((cell, k))
        return out


def run_trial(oracle: Oracle, lm: LastMetre, info, rng: random.Random, max_cells: int,
              tick_budget: int, decision_cap: int) -> dict:
    world = oracle.world
    cell, k = rng.choice(oracle.cells_near_goal(max_cells))
    oracle.go_to_cell(cell)
    oracle.turn_to(rng.uniform(-math.pi, math.pi))
    if oracle.frame is None:
        oracle.step(sim.LEFT)
    start_d, start_h = world.distance_heading()

    lm.start(info)
    frame = oracle.look(oracle.frame)
    ticks0, outcome, decisions = world.steps, "out of ticks", 0
    t0 = time.perf_counter()
    while world.steps - ticks0 < tick_budget and decisions < decision_cap:
        decisions += 1
        move = lm.step(frame)
        if isinstance(move, str):  # "LOST"
            outcome = move
            break
        if move is None:
            continue  # WAIT: same frame again
        action, n = move if isinstance(move, tuple) else (move, 1)
        if int(action) & int(Action.CHECKIN):
            outcome = "checked in"
            break
        if int(action) == 0:
            continue
        frame = oracle.look(oracle.step(int(action), n))
    d, h = world.distance_heading()
    return {
        "maze": world.seed, "start_cells": k, "start_dist_m": round(start_d, 3),
        "start_head_deg": round(start_h, 1), "outcome": outcome,
        "dist_m": round(d, 3), "head_deg": round(h, 1),
        "pass": outcome == "checked in" and d <= 0.2 and h <= 20,
        "pts": points(d, h) if outcome == "checked in" else 0,
        "ticks": world.steps - ticks0, "secs": round(time.perf_counter() - t0, 1),
        "last_status": lm.status,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mazes", default="7", help="comma-separated maze seeds, e.g. 7,8,9")
    p.add_argument("--trials", type=int, default=3, help="random starts per maze")
    p.add_argument("--max-cells", type=int, default=2, help="start up to this many 0.2 m cells away")
    p.add_argument("--ticks", type=int, default=600, help="tick budget for the last metre")
    p.add_argument("--seed", type=int, default=0, help="random seed for the starts")
    p.add_argument("--no-noise", action="store_true", help="exact motion (default: course noise)")
    p.add_argument("--no-jpeg", action="store_true", help="raw frames (default: server-like JPEG)")
    args = p.parse_args()

    rng = random.Random(args.seed)
    textures = sim.Textures.find()
    noise = None if args.no_noise else sim.DEFAULT_NOISE
    lm, rows = LastMetre(), []

    header = f"{'maze':>4} {'cells':>5} {'start m':>7} {'start deg':>9}  {'outcome':<13} {'dist m':>6} {'deg':>5} {'pts':>3} {'ticks':>5}  last status"
    print(header)
    print("-" * len(header))
    for maze in [int(s) for s in args.mazes.split(",")]:
        world = sim.Simulator(textures, maze, motion_noise=noise,
                              motion_seed=rng.randrange(2**31), msaa=4)
        for _ in range(args.trials):
            world.reset(maze, rng.randrange(2**31))
            oracle = Oracle(world, jpeg=not args.no_jpeg)
            targets = [oracle.look(t) for t in world.targets()]
            info = SimpleNamespace(targets=targets,
                                   camera=SimpleNamespace(intrinsic_matrix=world.intrinsic_matrix))
            try:
                r = run_trial(oracle, lm, info, rng, args.max_cells, args.ticks, decision_cap=3000)
            except RuntimeError as e:
                print(f"{maze:>4}  setup failed: {e}")
                continue
            rows.append(r)
            mark = "PASS" if r["pass"] else r["outcome"]
            print(f"{r['maze']:>4} {r['start_cells']:>5} {r['start_dist_m']:>7.2f} {r['start_head_deg']:>9.0f}  "
                  f"{mark:<13} {r['dist_m']:>6.2f} {r['head_deg']:>5.0f} {r['pts']:>3} {r['ticks']:>5}  {r['last_status']}")

    if rows:
        passed = [r for r in rows if r["pass"]]
        print("-" * len(header))
        print(f"passed {len(passed)}/{len(rows)}   "
              f"full marks (6 pts) {sum(r['pts'] == 6 for r in rows)}/{len(rows)}   "
              f"median ticks {int(np.median([r['ticks'] for r in rows]))}")
        for k in sorted({r["start_cells"] for r in rows}):
            group = [r for r in rows if r["start_cells"] == k]
            print(f"  starting {k} cell(s) away: passed {sum(r['pass'] for r in group)}/{len(group)}")
        out = Path("results")
        out.mkdir(exist_ok=True)
        with open(out / "last_metre_runs.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"saved {out / 'last_metre_runs.csv'}")


if __name__ == "__main__":
    main()
