"""
Testbed: run whole attempts on practice mazes and score them with the answer key. Ready to use.

Like the server: 4x anti-aliasing, JPEG frames, motion noise (--bias adds a per-run motion
bias as well). One line per run, then a summary.

    uv run source/testbed/run_nav.py --agent baseline --mazes 3     # today's my_agent.py
    uv run source/testbed/run_nav.py --mazes 3,7,11                 # nav_agent.py
    uv run source/testbed/run_nav.py --mazes 3 --runs 3 --log       # every step to results/

Columns:
  outcome     checked in / out of steps / crashed (with the traceback above it)
  dist, head  metres and degrees from the goal pose at check-in
  pts         rubric points (0 unless within 0.2 m and 20 degrees)
  ticks/best  ticks used, and the shortest possible route's ticks
  spl         best / ticks, the efficiency score (0 for a miss)
  loc_ok      how often the agent's "I am at this photo" was really within 0.4 m of the robot
  handover    true distance to the goal when the agent first handed over to the last metre
"""

from __future__ import annotations

import argparse
import csv
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # source/

import numpy as np
import vis_nav_sim as sim
from nav.data import load_answer_key, photo_key
from nav.place_recognition import RECOGNIZERS
from vis_nav_sdk import Action
from vis_nav_sdk.session import Camera, Limits, Observation, SessionInfo

from testbed.tools import distance_m, make_world, parse_ints, record, rubric_points, server_like


def make_agent(kind: str, data_dir: Path, recognizer: str = "vlad"):
    if kind == "nav":
        from nav_agent import NavAgent

        return NavAgent(data_dir, recognizer)
    from my_agent import MyAgent

    return MyAgent(data_dir, cache_dir=Path("cache") / data_dir.name / "baseline")


def guessed_photo(agent) -> str | None:
    """The photo the agent currently thinks it is at, as ``"traj_0/12.jpg"``."""
    if hasattr(agent, "estimated_photo_path"):
        path = agent.estimated_photo_path()
        return None if path is None else photo_key(path)
    current = getattr(agent, "current", None)  # the baseline's best match
    if current is not None and hasattr(agent, "frames"):
        return photo_key(agent.frames[current].path)
    return None


def cell(row: dict, key: str, width: int, spec: str) -> str:
    value = row.get(key)
    return f"{'-':>{width}}" if value is None else f"{value:>{width}{spec}}"


def run_once(agent, world: sim.Simulator, max_steps: int, answer: dict, log: list) -> dict:
    """One attempt, from the spawn until the agent checks in or the steps run out."""
    info = SessionInfo(
        session_id="testbed",
        challenge={"id": "testbed", "name": f"maze {world.seed}"},
        camera=Camera(320, 240, "bgr", "jpeg", np.asarray(world.intrinsic_matrix)),
        limits=Limits(max_steps, sim.MAX_REPEAT, 0, None, 120.0, 300.0, 3600.0),
        targets=[server_like(view, world) for view in world.targets()],
    )
    agent.setup(info)
    image = server_like(world.initial_frame(), world)
    outcome, decisions, logged_step = "out of steps", 0, -1
    errors: list[float] = []
    handover = None

    while world.steps < max_steps:
        decisions += 1
        if decisions > 4 * max_steps:
            outcome = "stuck waiting"  # WAIT forever without moving
            break
        reply = agent.act(Observation(image, world.steps, max_steps - world.steps))

        # What the agent believes about the frame it just saw, against the truth.
        if world.steps != logged_step:
            logged_step = world.steps
            x, y, yaw = world.pose()
            guess = guessed_photo(agent)
            error = distance_m(answer[guess], (x, y)) if guess in answer else None
            if error is not None:
                errors.append(error)
            log.append(
                {
                    "step": world.steps,
                    "x": round(x, 3),
                    "y": round(y, 3),
                    "yaw": round(yaw, 3),
                    "guess": guess,
                    "error_m": None if error is None else round(error, 3),
                    "mode": getattr(agent, "mode", ""),
                }
            )
        if handover is None and getattr(agent, "mode", None) == "last_metre":
            handover = world.distance_heading()

        action, ticks = reply if isinstance(reply, tuple) else (reply, 1)
        if int(action) & int(Action.CHECKIN):
            outcome = "checked in"
            break
        if int(action) & int(Action.QUIT):
            outcome = "quit"
            break
        if ticks <= 0:
            continue  # WAIT: the same frame again, no tick spent
        ticks = min(int(ticks), sim.MAX_REPEAT, max_steps - world.steps)
        image = server_like(world.step(int(action), ticks)[-1], world)

    distance, heading = world.distance_heading()
    arrived = outcome == "checked in" and distance <= 0.2 and heading <= 20
    spl = world.spl() if arrived else 0.0
    return {
        "outcome": outcome,
        "dist": round(distance, 3),
        "head": round(heading, 1),
        "pts": rubric_points(distance, heading, spl) if outcome == "checked in" else 0,
        "ticks": world.steps,
        "best": int(world.shortest_path["oracle_steps"]),
        "spl": round(spl, 3),
        "loc_ok": round(float(np.mean(np.array(errors) <= 0.4)), 3) if errors else None,
        "handover_m": None if handover is None else round(handover[0], 2),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--agent", choices=["nav", "baseline"], default="nav")
    p.add_argument("--mazes", default="3", help="maze seeds, comma-separated")
    p.add_argument("--size", type=int, default=21, help="cells per side: 21, 31 or 41")
    p.add_argument("--runs", type=int, default=1, help="attempts per maze (fresh motion noise)")
    p.add_argument("--max-steps", type=int, default=10000, help="the step budget")
    p.add_argument("--bias", default=None, help="motion bias sigmas 'speed,turn,curve'")
    p.add_argument("--log", action="store_true", help="write every step to results/")
    p.add_argument(
        "--recognizer",
        choices=RECOGNIZERS,
        default="vlad",
        help="fingerprints for --agent nav (the baseline always uses its own)",
    )
    args = p.parse_args()

    bias = tuple(float(v) for v in args.bias.split(",")) if args.bias else None
    out = Path("results")
    out.mkdir(exist_ok=True)
    rows = []
    header = (
        f"{'maze':>4} {'run':>3}  {'outcome':<14} {'dist':>5} {'head':>5} {'pts':>3} "
        f"{'ticks':>5} {'best':>5} {'spl':>5} {'loc_ok':>6} {'handover':>8}"
    )

    for seed in parse_ints(args.mazes):
        data_dir = record(args.size, seed)
        answer = load_answer_key(data_dir)
        try:
            agent = make_agent(args.agent, data_dir, args.recognizer)
        except Exception:
            traceback.print_exc()
            print(f"maze {seed}: the agent could not be built (see the error above)")
            continue
        print(header)
        for run in range(args.runs):
            world = make_world(args.size, seed, motion_seed=1000 + run, bias=bias)
            log: list = []
            try:
                row = run_once(agent, world, args.max_steps, answer, log)
            except Exception as exc:
                traceback.print_exc()
                row = {"outcome": f"crashed: {type(exc).__name__}", "ticks": world.steps}
            row = {"maze": seed, "run": run, **row}
            rows.append(row)
            print(
                f"{seed:>4} {run:>3}  {row['outcome']:<14} {cell(row, 'dist', 5, '.2f')} "
                f"{cell(row, 'head', 5, '.0f')} {cell(row, 'pts', 3, 'd')} "
                f"{cell(row, 'ticks', 5, 'd')} {cell(row, 'best', 5, 'd')} "
                f"{cell(row, 'spl', 5, '.2f')} {cell(row, 'loc_ok', 6, '.0%')} "
                f"{cell(row, 'handover_m', 8, '.2f')}"
            )
            if args.log and log:
                with open(out / f"nav_steps_{args.size}-{seed}-{run}.csv", "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(log[0]))
                    writer.writeheader()
                    writer.writerows(log)

    if rows:
        arrived = [r for r in rows if r.get("pts", 0) > 0]
        print(
            f"\narrived {len(arrived)}/{len(rows)}   "
            f"mean points {np.mean([r.get('pts', 0) for r in rows]):.1f}/10"
        )
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with open(out / "nav_runs.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved {out / 'nav_runs.csv'}")


if __name__ == "__main__":
    main()
