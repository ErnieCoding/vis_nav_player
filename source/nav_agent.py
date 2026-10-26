"""
Our autonomous agent: navigation (source/nav/) and the last metre (last_metre.py) together.

    uv run source/build_map.py --local 7         # 1. before the attempt: build the map
    uv run source/nav_agent.py --local 7         # 2. drive it, and watch in the window
    uv run source/testbed/run_nav.py --mazes 3   #    or score it on practice mazes, headless

The agent is always in one of two modes:

  "navigate"     each act(): work out where we are, plan a path to the goal, make one move
                 along it; hand over to the last metre once close to the goal
  "last_metre"   each act(): LastMetre.step() decides. It returns CHECKIN when it is done,
                 or "LOST" to give control back to navigation
"""

from __future__ import annotations

from pathlib import Path

import cli
import numpy as np
from last_metre import LastMetre
from nav.data import read_image
from nav.follower import Follower
from nav.localizer import Localizer
from nav.matching import Matcher
from nav.place_map import PlaceMap, build_place_map, map_file
from nav.place_recognition import VladRecognizer
from nav.planner import plan_path
from vis_nav_sdk import WAIT, Action, Agent, Observation, SessionInfo, run

LOOKAHEAD = 3  # drive toward the photo this many nodes ahead on the path
HANDOVER_HOPS = 4  # consider handing over within this many nodes of a goal photo
MIN_CONFIDENCE = 0.5  # ...and only when the localizer is at least this sure
COOLDOWN_TICKS = 30  # after the last metre says LOST, navigate this long before retrying

HERE = (76, 209, 132)
TOWARD = (108, 180, 255)
GOAL = (245, 177, 76)


class NavAgent(Agent):
    # -- before the attempt: no clock, crashes cost nothing

    def __init__(self, data_dir: Path) -> None:
        path = map_file(data_dir)
        self.recognizer = VladRecognizer(path.parent / "vlad")  # or DinoRecognizer, later
        self.matcher = Matcher()
        if path.exists():
            self.map = PlaceMap.load(path)
        else:
            self.map = build_place_map(data_dir, self.recognizer, self.matcher)
            self.map.save(path)
        self.recognizer.fit(self.map.photos)  # reloads the cached vocabulary: quick
        self.localizer = Localizer(self.map.graph, self.map.fingerprints)
        self.follower = Follower(self.matcher)
        self.lm = LastMetre()
        self._reset_run()

    def _reset_run(self) -> None:
        self.mode = "navigate"
        self.node: int | None = None  # the photo we think we are at
        self.confidence = 0.0
        self.path: list[int] = []  # nodes from self.node to a goal photo
        self.last_step: int | None = None  # obs.step of the last frame we localized on
        self.retry_after = 0  # no handover before this step (set after a LOST)
        self.handovers = 0

    # -- once per attempt: the clock is running, keep it cheap

    def setup(self, info: SessionInfo) -> None:
        """Also called once before the attempt with random frames, as a pre-flight check."""
        self._reset_run()
        self.follower.fx = float(np.asarray(info.camera.intrinsic_matrix)[0, 0])
        self.lm.start(info)
        self.localizer.reset(self.start_nodes())

    def start_nodes(self) -> list[int]:
        """The first photo of each route. Every recorded drive started where our run starts."""
        firsts: dict[int, int] = {}
        for photo in self.map.photos:
            firsts.setdefault(photo.route, photo.index)
        return list(firsts.values())

    # -- every step: one frame in, one move out

    def act(self, obs: Observation) -> Action | tuple[Action, int]:
        if self.mode == "last_metre":
            return self._last_metre(obs)

        # 1. Where am I? Once per new frame (after a WAIT the same frame comes back).
        if obs.step != self.last_step:
            if self.last_step is not None:
                self.localizer.predict(obs.step - self.last_step)  # ticks moved since then
            self.localizer.update(self.recognizer.describe(obs.image))
            self.last_step = obs.step
        self.node, self.confidence = self.localizer.estimate()

        # 2. Which way to the goal?
        self.path = plan_path(self.map.graph, self.node, self.map.goal_nodes)
        if not self.path:
            # No goal photo can be reached from where we think we are, which usually means
            # the guess is wrong. Turn to see something new and let the localizer recover.
            return Action.LEFT, 8

        # 3. Close enough for the last metre?
        if obs.step >= self.retry_after and self.should_hand_over(obs.image):
            self.mode = "last_metre"
            self.handovers += 1
            self.lm.reset()
            return self._last_metre(obs)

        # 4. One move toward a photo a few nodes ahead on the path.
        waypoint = self.map.photos[self.path[min(LOOKAHEAD, len(self.path) - 1)]]
        return self.follower.next_move(obs.image, waypoint)

    def should_hand_over(self, image: np.ndarray) -> bool:
        """Is it time to let the last metre take over?"""
        # TODO: decide the handover. Hand over only when ALL of these hold:
        #   1. the goal is close: len(self.path) - 1 <= HANDOVER_HOPS
        #   2. we are sure where we are: self.confidence >= MIN_CONFIDENCE
        #   3. the last metre can see the goal: self.lm.wants_control(image)
        #   Check the cheap tests first; wants_control runs SIFT on the frame.
        #   Tune HANDOVER_HOPS with source/testbed/run_nav.py: its "handover" column is the
        #   true distance to the goal at the handover. The last metre wants it under ~0.5 m.
        raise NotImplementedError("NavAgent.should_hand_over")

    def _last_metre(self, obs: Observation) -> Action | tuple[Action, int]:
        """The last metre is driving."""
        move = self.lm.step(obs.image)
        if move == "LOST":  # it gives control back
            self.mode = "navigate"
            self.retry_after = obs.step + COOLDOWN_TICKS
            return WAIT  # think again about this same frame, now navigating
        if move is None:
            return WAIT  # it wants another look before deciding
        return move  # a move, or Action.CHECKIN

    # -- for the window and the testbed

    def estimated_photo_path(self) -> Path | None:
        """The photo we think we are at. The testbed compares it with the truth."""
        return None if self.node is None else self.map.photos[self.node].path

    def finish(self, result) -> None:
        print("unscored" if result is None else f"{result.goal_tier}: {result.points} points")

    def hud(self) -> list[str]:
        lines = [f"mode: {self.mode}   handovers: {self.handovers}"]
        if self.node is not None:
            photo = self.map.photos[self.node]
            lines.append(f"I think I'm at {photo.key} (confidence {self.confidence:.2f})")
            lines.append(f"{max(len(self.path) - 1, 0)} nodes to the goal")
        if self.mode == "last_metre":
            lines.append(f"last metre: {self.lm.status}")
        return lines

    def panel(self) -> list[tuple[np.ndarray, str, tuple[int, int, int]]]:
        tiles = []
        if self.node is not None:
            tiles.append((read_image(self.map.photos[self.node]), "I think I'm here", HERE))
        if len(self.path) > 1:
            waypoint = self.map.photos[self.path[min(LOOKAHEAD, len(self.path) - 1)]]
            tiles.append((read_image(waypoint), "driving toward", TOWARD))
        if self.map.goal_nodes:
            tiles.append((read_image(self.map.photos[self.map.goal_nodes[0]]), "goal photo", GOAL))
        return tiles


if __name__ == "__main__":
    parser = cli.parser(__doc__)
    parser.add_argument("--data", default=None, help="exploration data folder")
    args = parser.parse_args()
    agent = NavAgent(cli.exploration_data(args, args.data))
    run(agent, cli.challenge(args), **cli.run_options(args))
