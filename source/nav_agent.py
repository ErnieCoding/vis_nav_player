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

Navigation moves spot to spot (nav/spots.py): the planned path is a chain of photos, many of
them taken from the same spot while the explorer turned. The next *leg* skips past those to
where the path leaves the current spot, faces the way the explorer left (the last photo
before leaving), and drives: forward if the path follows a recorded drive, backward if it
runs backwards along one.
"""

from __future__ import annotations

from pathlib import Path

import cli
import numpy as np
from last_metre import LastMetre
from nav.data import read_image
from nav.follower import Follower, Leg
from nav.homing import Facing, turn_ticks
from nav.localizer import Localizer
from nav.matching import Matcher
from nav.place_map import PlaceMap, build_place_map, consistent_goals, map_file
from nav.place_recognition import RECOGNIZERS, make_recognizer
from nav.planner import plan_path
from nav.spots import load_or_build_spots
from vis_nav_sdk import WAIT, Action, Agent, Observation, SessionInfo, run

HANDOVER_SPOTS = 1  # hand over at the goal photos' spot or one before it (within ~0.1 m)
MIN_CONFIDENCE = 0.5  # ...and only when the localizer is at least this sure
COOLDOWN_TICKS = 30  # after the last metre says LOST, navigate this long before retrying
LM_MAX_TICKS = 300  # the last metre gets this many ticks per handover, then it counts as LOST
CHECKIN_TOL_DEG = 5.0  # a last-metre check-in facing further off view 0 is turned first
LM_TRIES = 3  # after this many handovers end without a check-in, finish alone (Facing)
WAIT_AT_GOAL = 60  # decisions at the goal spot without a handover: finish alone too
FINISH_MAX = 80  # finishing alone: check in after this many decisions whatever happens
STALL_DECISIONS = 40  # no spot closer to the goal for this many decisions: shake loose
STALL_TURN_TICKS = 26  # ...by backing off and turning this much (~90 deg)
MAX_MEMBERS = 12  # spot photos tried per decision to recognize which way we face
STRAIGHT_SPOTS = 3  # this many one-photo spots ahead along one drive: a straight stretch
LEG_COMMIT = 12  # keep a leg this many decisions at most, unless its next spot is reached
BLOCKED_SHORTCUT_COST = 1000.0  # weight given to a shortcut we got stuck at (see stuck())
BLAME_SPOTS = 4  # ...if it is within this many spots of us

HERE = (76, 209, 132)
TOWARD = (108, 180, 255)
GOAL = (245, 177, 76)


class NavAgent(Agent):
    def __init__(self, data_dir: Path, recognizer: str = "vlad") -> None:
        path = map_file(data_dir, recognizer)
        self.recognizer = make_recognizer(recognizer, path.parent)
        self.matcher = Matcher()
        if path.exists():
            self.map = PlaceMap.load(path)
        else:
            self.map = build_place_map(data_dir, self.recognizer, self.matcher)
            self.map.save(path)
        self.recognizer.fit(self.map.photos)  # reloads the cached vocabulary: quick
        self.goal_nodes = consistent_goals(self.map.graph, self.map.goal_nodes)
        self.weights = {(a, b): w for a, b, w in self.map.graph.edges(data="weight")}
        self.spots = load_or_build_spots(path, self.map.photos, self.map.graph, self.matcher)
        self.localizer = Localizer(self.map.graph, self.map.fingerprints)
        self.follower = Follower(self.matcher)
        self.lm = LastMetre()
        self.facing = Facing(self.matcher)
        self._reset_run()

    def _reset_run(self) -> None:
        self.mode = "navigate"
        self.node: int | None = None  # the photo we think we are at
        self.confidence = 0.0
        self.path: list[int] = []  # nodes from self.node to a goal photo
        self.spots_left = 0  # spot changes along self.path
        self.leg: Leg | None = None
        self.leg_age = 0  # decisions the current leg has been kept
        self.last_step: int | None = None  # obs.step of the last frame we localized on
        self.retry_after = 0  # no handover before this step (set after a LOST)
        self.handovers = 0
        self.handover_step = 0  # obs.step of the latest handover
        self.lm_failures = 0  # handovers that ended without a check-in
        self.at_goal = 0  # decisions at the goal spot without handing over
        self.finishing = 0  # decisions spent finishing alone
        self.best_spots: int | None = None  # fewest spots to the goal so far (since a stall)
        self.stalled = 0  # decisions since best_spots last improved
        self.blocked_shortcuts = 0  # shortcuts given up on this attempt
        self.status = ""  # what navigation is doing (for the window and the testbed log)

    def setup(self, info: SessionInfo) -> None:
        """Also called once before the attempt with random frames, as a pre-flight check."""
        self._reset_run()
        for (a, b), weight in self.weights.items():  # undo the last attempt's repairs
            self.map.graph[a][b]["weight"] = weight
        self.follower.fx = float(np.asarray(info.camera.intrinsic_matrix)[0, 0])
        self.follower.reset()
        self.lm.start(info)
        self.facing.start(info)
        self.localizer.reset(self.start_nodes())

    def start_nodes(self) -> list[int]:
        """The first photo of each route. Every recorded drive started where our run starts."""
        firsts: dict[int, int] = {}
        for photo in self.map.photos:
            firsts.setdefault(photo.route, photo.index)
        return list(firsts.values())

    def act(self, obs: Observation) -> Action | tuple[Action, int]:
        if self.mode == "last_metre":
            return self._last_metre(obs)
        if self.mode == "finish":
            return self._finish(obs)

        # 1. Where am I? Once per new frame (after a WAIT the same frame comes back).
        if obs.step != self.last_step:
            if self.last_step is not None:
                self.localizer.predict(obs.step - self.last_step)  # ticks moved since then
            self.localizer.update(self.recognizer.describe(obs.image))
            self.last_step = obs.step
        self.node, self.confidence = self.localizer.estimate()

        # 2. Which way to the goal?
        self.path = plan_path(self.map.graph, self.node, self.goal_nodes)
        if not self.path:
            # No goal photo can be reached from where we think we are, which usually means
            # the guess is wrong. Turn to see something new and let the localizer recover.
            return Action.LEFT, 8
        leg, self.spots_left = self.plan_leg()
        if not self.keep_leg():
            self.leg, self.leg_age = leg, 0

        # 3. Close enough for the last metre? If it has given up twice, or will not take
        # over at the goal spot, finish alone: face view 0 and check in here.
        self.at_goal = self.at_goal + 1 if self.spots_left == 0 else 0
        if self.spots_left <= HANDOVER_SPOTS and (
            self.lm_failures >= LM_TRIES or self.at_goal >= WAIT_AT_GOAL
        ):
            self.mode = "finish"
            return self._finish(obs)
        if obs.step >= self.retry_after and self.should_hand_over(obs.image):
            self.mode = "last_metre"
            self.handovers += 1
            self.handover_step = obs.step
            self.lm.reset()
            return self._last_metre(obs)

        # 4. One move along the leg. If no spot closer to the goal has been reached for a
        # while, something is stuck (an arc pushing into a wall, a loop...): shake loose.
        if self.best_spots is None or self.spots_left < self.best_spots:
            self.best_spots, self.stalled = self.spots_left, 0
        else:
            self.stalled += 1
        if self.stalled >= STALL_DECISIONS:
            self.stuck()
            self.best_spots, self.stalled = self.spots_left, 0
        move = self.follower.next_move(obs.image, self.leg)
        way = " backward" if self.leg.reverse else ""
        self.status = (
            f"{self.spots_left} spots to go; face {self.leg.align.key}{way}: {self.follower.status}"
        )
        return move

    def stuck(self) -> None:
        """No progress for a while. If the path takes a shortcut just ahead, that shortcut is
        the likely culprit: a wrong one joins two look-alike places a wall apart, and the
        planner keeps sending us to it. Make it too expensive to use for the rest of this
        attempt, so the path goes another way. Either way, back off and turn."""
        spot = self.spots.spot
        spots_seen = 0
        for a, b in zip(self.path, self.path[1:], strict=False):
            if self.map.graph[a][b]["kind"] == "shortcut":
                self.map.graph[a][b]["weight"] = BLOCKED_SHORTCUT_COST
                self.blocked_shortcuts += 1
                break
            spots_seen += spot[a] != spot[b]
            if spots_seen > BLAME_SPOTS:
                break
        self.follower.escape(Action.BACKWARD, STALL_TURN_TICKS, "no progress")

    def keep_leg(self) -> bool:
        """Keep driving the current leg? The localizer often wobbles between neighbouring
        photos, and re-planning on every wobble flips the target heading by 45+ degrees
        (measured: 55-111 times a run), so the robot turns back and forth. Keep the leg
        until the localizer reaches the spot it heads for, or LEG_COMMIT decisions."""
        leg = self.leg
        if leg is None or not leg.drive or self.leg_age >= LEG_COMMIT:
            return False
        if self.spots.spot[self.node] == leg.end_spot or leg.align.index not in self.path:
            return False
        self.leg_age += 1
        return True

    def plan_leg(self) -> tuple[Leg, int]:
        """The next leg of the path, and how many spot changes are left to the goal."""
        spot = self.spots.spot
        path = self.path
        changes = sum(spot[a] != spot[b] for a, b in zip(path, path[1:], strict=False))
        here = spot[path[0]]
        k = 1
        while k < len(path) and spot[path[k]] == here:
            k += 1
        if k == len(path):  # the goal photos' spot: line up with the goal photo, no driving
            return self._leg(path[-1], reverse=False, drive=False), 0
        before, after = path[k - 1], path[k]
        if self.map.graph[before][after]["kind"] == "route":
            # Leaving the spot along a recorded drive. Forward: the explorer left `before`
            # facing the way it drove. Backward (after = before - 1): it arrived at `before`
            # facing away from `after`, so face that way and back up.
            leg = self._leg(before, reverse=after < before)
            leg.straight = self.straight_ahead(k)
            leg.start_spot, leg.end_spot = here, spot[after]
            return leg, changes
        # A shortcut to another spot: steer by the photo on the other side, and drive the way
        # the path then follows that photo's drive (backward if it runs backwards along it).
        reverse = False
        if k + 1 < len(path) and self.map.graph[after][path[k + 1]]["kind"] == "route":
            reverse = path[k + 1] < after
        leg = self._leg(after, reverse=reverse)
        leg.start_spot, leg.end_spot = here, spot[after]
        return leg, changes

    def straight_ahead(self, k: int) -> bool:
        """Is path[k:] a straight stretch of one recorded drive for the next few spots? The
        explorer turns in place at corners, which leaves spots of several photos; driving
        straight leaves one photo per spot, ~10 cm apart."""
        path, spots = self.path, self.spots
        step = path[k] - path[k - 1]  # +1 forward along the drive, -1 backward
        for a, b in zip(
            path[k - 1 : k - 1 + STRAIGHT_SPOTS], path[k : k + STRAIGHT_SPOTS], strict=False
        ):
            if b - a != step or self.map.graph[a][b]["kind"] != "route":
                return False
            if len(spots.members[spots.spot[b]]) > 1:
                return False
        return k + STRAIGHT_SPOTS <= len(path)

    def _leg(self, align: int, reverse: bool, drive: bool = True) -> Leg:
        """Line up with photo ``align``. If it is from the spot we are at, recognize any of the
        spot's photos (nearest heading first) and use their known turns to ``align``."""
        if self.spots.spot[align] == self.spots.spot[self.node]:
            members = self.spots.members[self.spots.spot[align]]
            members = sorted(members, key=lambda m: abs(self.spots.turn_between(m, align)))
            members = members[:MAX_MEMBERS]
        else:
            members = [align]
        return Leg(
            align=self.map.photos[align],
            reverse=reverse,
            members=[self.map.photos[m] for m in members],
            turns=[self.spots.turn_between(m, align) for m in members],
            drive=drive,
        )

    def should_hand_over(self, image: np.ndarray) -> bool:
        """Is it time to let the last metre take over? Only when the goal is a few spots away,
        we are sure where we are, and the last metre can see a goal view. The cheap checks
        come first; wants_control runs SIFT on the frame. Tune HANDOVER_SPOTS with
        source/testbed/run_nav.py: its "handover" column is the true distance to the goal at
        the handover, which the last metre wants under ~0.5 m."""
        if self.spots_left > HANDOVER_SPOTS:
            return False
        if self.confidence < MIN_CONFIDENCE:
            return False
        return self.lm.wants_control(image)

    def _last_metre(self, obs: Observation) -> Action | tuple[Action, int]:
        """The last metre is driving. After LM_MAX_TICKS without a check-in, navigation takes
        control back as if it had said LOST (a module stuck turning would otherwise spend the
        whole step budget)."""
        if obs.step - self.handover_step > LM_MAX_TICKS:
            move = "LOST"
            self.lm.status = f"no check-in after {LM_MAX_TICKS} ticks: navigation takes over"
        else:
            move = self.lm.step(obs.image)
        if move == "LOST":  # it gives control back
            self.mode = "navigate"
            self.lm_failures += 1
            self.retry_after = obs.step + COOLDOWN_TICKS
            return WAIT  # think again about this same frame, now navigating
        if move is None:
            return WAIT  # it wants another look before deciding
        if move == Action.CHECKIN:
            # One check-in, scored once: make sure we face view 0 first.
            measured = self.facing.turn_to_front(obs.image)
            if measured is None:
                self.lm.status = "check-in held: no goal view confirms the heading, looking"
                return (Action.LEFT, 2)
            if abs(measured[0]) > CHECKIN_TOL_DEG:
                self.lm.status = f"check-in held: {measured[0]:+.1f} deg off view 0, turning"
                return turn_ticks(measured[0])
        return move  # a move, or Action.CHECKIN

    def _finish(self, obs: Observation) -> Action | tuple[Action, int]:
        """Finishing alone at the goal spot: turn to face view 0, then check in."""
        self.finishing += 1
        if self.finishing > FINISH_MAX:
            self.facing.status = "finishing: out of patience, checking in here"
            move = Action.CHECKIN
        else:
            move = self.facing.finish(obs.image)
        self.status = self.facing.status
        return move

    # -- for the window and the testbed

    def estimated_photo_path(self) -> Path | None:
        """The photo we think we are at. The testbed compares it with the truth."""
        return None if self.node is None else self.map.photos[self.node].path

    def finish(self, result) -> None:
        print("unscored" if result is None else f"{result.goal_tier}: {result.points} points")

    def hud(self) -> list[str]:
        lines = [
            f"mode: {self.mode}   handovers: {self.handovers}   "
            f"shortcuts given up: {self.blocked_shortcuts}"
        ]
        if self.node is not None:
            photo = self.map.photos[self.node]
            lines.append(f"I think I'm at {photo.key} (confidence {self.confidence:.2f})")
        if self.mode == "last_metre":
            lines.append(f"last metre: {self.lm.status}")
        elif self.mode == "finish":
            lines.append(self.facing.status)
        elif self.status:
            lines.append(self.status)
        return lines

    def panel(self) -> list[tuple[np.ndarray, str, tuple[int, int, int]]]:
        tiles = []
        if self.node is not None:
            tiles.append((read_image(self.map.photos[self.node]), "I think I'm here", HERE))
        if self.leg is not None and self.mode == "navigate":
            tiles.append((read_image(self.leg.align), "facing this way", TOWARD))
        if self.goal_nodes:
            tiles.append((read_image(self.map.photos[self.goal_nodes[0]]), "goal photo", GOAL))
        return tiles


if __name__ == "__main__":
    parser = cli.parser(__doc__)
    parser.add_argument("--data", default=None, help="exploration data folder")
    parser.add_argument("--recognizer", choices=RECOGNIZERS, default="vlad", help="fingerprints")
    args = parser.parse_args()
    agent = NavAgent(cli.exploration_data(args, args.data), args.recognizer)
    run(agent, cli.challenge(args), **cli.run_options(args))
