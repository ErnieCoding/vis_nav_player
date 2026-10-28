"""Facing the goal: how far to turn to face goal view 0, and a simple finisher.

The four goal photos were all taken from the goal spot, facing front (view 0), left, back
and right, so view k faces 90k degrees left of view 0. From (or very near) the goal spot,
any goal view the camera matches gives the exact turn to face view 0: the two views differ
by a pure rotation. One catch, measured: the goal photos were taken from a camera higher
than the robot's, which shifts points up or down by an amount that depends on their
distance. A turn only shifts points sideways, so the fit uses horizontal positions only.

Accuracy of the turn to view 0 (median / 90th percentile): 0.2 / 0.5 deg on the spot,
1-2 / 3-5 deg 3 cm away, about 3 / 6-12 deg 5 cm away, 2-9 / 10-18 deg 8 cm away.

nav_agent.py uses it to check the last metre's check-ins (a check-in facing more than a few
degrees off is turned first) and, if the last metre gives up at the goal, to finish alone:
turn to face view 0 and check in where the robot is.
"""

from __future__ import annotations

import numpy as np
from vis_nav_sdk import Action

from nav.geometry import MIN_MATCHES, rotation_fit
from nav.matching import Features, Matcher

VIEW_TURN = (0.0, 90.0, 180.0, -90.0)  # view k faces this far left of view 0
FACE_TOL_DEG = 2.5  # the finisher checks in once facing view 0 within this...
SURE_EXPLAINED = 20  # ...and only on strong evidence: this many matches explained...
SURE_SHARE = 0.5  # ...and this share of them (both only hold within a few cm of the goal)
TURN_EXPLAINED = 12  # turning on weaker evidence is fine: the next look checks it
SEARCH_TICKS = 5  # one look-around turn when no goal view matches
TICK_DEG = 3.44


class Facing:
    def __init__(self, matcher: Matcher | None = None) -> None:
        self.matcher = matcher or Matcher()
        self.views: list[Features] = []
        self.status = ""

    def start(self, info) -> None:
        """Once per attempt: describe the four goal views."""
        self.views = [self.matcher.features(view) for view in info.targets]

    def turn_to_front(self, image: np.ndarray) -> tuple[float, int, int] | None:
        """Degrees to turn (+ = left) to face goal view 0, measured through whichever goal
        view a pure rotation explains best; with how many matches it explains, out of how
        many. None if no goal view matches."""
        now = self.matcher.features(image)
        best: tuple[float, int, int] | None = None
        for k, view in enumerate(self.views):
            points_now, points_view = self.matcher.match(now, view)
            if len(points_now) < MIN_MATCHES:
                continue
            turn, explained = rotation_fit(points_now, points_view, horizontal_only=True)
            if explained >= MIN_MATCHES and (best is None or explained > best[1]):
                best = ((turn - VIEW_TURN[k] + 180.0) % 360.0 - 180.0, explained, len(points_now))
        return best

    def finish(self, image: np.ndarray):
        """One step of finishing alone: turn to face view 0, then check in."""
        measured = self.turn_to_front(image)
        if measured is None or measured[1] < TURN_EXPLAINED:
            self.status = "finishing: no goal view clearly in sight, looking around"
            return (Action.LEFT, SEARCH_TICKS)
        turn, explained, matches = measured
        sure = explained >= SURE_EXPLAINED and explained >= SURE_SHARE * matches
        if abs(turn) <= FACE_TOL_DEG and sure:
            self.status = f"finishing: facing view 0 ({turn:+.1f} deg): checking in"
            return Action.CHECKIN
        if abs(turn) <= FACE_TOL_DEG:
            # Lined up by a weak fit: probably not quite at the goal. Look a little to the
            # side for a better view rather than check in on it.
            self.status = f"finishing: lined up but unsure ({explained}/{matches}), looking"
            return (Action.LEFT, 2)
        self.status = f"finishing: turning {turn:+.1f} deg to face view 0 ({explained} matches)"
        return turn_ticks(turn)


def turn_ticks(degrees: float) -> tuple[Action, int]:
    """The turn as a move: LEFT or RIGHT for the nearest number of ticks (at least 1)."""
    ticks = max(1, round(abs(degrees) / TICK_DEG))
    return (Action.LEFT if degrees > 0 else Action.RIGHT, ticks)
