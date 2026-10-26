"""The follower: "what move do I make now?"

The planner returns a chain of photos. The follower drives toward one of them a few nodes
ahead, the *waypoint*: turn until the waypoint photo's scene lines up with the camera, then
drive forward a few ticks, and repeat. Steering by comparing what you see with what you want
to see is called *visual servoing*.

Why not replay the moves recorded with the photos? They are rough: one recorded "FORWARD"
covers about 5 ticks of mixed motion and moved 9.5 cm on average instead of 15. Images do not
drift.
"""

from __future__ import annotations

import numpy as np
from vis_nav_sdk import Action

from nav.data import Photo
from nav.matching import Matcher

TICK_DEG = 3.44  # one LEFT or RIGHT tick turns this much
TICK_M = 0.03  # one FORWARD tick drives this far
ALIGNED_DEG = 10.0  # closer than this to lined up: drive instead of turning
MIN_MATCHES = 15  # fewer matches than this: we cannot see the waypoint's scene
FORWARD_TICKS = 4  # one forward burst


class Follower:
    def __init__(self, matcher: Matcher, fx: float = 92.0) -> None:
        self.matcher = matcher
        self.fx = fx  # focal length in pixels; nav_agent.setup() sets the real one

    def next_move(self, image: np.ndarray, waypoint: Photo) -> tuple[Action, int]:
        """One move toward ``waypoint``: ``(action, ticks)``."""
        # TODO: one move toward the waypoint photo.
        #   1. now = self.matcher.features(image)
        #      target = self.matcher.photo_features(waypoint)
        #      points_now, points_target = self.matcher.match(now, target)
        #   2. If there are at least MIN_MATCHES pairs:
        #        angle = self.matcher.bearing_deg(points_now, points_target, self.fx)
        #        |angle| > ALIGNED_DEG: turn toward it, Action.LEFT if angle > 0 else
        #                               Action.RIGHT, for round(|angle| / TICK_DEG) ticks
        #                               (at least 1)
        #        otherwise:             (Action.FORWARD, FORWARD_TICKS). Later, try
        #                               Action.FORWARD | Action.LEFT (or RIGHT) to drive and
        #                               correct a small angle in the same ticks.
        #   3. Too few pairs: the waypoint faces another way (around a corner, say). Turn a
        #      few ticks to look for it. waypoint.actions holds the move recorded at that
        #      photo: if it contains "LEFT" or "RIGHT", turning that way is a good guess.
        #   Later: if FORWARD barely changed the image, the robot is pushing against a wall.
        #   Back up a couple of ticks (Action.BACKWARD) and turn.
        raise NotImplementedError("Follower.next_move")
