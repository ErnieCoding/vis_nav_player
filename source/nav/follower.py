"""The follower: "what move do I make now?"

nav_agent.py turns the planned path into a *leg*: the photo to line up with (``align``) and
whether to drive forward or backward from there. The follower turns until the camera faces
the way that photo faces, then drives a few ticks, and repeats. Steering by comparing what
you see with what you want to see is called *visual servoing*.

How the turn is measured: the leg also lists the other photos taken from the same spot as
``align``, each with its known turn to ``align`` (nav/spots.py's heading table). The camera
frame is matched against them; any member that matches gives the exact turn, so at a corner
the robot recognizes where it is facing and turns once, straight to the way out, even when
``align`` itself is out of view.

Why not replay the moves recorded with the photos? They are rough: one recorded "FORWARD"
covers about 5 ticks of mixed motion and moved 9.5 cm on average instead of 15. Images do not
drift.

Safety nets:

  blocked   A straight drive that barely changed the image hit a wall: back off, turn ~45 deg.
  spinning  A circle's worth of turning without driving (nothing at this spot recognized):
            drive a few ticks to see something new.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from vis_nav_sdk import Action

from nav.data import Photo
from nav.geometry import relative_pose, rotation_fit
from nav.matching import Matcher

TICK_DEG = 3.44  # one LEFT or RIGHT tick turns this much
TICK_M = 0.03  # one FORWARD tick drives this far
ALIGNED_DEG = 20.0  # closer than this to lined up: drive (correcting in an arc) instead of turning
ARC_DEG = 3.0  # between this and ALIGNED_DEG: drive and turn in the same ticks (an arc)
MIN_EXPLAINED = 8  # a spot photo counts as recognized when the rotation explains this many
GOOD_EXPLAINED = 20  # ...and this many is good enough to stop trying the other members
FORWARD_TICKS = 4  # one forward burst...
LONG_TICKS = 8  # ...or this long on a straight stretch (Leg.straight)
SEARCH_TICKS = 5  # one search turn
MAX_TURN_TICKS = 30  # the most turned in one decision (~100 deg), so the result is re-checked
WEAK_TURN_TICKS = 9  # ...or ~30 deg on weak evidence (< GOOD_EXPLAINED): measured, a quarter
#                       of weak readings are > 20 deg off, so turn part way and look again
TURN_LIMIT = 105  # ticks of turning in place without a drive (~1 circle): then drive anyway
STUCK_DIFF = 14.0  # a straight drive that changed the mean pixel by less than this hit a wall
BACK_OFF_TICKS = 3  # blocked: back off this far...
ESCAPE_TICKS = 13  # ...then turn this much (~45 deg)

OPPOSITE = {Action.LEFT: Action.RIGHT, Action.RIGHT: Action.LEFT}
DRIVES = Action.FORWARD | Action.BACKWARD


@dataclass
class Leg:
    """One stretch of the path: face the way ``align`` faces, then drive."""

    align: Photo  # line up with this photo's heading...
    reverse: bool  # ...then drive BACKWARD (the path runs backwards along a recorded drive)
    members: list[Photo]  # photos from align's spot, to recognize which way we face
    turns: list[float]  # turns[k]: degrees from members[k]'s heading to align's (+ = left)
    drive: bool = True  # False at the goal's spot: line up only, the last metre takes over
    straight: bool = False  # a straight stretch of the recorded drive lies ahead: longer bursts
    start_spot: int = -1  # the spot this leg leaves...
    end_spot: int = -1  # ...and the one it heads for (nav_agent keeps the leg until there)


class Follower:
    def __init__(self, matcher: Matcher, fx: float = 92.0) -> None:
        self.matcher = matcher
        self.fx = fx  # focal length in pixels; nav_agent.setup() sets the real one
        self.reset()

    def reset(self) -> None:
        """Start of an attempt: no search in progress, nothing remembered."""
        self.searching = False  # turning to find something we recognize
        self.search_turn = Action.LEFT  # which way the current (or last) search turns
        self.turn_ticks = 0  # ticks turned in place since the last drive
        self.queue: list[tuple[Action, int]] = []  # moves to make before steering again
        self.last_image: np.ndarray | None = None
        self.check_wall = False  # was the last move a straight drive worth checking?
        self.last_drive = Action.FORWARD  # the last move's action
        self.status = ""  # why the last move was chosen (for the window and the testbed log)
        self.escape_reason = ""

    def next_move(self, image: np.ndarray, leg: Leg) -> tuple[Action, int]:
        """One move along ``leg``: ``(action, ticks)``."""
        if self._blocked(image):
            back = Action.BACKWARD if self.last_drive == Action.FORWARD else Action.FORWARD
            self.escape(back, ESCAPE_TICKS, "blocked by a wall")
        if self.queue:  # recovery moves: never checked for walls, never counted as spinning
            move, self.check_wall = self.queue.pop(0), False
            self.status = f"escape ({self.escape_reason})"
        else:
            move = self._steer(image, leg)
            if move[0] & DRIVES:
                self.turn_ticks = 0
            else:
                self.turn_ticks += move[1]
                if self.turn_ticks > TURN_LIMIT and leg.drive:
                    # A circle's worth of turning without driving: nothing here is recognized
                    # (or it keeps slipping away). Drive to get a new view; a wall in the way
                    # is caught next step.
                    self.turn_ticks, self.searching = 0, False
                    move = (Action.FORWARD, FORWARD_TICKS)
                    self.status = "spin limit: drive for a new view"
            self.check_wall = move[0] in (Action.FORWARD, Action.BACKWARD) and move[1] >= 3
        self.last_image, self.last_drive = image, move[0]
        return move

    def heading_error(self, image: np.ndarray, leg: Leg) -> tuple[float, int] | None:
        """Degrees to turn (+ = left) to face the way ``leg.align`` faces, measured through
        whichever photo of the spot the camera recognizes best; with how many matches the
        rotation explains. None if no photo of the spot is recognized."""
        now = self.matcher.features(image)
        best: tuple[float, int] | None = None
        most: tuple[int, np.ndarray, np.ndarray, float] | None = None  # for the fallback
        for member, member_to_align in zip(leg.members, leg.turns, strict=True):
            points_now, points_member = self.matcher.match(now, self.matcher.photo_features(member))
            if len(points_now) < MIN_EXPLAINED:
                continue
            if most is None or len(points_now) > most[0]:
                most = (len(points_now), points_now, points_member, member_to_align)
            to_member, explained = rotation_fit(points_now, points_member)
            if explained >= MIN_EXPLAINED and (best is None or explained > best[1]):
                best = ((to_member + member_to_align + 180.0) % 360.0 - 180.0, explained)
                if explained >= GOOD_EXPLAINED:
                    break  # members are ordered nearest-heading first: good enough
        if best is None and most is not None:
            # No member fits a pure rotation: the robot has moved off the spot (driving away
            # from it, say). Allow for that movement, on the member with the most matches.
            _, points_now, points_member, member_to_align = most
            moved = relative_pose(points_now, points_member)
            if moved is not None and moved[2] >= MIN_EXPLAINED:
                best = ((moved[0] + member_to_align + 180.0) % 360.0 - 180.0, moved[2])
        return best

    def escape(
        self, back: Action = Action.BACKWARD, turn_ticks: int = ESCAPE_TICKS, why: str = ""
    ) -> None:
        """Queue a way out: back off, then turn. Each escape turns the other way from the
        last one, so repeated escapes do not just undo each other."""
        self.queue = [(back, BACK_OFF_TICKS), (self.search_turn, turn_ticks)]
        self.search_turn = OPPOSITE[self.search_turn]
        self.searching, self.turn_ticks = False, 0
        self.escape_reason = why

    def _blocked(self, image: np.ndarray) -> bool:
        """Did the last straight drive hit a wall? A 12 cm drive changes the mean pixel by
        16-73 (measured); pushing against a wall changes it by 0-1, or by 5-23 when the
        collision also twists the robot a degree or two."""
        if not self.check_wall or self.last_image is None:
            return False
        change = np.mean(np.abs(image.astype(np.int16) - self.last_image.astype(np.int16)))
        return change < STUCK_DIFF

    def _steer(self, image: np.ndarray, leg: Leg) -> tuple[Action, int]:
        drive = Action.BACKWARD if leg.reverse else Action.FORWARD
        measured = self.heading_error(image, leg)

        if measured is not None:
            self.searching = False
            angle, explained = measured
            turn = Action.LEFT if angle > 0 else Action.RIGHT
            cap = MAX_TURN_TICKS if explained >= GOOD_EXPLAINED else WEAK_TURN_TICKS
            ticks = min(cap, max(1, round(abs(angle) / TICK_DEG)))
            seen = f"{explained} matches explained, off by {angle:+.0f} deg"
            if abs(angle) > ALIGNED_DEG:
                self.status = f"turn to line up ({seen})"
                return (turn, ticks)
            if not leg.drive:
                # At the goal's spot and lined up: look around a little so the last metre can
                # see a goal view and take over.
                self.status = f"at the goal spot, waiting for the last metre ({seen})"
                return (self.search_turn, SEARCH_TICKS)
            if abs(angle) > ARC_DEG:
                self.status = f"arc ({seen})"
                return (drive | turn, ticks)  # a little off: correct while driving
            burst = LONG_TICKS if leg.straight else FORWARD_TICKS
            self.status = f"drive {burst} ({seen})"
            return (drive, burst)

        # Nothing from this spot recognized. The recorded move at the photo picks the search
        # direction when a search starts (mirrored when going backwards); during a search
        # keep turning the same way so the robot sweeps the whole circle.
        if not self.searching:
            if "LEFT" in leg.align.actions:
                self.search_turn = Action.RIGHT if leg.reverse else Action.LEFT
            elif "RIGHT" in leg.align.actions:
                self.search_turn = Action.LEFT if leg.reverse else Action.RIGHT
            self.searching = True
        self.status = "search (nothing from this spot recognized)"
        return (self.search_turn, SEARCH_TICKS)
