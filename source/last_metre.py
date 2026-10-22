"""Last-metre module: lines the robot up with the goal's front view, then checks in."""

from __future__ import annotations

import math

import cv2
import numpy as np
from vis_nav_sdk import Action

VIEW_NAMES = ("front", "left", "back", "right")
# If this goal view matches best (and we're on the spot), turn this much to face front.
TURN_FOR_VIEW = {"front": 0, "left": 90, "back": 180, "right": -90}

MIN_MATCHES = 10      # fewer than this: we can't see that goal view
MARGIN = 2.0          # best view must beat the runner-up by this factor to be trusted
ALIGNED_DEG = 3.0     # facing front when within this many degrees
SCALE_TOL = 0.05      # at the right distance when keypoint sizes match within 5%
STEADY = 2            # checks in a row that must agree before checking in
STUCK_STEPS = 15      # this many coarse-turning decisions in a row = stuck: scan instead
SCAN_TICKS = 2        # ticks turned per scan increment
SCAN_MAX = 200        # safety cap on scan increments
FOUND_FRACTION = 0.8  # after the scan, stop turning when front reaches 80% of its best


def _focal_length(camera) -> float | None:
    """Find the 3x3 intrinsic matrix K in the camera info and return its focal length."""
    for name in dir(camera):
        if name.startswith("_"):
            continue
        value = getattr(camera, name)
        if isinstance(value, np.ndarray) and value.shape == (3, 3):
            return float(value[0, 0])
    return None


class LastMetre:
    def __init__(self, ratio: float = 0.75) -> None:
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.ratio = ratio
        self.targets = []
        self.fx = 277.0  # guess (60 degree view); replaced by the real value in start()
        self.reset()

    # ---------- setup ----------

    def reset(self) -> None:
        """Forget everything about the current attempt (call when taking over)."""
        self.mode = "normal"  # normal | scan | front
        self.coarse_steps, self.steady, self.lost = 0, 0, 0
        self.last_yaw, self.last_turn = None, Action.LEFT
        self.view, self.counts, self.yaw, self.scale = "?", [0, 0, 0, 0], None, None
        self.status = "idle"

    def start(self, info) -> None:
        """Once per run: fingerprint the 4 goal photos and read the camera's focal length."""
        self.targets = [self._describe(image) for image in info.targets]
        fx = _focal_length(info.camera)
        if fx:
            self.fx = fx
        self.reset()

    # ---------- seeing ----------

    def _describe(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.sift.detectAndCompute(gray, None)

    def _match(self, des_a, des_b) -> list:
        if des_a is None or des_b is None or len(des_a) < 2 or len(des_b) < 2:
            return []
        pairs = self.matcher.knnMatch(des_a, des_b, k=2)
        return [p[0] for p in pairs if len(p) == 2 and p[0].distance < self.ratio * p[1].distance]

    def look(self, image):
        """Match the camera against the 4 goal photos (no decisions, no state changes)."""
        kp, des = self._describe(image)
        matches = [self._match(des, t_des) for _, t_des in self.targets]
        self.counts = [len(m) for m in matches]
        self.view = VIEW_NAMES[int(np.argmax(self.counts))]
        return kp, des, matches

    def _fine_heading(self, kp, front_matches) -> float:
        """Degrees to turn to line up with the front photo: + = turn left, - = turn right."""
        front_kp = self.targets[0][0]
        shifts = [front_kp[m.trainIdx].pt[0] - kp[m.queryIdx].pt[0] for m in front_matches]
        return math.degrees(math.atan(float(np.median(shifts)) / self.fx))

    def _scale_ratio(self, kp, front_matches) -> float:
        """Below 1 = things look smaller than in the front photo = too far. Above 1 = too close."""
        front_kp = self.targets[0][0]
        ratios = [kp[m.queryIdx].size / front_kp[m.trainIdx].size for m in front_matches]
        return float(np.median(ratios))

    def _go(self, move, status):
        """Any move resets the 'steady' count: we only check in after holding still."""
        self.steady, self.status = 0, status
        return move

    # ---------- deciding ----------

    def step(self, image):
        """Look, then decide one move. Returns a move, Action.CHECKIN, or None (hold still)."""
        kp, des, matches = self.look(image)

        if self.mode == "scan":
            return self._scan_step(kp, des, matches)

        if self.mode == "front":  # after a scan: trust only the front photo
            if self.counts[0] < MIN_MATCHES:
                self.lost += 1
                if self.lost > 5:
                    return self._begin_scan(des, "lost the front view")
                return self._go(None, "front view weak: holding")
            self.lost = 0
            return self._fine_and_approach(kp, matches[0])

        # normal mode: coarse heading from the 4 goal photos (works on or near the spot)
        move = self._coarse()
        if move == "scan":
            return self._begin_scan(des, "can't see the goal")
        if move is not None:
            self.coarse_steps += 1
            if self.coarse_steps > STUCK_STEPS:
                return self._begin_scan(des, "stuck turning")
            return move
        self.coarse_steps = 0
        return self._fine_and_approach(kp, matches[0])

    def _coarse(self):
        """Turn toward front using which goal photo matches best. None = facing front."""
        order = np.argsort(self.counts)[::-1]
        best, second = self.counts[order[0]], self.counts[order[1]]
        if best < MIN_MATCHES:
            return "scan"
        if best < MARGIN * max(second, 1):
            pair = (VIEW_NAMES[order[0]], VIEW_NAMES[order[1]])
            if "front" in pair and ("left" in pair or "right" in pair):
                self.last_turn = Action.RIGHT if "left" in pair else Action.LEFT
            return self._go((self.last_turn, 1), "unsure which view: keep turning")
        if self.view != "front":
            self.last_turn = Action.RIGHT if TURN_FOR_VIEW[self.view] > 0 else Action.LEFT
            return self._go((self.last_turn, 3), f"facing goal's {self.view}: turning toward front")
        return None

    # ---------- the scan ----------

    def _begin_scan(self, des, why):
        """Spin a full circle, remembering where the front photo matched best."""
        self.mode, self.scan_phase = "scan", "circle"
        self.scan_first = des  # the view we started from: seeing it again = full circle
        self.scan_first_n = 0 if des is None else len(des)
        self.scan_i, self.scan_best, self.scan_left_home = 0, self.counts[0], False
        return self._go((Action.RIGHT, SCAN_TICKS), f"{why}: starting a 360 scan")

    def _scan_step(self, kp, des, matches):
        self.scan_i += 1
        front = self.counts[0]

        if self.scan_phase == "circle":
            self.scan_best = max(self.scan_best, front)
            home = len(self._match(des, self.scan_first))
            if home < max(5, 0.1 * self.scan_first_n):
                self.scan_left_home = True  # turned far enough that the start view is gone
            back_home = self.scan_left_home and home >= max(20, 0.3 * self.scan_first_n)
            if back_home or self.scan_i >= SCAN_MAX:
                if self.scan_best < MIN_MATCHES:
                    self.mode, self.coarse_steps = "normal", 0
                    return self._go(None, "scan done: front view not visible from here")
                self.scan_phase = "return"
            return self._go((Action.RIGHT, SCAN_TICKS),
                            f"scanning 360: front {front} (best so far {self.scan_best})")

        # return phase: keep turning the same way until front is nearly as good as its best
        if front >= FOUND_FRACTION * self.scan_best or self.scan_i >= 2 * SCAN_MAX:
            self.mode, self.lost = "front", 0
            return self._fine_and_approach(kp, matches[0])
        return self._go((Action.RIGHT, SCAN_TICKS),
                        f"turning to best direction: front {front} of {self.scan_best}")

    # ---------- fine heading + approach + check-in ----------

    def _fine_and_approach(self, kp, front_matches):
        self.yaw = self._fine_heading(kp, front_matches)
        self.scale = self._scale_ratio(kp, front_matches)
        flipped = self.last_yaw is not None and self.yaw * self.last_yaw < 0  # overshot
        self.last_yaw = self.yaw

        side = "left" if self.yaw > 0 else "right"
        turn = Action.LEFT if self.yaw > 0 else Action.RIGHT
        need_turn = abs(self.yaw) > ALIGNED_DEG and not flipped
        if self.scale < 1 - SCALE_TOL:
            drive, how = Action.FORWARD, "too far"
        elif self.scale > 1 + SCALE_TOL:
            drive, how = Action.BACKWARD, "too close"
        else:
            drive, how = None, "right distance"

        if abs(self.yaw) > 10.0:
            return self._go((turn, 1), f"turning {abs(self.yaw):.1f} deg {side} first")
        if drive is not None and need_turn:
            return self._go((drive | turn, 1), f"{how} (size {self.scale:.2f}) + {side}: arc")
        if drive is not None:
            return self._go((drive, 1), f"{how} (size {self.scale:.2f}): moving")
        if need_turn:
            return self._go((turn, 1), f"right distance, turning {abs(self.yaw):.1f} deg {side}")

        self.steady += 1
        if self.steady >= STEADY:
            self.status = "AT THE GOAL: checking in"
            return Action.CHECKIN
        self.status = f"lined up, size {self.scale:.2f}: confirming ({self.steady}/{STEADY})"
        return None