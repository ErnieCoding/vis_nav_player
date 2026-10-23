"""
Drive near the goal, press L to let the last-metre module take over.
Any arrow key takes control back. Space checks in, Esc quits.

    uv run source/test_last_metre.py --local 7
"""

from __future__ import annotations

import cli
from keyboard_agent import KeyboardAgent
from last_metre import LastMetre
from vis_nav_sdk import WAIT, Action, Observation, SessionInfo, run

ARROWS = ("up", "down", "left", "right")
AUTO_CHECKIN = False  # True = the module checks in by itself, as in the final agent


class TestAgent(KeyboardAgent):
    def __init__(self) -> None:
        self.lm = LastMetre()
        self.auto = False
        self._last_step = -1

    def setup(self, info: SessionInfo) -> None:
        self.lm.start(info)
        self.auto = False

    def act(self, obs: Observation):
        keys = self.viewer.keys() if self.viewer is not None else ()
        if "l" in keys and not self.auto:
            self.lm.reset()  # fresh start every handover
            self.auto = True
        if any(k in keys for k in ARROWS):
            self.auto = False

        if not self.auto:
            if obs.step != self._last_step:  # keep the display updated while you drive
                self.lm.look(obs.image)
                self.lm.status = f"manual: best view {self.lm.view}"
                self._last_step = obs.step
            return super().act(obs)

        if "escape" in keys:
            return Action.QUIT
        if "space" in keys:
            return Action.CHECKIN
        move = self.lm.step(obs.image)
        if move == "LOST":  # the module gives control back (navigation would take over)
            self.auto = False
            return WAIT
        if move == Action.CHECKIN and not AUTO_CHECKIN:
            self.lm.status = "READY: the module would check in now (press space)"
            return WAIT
        return WAIT if move is None else move

    def hud(self) -> list[str]:
        f, l, b, r = self.lm.counts
        yaw = "-" if self.lm.yaw is None else f"{self.lm.yaw:+.1f} deg"
        mode = f"AUTO, {self.lm.mode} mode" if self.auto else "manual"
        return super().hud() + [
            f"mode: {mode}   press L to hand over",
            f"{self.lm.status}",
            f"matches  front {f}  left {l}  back {b}  right {r}   fine yaw {yaw}",
        ]


if __name__ == "__main__":
    args = cli.parser(__doc__).parse_args()
    run(TestAgent(), cli.challenge(args), **cli.run_options(args))