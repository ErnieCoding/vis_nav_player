"""
My agent: the baseline, driving itself.

    uv run source/my_agent.py --local 7             # on your machine, maze 7, free
    uv run source/my_agent.py                       # on the server, one attempt
"""

from __future__ import annotations

from pathlib import Path

import cli
from baseline_agent import BaselineAgent
from vis_nav_sdk import Action, Observation, run


class MyAgent(BaselineAgent):
    def act(self, obs: Observation) -> Action | tuple[Action, int]:
        # 1. Where am I? Match this frame to a node and replan the path to the goal.
        if obs.step != self._localised_step:
            self._localise(obs.image)
            self._localised_step = obs.step

        # 2. Path has one node left: the baseline thinks we're at the goal.
        if len(self.path) < 2:
            return Action.CHECKIN

        # 3. Otherwise, do the move recorded on the path's first edge.
        move = self._edge_action(self.path[0], self.path[1])
        if move == "?":
            # First edge is a visual shortcut: there's no recorded move for it.
            return Action.FORWARD, 5
        return Action[move], 5


if __name__ == "__main__":
    parser = cli.parser(__doc__)
    parser.add_argument("--data", default=None, help="exploration data directory")
    args = parser.parse_args()

    challenge = cli.challenge(args)
    agent = MyAgent(
        cli.exploration_data(args, args.data),
        cache_dir=Path("cache") / challenge.replace(":", "-"),
    )
    run(agent, challenge, **cli.run_options(args))