"""The localizer: "where am I?"

The baseline trusts the single best-matching photo, so one look-alike wall makes it jump to
the other end of the maze. Instead, keep a *belief*: for every node of the map, the
probability that we are at that photo's spot, facing its way. The numbers add up to 1.
Every step does two things:

  predict  We just moved, so we may now be at a neighbouring node. Spread some belief from
           each node to its neighbours in the graph. More ticks moved = more spreading.
  update   We see a new camera frame. Nodes whose photos look like it become more likely,
           the others less likely: multiply by a *likelihood*, then rescale to add up to 1.

This is a discrete Bayes filter (also called an HMM filter: hidden Markov model). A look-alike
far away cannot win in one step, because the belief there was never built up by a sequence of
consistent views.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

STAY = 0.35  # predict: fraction of a node's belief that stays put per photo-length of motion
TICKS_PER_PHOTO = 5  # recorded photos are about this many ticks apart
TEMPERATURE = 0.02  # update: smaller = trust each single frame more.
# VLAD similarities are small (best matches ~0.05-0.15);
# DINOv2's are larger, so retune it after switching.


class Localizer:
    def __init__(self, graph: nx.Graph, fingerprints: np.ndarray) -> None:
        self.fingerprints = fingerprints
        self.n = len(fingerprints)
        # neighbours[i]: the nodes joined to node i by an edge. Handy for predict().
        self.neighbours = [list(graph.neighbors(i)) for i in range(self.n)]
        self.belief = np.full(self.n, 1.0 / self.n)

    def reset(self, start_nodes: list[int] | None = None) -> None:
        """Start of a run. With ``start_nodes``, all belief on those nodes (shared equally);
        without, the same everywhere: "no idea"."""
        self.belief = np.zeros(self.n)
        if start_nodes:
            self.belief[start_nodes] = 1.0 / len(start_nodes)
        else:
            self.belief[:] = 1.0 / self.n

    def predict(self, ticks: int) -> None:
        """We moved for ``ticks`` ticks since the last frame: spread the belief."""
        # TODO: motion update.
        #   Version 3 (later): spread more in the direction the robot drove (FORWARD favours
        #   the next photo of a route that was driving forward there).
        if ticks <= 0:
            return

        num_rounds = max(1, round(ticks / TICKS_PER_PHOTO))

        for _ in range(num_rounds):  # over rounds
            new_belief = np.zeros_like(self.belief)

            for j in range(len(new_belief)):  # over photos
                num_neighbors = len(self.neighbours[j])
                if num_neighbors > 0:  # if photo has neighbors, otherwise belief stays as is
                    new_belief[j] += STAY * self.belief[j]
                    shared_belief = (1 - STAY) * self.belief[j] / num_neighbors
                    for k in self.neighbours[j]:
                        new_belief[k] += shared_belief
                else:
                    new_belief[j] += self.belief[j]

            self.belief = new_belief

    def update(self, fingerprint: np.ndarray) -> None:
        """We see a new frame (here, its fingerprint): reweight the belief."""
        # TODO: measurement update.
        #   1. similarity = self.fingerprints @ fingerprint   (one number per node)
        #   2. likelihood = np.exp((similarity - similarity.max()) / TEMPERATURE)
        #      (subtracting the max keeps exp() from overflowing; it does not change the result)
        #   3. belief = belief * likelihood, plus a tiny floor (say 1e-12) so no node ever
        #      reaches exactly 0 and can never recover, then divide by the sum.
        similarity = self.fingerprints @ fingerprint
        likelihood = np.exp((similarity - similarity.max()) / TEMPERATURE)

        self.belief = self.belief * likelihood + 1e-12
        self.belief = self.belief / sum(self.belief)

    def estimate(self) -> tuple[int, float]:
        """``(best node, confidence)``: where we most likely are, and how sure we are (0..1)."""
        # TODO: read the belief out.
        #   best node: the node with the highest belief (np.argmax).
        #   confidence: how much belief sits on or next to it, e.g. belief[best] plus the
        #   belief of its neighbours. Near 1 = sure. Small = the belief is spread over several
        #   places: we are lost, so do not hand over to the last metre.
        best_node = int(np.argmax(self.belief))  # photo with the highest belief

        confidence = self.belief[best_node] + self.belief[self.neighbours[best_node]].sum()

        return best_node, confidence
