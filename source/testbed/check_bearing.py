import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # source/, so `nav` imports

import numpy as np
from nav.data import load_answer_key, load_photos
from nav.matching import Matcher
from testbed.tools import make_world, server_like

data = Path("data/eval-21-3")
photos, answer = load_photos(data), load_answer_key(data)
world = make_world(21, 3, motion_seed=0)  # the same maze as data/eval-21-3
fx = float(np.asarray(world.intrinsic_matrix)[0, 0])
matcher = Matcher()
rng = random.Random(0)

for photo in rng.sample(photos, 10):
    x, y, yaw = answer[photo.key]  # a real spot in the maze
    theta = rng.uniform(-30, 30)  # a known turn, in degrees (+ = left)
    target = server_like(world.render(x, y, yaw), world)
    now = server_like(world.render(x, y, yaw + math.radians(theta)), world)
    pts_now, pts_target = matcher.match(matcher.features(now), matcher.features(target))
    got = matcher.bearing_deg(pts_now, pts_target, fx)
    print(f"{photo.key}: expected {-theta:+6.1f}  got {got:+6.1f}  ({len(pts_now)} matches)")
