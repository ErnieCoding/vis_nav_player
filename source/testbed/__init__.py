"""The testbed: practice mazes on this machine, graded with the simulator's answer key.

On the server nobody tells you where the robot is, so a failed run gives you nothing to debug
with. The simulator on your own machine knows everything. These scripts use that knowledge to
grade each part of the agent separately. The agent itself never imports this package.

    record_eval.py              record practice mazes WITH an answer key (poses.json)
    check_place_recognition.py  grade fingerprints: is the most similar photo really nearby?
    check_map.py                grade a built map: wrong shortcuts? is the goal photo right?
    show_map.py                 draw a built map over the real maze, and the photos along a path
    run_nav.py                  run whole attempts, server-like, and score them
"""
