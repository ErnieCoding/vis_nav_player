"""Localization and navigation: the parts of our agent that find the way to the goal.

Built BEFORE the attempt (by build_map.py):

    data.py               read the exploration data folder: photos, moves, the goal photo
    place_recognition.py  turn a photo into a fingerprint; "which photo looks like this?"
    matching.py           "do these two photos really show the same spot?" and turn angles
    place_map.py          the map: a graph of the recorded photos, plus where the goal is

Used DURING the attempt (by nav_agent.py, every step):

    localizer.py          "where am I?": a probability for every photo in the map
    planner.py            "which way to the goal?": the shortest chain of photos
    follower.py           "what move now?": steer toward the next photo on that chain
"""
