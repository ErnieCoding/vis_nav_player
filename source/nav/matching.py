"""Matching keypoints between two photos: "do they really show the same spot, and how far do I
turn to line them up?"

A *keypoint* is a small, distinctive spot in an image (a corner, a blob of texture) that can be
found again from a slightly different viewpoint. SIFT finds a few hundred per photo and gives
each one a *descriptor*: 128 numbers describing the patch around it. *Matching* pairs up
keypoints of photo A with keypoints of photo B whose descriptors are nearly the same.

Fingerprints (place_recognition.py) only say "these look alike overall", and the maze's
repeated wall textures fool them. Matches can be checked with geometry: if both photos really
show the same scene from two camera positions, most matches agree with ONE camera movement.
RANSAC finds that movement by trial and error and counts the matches that agree with it, the
*inliers*. Two look-alike walls in different places give few inliers.

This uses OpenCV's SIFT, which is installed. SuperPoint + LightGlue are the stronger upgrade.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from nav.data import Photo, read_image

MIN_INLIERS = 25  # tune with source/testbed/check_map.py: enough to pass true pairs only


@dataclass
class Features:
    points: np.ndarray  # (n, 2): each keypoint's (x, y) position in pixels
    descriptors: np.ndarray | None  # (n, 128): each keypoint's descriptor; None if no keypoints


class Matcher:
    def __init__(self, ratio: float = 0.75, max_features: int = 500) -> None:
        self.sift = cv2.SIFT_create(nfeatures=max_features)
        self.bf = cv2.BFMatcher(cv2.NORM_L2)
        self.ratio = ratio
        self._cache: dict[str, Features] = {}

    def features(self, image: np.ndarray) -> Features:
        """SIFT keypoints and descriptors of one BGR image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        points = np.array([kp.pt for kp in keypoints], np.float32).reshape(-1, 2)
        return Features(points, descriptors)

    def photo_features(self, photo: Photo) -> Features:
        """Same as ``features``, for a recorded photo, remembered after the first time."""
        if photo.key not in self._cache:
            self._cache[photo.key] = self.features(read_image(photo))
        return self._cache[photo.key]

    def match(self, a: Features, b: Features) -> tuple[np.ndarray, np.ndarray]:
        """Pairs of keypoints that look the same: ``(points_in_a, points_in_b)``, both (m, 2);
        row k of one goes with row k of the other. Lowe's ratio test keeps a pair only when
        its best partner is clearly better than the second best."""
        empty = np.empty((0, 2), np.float32)
        if a.descriptors is None or b.descriptors is None:
            return empty, empty
        if len(a.descriptors) < 2 or len(b.descriptors) < 2:
            return empty, empty
        pairs = self.bf.knnMatch(a.descriptors, b.descriptors, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < self.ratio * p[1].distance]
        if not good:
            return empty, empty
        points_a = a.points[[m.queryIdx for m in good]]
        points_b = b.points[[m.trainIdx for m in good]]
        return points_a, points_b

    def same_place(self, a: Features, b: Features, min_inliers: int = MIN_INLIERS) -> bool:
        """Do the two photos show the same spot? Enough matches that agree geometrically."""
        return self.count_inliers(*self.match(a, b)) >= min_inliers

    def count_inliers(self, points_a: np.ndarray, points_b: np.ndarray) -> int:
        """How many matched pairs agree with one camera movement (RANSAC). 0 if too few pairs."""
        if len(points_a) < 8:
            return 0
        
        matches = 0
        F, mask = cv2.findFundamentalMat(points_a, points_b, cv2.FM_RANSAC, 1.0, 0.99)

        if mask is None:
            return 0
        
        for match in mask:
            if match == 1:
                matches+=1

        return matches

    def bearing_deg(self, points_now: np.ndarray, points_target: np.ndarray, fx: float, cx:int = 160) -> float:
        """How many degrees to turn so that the target photo's view lines up with ours.
        Positive = turn LEFT, negative = turn RIGHT (the convention last_metre.py uses).
        ``fx`` is the camera's focal length in pixels (about 92 for this camera)."""
        if len(points_now) == 0:
            return 0.0
        
        angle_diff = []

        for i in range(len(points_now)):
            x_now = points_now[i][0]
            x_target = points_target[i][0]

            angle_now = np.atan((x_now - cx) / fx)
            angle_target = np.atan((x_target - cx) / fx)

            angle_diff.append(angle_target - angle_now)

        return float(np.degrees(np.median(angle_diff)))