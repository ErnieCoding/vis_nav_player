"""Two-view geometry for a camera on a flat floor.

The robot only turns about the vertical axis and moves along the floor, so two camera views
differ by just two numbers: the turn between them and the direction from one camera to the
other. Matched keypoints (nav/matching.py) pin both down:

  rotation_fit    both views taken from the same spot: the turn, to a fraction of a degree,
                  and how many matches a pure rotation explains (most of them = same spot)
  relative_pose   the views taken a few tens of centimetres apart: the turn and the direction
                  to the other camera, from the epipolar constraint, by grid search

Pixels -> rays uses the course camera: focal length 92 px, centre (160, 120), the same in
every maze. All angles are in degrees, + = left (counter-clockwise from above).
"""

from __future__ import annotations

import math

import numpy as np

FX, CX, CY = 92.0, 160.0, 120.0
MIN_MATCHES = 8  # fewer matched pairs than this: no estimate


def _rays(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (points[:, 0] - CX) / FX, (points[:, 1] - CY) / FX


def rotation_fit(
    points_now: np.ndarray,
    points_ref: np.ndarray,
    fit_px: float = 2.0,
    horizontal_only: bool = False,
):
    """Assuming both views were taken from the same spot: the turn (+ = left) that faces the
    way the reference view faces, and how many matches that rotation explains.
    Measured on recorded photos: median error 0.13 deg, 99th percentile 0.6 deg.

    ``horizontal_only``: ignore the vertical image coordinate. Needed for the goal views,
    which were taken from a camera higher than the robot's: that moves points up or down
    by an amount that depends on their distance, but never sideways, and a turn about the
    vertical axis only moves them sideways."""
    if len(points_now) < 4:
        return 0.0, 0
    a1, b1 = _rays(points_now)
    a2, b2 = _rays(points_ref)
    threshold = (fit_px / FX) ** 2

    def explained(thetas: np.ndarray) -> np.ndarray:
        c, s = np.cos(thetas)[:, None], np.sin(thetas)[:, None]
        den = c - a1 * s  # the point's depth factor after rotating: must stay in front
        safe = np.where(np.abs(den) < 1e-6, 1e-6, den)
        error = ((a1 * c + s) / safe - a2) ** 2
        if not horizontal_only:
            error = error + (b1 / safe - b2) ** 2
        return ((den > 0) & (error < threshold)).sum(1)

    coarse = np.radians(np.arange(-180.0, 180.0, 1.0))
    best = coarse[int(np.argmax(explained(coarse)))]
    fine = best + np.radians(np.arange(-1.0, 1.01, 0.1))
    counts = explained(fine)
    return math.degrees(fine[int(np.argmax(counts))]), int(counts.max())


def relative_pose(points_now: np.ndarray, points_ref: np.ndarray):
    """The reference camera relative to this one, allowing for both a turn and a move:
    ``(turn, direction, agree)`` -- the turn (+ = left) that faces the reference view's way,
    the direction (+ = left, 0 = straight ahead) toward the reference camera's position, and
    how many matches agree. None with fewer than MIN_MATCHES matches.

    Measured on the follower's real situations: turn error median 3-4 deg. The direction is
    rougher (median 15-40 deg at 5-40 cm apart) and meaningless when both were taken from
    the same spot: check that with rotation_fit first."""
    if len(points_now) < MIN_MATCHES:
        return None
    a1, b1 = _rays(points_now)
    a2, b2 = _rays(points_ref)
    threshold = (1.5 / FX) ** 2

    def agree(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        # Planar motion: R = rotation by theta about the vertical, t = (sin phi, 0, cos phi).
        # x_ref . (t x R x_now) = 0 reduces to the residual below; divide by its gradient
        # (Sampson distance) so the threshold is in pixels.
        psi = phi - theta
        cphi, sphi = np.cos(phi)[:, None], np.sin(phi)[:, None]
        cpsi, spsi = np.cos(psi)[:, None], np.sin(psi)[:, None]
        residual = b1 * (sphi - a2 * cphi) + b2 * (a1 * cpsi - spsi)
        scale = (b1 * cphi) ** 2 + (a1 * cpsi - spsi) ** 2 + (b2 * cpsi) ** 2
        scale = scale + (sphi - a2 * cphi) ** 2
        return residual**2 / np.maximum(scale, 1e-12) < threshold

    grid = np.radians(np.arange(-180.0, 180.0, 3.0))
    theta, phi = (g.ravel() for g in np.meshgrid(grid, grid, indexing="ij"))
    best = int(np.argmax(agree(theta, phi).sum(1)))
    step = np.radians(np.arange(-3.0, 3.01, 0.5))
    theta, phi = (g.ravel() for g in np.meshgrid(theta[best] + step, phi[best] + step))
    masks = agree(theta, phi)
    best = int(np.argmax(masks.sum(1)))
    theta, phi, inliers = float(theta[best]), float(phi[best]), masks[best]

    # t and -t satisfy the same constraint: keep the one that puts the points in front of
    # both cameras.
    ct, st = math.cos(theta), math.sin(theta)
    den = a1 * ct + st - a2 * (ct - a1 * st)
    den = np.where(np.abs(den) < 1e-9, 1e-9, den)

    def in_front(phi_: float) -> int:
        depth_now = (a2 * math.cos(phi_) - math.sin(phi_)) / den
        depth_ref = depth_now * (ct - a1 * st) + math.cos(phi_)
        return int(np.sum((depth_now[inliers] > 0) & (depth_ref[inliers] > 0)))

    if in_front(phi + math.pi) > in_front(phi):
        phi += math.pi
    # the reference camera's centre in this camera's frame: -R^T t (x right, z forward)
    centre_x = -(ct * math.sin(phi) - st * math.cos(phi))
    centre_z = -(st * math.sin(phi) + ct * math.cos(phi))
    turn = (math.degrees(theta) + 180.0) % 360.0 - 180.0
    direction = math.degrees(math.atan2(-centre_x, centre_z))
    return turn, direction, int(inliers.sum())
