"""Reading the exploration data folder.

A challenge hands out a folder like this (``--local`` and the testbed record the same format):

    data/<id>/
      target.jpg            the four goal photos side by side: front, left, back, right
      traj_0/               route 0: one recorded drive through the maze
        0.jpg 1.jpg ...     photos, 320x240, one every 5 ticks
        data_info.json      [{"step": 0, "image": "0.jpg", "action": ["LEFT"]}, ...]
        poses.json          ONLY in testbed recordings: where each photo was really taken
      traj_1/ ...

``load_answer_key`` reads poses.json. Only the testbed calls it; the agent never does, because
a real challenge folder has no poses.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

VIEW_NAMES = ("front", "left", "back", "right")


@dataclass(frozen=True)
class Photo:
    """One recorded photo. ``index`` is its position in the list ``load_photos`` returns, and
    that is also its node number in the map."""

    index: int  # node number
    route: int  # which recorded drive: the i of traj_<i>/
    number: int  # which record of that drive: the row in data_info.json (and poses.json)
    path: Path
    actions: tuple[str, ...]  # the move(s) recorded right after this photo; a rough hint only

    @property
    def key(self) -> str:
        """``"traj_0/12.jpg"``: names the photo however the list was built."""
        return photo_key(self.path)


def photo_key(path: Path) -> str:
    return f"{Path(path).parent.name}/{Path(path).name}"


def route_dirs(data_dir: Path) -> list[Path]:
    """``traj_0/``, ``traj_1/``, ... in order."""
    dirs = sorted(
        (d for d in Path(data_dir).iterdir() if d.is_dir() and d.name.startswith("traj_")),
        key=lambda d: int(d.name.split("_")[1]),
    )
    if not dirs:
        raise FileNotFoundError(f"no traj_*/ folders under {data_dir}")
    return dirs


def load_photos(data_dir: Path, every: int = 1) -> list[Photo]:
    """Every photo of every route, in recording order, keeping one in ``every``.

    Unlike ``vlad.load_frames``, nothing else is skipped: photos followed by IDLE or by a
    combined move are places too."""
    photos: list[Photo] = []
    for directory in route_dirs(data_dir):
        route = int(directory.name.split("_")[1])
        records = json.loads((directory / "data_info.json").read_text())
        for number in range(0, len(records), every):
            record = records[number]
            photos.append(
                Photo(
                    index=len(photos),
                    route=route,
                    number=number,
                    path=directory / record["image"],
                    actions=tuple(record["action"]),
                )
            )
    return photos


def read_image(photo: Photo) -> np.ndarray:
    """The photo as a (240, 320, 3) BGR array, the same format as ``obs.image``."""
    image = cv2.imread(str(photo.path))
    if image is None:
        raise FileNotFoundError(photo.path)
    return image


def split_goal_photo(data_dir: Path) -> list[np.ndarray]:
    """``target.jpg`` cut into the four goal views ``[front, left, back, right]``. All four were
    taken from the goal spot. View 0 (front) is the way to face when checking in."""
    image = cv2.imread(str(Path(data_dir) / "target.jpg"))
    if image is None:
        raise FileNotFoundError(Path(data_dir) / "target.jpg")
    width = image.shape[1] // 4
    return [image[:, i * width : (i + 1) * width].copy() for i in range(4)]


def load_answer_key(data_dir: Path) -> dict[str, tuple[float, float, float]]:
    """TESTBED ONLY. Where each photo was really taken: ``{"traj_0/12.jpg": (x, y, yaw)}``,
    x and y in metres, yaw (the direction the camera faced) in radians."""
    key: dict[str, tuple[float, float, float]] = {}
    for directory in route_dirs(data_dir):
        poses_file = directory / "poses.json"
        if not poses_file.exists():
            raise FileNotFoundError(
                f"{poses_file} is missing: this folder has no answer key. "
                "Record one with source/testbed/record_eval.py."
            )
        records = json.loads((directory / "data_info.json").read_text())
        poses = json.loads(poses_file.read_text())
        for record, pose in zip(records, poses, strict=True):
            key[f"{directory.name}/{record['image']}"] = (
                float(pose[0]),
                float(pose[1]),
                float(pose[2]),
            )
    return key
