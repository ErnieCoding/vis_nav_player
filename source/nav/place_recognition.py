"""Place recognition: "which recorded photo looks most like this one?"

Each photo is turned into a *fingerprint* (the textbook name is *global descriptor*): one long
list of numbers, scaled to length 1, built so that photos of the same place get similar lists.
How alike two photos are is then a single dot product: near 1 = very alike, near 0 = unrelated.
Stack the fingerprints of all recorded photos into a matrix, and one matrix-vector product
compares the camera frame with every photo at once.

Two recognizers share the same two methods, so the rest of the code does not care which one
is used:

    VladRecognizer   the course baseline's RootSIFT + VLAD (vlad.py). Works now.
    DinoRecognizer   DINOv2, a pretrained neural network. Usually much better. Needs torch.

Pick one with ``--recognizer vlad|dino`` on build_map.py, nav_agent.py and the testbed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from vlad import Frame, VLADExtractor

from nav.data import Photo


class VladRecognizer:
    """The baseline's fingerprint (RootSIFT keypoints -> visual words -> VLAD), behind our
    interface. The vocabulary and the photos' keypoints are cached under ``cache_dir``."""

    def __init__(self, cache_dir: Path, n_clusters: int = 128) -> None:
        self.extractor = VLADExtractor(n_clusters, Path(cache_dir))

    def fit(self, photos: list[Photo]) -> np.ndarray:
        """Fingerprints of all recorded photos, shape (N, D); row i belongs to photos[i]. The
        first call learns the vocabulary (a minute or two); later calls read the cache."""
        frames = [Frame(p.path, p.actions[0] if p.actions else "IDLE", p.route) for p in photos]
        return self.extractor.fit(frames)

    def describe(self, image: np.ndarray) -> np.ndarray:
        """The fingerprint of one image (a camera frame or a goal view), shape (D,)."""
        return self.extractor.extract(image)


class DinoRecognizer:
    """DINOv2 fingerprints. An upgrade for once the whole pipeline works with VLAD: same two
    methods, so ``--recognizer dino`` switches every script over. Needs torch,
    which the default install does not have (`uv sync --group rl-cuda` adds the CUDA build)."""

    def __init__(self, cache_dir: Path, model_name: str = "dinov2_vits14", device: str = "cuda"):
        self.cache_dir = Path(cache_dir)
        self.model_name = model_name
        self.device = device
        self.model = None

    def fit(self, photos: list[Photo]) -> np.ndarray:
        # TODO (upgrade): fingerprints of every recorded photo, shape (N, D).
        #   Read each photo with nav.data.read_image, run self.describe on it, and stack the
        #   results with np.stack. Save the array to self.cache_dir / f"{self.model_name}.npy"
        #   and load that file instead when it already exists (the network is slow-ish).
        raise NotImplementedError("DinoRecognizer.fit")

    def describe(self, image: np.ndarray) -> np.ndarray:
        # TODO (upgrade): one BGR image -> one fingerprint of length 1.
        #   1. First call only: load the network.
        #        import torch
        #        self.model = torch.hub.load("facebookresearch/dinov2", self.model_name)
        #        self.model = self.model.to(self.device).eval()
        #   2. Prepare the image the way DINOv2 was trained: BGR -> RGB, resize to 322 x 238
        #      (DINOv2 wants sides divisible by 14), scale to 0..1, subtract the ImageNet mean
        #      (0.485, 0.456, 0.406), divide by the std (0.229, 0.224, 0.225), make it a
        #      (1, 3, 238, 322) tensor on self.device.
        #   3. Under torch.no_grad(): out = self.model.forward_features(tensor).
        #      out["x_norm_clstoken"] is one vector summarising the whole image. Start with it.
        #   4. Back to numpy, divide by its length (np.linalg.norm), return it.
        raise NotImplementedError("DinoRecognizer.describe")


RECOGNIZERS = ("vlad", "dino")


def make_recognizer(name: str, cache_dir: Path):
    """The recognizer called ``name`` (one of RECOGNIZERS), caching under ``cache_dir/name``.
    Every script takes ``--recognizer`` and builds it here."""
    if name == "vlad":
        return VladRecognizer(Path(cache_dir) / "vlad")
    if name == "dino":
        return DinoRecognizer(Path(cache_dir) / "dino")
    raise ValueError(f"unknown recognizer {name!r}; choose from {RECOGNIZERS}")


def most_similar(
    fingerprints: np.ndarray, query: np.ndarray, k: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """The k rows of ``fingerprints`` most like ``query``: (row numbers, similarities), best
    first. Fingerprints have length 1, so the dot product is the similarity."""
    similarity = fingerprints @ query
    best = np.argsort(similarity)[::-1][:k]
    return best, similarity[best]
