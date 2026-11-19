"""Dense vector indexes: exact brute force and an IVF (inverted file) index.

Vectors are assumed L2-normalized, so inner product == cosine similarity.
The IVF index partitions vectors with spherical k-means and only scans the
`nprobe` closest partitions at query time: the classic recall/latency knob.
"""
from __future__ import annotations

import numpy as np

from .bm25 import top_k_indices


def _normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


class FlatIndex:
    """Exact maximum-inner-product search via one matrix-vector product."""

    def __init__(self, vectors: np.ndarray) -> None:
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    def __len__(self) -> int:
        return self.vectors.shape[0]

    def search(self, qvec: np.ndarray, k: int = 10, mask: np.ndarray | None = None) -> list[tuple[int, float]]:
        if len(self) == 0:
            return []
        scores = self.vectors @ qvec.astype(np.float32)
        return [(int(i), float(scores[i])) for i in top_k_indices(scores, k, mask)]


def spherical_kmeans(
    x: np.ndarray, n_clusters: int, n_iter: int = 25, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Cosine k-means with deterministic k-means++ seeding. Returns (centroids, labels)."""
    n = x.shape[0]
    n_clusters = max(1, min(n_clusters, n))
    rng = np.random.default_rng(seed)
    centroids = np.empty((n_clusters, x.shape[1]), dtype=np.float32)
    centroids[0] = x[rng.integers(n)]
    closest = 1.0 - x @ centroids[0]
    for c in range(1, n_clusters):
        d = np.clip(closest, 0, None) ** 2
        total = d.sum()
        idx = rng.choice(n, p=d / total) if total > 0 else rng.integers(n)
        centroids[c] = x[idx]
        closest = np.minimum(closest, 1.0 - x @ centroids[c])
    labels = np.full(n, -1)
    for _ in range(n_iter):
        new_labels = np.argmax(x @ centroids.T, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(n_clusters):
            members = x[labels == c]
            if len(members):
                centroids[c] = members.sum(axis=0)
            else:  # re-seed an empty cluster with the worst-served point
                worst = int(np.argmin(np.max(x @ centroids.T, axis=1)))
                centroids[c] = x[worst]
        centroids = _normalize(centroids).astype(np.float32)
    return centroids, labels


class IVFIndex:
    """Inverted-file index: coarse quantizer + exhaustive scan of probed lists."""

    def __init__(self, vectors: np.ndarray, n_lists: int = 16, nprobe: int = 4, seed: int = 0) -> None:
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.nprobe = nprobe
        if len(self.vectors):
            self.centroids, labels = spherical_kmeans(self.vectors, n_lists, seed=seed)
        else:
            self.centroids, labels = np.zeros((0, vectors.shape[1]), np.float32), np.zeros(0, int)
        self.lists: list[np.ndarray] = [np.flatnonzero(labels == c) for c in range(len(self.centroids))]
        # contiguous per-list copies: probing scans memory sequentially, no gather
        self._list_vecs: list[np.ndarray] = [np.ascontiguousarray(self.vectors[ids]) for ids in self.lists]
        self.last_scanned = 0

    @property
    def n_lists(self) -> int:
        return len(self.lists)

    def __len__(self) -> int:
        return self.vectors.shape[0]

    def search(
        self,
        qvec: np.ndarray,
        k: int = 10,
        mask: np.ndarray | None = None,
        nprobe: int | None = None,
    ) -> list[tuple[int, float]]:
        if len(self) == 0:
            return []
        probe = max(1, min(nprobe or self.nprobe, self.n_lists))
        cscores = self.centroids @ qvec.astype(np.float32)
        probed = np.argsort(-cscores, kind="stable")[:probe]
        q = qvec.astype(np.float32)
        cand = np.concatenate([self.lists[c] for c in probed])
        scores = np.concatenate([self._list_vecs[c] @ q for c in probed])
        self.last_scanned = int(cand.size)
        if mask is not None:
            keep = mask[cand]
            cand, scores = cand[keep], scores[keep]
        if cand.size == 0:
            return []
        order = top_k_indices(scores, k)
        return [(int(cand[i]), float(scores[i])) for i in order]
