import numpy as np
from scipy.spatial.distance import cdist


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i: int, j: int) -> None:
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j


def min_contour_distance(cnt1: np.ndarray, cnt2: np.ndarray) -> float:
    pts1 = cnt1.reshape(-1, 2)
    pts2 = cnt2.reshape(-1, 2)
    dists = cdist(pts1, pts2, metric="euclidean")
    return float(np.min(dists))


def get_horizontal_extremes(cnt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = cnt.reshape(-1, 2)
    left_pt = pts[np.argmin(pts[:, 0])]
    right_pt = pts[np.argmax(pts[:, 0])]
    return left_pt, right_pt
