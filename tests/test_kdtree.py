import os
import sys
import math
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kdtree import KDTree


def brute_force_knn(points, data_list, target, k):
    dists = [(math.dist(target, p), p, d) for p, d in zip(points, data_list)]
    dists.sort(key=lambda x: x[0])
    return dists[:k]


def brute_force_radius(points, data_list, target, r):
    dists = [(math.dist(target, p), p, d) for p, d in zip(points, data_list)]
    return sorted([x for x in dists if x[0] <= r], key=lambda x: x[0])


def random_dataset(n=500, seed=0):
    rng = random.Random(seed)
    points = [(rng.uniform(-10, 10), rng.uniform(-10, 10)) for _ in range(n)]
    data = [{"id": i} for i in range(n)]
    return points, data


def test_knn_matches_brute_force():
    points, data = random_dataset(500, seed=1)
    tree = KDTree(points, data)
    rng = random.Random(2)
    for _ in range(20):
        target = (rng.uniform(-10, 10), rng.uniform(-10, 10))
        got = tree.k_nearest(target, k=5)
        expected = brute_force_knn(points, data, target, 5)
        got_ids = sorted(d["id"] for _, _, d in got)
        expected_ids = sorted(d["id"] for _, _, d in expected)
        assert got_ids == expected_ids, f"mismatch for target {target}"


def test_radius_matches_brute_force():
    points, data = random_dataset(500, seed=3)
    tree = KDTree(points, data)
    rng = random.Random(4)
    for _ in range(20):
        target = (rng.uniform(-10, 10), rng.uniform(-10, 10))
        r = rng.uniform(0.5, 4)
        got = tree.radius_query(target, r)
        expected = brute_force_radius(points, data, target, r)
        got_ids = sorted(d["id"] for _, _, d in got)
        expected_ids = sorted(d["id"] for _, _, d in expected)
        assert got_ids == expected_ids, f"mismatch for target {target}, r={r}"


def test_empty_tree():
    tree = KDTree([], [])
    assert tree.k_nearest((0, 0), k=5) == []
    assert tree.radius_query((0, 0), 1.0) == []


def test_single_point():
    tree = KDTree([(1.0, 1.0)], [{"id": 0}])
    result = tree.k_nearest((0.0, 0.0), k=1)
    assert len(result) == 1
    assert result[0][2]["id"] == 0
