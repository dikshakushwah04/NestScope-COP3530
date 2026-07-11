import os
import sys
import math
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quadtree import QuadTree, BoundingBox, Point


def brute_force_radius(points, target, r):
    x, y = target
    dists = [(math.dist((x, y), (p.x, p.y)), (p.x, p.y), p.data) for p in points]
    return sorted([d for d in dists if d[0] <= r], key=lambda a: a[0])


def random_points(n=500, seed=0):
    rng = random.Random(seed)
    return [Point(rng.uniform(-10, 10), rng.uniform(-10, 10), {"id": i}) for i in range(n)]


def build_tree(points):
    qt = QuadTree(BoundingBox(0, 0, 11, 11))
    for p in points:
        qt.insert(p)
    return qt


def test_radius_matches_brute_force():
    points = random_points(500, seed=1)
    qt = build_tree(points)
    rng = random.Random(2)
    for _ in range(20):
        target = (rng.uniform(-10, 10), rng.uniform(-10, 10))
        r = rng.uniform(0.5, 4)
        got = qt.radius_query(target[0], target[1], r)
        expected = brute_force_radius(points, target, r)
        got_ids = sorted(d["id"] for _, _, d in got)
        expected_ids = sorted(d["id"] for _, _, d in expected)
        assert got_ids == expected_ids, f"mismatch for target {target}, r={r}"


def test_all_points_retained():
    points = random_points(500, seed=3)
    qt = build_tree(points)
    assert qt.count() == 500


def test_knn_returns_k_closest_when_available():
    points = random_points(1000, seed=4)
    qt = build_tree(points)
    got = qt.k_nearest(0, 0, k=10)
    expected = sorted(
        [(math.dist((0, 0), (p.x, p.y)), (p.x, p.y), p.data) for p in points],
        key=lambda a: a[0],
    )[:10]
    got_ids = sorted(d["id"] for _, _, d in got)
    expected_ids = sorted(d["id"] for _, _, d in expected)
    assert got_ids == expected_ids


def test_empty_tree():
    qt = QuadTree(BoundingBox(0, 0, 10, 10))
    assert qt.radius_query(0, 0, 1.0) == []
    assert qt.count() == 0
