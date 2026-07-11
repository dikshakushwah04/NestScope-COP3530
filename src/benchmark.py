"""
benchmark.py

Benchmarks KDTree vs QuadTree build time and query time as dataset size
scales, and saves results as a CSV + chart in results/.
"""

import os
import sys
import time
import random
import csv

sys.path.insert(0, os.path.dirname(__file__))

from kdtree import KDTree
from quadtree import QuadTree, BoundingBox, Point
from data_loader import generate_synthetic_data

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def build_quadtree(points, data_list):
    lats = [p[0] for p in points]
    lons = [p[1] for p in points]
    cx = (min(lats) + max(lats)) / 2
    cy = (min(lons) + max(lons)) / 2
    hw = (max(lats) - min(lats)) / 2 + 0.01
    hh = (max(lons) - min(lons)) / 2 + 0.01
    qt = QuadTree(BoundingBox(cx, cy, hw, hh))
    for (lat, lon), data in zip(points, data_list):
        qt.insert(Point(lat, lon, data))
    return qt


def run_benchmark(sizes=(1_000, 5_000, 10_000, 50_000, 100_000), n_queries=200, seed=1):
    rng = random.Random(seed)
    rows = []

    # generate the largest dataset once, slice for smaller sizes so results are comparable
    max_n = max(sizes)
    all_points, all_data = generate_synthetic_data(max_n, seed=seed)

    for n in sizes:
        points, data_list = all_points[:n], all_data[:n]

        t0 = time.perf_counter()
        kdt = KDTree(points, data_list)
        kd_build = time.perf_counter() - t0

        t0 = time.perf_counter()
        qt = build_quadtree(points, data_list)
        qt_build = time.perf_counter() - t0

        # sample query targets from existing points (worst case realistic queries)
        targets = [points[rng.randrange(n)] for _ in range(n_queries)]

        t0 = time.perf_counter()
        for tx, ty in targets:
            kdt.k_nearest((tx, ty), k=10)
        kd_knn = (time.perf_counter() - t0) / n_queries

        t0 = time.perf_counter()
        for tx, ty in targets:
            qt.k_nearest(tx, ty, k=10)
        qt_knn = (time.perf_counter() - t0) / n_queries

        t0 = time.perf_counter()
        for tx, ty in targets:
            kdt.radius_query((tx, ty), radius=0.05)
        kd_radius = (time.perf_counter() - t0) / n_queries

        t0 = time.perf_counter()
        for tx, ty in targets:
            qt.radius_query(tx, ty, 0.05)
        qt_radius = (time.perf_counter() - t0) / n_queries

        rows.append({
            "n": n,
            "kd_build_s": kd_build, "qt_build_s": qt_build,
            "kd_knn_ms": kd_knn * 1000, "qt_knn_ms": qt_knn * 1000,
            "kd_radius_ms": kd_radius * 1000, "qt_radius_ms": qt_radius * 1000,
        })
        print(f"n={n:>7}  build(kd/qt)={kd_build:.3f}s/{qt_build:.3f}s  "
              f"kNN(kd/qt)={kd_knn*1000:.3f}ms/{qt_knn*1000:.3f}ms  "
              f"radius(kd/qt)={kd_radius*1000:.3f}ms/{qt_radius*1000:.3f}ms")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "benchmark_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved results to {csv_path}")
    return rows


def plot_results(rows):
    import matplotlib.pyplot as plt

    ns = [r["n"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].plot(ns, [r["kd_build_s"] for r in rows], marker="o", label="KD-Tree")
    axes[0].plot(ns, [r["qt_build_s"] for r in rows], marker="s", label="Quadtree")
    axes[0].set_title("Build Time")
    axes[0].set_xlabel("N points")
    axes[0].set_ylabel("Seconds")
    axes[0].legend()

    axes[1].plot(ns, [r["kd_knn_ms"] for r in rows], marker="o", label="KD-Tree")
    axes[1].plot(ns, [r["qt_knn_ms"] for r in rows], marker="s", label="Quadtree")
    axes[1].set_title("k-NN Query Time (avg)")
    axes[1].set_xlabel("N points")
    axes[1].set_ylabel("Milliseconds")
    axes[1].legend()

    axes[2].plot(ns, [r["kd_radius_ms"] for r in rows], marker="o", label="KD-Tree")
    axes[2].plot(ns, [r["qt_radius_ms"] for r in rows], marker="s", label="Quadtree")
    axes[2].set_title("Radius Query Time (avg)")
    axes[2].set_xlabel("N points")
    axes[2].set_ylabel("Milliseconds")
    axes[2].legend()

    fig.suptitle("KD-Tree vs Quadtree: NestScope Benchmark")
    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "benchmark_chart.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved chart to {out_path}")


if __name__ == "__main__":
    results = run_benchmark()
    plot_results(results)
