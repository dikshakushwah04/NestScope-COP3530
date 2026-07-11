"""
cli.py

Simple menu-driven interface for NestScope: STR Arbitrage Scout.
Loads synthetic data by default; point DATA_MODE at "real" and set
CSV_PATHS to use actual Inside Airbnb data.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from kdtree import KDTree
from quadtree import QuadTree, BoundingBox, Point
from data_loader import generate_synthetic_data, load_real_data

DATA_MODE = "synthetic"  # "synthetic" or "real"
CSV_PATHS = []           # e.g. ["data/miami_listings.csv", "data/nyc_listings.csv"]
N_SYNTHETIC = 100_000


def load_data():
    if DATA_MODE == "real" and CSV_PATHS:
        print(f"Loading real data from {CSV_PATHS} ...")
        return load_real_data(CSV_PATHS)
    print(f"Generating {N_SYNTHETIC:,} synthetic listings ...")
    return generate_synthetic_data(N_SYNTHETIC)


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


def print_banner():
    print("=" * 55)
    print("  NestScope -- STR Arbitrage Scout")
    print("=" * 55)


def print_results(label, elapsed_ms, results):
    print(f"\n[{label}] query time: {elapsed_ms:.3f} ms")
    if not results:
        print("  No comparable listings found in range.")
        return
    prices = [r[2].get("price", 0) for r in results]
    avg_price = sum(prices) / len(prices)
    print(f"  {len(results)} comparable listings found | Avg nightly rate: ${avg_price:.0f}")
    for dist, point, data in results[:5]:
        print(f"   - id={data.get('id')} price=${data.get('price')} "
              f"room={data.get('room_type')} dist={dist:.4f}")


def main():
    print_banner()
    points, data_list = load_data()
    print(f"Loaded {len(points):,} listings.\n")

    print("Building KD-Tree...")
    t0 = time.perf_counter()
    kdt = KDTree(points, data_list)
    print(f"  done in {time.perf_counter() - t0:.2f}s")

    print("Building Quadtree...")
    t0 = time.perf_counter()
    qt = build_quadtree(points, data_list)
    print(f"  done in {time.perf_counter() - t0:.2f}s\n")

    while True:
        print("-" * 55)
        raw = input("Enter location as 'lat,long' (or 'q' to quit): ").strip()
        if raw.lower() == "q":
            break
        try:
            lat_str, lon_str = raw.split(",")
            lat, lon = float(lat_str), float(lon_str)
        except ValueError:
            print("Couldn't parse that. Format: 25.76,-80.19")
            continue

        radius_str = input("Search radius in degrees (default 0.05 ~ a few miles): ").strip()
        radius = float(radius_str) if radius_str else 0.05

        structure = input("Structure -- [1] KD-Tree  [2] Quadtree  [3] Both (default 3): ").strip()
        structure = structure or "3"

        if structure in ("1", "3"):
            t0 = time.perf_counter()
            results = kdt.radius_query((lat, lon), radius)
            elapsed = (time.perf_counter() - t0) * 1000
            print_results("KD-Tree", elapsed, results)

        if structure in ("2", "3"):
            t0 = time.perf_counter()
            results = qt.radius_query(lat, lon, radius)
            elapsed = (time.perf_counter() - t0) * 1000
            print_results("Quadtree", elapsed, results)

    print("\nGoodbye!")


if __name__ == "__main__":
    main()
