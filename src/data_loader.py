"""
data_loader.py

Two ways to get listing data:
1. load_real_data(csv_paths)   -- load actual Inside Airbnb CSVs
   Download from https://insideairbnb.com/get-the-data/ (pick 2-3 cities'
   "listings.csv.gz" files to comfortably exceed 100,000 rows combined).
2. generate_synthetic_data(n)  -- generates realistic-looking listing data
   for development/testing without needing the download first.
"""

import csv
import random


ROOM_TYPES = ["Entire home/apt", "Private room", "Shared room", "Hotel room"]

# Rough (lat, long, radius_deg) for a few metro areas, used only for
# generating synthetic dev/test data that resembles real geographic clustering.
METRO_CENTERS = [
    (25.7617, -80.1918, 0.35),   # Miami
    (40.7128, -74.0060, 0.30),   # NYC
    (34.0522, -118.2437, 0.40),  # LA
    (29.9511, -90.0715, 0.20),   # New Orleans
    (47.6062, -122.3321, 0.25),  # Seattle
]


def generate_synthetic_data(n=100_000, seed=42):
    """Generates n synthetic listings clustered around several metro areas,
    mimicking the real Inside Airbnb schema."""
    rng = random.Random(seed)
    points = []
    data_list = []
    for i in range(n):
        cx, cy, spread = rng.choice(METRO_CENTERS)
        # Gaussian-ish clustering around city center, denser near downtown
        lat = rng.gauss(cx, spread / 3)
        lon = rng.gauss(cy, spread / 3)
        price = max(20, round(rng.gauss(150, 80)))
        room_type = rng.choices(ROOM_TYPES, weights=[55, 35, 5, 5])[0]
        min_nights = rng.choice([1, 2, 3, 7, 30])
        availability = rng.randint(0, 365)
        reviews = rng.randint(0, 400)

        points.append((lat, lon))
        data_list.append({
            "id": i,
            "price": price,
            "room_type": room_type,
            "minimum_nights": min_nights,
            "availability_365": availability,
            "number_of_reviews": reviews,
        })
    return points, data_list


def load_real_data(csv_paths):
    """
    Load one or more Inside Airbnb `listings.csv` files and combine them.
    Expects columns: id, latitude, longitude, price, room_type,
    minimum_nights, availability_365, number_of_reviews.

    Returns (points, data_list) in the same format as generate_synthetic_data.
    """
    points, data_list = [], []
    for path in csv_paths:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    lat = float(row["latitude"])
                    lon = float(row["longitude"])
                except (KeyError, ValueError):
                    continue
                price_raw = row.get("price", "0")
                price = _parse_price(price_raw)
                points.append((lat, lon))
                data_list.append({
                    "id": row.get("id"),
                    "price": price,
                    "room_type": row.get("room_type"),
                    "minimum_nights": row.get("minimum_nights"),
                    "availability_365": row.get("availability_365"),
                    "number_of_reviews": row.get("number_of_reviews"),
                })
    return points, data_list


def _parse_price(raw):
    if raw is None:
        return 0.0
    cleaned = str(raw).replace("$", "").replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return 0.0
