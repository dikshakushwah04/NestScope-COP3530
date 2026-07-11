# NestScope: A Spatial Indexing Engine for STR Arbitrage Discovery

Solo project for COP3530 (Data Structures and Algorithms), Project 2 — instructor-approved solo submission.

**Author:** Diksha Kushwah

## Overview

NestScope compares two spatial data structures — a **KD-Tree** and a **Quadtree**, both implemented from scratch — for solving the same task: fast nearest-neighbor and radius queries over short-term rental listing data. The goal is to help identify comparable properties near a given location quickly, at scale, and to benchmark how each structure performs as the dataset grows.

## Problem

See `report/Report.pdf` for the full write-up. In short: evaluating a short-term rental location for co-hosting or investment requires comparing it against many nearby listings, which doesn't scale with naive linear search.

## Data

[Inside Airbnb](https://insideairbnb.com/get-the-data/) — public, CC-licensed listing data. See `data/README.md` for which cities/files were used and how to regenerate the combined dataset (100,000+ rows).

## How to run

```bash
# 1. Clone the repo
git clone <repo-url>
cd NestScope-COP3530

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the CLI
python src/cli.py

# 4. Run benchmarks
python src/benchmark.py

# 5. Run tests
pytest tests/
```

## Repo structure

```
src/        Core implementation (KD-Tree, Quadtree, CLI, benchmarking)
data/       Raw and cleaned listing data
tests/      Unit tests for both data structures
results/    Benchmark output and charts
report/     Final PDF report
```

## Features

- k-nearest-neighbor query on a given location
- Radius query for all comparable listings within a distance
- Side-by-side performance comparison (build time, query time) between KD-Tree and Quadtree as dataset size scales

## License

Data used under Inside Airbnb's data license. Code is original work for COP3530.
