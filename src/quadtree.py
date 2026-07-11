"""
quadtree.py

From-scratch implementation of a 2D Quadtree (region quadtree, point-based)
for spatial nearest-neighbor and radius queries over listing data (lat, long).

Big-O summary (n = number of points, reasonably uniform distribution):
    Build:          O(n log n)   -- amortized, n inserts each O(log n)
    Radius query:   O(log n + m) average, O(n) worst case (heavily clustered)
    k-NN query:     O(log n + k) average via expanding-radius search over
                     radius_query, O(n) worst case
"""

import math


class Point:
    __slots__ = ("x", "y", "data")

    def __init__(self, x, y, data=None):
        self.x = x
        self.y = y
        self.data = data if data is not None else {}


class BoundingBox:
    __slots__ = ("cx", "cy", "half_w", "half_h")

    def __init__(self, cx, cy, half_w, half_h):
        self.cx = cx
        self.cy = cy
        self.half_w = half_w
        self.half_h = half_h

    def contains(self, point):
        return (self.cx - self.half_w <= point.x <= self.cx + self.half_w and
                self.cy - self.half_h <= point.y <= self.cy + self.half_h)

    def intersects_circle(self, x, y, r):
        """Closest-point-on-rect-to-circle-center distance check."""
        closest_x = max(self.cx - self.half_w, min(x, self.cx + self.half_w))
        closest_y = max(self.cy - self.half_h, min(y, self.cy + self.half_h))
        return (closest_x - x) ** 2 + (closest_y - y) ** 2 <= r * r


class QuadTree:
    CAPACITY = 8  # max points per node before subdividing

    def __init__(self, boundary, depth=0, max_depth=20):
        self.boundary = boundary
        self.points = []
        self.divided = False
        self.depth = depth
        self.max_depth = max_depth
        self.nw = self.ne = self.sw = self.se = None

    def _subdivide(self):
        cx, cy, hw, hh = self.boundary.cx, self.boundary.cy, self.boundary.half_w / 2, self.boundary.half_h / 2
        self.nw = QuadTree(BoundingBox(cx - hw, cy + hh, hw, hh), self.depth + 1, self.max_depth)
        self.ne = QuadTree(BoundingBox(cx + hw, cy + hh, hw, hh), self.depth + 1, self.max_depth)
        self.sw = QuadTree(BoundingBox(cx - hw, cy - hh, hw, hh), self.depth + 1, self.max_depth)
        self.se = QuadTree(BoundingBox(cx + hw, cy - hh, hw, hh), self.depth + 1, self.max_depth)
        self.divided = True
        # push existing points down
        old_points = self.points
        self.points = []
        for p in old_points:
            self._insert_into_children(p)

    def _insert_into_children(self, point):
        for child in (self.nw, self.ne, self.sw, self.se):
            if child.boundary.contains(point):
                child.insert(point)
                return

    def insert(self, point):
        if not self.boundary.contains(point):
            return False
        if not self.divided and (len(self.points) < self.CAPACITY or self.depth >= self.max_depth):
            self.points.append(point)
            return True
        if not self.divided:
            self._subdivide()
        self._insert_into_children(point)
        return True

    def radius_query(self, x, y, r, found=None):
        if found is None:
            found = []
        if not self.boundary.intersects_circle(x, y, r):
            return found
        r2 = r * r
        for p in self.points:
            d2 = (p.x - x) ** 2 + (p.y - y) ** 2
            if d2 <= r2:
                found.append((math.sqrt(d2), (p.x, p.y), p.data))
        if self.divided:
            for child in (self.nw, self.ne, self.sw, self.se):
                child.radius_query(x, y, r, found)
        return found

    def k_nearest(self, x, y, k=5, initial_radius=0.05, max_radius=180.0):
        """Expanding-radius search: doubles the search radius until >= k
        points are found, then returns the k closest."""
        radius = initial_radius
        found = []
        while len(found) < k and radius <= max_radius:
            found = self.radius_query(x, y, radius)
            radius *= 2
        found.sort(key=lambda item: item[0])
        return found[:k]

    def count(self):
        total = len(self.points)
        if self.divided:
            total += self.nw.count() + self.ne.count() + self.sw.count() + self.se.count()
        return total
