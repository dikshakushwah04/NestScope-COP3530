"""
kdtree.py

From-scratch implementation of a 2D KD-Tree for spatial nearest-neighbor
and radius queries over listing data (lat, long).

Big-O summary (n = number of points, balanced tree):
    Build:          O(n log n)   -- median-of-points split at each level
    k-NN query:     O(log n + k) average, O(n) worst case (degenerate tree)
    Radius query:   O(sqrt(n) + m) average, m = points returned
"""

import heapq
import math


class KDNode:
    __slots__ = ("point", "data", "left", "right", "axis")

    def __init__(self, point, data, axis):
        self.point = point      # (lat, long)
        self.data = data        # metadata dict (price, room_type, etc.)
        self.left = None
        self.right = None
        self.axis = axis        # 0 = split on lat, 1 = split on long


def _dist2(a, b):
    """Squared Euclidean distance (avoids sqrt for comparisons)."""
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


class KDTree:
    def __init__(self, points, data_list=None):
        """
        points: list of (lat, long) tuples
        data_list: parallel list of metadata dicts (optional)
        """
        if data_list is None:
            data_list = [{} for _ in points]
        items = list(zip(points, data_list))
        self.size = len(items)
        self.root = self._build(items, depth=0)

    def _build(self, items, depth):
        if not items:
            return None
        axis = depth % 2
        items.sort(key=lambda p: p[0][axis])
        mid = len(items) // 2
        point, data = items[mid]
        node = KDNode(point, data, axis)
        node.left = self._build(items[:mid], depth + 1)
        node.right = self._build(items[mid + 1:], depth + 1)
        return node

    def k_nearest(self, target, k=5):
        """Return the k nearest (dist, point, data) tuples, sorted nearest-first."""
        heap = []  # max-heap via negative distance
        counter = [0]  # unique tie-breaker so heap never compares points/data directly

        def visit(node):
            if node is None:
                return
            d2 = _dist2(target, node.point)
            counter[0] += 1
            if len(heap) < k:
                heapq.heappush(heap, (-d2, counter[0], node.point, node.data))
            elif d2 < -heap[0][0]:
                heapq.heapreplace(heap, (-d2, counter[0], node.point, node.data))

            axis = node.axis
            diff = target[axis] - node.point[axis]
            near, far = (node.left, node.right) if diff < 0 else (node.right, node.left)
            visit(near)
            # Only explore the far side if it could contain a closer point
            if len(heap) < k or diff * diff < -heap[0][0]:
                visit(far)

        visit(self.root)
        results = sorted(((-d2, p, dat) for d2, _, p, dat in heap), key=lambda x: x[0])
        return [(math.sqrt(d2), p, dat) for d2, p, dat in results]

    def radius_query(self, target, radius):
        """Return all (dist, point, data) within `radius` of target."""
        r2 = radius * radius
        results = []

        def visit(node):
            if node is None:
                return
            d2 = _dist2(target, node.point)
            if d2 <= r2:
                results.append((math.sqrt(d2), node.point, node.data))

            axis = node.axis
            diff = target[axis] - node.point[axis]
            if diff < 0:
                visit(node.left)
                if diff * diff <= r2:
                    visit(node.right)
            else:
                visit(node.right)
                if diff * diff <= r2:
                    visit(node.left)

        visit(self.root)
        results.sort(key=lambda x: x[0])
        return results
