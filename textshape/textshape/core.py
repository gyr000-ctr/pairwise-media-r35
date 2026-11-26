"""Glyph contour triangulation.

Public API::

    TextShape().triangulate(contours) -> list[tuple[int, int, int]]
    TextShape().area(contours) -> float

``contours`` is a sequence of rings; each ring is a sequence of ``(x, y)``
points.  Rings are classified by nesting depth: a ring contained in an even
number of other rings is a filled outline, a ring contained in an odd number
of other rings is a hole.  Equivalently the filled region follows the
non-zero winding rule once every ring's orientation is taken from its depth
(which is how glyph outlines are normally emitted).  The hole test in the
project spec uses two counter-clockwise, nested rings, so depth parity -- not
the raw winding number -- is the authoritative rule here.

The returned triangles index into the flattened input vertex list (rings
concatenated in argument order, points in ring order).  Every triangle is
counter-clockwise (positive signed area), triangles never overlap, and holes
are left unfilled.

The implementation is self-contained (standard library only) and keeps no
mutable per-instance state, so one :class:`TextShape` may safely be shared
between threads.

Invalid input is rejected with distinct :class:`TextShapeError` subclasses
(all of which derive from :class:`ValueError`).
"""

from __future__ import annotations

import math

__all__ = [
    "TextShape",
    "triangulate",
    "area",
    "TextShapeError",
    "InvalidContourError",
    "ContourTypeError",
    "EmptyContourError",
    "NonFiniteCoordinateError",
    "NonFiniteInputError",
    "DegenerateContourError",
    "SelfIntersectionError",
    "ContourIntersectionError",
    "TriangulationError",
    "MalformedContourError",
    "BadContourError",
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class TextShapeError(ValueError):
    """Base class for invalid contour input."""


class InvalidContourError(TextShapeError):
    """The contour structure is malformed."""


class ContourTypeError(TypeError, InvalidContourError):
    """A contour, point or coordinate has the wrong type.

    Inherits from both :class:`TypeError` and :class:`ValueError` so callers
    using either built-in category (or the TextShape hierarchy) can catch it.
    """


# Compatibility aliases for the same structural problem.
MalformedContourError = ContourTypeError
BadContourError = InvalidContourError


class EmptyContourError(InvalidContourError):
    """No contours were supplied."""


class NonFiniteCoordinateError(InvalidContourError):
    """A coordinate is NaN or infinite (or the range overflows)."""


# Alternate naming used by some callers/spec sheets.
NonFiniteInputError = NonFiniteCoordinateError


class DegenerateContourError(InvalidContourError):
    """A ring has fewer than three distinct points or zero area."""


class SelfIntersectionError(InvalidContourError):
    """A ring intersects itself."""


class ContourIntersectionError(InvalidContourError):
    """Two distinct rings intersect or touch each other."""


class TriangulationError(InvalidContourError):
    """The ear-clipping stage could not finish on a nominally valid ring."""


# Intersection tests operate on coordinates normalised to the unit square.
_EPS = 1e-12


# ---------------------------------------------------------------------------
# Small numeric helpers (normalised coordinates)
# ---------------------------------------------------------------------------

def _cross(ax, ay, bx, by, cx, cy):
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _sign(v):
    if v > _EPS:
        return 1
    if v < -_EPS:
        return -1
    return 0


def _on_segment(px, py, qx, qy, rx, ry):
    """True when q lies on segment p-r (collinear case)."""
    return (min(px, rx) - _EPS <= qx <= max(px, rx) + _EPS and
            min(py, ry) - _EPS <= qy <= max(py, ry) + _EPS)


def _segments_intersect(ax, ay, bx, by, cx, cy, dx, dy):
    """Strict-or-touching intersection of segments a-b and c-d."""
    o1 = _sign(_cross(ax, ay, bx, by, cx, cy))
    o2 = _sign(_cross(ax, ay, bx, by, dx, dy))
    o3 = _sign(_cross(cx, cy, dx, dy, ax, ay))
    o4 = _sign(_cross(cx, cy, dx, dy, bx, by))
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _on_segment(ax, ay, cx, cy, bx, by):
        return True
    if o2 == 0 and _on_segment(ax, ay, dx, dy, bx, by):
        return True
    if o3 == 0 and _on_segment(cx, cy, ax, ay, dx, dy):
        return True
    if o4 == 0 and _on_segment(cx, cy, bx, by, dx, dy):
        return True
    return False


def _point_in_ring(px, py, pts):
    """Crossing-number point-in-polygon test for a closed ring."""
    inside = False
    n = len(pts)
    jx, jy = pts[-1][1], pts[-1][2]
    for k in range(n):
        ix, iy = pts[k][1], pts[k][2]
        if (iy > py) != (jy > py):
            xint = ix + (py - iy) * (jx - ix) / (jy - iy)
            if px < xint:
                inside = not inside
        jx, jy = ix, iy
    return inside


def _ring_signed_area(pts):
    """Twice the signed area of a ring given as (index, x, y) tuples."""
    total = 0.0
    n = len(pts)
    jx, jy = pts[-1][1], pts[-1][2]
    for k in range(n):
        ix, iy = pts[k][1], pts[k][2]
        total += jx * iy - ix * jy
        jx, jy = ix, iy
    return 0.5 * total


# ---------------------------------------------------------------------------
# Input parsing / validation
# ---------------------------------------------------------------------------

def _is_point_like(obj):
    if isinstance(obj, (str, bytes)):
        return False
    try:
        if len(obj) != 2:
            return False
    except TypeError:
        return False
    a, b = obj[0], obj[1]
    if isinstance(a, (str, bytes)) or isinstance(b, (str, bytes)):
        return False
    try:
        float(a)
        float(b)
    except (TypeError, ValueError, OverflowError):
        return False
    return True


def _as_float(value):
    if isinstance(value, (str, bytes)):
        raise ContourTypeError("coordinate must be numeric, got text")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ContourTypeError(
            "coordinate must be a real number, got %r" % (value,)
        ) from exc
    if not math.isfinite(result):
        raise NonFiniteCoordinateError(
            "coordinate must be finite, got %r" % (value,)
        )
    return result


class _Ring:
    __slots__ = ("pts", "area_raw", "area_n", "bbox", "depth", "parent",
                 "rep")

    def __init__(self):
        self.pts = []          # list of (global_index, nx, ny)
        self.area_raw = 0.0
        self.area_n = 0.0
        self.bbox = (0.0, 0.0, 0.0, 0.0)  # minx, miny, maxx, maxy
        self.depth = 0
        self.parent = -1
        self.rep = None


def _parse_contours(contours):
    """Validate and normalise ``contours``.

    Returns the list of rings in input order; every ring carries normalised
    vertices, its signed area, bounding box and nesting depth.
    """
    if contours is None:
        raise ContourTypeError("contours must be a sequence of rings")

    if isinstance(contours, (str, bytes)):
        raise ContourTypeError("contours must be a sequence of rings")
    try:
        rings_in = list(contours)
    except TypeError as exc:
        raise ContourTypeError(
            "contours must be a sequence of rings"
        ) from exc

    if not rings_in:
        raise EmptyContourError("at least one contour is required")

    # Convenience: accept a bare ring (a sequence of points) as one contour.
    if all(_is_point_like(item) for item in rings_in):
        rings_in = [rings_in]

    # --- parse raw coordinates ------------------------------------------
    raw_rings = []  # list[list[(global_idx, x, y)]]
    global_idx = 0
    for ring in rings_in:
        if isinstance(ring, (str, bytes)):
            raise ContourTypeError("a contour must be a sequence of points")
        try:
            points = list(ring)
        except TypeError as exc:
            raise ContourTypeError(
                "a contour must be a sequence of points"
            ) from exc
        parsed = []
        for point in points:
            if isinstance(point, (str, bytes)):
                raise ContourTypeError(
                    "a point must be a pair of numbers"
                )
            try:
                if len(point) != 2:
                    raise InvalidContourError(
                        "a point must have exactly two coordinates"
                    )
                px = _as_float(point[0])
                py = _as_float(point[1])
            except (TypeError, LookupError) as exc:
                raise ContourTypeError(
                    "a point must be a pair of numbers"
                ) from exc
            parsed.append((global_idx, px, py))
            global_idx += 1

        # Remove consecutive duplicates and a duplicated closing vertex.
        cleaned = []
        for item in parsed:
            if cleaned and item[1] == cleaned[-1][1] and \
                    item[2] == cleaned[-1][2]:
                continue
            cleaned.append(item)
        if len(cleaned) > 1 and cleaned[0][1] == cleaned[-1][1] and \
                cleaned[0][2] == cleaned[-1][2]:
            cleaned.pop()

        if len(cleaned) < 3:
            raise DegenerateContourError(
                "a contour needs at least three distinct points"
            )
        raw_rings.append(cleaned)

    # --- global normalisation -------------------------------------------
    xs = [p[1] for ring in raw_rings for p in ring]
    ys = [p[2] for ring in raw_rings for p in ring]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    width = maxx - minx
    height = maxy - miny
    if not (math.isfinite(width) and math.isfinite(height)):
        raise NonFiniteCoordinateError(
            "coordinate range is not representable as a finite number"
        )
    scale = max(width, height)
    if scale <= 0.0:
        raise DegenerateContourError("all contours collapse to a single point")
    inv = 1.0 / scale

    rings = []
    for raw in raw_rings:
        ring = _Ring()
        ring.pts = [((i, (x - minx) * inv, (y - miny) * inv))
                    for (i, x, y) in raw]
        n = len(ring.pts)
        terms = []
        jx, jy = ring.pts[-1][1], ring.pts[-1][2]
        for (_, ix, iy) in ring.pts:
            terms.append(jx * iy - ix * jy)
            jx, jy = ix, iy
        ring.area_n = 0.5 * math.fsum(terms)
        # Raw signed area, computed with a compensated sum for accuracy.
        terms = []
        jx, jy = raw[-1][1], raw[-1][2]
        for (_, ix, iy) in raw:
            terms.append(jx * iy - ix * jy)
            jx, jy = ix, iy
        ring.area_raw = 0.5 * math.fsum(terms)
        xs = [p[1] for p in ring.pts]
        ys = [p[2] for p in ring.pts]
        ring.bbox = (min(xs), min(ys), max(xs), max(ys))
        bw = ring.bbox[2] - ring.bbox[0]
        bh = ring.bbox[3] - ring.bbox[1]
        if bw <= 0.0 or bh <= 0.0:
            raise DegenerateContourError(
                "a contour has zero area or collinear points"
            )
        ring.rep = ring.pts[0]
        rings.append(ring)

    # Self-intersection is diagnosed before the signed-area check: a bowtie
    # has cancelling signed area but is first and foremost self-intersecting.
    _check_intersections(rings)

    # For simple rings a near-zero signed area means a collinear/degenerate.
    for ring in rings:
        bw = ring.bbox[2] - ring.bbox[0]
        bh = ring.bbox[3] - ring.bbox[1]
        if abs(ring.area_n) <= 1e-12 * bw * bh:
            raise DegenerateContourError(
                "a contour has zero area or collinear points"
            )

    _compute_depths(rings)
    return rings


def _sweep_pairs(segments, nsegs):
    """Yield candidate segment-index pairs via a Shamos-Hoey style sweep.

    Only segments that become vertical neighbours in y-order at an x-event are
    yielded (plus small neighbour radii and explicit vertical handling), which
    is an exact superset of intersecting pairs for existence detection while
    staying near O(n log n) on outlines made of many long radial edges.
    """
    starts = {}
    ends = {}
    verticals = {}
    x_events = set()
    slopes = [0.0] * nsegs

    for i, seg in enumerate(segments):
        ax, ay, bx, by = seg[7], seg[8], seg[9], seg[10]
        if ax == bx:
            verticals.setdefault(ax, []).append(i)
            x_events.add(ax)
            continue
        slope = (by - ay) / (bx - ax)
        slopes[i] = slope
        if ax < bx:
            starts.setdefault(ax, []).append(i)
            ends.setdefault(bx, []).append(i)
            x_events.add(ax)
            x_events.add(bx)
        else:
            # Segment records keep original orientation; order by x here.
            starts.setdefault(bx, []).append(i)
            ends.setdefault(ax, []).append(i)
            x_events.add(ax)
            x_events.add(bx)

    def y_at(i, x):
        seg = segments[i]
        ax, ay, bx, by = seg[7], seg[8], seg[9], seg[10]
        if bx == ax:
            return 0.5 * (ay + by)
        return ay + (x - ax) * (by - ay) / (bx - ax)

    def key_at(i, x):
        return (y_at(i, x), slopes[i], i)

    active = []
    for x in sorted(x_events):
        for i in starts.get(x, ()):  # insertion order irrelevant with radius
            ki = key_at(i, x)
            lo, hi = 0, len(active)
            while lo < hi:
                mid = (lo + hi) // 2
                if key_at(active[mid], x) < ki:
                    lo = mid + 1
                else:
                    hi = mid
            pos = lo
            for q in range(max(0, pos - 2), min(len(active), pos + 3)):
                j = active[q]
                yield (i, j) if i < j else (j, i)
            active.insert(pos, i)

        at_x = active
        for vi in verticals.get(x, ()):  # vertical segments at this x
            seg = segments[vi]
            y0 = min(seg[8], seg[10])
            y1 = max(seg[8], seg[10])
            lo, hi = 0, len(at_x)
            while lo < hi:
                mid = (lo + hi) // 2
                if y_at(at_x[mid], x) < y0 - _EPS:
                    lo = mid + 1
                else:
                    hi = mid
            p = lo
            while p < len(at_x):
                j = at_x[p]
                if y_at(j, x) > y1 + _EPS:
                    break
                yield (vi, j) if vi < j else (j, vi)
                p += 1
        vl = verticals.get(x, ())
        for u in range(len(vl)):
            for v in range(u + 1, len(vl)):
                a, b = vl[u], vl[v]
                yield (a, b) if a < b else (b, a)

        for i in ends.get(x, ()):
            ki = key_at(i, x)
            lo, hi = 0, len(active)
            while lo < hi:
                mid = (lo + hi) // 2
                if key_at(active[mid], x) < ki:
                    lo = mid + 1
                else:
                    hi = mid
            found = -1
            for q in range(max(0, lo - 3), min(len(active), lo + 4)):
                if active[q] == i:
                    found = q
                    break
            if found < 0:
                try:
                    found = active.index(i)
                except ValueError:
                    continue
            if found - 1 >= 0:
                j = active[found - 1]
                yield (i, j) if i < j else (j, i)
            if found + 1 < len(active):
                j = active[found + 1]
                yield (i, j) if i < j else (j, i)
            left = active[found - 1] if found - 1 >= 0 else -1
            right = active[found + 1] if found + 1 < len(active) else -1
            if left >= 0 and right >= 0:
                yield (left, right) if left < right else (right, left)
            active.pop(found)


def _check_intersections(rings):
    """Detect self-intersections and ring/ring crossings."""
    segments = []
    for ridx, ring in enumerate(rings):
        pts = ring.pts
        n = len(pts)
        for k in range(n):
            ax, ay = pts[k][1], pts[k][2]
            bx, by = pts[(k + 1) % n][1], pts[(k + 1) % n][2]
            if ax == bx and ay == by:
                continue
            segments.append((
                min(ax, bx), max(ax, bx), min(ay, by), max(ay, by),
                ridx, k, n, ax, ay, bx, by,
            ))
    nsegs = len(segments)
    if nsegs < 2:
        return

    if nsegs <= 64:
        pairs = ((i, j) for i in range(nsegs) for j in range(i + 1, nsegs))
    else:
        pairs = _sweep_pairs(segments, nsegs)

    for a, b in pairs:
        sa = segments[a]
        sb = segments[b]
        a0, a1, a2, a3 = sa[0], sa[1], sa[2], sa[3]
        if sb[0] > a1 + _EPS or sb[1] < a0 - _EPS:
            continue
        if sb[2] > a3 + _EPS or sb[3] < a2 - _EPS:
            continue
        if sb[4] == sa[4]:
            # Adjacent edges of one ring share an endpoint by construction.
            if (sb[5] - sa[5]) % sa[6] in (1, sa[6] - 1):
                continue
            if _segments_intersect(sa[7], sa[8], sa[9], sa[10],
                                   sb[7], sb[8], sb[9], sb[10]):
                raise SelfIntersectionError(
                    "contour %d intersects itself" % sa[4]
                )
        elif _segments_intersect(sa[7], sa[8], sa[9], sa[10],
                                 sb[7], sb[8], sb[9], sb[10]):
            lo, hi = (sa[4], sb[4]) if sa[4] < sb[4] else (sb[4], sa[4])
            raise ContourIntersectionError(
                "contours %d and %d intersect" % (lo, hi)
            )


def _compute_depths(rings):
    """Assign each ring its nesting depth and immediate parent.

    Every ring containing this ring's representative point contributes one
    nesting level.  Because non-intersecting, non-degenerate rings form a
    nesting forest, the immediate parent is simply the innermost container,
    i.e. the container with the smallest absolute area.
    """
    for i, ring in enumerate(rings):
        px, py = ring.rep[1], ring.rep[2]
        depth = 0
        parent = -1
        parent_area = math.inf
        for j, other in enumerate(rings):
            if i == j:
                continue
            bx0, by0, bx1, by1 = other.bbox
            if not (bx0 - _EPS <= px <= bx1 + _EPS and
                    by0 - _EPS <= py <= by1 + _EPS):
                continue
            if _point_in_ring(px, py, other.pts):
                depth += 1
                area = abs(other.area_n)
                if area < parent_area:
                    parent_area = area
                    parent = j
        ring.depth = depth
        ring.parent = parent if depth > 0 else -1


# ---------------------------------------------------------------------------
# Earcut (port of mapbox/earcut, ISC licence, adapted to Python)
# ---------------------------------------------------------------------------

class _Node:
    __slots__ = ("i", "x", "y", "prev", "next", "z", "prevZ", "nextZ",
                 "steiner")

    def __init__(self, i, x, y):
        self.i = i
        self.x = x
        self.y = y
        self.prev = None
        self.next = None
        self.z = 0
        self.prevZ = None
        self.nextZ = None
        self.steiner = False


def _equals(p, q):
    return p.x == q.x and p.y == q.y


def _node_index(node):
    return node.i


def _area(p, q, r):
    return (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)


def _insert_node(i, x, y, last):
    p = _Node(i, x, y)
    if last is None:
        p.prev = p
        p.next = p
    else:
        p.next = last.next
        p.prev = last
        last.next.prev = p
        last.next = p
    return p


def _remove_node(p):
    p.next.prev = p.prev
    p.prev.next = p.next
    if p.prevZ is not None:
        p.prevZ.nextZ = p.nextZ
    if p.nextZ is not None:
        p.nextZ.prevZ = p.prevZ


def _linked_list(pts, clockwise):
    """Build a circular linked list, matching earcut's own convention.

    earcut's ``signedArea`` is the standard shoelace sum, so
    ``clockwise == (signed > 0)`` keeps the original order.  Callers pass
    ``clockwise=True`` for outer rings (ending up counter-clockwise) and
    ``False`` for holes (ending up clockwise).
    """
    last = None
    signed = _ring_signed_area(pts)
    if clockwise == (signed > 0):
        for (i, x, y) in pts:
            last = _insert_node(i, x, y, last)
    else:
        for (i, x, y) in reversed(pts):
            last = _insert_node(i, x, y, last)
    if last is not None and _equals(last, last.next):
        _remove_node(last)
        last = last.next
    return last


def _filter_points(start, end=None):
    if start is None:
        return start
    if end is None:
        end = start
    p = start
    again = True
    while again or p is not end:
        again = False
        if not p.steiner and (
                _equals(p, p.next) or _area(p.prev, p, p.next) == 0):
            _remove_node(p)
            p = p.prev
            end = p
            if p is p.next:
                break
            again = True
        else:
            p = p.next
    return end


def _sector_contains_sector(m, p):
    return (m.prev is not None and p.prev is not None and
            _area(m.prev, m, p.prev) < 0 and
            _area(p.next, m, m.next) < 0)


def _find_hole_bridge(hole, outer_node):
    p = outer_node
    hx = hole.x
    hy = hole.y
    qx = -math.inf
    m = None

    while True:
        if (hy <= p.y and hy >= p.next.y and p.next.y != p.y):
            x = p.x + (hy - p.y) * (p.next.x - p.x) / (p.next.y - p.y)
            if x <= hx and x > qx:
                qx = x
                m = p if p.x < p.next.x else p.next
                if x == hx:
                    return m
        p = p.next
        if p is outer_node:
            break

    if m is None:
        return None

    stop = m
    mx = m.x
    my = m.y
    tan_min = math.inf
    p = m
    while True:
        if (hx >= p.x and p.x >= mx and hx != p.x and
                _point_in_triangle(
                    hx if hy < my else qx, hy,
                    mx, my,
                    qx if hy < my else hx, hy,
                    p.x, p.y)):
            tan = abs(hy - p.y) / (hx - p.x)
            if (_locally_inside(p, hole) and
                    (tan < tan_min or
                     (tan == tan_min and
                      (p.x > m.x or
                       (p.x == m.x and _sector_contains_sector(m, p)))))):
                m = p
                tan_min = tan
        p = p.next
        if p is stop:
            break

    return m


def _eliminate_hole(hole, outer_node):
    bridge = _find_hole_bridge(hole, outer_node)
    if bridge is None:
        return outer_node
    bridge_reverse = _split_polygon(bridge, hole)
    _filter_points(bridge_reverse, bridge_reverse.next)
    return _filter_points(bridge, bridge.next)


def _eliminate_holes(holes, outer_node):
    queue = []
    for pts in holes:
        lst = _linked_list(pts, False)
        if lst is None:
            continue
        if lst is lst.next:
            lst.steiner = True
        queue.append(_get_leftmost(lst))
    queue.sort(key=lambda node: node.x)
    for hole in queue:
        outer_node = _eliminate_hole(hole, outer_node)
    return outer_node


def _get_leftmost(start):
    p = start
    leftmost = start
    while True:
        if p.x < leftmost.x or (p.x == leftmost.x and p.y < leftmost.y):
            leftmost = p
        p = p.next
        if p is start:
            break
    return leftmost


def _point_in_triangle(ax, ay, bx, by, cx, cy, px, py):
    return ((cx - px) * (ay - py) >= (ax - px) * (cy - py) and
            (ax - px) * (by - py) >= (bx - px) * (ay - py) and
            (bx - px) * (cy - py) >= (cx - px) * (by - py))


def _is_ear(ear):
    a = ear.prev
    b = ear
    c = ear.next
    if _area(a, b, c) >= 0:
        return False
    ax, ay = a.x, a.y
    bx, by = b.x, b.y
    cx, cy = c.x, c.y
    x0 = min(ax, bx, cx)
    y0 = min(ay, by, cy)
    x1 = max(ax, bx, cx)
    y1 = max(ay, by, cy)
    p = c.next
    while p is not a:
        if (x0 <= p.x <= x1 and y0 <= p.y <= y1 and
                _point_in_triangle(ax, ay, bx, by, cx, cy, p.x, p.y) and
                _area(p.prev, p, p.next) >= 0):
            return False
        p = p.next
    return True


def _z_order(x, y, min_x, min_y, inv_size):
    ix = int((x - min_x) * inv_size) & 0xFFFF
    iy = int((y - min_y) * inv_size) & 0xFFFF
    ix = (ix | (ix << 8)) & 0x00FF00FF
    ix = (ix | (ix << 4)) & 0x0F0F0F0F
    ix = (ix | (ix << 2)) & 0x33333333
    ix = (ix | (ix << 1)) & 0x55555555
    iy = (iy | (iy << 8)) & 0x00FF00FF
    iy = (iy | (iy << 4)) & 0x0F0F0F0F
    iy = (iy | (iy << 2)) & 0x33333333
    iy = (iy | (iy << 1)) & 0x55555555
    return ix | (iy << 1)


def _index_curve(start, min_x, min_y, inv_size):
    p = start
    while True:
        if p.z == 0:
            p.z = _z_order(p.x, p.y, min_x, min_y, inv_size)
        p.prevZ = p.prev
        p.nextZ = p.next
        p = p.next
        if p is start:
            break

    p.prevZ.nextZ = None
    p.prevZ = None
    _sort_linked(p)


def _sort_linked(lst):
    in_size = 1
    while True:
        p = lst
        lst = None
        tail = None
        num_merges = 0
        while p is not None:
            num_merges += 1
            q = p
            p_size = 0
            for _ in range(in_size):
                p_size += 1
                q = q.nextZ
                if q is None:
                    break
            q_size = in_size
            while p_size > 0 or (q_size > 0 and q is not None):
                if p_size != 0 and (q_size == 0 or q is None or
                                    p.z <= q.z):
                    e = p
                    p = p.nextZ
                    p_size -= 1
                else:
                    e = q
                    q = q.nextZ
                    q_size -= 1
                if tail is not None:
                    tail.nextZ = e
                else:
                    lst = e
                e.prevZ = tail
                tail = e
            p = q
        tail.nextZ = None
        if num_merges <= 1:
            return lst
        in_size *= 2


def _is_ear_hashed(ear, min_x, min_y, inv_size):
    a = ear.prev
    b = ear
    c = ear.next
    if _area(a, b, c) >= 0:
        return False
    ax, ay = a.x, a.y
    bx, by = b.x, b.y
    cx, cy = c.x, c.y
    x0 = min(ax, bx, cx)
    y0 = min(ay, by, cy)
    x1 = max(ax, bx, cx)
    y1 = max(ay, by, cy)
    min_z = _z_order(x0, y0, min_x, min_y, inv_size)
    max_z = _z_order(x1, y1, min_x, min_y, inv_size)

    p = ear.prevZ
    n = ear.nextZ
    while p is not None and p.z >= min_z and n is not None and n.z <= max_z:
        if (p is not ear.prev and p is not ear.next and
                _point_in_triangle(ax, ay, bx, by, cx, cy, p.x, p.y) and
                _area(p.prev, p, p.next) >= 0):
            return False
        p = p.prevZ
        if (n is not ear.prev and n is not ear.next and
                _point_in_triangle(ax, ay, bx, by, cx, cy, n.x, n.y) and
                _area(n.prev, n, n.next) >= 0):
            return False
        n = n.nextZ
    while p is not None and p.z >= min_z:
        if (p is not ear.prev and p is not ear.next and
                _point_in_triangle(ax, ay, bx, by, cx, cy, p.x, p.y) and
                _area(p.prev, p, p.next) >= 0):
            return False
        p = p.prevZ
    while n is not None and n.z <= max_z:
        if (n is not ear.prev and n is not ear.next and
                _point_in_triangle(ax, ay, bx, by, cx, cy, n.x, n.y) and
                _area(n.prev, n, n.next) >= 0):
            return False
        n = n.nextZ
    return True


def _locally_inside(a, b):
    if _area(a.prev, a, a.next) < 0:
        return _area(a, b, a.next) >= 0 and _area(a, a.prev, b) >= 0
    return _area(a, b, a.prev) < 0 or _area(a, a.next, b) < 0


def _middle_inside(a, b):
    p = a
    inside = False
    px = (a.x + b.x) / 2
    py = (a.y + b.y) / 2
    while True:
        if (((p.y > py) != (p.next.y > py)) and p.next.y != p.y and
                (px < (p.next.x - p.x) * (py - p.y) /
                 (p.next.y - p.y) + p.x)):
            inside = not inside
        p = p.next
        if p is a:
            break
    return inside


def _intersects(p1, q1, p2, q2):
    o1 = _sign(_area(p1, q1, p2))
    o2 = _sign(_area(p1, q1, q2))
    o3 = _sign(_area(p2, q2, p1))
    o4 = _sign(_area(p2, q2, q1))
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _on_segment(p1.x, p1.y, p2.x, p2.y, q1.x, q1.y):
        return True
    if o2 == 0 and _on_segment(p1.x, p1.y, q2.x, q2.y, q1.x, q1.y):
        return True
    if o3 == 0 and _on_segment(p2.x, p2.y, p1.x, p1.y, q2.x, q2.y):
        return True
    if o4 == 0 and _on_segment(p2.x, p2.y, q1.x, q1.y, q2.x, q2.y):
        return True
    return False


def _intersects_polygon(a, b):
    p = a
    while True:
        if (p.i != a.i and p.next.i != a.i and
                p.i != b.i and p.next.i != b.i and
                _intersects(p, p.next, a, b)):
            return True
        p = p.next
        if p is a:
            break
    return False


def _is_valid_diagonal(a, b):
    return (a.next.i != b.i and a.prev.i != b.i and
            not _intersects_polygon(a, b) and
            ((_locally_inside(a, b) and _locally_inside(b, a) and
              _middle_inside(a, b) and
              (_area(a.prev, a, b.prev) or _area(a, b.prev, b))) or
             (_equals(a, b) and
              _area(a.prev, a, a.next) > 0 and
              _area(b.prev, b, b.next) > 0)))


def _split_polygon(a, b):
    a2 = _Node(a.i, a.x, a.y)
    b2 = _Node(b.i, b.x, b.y)
    an = a.next
    bp = b.prev

    a.next = b
    b.prev = a

    a2.next = an
    an.prev = a2

    b2.next = a2
    a2.prev = b2

    bp.next = b2
    b2.prev = bp

    return b2


def _cure_local_intersections(start, triangles):
    p = start
    while True:
        a = p.prev
        b = p.next.next
        if (not _equals(a, b) and
                _intersects(a, p, p.next, b) and
                _locally_inside(a, b) and _locally_inside(b, a)):
            triangles.append((a.i, p.i, b.i))
            _remove_node(p)
            _remove_node(p.next)
            p = b
            start = b
        p = p.next
        if p is start:
            break
    return _filter_points(p)


def _split_earcut(start, triangles, min_x, min_y, inv_size):
    a = start
    while True:
        b = a.next.next
        while b is not a.prev:
            if a.i != b.i and _is_valid_diagonal(a, b):
                c = _split_polygon(a, b)
                a = _filter_points(a, a.next)
                c = _filter_points(c, c.next)
                _earcut_linked(a, triangles, min_x, min_y, inv_size, 0)
                _earcut_linked(c, triangles, min_x, min_y, inv_size, 0)
                return
            b = b.next
        a = a.next
        if a is start:
            break


def _earcut_linked(ear, triangles, min_x, min_y, inv_size, pass_):
    if ear is None:
        return
    if pass_ == 0 and inv_size != 0:
        _index_curve(ear, min_x, min_y, inv_size)

    stop = ear
    while ear.prev is not ear.next:
        prev = ear.prev
        nxt = ear.next
        if (inv_size != 0 and _is_ear_hashed(ear, min_x, min_y, inv_size)) or \
                (inv_size == 0 and _is_ear(ear)):
            triangles.append((prev.i, ear.i, nxt.i))
            _remove_node(ear)
            ear = nxt.next
            stop = nxt.next
            continue
        ear = nxt
        if ear is stop:
            if pass_ == 0:
                _earcut_linked(_filter_points(ear), triangles,
                               min_x, min_y, inv_size, 1)
            elif pass_ == 1:
                ear = _cure_local_intersections(
                    _filter_points(ear), triangles)
                _earcut_linked(ear, triangles, min_x, min_y, inv_size, 2)
            elif pass_ == 2:
                _split_earcut(ear, triangles, min_x, min_y, inv_size)
            break


def _fast_clip_seq(seq):
    """Queue-scheduled, grid-accelerated ear clipping.

    ``seq`` is a list of ``(global_index, x, y)`` forming a counter-clockwise
    simple (or weakly simple, after hole bridging) ring.  Returns index triples
    in the same index space, or ``None`` when the schedule stalls so the caller
    can fall back to the reference earcut implementation.

    The ear test uses the standard earcut criterion: a convex vertex is an ear
    when no *reflex* vertex lies inside its triangle.  Candidate reflex vertices
    are looked up through a coarse spatial hash, which keeps the work near
    linear on large outlines.
    """
    n = len(seq)
    if n < 3:
        return []
    ids = [p[0] for p in seq]
    xs = [float(p[1]) for p in seq]
    ys = [float(p[2]) for p in seq]

    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    span = max(maxx - minx, maxy - miny)
    if span <= 0.0:
        return None

    grid_r = 64
    inv_cell = grid_r / span

    prv = [(i - 1) % n for i in range(n)]
    nxt = [(i + 1) % n for i in range(n)]
    alive = bytearray(b"\x01") * n
    alive_count = n
    head = 0

    reflex = bytearray(n)
    reflex_set = set()
    for i in range(n):
        if _cross_seq(xs, ys, prv[i], i, nxt[i]) < 0.0:
            reflex[i] = 1
            reflex_set.add(i)

    buckets = {}
    for i in range(n):
        gx = int(math.floor((xs[i] - minx) * inv_cell))
        gy = int(math.floor((ys[i] - miny) * inv_cell))
        buckets.setdefault((gx, gy), []).append(i)

    seed = [i for i in range(n) if not reflex[i]]
    urgent = []
    in_queue = bytearray(n)
    for i in seed:
        in_queue[i] = 1

    triangles = []
    pops = 0
    budget = 60 * n + 100000

    def refresh_reflex(i):
        if not alive[i]:
            return
        is_reflex = 1 if _cross_seq(xs, ys, prv[i], i, nxt[i]) < 0.0 else 0
        if is_reflex and not reflex[i]:
            reflex[i] = 1
            reflex_set.add(i)
        elif not is_reflex and reflex[i]:
            reflex[i] = 0
            reflex_set.discard(i)

    def ear_blocked(i):
        if not reflex_set:
            return False
        a = prv[i]
        c = nxt[i]
        ax, ay = xs[a], ys[a]
        bx, by = xs[i], ys[i]
        cx, cy = xs[c], ys[c]
        x0 = min(ax, bx, cx)
        x1 = max(ax, bx, cx)
        y0 = min(ay, by, cy)
        y1 = max(ay, by, cy)

        if len(reflex_set) <= 200:
            for p in reflex_set:
                if p == a or p == c:
                    continue
                px, py = xs[p], ys[p]
                if px < x0 or px > x1 or py < y0 or py > y1:
                    continue
                if (bx - ax) * (py - ay) - (by - ay) * (px - ax) < -1e-9:
                    continue
                if (cx - bx) * (py - by) - (cy - by) * (px - bx) < -1e-9:
                    continue
                if (ax - cx) * (py - cy) - (ay - cy) * (px - cx) >= -1e-9:
                    return True
            return False

        gx0 = int(math.floor((x0 - minx) * inv_cell))
        gx1 = int(math.floor((x1 - minx) * inv_cell))
        gy0 = int(math.floor((y0 - miny) * inv_cell))
        gy1 = int(math.floor((y1 - miny) * inv_cell))
        ncells = max(0, gx1 - gx0 + 1) * max(0, gy1 - gy0 + 1)
        if ncells <= 256:
            for gx in range(gx0, gx1 + 1):
                for gy in range(gy0, gy1 + 1):
                    for p in buckets.get((gx, gy), ()):  # noqa: B007
                        if not reflex[p] or p == a or p == c:
                            continue
                        px, py = xs[p], ys[p]
                        if px < x0 or px > x1 or py < y0 or py > y1:
                            continue
                        if (bx - ax) * (py - ay) - (by - ay) * (px - ax) < -1e-9:
                            continue
                        if (cx - bx) * (py - by) - (cy - by) * (px - bx) < -1e-9:
                            continue
                        if (ax - cx) * (py - cy) - (ay - cy) * (px - cx) >= -1e-9:
                            return True
            return False

        for p in reflex_set:
            if p == a or p == c:
                continue
            px, py = xs[p], ys[p]
            if px < x0 or px > x1 or py < y0 or py > y1:
                continue
            if (bx - ax) * (py - ay) - (by - ay) * (px - ax) < -1e-9:
                continue
            if (cx - bx) * (py - by) - (cy - by) * (px - bx) < -1e-9:
                continue
            if (ax - cx) * (py - cy) - (ay - cy) * (px - cx) >= -1e-9:
                return True
        return False

    def push(j):
        if alive[j] and not in_queue[j]:
            in_queue[j] = 1
            urgent.append(j)

    while alive_count > 3:
        if urgent:
            i = urgent.pop()
        elif seed:
            i = seed.pop()
            if in_queue[i]:
                in_queue[i] = 0
        else:
            p = head
            any_convex = False
            while True:
                if alive[p] and not reflex[p]:
                    in_queue[p] = 0
                    urgent.append(p)
                    any_convex = True
                p = nxt[p]
                if p == head:
                    break
            if not any_convex:
                return None
            i = urgent.pop()

        pops += 1
        if pops > budget:
            return None
        in_queue[i] = 0
        if not alive[i]:
            continue

        a = prv[i]
        c = nxt[i]
        turn = _cross_seq(xs, ys, a, i, c)

        if turn == 0.0:
            bx, by = xs[i], ys[i]
            if (min(xs[a], xs[c]) <= bx <= max(xs[a], xs[c]) and
                    min(ys[a], ys[c]) <= by <= max(ys[a], ys[c])):
                nxt[a] = c
                prv[c] = a
                alive[i] = 0
                alive_count -= 1
                if reflex[i]:
                    reflex[i] = 0
                    reflex_set.discard(i)
                if i == head:
                    head = c
                refresh_reflex(a)
                refresh_reflex(c)
                push(a)
                push(c)
            continue

        if turn < 0.0:
            if not reflex[i]:
                reflex[i] = 1
                reflex_set.add(i)
            continue

        if ear_blocked(i):
            continue

        triangles.append((ids[a], ids[i], ids[c]))
        nxt[a] = c
        prv[c] = a
        alive[i] = 0
        alive_count -= 1
        if reflex[i]:
            reflex[i] = 0
            reflex_set.discard(i)
        if i == head:
            head = c
        refresh_reflex(a)
        refresh_reflex(c)
        push(c)
        push(a)

    if alive_count == 3:
        a = head
        b = nxt[a]
        c = nxt[b]
        if nxt[c] != a:
            return None
        if _cross_seq(xs, ys, a, b, c) > 0.0:
            triangles.append((ids[a], ids[b], ids[c]))
        else:
            triangles.append((ids[a], ids[c], ids[b]))
    return triangles


def _cross_seq(xs, ys, a, b, c):
    return ((xs[b] - xs[a]) * (ys[c] - ys[a])
            - (ys[b] - ys[a]) * (xs[c] - xs[a]))


def _fast_group(outer):
    """Fast triangulation of a bridged node cycle, with an area gate."""
    seq = []
    p = outer
    while True:
        seq.append((p.i, p.x, p.y))
        p = p.next
        if p is outer:
            break
    m = len(seq)
    if m < 3:
        return []
    if m <= 128:
        return None  # small inputs stay on the reference path

    area2 = 0.0
    for k in range(m):
        j = k - 1
        area2 += seq[j][1] * seq[k][2] - seq[k][1] * seq[j][2]
    if area2 < 0.0:
        seq = seq[::-1]
        area2 = -area2

    tris = _fast_clip_seq(seq)
    if not tris:
        return tris if tris == [] else None

    coords = {}
    for idx, x, y in seq:
        coords[idx] = (x, y)
    tri_area2 = 0.0
    for a, b, c in tris:
        if a == b or b == c or a == c:
            return None
        ax, ay = coords[a]
        bx, by = coords[b]
        cx, cy = coords[c]
        cr = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        if cr <= 0.0:
            return None
        tri_area2 += cr
    if abs(tri_area2 - area2) > 1e-7 * max(1.0, abs(area2)):
        return None
    return tris


def _earcut_group(outer_pts, hole_rings):
    """Triangulate one outer ring plus its holes."""
    outer = _linked_list(outer_pts, True)
    if outer is None or outer.next is outer.prev:
        return []
    if hole_rings:
        outer = _eliminate_holes(hole_rings, outer)

    fast = _fast_group(outer)
    if fast is not None:
        return fast

    triangles = []
    min_x = min_y = 0.0
    inv_size = 0.0
    vertex_count = sum(len(r) for r in hole_rings) + len(outer_pts)
    if vertex_count > 80:
        bxs = [p[1] for p in outer_pts]
        bys = [p[2] for p in outer_pts]
        min_x = min(bxs)
        min_y = min(bys)
        extent = max(max(bxs) - min_x, max(bys) - min_y)
        inv_size = 32767.0 / extent if extent > 0.0 else 0.0

    _earcut_linked(outer, triangles, min_x, min_y, inv_size, 0)
    return triangles


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class TextShape:
    """Triangulate glyph outlines made of nested contours."""

    def triangulate(self, contours):
        """Triangulate ``contours``.

        Returns a list of ``(a, b, c)`` vertex-index triples into the
        flattened contour point list.  Every triangle is counter-clockwise.
        """
        rings = _parse_contours(contours)

        groups = {}
        order = []
        for idx, ring in enumerate(rings):
            if ring.depth % 2 == 0:
                groups[idx] = []
                order.append(idx)
        for idx, ring in enumerate(rings):
            if ring.depth % 2 == 1:
                groups[ring.parent].append(idx)

        triangles = []
        for outer_idx in order:
            outer_pts = rings[outer_idx].pts
            hole_rings = [rings[h].pts for h in groups[outer_idx]]
            triangles.extend(_earcut_group(outer_pts, hole_rings))

        # Coordinate lookup in normalised space, keyed by global index.
        coords = {}
        for ring in rings:
            for (i, nx, ny) in ring.pts:
                coords[i] = (nx, ny)

        # Enforce a single (counter-clockwise) orientation and drop exact
        # zero-area slivers produced by degenerate ears.
        normalised = []
        for (a, b, c) in triangles:
            if a == b or b == c or a == c:
                continue
            ax, ay = coords[a]
            bx, by = coords[b]
            cx, cy = coords[c]
            cross2 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
            if cross2 > 0.0:
                normalised.append((a, b, c))
            elif cross2 < 0.0:
                normalised.append((a, c, b))
            # cross2 == 0.0: skip zero-area triangle
        return normalised

    def area(self, contours):
        """Area of the filled region under the nesting (non-zero) rule."""
        rings = _parse_contours(contours)
        total = 0.0
        gross = 0.0
        for ring in rings:
            magnitude = abs(ring.area_raw)
            gross += magnitude
            if ring.depth % 2 == 0:
                total += magnitude
            else:
                total -= magnitude
        # Remove a tiny negative residue from canceling shoelace sums.
        if total < 0.0 and abs(total) <= 1e-9 * max(1.0, gross):
            total = 0.0
        return float(total)


# A shared, stateless instance makes the module-level helpers safe to call
# from any number of threads.
_default_shape = TextShape()


def triangulate(contours):
    """Module-level convenience wrapper around :class:`TextShape`."""
    return _default_shape.triangulate(contours)


def area(contours):
    """Module-level convenience wrapper around :class:`TextShape.area`."""
    return _default_shape.area(contours)
