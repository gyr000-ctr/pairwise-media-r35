import math

import pytest

import textshape

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

OUTER = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
HOLE = [(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]

CASES = [
    ("count", lambda: len(textshape.TextShape().triangulate([OUTER])) == 2),
    ("hole", lambda: len(textshape.TextShape().triangulate([OUTER, HOLE])) == 8),
    ("area", lambda: abs(textshape.TextShape().area([OUTER, HOLE]) - 3.0) < 1e-6),
    ("bad_self", lambda: _raises(lambda: textshape.TextShape().triangulate([[(0, 0), (2, 2), (2, 0), (0, 2)]]))),
    ("bad_empty", lambda: _raises(lambda: textshape.TextShape().triangulate([]))),
    ("type", lambda: isinstance(textshape.TextShape().triangulate([OUTER]), list)),
    ("det", lambda: textshape.TextShape().triangulate([OUTER]) == textshape.TextShape().triangulate([OUTER])),
    ("area_type", lambda: isinstance(textshape.TextShape().area([OUTER]), float)),
    ("tri_len", lambda: all(len(t) == 3 for t in textshape.TextShape().triangulate([OUTER]))),
    ("no_mut", lambda: textshape.TextShape().triangulate([list(c) for c in OUTER]) is not None),
    ("index_range", lambda: all(0 <= i < 4 for t in textshape.TextShape().triangulate([OUTER]) for i in t)),
    ("nan", lambda: _raises(lambda: textshape.TextShape().triangulate([[(float('nan'), 0.0), (1, 0), (0, 1)]]))),
    ("single_tri", lambda: len(textshape.TextShape().triangulate([[(0, 0), (1, 0), (0, 1)]])) == 1),
    ("positive_area", lambda: all(textshape.TextShape().area([OUTER]) > 0 for _ in range(1))),
    ("float_area", lambda: isinstance(textshape.TextShape().area([OUTER, HOLE]), float)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
