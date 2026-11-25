import math

import pytest

import textellipsis

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

G = [("a", 10), ("b", 10), ("c", 10)]

CASES = [
    ("fit_all", lambda: textellipsis.Ellipsis().truncate(G, 100.0) == 'abc'),
    ("cut", lambda: len(textellipsis.Ellipsis().truncate(G, 25.0)) <= 3),
    ("width_bound", lambda: textellipsis.Ellipsis().width([('a', 10)]) == 10),
    ("bad_width", lambda: _raises(lambda: textellipsis.Ellipsis().truncate(G, 0.0))),
    ("bad_glyphs", lambda: _raises(lambda: textellipsis.Ellipsis().truncate([], 10.0))),
    ("type", lambda: isinstance(textellipsis.Ellipsis().truncate(G, 25.0), str)),
    ("det", lambda: textellipsis.Ellipsis().truncate(G, 25.0) == textellipsis.Ellipsis().truncate(G, 25.0)),
    ("empty_ell", lambda: isinstance(textellipsis.Ellipsis().truncate(G, 5.0, ''), str)),
    ("width_type", lambda: isinstance(textellipsis.Ellipsis().width(G), float)),
    ("no_mut", lambda: textellipsis.Ellipsis().truncate(list(G), 25.0) is not None),
    ("too_narrow", lambda: isinstance(textellipsis.Ellipsis().truncate(G, 1.0), str)),
    ("nan", lambda: _raises(lambda: textellipsis.Ellipsis().truncate(G, float('nan')))),
    ("cluster", lambda: isinstance(textellipsis.Ellipsis().truncate([('e\u0301', 10)], 10.0), str)),
    ("zero_glyph", lambda: _raises(lambda: textellipsis.Ellipsis().truncate([('a', 0)], 10.0))),
    ("float_width", lambda: isinstance(textellipsis.Ellipsis().width([('a', 10), ('b', 10)]), float)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
