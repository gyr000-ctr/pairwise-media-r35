import math

import pytest

import textmetrics

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _m():
    return textmetrics.Metrics(0.8, 0.2, 0.1)

CASES = [
    ("line_height", lambda: abs(_m().line_height(10.0) - 11.0) < 1e-12),
    ("layout_len", lambda: len(_m().layout(['ab', 'cd'], 10.0)) == 2),
    ("baseline", lambda: abs(_m().layout(['ab'], 10.0)[0][1] - 8.0) < 1e-9),
    ("bad_size", lambda: _raises(lambda: _m().line_height(0.0))),
    ("bad_metrics", lambda: _raises(lambda: textmetrics.Metrics(0.0, 0.0, 0.0))),
    ("empty", lambda: len(_m().layout([], 10.0)) == 0),
    ("type", lambda: isinstance(_m().layout(['a'], 10.0), list)),
    ("det", lambda: _m().layout(['a'], 10.0) == _m().layout(['a'], 10.0)),
    ("line_height_type", lambda: isinstance(_m().line_height(10.0), float)),
    ("no_mut", lambda: _m().layout(['a'], 10.0) is not None),
    ("single_line", lambda: len(_m().layout(['a'], 12.0)) == 1),
    ("nan", lambda: _raises(lambda: _m().layout(['a'], float('nan')))),
    ("three_lines", lambda: len(_m().layout(['a', 'b', 'c'], 10.0)) == 3),
    ("accumulate", lambda: abs(_m().layout(['a', 'b'], 10.0)[1][1] - 19.0) < 1e-9),
    ("float_y", lambda: isinstance(_m().layout(['a'], 10.0)[0][1], float)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
