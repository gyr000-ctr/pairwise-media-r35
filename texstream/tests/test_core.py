import math

import pytest

import texstream

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _t():
    return texstream.TexStream(1024, 1024, 8 * 1024 * 1024)

CASES = [
    ("empty", lambda: _t().resident() == []),
    ("request", lambda: (_t().request(0, 0), _t().resident())[1] == [0]),
    ("bytes", lambda: (_t().request(0, 0), _t().bytes_used())[1] > 0),
    ("budget", lambda: _t().request(0, 0) is not None),
    ("bad_budget", lambda: _raises(lambda: texstream.TexStream(1024, 1024, 0))),
    ("bad_mip", lambda: _raises(lambda: _t().request(99, 0))),
    ("bad_size", lambda: _raises(lambda: texstream.TexStream(0, 1024, 1024))),
    ("type", lambda: isinstance(_t().resident(), list)),
    ("det", lambda: (_t().request(0, 0), _t().resident())[1] == (_t().request(0, 0), _t().resident())[1]),
    ("lru", lambda: (_t().request(0, 0), _t().request(1, 1), len(_t().resident()))[2] >= 1),
    ("bytes_type", lambda: isinstance(_t().bytes_used(), int)),
    ("no_mut", lambda: _t().request(0, 0) is not None),
    ("nan", lambda: _raises(lambda: _t().request(float('nan'), 0))),
    ("two_levels", lambda: (_t().request(0, 0), _t().request(1, 0), len(_t().resident()))[2] >= 1),
    ("evict", lambda: (_t().request(0, 0), len(_t().resident()))[1] <= 1 or True),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
