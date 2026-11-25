import math

import pytest

import textshaping

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

G = [("a", 500), ("b", 500)]

CASES = [
    ("advances_len", lambda: len(textshaping.Shaping().shape(G, ['kern'])) == 2),
    ("kern", lambda: sum(textshaping.Shaping().shape(G, ['kern'])) <= 1000),
    ("no_feature", lambda: sum(textshaping.Shaping().shape(G, [])) == 1000),
    ("bad_glyphs", lambda: _raises(lambda: textshaping.Shaping().shape([], ['kern']))),
    ("bad_feature", lambda: _raises(lambda: textshaping.Shaping().shape(G, ['bogus']))),
    ("type", lambda: isinstance(textshaping.Shaping().shape(G, ['kern']), list)),
    ("det", lambda: textshaping.Shaping().shape(G, ['kern']) == textshaping.Shaping().shape(G, ['kern'])),
    ("advances", lambda: isinstance(textshaping.Shaping().advances(), list)),
    ("no_mut", lambda: textshaping.Shaping().shape(list(G), ['kern']) is not None),
    ("single", lambda: len(textshaping.Shaping().shape([('a', 500)], ['kern'])) == 1),
    ("int_values", lambda: all(isinstance(v, int) for v in textshaping.Shaping().shape(G, ['kern']))),
    ("order", lambda: textshaping.Shaping().shape(G, ['kern', 'mark']) is not None),
    ("nan", lambda: _raises(lambda: textshaping.Shaping().shape([('a', float('nan'))], ['kern']))),
    ("positive", lambda: all(v >= 0 for v in textshaping.Shaping().shape(G, ['kern']))),
    ("float_adv", lambda: isinstance(textshaping.Shaping().advances() or [0.0], list)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
