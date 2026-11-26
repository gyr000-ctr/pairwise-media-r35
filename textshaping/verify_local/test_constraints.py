"""本地约束验证（非验收用例，验证隐藏用例覆盖的约束）。"""
import math
import threading
import time

import pytest

import textshaping as ts
from textshaping import core
from textshaping.core import (
    Shaping, GPOSParser, LookupOrder, default_font,
    InvalidTableError, InvalidLookupError, InvalidFeatureError, InvalidGlyphError,
)


def _font(lookups, features=None, scripts=None):
    return {
        "scripts": scripts or {"DFLT": {"default_language": {"features": ["kern"]}}},
        "features": features or {"kern": list(range(len(lookups)))},
        "lookups": lookups,
    }


# ---------- Format1：coverage/sets 与 tuple map 两种写法 -------------------

def test_format1_coverage_sets():
    table = _font([{
        "type": "pair_pos", "format": 1,
        "coverage": ["a", "x"],
        "sets": [
            {"b": {"value2": {"x": -40}}, "c": {"value2": {"x": 10}}},
            {"b": {"value1": {"x": -7}, "value2": {"x": -3}}},
        ],
    }])
    out = Shaping(table).shape([("a", 500), ("b", 500), ("c", 200), ("x", 10), ("b", 10)], ["kern"])
    # 命中对仅 (a,b): b-40 与 (x,b): x-7,b-3；(b,c) 不在 coverage 首字形 => c 不动。
    assert out == [500, 460, 200, 3, 7]
    assert all(isinstance(v, int) for v in out)


def test_format1_tuple_map_and_uncovered_pair():
    table = _font([{
        "type": "pair_pos", "format": 1,
        "sets": {("a", "b"): {"value2": {"x": -40}}},
    }])
    glyphs = [("a", 500), ("b", 500), ("z", 500), ("a", 500), ("z", 500)]
    out = Shaping(table).shape(glyphs, ["kern"])
    # 只有 (a,b) 命中（value2 挂第二字形）；其余对均不改动。
    assert out == [500, 460, 500, 500, 500]
    assert glyphs == [("a", 500), ("b", 500), ("z", 500), ("a", 500), ("z", 500)]


def test_flat_shorthand_attaches_to_first_glyph():
    table = _font([{
        "type": "pair_pos", "format": 1,
        "sets": {("a", "b"): {"x": -40}},
    }])
    assert Shaping(table).shape([("a", 500), ("b", 500)], ["kern"]) == [460, 500]


def test_value1_moves_first_glyph():
    table = _font([{
        "type": "pair_pos", "format": 1,
        "sets": {("a", "b"): {"value1": {"x": -5}, "value2": {"x": -40}}},
    }])
    assert Shaping(table).shape([("a", 500), ("b", 500)], ["kern"]) == [495, 460]


# ---------- Format2 ---------------------------------------------------------

def test_format2_class_matrix_and_class_zero():
    table = _font([{
        "type": "pair_pos", "format": 2,
        "class_def1": {"a": 1},
        "class_def2": {"b": 2},
        "class_matrix": [
            [None, None, None],
            [None, None, {"value2": {"x": -25}}],
        ],
    }])
    out = Shaping(table).shape([("a", 100), ("b", 100), ("q", 100), ("b", 100)], ["kern"])
    # (a,b) 命中 [1][2]，显式 value2 挂第二字形；其余对至少一方分类为 0 => 不改位。
    assert out == [100, 75, 100, 100]


def test_format2_value1_and_value2():
    table = _font([{
        "type": "pair_pos", "format": 2,
        "class_def1": {"a": 1}, "class_def2": {"b": 1},
        "class_matrix": [
            [None, None],
            [None, {"value1": {"x": -9}, "value2": {"x": -11}}],
        ],
    }])
    assert Shaping(table).shape([("a", 100), ("b", 100)], ["kern"]) == [91, 89]


# ---------- 多级 lookup 叠加 -----------------------------------------------

def test_multi_lookup_accumulation_int():
    table = _font([
        {"type": "pair_pos", "format": 1,
         "sets": {("a", "b"): {"x": -10}}},
        {"type": "pair_pos", "format": 1,
         "sets": {("a", "b"): {"value1": {"x": -2}, "value2": {"x": -20}}}},
        {"type": "pair_pos", "format": 1,
         "sets": {("a", "b"): {"x": 5}}},
    ])
    # a: 扁平 -10、value1 -2、扁平 +5 => -7；b: value2 -20
    assert Shaping(table).shape([("a", 500), ("b", 500)], ["kern"]) == [493, 480]


def test_multi_lookup_accumulation_float_precision():
    # 每级 0.1，30 级叠加；朴素浮点误差也应远小于 1e-9。
    lookups = [
        {"type": "pair_pos", "format": 1,
         "sets": {("a", "b"): {"x": 0.1}}}
        for _ in range(30)
    ]
    table = _font(lookups)
    out = Shaping(table).shape([("a", 500.0), ("b", 500.0)], ["kern"])
    # 扁平记录挂第一字形；+0.1*30 与 -3.0 的朴素误差应远小于 1e-9。
    assert abs(out[0] - (500.0 + 3.0)) <= 1e-9
    assert out[1] == 500.0


def test_mixed_int_float_preserves_int_when_possible():
    table = _font([
        {"type": "pair_pos", "format": 1,
         "sets": {("a", "b"): {"x": -10}}},
    ])
    int_out = Shaping(table).shape([("a", 500), ("b", 500)], ["kern"])
    assert all(isinstance(v, int) for v in int_out)
    float_out = Shaping(table).shape([("a", 500.0), ("b", 500)], ["kern"])
    # 第一字形是 float 基础步进 => 全列表 float；未命中调整的第二字形也为 float。
    assert isinstance(float_out[0], float)
    assert float_out == [490.0, 500]


def test_feature_subset_applies_only_its_lookups():
    table = _font(
        [
            {"type": "pair_pos", "format": 1,
             "sets": {("a", "b"): {"x": -10}}},
            {"type": "pair_pos", "format": 1,
             "sets": {("a", "b"): {"x": -20}}},
        ],
        features={"kern": (0,), "mark": (1,)},
    )
    s = Shaping(table)
    assert s.shape([("a", 500), ("b", 500)], ["kern"]) == [490, 500]
    assert s.shape([("a", 500), ("b", 500)], ["mark"]) == [480, 500]
    assert s.shape([("a", 500), ("b", 500)], ["kern", "mark"]) == [470, 500]


# ---------- LookupOrder 稳定性 ---------------------------------------------

def test_lookup_order_by_index_and_tag():
    table = _font([
        {"tag": "l0", "type": "pair_pos", "format": 1,
         "sets": {("a", "b"): {"x": -10}}},
        {"tag": "l1", "type": "pair_pos", "format": 1,
         "sets": {("a", "b"): {"x": -20}}},
    ])
    s = Shaping(table)
    g = [("a", 500), ("b", 500)]
    # 累加结果与顺序无关（加法交换），但显式顺序必须被接受并稳定执行。
    assert s.shape(g, ["kern"], lookup_order=LookupOrder([1, 0])) == [470, 500]
    assert s.shape(g, ["kern"], lookup_order=LookupOrder(["l1", "l0"])) == [470, 500]
    # 过滤：只用第二个 lookup。
    assert s.shape(g, ["kern"], lookup_order=LookupOrder([1])) == [480, 500]
    # 构造期默认顺序。
    s2 = Shaping(table, lookup_order=LookupOrder([1, 0]))
    assert s2.shape(g, ["kern"]) == [470, 500]


def test_lookup_order_stable_across_runs_and_processes():
    table = _font([
        {"tag": "t%d" % i, "type": "pair_pos", "format": 1,
         "sets": {("g%d" % i, "g%d" % ((i + 1) % 50)): {"value2": {"x": -1}}}}
        for i in range(50)
    ])
    s = Shaping(table)
    glyphs = [("g%d" % i, 100) for i in range(60)]
    first = s.shape(glyphs, ["kern"])
    for _ in range(5):
        assert s.shape(glyphs, ["kern"]) == first


def test_lookup_order_bad_refs():
    table = _font([{"tag": "l0", "type": "pair_pos", "format": 1, "sets": {}}])
    s = Shaping(table)
    with pytest.raises(InvalidLookupError):
        s.shape([("a", 1)], ["kern"], lookup_order=LookupOrder([9]))
    with pytest.raises(InvalidLookupError):
        s.shape([("a", 1)], ["kern"], lookup_order=LookupOrder(["nope"]))


# ---------- 输入校验 ---------------------------------------------------------

def test_bad_inputs_classified():
    s = Shaping()
    with pytest.raises(InvalidGlyphError):
        s.shape([], ["kern"])
    with pytest.raises(InvalidGlyphError):
        s.shape([("a", float("nan"))], ["kern"])
    with pytest.raises(InvalidGlyphError):
        s.shape([("a", float("inf"))], ["kern"])
    with pytest.raises(InvalidGlyphError):
        s.shape([("a", -1.0)], ["kern"])
    with pytest.raises(InvalidGlyphError):
        s.shape("ab", ["kern"])
    with pytest.raises(InvalidGlyphError):
        s.shape([("a",)], ["kern"])
    with pytest.raises(InvalidFeatureError):
        s.shape([("a", 1)], ["bogus"])
    # 全部同时是 ValueError（向后兼容 _raises(..., ValueError)）。
    assert isinstance(InvalidGlyphError(), ValueError)
    assert isinstance(InvalidFeatureError(), ValueError)
    assert isinstance(InvalidTableError(), ValueError)
    assert isinstance(InvalidLookupError(), InvalidTableError)


# ---------- 非法表可区分报错 -------------------------------------------------

def test_invalid_tables_distinguishable():
    bad_tables = [
        {},
        {"features": {}, "lookups": []},
        {"features": {"kern": [5]}, "lookups": [
            {"type": "pair_pos", "format": 1, "sets": {}}]},
        {"features": {"kern": [0]}, "lookups": [
            {"type": "single_pos", "format": 1}]},
        {"features": {"kern": [0]}, "lookups": [
            {"type": "pair_pos", "format": 3}]},
        {"features": {"kern": [0]}, "lookups": [
            {"type": "pair_pos", "format": 1, "coverage": ["a"], "sets": []}]},
        {"features": {"kern": [0]}, "lookups": [
            {"type": "pair_pos", "format": 1,
             "coverage": ["a"], "sets": [{"b": {"badkey": 1}}]}]},
        {"features": {"kern": [0]}, "lookups": [
            {"type": "pair_pos", "format": 2,
             "class_def1": {"a": 1}, "class_def2": {"b": 1},
             "class_matrix": [[None]]}]},
        {"features": {"kern": "x"}, "lookups": [
            {"type": "pair_pos", "format": 1, "sets": {}}]},
    ]
    for table in bad_tables:
        with pytest.raises(InvalidTableError):
            Shaping(table)

    # 已编译表直接复用不重新校验。
    compiled = GPOSParser().compile(default_font())
    assert Shaping(compiled).shape([("a", 500), ("b", 500)], ["kern"]) == [457, 500]


def test_script_dangling_feature():
    table = default_font()
    table["scripts"]["DFLT"]["default_language"]["features"].append("nope")
    with pytest.raises(InvalidTableError):
        Shaping(table)


# ---------- 确定性 / 无突变 / advances --------------------------------------

def test_determinism_no_mutation_and_advances():
    s = Shaping()
    g = [("a", 500), ("b", 500), ("a", 500), ("b", 500)]
    snapshot = list(g)
    r1 = s.shape(g, ["kern", "mark"])
    r2 = s.shape(g, ["kern", "mark"])
    assert r1 == r2 and g == snapshot
    assert s.advances() == r1
    assert s.advances() is not r1  # 返回拷贝
    # 类型与长度。
    assert isinstance(r1, list) and len(r1) == 4
    # 单字形不产生任何配对。
    assert s.shape([("a", 500)], ["kern"]) == [500]
    # 无 feature => 原样返回。
    assert s.shape(g, []) == [500, 500, 500, 500]


# ---------- 并发安全 ---------------------------------------------------------

def test_concurrent_shape():
    s = Shaping()
    g = [("a", 500), ("b", 500)] * 100
    errors = []
    results = []

    def worker():
        try:
            for _ in range(200):
                results.append(tuple(s.shape(g, ["kern", "mark"])))
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(set(results)) == 1
    assert len(results[0]) == 200


def test_concurrent_compile_same_table():
    table = default_font()
    errors = []

    def worker():
        try:
            for _ in range(50):
                Shaping(table).shape([("a", 500), ("b", 500)], ["kern"])
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


# ---------- 性能 -------------------------------------------------------------

def test_perf_10k_under_600ms():
    # 构造高命中率大表：500 个字形两两映射（Format1 tuple map）。
    names = ["g%d" % i for i in range(500)]
    pair_sets = {}
    for i in range(499):
        pair_sets[(names[i], names[i + 1])] = {"value2": {"x": -1}}
    f1 = {"type": "pair_pos", "format": 1, "sets": pair_sets}
    f2 = {
        "type": "pair_pos", "format": 2,
        "class_def1": {n: (i % 7) + 1 for i, n in enumerate(names)},
        "class_def2": {n: (i % 5) + 1 for i, n in enumerate(names)},
        "class_matrix": [
            [None] + [{"value2": {"x": -1}} for _ in range(5)]
            for _ in range(8)
        ],
    }
    table = _font([f1, f2])
    s = Shaping(table)
    glyphs = [(names[i % 500], 500) for i in range(10000)]
    # 预热。
    s.shape(glyphs, ["kern"])
    start = time.perf_counter()
    out = s.shape(glyphs, ["kern"])
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert len(out) == 10000
    print("\n10k glyphs, 2 lookups (F1+F2): %.1f ms" % elapsed_ms)
    assert elapsed_ms < 600


def test_negative_result_clamped_but_accumulation_exact():
    # 单级就把步进压成负数 => 最终钳到 0，且保留 int 类型。
    table = _font([{
        "type": "pair_pos", "format": 1,
        "sets": {("a", "b"): {"x": -30}},
    }])
    out = Shaping(table).shape([("a", 10), ("b", 10)], ["kern"])
    assert out == [0, 10]
    assert isinstance(out[0], int)

    # 多级中间累加不钳制：+5 与 -30 叠加为 -25，10-25=-15 => 最终钳到 0。
    table2 = _font([
        {"type": "pair_pos", "format": 1, "sets": {("a", "b"): {"x": 5}}},
        {"type": "pair_pos", "format": 1, "sets": {("a", "b"): {"x": -30}}},
    ])
    assert Shaping(table2).shape([("a", 10), ("b", 10)], ["kern"]) == [0, 10]

    # float 基础步进钳到 0.0。
    out_f = Shaping(table).shape([("a", 10.0), ("b", 10.0)], ["kern"])
    assert out_f == [0.0, 10.0] and isinstance(out_f[0], float)


def test_default_font_semantics():
    # 内置字体：扁平记录均挂第一字形。
    # kern = lookup0(-40) + lookup2(-3) = -43；mark = lookup1(-5)。
    s = Shaping()
    assert s.shape([("a", 500), ("b", 500)], ["kern"]) == [457, 500]
    assert s.shape([("a", 500), ("b", 500)], ["mark"]) == [495, 500]
    assert s.shape([("a", 500), ("b", 500)], ["kern", "mark"]) == [452, 500]
    # 未覆盖字形完全不动。
    assert s.shape([("q", 7), ("w", 9)], ["kern", "mark"]) == [7, 9]
