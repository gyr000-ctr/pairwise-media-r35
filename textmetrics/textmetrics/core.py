"""textmetrics 核心 API：行高、基线、行盒与上下标度量。

约束实现：
- 行高 = (ascent + descent + lineGap) * size；
- 多行基线按行盒累积：baseline_i = ascent*size + i*line_height；
- 上下标缩放与偏移可配，且不改变行盒；溢出行盒时通过结果中的
  ``overflow`` / ``overflow_amount`` 报告；
- 行盒矩形与基线一致（``line_box`` / ``measure``）；
- 空文本返回单行度量（``measure('')`` 仍为一行）；
- 非法输入分类报错：类型错误 -> ``TypeError``；
  非有限（NaN/inf）或非正数值 -> ``ValueError``；
- ``Metrics`` 构造后不可变，布局计算无共享状态，可安全并发使用。
"""

from __future__ import annotations

import math
from collections import namedtuple

__all__ = [
    "Metrics",
    "LineBox",
    "ScriptMetrics",
    "TextLayout",
]


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_finite(name, value):
    """校验为有限实数：类型错 -> TypeError；非有限 -> ValueError。"""
    if not _is_number(value):
        raise TypeError(
            "%s must be a real number, got %s" % (name, type(value).__name__)
        )
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("%s must be finite, got %r" % (name, value))
    return value


def _check_non_negative(name, value):
    value = _check_finite(name, value)
    if value < 0.0:
        raise ValueError("%s must be non-negative, got %r" % (name, value))
    return value


def _check_positive(name, value):
    value = _check_finite(name, value)
    if value <= 0.0:
        raise ValueError("%s must be positive, got %r" % (name, value))
    return value


def _check_size(size):
    return _check_positive("size", size)


class LineBox(namedtuple("LineBox", ("top", "bottom", "height"))):
    """行盒矩形（y 轴向下为正，坐标与基线同一原点）。

    ``top`` / ``bottom`` 为行盒上下边界的 y 坐标，``height`` 恒等于行高。
    """

    __slots__ = ()

    def __contains__(self, y):
        return self.top <= y <= self.bottom


class ScriptMetrics(
    namedtuple("ScriptMetrics", ("size", "offset", "overflow", "overflow_amount"))
):
    """上下标度量结果。

    - ``size``：缩放后的字号；
    - ``offset``：相对主基线的偏移量（向上为正，y 轴向下为正坐标系）；
    - ``overflow``：字形是否溢出（未缩放的）行盒；
    - ``overflow_amount``：溢出总量（上溢 + 下溢，>= 0）。

    上下标只缩放并平移字形，**不改变行盒**。
    """

    __slots__ = ()


class TextLayout(
    namedtuple("TextLayout", ("lines", "baselines", "boxes", "line_height", "size"))
):
    """多行度量结果。

    ``lines`` 为原文各行，``baselines`` 为对应基线 y 坐标，
    ``boxes`` 为与基线一致的行盒矩形。空文本仍返回单行度量。
    """

    __slots__ = ()

    @property
    def line_count(self):
        return len(self.lines)

    @property
    def total_height(self):
        return self.line_height * len(self.lines)


class Metrics(object):
    """字体垂直度量：ascent / descent / lineGap 均为字号的比例系数。"""

    __slots__ = (
        "_ascent",
        "_descent",
        "_line_gap",
        "_unit_height",
        "_superscript_scale",
        "_superscript_offset",
        "_subscript_scale",
        "_subscript_offset",
    )

    def __init__(
        self,
        ascent,
        descent,
        line_gap,
        superscript_scale=0.7,
        superscript_offset=0.2,
        subscript_scale=0.7,
        subscript_offset=0.15,
    ):
        """构造度量对象。

        ``ascent`` / ``descent`` / ``line_gap`` 必须为非负有限实数，
        且三者之和为正。上下标的缩放（相对字号的倍数，必须为正）与
        偏移（相对字号的比例，向上为正）可在此配置，也可在调用
        ``superscript`` / ``subscript`` 时按次覆盖。
        """
        self._ascent = _check_non_negative("ascent", ascent)
        self._descent = _check_non_negative("descent", descent)
        self._line_gap = _check_non_negative("line_gap", line_gap)
        unit = self._ascent + self._descent + self._line_gap
        if unit <= 0.0:
            raise ValueError("ascent + descent + line_gap must be positive")
        self._unit_height = unit
        self._superscript_scale = _check_positive(
            "superscript_scale", superscript_scale
        )
        self._superscript_offset = _check_finite(
            "superscript_offset", superscript_offset
        )
        self._subscript_scale = _check_positive("subscript_scale", subscript_scale)
        self._subscript_offset = _check_finite("subscript_offset", subscript_offset)

    # -- 只读属性（构造后不可变，保证并发安全） ----------------------------

    @property
    def ascent(self):
        return self._ascent

    @property
    def descent(self):
        return self._descent

    @property
    def line_gap(self):
        return self._line_gap

    @property
    def superscript_scale(self):
        return self._superscript_scale

    @property
    def superscript_offset(self):
        return self._superscript_offset

    @property
    def subscript_scale(self):
        return self._subscript_scale

    @property
    def subscript_offset(self):
        return self._subscript_offset

    def __repr__(self):
        return "Metrics(ascent=%r, descent=%r, line_gap=%r)" % (
            self._ascent,
            self._descent,
            self._line_gap,
        )

    # -- 行高 / 行盒 ------------------------------------------------------

    def line_height(self, size):
        """行高 = (ascent + descent + lineGap) * size。"""
        size = _check_size(size)
        return self._unit_height * size

    def line_box(self, size, baseline=0.0):
        """返回与给定基线一致的行盒矩形（y 轴向下为正）。

        行盒上边距基线 ascent*size，下边距基线 (descent+lineGap)*size，
        高度恒等于 ``line_height(size)``。
        """
        size = _check_size(size)
        baseline = _check_finite("baseline", baseline)
        top = baseline - self._ascent * size
        bottom = baseline + (self._descent + self._line_gap) * size
        return LineBox(top, bottom, bottom - top)

    # -- 布局 -------------------------------------------------------------

    def layout(self, lines, size):
        """多行布局：返回 ``[(line, baseline_y), ...]`` 列表。

        首行基线为 ``ascent*size``，之后每行按行高累积。
        空列表返回空列表；单个字符串按一行处理。
        """
        size = _check_size(size)
        if isinstance(lines, str):
            lines = (lines,)
        try:
            iterator = iter(lines)
        except TypeError:
            raise TypeError(
                "lines must be an iterable of strings, got %s"
                % type(lines).__name__
            )
        step = self._unit_height * size
        first = self._ascent * size
        out = []
        append = out.append
        i = 0
        for line in iterator:
            if not isinstance(line, str):
                raise TypeError(
                    "each line must be a str, got %s" % type(line).__name__
                )
            append((line, first + i * step))
            i += 1
        return out

    def measure(self, text, size):
        """度量一段文本（可含 ``\\n`` 换行）；空文本返回单行度量。"""
        size = _check_size(size)
        if not isinstance(text, str):
            raise TypeError("text must be a str, got %s" % type(text).__name__)
        lines = text.split("\n") if text else [""]
        entries = self.layout(lines, size)
        baselines = tuple(b for _, b in entries)
        boxes = tuple(self.line_box(size, b) for b in baselines)
        return TextLayout(
            tuple(lines), baselines, boxes, self._unit_height * size, size
        )

    # -- 上下标 -------------------------------------------------------------

    def superscript(self, size, scale=None, offset=None):
        """上标度量：缩放与偏移可配，不改变行盒，溢出会被报告。"""
        return self._script(size, True, scale, offset)

    def subscript(self, size, scale=None, offset=None):
        """下标度量：缩放与偏移可配，不改变行盒，溢出会被报告。"""
        return self._script(size, False, scale, offset)

    def _script(self, size, superscript, scale, offset):
        size = _check_size(size)
        if scale is None:
            scale = (
                self._superscript_scale if superscript else self._subscript_scale
            )
        if offset is None:
            offset = (
                self._superscript_offset if superscript else self._subscript_offset
            )
        scale = _check_positive("scale", scale)
        offset = _check_finite("offset", offset)
        if not superscript:
            offset = -offset

        script_size = size * scale
        shift = offset * size  # 向上为正
        # 行盒（未缩放）：top = -ascent*size，bottom = (descent+lineGap)*size
        box_top = -(self._ascent * size)
        box_bottom = (self._descent + self._line_gap) * size
        # 上下标字形（y 向下为正，基线上移 shift）
        glyph_top = -shift - self._ascent * script_size
        glyph_bottom = -shift + self._descent * script_size
        over_above = box_top - glyph_top
        over_below = glyph_bottom - box_bottom
        amount = 0.0
        if over_above > 0.0:
            amount += over_above
        if over_below > 0.0:
            amount += over_below
        return ScriptMetrics(script_size, shift, amount > 0.0, amount)
