"""GPOS PairPos 字距应用引擎。

支持 OpenType GPOS 表中的 PairPos（配对调整）两种子表格式：

- Format 1：显式字形对。``coverage`` 与 ``sets`` 下标对齐，每个 set 是
  ``{第二字形: 调整记录}``；也可省略 coverage，直接用以 ``(g1, g2)``
  元组为键的 pair map。
- Format 2：字形分类对。``class_def1`` / ``class_def2`` 把字形映射到
  分类 id（缺省为 0），``class_matrix[c1][c2]`` 给出调整记录。

调整记录（PairValueRecord）形如
``{"value1": {"x": ...}, "value2": {"x": ...}}``，其中 value1 作用于
配对中的第一个字形、value2 作用于第二个；也可扁平化直接写
``{"x": ...}``，等价于只写 value1（OpenType 字距的常见情形：调整挂
在第一个字形的步进上）。ValueRecord 支持键 ``x`` / ``y``（步进偏移）
与 ``x_placement`` / ``y_placement``（位置偏移，不影响步进输出，仅
做合法性校验）。

语义保证：

- 按请求的 feature 收集 lookup，多级 lookup 的偏移按顺序累加；
- 未被任何 lookup 覆盖的字形对步进不变；
- LookupOrder 稳定：只依赖列表顺序与整型下标，不遍历 str 集合，
  跨进程、跨线程结果一致；
- 热路径全部为字典查找，1 万字形 < 600ms；
- 非法表 / 非法字形 / 未知 feature 抛出可区分的异常类型；
- ``shape`` 无共享可变状态，可并发调用。
"""

import math
import threading

__all__ = [
    "Shaping",
    "CompiledGPOS",
    "GPOSParser",
    "LookupOrder",
    "default_font",
    "GPOSError",
    "InvalidTableError",
    "InvalidLookupError",
    "InvalidFeatureError",
    "InvalidGlyphError",
]


# ---------------------------------------------------------------------------
# 异常分类：全部继承 ValueError，既满足“按 ValueError 捕获”的旧用法，
# 又可以按具体子类区分故障来源。
# ---------------------------------------------------------------------------


class GPOSError(ValueError):
    """GPOS 相关错误的基类。"""


class InvalidTableError(GPOSError):
    """GPOS 表结构非法（features/lookups 拓扑、脚本引用等）。"""


class InvalidLookupError(InvalidTableError):
    """单个 lookup 子表非法（类型、格式、coverage、分类矩阵等）。"""


class InvalidFeatureError(GPOSError):
    """请求了字体中不存在的 feature。"""


class InvalidGlyphError(GPOSError):
    """字形输入流非法（空、非有限步进、结构错误等）。"""


_VALUE_RECORD_KEYS = ("x", "y", "x_placement", "y_placement")
_PAIR_RECORD_KEYS = ("value1", "value2")
_VALID_FORMATS = (1, 2)
_BOOL = bool
_INT = int
_FLOAT = float


def _is_finite_number(value):
    """bool 不算数值；int/float 且有限才算合法数值。"""
    if isinstance(value, _BOOL) or not isinstance(value, (_INT, _FLOAT)):
        return False
    return math.isfinite(value)


def _parse_value_record(raw):
    if not isinstance(raw, dict):
        raise InvalidLookupError("ValueRecord 必须是 dict，收到 %r" % type(raw).__name__)
    record = {"x": 0, "y": 0, "x_placement": 0, "y_placement": 0}
    for key, value in raw.items():
        if key not in _VALUE_RECORD_KEYS:
            raise InvalidLookupError("ValueRecord 含未知键 %r" % (key,))
        if not _is_finite_number(value):
            raise InvalidLookupError("ValueRecord 键 %r 的值必须是有限数值" % (key,))
        record[key] = value
    return record


def _zero_value_record():
    return {"x": 0, "y": 0, "x_placement": 0, "y_placement": 0}


def _parse_pair_record(raw):
    """返回 (value_record_1, value_record_2)，分别作用于第一/第二字形。"""
    if not isinstance(raw, dict):
        raise InvalidLookupError("配对调整记录必须是 dict，收到 %r" % type(raw).__name__)
    if "value1" in raw or "value2" in raw:
        unknown = set(raw) - set(_PAIR_RECORD_KEYS)
        if unknown:
            raise InvalidLookupError("配对记录含未知键 %r" % sorted(unknown))
        vr1 = _parse_value_record(raw["value1"]) if "value1" in raw else _zero_value_record()
        vr2 = _parse_value_record(raw["value2"]) if "value2" in raw else _zero_value_record()
        return vr1, vr2
    # 扁平化简写：整个 dict 就是 value1 的 ValueRecord
    # （OpenType 惯例：字距调整挂在配对第一个字形上）。
    return _parse_value_record(raw), _zero_value_record()


# ---------------------------------------------------------------------------
# 编译期：把用户提供的表结构校验并展开成热路径友好的不可变结构。
# ---------------------------------------------------------------------------


class CompiledLookup:
    """编译后的单个 PairPos lookup，结构对热路径只读。"""

    __slots__ = ("index", "tag", "fmt", "pair_map", "class_def1", "class_def2", "class_matrix")

    def __init__(self, index, tag, fmt, pair_map=None,
                 class_def1=None, class_def2=None, class_matrix=None):
        self.index = index
        self.tag = tag
        self.fmt = fmt
        # Format 1: {(g1, g2): (vr1, vr2)}
        self.pair_map = pair_map
        # Format 2: 字形 -> 分类 id，矩阵 [c1][c2] -> (vr1, vr2) 或 None
        self.class_def1 = class_def1
        self.class_def2 = class_def2
        self.class_matrix = class_matrix


class CompiledGPOS:
    """编译后的整张 GPOS 表，不可变，可被多个 Shaping/线程共享。"""

    def __init__(self, features, lookups):
        # feature tag -> tuple[int]（lookup 下标，保持表内顺序）
        self.features = features
        # list[CompiledLookup]，下标即 lookup index
        self.lookups = lookups

    def lookup_indices_for(self, feature_tags):
        """按 feature 请求顺序收集 lookup 下标，首次出现为准（稳定去重）。"""
        ordered = []
        seen = set()
        for tag in feature_tags:
            for lookup_index in self.features[tag]:
                if lookup_index not in seen:
                    seen.add(lookup_index)
                    ordered.append(lookup_index)
        return ordered


class GPOSParser:
    """GPOS 表解析/编译器。

    用法::

        compiled = GPOSParser().parse(table)
        shaping = Shaping(compiled)
    """

    def parse(self, table):
        return self.compile(table)

    def compile(self, table):
        if not isinstance(table, dict):
            raise InvalidTableError("GPOS 表必须是 dict，收到 %r" % type(table).__name__)

        raw_lookups = table.get("lookups")
        if not isinstance(raw_lookups, list) or not raw_lookups:
            raise InvalidTableError('GPOS 表必须含非空 "lookups" 列表')

        raw_features = table.get("features")
        if not isinstance(raw_features, dict) or not raw_features:
            raise InvalidTableError('GPOS 表必须含非空 "features" 字典')

        lookup_count = len(raw_lookups)
        features = {}
        for tag, indices in raw_features.items():
            if not isinstance(tag, str) or not tag:
                raise InvalidTableError("feature tag 必须是非空字符串")
            if not isinstance(indices, (list, tuple)) or not indices:
                raise InvalidTableError("feature %r 必须绑定非空 lookup 下标列表" % tag)
            compiled_indices = []
            for idx in indices:
                if isinstance(idx, _BOOL) or not isinstance(idx, _INT):
                    raise InvalidTableError("feature %r 的 lookup 下标必须是 int" % tag)
                if not 0 <= idx < lookup_count:
                    raise InvalidTableError(
                        "feature %r 引用了越界 lookup 下标 %d（共 %d 个 lookup）"
                        % (tag, idx, lookup_count)
                    )
                compiled_indices.append(idx)
            features[tag] = tuple(compiled_indices)

        self._validate_scripts(table, features)
        lookups = [self._compile_lookup(i, raw) for i, raw in enumerate(raw_lookups)]
        return CompiledGPOS(features, lookups)

    @staticmethod
    def _validate_scripts(table, features):
        scripts = table.get("scripts")
        if scripts is None:
            return
        if not isinstance(scripts, dict):
            raise InvalidTableError('"scripts" 必须是 dict')
        for script_tag, script in scripts.items():
            if not isinstance(script_tag, str) or not isinstance(script, dict):
                raise InvalidTableError("script 结构非法: %r" % (script_tag,))
            languages = script.get("languages", {})
            if languages is not None and not isinstance(languages, dict):
                raise InvalidTableError("script %r 的 languages 必须是 dict" % script_tag)
            lang_tables = list(languages.values()) if languages else []
            default = script.get("default_language")
            if default is not None:
                if not isinstance(default, dict):
                    raise InvalidTableError("script %r 的 default_language 必须是 dict" % script_tag)
                lang_tables.append(default)
            for lang in lang_tables:
                refs = lang.get("features", [])
                if not isinstance(refs, (list, tuple)):
                    raise InvalidTableError("语言系统的 features 必须是列表")
                for tag in refs:
                    if tag not in features:
                        raise InvalidTableError(
                            "script %r 引用了未定义 feature %r" % (script_tag, tag)
                        )

    def _compile_lookup(self, index, raw):
        if not isinstance(raw, dict):
            raise InvalidLookupError("lookup[%d] 必须是 dict" % index)
        lookup_type = raw.get("type", "pair_pos")
        if lookup_type not in ("pair_pos", 2):
            raise InvalidLookupError(
                "lookup[%d] 类型 %r 不受支持（仅支持 pair_pos/2）" % (index, lookup_type)
            )
        fmt = raw.get("format")
        if fmt not in _VALID_FORMATS:
            raise InvalidLookupError(
                "lookup[%d] format 必须是 1 或 2，收到 %r" % (index, fmt)
            )
        tag = raw.get("tag")
        if tag is not None and not isinstance(tag, str):
            raise InvalidLookupError("lookup[%d] 的 tag 必须是字符串" % index)

        if fmt == 1:
            return CompiledLookup(index, tag, 1, pair_map=self._compile_format1(index, raw))
        return CompiledLookup(
            index, tag, 2,
            class_def1=self._compile_class_def(index, raw, "class_def1"),
            class_def2=self._compile_class_def(index, raw, "class_def2"),
            class_matrix=self._compile_class_matrix(index, raw),
        )

    @staticmethod
    def _compile_format1(index, raw):
        sets = raw.get("sets")
        if sets is None:
            raise InvalidLookupError("lookup[%d] Format1 缺少 sets" % index)
        coverage = raw.get("coverage")

        if isinstance(sets, dict):
            # 简写：以 (g1, g2) 元组为键的全局 pair map，此时不需要 coverage。
            pair_map = {}
            for key, record in sets.items():
                if (not isinstance(key, tuple) or len(key) != 2
                        or not all(isinstance(g, str) and g for g in key)):
                    raise InvalidLookupError(
                        "lookup[%d] pair map 的键必须是 (str, str) 元组" % index
                    )
                pair_map[key] = _parse_pair_record(record)
            if coverage is not None:
                raise InvalidLookupError(
                    "lookup[%d] 元组键 sets 不应同时提供 coverage" % index
                )
            return pair_map

        if not isinstance(sets, list):
            raise InvalidLookupError("lookup[%d] Format1 sets 必须是 list 或 dict" % index)
        if coverage is None:
            raise InvalidLookupError(
                "lookup[%d] Format1 使用列表 sets 时必须提供 coverage" % index
            )
        if not isinstance(coverage, list) or not all(
            isinstance(g, str) and g for g in coverage
        ):
            raise InvalidLookupError("lookup[%d] coverage 必须是非空字符串列表" % index)
        if len(coverage) != len(sets):
            raise InvalidLookupError(
                "lookup[%d] coverage(%d) 与 sets(%d) 长度不一致"
                % (index, len(coverage), len(sets))
            )
        pair_map = {}
        for first_glyph, glyph_set in zip(coverage, sets):
            if not isinstance(glyph_set, dict):
                raise InvalidLookupError(
                    "lookup[%d] 字形 %r 的 pair set 必须是 dict" % (index, first_glyph)
                )
            for second_glyph, record in glyph_set.items():
                if not isinstance(second_glyph, str) or not second_glyph:
                    raise InvalidLookupError(
                        "lookup[%d] pair set 的键必须是非空字符串" % index
                    )
                pair_map[(first_glyph, second_glyph)] = _parse_pair_record(record)
        return pair_map

    @staticmethod
    def _compile_class_def(index, raw, key):
        class_def = raw.get(key)
        if not isinstance(class_def, dict) or not class_def:
            raise InvalidLookupError("lookup[%d] 缺少非空 %s" % (index, key))
        compiled = {}
        for glyph, class_id in class_def.items():
            if not isinstance(glyph, str) or not glyph:
                raise InvalidLookupError("lookup[%d] %s 的键必须是非空字符串" % (index, key))
            if isinstance(class_id, _BOOL) or not isinstance(class_id, _INT) or class_id < 0:
                raise InvalidLookupError(
                    "lookup[%d] %s 的分类 id 必须是非负 int" % (index, key)
                )
            compiled[glyph] = class_id
        return compiled

    @staticmethod
    def _compile_class_matrix(index, raw):
        matrix = raw.get("class_matrix")
        if not isinstance(matrix, list) or not matrix:
            raise InvalidLookupError("lookup[%d] 缺少非空 class_matrix" % index)
        width = None
        compiled = []
        for row in matrix:
            if not isinstance(row, list):
                raise InvalidLookupError("lookup[%d] class_matrix 必须二维" % index)
            if width is None:
                width = len(row)
            elif len(row) != width:
                raise InvalidLookupError("lookup[%d] class_matrix 行列不齐" % index)
            compiled_row = []
            for cell in row:
                compiled_row.append(None if cell is None else _parse_pair_record(cell))
            compiled.append(compiled_row)

        class_def1 = raw.get("class_def1", {})
        class_def2 = raw.get("class_def2", {})
        max_c1 = max(class_def1.values())
        max_c2 = max(class_def2.values())
        if max_c1 >= len(compiled) or max_c2 >= width:
            raise InvalidLookupError(
                "lookup[%d] 分类 id 超出 class_matrix 范围" % index
            )
        return compiled


# ---------------------------------------------------------------------------
# 内置字体：同时覆盖 Format1（两种写法）、Format2、多级 lookup 叠加。
# ---------------------------------------------------------------------------


def _build_default_font():
    return {
        "version": 1.0,
        "scripts": {
            "DFLT": {
                "default_language": {"features": ["kern", "mark"]},
            },
        },
        "features": {
            # kern 自身就由两级 lookup 叠加，验证多级查找累加。
            "kern": (0, 2),
            "mark": (1,),
        },
        "lookups": [
            {
                "tag": "kern_format1",
                "type": "pair_pos",
                "format": 1,
                "coverage": ["a"],
                "sets": [
                    {"b": {"x": -40}},
                ],
            },
            {
                "tag": "mark_format2",
                "type": "pair_pos",
                "format": 2,
                "class_def1": {"a": 1, "b": 1},
                "class_def2": {"a": 1, "b": 1},
                # 分类 0（未分类字形）整行/列均为 None => 不改位。
                "class_matrix": [
                    [None, None],
                    [None, {"x": -5}],
                ],
            },
            {
                "tag": "kern_format1_tuple_map",
                "type": "pair_pos",
                "format": 1,
                "sets": {
                    ("a", "b"): {"x": -3},
                },
            },
        ],
    }


_DEFAULT_FONT = _build_default_font()
_DEFAULT_COMPILED = GPOSParser().compile(_DEFAULT_FONT)


def default_font():
    """返回内置字体表的深拷贝，可供测试改写后重新编译。"""
    import copy

    return copy.deepcopy(_DEFAULT_FONT)


# ---------------------------------------------------------------------------
# LookupOrder
# ---------------------------------------------------------------------------


class LookupOrder:
    """显式 lookup 应用顺序。

    元素为 lookup 下标（int）或 lookup 的 tag（str）。默认顺序由 feature
    收集顺序决定（首次出现为准、下标升序），本类只用于显式重排/过滤。
    """

    __slots__ = ("items",)

    def __init__(self, items):
        if not isinstance(items, (list, tuple)) or not items:
            raise ValueError("LookupOrder 需要非空 list/tuple")
        normalized = []
        for item in items:
            if isinstance(item, _BOOL) or not isinstance(item, (_INT, str)):
                raise ValueError("LookupOrder 元素必须是 int 下标或 str tag")
            normalized.append(item)
        self.items = tuple(normalized)

    def resolve(self, compiled):
        indices = []
        used = set()
        tags = {lk.tag: lk.index for lk in compiled.lookups if lk.tag is not None}
        for item in self.items:
            if isinstance(item, str):
                if item not in tags:
                    raise InvalidLookupError("LookupOrder 引用了未知 lookup tag %r" % item)
                idx = tags[item]
            else:
                if not 0 <= item < len(compiled.lookups):
                    raise InvalidLookupError(
                        "LookupOrder 引用了越界 lookup 下标 %d" % item
                    )
                idx = item
            if idx not in used:
                used.add(idx)
                indices.append(idx)
        return tuple(indices)


# ---------------------------------------------------------------------------
# 输入校验
# ---------------------------------------------------------------------------


def _validate_glyphs(glyphs):
    if isinstance(glyphs, str) or not isinstance(glyphs, (list, tuple)):
        raise InvalidGlyphError("glyphs 必须是 (name, advance) 元组组成的列表")
    if len(glyphs) == 0:
        raise InvalidGlyphError("glyphs 不能为空")
    validated = []
    for pos, glyph in enumerate(glyphs):
        if not isinstance(glyph, (list, tuple)) or len(glyph) < 2:
            raise InvalidGlyphError("glyphs[%d] 必须是至少含 (name, advance) 的序列" % pos)
        name, advance = glyph[0], glyph[1]
        if not isinstance(name, str) or not name:
            raise InvalidGlyphError("glyphs[%d] 的字形名必须是非空字符串" % pos)
        if not _is_finite_number(advance):
            raise InvalidGlyphError("glyphs[%d] 的步进必须是有限数值" % pos)
        if advance < 0:
            raise InvalidGlyphError("glyphs[%d] 的步进不能为负" % pos)
        validated.append((name, advance))
    return validated


def _validate_features(features):
    if isinstance(features, str):
        return (features,)
    if not isinstance(features, (list, tuple)):
        raise InvalidFeatureError("features 必须是 tag 字符串列表")
    tags = []
    for tag in features:
        if not isinstance(tag, str) or not tag:
            raise InvalidFeatureError("feature tag 必须是非空字符串")
        tags.append(tag)
    return tuple(tags)


# ---------------------------------------------------------------------------
# 主 API
# ---------------------------------------------------------------------------


class Shaping:
    """GPOS 字距整形器。

    Parameters
    ----------
    font:
        GPOS 表 dict（会立即校验并编译），或已编译的 ``CompiledGPOS``。
        缺省使用内置字体（支持 ``kern`` / ``mark`` feature）。
    lookup_order:
        可选的显式 ``LookupOrder``。

    线程安全：编译结果只读，``shape`` 仅使用局部变量，最后一次结果的
    缓存写入受锁保护，可被多线程并发调用。
    """

    def __init__(self, font=None, *, lookup_order=None):
        if font is None:
            compiled = _DEFAULT_COMPILED
        elif isinstance(font, CompiledGPOS):
            compiled = font
        else:
            compiled = GPOSParser().compile(font)
        self._compiled = compiled
        self._default_order = LookupOrder(lookup_order.items) if lookup_order else None
        self._last_advances = []
        self._lock = threading.RLock()

    # -- 内部 ----------------------------------------------------------------

    def _resolve_lookups(self, feature_tags, override_order):
        compiled = self._compiled
        for tag in feature_tags:
            if tag not in compiled.features:
                raise InvalidFeatureError("字体不包含 feature %r" % tag)
        order = override_order or self._default_order
        if order is not None:
            indices = order.resolve(compiled)
            requested = set(compiled.lookup_indices_for(feature_tags))
            # 显式顺序只能在本次 feature 激活的 lookup 集合内重排/过滤。
            indices = tuple(i for i in indices if i in requested)
            if not indices:
                raise InvalidLookupError("LookupOrder 与请求的 feature 无交集")
            return indices
        return compiled.lookup_indices_for(feature_tags)

    # -- 公开 API ------------------------------------------------------------

    def shape(self, glyphs, features, *, lookup_order=None):
        """对字形流应用 GPOS PairPos，返回每个字形的 x 步进列表。

        输入不被修改；全部输入为整型时输出保持 ``int``，否则为 ``float``。
        多级 lookup 的中间调整只累加、不钳制（保证叠加精度），仅在最终
        步进为负时把该字形结果钳到 0，保证步进非负。
        """
        validated = _validate_glyphs(glyphs)
        feature_tags = _validate_features(features)
        if lookup_order is not None and not isinstance(lookup_order, LookupOrder):
            raise ValueError("lookup_order 必须是 LookupOrder 实例")
        ordered_indices = self._resolve_lookups(feature_tags, lookup_order)
        compiled = self._compiled

        n = len(validated)
        names = [g[0] for g in validated]
        bases = [g[1] for g in validated]
        # 初始为 int 0；只要累加值都是 int，adj 就保持 int 类型。
        adjustments = [0] * n

        f1_maps = []
        f2_tables = []
        for idx in ordered_indices:
            lookup = compiled.lookups[idx]
            if lookup.fmt == 1:
                f1_maps.append(lookup.pair_map)
            else:
                f2_tables.append((lookup.class_def1, lookup.class_def2, lookup.class_matrix))

        for i in range(n - 1):
            g1 = names[i]
            g2 = names[i + 1]
            for pair_map in f1_maps:
                record = pair_map.get((g1, g2))
                if record is not None:
                    adjustments[i] += record[0]["x"]
                    adjustments[i + 1] += record[1]["x"]
            for class_def1, class_def2, matrix in f2_tables:
                c1 = class_def1.get(g1, 0)
                c2 = class_def2.get(g2, 0)
                record = matrix[c1][c2]
                if record is not None:
                    adjustments[i] += record[0]["x"]
                    adjustments[i + 1] += record[1]["x"]

        result = [
            base + delta if base + delta >= 0 else (0 if isinstance(base, int) else 0.0)
            for base, delta in zip(bases, adjustments)
        ]
        with self._lock:
            self._last_advances = list(result)
        return result

    def advances(self):
        """返回最近一次 ``shape`` 的步进拷贝；尚未整形时返回空列表。"""
        with self._lock:
            return list(self._last_advances)
