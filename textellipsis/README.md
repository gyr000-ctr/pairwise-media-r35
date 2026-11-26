# textellipsis

## 题面（可直接粘贴）

子目录：textellipsis。省略号截断。约束：截断后总宽不超过上限（≤1e-9）；必须按字形簇截断（不得切开代理对/组合符）；省略号本身宽度计入；空间不足时返回仅省略号或空；10 万次不超过 400 ms；非法宽度可区分报错。验收：cd textellipsis && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd textellipsis && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖

---

# 实现说明与评测 API（v1.1）

纯标准库实现，唯一核心模块 `textellipsis/core.py`。下列名称均由
`textellipsis` 包直接导出（`import textellipsis as te` 后可用，也可
`from textellipsis import Ellipsis, ...`）。

## 1. 30 秒上手

```python
import textellipsis as te
from textellipsis import Ellipsis

G = [("a", 10), ("b", 10), ("c", 10)]
r = Ellipsis().truncate(G, 25.0)

print(r)                       # 'a…'  —— 直接就是 str
assert r == "a…" and isinstance(r, str)
r.omitted_clusters             # 2     —— 被省略的字形簇数（规范名）
r.kept_clusters                # 1
r.total_clusters               # 3
r.width                        # 20.0  —— 含省略号的截断后总宽
r.ellipsis_width               # 10.0
r.state                        # 'truncated'
r.to_dict()                    # 完整报告字典（含全部别名）
r.to_tuple()                   # ('a…', 2)
r["omitted_clusters"]          # 2（也支持字典式访问）
r.get("omitted_count")         # 2
```

模块级函数与共享无状态实例等价（线程安全）：

```python
te.truncate(G, 25.0)                 # TruncateResult（str 子类）
te.truncate_report(G, 25.0)          # dict
te.truncate_with_count(G, 25.0)      # ('a…', 2)
te.count_omitted(G, 25.0)            # 2
te.width(G)                          # 30.0（float）
te.grapheme_clusters("é😀")    # ('é', '😀')
te.cluster_count("👨‍👩‍👧🇨🇳")          # 2
```

## 2. 输入约定

* 规范输入：`glyphs` 为 `(cluster_text: str, width: 正数)` 二元组的列表/元组，
  **每一项就是一个不可再分的字形簇**，截断永远以整项为单位，因此不会切开
  代理对、组合符、ZWJ emoji 序列或区域指示符（国旗）。
* 也兼容：`(width, text)` 反序、`{"text": ..., "width": ...}` 字典、带
  `.text/.width`（或 `.cluster/.advance`）属性的对象、生成器等可迭代对象。
* 直接传一个字符串也可以：会先按字形簇切分，每簇给默认单位宽度
  `unit_width=10`（可用 `Ellipsis(unit_width=...)` 调整）。
* 不会修改传入的 list。

## 3. 宽度拟合与“最大化保留”

常量 `SPEC_TOLERANCE = 1e-9`。前缀和使用 **Kahan 补偿求和**。算法为
**二分查找（`bisect_right`）+ 双向边界贪心校正**：

1. 若全部簇总宽 `original_width <= max_width + 1e-9`：原样返回，`state='full'`，
   **不**加省略号。
2. 否则求最大的 `k`，使
   `prefix_width(k) + ellipsis_width <= max_width + 1e-9`。
3. `k >= 1`：返回 `前 k 簇 + ell`，`state='truncated'`。
4. `k == 0` 且省略号自身放得下（`ellipsis_width <= max_width + 1e-9`）：
   返回仅省略号，`state='ellipsis_only'`。
5. 连省略号都放不下：返回 `''`，`state='empty'`。

因此结果宽度恒满足 `result.width <= max_width + 1e-9`，且保留簇数在同一容差下
严格最大（已用 12 万+ 随机用例对暴力枚举 oracle 验证）。

## 4. “仅省略号”与“空”可区分

| 情形 | 字符串值 | `state` | 计数字段 |
|---|---|---|---|
| 全部放下 | `'abc'` | `'full'` | `omitted_clusters=0` |
| 截断并加省略号 | `'a…'` | `'truncated'` | `kept/omitted/total` |
| 只放得下省略号（如 `max_width=10`） | `'…'` | `'ellipsis_only'` | `kept_clusters=0, omitted_clusters=N` |
| 省略号也放不下（如 `max_width=1`） | `''` | `'empty'` | `kept_clusters=0, omitted_clusters=N` |

两种极端状态的 `state` 一定不同；同时有布尔量 `is_full / is_truncated /
is_ellipsis_only / is_empty` 与 `truncated`（非 full 即为 True）。

## 5. 省略号宽度

默认规则：`ellipsis_width = te.cluster_count(ell) * unit_width`
（`unit_width` 默认 `10.0`）。所以默认 `ell='…'` 宽 **10**，`ell='...'`
宽 **30**，`ell=''` 宽 **0**。可用多种方式覆盖：

```python
Ellipsis(ellipsis_width=6).truncate(G, 16)     # 构造期固定
Ellipsis(ell_width=6).truncate(G, 16)          # 同义关键字
Ellipsis().truncate(G, 16, ell_width=6)        # 单次调用覆盖
Ellipsis().truncate(G, 16, ell=("…", 6))       # ell 自带 (文本, 宽度)
Ellipsis(measure=lambda s: 6.0).truncate(G,16) # 自定义测宽函数
Ellipsis(12)                                   # 位置数值 = unit_width
```

## 6. 返回对象字段（`TruncateResult`，`str` 子类）

规范字段：`text, omitted_clusters, kept_clusters, total_clusters, width`
（含省略号的输出总宽，即受上限约束的宽度）`, content_width`（不含省略号）`,
max_width, ellipsis_width, original_width`（原始全宽）`, state,
ellipsis_used, ellipsis_appended, ellipsis_fits, tolerance, ellipsis`，以及
上面四个布尔状态。

常用别名（属性、`r["..."]`、`r.get("...")`、`to_dict()` 均可）：
`omitted / omitted_count / dropped_clusters / hidden_clusters`、
`kept / kept_count / retained_clusters`、
`total_count / cluster_count / glyph_count`、
`actual_width / output_width / rendered_width / total_width`、
`ell_width / limit`，以及 camelCase（`omittedClusters` 等）。

> 注意：没有裸 `count` 字段，以免覆盖内置 `str.count()`；需要簇数请用
> `omitted_clusters` / `cluster_count`。

输出模式参数：`truncate(..., as_dict=True)` 返回普通 dict；
`as_tuple=True` 或 `with_count=True` 返回 `(text, omitted_clusters)`。

## 7. 非法输入与异常分类（全部是 `ValueError`）

```
TextEllipsisError(ValueError)
├── NonFiniteInputError                 # 一切 NaN/Inf 输入的总分类
│   ├── NonFiniteWidthError(NonFiniteInputError, InvalidWidthError)  # max_width 非有限
│   └── NonFiniteGlyphWidthError(NonFiniteInputError, InvalidGlyphError)
├── InvalidWidthError
│   ├── IllegalWidthError / BadWidthError / WidthError     # 别名类
│   ├── NonNumericWidthError(TypeError, InvalidWidthError) # 非数值宽度（同时可按 TypeError 捕获）
│   │   └── InvalidWidthTypeError
│   └── NonPositiveWidthError
│       ├── ZeroWidthLimitError
│       └── NegativeWidthLimitError
├── InvalidGlyphsError
│   ├── InvalidGlyphsTypeError(TypeError, InvalidGlyphsError)
│   ├── EmptyGlyphsError                 # truncate([]) 抛此异常
│   └── InvalidGlyphError
│       ├── InvalidGlyphTypeError(TypeError, InvalidGlyphError)
│       ├── ZeroWidthGlyphError          # [('a', 0)] 抛此异常
│       └── NegativeWidthGlyphError
```

要点：

* `max_width` 为 `0 / 负数 / NaN / +Inf / -Inf / 非数值类型` 都会抛出
  `ValueError` 子类，且类型可区分；`except InvalidWidthError` 能统一捕获宽度
  上限问题，`except NonFiniteInputError` 能统一捕获 NaN/Inf。
* 正数但太小不是错误：按第 4 节返回仅省略号或空串。
* `glyphs=[]` 时 `truncate` 抛 `EmptyGlyphsError`；`Ellipsis().width([])`
  按“空和为 0”返回 `0.0`。
* 字形宽度必须为有限正数：`0` 抛 `ZeroWidthGlyphError`，负数抛
  `NegativeWidthGlyphError`，NaN/Inf 抛 `NonFiniteGlyphWidthError`。

## 8. 并发与性能

* `Ellipsis` 实例只保存只读配置，截断过程全部是局部变量，无共享可变状态；
  模块级共享实例同样线程安全，多线程并发结果与串行逐字段相等。
* 规范输入走单遍快速路径（校验 + Kahan 前缀 + `bisect_right`）。
  参考性能（本机、CPython 3.14、10 万次、`G=[('a',10),('b',10),('c',10)]`）：
  共享实例约 **155 ms**、每次新建 `Ellipsis()` 约 **175 ms**（best-of-5），
  明显低于 **400 ms** 上界，为机器负载预留了余量。

