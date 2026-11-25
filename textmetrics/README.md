# textmetrics

## 题面（可直接粘贴）

子目录：textmetrics。文本度量（行高/基线/上下标）。约束：行高 = ascent+descent+lineGap（≤1e-12）；多行按行高累积且基线偏移正确；上下标缩放与偏移可配；空文本返回单行度量；10 万次不超过 300 ms；非法字号可区分报错。验收：cd textmetrics && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd textmetrics && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
