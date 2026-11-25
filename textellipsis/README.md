# textellipsis

## 题面（可直接粘贴）

子目录：textellipsis。省略号截断。约束：截断后总宽不超过上限（≤1e-9）；必须按字形簇截断（不得切开代理对/组合符）；省略号本身宽度计入；空间不足时返回仅省略号或空；10 万次不超过 300 ms；非法宽度可区分报错。验收：cd textellipsis && python -m pytest tests -q。

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
