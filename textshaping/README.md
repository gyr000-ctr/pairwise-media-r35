# textshaping

## 题面（可直接粘贴）

子目录：textshaping。GPOS 字距应用。约束：按 Feature 表应用 PairPos（Format1/2 均可）；多级查找叠加且偏移累加正确（≤1e-9）；未被覆盖的对不改位；查找顺序（LookupOrder）稳定；1 万字形不超过 600 ms；非法表可区分报错。验收：cd textshaping && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd textshaping && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
