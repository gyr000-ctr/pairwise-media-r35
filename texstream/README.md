# texstream

## 题面（可直接粘贴）

子目录：texstream。纹理流式驻留（mip 驻留与预算）。约束：按屏幕空间误差选目标 mip 级，与逐级暴力比较一致（≤1e-9）；驻留集合始终满足字节预算（≤1e-12 误差）；淘汰用 LRU 且切换级时不得闪烁（相邻帧可见误差上界可测）；请求顺序确定，同序列两次运行驻留轨迹一致；10 万次查询不超过 400 ms；非法预算与越界 mip 可区分报错。验收：cd texstream && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd texstream && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
