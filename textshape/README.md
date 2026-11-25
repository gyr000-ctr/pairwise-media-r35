# textshape

## 题面（可直接粘贴）

子目录：textshape。字形轮廓三角化。约束：轮廓按非零环绕规则填充（与扫描线面积差 ≤1e-6）；内环（孔洞）被挖空；三角形不重叠且方向一致；自交轮廓报错；1 万顶点不超过 900 ms；非法轮廓可区分报错。验收：cd textshape && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd textshape && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
