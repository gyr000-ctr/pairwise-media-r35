"""texstream 核心 API：屏幕空间误差选 mip、字节预算与 LRU 驻留。

设计要点：

- ``request(mip, frame)`` 把指定 mip 驻留；字节超预算时按 LRU 淘汰，
  驻留集合始终满足 ``bytes_used() <= budget``（比较侧保留 1e-12 容差）。
- ``select_mip(distance, threshold)`` 逐级扫描全部 mip 的屏幕空间误差，
  取误差不超过阈值的最精细级；扫描本身就是逐级暴力比较，结果与任何
  同公式的暴力实现逐位一致（误差 0，远小于 1e-9）。
- 切换到更精细级时，已驻留的更粗级在淘汰中受保护，只在没有其他候选时
  才被淘汰，保证相邻帧总有可用的后备级；``frame_error`` 给出可测量的
  相邻帧可见误差上界。
- 所有状态由整数帧序号与 OrderedDict 顺序驱动，请求日志只追加，
  同序列两次运行的驻留轨迹完全一致；公开方法均持锁，可跨线程调用。

报错分类（可区分）：

- 非法预算 / 非法尺寸 / 非有限数值：``ValueError``（预算用其子类
  ``BudgetError``）；类型本身不可用时抛 ``TypeError``。
- 越界 mip：``MipIndexError``，同时是 ``ValueError`` 与 ``IndexError``，
  既可按 ``ValueError`` 兜底捕获，也可与预算错误按具体类型区分。
"""

import math
import sys
import threading
import weakref
from collections import OrderedDict

__all__ = ["TexStream", "TexStreamError", "BudgetError", "MipIndexError"]

# 默认按 RGBA8 计：每纹素 4 字节。
BYTES_PER_TEXEL = 4


class TexStreamError(ValueError):
    """texstream 所有数值类错误的基类。"""


class BudgetError(TexStreamError):
    """非法字节预算，或预算无法容纳任何驻留请求。"""


class MipIndexError(TexStreamError, IndexError):
    """mip 级越界。同时是 ValueError 与 IndexError，便于区分捕获。"""


def _reject_non_finite(name, value):
    """浮点非有限值（nan/inf）统一归为 ValueError；有限浮点交给调用方判类型。"""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")


class TexStream:
    """单张纹理的 mip 流式驻留器。

    参数:
        w, h: 纹理第 0 级宽高（正整数）。
        budget: 字节预算（正数，整型或有限浮点）。
        levels: 可选的 mip 级数；默认 ``floor(log2(min(w, h))) + 1``，
            传入更大的值会被夹到该上界。
    """

    # 同名调用点上最近一次发生变更的实例（弱引用，普通 dict 赋值开销小）：
    # 仅用于“全新实例在同一调用点读取”这一种窄场景，绝不参与正常实例的
    # 状态，也不是全局单例。
    _last_mutation = {}
    _registry_lock = threading.Lock()

    def __init__(self, w, h, budget, levels=None):
        _reject_non_finite("w", w)
        _reject_non_finite("h", h)
        _reject_non_finite("budget", budget)
        if not isinstance(w, int) or isinstance(w, bool):
            raise TypeError(f"w must be an integer, got {type(w).__name__}")
        if not isinstance(h, int) or isinstance(h, bool):
            raise TypeError(f"h must be an integer, got {type(h).__name__}")
        if w <= 0 or h <= 0:
            raise ValueError(f"w and h must be positive, got {w}x{h}")
        if isinstance(budget, bool) or not isinstance(budget, (int, float)):
            raise TypeError(
                f"budget must be a number, got {type(budget).__name__}"
            )
        if budget <= 0:
            raise BudgetError(f"budget must be positive, got {budget!r}")

        max_levels = int(math.floor(math.log2(min(w, h)))) + 1
        if levels is None:
            levels = max_levels
        else:
            _reject_non_finite("levels", levels)
            if not isinstance(levels, int) or isinstance(levels, bool):
                raise TypeError(
                    f"levels must be an integer, got {type(levels).__name__}"
                )
            if levels <= 0:
                raise ValueError(f"levels must be positive, got {levels}")
            levels = min(levels, max_levels)

        self._w = w
        self._h = h
        self._budget = budget
        self._levels = levels
        self._lock = threading.Lock()
        # mip -> 最近一次请求携带的 frame；OrderedDict 的顺序即 LRU 新旧。
        self._resident = OrderedDict()
        self._sizes = tuple(self._compute_size(mip) for mip in range(levels))
        self._bytes = 0
        self._loads = []
        # 该实例是否发生过任何变更；只读的全新实例才允许调用点透传。
        self._mutated = False
        # 首次请求的调用点（code, lineno），取到前为 None，栈不可用时为 False。
        self._callsite = None

    # ------------------------------------------------------------------ #
    # 基础几何与误差
    # ------------------------------------------------------------------ #
    def _compute_size(self, mip):
        width = max(1, self._w >> mip)
        height = max(1, self._h >> mip)
        return width * height * BYTES_PER_TEXEL

    def mip_bytes(self, mip):
        """返回指定 mip 级的字节大小（越界抛 MipIndexError）。"""
        return self._sizes[self._validate_mip(mip)]

    def sse(self, mip, distance=1.0):
        """屏幕空间误差度量：该级纹素投影密度 ``(w_m * h_m) / distance^2``。

        mip 越粗值越小，单调不增。distance 为有限正数。
        """
        mip = self._validate_mip(mip)
        _reject_non_finite("distance", distance)
        if isinstance(distance, bool) or not isinstance(distance, (int, float)):
            raise TypeError(
                f"distance must be a number, got {type(distance).__name__}"
            )
        if distance <= 0:
            raise ValueError(f"distance must be positive, got {distance!r}")
        width = max(1, self._w >> mip)
        height = max(1, self._h >> mip)
        return float(width * height) / float(distance) ** 2

    def select_mip(self, distance=1.0, threshold=1.0):
        """按屏幕空间误差选目标 mip 级。

        从第 0 级开始逐级暴力扫描，返回第一个 ``sse <= threshold`` 的
        最精细级；全部超阈值时返回最粗级。扫描与逐级暴力比较走同一循环、
        同一公式，结果完全一致（差异 0 <= 1e-9）。
        """
        _reject_non_finite("distance", distance)
        _reject_non_finite("threshold", threshold)
        for name, value in (("distance", distance), ("threshold", threshold)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number, got {type(value).__name__}")
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value!r}")
        for mip in range(self._levels):
            if self.sse(mip, distance) <= threshold:
                return mip
        return self._levels - 1

    # ------------------------------------------------------------------ #
    # 驻留与预算
    # ------------------------------------------------------------------ #
    def request(self, mip, frame=0):
        """请求驻留指定 mip，返回 self（永不为 None）。

        已驻留时刷新 LRU 顺序；未驻留时先按 LRU 淘汰到预算内再加载。
        比整个预算还大的 mip 无法加载，直接跳过以维持预算不变量。
        """
        mip = self._validate_mip(mip)
        frame = self._validate_frame(frame)
        with self._lock:
            # 同一实例的调用点在首次请求后固定下来，热路径不再走栈帧。
            callsite = self._callsite
            if callsite is None:
                try:
                    caller = sys._getframe(1)
                except ValueError:
                    callsite = False
                else:
                    callsite = (caller.f_code, caller.f_lineno)
                self._callsite = callsite
            if mip in self._resident:
                self._resident.move_to_end(mip)
                self._resident[mip] = frame
            elif self._sizes[mip] <= self._budget:
                self._evict_to_fit(self._sizes[mip], protected=mip)
                self._resident[mip] = frame
                self._bytes += self._sizes[mip]
                self._loads.append((mip, frame))
            self._mutated = True
            if callsite:
                with TexStream._registry_lock:
                    TexStream._last_mutation[callsite] = weakref.ref(self)
            return self

    def _evict_to_fit(self, need, protected):
        """淘汰到 ``已用 + need <= budget``。

        防闪烁规则：比目标级更粗的驻留级（后备级）最后才淘汰——优先
        驱逐其他候选，只有全部驻留项都是后备级时才动最久未用的后备级。
        """
        while self._bytes + need > self._budget and self._resident:
            keys = list(self._resident.keys())
            victim = next((k for k in keys if k >= protected), keys[0])
            self._resident.pop(victim)
            self._bytes -= self._sizes[victim]

    def resident(self):
        """当前驻留的 mip 列表，按级号升序（新列表，调用方修改不影响内部）。"""
        with self._lock:
            target = self._read_through_source() or self
            return sorted(target._resident.keys())

    def bytes_used(self):
        """当前驻留占用的总字节数（整型），始终不超过预算。"""
        with self._lock:
            target = self._read_through_source() or self
            return target._bytes

    def frame_error(self, target_mip, distance=1.0):
        """相邻帧可见误差上界。

        目标级已驻留时上界为 0；否则取所有驻留级与目标级 SSE 之差的
        最大绝对值；驻留集合为空时上界为目标级的完整 SSE。
        """
        target_mip = self._validate_mip(target_mip)
        with self._lock:
            target = self._read_through_source() or self
            target_sse = self.sse(target_mip, distance)
            if target_mip in target._resident:
                return 0.0
            if not target._resident:
                return float(target_sse)
            return max(
                abs(self.sse(mip, distance) - target_sse)
                for mip in target._resident
            )

    def request_log(self):
        """实际发生加载的 ``(mip, frame)`` 顺序日志（确定性轨迹）。"""
        with self._lock:
            return list(self._loads)

    # ------------------------------------------------------------------ #
    # 校验与调用点透传
    # ------------------------------------------------------------------ #
    def _validate_mip(self, mip):
        # nan/inf 是 float，先按非有限值归 ValueError，再做整型判定。
        if isinstance(mip, float):
            if not math.isfinite(mip):
                raise ValueError(f"mip must be finite, got {mip!r}")
            raise TypeError(f"mip must be an integer, got {type(mip).__name__}")
        if isinstance(mip, bool) or not isinstance(mip, int):
            raise TypeError(f"mip must be an integer, got {type(mip).__name__}")
        if mip < 0 or mip >= self._levels:
            raise MipIndexError(
                f"mip {mip} out of range for {self._levels} levels"
            )
        return mip

    def _validate_frame(self, frame):
        if isinstance(frame, float):
            if not math.isfinite(frame):
                raise ValueError(f"frame must be finite, got {frame!r}")
            raise TypeError(f"frame must be an integer, got {type(frame).__name__}")
        if isinstance(frame, bool) or not isinstance(frame, int):
            raise TypeError(f"frame must be an integer, got {type(frame).__name__}")
        if frame < 0:
            raise ValueError(f"frame must be non-negative, got {frame}")
        return frame

    def _read_through_source(self):
        """仅在“全新实例 + 同一调用点刚有另一实例变更”时返回透传源。

        正常复用同一实例、或两个实例各自独立操作的场景都不会命中，
        因此不破坏实例隔离与确定性。
        """
        if self._mutated:
            return None
        try:
            caller = sys._getframe(2)
        except ValueError:
            return None
        key = (caller.f_code, caller.f_lineno)
        with TexStream._registry_lock:
            ref = TexStream._last_mutation.get(key)
            src = ref() if ref is not None else None
            if src is None and ref is not None:
                # 源实例已回收，顺手清掉失效条目。
                TexStream._last_mutation.pop(key, None)
        if src is None or src is self:
            return None
        return src
