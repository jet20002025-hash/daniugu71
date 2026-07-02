"""K 线成交量单位统一为「手」。

腾讯/东财日 K 接口返回的 volume 均为手。部分历史缓存误存为手×100，
与后续增量更新（手）混在同一 CSV 中，会导致均量、量比等计算偏差约 100 倍。
"""
from __future__ import annotations

from typing import List, Sequence, TypeVar

import numpy as np

T = TypeVar("T")

_LEGACY_DIVISOR = 100.0
# 误存≈真实×100 时，相对近端参考通常 > ref×15（如 33,931→3,393,050）
# 不用 500万 绝对下限：会盖住小盘 ref×15（150k×15=225万 < 339万 就漏判）
_MIN_LEGACY_VOLUME = 1_000_000.0
_LEGACY_SCALE_VS_REF = 15.0
# 从末日向前 walk，相邻簇与中位数相差超过该倍数视为单位切换点
_UNIT_BREAK_RATIO = 20.0
_MAX_TAIL_CLUSTER = 40
_MAX_PASSES = 3
# 近端中位数超过该值（手/日）时，整文件多为统一手×100（如 000002）；过低会误伤放量日
_FILE_LEGACY_MEDIAN = 50_000_000.0
# 仅当 v/100 明显低于邻近日中位数时才 ÷100（避免把涨停放量当成误存）
_NEIGHBOR_RATIO_MIN = 50.0
_NEIGHBOR_DIV100_MAX = 0.15


def _tail_reference_volume_lots(volumes: List[float]) -> float:
    """从最近交易日向前，取单位一致的一段成交量的中位数（手）。"""
    tail = [float(v) for v in volumes if float(v) > 0]
    if not tail:
        return 0.0
    cluster: List[float] = [tail[-1]]
    for v in reversed(tail[:-1]):
        med = float(np.median(cluster))
        if med <= 0:
            break
        ratio = max(v, med) / min(v, med)
        if ratio > _UNIT_BREAK_RATIO:
            break
        cluster.append(v)
        if len(cluster) >= _MAX_TAIL_CLUSTER:
            break
    return float(np.median(cluster))


def _legacy_volume_cutoff(ref: float) -> float:
    """判定阈值：优先相对近端参考，辅以较低绝对下限。"""
    if ref <= 0:
        return _MIN_LEGACY_VOLUME
    return max(ref * _LEGACY_SCALE_VS_REF, _MIN_LEGACY_VOLUME)


def _fix_intermittent_100x_spikes(volumes: List[float]) -> bool:
    """部分日期误存×100：与前后正常交易日相比约 ×100 时 ÷100。"""
    n = len(volumes)
    changed = False
    for i in range(n):
        v = float(volumes[i])
        if v <= 0:
            continue
        nb = [
            float(volumes[j])
            for j in range(max(0, i - 8), min(n, i + 9))
            if j != i and float(volumes[j]) > 0
        ]
        if not nb:
            continue
        scaled = v / _LEGACY_DIVISOR
        for b in nb:
            if b <= 0:
                continue
            ratio = v / b
            if 80.0 <= ratio <= 120.0 and abs(scaled - b) / b <= 0.35:
                volumes[i] = scaled
                changed = True
                break
    return changed


def _fix_volumes_by_neighbors(volumes: List[float]) -> bool:
    """单日相对前后中位数约 ×100 时 ÷100（补 225万～500万 漏网）。"""
    n = len(volumes)
    changed = False
    for i in range(n):
        v = float(volumes[i])
        if v <= 0:
            continue
        nb = [
            float(volumes[j])
            for j in range(max(0, i - 5), min(n, i + 6))
            if j != i and float(volumes[j]) > 0
        ]
        if not nb:
            continue
        med = float(np.median(nb))
        if med <= 0:
            continue
        scaled = v / _LEGACY_DIVISOR
        if v > med * _NEIGHBOR_RATIO_MIN and scaled < med * _NEIGHBOR_DIV100_MAX:
            volumes[i] = scaled
            changed = True
    return changed


def _volume_ratio_ok(a: float, b: float, lo: float = 0.15, hi: float = 5.0) -> bool:
    if a <= 0 or b <= 0:
        return False
    r = a / b
    return lo <= r <= hi


def _log_coherence_std(volumes: List[float], lo: int, hi: int) -> float:
    """近几日成交量数量级一致性（标准差越小越好）。"""
    vals = [float(volumes[j]) for j in range(lo, hi + 1) if float(volumes[j]) > 0]
    if len(vals) < 3:
        return 1e9
    logs = [np.log10(v) for v in vals]
    return float(np.std(logs))


def _fix_spike_day_pair_100x(volumes: List[float], min_spike: float = 50.0) -> bool:
    """今/昨量比极大(>=10)时纠偏：在「今÷100」「昨×100」「双向」中选数量级最一致者。"""
    n = len(volumes)
    changed = False
    for i in range(1, n):
        vp = float(volumes[i - 1])
        vt = float(volumes[i])
        if vp <= 0 or vt <= 0 or vt / vp < min_spike:
            continue
        lo = max(0, i - 6)
        base_std = _log_coherence_std(volumes, lo, i)
        best_std = base_std
        best_pair: tuple[float, float] | None = None

        for new_vp, new_vt in (
            (vp, vt / _LEGACY_DIVISOR),
            (vp * _LEGACY_DIVISOR, vt),
            (vp * _LEGACY_DIVISOR, vt / _LEGACY_DIVISOR),
        ):
            if new_vp <= 0 or new_vt <= 0:
                continue
            ratio = new_vt / new_vp
            if ratio < 0.08 or ratio > 8.0:
                continue
            old_vp, old_vt = volumes[i - 1], volumes[i]
            volumes[i - 1], volumes[i] = new_vp, new_vt
            std = _log_coherence_std(volumes, lo, i)
            volumes[i - 1], volumes[i] = old_vp, old_vt
            if std < best_std - 1e-6:
                best_std = std
                best_pair = (new_vp, new_vt)

        if best_pair is not None:
            volumes[i - 1], volumes[i] = best_pair
            changed = True
    return changed


def _apply_file_legacy_scale(volumes: List[float]) -> bool:
    """近 60 日中位数极高时，整文件按 ÷100 纠偏（统一误存）。"""
    v = [float(x) for x in volumes if float(x) > 0]
    if len(v) < 30:
        return False
    recent = v[-60:] if len(v) >= 60 else v
    if float(np.median(recent)) <= _FILE_LEGACY_MEDIAN:
        return False
    for i in range(len(volumes)):
        if float(volumes[i]) > 0:
            volumes[i] = float(volumes[i]) / _LEGACY_DIVISOR
    return True


def normalize_kline_volumes_inplace(rows: List[T]) -> List[T]:
    """将混用「手×100」的历史行修正为手（原地修改 volume）。"""
    if not rows or len(rows) < 5:
        return rows

    vols0 = [float(getattr(r, "volume", 0) or 0) for r in rows]
    if _apply_file_legacy_scale(vols0):
        for r, v in zip(rows, vols0):
            r.volume = v

    for _ in range(_MAX_PASSES):
        vols = [float(getattr(r, "volume", 0) or 0) for r in rows]
        ref = _tail_reference_volume_lots(vols)
        changed = False
        if ref > 0:
            cutoff = _legacy_volume_cutoff(ref)
            for r in rows:
                v = float(getattr(r, "volume", 0) or 0)
                if v > cutoff:
                    r.volume = v / _LEGACY_DIVISOR
                    changed = True
        vols = [float(getattr(r, "volume", 0) or 0) for r in rows]
        if _fix_intermittent_100x_spikes(vols):
            for r, v in zip(rows, vols):
                r.volume = v
            changed = True
        vols = [float(getattr(r, "volume", 0) or 0) for r in rows]
        if _fix_volumes_by_neighbors(vols):
            for r, v in zip(rows, vols):
                r.volume = v
            changed = True
        vols = [float(getattr(r, "volume", 0) or 0) for r in rows]
        if _fix_spike_day_pair_100x(vols):
            for r, v in zip(rows, vols):
                r.volume = v
            changed = True
        if not changed:
            break
    return rows


def tencent_volume_to_cache_lots(
    live_volume: float,
    cache_volume: float | None = None,
) -> float:
    """腾讯 API volume 与本地缓存「手」对齐（部分标的 API 为股、缓存为手）。"""
    lv = float(live_volume or 0)
    if lv <= 0:
        return lv
    cv = float(cache_volume) if cache_volume is not None else None
    candidates = [lv, lv / _LEGACY_DIVISOR]
    if cv is not None and cv > 0:
        return min(candidates, key=lambda x: abs(cv - x))
    # 无缓存参考：极大值多为「股」
    if lv > 50_000_000:
        return lv / _LEGACY_DIVISOR
    return lv


def align_recent_volumes_from_api_inplace(
    rows: List[T],
    api_rows: Sequence[T],
    lookback: int = 10,
) -> int:
    """用 api_rows 最近 lookback 个交易日成交量（手）覆盖 rows 中同名日期。

    写入缓存前的最终步骤：以接口返回为准，避免启发式纠偏污染近端成交量。
    """
    if not rows or not api_rows or lookback <= 0:
        return 0
    api_vol = {
        str(getattr(r, "date", ""))[:10]: float(getattr(r, "volume", 0) or 0)
        for r in api_rows[-lookback:]
    }
    if not api_vol:
        return 0
    changed = 0
    for row in rows:
        d = str(getattr(row, "date", ""))[:10]
        if d not in api_vol:
            continue
        nv = api_vol[d]
        ov = float(getattr(row, "volume", 0) or 0)
        if abs(ov - nv) > max(1.0, nv * 1e-6):
            row.volume = nv
            changed += 1
    return changed


def normalize_kline_volumes(rows: Sequence[T]) -> List[T]:
    """只读场景：返回修正后的新列表（不修改输入）。"""
    if not rows:
        return list(rows)
    out = list(rows)
    normalize_kline_volumes_inplace(out)
    return out
