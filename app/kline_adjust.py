"""K 线除权除息简易前复权：按开盘相对昨收跳空检测权息日，下调历史 OHLC。

腾讯 qfqday 在部分权息日后历史 bar 未必及时下调，本地缓存合并后易出现
「除权前高价 + 除权后低价」混算均线。本模块按 K 线自身跳空做前复权修正。
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from app.scanner import KlineRow


def ex_right_gap_ratio(prev_close: float, cur_open: float, gap_pct: float) -> bool:
    if prev_close <= 0:
        return False
    ratio = cur_open / prev_close
    return ratio < 1.0 - gap_pct or ratio > 1.0 + gap_pct


def forward_adj_factor(
    opens: np.ndarray,
    closes: np.ndarray,
    gap_pct: float = 0.12,
) -> np.ndarray:
    """前复权累计因子：下标 i 处价格乘以 factor[i] 与最新价同尺度。"""
    n = len(closes)
    factor = np.ones(n, dtype=float)
    for i in range(n - 1, 0, -1):
        if ex_right_gap_ratio(float(closes[i - 1]), float(opens[i]), gap_pct):
            factor[:i] *= float(opens[i]) / float(closes[i - 1])
    return factor


def forward_adj_ohlc_arrays(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    gap_pct: float = 0.12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    factor = forward_adj_factor(opens, closes, gap_pct=gap_pct)
    return (
        opens * factor,
        highs * factor,
        lows * factor,
        closes * factor,
        factor,
    )


def forward_adj_ohlc_rows(
    rows: Sequence[KlineRow],
    gap_pct: float = 0.12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    opens = np.array([float(r.open) for r in rows], dtype=float)
    highs = np.array([float(r.high) for r in rows], dtype=float)
    lows = np.array([float(r.low) for r in rows], dtype=float)
    closes = np.array([float(r.close) for r in rows], dtype=float)
    return forward_adj_ohlc_arrays(opens, highs, lows, closes, gap_pct=gap_pct)
