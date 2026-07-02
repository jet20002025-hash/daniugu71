#!/usr/bin/env python3
"""仅修正指定交易日相对前一日成交量异常放大(>=10倍)的×100/÷100 错误，不做全文件缩放量。

用法:
  python3 scripts/fix_volume_spike_day.py --date 2026-07-01
  python3 scripts/fix_volume_spike_day.py --date 2026-06-29,2026-06-30,2026-07-01
  python3 scripts/fix_volume_spike_day.py --date 2026-07-01 --dry-run
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.paths import GPT_DATA_DIR

CSV_FIELDS = [
    "date", "open", "close", "high", "low", "volume",
    "amount", "amplitude", "pct_chg", "chg", "turnover",
]
DIV = 100.0
MIN_SPIKE = 10.0
LOW_VOL = 5_000.0
LOW_RUN_MAX = 2_000.0
LOW_RUN_DAYS = 3
HIGH_VOL = 200_000.0
RATIO_LO = 0.08
RATIO_HI = 20.0


def _ratio_ok(a: float, b: float) -> bool:
    if a <= 0 or b <= 0:
        return False
    r = a / b
    return RATIO_LO <= r <= RATIO_HI


def decide_spike_pair_fix(vp: float, vt: float, vol_prev2: float | None) -> Tuple[float, float] | None:
    """今/昨量比>=10 时，在三种 ×100/÷100 组合中选择。"""
    if vp <= 0 or vt <= 0 or vt / vp < MIN_SPIKE:
        return None

    r_div_today = (vt / DIV) / vp
    r_mul_yday = vt / (vp * DIV)
    r_both = (vt / DIV) / (vp * DIV)

    only_today = _ratio_ok(vt / DIV, vp)
    only_yday = _ratio_ok(vt, vp * DIV)
    both = _ratio_ok(vt / DIV, vp * DIV)

    prev2 = float(vol_prev2 or 0)

    # 昨量已正常(>=5万)且今量极大(疑似×100)：只除今日
    if vp >= LOW_VOL and only_today and vt > HIGH_VOL:
        return vp, vt / DIV

    # 今量已处于正常手数区间时，不因量比大而去缩放（多为真实放量）
    if vt < HIGH_VOL and vp < LOW_VOL * 3 and vt / vp >= MIN_SPIKE:
        return None

    # 昨量偏小(<2千)且今量正常：只乘昨日
    if vp < LOW_RUN_MAX and vt >= LOW_VOL and only_yday:
        return vp * DIV, vt

    # 昨量与再前一日同量级、今量极大：多为真实放量（如今/昨比 80+）
    if (
        prev2 > 0
        and prev2 * 0.85 <= vp <= prev2 * 1.15
        and vt / vp > 50
        and only_today
    ):
        return None

    # 昨量相对再前一日异常放大(×100 漏乘)，今量亦大：乘昨日
    if (
        prev2 > 0
        and vp > prev2 * 2.5
        and vp < prev2 * 5.0
        and vt / vp >= MIN_SPIKE
        and _ratio_ok(vt, vp * DIV)
    ):
        return vp * DIV, vt

    if only_today and not only_yday:
        return vp, vt / DIV
    if only_yday and not only_today:
        return vp * DIV, vt
    if both:
        # 昨量不大时优先只除今日，避免误乘昨日
        if vp < LOW_VOL:
            return vp, vt / DIV
        return vp * DIV, vt / DIV
    if only_today:
        return vp, vt / DIV
    if only_yday:
        return vp * DIV, vt
    return None


def _scale_up_low_run_before_spike(volumes: List[float], i: int) -> bool:
    """目标日前连续过小成交量(疑似÷100)，而后几日已恢复正常量级时，整体×100。"""
    if i <= 0:
        return False
    future = [float(volumes[j]) for j in range(i, min(len(volumes), i + 4)) if float(volumes[j]) > 0]
    if not future or max(future) < LOW_VOL:
        return False
    changed = False
    j = i - 1
    steps = 0
    while j >= 0 and steps < LOW_RUN_DAYS:
        v = float(volumes[j])
        if v <= 0 or v >= LOW_RUN_MAX:
            break
        volumes[j] = v * DIV
        changed = True
        j -= 1
        steps += 1
    return changed


def _scale_down_high_spike(volumes: List[float], i: int) -> bool:
    """单日成交量疑似×100 过大。"""
    v = float(volumes[i])
    if v < HIGH_VOL:
        return False
    nb = [
        float(volumes[j])
        for j in range(max(0, i - 6), min(len(volumes), i + 4))
        if j != i and float(volumes[j]) > 0
    ]
    if not nb:
        return False
    scaled = v / DIV
    med = sorted(nb)[len(nb) // 2]
    if med <= 0:
        return False
    if v < med * 8:
        return False
    if abs(scaled - med) + 1.0 < abs(v - med):
        volumes[i] = scaled
        return True
    return False


def _best_pair_fix(volumes: List[float], i: int) -> Tuple[float, float] | None:
    vp = float(volumes[i - 1])
    vt = float(volumes[i])
    vol_prev2 = float(volumes[i - 2]) if i >= 2 else None
    return decide_spike_pair_fix(vp, vt, vol_prev2)


def process_file(path: str, targets: set[str], dry_run: bool) -> dict | None:
    rows: List[dict] = []
    volumes: List[float] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            try:
                volumes.append(float(row.get("volume") or 0))
            except (TypeError, ValueError):
                volumes.append(0.0)
    if not rows:
        return None

    fixed = 0
    ordered_targets = sorted(targets)
    for target in ordered_targets:
        for i in range(1, len(rows)):
            if str(rows[i].get("date", ""))[:10] != target:
                continue
            if _scale_up_low_run_before_spike(volumes, i):
                fixed += 1
            pair = _best_pair_fix(volumes, i)
            if pair is not None:
                new_vp, new_vt = pair
                if new_vp != volumes[i - 1] or new_vt != volumes[i]:
                    volumes[i - 1], volumes[i] = new_vp, new_vt
                    fixed += 1
            elif _scale_down_high_spike(volumes, i):
                fixed += 1

    if fixed == 0:
        return None

    if not dry_run:
        for row, vol in zip(rows, volumes):
            row["volume"] = vol
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    code = os.path.splitext(os.path.basename(path))[0]
    return {"code": code, "fixed_pairs": fixed}


def main() -> int:
    ap = argparse.ArgumentParser(description="修正指定日成交量今/昨>=10倍的单位错误")
    ap.add_argument("--date", required=True, help="目标交易日 YYYY-MM-DD，可逗号分隔多个")
    ap.add_argument("--cache-dir", default=os.path.join(GPT_DATA_DIR, "kline_cache_tencent"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    targets = {d.strip()[:10] for d in args.date.split(",") if d.strip()}

    files = sorted(
        os.path.join(args.cache_dir, fn)
        for fn in os.listdir(args.cache_dir)
        if fn.endswith(".csv")
    )
    changed = []
    for path in files:
        try:
            st = process_file(path, targets, args.dry_run)
        except Exception:
            continue
        if st:
            changed.append(st)

    mode = "DRY-RUN" if args.dry_run else "WRITE"
    print(f"[{mode}] 日期 {','.join(sorted(targets))}  修正文件 {len(changed)} 只")
    for st in changed[:30]:
        print(f"  {st['code']}  修正 {st['fixed_pairs']} 处")
    if len(changed) > 30:
        print(f"  ... 另有 {len(changed) - 30} 只")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
