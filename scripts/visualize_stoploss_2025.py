import argparse
import csv
import json
import os
from datetime import datetime
from statistics import mean, median
from typing import Dict, List, Optional, Tuple

from app.eastmoney import read_cached_kline_by_code
from app.paths import GPT_DATA_DIR


def _parse_date(value: str) -> Optional[datetime.date]:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except Exception:
        return None


def _find_index(rows, date_str: str) -> Optional[int]:
    for i, row in enumerate(rows):
        if row.date == date_str:
            return i
    return None


def _calc_trade(
    rows,
    buy_idx: int,
    hold_days: int,
    stop_pct: float,
) -> Optional[Tuple[int, str, float, float]]:
    if buy_idx < 0 or buy_idx >= len(rows):
        return None
    exit_idx = buy_idx + hold_days
    if exit_idx >= len(rows):
        return None
    buy_price = rows[buy_idx].open
    if buy_price <= 0:
        return None
    stop_price = buy_price * (1 - stop_pct)
    exit_price = rows[exit_idx].close
    exit_reason = "hold"
    for i in range(buy_idx, exit_idx + 1):
        if rows[i].low <= stop_price:
            exit_idx = i
            exit_price = stop_price
            exit_reason = "stop"
            break
    ret = exit_price / buy_price - 1
    return exit_idx, exit_reason, exit_price, ret


def _write_svg(path: str, series: Dict[str, List[float]]) -> None:
    if not series:
        return
    width = 1000
    height = 360
    padding = 40
    all_values = [v for values in series.values() for v in values]
    if not all_values:
        return
    min_v = min(all_values)
    max_v = max(all_values)
    span = max_v - min_v if max_v > min_v else 1.0

    def _x(i: int, n: int) -> float:
        return padding + (width - 2 * padding) * (i / max(1, n - 1))

    def _y(v: float) -> float:
        return height - padding - (height - 2 * padding) * ((v - min_v) / span)

    colors = ["#fbbf24", "#38bdf8", "#a78bfa"]
    polylines = []
    legend = []
    for idx, (label, values) in enumerate(series.items()):
        if not values:
            continue
        points = " ".join(f"{_x(i, len(values)):.2f},{_y(v):.2f}" for i, v in enumerate(values))
        color = colors[idx % len(colors)]
        polylines.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}" />'
        )
        legend.append((label, color))

    legend_items = []
    for i, (label, color) in enumerate(legend):
        x = padding + i * 160
        legend_items.append(
            f'<rect x="{x}" y="{height - padding + 12}" width="12" height="12" fill="{color}" />'
        )
        legend_items.append(
            f'<text x="{x + 18}" y="{height - padding + 22}" fill="#e2e8f0" font-size="12">{label}</text>'
        )

    svg = f"""<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
  <rect width="100%" height="100%" fill="#0f172a"/>
  <line x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{height - padding}" stroke="#334155" />
  <line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height - padding}" stroke="#334155" />
  {''.join(polylines)}
  <text x="{padding}" y="{padding - 10}" fill="#e2e8f0" font-size="12">Equity Curve (per-trade compounded)</text>
  {''.join(legend_items)}
</svg>
"""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(svg)


def _write_compare_svg(path: str, series: Dict[str, List[float]]) -> None:
    _write_svg(path, series)


def _run_stoploss(
    input_path: str,
    mode: str,
    cache_dir: str,
    hold_days: int,
    stop_pct: float,
    output_path: str,
    summary_path: str,
    svg_path: Optional[str] = None,
) -> Tuple[List[float], Dict[str, object]]:
    with open(input_path, "r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    filtered = [r for r in rows if r.get("mode") == mode]
    if not filtered:
        raise RuntimeError("未找到对应模式的记录")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output_rows: List[Dict[str, object]] = []
    returns: List[float] = []
    equity: List[float] = []
    equity_value = 1.0
    missing = 0

    kline_cache: Dict[str, Optional[List]] = {}

    def _get_rows(code: str):
        if code in kline_cache:
            return kline_cache[code]
        data = read_cached_kline_by_code(cache_dir, code)
        kline_cache[code] = data
        return data

    filtered.sort(key=lambda r: r.get("buy_date") or "")

    for r in filtered:
        code = r.get("code") or ""
        name = r.get("name") or ""
        buy_date = r.get("buy_date") or ""
        signal_date = r.get("date") or ""

        data = _get_rows(code)
        if not data:
            missing += 1
            continue
        buy_idx = _find_index(data, buy_date)
        if buy_idx is None:
            continue
        trade = _calc_trade(data, buy_idx, hold_days, stop_pct)
        if trade is None:
            continue
        exit_idx, exit_reason, exit_price, ret = trade
        exit_date = data[exit_idx].date
        buy_price = data[buy_idx].open
        mult = exit_price / buy_price if buy_price > 0 else 0.0

        output_rows.append(
            {
                "signal_date": signal_date,
                "buy_date": buy_date,
                "code": code,
                "name": name,
                "buy_price": round(buy_price, 4),
                "exit_date": exit_date,
                "exit_price": round(exit_price, 4),
                "exit_reason": exit_reason,
                "return": round(ret, 4),
                "multiple": round(mult, 4),
            }
        )
        returns.append(ret)
        equity_value *= (1 + ret)
        equity.append(equity_value)

    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "signal_date",
                "buy_date",
                "code",
                "name",
                "buy_price",
                "exit_date",
                "exit_price",
                "exit_reason",
                "return",
                "multiple",
            ],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    if svg_path:
        _write_svg(svg_path, {f"{int(stop_pct*100)}%": equity})

    summary: Dict[str, object] = {}
    if returns:
        win = sum(1 for r in returns if r > 0)
        summary = {
            "mode": mode,
            "stop_pct": stop_pct,
            "hold_days": hold_days,
            "signals": len(filtered),
            "trades": len(returns),
            "missing_kline": missing,
            "win_rate": round(win / len(returns), 4),
            "avg_return": round(mean(returns), 4),
            "median_return": round(median(returns), 4),
            "min_return": round(min(returns), 4),
            "max_return": round(max(returns), 4),
        }

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    return equity, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize stop-loss backtest for 2025.")
    parser.add_argument("--input", default="data/results/startup_mode_compare_2025_full.csv")
    parser.add_argument("--mode", default="mode3")
    parser.add_argument("--cache-dir", default=os.path.join(GPT_DATA_DIR, "kline_cache_tencent"))
    parser.add_argument("--hold-days", type=int, default=20)
    parser.add_argument("--stop-pct", type=float, default=0.15)
    parser.add_argument("--output", default="data/results/stoploss_mode3_2025_15.csv")
    parser.add_argument("--svg", default="data/results/stoploss_mode3_2025_15.svg")
    parser.add_argument("--summary", default="data/results/stoploss_mode3_2025_15_summary.json")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Generate 10% vs 15% comparison SVG.",
    )
    parser.add_argument(
        "--compare-svg",
        default="data/results/stoploss_mode3_2025_compare.svg",
    )
    args = parser.parse_args()

    if args.compare:
        equity10, summary10 = _run_stoploss(
            input_path=args.input,
            mode=args.mode,
            cache_dir=args.cache_dir,
            hold_days=args.hold_days,
            stop_pct=0.10,
            output_path="data/results/stoploss_mode3_2025_10.csv",
            summary_path="data/results/stoploss_mode3_2025_10_summary.json",
            svg_path=None,
        )
        equity15, summary15 = _run_stoploss(
            input_path=args.input,
            mode=args.mode,
            cache_dir=args.cache_dir,
            hold_days=args.hold_days,
            stop_pct=0.15,
            output_path="data/results/stoploss_mode3_2025_15.csv",
            summary_path="data/results/stoploss_mode3_2025_15_summary.json",
            svg_path=None,
        )
        _write_compare_svg(
            args.compare_svg,
            {
                "10% stop": equity10,
                "15% stop": equity15,
            },
        )
        print(f"输出对比曲线: {args.compare_svg}")
        print("10% 统计:", summary10)
        print("15% 统计:", summary15)
        return

    equity, summary = _run_stoploss(
        input_path=args.input,
        mode=args.mode,
        cache_dir=args.cache_dir,
        hold_days=args.hold_days,
        stop_pct=args.stop_pct,
        output_path=args.output,
        summary_path=args.summary,
        svg_path=args.svg,
    )

    print(f"输出交易明细: {args.output}")
    print(f"输出曲线图: {args.svg}")
    print(f"输出统计: {args.summary}")
    if summary:
        print("统计:", summary)


if __name__ == "__main__":
    main()
