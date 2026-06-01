# mode34 模型说明（底部突破回踩二波）

> **定位**：股价处于**阶段底部**后，出现**强阳/近涨停突破**，再**缩量回踩平台不破启动低点**，信号日为**二波放量阳线确认**。  
> **参考样本**：电科数字 600850（底 5/15 → 突破 5/18～19 → 回踩 5/21～25 → 买点 **5/26**）。

---

## 一、四段结构

```text
价格
      │                    ╭── 二波确认（信号日）
      │               ╭────╯
      │          ╭────╯ 缩量回踩平台
      │     ╭────╯ 强阳突破
阶段底├─────╯
      └────────────────────────→ 时间
        ①底    ②突破    ③回踩    ④二波
```

| 阶段 | 含义 | 600850 参照 |
|------|------|-------------|
| ① 阶段底 | 60 日区间下沿，启动前低点 | 5/15 收 18.60，区间位置约 1% |
| ② 强阳突破 | 1～4 日内自底部涨幅 ≥12%，含大阳线/涨停 | 5/18～19 连拉约 +20% |
| ③ 缩量回踩 | 峰后 2～8 日，回撤 4%～20%，量缩，不破启动低点 | 5/21～25 回踩 21 元一带 |
| ④ 二波确认 | 信号日收阳、涨幅≥1.5%、放量、收盘高于回踩高点 | **5/26** 收 22.85 |

---

## 二、与相近模式的区别

| 模式 | 位置 | 买点 |
|------|------|------|
| mode底部大阳线 | 低位 | **第一根**大阳线 |
| 中位大阳线 | 锚点后已涨一截 | 平台**首阳**突破 |
| 天量锚点支撑回踩 | 月级锚点后深调 | 回踩 L0/VWAP |
| **mode34** | **阶段底** | 突破后**回踩再起** |

---

## 三、核心参数（`ScanConfig` / `mode34_default_kw`）

| 参数 | 默认 | 说明 |
|------|------|------|
| `bottom_lookback` | 60 | 底部区间回溯日 |
| `bottom_pos_max` | 0.30 | 启动低点在 60 日区间位置上限 |
| `surge_cum_pct_min` | 12% | 底部→突破峰最小涨幅 |
| `surge_big_pct_min/main` | 7% / 4.5% | 突破段大阳线阈值 |
| `pullback_days_min/max` | 2 / 8 | 峰后到信号日交易日数 |
| `pullback_dd_min/max` | 4% / 20% | 峰到回踩低点回撤 |
| `pullback_vol_ratio_max` | 0.75 | 回踩均量 / 突破峰值量 |
| `platform_break_tol` | 3% | 允许低于启动低点比例 |
| `signal_pct_min` | 1.5% | 信号日最小涨幅 |
| `signal_vol_mult` | 1.10 | 信号日量 / vol20 |

---

## 四、盘中买点工作流（观察日 + 预案日）

以德福科技为例：**5/22 入观察 → 5/25 出建议 → 5/26 盘中突破买入**。

| 阶段 | 日期示例 | 脚本 | 含义 |
|------|----------|------|------|
| 观察日 | 5/22 | `--watch-date` | 突破后回踩平台、缩量，尚未二波确认；**对齐电科**需贴近底部启动（见下） |

**观察池电科模版过滤（相对旧版全市场 126 只）：**

- `bottom_pos_pct` ≤ 12%（电科 5/22 约 5%）
- 底→峰涨幅 `surge_rise_pct` ≤ 35%（电科约 25%）
- 相对铁底涨幅 `rise_from_base_pct` ≤ 20%
- 收盘距平台上沿 `dist_to_pull_high_pct` ≤ 8%（电科约 3.6%）
- **启动段最低价** = 自 `base_date` 至观察日区间内的最低 `low`；须为观察日前 **N 个自然月**内最低价，**N≥12**（默认 12，电科 5/15 低 18.53 为约 17 月新低）
- **放量涨停启动**：铁底日至突破峰之间，至少 **1 日涨停**，且该日 **20 日量比 ≥ 2.0**（电科 5/19 涨停、量比约 3.06）
- **6 个月最大量**：启动段（铁底日→突破峰）须含观察日前 **6 个自然月**内**最大成交量**（电科 5/20 为近 6 月天量；不满足则不入观察池）

旧版观察分曾奖励「涨得多、离平台远」，与电科不符，已改为按上表打分。

| 预案日 | 5/25 | `--prebuy-date` | 小阳缩量企稳，输出偏多/试探/观察/放弃 + **突破昨高触发价** |
| 确认日 | 5/26 | `scan_mode34_today` | 完整 mode34 二波阳线（可对照验证） |

**区间扫描（二波确认 / 电科观察池）：**

```bash
python3 scripts/scan_mode34_period.py --start 2026-05-01 --end 2026-05-31
python3 scripts/scan_mode34_watch_period.py --start 2026-05-01 --end 2026-05-31
```

```bash
# 1) 观察日：全市场入池
python3 scripts/scan_mode34_watch_prebuy.py --watch-date 2026-05-22

# 1b) 观察池区间（如整月电科模版）
python3 scripts/scan_mode34_watch_period.py --start 2026-05-01 --end 2026-05-31

# 2) 预案日：仅对观察池给买卖建议
python3 scripts/scan_mode34_watch_prebuy.py --prebuy-date 2026-05-25 \
  --from-watch-csv data/gpt/results/mode34_watch_20260522.csv

# 单股
python3 scripts/scan_mode34_watch_prebuy.py --watch-date 2026-05-22 --prebuy-date 2026-05-25 --code 600850
```

**预案日输出字段**

- `advice`：偏多买入 / 轻仓试探 / 继续观察 / 放弃
- `buy_trigger_above`：次日盘中参考触发价（≈ 预案日最高价）
- `stop_below`：平台铁底下方止损参考
- `next_day_mode34`：次日是否满足完整 mode34（如是则成功率更高）

## 五、用法（全市场确认日扫描）

```bash
# 单日扫描（默认最新交易日）
python3 scripts/scan_mode34_today.py

# 指定日期 / 单股验证
python3 scripts/scan_mode34_today.py --date 2026-05-26 --code 600850 --min-score 55

# 区间逐日
python3 scripts/scan_mode34_period.py --start 2026-05-01 --end 2026-05-31
```

输出：`data/gpt/results/mode34_bottom_break_pullback_YYYYMMDD.csv`

---

## 五、实现位置

- 逻辑：`app/mode34_bottom_break_pullback.py`
- 扫描：`scripts/scan_mode34_today.py`、`scripts/scan_mode34_period.py`
