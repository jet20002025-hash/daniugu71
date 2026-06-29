import json
import math
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from .eastmoney import KlineRow, StockItem
from .weekly_ma import (
    daily_to_monthly_with_last_index,
    daily_to_weekly_with_last_index,
    daily_to_weekly_with_volume_and_last_index,
    _rolling_mean,
    weekly_convergence_value_series,
    weekly_kdj,
)


@dataclass
class ScanResult:
    code: str
    name: str
    score: int
    latest_close: float
    change_pct: float
    reasons: List[str]
    metrics: Dict[str, Any]


@dataclass
class ScanConfig:
    min_score: int = 70
    max_results: int = 200
    volume_ratio: float = 1.2
    near_high_pct: float = 3.0
    breakout_lookback: int = 20
    breakout_recent: int = 3
    year_lookback: int = 240
    year_return_limit: float = 500.0
    year_high_low_ratio_limit: float = 4.0  # 近一年最高/最低超4倍则排除
    cache_days: int = 2
    workers: int = 6
    weight_trend: float = 1.3
    weight_volume: float = 1.4
    weight_breakout: float = 1.0
    weight_strength: float = 1.0
    weight_risk: float = 1.0
    max_market_cap: Optional[float] = 15_000_000_000.0
    mode8_n_bars: int = 60  # mode8 起算 K 线根数（70/80/90 等），仅 use_mode8 时生效
    mode10_conv_max: float = 1.0  # mode10 买点前周线拟合值上限（%），满足 拟合<conv_max 才出信号
    mode10_ma30_turn_weeks: int = 5  # mode10 信号周前 N 周内需存在 MA30 拐头向上
    mode11_accel_th: float = 2.5  # mode11 同 mode10
    mode11_body_ratio_max: float = 0.35  # mode11 拐点形态：实体/振幅上限
    mode11_vol_ratio_min: float = 1.5  # mode11 当周量 >= 该倍数 * 过去 N 周均量
    mode11_vol_weeks: int = 20  # mode11 均量回溯周数
    mode12_accel_th: float = 2.5  # mode12 同 mode10
    mode12_ma30_turn_weeks: int = 5  # mode12 信号周前 N 周内需存在周线 MA30 拐头向上
    mode88_d_min: float = 0.03  # 震仓周相对前周最小跌幅
    mode88_d_max: float = 0.15  # 震仓周相对前周最大跌幅
    mode88_r_min: float = 0.03  # 第3周相对第2周最小涨幅
    mode88_acc_L: int = 8
    mode88_acc_R: int = 20
    mode88_A_min: float = 15.0
    mode88_A_max: float = 55.0
    mode88_epsilon: float = 0.02
    mode88_wash_L: int = 2
    mode88_wash_R: int = 10
    mode88_R_rise: float = 8.0
    mode88_D_pull: float = 3.0
    mode88_K_vol: float = 1.0

    # mode90：日线 MACD 三项归一化加分参数
    macd_norm_factor: float = 1.0  # DIF/DEA 归一化因子：dif/dea 除以 (close * factor)
    mode90_macd_weight: float = 1.0  # 贴轴加分再乘此系数
    mode90_macd_max_bonus: float = 12.0  # s=0 时 MACD 最高加分
    mode90_macd_s_scale: float = 0.12  # s=DIF_norm+DEA_norm+HIST_norm 达到此值时 MACD 加分为 0

    # mode9/mode90：信号日全市场涨停行业 TopN 与本股行业一致时加分（0=关闭）
    mode9_hot_industry_bonus: int = 3
    mode9_hot_industry_top_n: int = 5
    # 涨停行业加分上限（含按该行业当日涨停家数追加的分，见 _score_mode9）
    mode9_hot_industry_bonus_max: int = 12
    # 信号日前 N 个交易日（含信号日）内，本股所属行业涨停家次累计：多则加分、0 则扣分（0=关闭）
    mode9_industry_limit_ndays: int = 0
    mode9_industry_ndays_penalty: int = 3
    mode9_industry_ndays_bonus_per_unit: int = 5  # 累计家次每满 per_unit 加 1 分
    mode9_industry_ndays_bonus_cap: int = 8

    # 东财：当日主力净流入TopN行业命中加分（需提前生成快照文件）
    em_industry_flow_top_n: int = 10
    em_industry_flow_bonus: int = 3

    # mode5：涨停锚点 + 缩量（相对涨停次日量）+ 涨停后至信号日低点≥MA10 + 信号日MA20向上 + 半年线之上
    mode5_shrink_max_days: int = 5
    mode5_half_year_bars: int = 120

    # mode93：低位(120日最低点出现在近10天) → 次日放量≥3倍且涨停 → 回调到涨停日最低价附近
    mode93_lookback_days: int = 20
    mode93_low_window: int = 120
    mode93_low_recent_days: int = 10
    mode93_vol_mult: float = 3.0
    mode93_pullback_min: float = 0.95
    mode93_pullback_max: float = 1.05
    mode93_pullback_max_days: int = 20

    # mode底部大阳线：低位区间 + 倍量大阳线 + 突发放量
    modebbd_low_lookback: int = 60
    modebbd_bottom_pos_max: float = 0.50
    modebbd_big_pct_min: float = 5.0
    modebbd_body_ratio_min: float = 0.55
    modebbd_vol_mult: float = 2.0
    modebbd_vol_ma: int = 20
    modebbd_sudden_days: int = 5
    modebbd_prior_vol_ratio_max: float = 0.65

    # mode平台突破首阳：约3个月吸筹震仓后，突破平台的第一根放量大阳线（买点）
    modepbs_phase_days_min: int = 45
    modepbs_phase_days_max: int = 95
    modepbs_rise_from_low_min: float = 0.20
    modepbs_rise_from_low_max: float = 0.55
    modepbs_consolid_days: int = 20
    modepbs_consolid_amp_max: float = 0.30
    modepbs_breakout_lookback: int = 60
    modepbs_breakout_near_min: float = 0.93  # 信号日最高 >= 近60日最高×该值（贴近或突破箱顶）
    modepbs_big_pct_min: float = 7.0  # 创业板/科创板等大阳涨幅下限
    modepbs_big_pct_min_main: float = 4.5  # 主板(10%板)大阳涨幅下限
    modepbs_body_ratio_min: float = 0.55
    modepbs_vol_mult: float = 1.25
    modepbs_vol_ma: int = 20
    modepbs_big_yang_gap: int = 15
    modepbs_gap_breakout_near_min: float = 0.93  # 前序大阳须贴顶(60日高比≥该值)才算占用首阳
    modepbs_high100_lookback: int = 100
    modepbs_high100_near_min: float = 0.93  # 信号日最高 >= 前100日最高×该值（贴近或刚突破100日新高）
    modepbs_vol_ratio_max: float = 4.0  # 量比上限，排除异常放量试探（0=不限）
    modepbs_vol_ratio_extended_max: float = 6.5  # 100日突破且震仓大阳≥3时允许更高量比
    modepbs_vol_high100_wash_min: int = 3
    modepbs_upper_ratio_max: float = 0.35  # 上影线/振幅上限，排除长上影假突破
    modepbs_upper_ratio_extended_max: float = 0.30  # 100日突破且量比≥4时放宽上影
    modepbs_upper_high100_vol_min: float = 4.0
    modepbs_wash_close_min_cnt: int = 2  # 震仓期大阳线≥该值时，要求收盘贴近箱顶
    modepbs_wash_close60_min: float = 0.98  # 上述情况下 close >= 近60日高×该值（100日突破可豁免）
    modepbs_pre_rise5_min: float = -0.05  # 信号前5日涨幅须 > 该值（默认-5%，排除急跌反弹）
    modepbs_pre_rise5_max: float = 0.10  # 信号前5日涨幅须 <= 该值，排除连涨追高（0=不限）
    modepbs_high_rise_wash_drop_rise_above: float = 0.38  # 自低点涨幅超该值时须有足够洗盘回撤
    modepbs_high_rise_wash_drop_min: float = 0.08  # 上述情况下震仓期峰值至低点回撤下限
    modepbs_weekly_conv_sig_max: float = 15.0  # 信号周5/10/20/30周均线拟合上限(%)，0=不限
    modepbs_weekly_conv_improve_min: float = -1.5  # 平台前半周拟合均值-后半周均值下限(%)，越大越要求后期收敛

    # mode中位大阳线：主力介入大阳(锚点)→吸筹震仓→突破大阳买点（参考埃科光电688610）
    mode_mby_anchor_days_min: int = 30
    mode_mby_anchor_days_max: int = 200
    mode_mby_anchor_vol_mult: float = 1.5
    mode_mby_rise_from_anchor_min: float = 0.20
    mode_mby_rise_from_anchor_max: float = 1.20
    mode_mby_consolid_days: int = 20
    mode_mby_consolid_amp_max: float = 0.35
    mode_mby_breakout_lookback: int = 60
    mode_mby_breakout_min: float = 1.0  # 信号日最高须严格突破近60日高
    mode_mby_high100_lookback: int = 100
    mode_mby_high100_min: float = 1.0
    mode_mby_tight_consolid_amp_max: float = 0.15  # 末段整理极窄时允许更早突破
    mode_mby_tight_vol_ratio_min: float = 1.8
    mode_mby_tight_rise_from_anchor_min: float = 0.10
    mode_mby_tight_high100_min: float = 0.985
    mode_mby_big_pct_min: float = 7.0
    mode_mby_big_pct_min_main: float = 4.5
    mode_mby_body_ratio_min: float = 0.55
    mode_mby_vol_mult: float = 1.25
    mode_mby_vol_ma: int = 20
    mode_mby_vol_ratio_max: float = 5.0
    mode_mby_upper_ratio_max: float = 0.40
    mode_mby_close_break60_min: float = 1.0
    mode_mby_pre_rise5_min: float = -0.05
    mode_mby_pre_rise5_max: float = 0.15  # 信号前5日涨幅须 <=15%，排除连板追高

    # mode底部支撑：底部起量大阳→拉升→回调至起量位获支撑→再拉升→再回调至支撑（抄底）
    mode_mbs_anchor_days_min: int = 30
    mode_mbs_anchor_days_max: int = 200
    mode_mbs_low_lookback: int = 60
    mode_mbs_bottom_pos_max: float = 0.50
    mode_mbs_anchor_vol_mult: float = 2.0
    mode_mbs_anchor_vol_ma: int = 20
    mode_mbs_big_pct_min: float = 5.0
    mode_mbs_body_ratio_min: float = 0.55
    mode_mbs_min_rally_pct: float = 0.15  # 锚点后须有明显拉升
    mode_mbs_support_near_max: float = 0.15  # 信号日低点距支撑 <=15%
    mode_mbs_support_break_min: float = 0.97  # 低点不可有效跌破支撑×该值
    mode_mbs_test_tol: float = 0.15  # 历史回踩容差
    mode_mbs_min_support_tests: int = 1  # 至少一次历史支撑验证
    mode_mbs_bounce_days: int = 5
    mode_mbs_weekly_vol_mult: float = 1.5  # 锚点周量 >= 该值×周均量

    # mode最后震仓：起量吸筹→箱体整理→最后震仓→反包/突破（参考金利华电300069）
    mode_mfs_phase_days_min: int = 30
    mode_mfs_phase_days_max: int = 90
    mode_mfs_anchor_vol_mult: float = 1.5
    mode_mfs_min_rally_pct: float = 0.10
    mode_mfs_consolid_days: int = 20
    mode_mfs_consolid_amp_max: float = 0.15
    mode_mfs_peak_lookback: int = 15
    mode_mfs_shakeout_days_min: int = 3
    mode_mfs_shakeout_days_max: int = 7
    mode_mfs_shakeout_drop_min: float = 0.10
    mode_mfs_shakeout_drop_max: float = 0.22
    mode_mfs_phase_low_lookback: int = 90
    mode_mfs_phase_low_break_min: float = 0.95
    mode_mfs_shakeout_vol_min: float = 0.6
    mode_mfs_shakeout_vol_max: float = 2.5
    mode_mfs_ma60_slope_days: int = 20
    mode_mfs_reversal_pct_min: float = 8.0
    mode_mfs_reversal_vol_min: float = 1.5
    mode_mfs_reversal_low_tol: float = 0.05
    mode_mfs_breakout_pct_min: float = 15.0
    mode_mfs_breakout_pct_min_main: float = 9.0
    mode_mfs_breakout_vol_min: float = 3.0
    mode_mfs_body_ratio_min: float = 0.55
    # 周线最后震仓：起量→2~4周缩量洗盘→放量反包/突破（参考688531/001259/300302等）
    mode_mfs_weekly_shakeout_weeks_min: int = 2
    mode_mfs_weekly_shakeout_weeks_max: int = 4
    mode_mfs_weekly_peak_weeks_back: int = 12
    mode_mfs_weekly_shakeout_drop_min: float = 0.08
    mode_mfs_weekly_shakeout_drop_max: float = 0.38
    mode_mfs_weekly_shakeout_vol_max: float = 1.05  # 洗盘周量 <= 周均量×该值
    mode_mfs_weekly_accum_lookback: int = 16
    mode_mfs_weekly_accum_vol_mult: float = 1.0
    mode_mfs_weekly_accum_pct_min: float = 3.0
    mode_mfs_weekly_signal_pct_min: float = 3.0
    mode_mfs_weekly_signal_strong_pct: float = 7.0
    mode_mfs_weekly_signal_vol_mult: float = 1.25
    mode_mfs_weekly_min_rally_pct: float = 0.08

    # mode98：日/周/月 KDJ（9,3,3）三线（K、D、J）均严格小于阈值
    mode98_kdj_threshold: float = 20.0
    mode98_kdj_n: int = 9
    mode98_kdj_m1: int = 3
    mode98_kdj_m2: int = 3

    # mode32（3+2）：实体首板后 5 日整理，信号日 = 首板后第 6 个交易日（尾盘上车语义）
    mode32_sideways_days: int = 60
    mode32_sideways_range_pct: float = 0.44  # (区间最高-最低)/区间均价；略放宽以覆盖震仓后首板（如000509）
    mode32_day1_body_max: float = 0.50  # 首板次日实体占振幅上限
    mode32_day1_vol_vs_limit_min: float = 1.0  # 次日量 ≥ 首板量 × 该值
    mode32_near_high_pct: float = 0.028  # 第2～3日收盘不低于 首板最高价×(1-该值)
    mode32_days23_low_min_frac: float = 0.97  # 第2～3日最低价不低于 首板最高价×该值
    mode32_day45_body_max: float = 0.95  # 第4～5日实体占振幅上限（允许末段震荡阴但仍缩量）
    mode32_vol_day43_vs_day3_max: float = 1.20  # 第4日量 ≤ 第3日量×该值（原1.08过严）
    mode32_vol_day5_vs_day4_max: float = 1.08  # 第5日量 ≤ 第4日量×该值
    mode32_vol_day45_vs_day1_max: float = 0.72  # 第4、5日量相对次日量的上限比例（低迷）
    mode32_min_close_vs_mid: float = 1.0  # 信号日收盘 ≥ 首板实体中轴×该值（1.0=不破中轴）
    # 信号日：MA120/MA250 向上但斜率平缓（参考000509@2026-05-15：5日斜率约0.16%/0.50%）
    mode32_ma_slope_days: int = 5
    mode32_ma120_slope_min_pct: float = 0.01  # 半年线须略向上（%）
    mode32_ma120_slope_max_pct: float = 0.55
    mode32_ma250_slope_min_pct: float = 0.01  # 年线须略向上（%）
    mode32_ma250_slope_max_pct: float = 1.05

    # mode33（锚试末洗）：锚点大阳吸筹→长震仓→中级反弹→试盘→末次放量震仓买点
    mode33_anchor_lookback: int = 300
    mode33_anchor_body_min: float = 0.32
    mode33_anchor_vol_mult: float = 1.20
    mode33_break_tol: float = 0.005
    mode33_shakeout_days_min: int = 55
    mode33_shakeout_days_max: int = 165
    mode33_sideways_range_pct: float = 0.62
    mode33_trial_lookback: int = 30
    mode33_trial_after_anchor_min: int = 35
    mode33_trial_box_pct: float = 0.10
    mode33_require_mid_rebound: bool = True
    mode33_mid_rebound_min_shake_frac: float = 0.28
    mode33_mid_rebound_min_rise_pct: float = 0.12
    mode33_mid_surge_pct_min: float = 5.5
    mode33_mid_surge_vol_mult: float = 1.25
    mode33_mid_min_days_before_trial: int = 8
    mode33_mid_pullback_min_pct: float = 0.06
    mode33_final_day_min: int = 6
    mode33_final_day_max: int = 20
    mode33_final_vol_mult: float = 1.50
    mode33_box_end_vol_min: float = 0.60
    mode33_box_end_vol_max: float = 1.40
    mode33_box_end_day_max: int = 7
    mode33_final_pct_max: float = 5.0
    mode33_final_body_max: float = 0.95
    mode33_ma_slope_days: int = 5
    mode33_ma120_slope_min_pct: float = 0.01
    mode33_ma120_slope_max_pct: float = 0.55
    mode33_ma250_slope_min_pct: float = 0.01
    mode33_ma250_slope_max_pct: float = 1.05
    mode33_vol_ma: int = 20

    # mode34（底部突破回踩二波）：阶段底→强阳突破→缩量平台→二波确认（参考600850@5/26）
    mode34_bottom_lookback: int = 60
    mode34_bottom_pos_max: float = 0.12
    mode34_surge_cum_pct_min: float = 12.0
    mode34_surge_big_pct_min: float = 7.0
    mode34_surge_big_pct_main: float = 4.5
    mode34_pullback_days_min: int = 2
    mode34_pullback_days_max: int = 8
    mode34_pullback_dd_max: float = 0.20
    mode34_signal_pct_min: float = 1.5
    mode34_min_score: int = 62

    # mode35（前高压顶洗盘突破）：前高锚点→压顶整理→A类放量突破
    mode35_min_score: int = 70

    # mode36（一阳穿多均线）：放量阳线一次穿越多条均线
    mode36_min_ma_cross: int = 6
    mode36_min_score: int = 60

    # mode37（跳空缺口支撑）：向上跳空未回补，回踩缺口区
    mode37_min_score: int = 60

    # mode38（大牛股关键位回踩）：大涨后回调踩 MA10/20/30/60/120
    mode38_min_score: int = 60

    # mode39（大阳锚点回踩再升）：放量大阳锚点 + 回踩企稳 / 长下影探底
    mode39_min_score: int = 60

    # mode40（新高回调踩60线回升）：阶段新高后回调7～10日触MA60回升
    mode40_min_score: int = 60

    # mode41（周线关键位回踩缩量）：周低点踩周均线 + 量能近5周最低附近
    mode41_min_score: int = 60

    # mode42（周线短均线回踩缩量回升）：踩周 MA5/10 + 当周阳线
    mode42_min_score: int = 60

    # mode43（周线爆量洗盘周）：主升途中放量分歧，收盘在周 MA10 之上
    mode43_min_score: int = 60

    # mode44（三连阴量价背离）：连续3阴、成交量逐日放大
    mode44_min_score: int = 60

    # mode45（涨停新高后缓升）：涨停/强阳放量创新高后横盘缓升
    mode45_min_score: int = 60

    # mode46（前高附近二次攻击）：回撤后再上攻贴近前高、收盘未突破
    mode46_min_score: int = 60


def _mode46_kw_from_config(config: ScanConfig) -> Dict[str, Any]:
    from app.mode46_prior_high_retest import mode46_kw_from_scan_config

    return mode46_kw_from_scan_config(config)


def _mode46_signal_at(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    **kwargs: Any,
) -> bool:
    from app.mode46_prior_high_retest import match_mode46_prior_high_retest

    return match_mode46_prior_high_retest(rows, idx, code, name, **kwargs) is not None


def _score_mode46(
    rows: List[KlineRow],
    idx: int,
    ma10: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
    vol20: np.ndarray,
    code: str = "",
    name: str = "",
    breakdown: Optional[List[tuple]] = None,
    **kwargs: Any,
) -> int:
    _ = (ma10, ma20, ma60, vol20, breakdown)
    from app.mode46_prior_high_retest import score_mode46_prior_high_retest

    return score_mode46_prior_high_retest(rows, idx, code, name, **kwargs)


def _mode46_metrics(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    from app.mode46_prior_high_retest import mode46_signal_metrics

    return mode46_signal_metrics(rows, idx, code, name, **kwargs)


def _mode45_kw_from_config(config: ScanConfig) -> Dict[str, Any]:
    from app.mode45_limitup_grind import mode45_kw_from_scan_config

    return mode45_kw_from_scan_config(config)


def _mode45_signal_at(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    **kwargs: Any,
) -> bool:
    from app.mode45_limitup_grind import match_mode45_limitup_grind

    return match_mode45_limitup_grind(rows, idx, code, name, **kwargs) is not None


def _score_mode45(
    rows: List[KlineRow],
    idx: int,
    ma10: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
    vol20: np.ndarray,
    code: str = "",
    name: str = "",
    breakdown: Optional[List[tuple]] = None,
    **kwargs: Any,
) -> int:
    _ = (ma10, ma20, ma60, vol20, breakdown)
    from app.mode45_limitup_grind import score_mode45_limitup_grind

    return score_mode45_limitup_grind(rows, idx, code, name, **kwargs)


def _mode45_metrics(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    from app.mode45_limitup_grind import mode45_signal_metrics

    return mode45_signal_metrics(rows, idx, code, name, **kwargs)


def _mode44_kw_from_config(config: ScanConfig) -> Dict[str, Any]:
    from app.mode44_triple_yin_vol_rise import mode44_kw_from_scan_config

    return mode44_kw_from_scan_config(config)


def _mode44_signal_at(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    **kwargs: Any,
) -> bool:
    from app.mode44_triple_yin_vol_rise import match_mode44_triple_yin_vol_rise

    return match_mode44_triple_yin_vol_rise(rows, idx, code, name, **kwargs) is not None


def _score_mode44(
    rows: List[KlineRow],
    idx: int,
    ma10: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
    vol20: np.ndarray,
    code: str = "",
    name: str = "",
    breakdown: Optional[List[tuple]] = None,
    **kwargs: Any,
) -> int:
    _ = (ma10, ma20, ma60, vol20, breakdown)
    from app.mode44_triple_yin_vol_rise import score_mode44_triple_yin_vol_rise

    return score_mode44_triple_yin_vol_rise(rows, idx, code, name, **kwargs)


def _mode44_metrics(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    from app.mode44_triple_yin_vol_rise import mode44_signal_metrics

    return mode44_signal_metrics(rows, idx, code, name, **kwargs)


def _mode43_kw_from_config(config: ScanConfig) -> Dict[str, Any]:
    from app.mode43_weekly_burst_churn import mode43_kw_from_scan_config

    return mode43_kw_from_scan_config(config)


def _mode43_signal_at(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    **kwargs: Any,
) -> bool:
    from app.mode43_weekly_burst_churn import match_mode43_weekly_burst_churn

    return match_mode43_weekly_burst_churn(rows, idx, code, name, **kwargs) is not None


def _score_mode43(
    rows: List[KlineRow],
    idx: int,
    ma10: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
    vol20: np.ndarray,
    code: str = "",
    name: str = "",
    breakdown: Optional[List[tuple]] = None,
    **kwargs: Any,
) -> int:
    _ = (ma10, ma20, ma60, vol20, breakdown)
    from app.mode43_weekly_burst_churn import score_mode43_weekly_burst_churn

    return score_mode43_weekly_burst_churn(rows, idx, code, name, **kwargs)


def _mode43_metrics(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    from app.mode43_weekly_burst_churn import mode43_signal_metrics

    return mode43_signal_metrics(rows, idx, code, name, **kwargs)


def _mode42_kw_from_config(config: ScanConfig) -> Dict[str, Any]:
    from app.mode42_weekly_short_ma_rebound import mode42_kw_from_scan_config

    return mode42_kw_from_scan_config(config)


def _mode42_signal_at(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    **kwargs: Any,
) -> bool:
    from app.mode42_weekly_short_ma_rebound import match_mode42_weekly_short_ma_rebound

    return match_mode42_weekly_short_ma_rebound(rows, idx, code, name, **kwargs) is not None


def _score_mode42(
    rows: List[KlineRow],
    idx: int,
    ma10: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
    vol20: np.ndarray,
    code: str = "",
    name: str = "",
    breakdown: Optional[List[tuple]] = None,
    **kwargs: Any,
) -> int:
    _ = (ma10, ma20, ma60, vol20, breakdown)
    from app.mode42_weekly_short_ma_rebound import score_mode42_weekly_short_ma_rebound

    return score_mode42_weekly_short_ma_rebound(rows, idx, code, name, **kwargs)


def _mode42_metrics(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    from app.mode42_weekly_short_ma_rebound import mode42_signal_metrics

    return mode42_signal_metrics(rows, idx, code, name, **kwargs)


def _mode41_kw_from_config(config: ScanConfig) -> Dict[str, Any]:
    from app.mode41_weekly_ma_pullback import mode41_kw_from_scan_config

    return mode41_kw_from_scan_config(config)


def _mode41_signal_at(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    **kwargs: Any,
) -> bool:
    from app.mode41_weekly_ma_pullback import match_mode41_weekly_ma_pullback

    return match_mode41_weekly_ma_pullback(rows, idx, code, name, **kwargs) is not None


def _score_mode41(
    rows: List[KlineRow],
    idx: int,
    ma10: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
    vol20: np.ndarray,
    code: str = "",
    name: str = "",
    breakdown: Optional[List[tuple]] = None,
    **kwargs: Any,
) -> int:
    _ = (ma10, ma20, ma60, vol20, breakdown)
    from app.mode41_weekly_ma_pullback import score_mode41_weekly_ma_pullback

    return score_mode41_weekly_ma_pullback(rows, idx, code, name, **kwargs)


def _mode41_metrics(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    from app.mode41_weekly_ma_pullback import mode41_signal_metrics

    return mode41_signal_metrics(rows, idx, code, name, **kwargs)


def _mode40_kw_from_config(config: ScanConfig) -> Dict[str, Any]:
    from app.mode40_high_pullback_ma60_rebound import mode40_kw_from_scan_config

    return mode40_kw_from_scan_config(config)


def _mode40_signal_at(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    **kwargs: Any,
) -> bool:
    from app.mode40_high_pullback_ma60_rebound import match_mode40_high_pullback_ma60_rebound

    return match_mode40_high_pullback_ma60_rebound(rows, idx, code, name, **kwargs) is not None


def _score_mode40(
    rows: List[KlineRow],
    idx: int,
    ma10: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
    vol20: np.ndarray,
    code: str = "",
    name: str = "",
    breakdown: Optional[List[tuple]] = None,
    **kwargs: Any,
) -> int:
    _ = (ma10, ma20, ma60, vol20, breakdown)
    from app.mode40_high_pullback_ma60_rebound import score_mode40_high_pullback_ma60_rebound

    return score_mode40_high_pullback_ma60_rebound(rows, idx, code, name, **kwargs)


def _mode40_metrics(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    from app.mode40_high_pullback_ma60_rebound import mode40_signal_metrics

    return mode40_signal_metrics(rows, idx, code, name, **kwargs)


def _mode38_kw_from_config(config: ScanConfig) -> Dict[str, Any]:
    from app.mode38_bull_ma_pullback import mode38_kw_from_scan_config

    return mode38_kw_from_scan_config(config)


def _mode38_signal_at(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    **kwargs: Any,
) -> bool:
    from app.mode38_bull_ma_pullback import match_mode38_bull_ma_pullback

    return match_mode38_bull_ma_pullback(rows, idx, code, name, **kwargs) is not None


def _score_mode38(
    rows: List[KlineRow],
    idx: int,
    ma10: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
    vol20: np.ndarray,
    code: str = "",
    name: str = "",
    breakdown: Optional[List[tuple]] = None,
    **kwargs: Any,
) -> int:
    _ = (ma10, ma20, ma60, vol20, breakdown)
    from app.mode38_bull_ma_pullback import score_mode38_bull_ma_pullback

    return score_mode38_bull_ma_pullback(rows, idx, code, name, **kwargs)


def _mode38_metrics(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    from app.mode38_bull_ma_pullback import mode38_signal_metrics

    return mode38_signal_metrics(rows, idx, code, name, **kwargs)


def _mode39_kw_from_config(config: ScanConfig) -> Dict[str, Any]:
    from app.mode39_bull_anchor_pullback import mode39_kw_from_scan_config

    return mode39_kw_from_scan_config(config)


def _mode39_signal_at(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    **kwargs: Any,
) -> bool:
    from app.mode39_bull_anchor_pullback import match_mode39_bull_anchor_pullback

    return match_mode39_bull_anchor_pullback(rows, idx, code, name, **kwargs) is not None


def _score_mode39(
    rows: List[KlineRow],
    idx: int,
    ma10: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
    vol20: np.ndarray,
    code: str = "",
    name: str = "",
    breakdown: Optional[List[tuple]] = None,
    **kwargs: Any,
) -> int:
    _ = (ma10, ma20, ma60, vol20, breakdown)
    from app.mode39_bull_anchor_pullback import score_mode39_bull_anchor_pullback

    return score_mode39_bull_anchor_pullback(rows, idx, code, name, **kwargs)


def _mode39_metrics(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    from app.mode39_bull_anchor_pullback import mode39_signal_metrics

    return mode39_signal_metrics(rows, idx, code, name, **kwargs)


def _mode37_kw_from_config(config: ScanConfig) -> Dict[str, Any]:
    from app.mode37_gap_support import mode37_kw_from_scan_config

    return mode37_kw_from_scan_config(config)


def _mode37_signal_at(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    **kwargs: Any,
) -> bool:
    from app.mode37_gap_support import match_mode37_gap_support

    return match_mode37_gap_support(rows, idx, code, name, **kwargs) is not None


def _score_mode37(
    rows: List[KlineRow],
    idx: int,
    ma10: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
    vol20: np.ndarray,
    code: str = "",
    name: str = "",
    breakdown: Optional[List[tuple]] = None,
    **kwargs: Any,
) -> int:
    _ = (ma10, ma20, ma60, vol20, breakdown)
    from app.mode37_gap_support import score_mode37_gap_support

    return score_mode37_gap_support(rows, idx, code, name, **kwargs)


def _mode37_metrics(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    from app.mode37_gap_support import mode37_signal_metrics

    return mode37_signal_metrics(rows, idx, code, name, **kwargs)


def _mode36_kw_from_config(config: ScanConfig) -> Dict[str, Any]:
    from app.mode36_yang_cross_ma import mode36_kw_from_scan_config

    return mode36_kw_from_scan_config(config)


def _mode36_signal_at(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    **kwargs: Any,
) -> bool:
    from app.mode36_yang_cross_ma import match_mode36_yang_cross_ma

    return match_mode36_yang_cross_ma(rows, idx, code, name, **kwargs) is not None


def _score_mode36(
    rows: List[KlineRow],
    idx: int,
    ma10: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
    vol20: np.ndarray,
    code: str = "",
    name: str = "",
    breakdown: Optional[List[tuple]] = None,
    **kwargs: Any,
) -> int:
    _ = (ma10, ma20, ma60, vol20, breakdown)
    from app.mode36_yang_cross_ma import score_mode36_yang_cross_ma

    return score_mode36_yang_cross_ma(rows, idx, code, name, **kwargs)


def _mode36_metrics(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    from app.mode36_yang_cross_ma import mode36_signal_metrics

    return mode36_signal_metrics(rows, idx, code, name, **kwargs)


def _mode35_kw_from_config(config: ScanConfig) -> Dict[str, Any]:
    from app.mode35_prior_high_breakout import mode35_kw_from_scan_config

    return mode35_kw_from_scan_config(config)


def _mode35_signal_at(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    **kwargs: Any,
) -> bool:
    from app.mode35_prior_high_breakout import match_mode35_prior_high_breakout

    return match_mode35_prior_high_breakout(rows, idx, code, name, **kwargs) is not None


def _score_mode35(
    rows: List[KlineRow],
    idx: int,
    ma10: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
    vol20: np.ndarray,
    code: str = "",
    name: str = "",
    breakdown: Optional[List[tuple]] = None,
    **kwargs: Any,
) -> int:
    _ = (ma10, ma20, ma60, vol20, breakdown)
    from app.mode35_prior_high_breakout import score_mode35_prior_high_breakout

    return score_mode35_prior_high_breakout(rows, idx, code, name, **kwargs)


def _mode35_metrics(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    from app.mode35_prior_high_breakout import mode35_signal_metrics

    return mode35_signal_metrics(rows, idx, code, name, **kwargs)


def _mode34_kw_from_config(config: ScanConfig) -> Dict[str, Any]:
    from app.mode34_bottom_break_pullback import mode34_kw_from_scan_config

    return mode34_kw_from_scan_config(config)


def _mode34_signal_at(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    **kwargs: Any,
) -> bool:
    from app.mode34_bottom_break_pullback import (
        match_mode34_prebuy_signal,
        match_mode34_watchlist,
    )

    return (
        match_mode34_prebuy_signal(rows, idx, code, name, **kwargs) is not None
        or match_mode34_watchlist(rows, idx, code, name, **kwargs) is not None
    )


def _score_mode34(
    rows: List[KlineRow],
    idx: int,
    ma10: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
    vol20: np.ndarray,
    code: str = "",
    name: str = "",
    breakdown: Optional[List[tuple]] = None,
    **kwargs: Any,
) -> int:
    _ = (ma10, ma20, ma60, vol20, breakdown)
    from app.mode34_bottom_break_pullback import (
        score_mode34_prebuy_signal,
        score_mode34_watchlist,
    )

    pb = score_mode34_prebuy_signal(rows, idx, code, name, **kwargs)
    if pb > 0:
        return pb
    return score_mode34_watchlist(rows, idx, code, name, **kwargs)


def _mode34_metrics(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    from app.mode34_bottom_break_pullback import (
        match_mode34_prebuy_signal,
        mode34_prebuy_signal_metrics,
        mode34_watch_signal_metrics,
    )

    if match_mode34_prebuy_signal(rows, idx, code, name, **kwargs):
        return mode34_prebuy_signal_metrics(rows, idx, code, name, **kwargs)
    return mode34_watch_signal_metrics(rows, idx, code, name, **kwargs)


def _normalize_code(code: str) -> str:
    value = str(code or "").strip()
    if value.isdigit() and len(value) < 6:
        return value.zfill(6)
    return value


def _is_st(name: str) -> bool:
    if not name:
        return True
    return "ST" in name or name.startswith("*ST") or name.startswith("退")


def _to_array(rows: List[KlineRow]) -> Dict[str, np.ndarray]:
    return {
        "close": np.array([r.close for r in rows], dtype=float),
        "high": np.array([r.high for r in rows], dtype=float),
        "low": np.array([r.low for r in rows], dtype=float),
        "volume": np.array([r.volume for r in rows], dtype=float),
        "pct": np.array([r.pct_chg for r in rows], dtype=float),
    }


def _rolling_mean(data: np.ndarray, window: int) -> np.ndarray:
    if len(data) < window:
        return np.array([])
    return np.convolve(data, np.ones(window) / window, mode="valid")


def _moving_mean(values: np.ndarray, window: int) -> np.ndarray:
    res = np.full_like(values, np.nan, dtype=float)
    if len(values) < window:
        return res
    weights = np.ones(window, dtype=float) / window
    res[window - 1 :] = np.convolve(values, weights, mode="valid")
    return res


def _ema_series(close: np.ndarray, n: int) -> np.ndarray:
    """指数移动平均，用于周线 MACD。"""
    out = np.full_like(close, np.nan, dtype=float)
    if len(close) < n or n <= 0:
        return out
    alpha = 2.0 / (n + 1)
    out[n - 1] = float(np.nanmean(close[:n]))
    for i in range(n, len(close)):
        if np.isnan(close[i]):
            continue
        if np.isnan(out[i - 1]):
            out[i] = close[i]
        else:
            out[i] = alpha * close[i] + (1.0 - alpha) * out[i - 1]
    return out


def _weekly_macd_dif_dea(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """周线 MACD：DIF = EMA(close,12)-EMA(close,26)，DEA = EMA(DIF,9)。
    返回 (dif_norm, dea_norm)：归一化值 = 原值 / 当周收盘价，便于跨股票、跨价位比较。"""
    ema_fast = _ema_series(close, fast)
    ema_slow = _ema_series(close, slow)
    dif = np.full_like(close, np.nan, dtype=float)
    for i in range(slow - 1, len(close)):
        if not (np.isnan(ema_fast[i]) or np.isnan(ema_slow[i])):
            dif[i] = ema_fast[i] - ema_slow[i]
    dea = np.full_like(close, np.nan, dtype=float)
    alpha_sig = 2.0 / (signal + 1)
    start = slow - 1
    if start < len(close) and not np.isnan(dif[start]):
        dea[start] = dif[start]
        for i in range(start + 1, len(close)):
            if np.isnan(dif[i]):
                continue
            dea[i] = alpha_sig * dif[i] + (1.0 - alpha_sig) * dea[i - 1]
    dif_norm = np.full_like(close, np.nan, dtype=float)
    dea_norm = np.full_like(close, np.nan, dtype=float)
    for i in range(len(close)):
        if close[i] > 0:
            if not np.isnan(dif[i]):
                dif_norm[i] = dif[i] / close[i]
            if not np.isnan(dea[i]):
                dea_norm[i] = dea[i] / close[i]
    return dif_norm, dea_norm


def _daily_macd_dif_dea(
    close: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    norm_factor: float = 1.0,
) -> tuple:
    """日线 MACD：DIF = EMA(close,fast)-EMA(close,slow)，DEA = EMA(DIF,signal)。
    返回 (dif_norm, dea_norm)：归一化值 = 原值 / (close * norm_factor)，便于跨价位比较。"""
    ema_fast = _ema_series(close, fast)
    ema_slow = _ema_series(close, slow)
    dif = np.full_like(close, np.nan, dtype=float)
    for i in range(len(close)):
        if not (np.isnan(ema_fast[i]) or np.isnan(ema_slow[i])):
            dif[i] = ema_fast[i] - ema_slow[i]

    # DEA = EMA(DIF, signal)
    dea = np.full_like(close, np.nan, dtype=float)
    alpha_sig = 2.0 / (signal + 1)
    start = max(slow - 1, 0)
    if start < len(close) and not np.isnan(dif[start]):
        dea[start] = dif[start]
        for i in range(start + 1, len(close)):
            if np.isnan(dif[i]) or np.isnan(dea[i - 1]):
                continue
            dea[i] = alpha_sig * dif[i] + (1.0 - alpha_sig) * dea[i - 1]

    dif_norm = np.full_like(close, np.nan, dtype=float)
    dea_norm = np.full_like(close, np.nan, dtype=float)
    for i in range(len(close)):
        denom = close[i] * (norm_factor if norm_factor and norm_factor > 0 else 1.0)
        if denom > 0:
            if not np.isnan(dif[i]):
                dif_norm[i] = dif[i] / denom
            if not np.isnan(dea[i]):
                dea_norm[i] = dea[i] / denom
    return dif_norm, dea_norm


def _mode18_signals(
    rows: List[KlineRow],
    start_date: Optional[str],
    end_date: Optional[str],
) -> List[int]:
    """
    mode18 信号：周线 MACD 金叉买入。条件：DIF 上穿 DEA（金叉），且 DIF、DEA 均在 0 轴以上（MACD 值由负变正）。
    信号日 = 当周最后交易日，买点 = 下一交易日开盘。
    """
    if not rows or len(rows) < 200:
        return []
    weekly_bars, last_indices = daily_to_weekly_with_last_index(rows)
    if len(weekly_bars) < 35:
        return []
    closes = np.array([w[4] for w in weekly_bars], dtype=float)
    dif_norm, dea_norm = _weekly_macd_dif_dea(closes, 12, 26, 9)
    signal_indices = []
    for i in range(34, len(weekly_bars)):
        if np.isnan(dif_norm[i]) or np.isnan(dea_norm[i]) or np.isnan(dif_norm[i - 1]) or np.isnan(dea_norm[i - 1]):
            continue
        if dif_norm[i] <= 0 or dea_norm[i] <= 0:
            continue
        if not (dif_norm[i] > dea_norm[i] and dif_norm[i - 1] <= dea_norm[i - 1]):
            continue
        idx = last_indices[i]
        if idx >= len(rows):
            continue
        d = rows[idx].date
        if start_date and d < start_date:
            continue
        if end_date and d > end_date:
            continue
        signal_indices.append(idx)
    return sorted(signal_indices)


def _score_mode18(
    rows: List[KlineRow],
    idx: int,
    ma10: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
    vol20: np.ndarray,
    code: str = "",
    name: str = "",
) -> int:
    """mode18 评分：以当周归一化 DIF（DIF/收盘价）为权重，可比 across 股票，50～100。"""
    if idx < 0 or idx >= len(rows):
        return 50
    sub = rows[: idx + 1]
    if len(sub) < 200:
        return 50
    weekly_bars, _ = daily_to_weekly_with_last_index(sub)
    if len(weekly_bars) < 35:
        return 50
    closes = np.array([w[4] for w in weekly_bars], dtype=float)
    dif_norm, dea_norm = _weekly_macd_dif_dea(closes, 12, 26, 9)
    wi = len(weekly_bars) - 1
    if np.isnan(dif_norm[wi]) or dif_norm[wi] <= 0:
        return 50
    score = 50 + min(50, dif_norm[wi] * 2500)
    return int(max(50, min(100, round(score))))


def _mode98_kdj_triplet_ok(
    k: np.ndarray,
    d: np.ndarray,
    j: np.ndarray,
    i: int,
    threshold: float,
) -> bool:
    if i < 0 or i >= len(k):
        return False
    if np.isnan(k[i]) or np.isnan(d[i]) or np.isnan(j[i]):
        return False
    return k[i] < threshold and d[i] < threshold and j[i] < threshold


def _mode98_signals(
    rows: List[KlineRow],
    start_date: Optional[str],
    end_date: Optional[str],
    threshold: float = 20.0,
    n: int = 9,
    m1: int = 3,
    m2: int = 3,
) -> List[int]:
    """
    mode98：信号日当日，日线、周线、月线 KDJ（参数 n,m1,m2，默认 9,3,3）的 K、D、J 均 < threshold（默认 20）。
    周线/月线按信号日为止的历史聚合（含未走完的当周、当月）。
    """
    if not rows or len(rows) < n:
        return []
    daily_bars = [(r.date, r.open, r.high, r.low, r.close) for r in rows]
    kd, dd, jd = weekly_kdj(daily_bars, n=n, m1=m1, m2=m2)
    if kd.size == 0:
        return []
    out: List[int] = []
    for idx in range(n - 1, len(rows)):
        d_str = rows[idx].date
        if start_date and d_str < start_date:
            continue
        if end_date and d_str > end_date:
            continue
        if not _mode98_kdj_triplet_ok(kd, dd, jd, idx, threshold):
            continue
        sub = rows[: idx + 1]
        wb, _ = daily_to_weekly_with_last_index(sub)
        if len(wb) < n:
            continue
        kw, dw, jw = weekly_kdj(wb, n=n, m1=m1, m2=m2)
        wi = len(wb) - 1
        if not _mode98_kdj_triplet_ok(kw, dw, jw, wi, threshold):
            continue
        mb, _ = daily_to_monthly_with_last_index(sub)
        if len(mb) < n:
            continue
        km, dm, jm = weekly_kdj(mb, n=n, m1=m1, m2=m2)
        mi = len(mb) - 1
        if not _mode98_kdj_triplet_ok(km, dm, jm, mi, threshold):
            continue
        out.append(idx)
    return out


def _mode98_kdj_metrics(
    rows: List[KlineRow],
    idx: int,
    n: int = 9,
    m1: int = 3,
    m2: int = 3,
) -> Dict[str, Any]:
    """信号日 K/D/J（日、周、月），供 ScanResult.metrics。"""
    out: Dict[str, Any] = {}
    if idx < 0 or idx >= len(rows):
        return out
    sub = rows[: idx + 1]
    daily_bars = [(r.date, r.open, r.high, r.low, r.close) for r in sub]
    kd, dd, jd = weekly_kdj(daily_bars, n=n, m1=m1, m2=m2)
    wb, _ = daily_to_weekly_with_last_index(sub)
    kw, dw, jw = (
        weekly_kdj(wb, n=n, m1=m1, m2=m2) if len(wb) >= n else (np.array([]), np.array([]), np.array([]))
    )
    mb, _ = daily_to_monthly_with_last_index(sub)
    km, dm, jm = (
        weekly_kdj(mb, n=n, m1=m1, m2=m2) if len(mb) >= n else (np.array([]), np.array([]), np.array([]))
    )
    di = idx
    wi = len(wb) - 1 if wb else -1
    mi = len(mb) - 1 if mb else -1

    if kd.size > di:
        for arr, letter in ((kd, "K"), (dd, "D"), (jd, "J")):
            if arr.size > di and not np.isnan(arr[di]):
                out[f"mode98_daily_{letter}"] = float(arr[di])
    if wi >= 0 and kw.size > wi:
        for arr, letter in ((kw, "K"), (dw, "D"), (jw, "J")):
            if arr.size > wi and not np.isnan(arr[wi]):
                out[f"mode98_weekly_{letter}"] = float(arr[wi])
    if mi >= 0 and km.size > mi:
        for arr, letter in ((km, "K"), (dm, "D"), (jm, "J")):
            if arr.size > mi and not np.isnan(arr[mi]):
                out[f"mode98_monthly_{letter}"] = float(arr[mi])
    return out


def _score_mode98(
    rows: List[KlineRow],
    idx: int,
    ma10: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
    vol20: np.ndarray,
    code: str = "",
    name: str = "",
    breakdown: Optional[List[tuple]] = None,
    threshold: float = 20.0,
    n: int = 9,
    m1: int = 3,
    m2: int = 3,
) -> int:
    """mode98 评分：三线距阈值越远（超卖越深）分越高，约 55～95。"""
    if idx < 0 or idx >= len(rows):
        return 55
    sub = rows[: idx + 1]
    daily_bars = [(r.date, r.open, r.high, r.low, r.close) for r in sub]
    kd, dd, jd = weekly_kdj(daily_bars, n=n, m1=m1, m2=m2)
    wb, _ = daily_to_weekly_with_last_index(sub)
    kw, dw, jw = weekly_kdj(wb, n=n, m1=m1, m2=m2)
    mb, _ = daily_to_monthly_with_last_index(sub)
    km, dm, jm = weekly_kdj(mb, n=n, m1=m1, m2=m2)
    wi, mi = len(wb) - 1, len(mb) - 1
    vals = []
    for arr, i in (
        (kd, idx),
        (dd, idx),
        (jd, idx),
        (kw, wi),
        (dw, wi),
        (jw, wi),
        (km, mi),
        (dm, mi),
        (jm, mi),
    ):
        if arr.size <= i or i < 0:
            return 55
        v = arr[i]
        if np.isnan(v):
            return 55
        vals.append(float(v))
    peak = max(vals)
    if peak >= threshold:
        return 55
    room = threshold - peak
    score = 55.0 + min(40.0, room * 3.0)
    return int(max(55, min(95, round(score))))


def _mode32_row_body_ratio(row: KlineRow) -> float:
    rng = float(row.high) - float(row.low)
    if rng <= 1e-12:
        return 1.0
    return abs(float(row.close) - float(row.open)) / rng


def _mode32_is_yizi(rows: List[KlineRow], t: int, prev_close: float) -> bool:
    if prev_close <= 0:
        return True
    return (float(rows[t].high) - float(rows[t].low)) / prev_close < 0.005


def _mode32_is_t_board(rows: List[KlineRow], t: int, prev_close: float, rate: float) -> bool:
    """近似 T 字：高开近涨停且下影线占振幅比例大。"""
    row = rows[t]
    o, h, l, c = float(row.open), float(row.high), float(row.low), float(row.close)
    rng = h - l
    if rng <= 1e-12:
        return False
    lim_ref = prev_close * (1.0 + rate)
    near_top_open = o >= max(h - max(0.003 * prev_close, 0.02), lim_ref * 0.988)
    lower_shadow = (min(o, c) - l) / rng
    return near_top_open and lower_shadow >= 0.34


def _mode32_solid_limit_ok(
    rows: List[KlineRow],
    t: int,
    code: str,
    name: str,
    *,
    body_min: float = 0.32,
) -> bool:
    if t < 1 or not _limit_up_day(rows, t, code, name):
        return False
    prev_close = float(rows[t - 1].close)
    if prev_close <= 0:
        return False
    rate = _limit_rate(code, name)
    if _mode32_is_yizi(rows, t, prev_close):
        return False
    if _mode32_is_t_board(rows, t, prev_close, rate):
        return False
    row = rows[t]
    h, c = float(row.high), float(row.close)
    br = _mode32_row_body_ratio(row)
    closed_near_high = c >= h - max(0.004 * prev_close, 0.02)
    return br >= body_min and closed_near_high


def _mode32_sideways_ok(
    rows: List[KlineRow],
    t: int,
    sideways_days: int,
    sideways_range_pct: float,
) -> bool:
    if t < sideways_days + 1:
        return False
    lo = t - sideways_days
    hi = t - 1
    highs = [float(rows[i].high) for i in range(lo, hi + 1)]
    lows = [float(rows[i].low) for i in range(lo, hi + 1)]
    closes = [float(rows[i].close) for i in range(lo, hi + 1)]
    mx, mn = max(highs), min(lows)
    mean_c = sum(closes) / max(1, len(closes))
    if mean_c <= 0:
        return False
    return (mx - mn) / mean_c <= sideways_range_pct


def _mode32_ma_slope_pct(
    closes: np.ndarray,
    idx: int,
    ma_window: int,
    slope_days: int,
) -> Optional[float]:
    """MA 在 slope_days 内的涨幅（%），用于「向上但近乎走平」。"""
    need = ma_window - 1 + slope_days
    if idx < need:
        return None
    ma = _moving_mean(closes, ma_window)
    if np.isnan(ma[idx]) or np.isnan(ma[idx - slope_days]):
        return None
    prev = float(ma[idx - slope_days])
    if prev <= 0:
        return None
    return (float(ma[idx]) - prev) / prev * 100.0


def _mode32_long_ma_flat_up_ok(
    rows: List[KlineRow],
    idx: int,
    *,
    ma120_window: int = 120,
    ma250_window: int = 250,
    slope_days: int = 5,
    ma120_slope_min_pct: float = 0.01,
    ma120_slope_max_pct: float = 0.55,
    ma250_slope_min_pct: float = 0.01,
    ma250_slope_max_pct: float = 1.05,
) -> bool:
    closes = np.array([float(r.close) for r in rows], dtype=float)
    s120 = _mode32_ma_slope_pct(closes, idx, ma120_window, slope_days)
    s250 = _mode32_ma_slope_pct(closes, idx, ma250_window, slope_days)
    if s120 is None or s250 is None:
        return False
    return (
        ma120_slope_min_pct <= s120 <= ma120_slope_max_pct
        and ma250_slope_min_pct <= s250 <= ma250_slope_max_pct
    )


def _mode32_long_ma_slopes(
    rows: List[KlineRow],
    idx: int,
    slope_days: int = 5,
) -> Tuple[Optional[float], Optional[float]]:
    closes = np.array([float(r.close) for r in rows], dtype=float)
    return (
        _mode32_ma_slope_pct(closes, idx, 120, slope_days),
        _mode32_ma_slope_pct(closes, idx, 250, slope_days),
    )


def _mode32_signal_at(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    sideways_days: int = 60,
    sideways_range_pct: float = 0.42,
    day1_body_max: float = 0.50,
    day1_vol_vs_limit_min: float = 1.0,
    near_high_pct: float = 0.028,
    days23_low_min_frac: float = 0.97,
    day45_body_max: float = 0.95,
    vol_day43_vs_day3_max: float = 1.20,
    vol_day5_vs_day4_max: float = 1.08,
    vol_day45_vs_day1_max: float = 0.72,
    min_close_vs_mid: float = 1.0,
    ma_slope_days: int = 5,
    ma120_slope_min_pct: float = 0.01,
    ma120_slope_max_pct: float = 0.55,
    ma250_slope_min_pct: float = 0.01,
    ma250_slope_max_pct: float = 1.05,
) -> bool:
    """
    信号日在 idx = T+6（首板日 T，其后 5 日为整理），且 ST、一字、T 字板已剔除。
    """
    if _is_st(name or ""):
        return False
    T = idx - 6
    if T < 1 or idx >= len(rows):
        return False
    if not _mode32_sideways_ok(rows, T, sideways_days, sideways_range_pct):
        return False
    if not _mode32_solid_limit_ok(rows, T, code, name):
        return False

    H0 = float(rows[T].high)
    O0, C0 = float(rows[T].open), float(rows[T].close)
    mid = 0.5 * (O0 + C0)
    Vlim = float(rows[T].volume)

    # Day1 = T+1
    d1 = rows[T + 1]
    if Vlim <= 0 or float(d1.volume) + 1e-9 < Vlim * day1_vol_vs_limit_min:
        return False
    if _mode32_row_body_ratio(d1) > day1_body_max:
        return False

    # Days 2–3：缩量梯形 + 收盘贴近首板高、低点不破过多
    v1 = float(rows[T + 1].volume)
    v2 = float(rows[T + 2].volume)
    v3 = float(rows[T + 3].volume)
    if not (v2 < v1 and v3 < v2):
        return False
    band_low = H0 * (1.0 - near_high_pct)
    floor_low = H0 * days23_low_min_frac
    for j in (T + 2, T + 3):
        rj = rows[j]
        if float(rj.close) < band_low:
            return False
        if float(rj.low) < floor_low:
            return False

    # Days 4–5：小实体 + 量能低迷
    v4 = float(rows[T + 4].volume)
    v5 = float(rows[T + 5].volume)
    if not (v4 <= v3 * vol_day43_vs_day3_max + 1e-9 and v5 <= v4 * vol_day5_vs_day4_max + 1e-9):
        return False
    if v4 > v1 * vol_day45_vs_day1_max + 1e-9 or v5 > v1 * vol_day45_vs_day1_max + 1e-9:
        return False
    for j in (T + 4, T + 5):
        if _mode32_row_body_ratio(rows[j]) > day45_body_max:
            return False

    # 信号日收盘仍在中轴之上（防守位语义）
    if float(rows[idx].close) + 1e-9 < mid * min_close_vs_mid:
        return False

    # 整理五日最低价不破首板实体中轴过多（承接仍有效）
    min_low_5 = min(float(rows[j].low) for j in range(T + 1, T + 6))
    if min_low_5 + 1e-9 < mid * 0.98:
        return False

    if not _mode32_long_ma_flat_up_ok(
        rows,
        idx,
        slope_days=ma_slope_days,
        ma120_slope_min_pct=ma120_slope_min_pct,
        ma120_slope_max_pct=ma120_slope_max_pct,
        ma250_slope_min_pct=ma250_slope_min_pct,
        ma250_slope_max_pct=ma250_slope_max_pct,
    ):
        return False

    return True


def _mode32_metrics(
    rows: List[KlineRow],
    idx: int,
) -> Dict[str, Any]:
    T = idx - 6
    H0 = float(rows[T].high)
    O0, C0 = float(rows[T].open), float(rows[T].close)
    mid = 0.5 * (O0 + C0)
    s120, s250 = _mode32_long_ma_slopes(rows, idx)
    out = {
        "mode32_limit_date": rows[T].date,
        "mode32_limit_high": round(H0, 4),
        "mode32_mid_stop": round(mid, 4),
        "mode32_vol_day1_vs_limit": round(float(rows[T + 1].volume) / max(float(rows[T].volume), 1e-9), 4),
        "mode32_vol_day5_vs_day1": round(float(rows[T + 5].volume) / max(float(rows[T + 1].volume), 1e-9), 4),
    }
    if s120 is not None:
        out["mode32_ma120_slope5_pct"] = round(s120, 4)
    if s250 is not None:
        out["mode32_ma250_slope5_pct"] = round(s250, 4)
    return out


def _score_mode32(
    rows: List[KlineRow],
    idx: int,
    ma10: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
    vol20: np.ndarray,
    code: str = "",
    name: str = "",
    breakdown: Optional[List[tuple]] = None,
) -> int:
    """mode32 评分：横盘越紧、量缩得越干净分略高，约 62～92。"""
    if idx < 6 or idx >= len(rows):
        return 62
    T = idx - 6
    prev_seg = rows[max(0, T - 60) : T]
    if len(prev_seg) < 30:
        return 62
    closes = np.array([float(r.close) for r in prev_seg], dtype=float)
    highs = np.array([float(r.high) for r in prev_seg], dtype=float)
    lows = np.array([float(r.low) for r in prev_seg], dtype=float)
    mean_c = float(np.mean(closes))
    if mean_c <= 0:
        return 62
    tight = (float(np.max(highs)) - float(np.min(lows))) / mean_c
    # tight 越小越好：0.25→加分多，0.4→少
    bonus_tight = max(0.0, min(22.0, (0.45 - tight) * 80.0))
    v1 = float(rows[T + 1].volume)
    v5 = float(rows[T + 5].volume)
    shrink = v5 / max(v1, 1e-9)
    bonus_vol = max(0.0, min(18.0, (0.65 - shrink) * 50.0))
    score = 62.0 + bonus_tight + bonus_vol
    return int(max(62, min(92, round(score))))


def _mode33_anchor_low(rows: List[KlineRow], t: int) -> float:
    return min(float(rows[t].open), float(rows[t].low))


def _mode33_find_anchor(
    rows: List[KlineRow],
    trial_i: int,
    code: str,
    name: str,
    *,
    anchor_lookback: int,
    trial_after_anchor_min: int,
    anchor_body_min: float,
    shakeout_days_min: int = 55,
    shakeout_days_max: int = 165,
    break_tol: float = 0.005,
    vol_ma: int = 20,
    anchor_vol_mult: float = 1.20,
) -> Optional[int]:
    """锚点大阳：震仓期不破 L0，在候选中取吸筹量更强、铁底更牢的一根（非简单最早）。"""
    lo = max(1, trial_i - anchor_lookback)
    hi = trial_i - trial_after_anchor_min
    if hi < lo:
        return None
    best_t: Optional[int] = None
    best_score = -1e18
    for t in range(lo, hi + 1):
        if not _mode32_solid_limit_ok(rows, t, code, name, body_min=anchor_body_min):
            continue
        shake = trial_i - t - 1
        if shake < shakeout_days_min or shake > shakeout_days_max:
            continue
        vr = _vol_ratio_at(rows, t, vol_ma)
        if vr + 1e-9 < anchor_vol_mult:
            continue
        L0 = _mode33_anchor_low(rows, t)
        if L0 <= 0:
            continue
        shake_lo, shake_hi = t + 1, trial_i - 1
        if shake_hi < shake_lo:
            continue
        seg_min = min(float(rows[j].low) for j in range(shake_lo, shake_hi + 1))
        if seg_min + 1e-9 < L0 * (1.0 - break_tol):
            continue
        defense = (seg_min - L0) / L0
        sweet = 1.0 if 85 <= shake <= 145 else 0.6
        score = vr * 10.0 + sweet * 6.0 - defense * 40.0
        if score > best_score:
            best_score = score
            best_t = t
    return best_t


def _mode33_mid_rebound_detail(
    rows: List[KlineRow],
    T0: int,
    shake_lo: int,
    shake_hi: int,
    t_trial: int,
    L0: float,
    code: str,
    name: str,
    *,
    mid_rebound_min_shake_frac: float = 0.28,
    mid_rebound_min_rise_pct: float = 0.12,
    mid_surge_pct_min: float = 5.5,
    mid_surge_vol_mult: float = 1.25,
    mid_min_days_before_trial: int = 8,
    mid_pullback_min_pct: float = 0.06,
    anchor_body_min: float = 0.32,
    vol_ma: int = 20,
) -> Optional[Dict[str, Any]]:
    """长震仓后半段须出现中级反弹（放量冲高），试盘前自峰值回落整理。"""
    if shake_hi <= shake_lo or L0 <= 0:
        return None
    shake_days = shake_hi - shake_lo + 1
    min_peak_i = shake_lo + max(1, int(shake_days * mid_rebound_min_shake_frac))

    t_peak = min_peak_i
    h_peak = float(rows[t_peak].high)
    for j in range(min_peak_i + 1, shake_hi + 1):
        h = float(rows[j].high)
        if h > h_peak:
            h_peak = h
            t_peak = j
    if h_peak < L0 * (1.0 + mid_rebound_min_rise_pct):
        return None

    vols = [float(r.volume) for r in rows]
    surge_i: Optional[int] = None
    surge_best = -1e18
    win_lo = max(shake_lo, T0 + 3, t_peak - 20)
    win_hi = min(shake_hi, t_peak + 3)
    for j in range(win_lo, win_hi + 1):
        pct = float(rows[j].pct_chg)
        is_limit = _mode32_solid_limit_ok(
            rows, j, code, name, body_min=anchor_body_min * 0.85
        )
        if not is_limit and pct + 1e-9 < mid_surge_pct_min:
            continue
        if j < vol_ma:
            continue
        vma = float(np.mean(vols[j - vol_ma : j]))
        if not is_limit and vols[j] + 1e-9 < mid_surge_vol_mult * max(vma, 1e-9):
            continue
        score = pct + (8.0 if is_limit else 0.0) - abs(j - t_peak) * 0.3
        if score > surge_best:
            surge_best = score
            surge_i = j
    if surge_i is None:
        return None

    if t_trial - t_peak < mid_min_days_before_trial:
        return None

    post_lows = [
        float(rows[j].low) for j in range(t_peak + 1, t_trial) if t_peak + 1 < t_trial
    ]
    if not post_lows:
        return None
    trough = min(post_lows)
    if h_peak <= 0:
        return None
    if (h_peak - trough) / h_peak + 1e-9 < mid_pullback_min_pct:
        return None

    return {
        "T_mid": t_peak,
        "H_mid": h_peak,
        "T_surge": surge_i,
        "mid_rise_from_L0_pct": round((h_peak - L0) / L0 * 100.0, 2),
        "mid_pullback_pct": round((h_peak - trough) / h_peak * 100.0, 2),
    }


def _mode33_signal_vol_kind(
    rows: List[KlineRow],
    idx: int,
    t_trial: int,
    vr_sig: float,
    *,
    dt: int,
    final_vol_mult: float,
    box_end_vol_min: float,
    box_end_vol_max: float,
    box_end_day_max: int,
    vol_ma: int = 20,
) -> Optional[str]:
    """⑤ 末洗：放量最后一洗(final_shake) 或 试盘后缩量整理收官(box_end，000509@5/15)。"""
    if vr_sig + 1e-9 >= final_vol_mult:
        return "final_shake"
    if dt > box_end_day_max:
        return None
    if vr_sig + 1e-9 < box_end_vol_min or vr_sig > box_end_vol_max + 1e-9:
        return None
    if t_trial + 2 > idx:
        return None
    vols = [float(r.volume) for r in rows]
    v_d1 = max(vols[t_trial + 1], 1e-9)
    post_mean = float(np.mean(vols[t_trial + 2 : idx + 1]))
    if post_mean > v_d1 * 1.02:
        return None
    if _vol_ratio_at(rows, t_trial, vol_ma) > 0 and vr_sig > _vol_ratio_at(rows, t_trial, vol_ma) * 0.85:
        return None
    return "box_end"


def _mode33_match_detail(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    *,
    anchor_lookback: int = 300,
    anchor_body_min: float = 0.32,
    anchor_vol_mult: float = 1.20,
    break_tol: float = 0.005,
    shakeout_days_min: int = 55,
    shakeout_days_max: int = 165,
    sideways_range_pct: float = 0.62,
    trial_after_anchor_min: int = 35,
    trial_box_pct: float = 0.10,
    require_mid_rebound: bool = True,
    mid_rebound_min_shake_frac: float = 0.28,
    mid_rebound_min_rise_pct: float = 0.12,
    mid_surge_pct_min: float = 5.5,
    mid_surge_vol_mult: float = 1.25,
    mid_min_days_before_trial: int = 8,
    mid_pullback_min_pct: float = 0.06,
    final_day_min: int = 6,
    final_day_max: int = 20,
    final_vol_mult: float = 1.50,
    box_end_vol_min: float = 0.60,
    box_end_vol_max: float = 1.40,
    box_end_day_max: int = 12,
    final_pct_max: float = 5.0,
    final_body_max: float = 0.95,
    vol_ma: int = 20,
    ma_slope_days: int = 5,
    ma120_slope_min_pct: float = 0.01,
    ma120_slope_max_pct: float = 0.55,
    ma250_slope_min_pct: float = 0.01,
    ma250_slope_max_pct: float = 1.05,
) -> Optional[Dict[str, Any]]:
    """mode33：锚点吸筹→长震仓→中级反弹→试盘→末洗信号日。"""
    if _is_st(name or ""):
        return None
    if idx < 250 + ma_slope_days or idx >= len(rows):
        return None

    sig = rows[idx]
    if _limit_up_day(rows, idx, code, name):
        return None
    if float(sig.pct_chg) > final_pct_max + 1e-9:
        return None
    if _mode32_row_body_ratio(sig) > final_body_max:
        return None

    vr_sig = _vol_ratio_at(rows, idx, vol_ma)

    if not _mode32_long_ma_flat_up_ok(
        rows,
        idx,
        slope_days=ma_slope_days,
        ma120_slope_min_pct=ma120_slope_min_pct,
        ma120_slope_max_pct=ma120_slope_max_pct,
        ma250_slope_min_pct=ma250_slope_min_pct,
        ma250_slope_max_pct=ma250_slope_max_pct,
    ):
        return None

    for dt in range(final_day_min, final_day_max + 1):
        t_trial = idx - dt
        if t_trial < 1:
            continue
        if not _mode32_solid_limit_ok(rows, t_trial, code, name, body_min=anchor_body_min):
            continue

        sig_kind = _mode33_signal_vol_kind(
            rows,
            idx,
            t_trial,
            vr_sig,
            dt=dt,
            final_vol_mult=final_vol_mult,
            box_end_vol_min=box_end_vol_min,
            box_end_vol_max=box_end_vol_max,
            box_end_day_max=box_end_day_max,
            vol_ma=vol_ma,
        )
        if sig_kind is None:
            continue

        T0 = _mode33_find_anchor(
            rows,
            t_trial,
            code,
            name,
            anchor_lookback=anchor_lookback,
            trial_after_anchor_min=trial_after_anchor_min,
            anchor_body_min=anchor_body_min,
            shakeout_days_min=shakeout_days_min,
            shakeout_days_max=shakeout_days_max,
            break_tol=break_tol,
            vol_ma=vol_ma,
            anchor_vol_mult=anchor_vol_mult,
        )
        if T0 is None:
            continue

        L0 = _mode33_anchor_low(rows, T0)
        if L0 <= 0:
            continue
        H0 = float(rows[T0].high)
        H_trial = float(rows[t_trial].high)
        trial_mid = 0.5 * (float(rows[t_trial].open) + float(rows[t_trial].close))
        floor_l0 = L0 * (1.0 - break_tol)
        box_low = H_trial * (1.0 - trial_box_pct)

        shake_lo = T0 + 1
        shake_hi = t_trial - 1
        if shake_hi < shake_lo:
            continue
        shake_days = shake_hi - shake_lo + 1

        seg_highs = [float(rows[j].high) for j in range(shake_lo, shake_hi + 1)]
        seg_lows = [float(rows[j].low) for j in range(shake_lo, shake_hi + 1)]
        seg_closes = [float(rows[j].close) for j in range(shake_lo, shake_hi + 1)]
        seg_min_low = min(seg_lows)
        if seg_min_low + 1e-9 < floor_l0:
            continue
        mean_c = sum(seg_closes) / max(1, len(seg_closes))
        if mean_c <= 0:
            continue
        if (max(seg_highs) - min(seg_lows)) / mean_c > sideways_range_pct:
            continue

        mid_det = _mode33_mid_rebound_detail(
            rows,
            T0,
            shake_lo,
            shake_hi,
            t_trial,
            L0,
            code,
            name,
            mid_rebound_min_shake_frac=mid_rebound_min_shake_frac,
            mid_rebound_min_rise_pct=mid_rebound_min_rise_pct,
            mid_surge_pct_min=mid_surge_pct_min,
            mid_surge_vol_mult=mid_surge_vol_mult,
            mid_min_days_before_trial=mid_min_days_before_trial,
            mid_pullback_min_pct=mid_pullback_min_pct,
            anchor_body_min=anchor_body_min,
            vol_ma=vol_ma,
        )
        if require_mid_rebound and mid_det is None:
            continue

        post_lows = [float(rows[j].low) for j in range(t_trial + 1, idx + 1)]
        if post_lows and min(post_lows) + 1e-9 < floor_l0:
            continue

        sig_low = float(sig.low)
        sig_close = float(sig.close)
        if sig_low + 1e-9 < box_low and sig_close + 1e-9 < trial_mid:
            continue

        s120, s250 = _mode32_long_ma_slopes(rows, idx, ma_slope_days)
        out: Dict[str, Any] = {
            "T0": T0,
            "T_trial": t_trial,
            "L0": L0,
            "H0": H0,
            "H_trial": H_trial,
            "shake_days": shake_days,
            "shake_min_low": seg_min_low,
            "shake_range_pct": (max(seg_highs) - min(seg_lows)) / mean_c,
            "days_trial_to_signal": dt,
            "final_vol_ratio20": vr_sig,
            "signal_kind": sig_kind,
            "ma120_slope5_pct": s120,
            "ma250_slope5_pct": s250,
            "anchor_vol_ratio20": round(_vol_ratio_at(rows, T0, vol_ma), 4),
        }
        if mid_det:
            out["T_mid"] = int(mid_det["T_mid"])
            out["H_mid"] = float(mid_det["H_mid"])
            out["T_surge"] = int(mid_det["T_surge"])
            out["mid_rise_from_L0_pct"] = float(mid_det["mid_rise_from_L0_pct"])
            out["mid_pullback_pct"] = float(mid_det["mid_pullback_pct"])
        return out

    return None


def _mode33_signal_at(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    **kwargs: Any,
) -> bool:
    return _mode33_match_detail(rows, idx, code, name, **kwargs) is not None


def _mode33_metrics(rows: List[KlineRow], idx: int, code: str, name: str, **kwargs: Any) -> Dict[str, Any]:
    det = _mode33_match_detail(rows, idx, code, name, **kwargs)
    if not det:
        return {}
    T0 = int(det["T0"])
    Tt = int(det["T_trial"])
    out: Dict[str, Any] = {
        "mode33_anchor_date": rows[T0].date,
        "mode33_L0": round(float(det["L0"]), 4),
        "mode33_H0": round(float(det["H0"]), 4),
        "mode33_trial_date": rows[Tt].date,
        "mode33_H_trial": round(float(det["H_trial"]), 4),
        "mode33_days_anchor_to_trial": int(Tt - T0),
        "mode33_days_trial_to_signal": int(det["days_trial_to_signal"]),
        "mode33_shakeout_days": int(det["shake_days"]),
        "mode33_shakeout_min_low": round(float(det["shake_min_low"]), 4),
        "mode33_shakeout_range_pct": round(float(det["shake_range_pct"]) * 100.0, 2),
        "mode33_final_vol_ratio20": round(float(det["final_vol_ratio20"]), 4),
    }
    if det.get("signal_kind"):
        out["mode33_signal_kind"] = str(det["signal_kind"])
    if det.get("ma120_slope5_pct") is not None:
        out["mode33_ma120_slope5_pct"] = round(float(det["ma120_slope5_pct"]), 4)
    if det.get("ma250_slope5_pct") is not None:
        out["mode33_ma250_slope5_pct"] = round(float(det["ma250_slope5_pct"]), 4)
    if det.get("T_mid") is not None:
        out["mode33_mid_rebound_date"] = rows[int(det["T_mid"])].date
        out["mode33_H_mid"] = round(float(det["H_mid"]), 4)
        out["mode33_mid_rise_from_L0_pct"] = float(det.get("mid_rise_from_L0_pct", 0))
        out["mode33_mid_pullback_pct"] = float(det.get("mid_pullback_pct", 0))
    if det.get("T_surge") is not None:
        out["mode33_mid_surge_date"] = rows[int(det["T_surge"])].date
    if det.get("anchor_vol_ratio20") is not None:
        out["mode33_anchor_vol_ratio20"] = float(det["anchor_vol_ratio20"])
    return out


def _mode33_volume_profile(
    rows: List[KlineRow],
    idx: int,
    det: Dict[str, Any],
    *,
    vol_ma: int = 20,
) -> Dict[str, float]:
    """量能结构特征（用于相似股匹配与涨幅预估）。"""
    T0 = int(det["T0"])
    Tt = int(det["T_trial"])
    vols = [float(r.volume) for r in rows]

    def _vr(i: int) -> float:
        if i < vol_ma or i >= len(vols):
            return 1.0
        vma = float(np.mean(vols[i - vol_ma : i]))
        return vols[i] / max(vma, 1e-9)

    v_trial = max(vols[Tt], 1e-9)
    v_d1 = max(vols[Tt + 1], 1e-9) if Tt + 1 < len(vols) else v_trial
    post: List[float] = []
    if Tt + 2 < idx:
        post = vols[Tt + 2 : idx]
    post_mean = float(np.mean(post)) if post else v_d1

    shake_vols = vols[T0 + 1 : Tt] if Tt > T0 + 1 else []
    shake_mean = float(np.mean(shake_vols)) if shake_vols else v_trial

    close_sig = float(rows[idx].close)
    L0 = float(det["L0"])
    return {
        "trial_vol_ratio20": round(_vr(Tt), 4),
        "day1_vol_vs_trial": round(v_d1 / v_trial, 4),
        "post_trial_vol_vs_day1": round(post_mean / v_d1, 4),
        "shake_vol_vs_trial": round(shake_mean / v_trial, 4),
        "final_vol_ratio20": round(float(det["final_vol_ratio20"]), 4),
        "rise_from_L0_pct": round((close_sig - L0) / L0 * 100.0, 2) if L0 > 0 else 0.0,
        "days_trial_to_signal": float(det["days_trial_to_signal"]),
        "shake_days": float(det["shake_days"]),
        "shake_range_pct": round(float(det["shake_range_pct"]) * 100.0, 2),
    }


def _mode33_forward_rise(
    rows: List[KlineRow],
    idx: int,
    *,
    horizons: Tuple[int, ...] = (20, 40, 60, 120),
    entry: str = "signal_close",
) -> Dict[str, float]:
    """信号后涨幅：默认以信号日收盘价为基准，统计各窗口内最高价涨幅。"""
    if idx >= len(rows):
        return {}
    if entry == "signal_close":
        base = float(rows[idx].close)
        base_i = idx
    else:
        base_i = idx + 1
        if base_i >= len(rows):
            return {}
        base = float(rows[base_i].open)
    if base <= 0:
        return {}
    out: Dict[str, float] = {}
    for d in horizons:
        end = min(len(rows) - 1, base_i + d)
        if end <= base_i:
            out[f"max_gain_{d}d_pct"] = float("nan")
            out[f"close_gain_{d}d_pct"] = float("nan")
            continue
        seg = rows[base_i + 1 : end + 1]
        hi = max(float(r.high) for r in seg)
        close_end = float(rows[end].close)
        out[f"max_gain_{d}d_pct"] = round((hi / base - 1.0) * 100.0, 2)
        out[f"close_gain_{d}d_pct"] = round((close_end / base - 1.0) * 100.0, 2)
    return out


def _score_mode33(
    rows: List[KlineRow],
    idx: int,
    ma10: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
    vol20: np.ndarray,
    code: str = "",
    name: str = "",
    breakdown: Optional[List[tuple]] = None,
    **kwargs: Any,
) -> int:
    """mode33 评分约 62～95。"""
    det = _mode33_match_detail(rows, idx, code, name, **kwargs)
    if not det:
        return 62
    score = 62.0
    tight = float(det["shake_range_pct"])
    score += max(0.0, min(18.0, (0.55 - tight) * 40.0))
    L0 = float(det["L0"])
    shake_min = float(det["shake_min_low"])
    if L0 > 0:
        lift = (shake_min - L0) / L0
        score += max(0.0, min(12.0, lift * 80.0))
    kind = str(det.get("signal_kind") or "")
    vr = float(det["final_vol_ratio20"])
    if kind == "final_shake":
        if 1.5 <= vr <= 2.5:
            score += 10.0
        elif 1.35 <= vr < 1.5:
            score += 5.0
        elif vr > 2.8:
            score -= 5.0
    elif kind == "box_end":
        if 0.75 <= vr <= 1.15:
            score += 10.0
        elif 0.60 <= vr <= 1.35:
            score += 6.0
    dt = int(det["days_trial_to_signal"])
    if 9 <= dt <= 12:
        score += 6.0
    elif 7 <= dt <= 18:
        score += 3.0
    mid_rise = det.get("mid_rise_from_L0_pct")
    if mid_rise is not None:
        score += max(0.0, min(8.0, (float(mid_rise) - 12.0) * 0.25))
    mid_pb = det.get("mid_pullback_pct")
    if mid_pb is not None and 8.0 <= float(mid_pb) <= 35.0:
        score += 4.0
    avr = det.get("anchor_vol_ratio20")
    if avr is not None and float(avr) >= 1.5:
        score += 3.0
    s120 = det.get("ma120_slope5_pct")
    s250 = det.get("ma250_slope5_pct")
    if s120 is not None and s250 is not None:
        flatness = abs(float(s120)) + abs(float(s250))
        score += max(0.0, min(8.0, (1.2 - flatness) * 6.0))
    if float(rows[idx].close) < float(rows[idx].open):
        score += 4.0
    return int(max(62, min(95, round(score))))


def _mode88_signals(
    rows: List[KlineRow],
    start_date: Optional[str],
    end_date: Optional[str],
    d_min: float = 0.03,
    d_max: float = 0.15,
    r_min: float = 0.03,
    acc_L: int = 8,
    acc_R: int = 20,
    A_min: float = 15.0,
    A_max: float = 55.0,
    epsilon: float = 0.02,
    wash_L: int = 2,
    wash_R: int = 10,
    R_rise: float = 8.0,
    D_pull: float = 3.0,
    K_vol: float = 1.0,
) -> List[int]:
    """
    mode88：吸筹 → 洗盘 → 震仓 → 拉升。信号日 = 震仓第 3 周最后交易日，买点 = 下一交易日开盘。
    仅当震仓三周形态成立且震仓前同时满足吸筹、洗盘时出信号。
    """
    if not rows or len(rows) < 260:
        return []
    weekly_bars, last_indices = daily_to_weekly_with_volume_and_last_index(rows)
    nw = len(weekly_bars)
    if nw < 25:
        return []
    # 周线: (week_key, open, high, low, close, volume)
    closes = np.array([w[4] for w in weekly_bars], dtype=float)
    highs = np.array([w[2] for w in weekly_bars], dtype=float)
    lows = np.array([w[3] for w in weekly_bars], dtype=float)
    vols = np.array([w[5] for w in weekly_bars], dtype=float)
    signal_indices = []
    for i in range(acc_R + wash_R + 2, nw):
        i1 = i - 2
        i2 = i - 1
        c1, c2, c3 = closes[i1], closes[i2], closes[i]
        v1, v2 = vols[i1], vols[i2]
        l2 = lows[i2]
        if c1 <= 0 or c2 <= 0 or c3 <= 0:
            continue
        if not (c2 < c1 and v2 < v1 and c3 > c2):
            continue
        drop_pct = (c1 - c2) / c1
        if drop_pct < d_min or drop_pct > d_max:
            continue
        if (c3 - c2) / c2 < r_min:
            continue
        shakeout_i = i2
        acc_start = shakeout_i - acc_R
        acc_end = shakeout_i - acc_L + 1
        if acc_start < 0 or acc_end <= acc_start + 2:
            continue
        acc_high = float(np.nanmax(highs[acc_start:acc_end]))
        acc_low = float(np.nanmin(lows[acc_start:acc_end]))
        acc_mid = (acc_high + acc_low) / 2
        if acc_mid <= 0:
            continue
        amplitude = (acc_high - acc_low) / acc_mid * 100
        if amplitude < A_min or amplitude > A_max:
            continue
        if l2 < acc_low * (1 - epsilon):
            continue
        wash_start = shakeout_i - wash_R
        wash_end = shakeout_i - wash_L + 1
        if wash_start < 0 or wash_end <= wash_start + 2:
            continue
        wash_high = float(np.nanmax(highs[wash_start:wash_end]))
        wash_low = float(np.nanmin(lows[wash_start:wash_end]))
        if wash_low <= 0:
            continue
        phase_rise = (wash_high - wash_low) / wash_low * 100
        if phase_rise < R_rise:
            continue
        close_before = closes[shakeout_i - 1]
        if wash_high <= 0:
            continue
        pullback = (wash_high - close_before) / wash_high * 100
        if pullback < D_pull:
            continue
        high_idx = wash_start + int(np.nanargmax(highs[wash_start:wash_end]))
        vol_up_slice = vols[wash_start : high_idx + 1]
        vol_down_slice = vols[high_idx : shakeout_i]
        if len(vol_up_slice) == 0 or len(vol_down_slice) == 0:
            continue
        v_up = float(np.nanmean(vol_up_slice))
        v_down = float(np.nanmean(vol_down_slice))
        if v_up <= 0 or v_down > v_up * K_vol:
            continue
        idx = last_indices[i]
        if idx >= len(rows):
            continue
        d = rows[idx].date
        if start_date and d < start_date:
            continue
        if end_date and d > end_date:
            continue
        signal_indices.append(idx)
    return sorted(signal_indices)


def _score_mode88(
    rows: List[KlineRow],
    idx: int,
    ma10: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
    vol20: np.ndarray,
    code: str = "",
    name: str = "",
) -> int:
    """
    mode88 评分：吸筹/洗盘/震仓形态 + 拉升周强度 + 周线 MA20/MA10/MACD + 日线偏离MA20。无上下限，按各项加减分汇总。
    特征：第3周涨幅/放量、震仓跌幅、不破前低缓冲；周线 MA20/MA10/MACD；日线收盘偏离日线MA20分档扣分（>15%扣5，>12%扣3，>8%扣2，>5%扣1）。
    """
    if idx < 0 or idx >= len(rows):
        return 50
    sub = rows[: idx + 1]
    if len(sub) < 260:
        return 50
    weekly_bars, _ = daily_to_weekly_with_volume_and_last_index(sub)
    nw = len(weekly_bars)
    if nw < 25:
        return 50
    closes = np.array([w[4] for w in weekly_bars], dtype=float)
    highs = np.array([w[2] for w in weekly_bars], dtype=float)
    lows = np.array([w[3] for w in weekly_bars], dtype=float)
    vols = np.array([w[5] for w in weekly_bars], dtype=float)
    i = nw - 1
    i2 = i - 1
    i1 = i - 2
    shakeout_i = i2
    c1, c2, c3 = closes[i1], closes[i2], closes[i]
    v1, v2, v3 = vols[i1], vols[i2], vols[i]
    l2 = lows[i2]
    # 第3周涨幅（拉升周强度，与后续收益正相关）
    week3_rise = (c3 - c2) / c2 * 100 if c2 > 0 else 0
    week3_volume_up = v3 > v2 if v2 > 0 else False
    shakeout_drop_pct = (c1 - c2) / c1 * 100 if c1 > 0 else 0

    acc_start = shakeout_i - 20
    acc_end = shakeout_i - 8 + 1
    if acc_start < 0 or acc_end <= acc_start + 2:
        return 50
    acc_high = float(np.nanmax(highs[acc_start:acc_end]))
    acc_low = float(np.nanmin(lows[acc_start:acc_end]))
    acc_mid = (acc_high + acc_low) / 2
    amplitude = (acc_high - acc_low) / acc_mid * 100 if acc_mid > 0 else 0
    # 震仓周最低相对吸筹低的缓冲（不破且留有余地加分）
    hold_buffer = (l2 - acc_low) / acc_low * 100 if acc_low > 0 else 0

    wash_start = shakeout_i - 10
    wash_end = shakeout_i - 2 + 1
    if wash_start < 0 or wash_end <= wash_start + 2:
        return 50
    wash_high = float(np.nanmax(highs[wash_start:wash_end]))
    wash_low = float(np.nanmin(lows[wash_start:wash_end]))
    phase_rise = (wash_high - wash_low) / wash_low * 100 if wash_low > 0 else 0
    close_before = closes[shakeout_i - 1]
    pullback = (wash_high - close_before) / wash_high * 100 if wash_high > 0 else 0
    high_idx = wash_start + int(np.nanargmax(highs[wash_start:wash_end]))
    v_up = float(np.nanmean(vols[wash_start : high_idx + 1]))
    v_down = float(np.nanmean(vols[high_idx : shakeout_i]))
    vol_ratio = (v_down / v_up) if v_up > 0 else 1.0

    # 周线 MA20 方向：信号周 MA20 与上周比较（与买点后收益正相关）
    ma20_weekly = np.full_like(closes, np.nan, dtype=float)
    for j in range(19, len(closes)):
        ma20_weekly[j] = float(np.mean(closes[j - 19 : j + 1]))
    ma20_up = (
        ma20_weekly[i] > ma20_weekly[i - 1]
        if (i >= 20 and not np.isnan(ma20_weekly[i]) and not np.isnan(ma20_weekly[i - 1]))
        else False
    )
    # 周线 MA10 方向：信号周 MA10 与上周比较，向下扣分
    ma10_weekly = np.full_like(closes, np.nan, dtype=float)
    for j in range(9, len(closes)):
        ma10_weekly[j] = float(np.mean(closes[j - 9 : j + 1]))
    ma10_up = (
        ma10_weekly[i] > ma10_weekly[i - 1]
        if (i >= 10 and not np.isnan(ma10_weekly[i]) and not np.isnan(ma10_weekly[i - 1]))
        else False
    )
    # 周线 MACD：信号周 DIF 与 DEA 比较，金叉加分、死叉扣分
    dif_norm, dea_norm = _weekly_macd_dif_dea(closes, 12, 26, 9)
    macd_golden = (
        dif_norm[i] > dea_norm[i]
        if (i >= 34 and not np.isnan(dif_norm[i]) and not np.isnan(dea_norm[i]))
        else None
    )
    # 日线偏离 MA20：信号日收盘相对日线 MA20 的偏离幅度，分档扣分（偏离越大扣分越多）
    day_close = float(rows[idx].close)
    ma20_day = float(ma20[idx]) if idx < len(ma20) and not np.isnan(ma20[idx]) and ma20[idx] > 0 else None
    if ma20_day is not None and ma20_day > 0:
        day_deviation_pct = abs(day_close - ma20_day) / ma20_day * 100
    else:
        day_deviation_pct = None

    base = 50
    # 吸筹振幅
    if 18 <= amplitude <= 45:
        base += 9
    elif 15 <= amplitude <= 55:
        base += 5
    # 洗盘阶段涨幅
    if phase_rise >= 15:
        base += 10
    elif phase_rise >= 8:
        base += 5
    # 洗盘回撤+缩量
    if pullback >= 5 and vol_ratio <= 0.85:
        base += 10
    elif pullback >= 3 and vol_ratio <= 1.0:
        base += 5
    # 第3周涨幅（与后续收益正相关，权重大）
    if week3_rise >= 8:
        base += 10
    elif week3_rise >= 6:
        base += 7
    elif week3_rise >= 5:
        base += 6
    elif week3_rise >= 4:
        base += 3
    elif week3_rise < 3.5 and not week3_volume_up:
        base -= 2
    # 第3周放量（放量拉升加分）
    if week3_volume_up:
        base += 5
    # 震仓跌幅适中加分、过深略扣
    if 5 <= shakeout_drop_pct <= 10:
        base += 3
    elif 4 <= shakeout_drop_pct <= 11:
        base += 1
    elif shakeout_drop_pct > 12:
        base -= 1
    # 不破前低且有缓冲
    if hold_buffer >= 1:
        base += 2
    elif hold_buffer >= 0:
        base += 1
    # 周线 MA20 向上（权重大，与买点后收益正相关）
    if ma20_up:
        base += 8
    else:
        base -= 3
    # 周线 MA10 向下扣分（权重较 MA20 小）
    if not ma10_up:
        base -= 1
    # 周线 MACD 金叉加分、死叉扣分
    if macd_golden is not None:
        if macd_golden:
            base += 4
        else:
            base -= 3
    # 日线偏离 MA20 分档扣分（追高或远离均线风险）
    if day_deviation_pct is not None:
        if day_deviation_pct > 15:
            base -= 5
        elif day_deviation_pct > 12:
            base -= 3
        elif day_deviation_pct > 8:
            base -= 2
        elif day_deviation_pct > 5:
            base -= 1
    return int(round(base))


def _pct_change(a: np.ndarray, period: int) -> Optional[float]:
    if len(a) <= period:
        return None
    base = a[-period - 1]
    if base == 0:
        return None
    return (a[-1] - base) / base * 100


def score_stock(
    item: StockItem,
    rows: List[KlineRow],
    index_return_10d: Optional[float],
    return_percentile_10d: Optional[float],
    config: ScanConfig,
) -> Optional[ScanResult]:
    if _is_st(item.name):
        return None
    if len(rows) < 80:
        return None

    arr = _to_array(rows)
    close = arr["close"]
    high = arr["high"]
    low = arr["low"]
    volume = arr["volume"]

    ma20 = _rolling_mean(close, 20)
    ma60 = _rolling_mean(close, 60)
    if len(ma20) == 0 or len(ma60) == 0:
        return None

    latest_close = close[-1]
    latest_change = rows[-1].pct_chg

    if len(close) <= config.year_lookback:
        return None
    base = close[-config.year_lookback - 1]
    if base > 0:
        year_return = (latest_close - base) / base * 100
        if year_return >= config.year_return_limit:
            return None

    trend_score = 0
    volume_score = 0
    breakout_score = 0
    strength_score = 0
    risk_score = 0
    reasons: List[str] = []

    # Trend structure (30)
    ma20_now = ma20[-1]
    ma60_now = ma60[-1]
    ma20_slope = ma20[-1] - ma20[-4] if len(ma20) >= 4 else 0
    ma60_slope = ma60[-1] - ma60[-4] if len(ma60) >= 4 else 0

    if ma20_now > ma60_now and ma20_slope > 0 and ma60_slope > 0:
        trend_score += 10
        reasons.append("20/60均线多头且上行")

    if len(close) >= 3 and np.all(close[-3:] > ma20_now):
        trend_score += 10
        reasons.append("连续3日站上20日均线")

    high_20 = np.max(close[-20:])
    if high_20 > 0 and (high_20 - latest_close) / high_20 * 100 <= config.near_high_pct:
        trend_score += 10
        reasons.append("接近20日新高")

    # Volume & flow (25)
    vol5 = np.mean(volume[-5:])
    vol20 = np.mean(volume[-20:])
    if vol20 > 0 and vol5 >= vol20 * config.volume_ratio:
        volume_score += 10
        reasons.append("5日均量显著放大")

    if len(close) >= 11:
        up_mask = close[1:] >= close[:-1]
        vol_up = np.sum(volume[1:][up_mask])
        vol_down = np.sum(volume[1:][~up_mask])
        if vol_up > vol_down:
            volume_score += 10
            reasons.append("上涨日量能占优")

    if vol20 > 0 and np.max(volume[-3:]) >= vol20 * 1.8:
        volume_score += 5
        reasons.append("近3日出现明显放量")

    # Breakout / pattern (20)
    lookback = config.breakout_lookback
    recent = config.breakout_recent
    if len(close) > lookback + recent:
        previous_high = np.max(close[-(lookback + recent) : -recent])
        recent_high = np.max(close[-recent:])
        if recent_high >= previous_high:
            breakout_score += 10
            reasons.append("近期突破前高")

    if len(low) >= 5:
        if np.min(low[-3:]) >= ma20_now * 0.98 and volume[-1] <= vol20:
            breakout_score += 10
            reasons.append("回踩不破+量能收敛")

    # Strength (15)
    if return_percentile_10d is not None and return_percentile_10d >= 90:
        strength_score += 10
        reasons.append("10日涨幅位列前10%")

    if index_return_10d is not None:
        stock_return_10d = _pct_change(close, 10)
        if stock_return_10d is not None and stock_return_10d >= index_return_10d + 2:
            strength_score += 5
            reasons.append("强于指数")

    # Risk penalties
    # Long upper shadow on heavy volume
    body = abs(rows[-1].close - rows[-1].open)
    upper = rows[-1].high - max(rows[-1].close, rows[-1].open)
    if vol20 > 0 and upper > body * 2 and volume[-1] >= vol20 * 1.5:
        risk_score -= 5
        reasons.append("放量长上影扣分")

    if vol20 > 0 and np.mean(volume[-5:]) >= vol20 * 1.5:
        prev_high = np.max(close[-25:-5]) if len(close) >= 30 else np.max(close[:-5])
        if prev_high > 0 and np.max(close[-5:]) < prev_high:
            risk_score -= 5
            reasons.append("放量未创新高扣分")

    raw_score = trend_score + volume_score + breakout_score + strength_score + risk_score
    weighted_score = (
        trend_score * config.weight_trend
        + volume_score * config.weight_volume
        + breakout_score * config.weight_breakout
        + strength_score * config.weight_strength
        + risk_score * config.weight_risk
    )
    score = int(round(weighted_score))

    metrics = {
        "ma20": float(ma20_now),
        "ma60": float(ma60_now),
        "vol5": float(vol5),
        "vol20": float(vol20),
        "high20": float(high_20),
        "score_raw": float(raw_score),
        "score_weighted": float(score),
        "score_trend": float(trend_score),
        "score_volume": float(volume_score),
        "score_breakout": float(breakout_score),
        "score_strength": float(strength_score),
        "score_risk": float(risk_score),
        "weight_trend": float(config.weight_trend),
        "weight_volume": float(config.weight_volume),
        "weight_breakout": float(config.weight_breakout),
        "weight_strength": float(config.weight_strength),
        "weight_risk": float(config.weight_risk),
    }

    return ScanResult(
        code=item.code,
        name=item.name,
        score=int(score),
        latest_close=float(latest_close),
        change_pct=float(latest_change),
        reasons=reasons,
        metrics=metrics,
    )


def percentile_ranks(values: List[float]) -> Dict[int, float]:
    if not values:
        return {}
    arr = np.array(values, dtype=float)
    order = np.argsort(arr)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(arr))
    percentiles = ranks / (len(arr) - 1) * 100 if len(arr) > 1 else np.array([100.0])
    return {idx: float(percentiles[idx]) for idx in range(len(arr))}


def _parse_date(value: Optional[str]) -> Optional[datetime.date]:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except Exception:
        return None


def _mode3_signals(
    rows: List[KlineRow],
    start_date: Optional[str],
    end_date: Optional[str],
) -> List[int]:
    """
    找出所有满足 mode3 启动点的信号日下标。
    测算起点：从信号日当天往前至少 60 根 K 线参与计算（MA60、vol20 等），
    即从买点前约 60 个交易日开始数据就参与测算；第一个可能出信号的日期是第 61 根 K 线（下标 60）。
    """
    signals: List[int] = []
    if len(rows) < 60:
        return signals
    close = np.array([r.close for r in rows], dtype=float)
    volume = np.array([r.volume for r in rows], dtype=float)

    ma10 = _moving_mean(close, 10)
    ma20 = _moving_mean(close, 20)
    ma60 = _moving_mean(close, 60)
    vol20 = _moving_mean(volume, 20)

    start_dt = _parse_date(start_date)
    end_dt = _parse_date(end_date)

    for i in range(60, len(rows)):
        if start_dt or end_dt:
            try:
                row_dt = datetime.strptime(rows[i].date, "%Y-%m-%d").date()
            except Exception:
                continue
            if start_dt and row_dt < start_dt:
                continue
            if end_dt and row_dt > end_dt:
                continue

        if (
            np.isnan(ma10[i])
            or np.isnan(ma20[i])
            or np.isnan(ma60[i])
            or np.isnan(vol20[i])
        ):
            continue

        if i - 20 >= 0 and close[i - 20] > 0:
            ret20 = (close[i] - close[i - 20]) / close[i - 20] * 100
            if ret20 > 25:
                continue

        ma10_slope = ma10[i] - ma10[i - 3]
        ma20_slope = ma20[i] - ma20[i - 3]
        ma60_slope = ma60[i] - ma60[i - 3]
        if not (
            ma10[i] > ma20[i] > ma60[i]
            and ma10_slope > 0
            and ma20_slope > 0
            and ma60_slope > 0
        ):
            continue
        if close[i] < ma20[i]:
            continue
        if volume[i] < vol20[i] * 1.2:
            continue
        signals.append(i)
    return signals


def _mode9_signals(
    rows: List[KlineRow],
    start_date: Optional[str],
    end_date: Optional[str],
) -> List[int]:
    """mode9：与 mode3（71倍）完全一致的信号逻辑，复制一份便于独立调参或扩展。"""
    return _mode3_signals(rows, start_date, end_date)


def _limit_up_day(rows: List[KlineRow], i: int, code: str, name: str) -> bool:
    """按 ST/板块涨停幅度判断第 i 日是否涨停。"""
    if i < 0 or i >= len(rows):
        return False
    rate = _limit_rate(code, name)
    limit_up = (rate * 100) - 0.5
    return float(rows[i].pct_chg) >= float(limit_up)


def _mode5_lows_on_or_above_ma10(
    rows: List[KlineRow],
    limit_idx: int,
    until_idx: int,
    ma10: np.ndarray,
) -> bool:
    """从涨停次日至 until_idx（含），每日最低价不得低于当日 MA10。"""
    if limit_idx < 0 or until_idx >= len(rows) or until_idx < limit_idx + 1:
        return False
    for j in range(limit_idx + 1, until_idx + 1):
        m = ma10[j]
        if np.isnan(m):
            return False
        if float(rows[j].low) < float(m):
            return False
    return True


def _mode5_anchor_detail(
    rows: List[KlineRow],
    s_idx: int,
    code: str,
    name: str,
    *,
    shrink_max_days: int = 5,
    half_year_bars: int = 120,
) -> Optional[Tuple[int, float, float]]:
    """
    mode5 单点判定（信号日 s_idx）：
    - 收盘在半年线（MA half_year_bars）之上；
    - 信号日 MA20 向上（MA20[s] > MA20[s-1]）；
    - 存在涨停日 T，使 s ∈ [T+2, T+shrink_max_days]；
    - 成交量：vol[s] < vol[T+1]/2（基准为涨停次日量）；
    - 从涨停次日至信号日：low 不低于当日 MA10。
    返回 (T, 涨停次日成交量, vol[s]/vol[T+1])；否则 None。
    """
    if s_idx < half_year_bars or s_idx >= len(rows):
        return None
    close = np.array([r.close for r in rows], dtype=float)
    vol = np.array([r.volume for r in rows], dtype=float)
    ma10 = _moving_mean(close, 10)
    ma20 = _moving_mean(close, 20)
    ma_h = _moving_mean(close, half_year_bars)
    if np.isnan(ma_h[s_idx]) or close[s_idx] <= ma_h[s_idx]:
        return None
    if (
        s_idx < 1
        or np.isnan(ma20[s_idx])
        or np.isnan(ma20[s_idx - 1])
        or ma20[s_idx] <= ma20[s_idx - 1]
    ):
        return None

    # 取 [s-shrink_max_days, s-2] 内最早满足条件的涨停日 T
    for T in range(s_idx - shrink_max_days, s_idx - 1):
        if T < 0:
            continue
        if not _limit_up_day(rows, T, code, name):
            continue
        if s_idx < T + 2 or s_idx > T + shrink_max_days:
            continue
        if T + 1 >= len(rows):
            continue
        v_ref = vol[T + 1]
        if v_ref <= 0:
            continue
        if vol[s_idx] >= v_ref * 0.5:
            continue
        if not _mode5_lows_on_or_above_ma10(rows, T, s_idx, ma10):
            continue
        return (T, float(v_ref), float(vol[s_idx] / v_ref))
    return None


def _mode93_anchor_detail(
    rows: List[KlineRow],
    s_idx: int,
    code: str,
    name: str,
    *,
    lookback_days: int = 20,
    low_window: int = 120,
    low_recent_days: int = 10,
    vol_mult: float = 3.0,
    pullback_min: float = 0.99,
    pullback_max: float = 1.02,
    pullback_max_days: int = 20,
) -> Optional[Dict[str, float]]:
    """
    mode93 单点判定（信号日 s_idx）：
    - 在最近 lookback_days 内，存在“低位放量涨停”事件：低位=近 low_recent_days 天出现 low_window 日最低点；
    - 最低点次日：成交量放大 ≥ vol_mult 倍，且当日涨停；
    - 涨停日最低价记为 A；信号日收盘价落在 [pullback_min*A, pullback_max*A]；
    - 信号日距离涨停日不超过 pullback_max_days。

    返回关键指标用于 reasons/metrics（否则 None）。
    """
    n = len(rows)
    if s_idx <= 0 or s_idx >= n:
        return None
    lookback_days = max(5, int(lookback_days))
    low_window = max(30, int(low_window))
    low_recent_days = max(2, int(low_recent_days))
    pullback_max_days = max(3, int(pullback_max_days))
    vol_mult = float(vol_mult or 0.0)
    if vol_mult <= 1.0:
        vol_mult = 3.0

    close = np.array([r.close for r in rows], dtype=float)
    low = np.array([r.low for r in rows], dtype=float)
    vol = np.array([r.volume for r in rows], dtype=float)

    # 先检查信号日回调区间（针对每个候选涨停日不同 A）
    s_close = float(close[s_idx])
    if not (s_close > 0):
        return None

    start = max(low_window, s_idx - lookback_days - pullback_max_days - 2)
    end = s_idx - 1
    if end < start:
        return None

    # 在 [start, end] 内找候选“涨停放量日”（即最低点次日）
    for limit_idx in range(max(start + 1, s_idx - pullback_max_days), end + 1):
        if limit_idx <= 0 or limit_idx >= n:
            continue
        # 低位：涨停日前 low_recent_days 天内，出现 low_window 日最低点（最低点日不要求紧挨涨停前一天）
        # 例如 low_recent_days=10：则 [limit_idx-9, limit_idx] 这10天内只要有一天是120日最低即可
        recent_start = max(low_window - 1, limit_idx - (low_recent_days - 1))
        recent_end = limit_idx
        if recent_end - recent_start + 1 < 2:
            continue
        is_low = False
        low_120 = float("nan")
        low_day_idx = None
        for j in range(recent_start, recent_end + 1):
            if j < low_window - 1:
                continue
            low_120_j = float(np.nanmin(low[j - low_window + 1 : j + 1]))
            if not (low_120_j > 0):
                continue
            # j 当天 low 接近该 120 日窗口最低
            if abs(float(low[j]) - low_120_j) / low_120_j <= 0.0008:
                is_low = True
                low_120 = low_120_j
                low_day_idx = j
                break
        if not is_low or low_day_idx is None:
            continue

        # 次日放量 >= vol_mult 倍
        # 仍按“涨停日相对前一日”放量（符合“第二天突然放大”）
        base_idx = limit_idx - 1
        v0 = float(vol[base_idx])
        v1 = float(vol[limit_idx])
        if not (v0 > 0 and v1 > 0):
            continue
        if v1 < v0 * vol_mult:
            continue

        # 次日涨停
        if not _limit_up_day(rows, limit_idx, code, name):
            continue

        # 信号日回调到涨停日最低价 A 附近
        A = float(rows[limit_idx].low)
        if not (A > 0):
            continue
        lo = A * float(pullback_min)
        hi = A * float(pullback_max)
        if not (lo <= s_close <= hi):
            continue

        # 回调期间“慢慢回调”的弱约束：从涨停次日至信号日，收盘不得大幅跌破 A（避免破位太深）
        if np.nanmin(close[limit_idx + 1 : s_idx + 1]) < A * 0.92:
            continue

        return {
            "base_low": float(low_120),
            "base_idx": float(low_day_idx),
            "limit_idx": float(limit_idx),
            "A": float(A),
            "vol_mult": float(v1 / v0),
            "pullback_pct": float((s_close - A) / A * 100.0),
        }
    return None


def _mode93_signals(
    rows: List[KlineRow],
    start_date: Optional[str],
    end_date: Optional[str],
    *,
    lookback_days: int = 20,
    low_window: int = 120,
    low_recent_days: int = 10,
    vol_mult: float = 3.0,
    pullback_min: float = 0.99,
    pullback_max: float = 1.02,
    pullback_max_days: int = 20,
) -> List[int]:
    if not rows or len(rows) < max(200, low_window + 10):
        return []
    # 日期过滤：沿用 mode3 等模式的做法（通过 rows[i].date 直接比字符串前10位）
    st = str(start_date).strip()[:10] if start_date else ""
    ed = str(end_date).strip()[:10] if end_date else ""
    signals: List[int] = []
    for i in range(low_window + 5, len(rows)):
        d = str(rows[i].date)[:10]
        if st and d < st:
            continue
        if ed and d > ed:
            continue
        det = _mode93_anchor_detail(
            rows,
            i,
            code="",  # 占位：真正判断涨停需要 code/name，在 scan_with_mode3 内会传入
            name="",
            lookback_days=lookback_days,
            low_window=low_window,
            low_recent_days=low_recent_days,
            vol_mult=vol_mult,
            pullback_min=pullback_min,
            pullback_max=pullback_max,
            pullback_max_days=pullback_max_days,
        )
        # 这里无法判涨停（缺 code/name），因此 signals 由 scan_with_mode3 内再判定更合理
        # 保留占位，避免被误用；实际 scan_with_mode3 会走 _mode93_anchor_detail 完整判定
        if det:
            signals.append(i)
    return signals


def _score_mode93(
    rows: List[KlineRow],
    idx: int,
    ma10: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
    vol20: np.ndarray,
    code: str = "",
    name: str = "",
    breakdown: Optional[List[Tuple[str, int]]] = None,
    *,
    lookback_days: int = 20,
    low_window: int = 120,
    low_recent_days: int = 10,
    vol_mult: float = 3.0,
    pullback_min: float = 0.99,
    pullback_max: float = 1.02,
    pullback_max_days: int = 20,
) -> int:
    """mode93 评分（0~100）：量比越大越好、回调越贴近A越好。"""
    _ = (ma10, ma20, ma60, vol20)  # 与其他 score_fn 签名保持一致；mode93 自身不依赖这些数组
    det = _mode93_anchor_detail(
        rows,
        idx,
        code,
        name,
        lookback_days=lookback_days,
        low_window=low_window,
        low_recent_days=low_recent_days,
        vol_mult=vol_mult,
        pullback_min=pullback_min,
        pullback_max=pullback_max,
        pullback_max_days=pullback_max_days,
    )
    if not det:
        return 0
    vmult = float(det.get("vol_mult") or 0.0)
    pull = float(det.get("pullback_pct") or 0.0)
    A = float(det.get("A") or 0.0)

    score = 70
    score += int(min(18.0, max(0.0, (vmult - float(vol_mult)) * 5.0)))
    score += int(min(12.0, max(0.0, (2.0 - abs(pull)) * 6.0)))
    score = int(max(0, min(100, score)))

    if breakdown is not None:
        breakdown.append(
            (f"低位{low_window}日最低(近{low_recent_days}日)→次日放量涨停(量比{vmult:.2f}x)", 0)
        )
        breakdown.append((f"回调到涨停日低点A附近(A={A:.2f},偏离{pull:.2f}%)", 0))
    return int(score)


def _is_big_yang_row_modebbd(
    r: KlineRow,
    code: str,
    name: str,
    *,
    big_pct_min: float,
    body_ratio_min: float,
) -> bool:
    o, c, h, l_ = float(r.open), float(r.close), float(r.high), float(r.low)
    if c <= o:
        return False
    pct = float(getattr(r, "pct_chg", 0.0) or 0.0)
    if pct < big_pct_min:
        return False
    if pct >= _limit_rate(code, name) * 100 - 0.6:
        return False
    rng = h - l_
    if rng <= 0:
        return False
    if (c - o) / rng < body_ratio_min:
        return False
    return True


def _match_mode_bottom_big_yang(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    *,
    low_lookback: int = 60,
    bottom_pos_max: float = 0.50,
    big_pct_min: float = 5.0,
    body_ratio_min: float = 0.55,
    vol_mult: float = 2.0,
    vol_ma: int = 20,
    sudden_days: int = 5,
    prior_vol_ratio_max: float = 0.65,
) -> Optional[Dict[str, float]]:
    n = len(rows)
    need = max(low_lookback + 1, vol_ma + 1, sudden_days + 2)
    if idx < need or idx >= n:
        return None
    r = rows[idx]
    if not _is_big_yang_row_modebbd(
        r, code, name, big_pct_min=big_pct_min, body_ratio_min=body_ratio_min
    ):
        return None
    close = float(r.close)
    volume = float(r.volume)
    high_arr = np.array([float(x.high) for x in rows], dtype=float)
    low_arr = np.array([float(x.low) for x in rows], dtype=float)
    vol_arr = np.array([float(x.volume) for x in rows], dtype=float)
    seg_lo = idx - low_lookback
    h_max = float(np.max(high_arr[seg_lo : idx + 1]))
    l_min = float(np.min(low_arr[seg_lo : idx + 1]))
    rng = h_max - l_min
    if rng <= 0:
        return None
    pos = (close - l_min) / rng
    if pos > bottom_pos_max:
        return None
    v_prev = float(vol_arr[idx - 1]) if idx >= 1 else 0.0
    v_ma = float(np.mean(vol_arr[idx - vol_ma : idx])) if idx >= vol_ma else 0.0
    v_base = max(v_prev, v_ma)
    if v_base <= 0 or volume < vol_mult * v_base:
        return None
    for j in range(idx - sudden_days, idx):
        if j < 0:
            continue
        if _is_big_yang_row_modebbd(
            rows[j], code, name, big_pct_min=big_pct_min, body_ratio_min=body_ratio_min
        ):
            return None
    if sudden_days > 0 and idx >= sudden_days:
        v_prior = float(np.mean(vol_arr[idx - sudden_days : idx]))
        if v_prior > prior_vol_ratio_max * volume:
            return None
    o, c, h, l_ = float(r.open), float(r.close), float(r.high), float(r.low)
    rng_d = h - l_
    body_ratio = (c - o) / rng_d if rng_d > 0 else 0.0
    pct = float(getattr(r, "pct_chg", 0.0) or 0.0)
    return {
        "close": close,
        "low_pos_pct": pos * 100.0,
        "range_low": l_min,
        "range_high": h_max,
        "vol_today": volume,
        "vol_base": v_base,
        "vol_ratio": volume / v_base,
        "pct_chg": pct,
        "body_ratio": body_ratio,
    }


def _score_mode_bottom_big_yang(
    rows: List[KlineRow],
    idx: int,
    ma10: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
    vol20: np.ndarray,
    code: str = "",
    name: str = "",
    breakdown: Optional[List[tuple]] = None,
    *,
    low_lookback: int = 60,
    bottom_pos_max: float = 0.50,
    big_pct_min: float = 5.0,
    body_ratio_min: float = 0.55,
    vol_mult: float = 2.0,
    vol_ma: int = 20,
    sudden_days: int = 5,
    prior_vol_ratio_max: float = 0.65,
) -> int:
    _ = (ma10, ma20, ma60, vol20)
    det = _match_mode_bottom_big_yang(
        rows,
        idx,
        code,
        name,
        low_lookback=low_lookback,
        bottom_pos_max=bottom_pos_max,
        big_pct_min=big_pct_min,
        body_ratio_min=body_ratio_min,
        vol_mult=vol_mult,
        vol_ma=vol_ma,
        sudden_days=sudden_days,
        prior_vol_ratio_max=prior_vol_ratio_max,
    )
    if not det:
        return 0
    vr = float(det["vol_ratio"])
    pct = float(det["pct_chg"])
    score = int(min(100, max(0, round(55 + vr * 8 + pct * 1.5))))
    if breakdown is not None:
        breakdown.append(
            (
                f"低位{low_lookback}日位置{det['low_pos_pct']:.1f}% 量比{vr:.2f} 涨{pct:.2f}%",
                0,
            )
        )
    return score


def _is_launch_big_yang_row(
    r: KlineRow,
    code: str,
    name: str,
    *,
    big_pct_min: float,
    body_ratio_min: float,
) -> bool:
    """起量锚点大阳：允许涨停起量（不限于未封板）。"""
    o, c, h, l_ = float(r.open), float(r.close), float(r.high), float(r.low)
    if c <= o:
        return False
    pct = float(getattr(r, "pct_chg", 0.0) or 0.0)
    if pct < big_pct_min:
        return False
    rng = h - l_
    if rng <= 0:
        return False
    return (c - o) / rng >= body_ratio_min


def _launch_bottom_big_yang_detail(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    *,
    low_lookback: int = 60,
    bottom_pos_max: float = 0.50,
    big_pct_min: float = 5.0,
    body_ratio_min: float = 0.55,
    vol_mult: float = 2.0,
    vol_ma: int = 20,
    sudden_days: int = 5,
    prior_vol_ratio_max: float = 0.65,
) -> Optional[Dict[str, float]]:
    """起量锚点：低位放量大阳线，以 min(open,low) 衡量底部位置（允许收盘拉到箱顶）。"""
    n = len(rows)
    need = max(low_lookback + 1, vol_ma + 1, sudden_days + 2)
    if idx < need or idx >= n:
        return None
    r = rows[idx]
    if not _is_launch_big_yang_row(
        r, code, name, big_pct_min=big_pct_min, body_ratio_min=body_ratio_min
    ):
        return None
    o, c, h, l_ = float(r.open), float(r.close), float(r.high), float(r.low)
    launch = min(o, l_)
    high_arr = np.array([float(x.high) for x in rows], dtype=float)
    low_arr = np.array([float(x.low) for x in rows], dtype=float)
    vol_arr = np.array([float(x.volume) for x in rows], dtype=float)
    seg_lo = idx - low_lookback
    h_max = float(np.max(high_arr[seg_lo : idx + 1]))
    l_min = float(np.min(low_arr[seg_lo : idx + 1]))
    rng = h_max - l_min
    if rng <= 0:
        return None
    pos = (launch - l_min) / rng
    if pos > bottom_pos_max:
        return None
    volume = float(r.volume)
    v_prev = float(vol_arr[idx - 1]) if idx >= 1 else 0.0
    v_ma = float(np.mean(vol_arr[idx - vol_ma : idx])) if idx >= vol_ma else 0.0
    v_base = max(v_prev, v_ma)
    if v_base <= 0 or volume < vol_mult * v_base:
        return None
    for j in range(idx - sudden_days, idx):
        if j < 0:
            continue
        if _is_launch_big_yang_row(
            rows[j], code, name, big_pct_min=big_pct_min, body_ratio_min=body_ratio_min
        ):
            return None
    if sudden_days > 0 and idx >= sudden_days:
        v_prior = float(np.mean(vol_arr[idx - sudden_days : idx]))
        if v_prior > prior_vol_ratio_max * volume:
            return None
    rng_d = h - l_
    pct = float(getattr(r, "pct_chg", 0.0) or 0.0)
    return {
        "support": launch,
        "close": c,
        "low_pos_pct": pos * 100.0,
        "vol_ratio": volume / v_base,
        "pct_chg": pct,
        "body_ratio": (c - o) / rng_d if rng_d > 0 else 0.0,
    }


def _find_mode_bottom_support_anchor(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    *,
    anchor_days_min: int,
    anchor_days_max: int,
    low_lookback: int,
    bottom_pos_max: float,
    vol_mult: float,
    vol_ma: int,
    big_pct_min: float,
    body_ratio_min: float,
    min_rally_pct: float,
    support_near_max: float,
    support_break_min: float,
    test_tol: float,
    min_support_tests: int,
    bounce_days: int,
    weekly_vol_mult: float,
) -> Optional[int]:
    """锚点：信号前窗口内，起量支撑位与当前价最接近且历史验证有效的低位放量大阳。"""
    lo = max(low_lookback + 1, vol_ma + 1, idx - anchor_days_max)
    hi = idx - anchor_days_min
    if hi < lo:
        return None
    sig_low = float(rows[idx].low)
    best: Optional[Tuple[float, float, int]] = None
    for j in range(lo, hi + 1):
        det = _launch_bottom_big_yang_detail(
            rows,
            j,
            code,
            name,
            low_lookback=low_lookback,
            bottom_pos_max=bottom_pos_max,
            big_pct_min=big_pct_min,
            body_ratio_min=body_ratio_min,
            vol_mult=vol_mult,
            vol_ma=vol_ma,
        )
        if det is None:
            continue
        if not _weekly_anchor_vol_ok(rows, j, weekly_vol_mult):
            continue
        support = float(det["support"])
        if support <= 0:
            continue
        dist = (sig_low - support) / support
        if dist > support_near_max or dist < 0 or sig_low < support * support_break_min:
            continue
        tests, rally, _ = _count_bottom_support_tests(
            rows,
            j,
            idx,
            support,
            min_rally_pct=min_rally_pct,
            test_tol=test_tol,
            break_min=support_break_min,
            bounce_days=bounce_days,
        )
        if tests < min_support_tests or rally < min_rally_pct * 100.0:
            continue
        key = (dist, -float(det["vol_ratio"]))
        if best is None or key < (best[0], best[1]):
            best = (dist, -float(det["vol_ratio"]), j)
    return best[2] if best else None


def _count_bottom_support_tests(
    rows: List[KlineRow],
    anchor_idx: int,
    signal_idx: int,
    support: float,
    *,
    min_rally_pct: float,
    test_tol: float,
    break_min: float,
    bounce_days: int,
) -> Tuple[int, float, float]:
    """锚点后、信号前：统计起量位有效回踩并反弹次数；返回 (次数, 最大涨幅%, 最近测试距今天数)。"""
    if support <= 0 or signal_idx <= anchor_idx + 1:
        return 0, 0.0, 9999.0
    high_arr = np.array([float(x.high) for x in rows], dtype=float)
    low_arr = np.array([float(x.low) for x in rows], dtype=float)
    close_arr = np.array([float(x.close) for x in rows], dtype=float)
    seg_high = float(np.max(high_arr[anchor_idx:signal_idx]))
    max_rally = (seg_high - support) / support * 100.0
    if max_rally < min_rally_pct * 100.0:
        return 0, max_rally, 9999.0
    tests = 0
    last_test_gap = 9999.0
    j = anchor_idx + 1
    while j < signal_idx:
        lo = float(low_arr[j])
        if lo > support * (1.0 + test_tol) or lo < support * break_min:
            j += 1
            continue
        base_close = float(close_arr[j])
        bounced = False
        for k in range(j + 1, min(signal_idx, j + bounce_days + 1)):
            if float(close_arr[k]) > base_close:
                bounced = True
                break
        if bounced:
            tests += 1
            last_test_gap = min(last_test_gap, float(signal_idx - j))
            j += bounce_days + 1
        else:
            j += 1
    return tests, max_rally, last_test_gap


def _weekly_anchor_vol_ok(
    rows: List[KlineRow],
    anchor_idx: int,
    weekly_vol_mult: float,
) -> bool:
    """锚点所在周为起量周：周量 >= weekly_vol_mult × 前20周均量。"""
    if weekly_vol_mult <= 0:
        return True
    wk, widx = daily_to_weekly_with_volume_and_last_index(rows)
    wi = next((i for i, end_i in enumerate(widx) if end_i >= anchor_idx), None)
    if wi is None:
        return False
    vols = [float(w[5]) for w in wk]
    if wi < 20:
        return vols[wi] >= weekly_vol_mult * float(np.mean(vols[: max(1, wi)]))
    avg = float(np.mean(vols[wi - 20 : wi]))
    if avg <= 0:
        return False
    return vols[wi] >= weekly_vol_mult * avg


def _match_mode_bottom_support(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    *,
    anchor_days_min: int = 30,
    anchor_days_max: int = 200,
    low_lookback: int = 60,
    bottom_pos_max: float = 0.50,
    anchor_vol_mult: float = 2.0,
    anchor_vol_ma: int = 20,
    big_pct_min: float = 5.0,
    body_ratio_min: float = 0.55,
    min_rally_pct: float = 0.15,
    support_near_max: float = 0.15,
    support_break_min: float = 0.97,
    test_tol: float = 0.15,
    min_support_tests: int = 1,
    bounce_days: int = 5,
    weekly_vol_mult: float = 1.5,
) -> Optional[Dict[str, float]]:
    """mode底部支撑：起量大阳支撑→拉升→回踩验证→当前再次回踩支撑（抄底）。"""
    n = len(rows)
    need = max(
        anchor_days_max + 1,
        low_lookback + 1,
        anchor_vol_ma + 1,
        bounce_days + 2,
    )
    if idx < need or idx >= n:
        return None

    i_anchor = _find_mode_bottom_support_anchor(
        rows,
        idx,
        code,
        name,
        anchor_days_min=anchor_days_min,
        anchor_days_max=anchor_days_max,
        low_lookback=low_lookback,
        bottom_pos_max=bottom_pos_max,
        vol_mult=anchor_vol_mult,
        vol_ma=anchor_vol_ma,
        big_pct_min=big_pct_min,
        body_ratio_min=body_ratio_min,
        min_rally_pct=min_rally_pct,
        support_near_max=support_near_max,
        support_break_min=support_break_min,
        test_tol=test_tol,
        min_support_tests=min_support_tests,
        bounce_days=bounce_days,
        weekly_vol_mult=weekly_vol_mult,
    )
    if i_anchor is None:
        return None

    r = rows[idx]
    r_a = rows[i_anchor]
    launch_det = _launch_bottom_big_yang_detail(
        rows,
        i_anchor,
        code,
        name,
        low_lookback=low_lookback,
        bottom_pos_max=bottom_pos_max,
        big_pct_min=big_pct_min,
        body_ratio_min=body_ratio_min,
        vol_mult=anchor_vol_mult,
        vol_ma=anchor_vol_ma,
    )
    if launch_det is None:
        return None
    support = float(launch_det["support"])

    tests, max_rally, last_test_gap = _count_bottom_support_tests(
        rows,
        i_anchor,
        idx,
        support,
        min_rally_pct=min_rally_pct,
        test_tol=test_tol,
        break_min=support_break_min,
        bounce_days=bounce_days,
    )
    if tests < min_support_tests:
        return None

    lo = float(r.low)
    close = float(r.close)
    if lo > support * (1.0 + support_near_max) or lo < support * support_break_min:
        return None
    if close < support * support_break_min:
        return None

    dist_pct = (lo - support) / support * 100.0
    close_dist = (close - support) / support * 100.0
    anchor_vr = float(launch_det["vol_ratio"])
    pct = float(getattr(r, "pct_chg", 0.0) or 0.0)

    return {
        "anchor_date_idx": float(i_anchor),
        "support": support,
        "anchor_close": float(r_a.close),
        "anchor_vol_ratio": anchor_vr,
        "anchor_pct_chg": float(getattr(r_a, "pct_chg", 0.0) or 0.0),
        "phase_days": float(idx - i_anchor),
        "support_tests": float(tests),
        "max_rally_pct": max_rally,
        "last_test_gap_days": last_test_gap,
        "low_dist_pct": dist_pct,
        "close_dist_pct": close_dist,
        "close": close,
        "pct_chg": pct,
    }


def _score_mode_bottom_support(
    rows: List[KlineRow],
    idx: int,
    ma10: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
    vol20: np.ndarray,
    code: str = "",
    name: str = "",
    breakdown: Optional[List[tuple]] = None,
    **kwargs,
) -> int:
    _ = (ma10, ma20, ma60, vol20, kwargs)
    det = _match_mode_bottom_support(rows, idx, code, name)
    if not det:
        return 0
    tests = int(det["support_tests"])
    dist = float(det["low_dist_pct"])
    rally = float(det["max_rally_pct"])
    vr = float(det["anchor_vol_ratio"])
    score = int(
        min(
            100,
            max(
                0,
                round(
                    50
                    + tests * 8
                    + max(0.0, 12.0 - abs(dist)) * 1.5
                    + min(rally, 50) * 0.3
                    + vr * 3
                ),
            ),
        )
    )
    if breakdown is not None:
        i_a = int(det["anchor_date_idx"])
        anchor_date = str(rows[i_a].date)[:10]
        breakdown.append(
            (
                f"锚点{anchor_date} 支撑{det['support']:.2f} "
                f"距支撑+{dist:.1f}% 验证{tests}次 最大拉升+{rally:.1f}%",
                0,
            )
        )
    return score


def _find_final_shakeout_trough(
    rows: List[KlineRow],
    idx: int,
    *,
    shakeout_days_min: int,
    shakeout_days_max: int,
) -> Optional[int]:
    """震仓低点：信号前 1～shakeout_days_max 日内最低低点（不含信号日）。"""
    lo = max(1, idx - shakeout_days_max)
    hi = idx - 1
    if hi < lo:
        return None
    low_arr = np.array([float(x.low) for x in rows], dtype=float)
    best_i = int(lo + np.argmin(low_arr[lo : hi + 1]))
    if idx - best_i < 1:
        return None
    return best_i


def _is_volume_surge_anchor_row(
    r: KlineRow,
    *,
    pct_min: float = 3.0,
) -> bool:
    o, c = float(r.open), float(r.close)
    if c <= o:
        return False
    pct = float(getattr(r, "pct_chg", 0.0) or 0.0)
    return pct >= pct_min


_WEEKLY_BUNDLE_CACHE: Dict[int, Tuple[List[tuple], List[int]]] = {}


def _get_weekly_bundle(rows: List[KlineRow]) -> Tuple[List[tuple], List[int]]:
    key = id(rows)
    cached = _WEEKLY_BUNDLE_CACHE.get(key)
    if cached is not None and len(cached[0]) > 0:
        return cached
    bundle = daily_to_weekly_with_volume_and_last_index(rows)
    _WEEKLY_BUNDLE_CACHE[key] = bundle
    if len(_WEEKLY_BUNDLE_CACHE) > 512:
        _WEEKLY_BUNDLE_CACHE.clear()
    return bundle


def _weekly_bar_partial(rows: List[KlineRow], idx: int) -> Optional[Tuple[int, float, float, float, float, float]]:
    """返回 (week_i, open, high, low, close, volume) 为 idx 所在周的截至 idx 的聚合 K 线。"""
    weekly, last_idx = _get_weekly_bundle(rows)
    if not weekly:
        return None
    wi: Optional[int] = None
    for i, li in enumerate(last_idx):
        if li >= idx and (i == 0 or last_idx[i - 1] < idx):
            wi = i
            break
    if wi is None:
        return None
    start_i = last_idx[wi - 1] + 1 if wi > 0 else 0
    seg = rows[start_i : idx + 1]
    if not seg:
        return None
    o = float(seg[0].open)
    h = max(float(r.high) for r in seg)
    l = min(float(r.low) for r in seg)
    c = float(seg[-1].close)
    v = sum(float(getattr(r, "volume", 0) or 0) for r in seg)
    return wi, o, h, l, c, v


def _match_mode_final_shakeout_weekly(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    *,
    shakeout_weeks_min: int = 2,
    shakeout_weeks_max: int = 4,
    peak_weeks_back: int = 12,
    shakeout_drop_min: float = 0.08,
    shakeout_drop_max: float = 0.38,
    shakeout_vol_max: float = 1.05,
    accum_lookback: int = 16,
    accum_vol_mult: float = 1.0,
    accum_pct_min: float = 3.0,
    signal_pct_min: float = 3.0,
    signal_strong_pct: float = 7.0,
    signal_vol_mult: float = 1.25,
    min_rally_pct: float = 0.08,
    ma60_slope_days: int = 20,
    breakout_pct_min: float = 15.0,
    breakout_pct_min_main: float = 9.0,
    breakout_vol_min: float = 2.5,
    vol_ma: int = 20,
) -> Optional[Dict[str, float]]:
    """周线最后震仓：前期放量介入→2~4周缩量回落→当周反包/突破。"""
    n = len(rows)
    need = max(accum_lookback + 5, ma60_slope_days + 60, vol_ma + 1)
    if idx < need or idx >= n:
        return None

    ctx = _weekly_bar_partial(rows, idx)
    if ctx is None:
        return None
    wi, sig_o, sig_h, sig_l, sig_c, sig_v = ctx
    if sig_c <= sig_o:
        return None

    weekly, last_idx = _get_weekly_bundle(rows)
    # 周线买点：默认仅在该周最后一个交易日确认（避免同一周重复触发）
    if idx != last_idx[wi]:
        return None
    if wi < shakeout_weeks_min + 2 or wi >= len(weekly):
        return None

    vols = np.array([float(w[5]) for w in weekly], dtype=float)
    vol_ma10 = _moving_mean(vols, 10)

    sh_start = max(0, wi - shakeout_weeks_max)
    sh_end = wi - 1
    if sh_end - sh_start + 1 < shakeout_weeks_min:
        return None

    highs = np.array([float(w[2]) for w in weekly], dtype=float)
    lows = np.array([float(w[3]) for w in weekly], dtype=float)

    # 震仓低：信号前若干完成周内最低
    trough_wi = int(sh_start + np.argmin(lows[sh_start : sh_end + 1]))
    trough_low = float(lows[trough_wi])
    if trough_low <= 0:
        return None

    # 箱顶：震仓低前 1~8 周内最高（近期平台顶，非数月前旧高）
    peak_back = min(8, max(2, trough_wi - sh_start + 2))
    peak_lo = max(0, trough_wi - peak_back)
    peak_hi = max(peak_lo, trough_wi - 1)
    if peak_hi <= peak_lo:
        return None
    peak_wi = int(peak_lo + np.argmax(highs[peak_lo : peak_hi + 1]))
    peak_high = float(highs[peak_wi])
    if peak_high <= 0 or peak_wi >= trough_wi:
        return None

    drop_pct = (peak_high - trough_low) / peak_high
    if drop_pct < shakeout_drop_min or drop_pct > shakeout_drop_max:
        return None

    shakeout_vols: List[float] = []
    low_vol_weeks = 0
    peak_vol = float(vols[peak_wi])
    for j in range(peak_wi + 1, wi):
        vj = float(vols[j])
        shakeout_vols.append(vj)
        vma = (
            float(vol_ma10[j])
            if j < len(vol_ma10) and not np.isnan(vol_ma10[j])
            else 0.0
        )
        vol_shrink = peak_vol > 0 and vj <= peak_vol * 0.80
        vol_quiet = vma > 0 and vj <= vma * shakeout_vol_max
        if vol_quiet or vol_shrink:
            low_vol_weeks += 1
    shakeout_avg_vol = float(np.mean(shakeout_vols)) if shakeout_vols else 0.0
    shakeout_avg_quiet = (
        peak_vol > 0 and shakeout_avg_vol <= peak_vol * shakeout_vol_max
    )
    if len(shakeout_vols) < shakeout_weeks_min:
        return None
    if low_vol_weeks < shakeout_weeks_min and not shakeout_avg_quiet:
        return None

    accum_lo = max(0, wi - accum_lookback)
    accum_hi = max(accum_lo, peak_wi - 1)
    anchor_wi: Optional[int] = None
    anchor_vr = 0.0
    for j in range(accum_lo, accum_hi + 1):
        o, c, v = float(weekly[j][1]), float(weekly[j][4]), float(vols[j])
        if o <= 0:
            continue
        pct_w = (c - o) / o * 100.0
        vma = float(vol_ma10[j]) if j < len(vol_ma10) and not np.isnan(vol_ma10[j]) else 0.0
        if vma <= 0:
            continue
        vr = v / vma
        yang = c > o
        hit = (yang and pct_w >= accum_pct_min and vr >= accum_vol_mult) or vr >= max(
            accum_vol_mult * 1.2, 1.25
        )
        if hit and (anchor_wi is None or vr > anchor_vr):
            anchor_wi = j
            anchor_vr = vr
    if anchor_wi is None:
        return None

    anchor_support = float(min(weekly[anchor_wi][1], weekly[anchor_wi][3]))
    if anchor_support <= 0:
        return None
    rally_pct = (peak_high - anchor_support) / anchor_support
    if rally_pct < min_rally_pct:
        return None

    sig_pct_w = (sig_c - sig_o) / sig_o * 100.0 if sig_o > 0 else 0.0
    sig_vol_ok = shakeout_avg_vol > 0 and sig_v >= shakeout_avg_vol * signal_vol_mult
    prior_high = float(np.max(highs[max(sh_start, wi - 2) : wi])) if wi >= 1 else 0.0
    price_break = prior_high > 0 and sig_c > prior_high
    if sig_pct_w < signal_strong_pct and not (
        sig_pct_w >= signal_pct_min and sig_vol_ok
    ) and not price_break:
        return None

    close_arr = np.array([float(x.close) for x in rows], dtype=float)
    ma20 = _moving_mean(close_arr, 20)
    ma60 = _moving_mean(close_arr, 60)
    if np.isnan(ma60[idx]) or np.isnan(ma60[idx - ma60_slope_days]):
        return None
    if ma60[idx] < ma60[idx - ma60_slope_days] * 0.95:
        return None

    daily_vr = _vol_ratio_at(rows, idx, vol_ma)
    pct_d = float(getattr(rows[idx], "pct_chg", 0.0) or 0.0)
    brk_pct_min = _modepbs_big_pct_threshold(
        code, name, big_pct_min=breakout_pct_min, big_pct_min_main=breakout_pct_min_main
    )
    breakout_candidate = (
        (sig_pct_w >= brk_pct_min or pct_d >= brk_pct_min)
        and daily_vr >= breakout_vol_min
        and not np.isnan(ma20[idx])
        and sig_c > ma20[idx]
        and sig_c > peak_high
    )
    reversal_candidate = sig_pct_w >= signal_strong_pct or (
        sig_pct_w >= signal_pct_min and sig_vol_ok
    ) or (
        idx >= 1 and sig_c > float(rows[idx - 1].high) and daily_vr >= 1.2
    )
    if breakout_candidate:
        signal_type = "breakout"
    elif reversal_candidate:
        signal_type = "reversal"
    else:
        return None

    rng_d = sig_h - sig_l
    body_ratio = (sig_c - sig_o) / rng_d if rng_d > 0 else 0.0
    anchor_i = int(last_idx[anchor_wi])
    trough_i = int(last_idx[trough_wi])
    peak_i = int(last_idx[peak_wi])

    return {
        "signal_type": 1.0 if signal_type == "breakout" else 0.0,
        "weekly_path": 1.0,
        "anchor_date_idx": float(anchor_i),
        "trough_date_idx": float(trough_i),
        "peak_date_idx": float(peak_i),
        "anchor_support": anchor_support,
        "anchor_vol_ratio": anchor_vr,
        "peak_high": peak_high,
        "trough_low": trough_low,
        "shakeout_drop_pct": drop_pct * 100.0,
        "phase_low": trough_low,
        "rally_from_anchor_pct": rally_pct * 100.0,
        "phase_days": float(idx - anchor_i),
        "shakeout_days": float((trough_wi - peak_wi) * 5),
        "shakeout_weeks": float(trough_wi - peak_wi),
        "vol_ratio": daily_vr if daily_vr > 0 else (sig_v / shakeout_avg_vol if shakeout_avg_vol > 0 else 0.0),
        "weekly_vol_ratio": sig_v / shakeout_avg_vol if shakeout_avg_vol > 0 else 0.0,
        "pct_chg": pct_d if abs(pct_d) > 0.01 else sig_pct_w,
        "weekly_pct": sig_pct_w,
        "body_ratio": body_ratio,
        "close": sig_c,
        "ma20": float(ma20[idx]) if not np.isnan(ma20[idx]) else 0.0,
        "ma60": float(ma60[idx]) if not np.isnan(ma60[idx]) else 0.0,
    }


def _match_mode_final_shakeout_daily(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    *,
    phase_days_min: int = 30,
    phase_days_max: int = 90,
    anchor_vol_mult: float = 1.5,
    min_rally_pct: float = 0.10,
    consolid_days: int = 20,
    consolid_amp_max: float = 0.15,
    peak_lookback: int = 15,
    shakeout_days_min: int = 3,
    shakeout_days_max: int = 7,
    shakeout_drop_min: float = 0.10,
    shakeout_drop_max: float = 0.22,
    phase_low_lookback: int = 90,
    phase_low_break_min: float = 0.95,
    shakeout_vol_min: float = 0.6,
    shakeout_vol_max: float = 2.5,
    ma60_slope_days: int = 20,
    reversal_pct_min: float = 8.0,
    reversal_vol_min: float = 1.5,
    reversal_low_tol: float = 0.05,
    breakout_pct_min: float = 15.0,
    breakout_pct_min_main: float = 9.0,
    breakout_vol_min: float = 3.0,
    body_ratio_min: float = 0.55,
    vol_ma: int = 20,
) -> Optional[Dict[str, float]]:
    """mode最后震仓：起量→箱体→最后洗盘→反包/突破买点。"""
    n = len(rows)
    need = max(
        phase_low_lookback + 1,
        phase_days_max + 1,
        peak_lookback + shakeout_days_max + 5,
        vol_ma + 1,
        ma60_slope_days + 60,
    )
    if idx < need or idx >= n:
        return None

    r = rows[idx]
    if not _is_launch_big_yang_row(
        r, code, name, big_pct_min=min(reversal_pct_min, breakout_pct_min_main), body_ratio_min=body_ratio_min
    ):
        return None

    trough_i = _find_final_shakeout_trough(
        rows, idx, shakeout_days_min=shakeout_days_min, shakeout_days_max=shakeout_days_max
    )
    if trough_i is None:
        return None

    high_arr = np.array([float(x.high) for x in rows], dtype=float)
    low_arr = np.array([float(x.low) for x in rows], dtype=float)
    close_arr = np.array([float(x.close) for x in rows], dtype=float)
    vol_arr = np.array([float(x.volume) for x in rows], dtype=float)

    trough_low = float(low_arr[trough_i])
    if trough_low <= 0:
        return None

    peak_lo = max(0, trough_i - peak_lookback)
    peak_hi = max(peak_lo, trough_i - 2)
    if peak_hi <= peak_lo:
        return None
    peak_i = int(peak_lo + np.argmax(high_arr[peak_lo : peak_hi + 1]))
    peak_high = float(high_arr[peak_i])
    if peak_high <= 0:
        return None

    drop_pct = (peak_high - trough_low) / peak_high
    if drop_pct < shakeout_drop_min or drop_pct > shakeout_drop_max:
        return None
    if trough_i - peak_i < shakeout_days_min or trough_i - peak_i > shakeout_days_max + 5:
        return None

    phase_lo_i = max(0, trough_i - phase_low_lookback)
    phase_low = float(np.min(low_arr[phase_lo_i:trough_i]))
    if phase_low <= 0 or trough_low < phase_low * phase_low_break_min:
        return None

    if trough_i >= consolid_days:
        c_seg = rows[trough_i - consolid_days : trough_i]
        mean_c = float(np.mean([float(x.close) for x in c_seg]))
        if mean_c > 0:
            amp = (
                max(float(x.high) for x in c_seg) - min(float(x.low) for x in c_seg)
            ) / mean_c
            if amp > consolid_amp_max:
                return None

    # 震仓期：起量后拉升 + 起量锚点
    search_lo = max(vol_ma + 1, idx - phase_days_max)
    search_hi = max(search_lo, trough_i - phase_days_min)
    anchor_i: Optional[int] = None
    anchor_vr = 0.0
    for j in range(search_lo, search_hi + 1):
        det = _launch_bottom_big_yang_detail(
            rows, j, code, name, vol_mult=anchor_vol_mult, vol_ma=vol_ma
        )
        if det is None:
            vr = _vol_ratio_at(rows, j, vol_ma)
            if vr >= anchor_vol_mult and _is_volume_surge_anchor_row(rows[j], pct_min=3.0):
                det = {"vol_ratio": vr}
            else:
                continue
        else:
            vr = float(det["vol_ratio"])
        support_j = float(min(rows[j].open, rows[j].low))
        if support_j <= 0:
            continue
        seg_high = float(np.max(high_arr[j : peak_i + 1]))
        rally_j = (seg_high - support_j) / support_j
        if rally_j < min_rally_pct:
            continue
        if anchor_i is None or vr > anchor_vr:
            anchor_i = j
            anchor_vr = vr
    if anchor_i is None:
        return None

    anchor_support = float(min(rows[anchor_i].open, rows[anchor_i].low))
    if anchor_support <= 0:
        return None
    seg_high = float(np.max(high_arr[anchor_i : peak_i + 1]))
    rally_pct = (seg_high - anchor_support) / anchor_support
    if rally_pct < min_rally_pct:
        return None

    # 震仓期量能温和
    for j in range(peak_i + 1, trough_i + 1):
        if j <= 0 or j >= n:
            continue
        vr = _vol_ratio_at(rows, j, vol_ma)
        if vr > 0 and (vr < shakeout_vol_min or vr > shakeout_vol_max):
            return None

    # MA60 中期趋势未坏
    ma60 = _moving_mean(close_arr, 60)
    if np.isnan(ma60[idx]) or np.isnan(ma60[idx - ma60_slope_days]):
        return None
    if ma60[idx] < ma60[idx - ma60_slope_days] * 0.995:
        return None

    ma5 = _moving_mean(close_arr, 5)
    ma10 = _moving_mean(close_arr, 10)
    ma20 = _moving_mean(close_arr, 20)
    washed = False
    for j in range(peak_i + 1, idx):
        if not np.isnan(ma5[j]) and not np.isnan(ma10[j]) and ma5[j] < ma10[j]:
            washed = True
            break
    if not washed:
        return None

    pct = float(getattr(r, "pct_chg", 0.0) or 0.0)
    vol_ratio = _vol_ratio_at(rows, idx, vol_ma)
    sig_low = float(r.low)
    sig_close = float(r.close)

    brk_pct_min = _modepbs_big_pct_threshold(
        code, name, big_pct_min=breakout_pct_min, big_pct_min_main=breakout_pct_min_main
    )
    breakout_candidate = (
        pct >= brk_pct_min
        and vol_ratio >= breakout_vol_min
        and not np.isnan(ma20[idx])
        and sig_close > ma20[idx]
        and sig_close > peak_high
    )
    low_tol = 0.15 if breakout_candidate else reversal_low_tol
    if sig_low > trough_low * (1.0 + low_tol):
        return None

    signal_type = ""
    if breakout_candidate:
        signal_type = "breakout"
    elif (
        pct >= reversal_pct_min
        and vol_ratio >= reversal_vol_min
        and idx >= 1
        and sig_close > float(rows[idx - 1].high)
    ):
        signal_type = "reversal"
    else:
        return None

    o, c, h, l_ = float(r.open), float(r.close), float(r.high), float(r.low)
    rng_d = h - l_
    body_ratio = (c - o) / rng_d if rng_d > 0 else 0.0

    return {
        "signal_type": 1.0 if signal_type == "breakout" else 0.0,
        "weekly_path": 0.0,
        "anchor_date_idx": float(anchor_i),
        "trough_date_idx": float(trough_i),
        "peak_date_idx": float(peak_i),
        "anchor_support": anchor_support,
        "anchor_vol_ratio": anchor_vr,
        "peak_high": peak_high,
        "trough_low": trough_low,
        "shakeout_drop_pct": drop_pct * 100.0,
        "phase_low": phase_low,
        "rally_from_anchor_pct": rally_pct * 100.0,
        "phase_days": float(idx - anchor_i),
        "shakeout_days": float(trough_i - peak_i),
        "vol_ratio": vol_ratio,
        "pct_chg": pct,
        "body_ratio": body_ratio,
        "close": sig_close,
        "ma20": float(ma20[idx]) if not np.isnan(ma20[idx]) else 0.0,
        "ma60": float(ma60[idx]) if not np.isnan(ma60[idx]) else 0.0,
    }


def _match_mode_final_shakeout(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    **kwargs,
) -> Optional[Dict[str, float]]:
    """mode最后震仓：日线快路径 + 周线结构兜底。"""
    daily = _match_mode_final_shakeout_daily(rows, idx, code, name, **kwargs)
    if daily is not None:
        return daily
    return _match_mode_final_shakeout_weekly(
        rows,
        idx,
        code,
        name,
        shakeout_weeks_min=int(kwargs.get("shakeout_weeks_min", 2) or 2),
        shakeout_weeks_max=int(kwargs.get("shakeout_weeks_max", 4) or 4),
        peak_weeks_back=int(kwargs.get("peak_weeks_back", 12) or 12),
        shakeout_drop_min=float(kwargs.get("weekly_shakeout_drop_min", 0.08) or 0.08),
        shakeout_drop_max=float(kwargs.get("weekly_shakeout_drop_max", 0.38) or 0.38),
        shakeout_vol_max=float(kwargs.get("weekly_shakeout_vol_max", 1.05) or 1.05),
        accum_lookback=int(kwargs.get("accum_lookback", 16) or 16),
        accum_vol_mult=float(kwargs.get("accum_vol_mult", 1.0) or 1.0),
        accum_pct_min=float(kwargs.get("accum_pct_min", 3.0) or 3.0),
        signal_pct_min=float(kwargs.get("signal_pct_min", 3.0) or 3.0),
        signal_strong_pct=float(kwargs.get("signal_strong_pct", 7.0) or 7.0),
        signal_vol_mult=float(kwargs.get("signal_vol_mult", 1.25) or 1.25),
        min_rally_pct=float(kwargs.get("weekly_min_rally_pct", 0.08) or 0.08),
        ma60_slope_days=int(kwargs.get("ma60_slope_days", 20) or 20),
        breakout_pct_min=float(kwargs.get("breakout_pct_min", 15.0) or 15.0),
        breakout_pct_min_main=float(kwargs.get("breakout_pct_min_main", 9.0) or 9.0),
        breakout_vol_min=float(kwargs.get("breakout_vol_min", 2.5) or 2.5),
        vol_ma=int(kwargs.get("vol_ma", 20) or 20),
    )


def _score_mode_final_shakeout(
    rows: List[KlineRow],
    idx: int,
    ma10: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
    vol20: np.ndarray,
    code: str = "",
    name: str = "",
    breakdown: Optional[List[tuple]] = None,
    **kwargs,
) -> int:
    _ = (ma10, ma20, ma60, vol20, kwargs)
    det = _match_mode_final_shakeout(rows, idx, code, name)
    if not det:
        return 0
    vr = float(det["vol_ratio"])
    pct = float(det["pct_chg"])
    drop = float(det["shakeout_drop_pct"])
    is_brk = int(det["signal_type"]) == 1
    score = int(
        min(
            100,
            max(
                0,
                round(
                    48
                    + vr * 4
                    + pct * 0.8
                    + drop * 0.5
                    + (12 if is_brk else 6)
                ),
            ),
        )
    )
    if breakdown is not None:
        i_a = int(det["anchor_date_idx"])
        i_t = int(det["trough_date_idx"])
        typ = "突破" if is_brk else "反包"
        wk = "周线" if int(det.get("weekly_path", 0)) == 1 else "日线"
        sw = det.get("shakeout_weeks")
        sh_label = f"震仓{int(sw)}周" if sw else f"震仓{int(det['shakeout_days'])}日"
        breakdown.append(
            (
                f"[{wk}]锚点{str(rows[i_a].date)[:10]} {sh_label} "
                f"回撤{drop:.1f}% 低{det['trough_low']:.2f} {typ} 量比{vr:.2f} 涨{pct:.1f}%",
                0,
            )
        )
    return score


def _modepbs_big_pct_threshold(
    code: str, name: str, *, big_pct_min: float, big_pct_min_main: float
) -> float:
    """主板(10%板)用 big_pct_min_main，科创/创业板/北交所等用 big_pct_min。"""
    if _limit_rate(code, name) >= 0.15 or _is_st(name or ""):
        return big_pct_min
    return big_pct_min_main


def _is_big_yang_row_modepbs(
    r: KlineRow,
    code: str,
    name: str,
    *,
    big_pct_min: float,
    big_pct_min_main: float = 4.5,
    body_ratio_min: float,
    for_signal: bool = True,
    allow_limit_up: bool = False,
) -> bool:
    """平台突破首阳大阳线判定。

    for_signal=True 时主板用 big_pct_min_main(4.5%)；震仓/首阳回溯用 big_pct_min(7%)。
    allow_limit_up=True 时不排除涨停日（信号日可为涨停突破）。
    """
    o, c, h, l_ = float(r.open), float(r.close), float(r.high), float(r.low)
    if c <= o:
        return False
    pct = float(getattr(r, "pct_chg", 0.0) or 0.0)
    if for_signal:
        pct_min = _modepbs_big_pct_threshold(
            code, name, big_pct_min=big_pct_min, big_pct_min_main=big_pct_min_main
        )
    else:
        pct_min = big_pct_min
    if pct < pct_min:
        return False
    if not allow_limit_up:
        limit_pct = _limit_rate(code, name) * 100
        if pct >= limit_pct - 0.001:
            return False
    rng = h - l_
    if rng <= 0:
        return False
    if (c - o) / rng < body_ratio_min:
        return False
    return True


def _modepbs_wash_drop_from_peak_pct(
    high_arr: np.ndarray, low_arr: np.ndarray, i_low: int, idx: int
) -> float:
    """震仓期内自阶段高点至低点最大回撤比例。"""
    if idx <= i_low:
        return 0.0
    peak_i = i_low + int(np.argmax(high_arr[i_low:idx]))
    peak_h = float(high_arr[peak_i])
    if peak_h <= 0:
        return 0.0
    trough_after = float(np.min(low_arr[peak_i:idx]))
    return (peak_h - trough_after) / peak_h


def _modepbs_weekly_conv_metrics(
    rows: List[KlineRow], i_low: int, idx: int
) -> Optional[Tuple[float, float]]:
    """返回 (信号周拟合%, 平台收敛改善=平台前半周拟合均值-后半周均值)。不足周线返回 None。"""
    weekly_bars, last_indices = daily_to_weekly_with_last_index(rows)
    if len(weekly_bars) < 30:
        return None
    conv = weekly_convergence_value_series(weekly_bars)
    if conv.size == 0:
        return None
    wi = None
    for k, li in enumerate(last_indices):
        if li >= idx:
            wi = k
            break
    if wi is None:
        wi = len(last_indices) - 1
    if wi >= len(conv) or np.isnan(conv[wi]):
        return None
    w_sig = float(conv[wi])

    wi_low = None
    for k, li in enumerate(last_indices):
        if li >= i_low:
            wi_low = k
            break
    if wi_low is None or wi_low > wi:
        return None
    wc = [
        float(conv[k])
        for k in range(wi_low, wi + 1)
        if k < len(conv) and not np.isnan(conv[k])
    ]
    if len(wc) < 4:
        return None
    h = len(wc) // 2
    if h < 1 or h >= len(wc):
        return None
    w_improve = float(np.mean(wc[:h]) - np.mean(wc[h:]))
    return w_sig, w_improve


def _match_mode_platform_breakout_first_yang(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    *,
    phase_days_min: int = 45,
    phase_days_max: int = 95,
    rise_from_low_min: float = 0.20,
    rise_from_low_max: float = 0.55,
    consolid_days: int = 20,
    consolid_amp_max: float = 0.30,
    breakout_lookback: int = 60,
    breakout_near_min: float = 0.93,
    big_pct_min: float = 7.0,
    big_pct_min_main: float = 4.5,
    body_ratio_min: float = 0.55,
    vol_mult: float = 1.25,
    vol_ma: int = 20,
    big_yang_gap: int = 15,
    gap_breakout_near_min: float = 0.93,
    high100_lookback: int = 100,
    high100_near_min: float = 0.93,
    vol_ratio_max: float = 4.0,
    vol_ratio_extended_max: float = 6.5,
    vol_high100_wash_min: int = 3,
    upper_ratio_max: float = 0.35,
    upper_ratio_extended_max: float = 0.30,
    upper_high100_vol_min: float = 4.0,
    wash_close_min_cnt: int = 2,
    wash_close60_min: float = 0.98,
    pre_rise5_min: float = -0.05,
    pre_rise5_max: float = 0.10,
    high_rise_wash_drop_rise_above: float = 0.38,
    high_rise_wash_drop_min: float = 0.08,
    weekly_conv_sig_max: float = 15.0,
    weekly_conv_improve_min: float = -1.5,
) -> Optional[Dict[str, float]]:
    """mode平台突破首阳：阶段低点→约3个月震仓整理→贴近/突破平台首根放量大阳线。"""
    n = len(rows)
    need = max(
        phase_days_max + 1,
        breakout_lookback + 1,
        high100_lookback + 1,
        consolid_days + 1,
        vol_ma + 1,
        big_yang_gap + 2,
    )
    if idx < need or idx >= n:
        return None

    r = rows[idx]
    if not _is_big_yang_row_modepbs(
        r,
        code,
        name,
        big_pct_min=big_pct_min,
        big_pct_min_main=big_pct_min_main,
        body_ratio_min=body_ratio_min,
        for_signal=True,
        allow_limit_up=True,
    ):
        return None

    high_arr = np.array([float(x.high) for x in rows], dtype=float)
    low_arr = np.array([float(x.low) for x in rows], dtype=float)
    close_arr = np.array([float(x.close) for x in rows], dtype=float)
    vol_arr = np.array([float(x.volume) for x in rows], dtype=float)

    lo = idx - phase_days_max
    hi = idx - phase_days_min
    if lo < 0 or hi < lo:
        return None
    seg = low_arr[lo : hi + 1]
    i_low = lo + int(np.argmin(seg))
    phase_days = idx - i_low
    if phase_days < phase_days_min or phase_days > phase_days_max:
        return None

    low_price = float(low_arr[i_low])
    if low_price <= 0:
        return None
    close = float(r.close)
    high = float(r.high)
    volume = float(r.volume)
    rise = (close - low_price) / low_price
    if rise < rise_from_low_min or rise > rise_from_low_max:
        return None

    wash_drop = _modepbs_wash_drop_from_peak_pct(high_arr, low_arr, i_low, idx)
    if (
        high_rise_wash_drop_rise_above > 0
        and high_rise_wash_drop_min > 0
        and rise > high_rise_wash_drop_rise_above
        and wash_drop < high_rise_wash_drop_min
    ):
        return None

    if idx < consolid_days:
        return None
    c_seg = rows[idx - consolid_days : idx]
    c_closes = [float(x.close) for x in c_seg]
    c_highs = [float(x.high) for x in c_seg]
    c_lows = [float(x.low) for x in c_seg]
    mean_c = float(np.mean(c_closes))
    if mean_c <= 0:
        return None
    consolid_amp = (max(c_highs) - min(c_lows)) / mean_c
    if consolid_amp > consolid_amp_max:
        return None

    if idx < breakout_lookback:
        return None
    prior_high = float(np.max(high_arr[idx - breakout_lookback : idx]))
    if prior_high <= 0:
        return None
    breakout_ratio = high / prior_high
    if breakout_ratio < breakout_near_min:
        return None
    breakout_pct = (high - prior_high) / prior_high * 100.0

    if idx < high100_lookback:
        return None
    prior_high100 = float(np.max(high_arr[idx - high100_lookback : idx]))
    if prior_high100 <= 0:
        return None
    high100_ratio = high / prior_high100
    if high100_ratio < high100_near_min:
        return None

    for j in range(idx - big_yang_gap, idx):
        if j < 0:
            continue
        if not _is_big_yang_row_modepbs(
            rows[j], code, name, big_pct_min=big_pct_min, big_pct_min_main=big_pct_min_main, body_ratio_min=body_ratio_min, for_signal=False
        ):
            continue
        if j < breakout_lookback:
            continue
        prior_high_j = float(np.max(high_arr[j - breakout_lookback : j]))
        if prior_high_j <= 0:
            continue
        if float(rows[j].high) / prior_high_j >= gap_breakout_near_min:
            return None

    pre_rise5 = 0.0
    if idx >= 6:
        pre_close = float(close_arr[idx - 1])
        base_close = float(close_arr[idx - 6])
        if base_close > 0:
            pre_rise5 = (pre_close - base_close) / base_close
            if pre_rise5 <= pre_rise5_min:
                return None
            if pre_rise5_max > 0 and pre_rise5 > pre_rise5_max:
                return None

    v_prev = float(vol_arr[idx - 1]) if idx >= 1 else 0.0
    v_ma = float(np.mean(vol_arr[idx - vol_ma : idx])) if idx >= vol_ma else 0.0
    v_base = max(v_prev, v_ma)
    if v_base <= 0 or volume < vol_mult * v_base:
        return None

    o, c, h, l_ = float(r.open), float(r.close), float(r.high), float(r.low)
    rng_d = h - l_
    body_ratio = (c - o) / rng_d if rng_d > 0 else 0.0
    upper_ratio = (h - max(o, c)) / rng_d if rng_d > 0 else 0.0
    pct = float(getattr(r, "pct_chg", 0.0) or 0.0)
    vol_ratio = volume / v_base

    wash_cnt = 0
    for j in range(i_low, idx):
        if _is_big_yang_row_modepbs(
            rows[j], code, name, big_pct_min=big_pct_min, big_pct_min_main=big_pct_min_main, body_ratio_min=body_ratio_min, for_signal=False
        ):
            wash_cnt += 1

    if vol_ratio_max > 0 and vol_ratio > vol_ratio_max:
        ext_ok = (
            vol_ratio_extended_max > 0
            and vol_ratio <= vol_ratio_extended_max
            and high100_ratio >= 1.0
            and vol_high100_wash_min > 0
            and wash_cnt >= vol_high100_wash_min
        )
        if not ext_ok:
            return None
    if upper_ratio_max > 0 and upper_ratio > upper_ratio_max:
        ext_ok = (
            upper_ratio_extended_max > 0
            and upper_ratio <= upper_ratio_extended_max
            and high100_ratio >= 1.0
            and upper_high100_vol_min > 0
            and vol_ratio >= upper_high100_vol_min
        )
        if not ext_ok:
            return None

    close_break60 = close / prior_high if prior_high > 0 else 0.0
    pre_rise5_pct = 0.0
    if idx >= 6 and float(close_arr[idx - 6]) > 0:
        pre_rise5_pct = (float(close_arr[idx - 1]) - float(close_arr[idx - 6])) / float(
            close_arr[idx - 6]
        ) * 100.0
    if wash_close_min_cnt > 0 and wash_cnt >= wash_close_min_cnt:
        if close_break60 < wash_close60_min and high100_ratio < 1.0:
            return None

    w_metrics = _modepbs_weekly_conv_metrics(rows, i_low, idx)
    w_sig_pct = float("nan")
    w_improve_pct = float("nan")
    if weekly_conv_sig_max > 0 or weekly_conv_improve_min > -900.0:
        if w_metrics is None:
            return None
        w_sig_pct, w_improve_pct = w_metrics
        if weekly_conv_sig_max > 0 and w_sig_pct > weekly_conv_sig_max:
            return None
        if w_improve_pct < weekly_conv_improve_min:
            return None

    return {
        "close": close,
        "low_date_idx": float(i_low),
        "low_price": low_price,
        "phase_days": float(phase_days),
        "rise_from_low_pct": rise * 100.0,
        "consolid_amp_pct": consolid_amp * 100.0,
        "prior_high": prior_high,
        "breakout_ratio": breakout_ratio,
        "breakout_pct": breakout_pct,
        "prior_high100": prior_high100,
        "high100_ratio": high100_ratio,
        "vol_today": volume,
        "vol_base": v_base,
        "vol_ratio": vol_ratio,
        "pct_chg": pct,
        "body_ratio": body_ratio,
        "upper_ratio": upper_ratio,
        "close_break60": close_break60,
        "pre_rise5_pct": pre_rise5_pct,
        "wash_big_yang_cnt": float(wash_cnt),
        "wash_drop_from_peak_pct": wash_drop * 100.0,
        "weekly_conv_sig_pct": w_sig_pct,
        "weekly_conv_improve_pct": w_improve_pct,
    }


def _score_mode_platform_breakout_first_yang(
    rows: List[KlineRow],
    idx: int,
    ma10: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
    vol20: np.ndarray,
    code: str = "",
    name: str = "",
    breakdown: Optional[List[tuple]] = None,
    *,
    phase_days_min: int = 45,
    phase_days_max: int = 95,
    rise_from_low_min: float = 0.20,
    rise_from_low_max: float = 0.55,
    consolid_days: int = 20,
    consolid_amp_max: float = 0.30,
    breakout_lookback: int = 60,
    breakout_near_min: float = 0.93,
    big_pct_min: float = 7.0,
    big_pct_min_main: float = 4.5,
    body_ratio_min: float = 0.55,
    vol_mult: float = 1.25,
    vol_ma: int = 20,
    big_yang_gap: int = 15,
    gap_breakout_near_min: float = 0.93,
    high100_lookback: int = 100,
    high100_near_min: float = 0.93,
    vol_ratio_max: float = 4.0,
    vol_ratio_extended_max: float = 6.5,
    vol_high100_wash_min: int = 3,
    upper_ratio_max: float = 0.35,
    upper_ratio_extended_max: float = 0.30,
    upper_high100_vol_min: float = 4.0,
    wash_close_min_cnt: int = 2,
    wash_close60_min: float = 0.98,
    pre_rise5_min: float = -0.05,
    pre_rise5_max: float = 0.10,
    high_rise_wash_drop_rise_above: float = 0.38,
    high_rise_wash_drop_min: float = 0.08,
    weekly_conv_sig_max: float = 15.0,
    weekly_conv_improve_min: float = -1.5,
) -> int:
    _ = (ma10, ma20, ma60, vol20)
    det = _match_mode_platform_breakout_first_yang(
        rows,
        idx,
        code,
        name,
        phase_days_min=phase_days_min,
        phase_days_max=phase_days_max,
        rise_from_low_min=rise_from_low_min,
        rise_from_low_max=rise_from_low_max,
        consolid_days=consolid_days,
        consolid_amp_max=consolid_amp_max,
        breakout_lookback=breakout_lookback,
        breakout_near_min=breakout_near_min,
        big_pct_min=big_pct_min,
        big_pct_min_main=big_pct_min_main,
        body_ratio_min=body_ratio_min,
        vol_mult=vol_mult,
        vol_ma=vol_ma,
        big_yang_gap=big_yang_gap,
        gap_breakout_near_min=gap_breakout_near_min,
        high100_lookback=high100_lookback,
        high100_near_min=high100_near_min,
        vol_ratio_max=vol_ratio_max,
        upper_ratio_max=upper_ratio_max,
        vol_ratio_extended_max=vol_ratio_extended_max,
        vol_high100_wash_min=vol_high100_wash_min,
        upper_ratio_extended_max=upper_ratio_extended_max,
        upper_high100_vol_min=upper_high100_vol_min,
        wash_close_min_cnt=wash_close_min_cnt,
        wash_close60_min=wash_close60_min,
        pre_rise5_min=pre_rise5_min,
        pre_rise5_max=pre_rise5_max,
        high_rise_wash_drop_rise_above=high_rise_wash_drop_rise_above,
        high_rise_wash_drop_min=high_rise_wash_drop_min,
        weekly_conv_sig_max=weekly_conv_sig_max,
        weekly_conv_improve_min=weekly_conv_improve_min,
    )
    if not det:
        return 0
    vr = float(det["vol_ratio"])
    pct = float(det["pct_chg"])
    brk = float(det["breakout_pct"])
    h100r = float(det["high100_ratio"])
    br = float(det["breakout_ratio"])
    platform_pts = max(brk * 1.5, 0.0) + max(h100r - high100_near_min, 0.0) * 100.0
    platform_pts += max(br - breakout_near_min, 0.0) * 80.0
    w_sig = float(det.get("weekly_conv_sig_pct", 0) or 0)
    w_imp = float(det.get("weekly_conv_improve_pct", 0) or 0)
    if w_sig == w_sig:
        platform_pts += max(0.0, (weekly_conv_sig_max - w_sig) if weekly_conv_sig_max > 0 else 0.0) * 0.5
    if w_imp == w_imp:
        platform_pts += max(0.0, w_imp) * 2.0
    score = int(min(100, max(0, round(50 + vr * 5 + pct * 1.2 + platform_pts))))
    if breakdown is not None:
        brk_label = f"突破+{brk:.1f}%" if brk >= 0 else f"贴顶{br:.2f}"
        wk = ""
        if w_sig == w_sig:
            wk = f" 周拟合{w_sig:.1f}%"
            if w_imp == w_imp:
                wk += f" 收敛+{w_imp:.1f}%"
        breakdown.append(
            (
                f"震仓{int(det['phase_days'])}日 自低+{det['rise_from_low_pct']:.1f}% "
                f"100日高比{h100r:.2f} {brk_label} 量比{vr:.2f} 涨{pct:.1f}%{wk}",
                0,
            )
        )
    return score


def _vol_ratio_at(rows: List[KlineRow], idx: int, vol_ma: int) -> float:
    vol_arr = np.array([float(x.volume) for x in rows], dtype=float)
    v_prev = float(vol_arr[idx - 1]) if idx >= 1 else 0.0
    v_ma = float(np.mean(vol_arr[idx - vol_ma : idx])) if idx >= vol_ma else 0.0
    v_base = max(v_prev, v_ma)
    if v_base <= 0:
        return 0.0
    return float(vol_arr[idx]) / v_base


def _find_mode_mid_big_yang_anchor(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    *,
    anchor_days_min: int,
    anchor_days_max: int,
    anchor_vol_mult: float,
    big_pct_min: float,
    big_pct_min_main: float,
    body_ratio_min: float,
    vol_ma: int,
) -> Optional[int]:
    """最近的主力介入大阳（锚点）：信号前 anchor_days_min～max 日内，取最近一根放量大阳线。"""
    lo = max(vol_ma + 1, idx - anchor_days_max)
    hi = idx - anchor_days_min
    if hi < lo:
        return None
    anchor_idx: Optional[int] = None
    for j in range(lo, hi + 1):
        if not _is_big_yang_row_modepbs(
            rows[j],
            code,
            name,
            big_pct_min=big_pct_min,
            big_pct_min_main=big_pct_min_main,
            body_ratio_min=body_ratio_min,
            for_signal=True,
        ):
            continue
        if _vol_ratio_at(rows, j, vol_ma) < anchor_vol_mult:
            continue
        if anchor_idx is None or j > anchor_idx:
            anchor_idx = j
    return anchor_idx


def _match_mode_mid_big_yang(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    *,
    anchor_days_min: int = 30,
    anchor_days_max: int = 200,
    anchor_vol_mult: float = 1.5,
    rise_from_anchor_min: float = 0.20,
    rise_from_anchor_max: float = 1.20,
    consolid_days: int = 20,
    consolid_amp_max: float = 0.35,
    breakout_lookback: int = 60,
    breakout_min: float = 1.0,
    high100_lookback: int = 100,
    high100_min: float = 1.0,
    tight_consolid_amp_max: float = 0.15,
    tight_vol_ratio_min: float = 1.8,
    tight_rise_from_anchor_min: float = 0.10,
    tight_high100_min: float = 0.985,
    big_pct_min: float = 7.0,
    big_pct_min_main: float = 4.5,
    body_ratio_min: float = 0.55,
    vol_mult: float = 1.25,
    vol_ma: int = 20,
    vol_ratio_max: float = 5.0,
    upper_ratio_max: float = 0.40,
    close_break60_min: float = 1.0,
    pre_rise5_min: float = -0.05,
    pre_rise5_max: float = 0.15,
) -> Optional[Dict[str, float]]:
    """mode中位大阳线：锚点主力大阳→震仓→突破大阳（自锚点已有较大涨幅）。"""
    n = len(rows)
    need = max(
        anchor_days_max + 1,
        breakout_lookback + 1,
        high100_lookback + 1,
        consolid_days + 1,
        vol_ma + 1,
        6,
    )
    if idx < need or idx >= n:
        return None

    r = rows[idx]
    if not _is_big_yang_row_modepbs(
        r,
        code,
        name,
        big_pct_min=big_pct_min,
        big_pct_min_main=big_pct_min_main,
        body_ratio_min=body_ratio_min,
        for_signal=True,
    ):
        return None

    i_anchor = _find_mode_mid_big_yang_anchor(
        rows,
        idx,
        code,
        name,
        anchor_days_min=anchor_days_min,
        anchor_days_max=anchor_days_max,
        anchor_vol_mult=anchor_vol_mult,
        big_pct_min=big_pct_min,
        big_pct_min_main=big_pct_min_main,
        body_ratio_min=body_ratio_min,
        vol_ma=vol_ma,
    )
    if i_anchor is None:
        return None

    high_arr = np.array([float(x.high) for x in rows], dtype=float)
    close_arr = np.array([float(x.close) for x in rows], dtype=float)
    close = float(r.close)
    high = float(r.high)
    volume = float(r.volume)
    anchor_close = float(close_arr[i_anchor])
    if anchor_close <= 0:
        return None
    phase_days = idx - i_anchor

    if idx < consolid_days:
        return None
    c_seg = rows[idx - consolid_days : idx]
    mean_c = float(np.mean([float(x.close) for x in c_seg]))
    if mean_c <= 0:
        return None
    consolid_amp = (
        max(float(x.high) for x in c_seg) - min(float(x.low) for x in c_seg)
    ) / mean_c
    if consolid_amp > consolid_amp_max:
        return None

    vol_ratio = _vol_ratio_at(rows, idx, vol_ma)
    tight_breakout = (
        consolid_amp <= tight_consolid_amp_max
        and vol_ratio >= tight_vol_ratio_min
    )
    eff_rise_min = (
        tight_rise_from_anchor_min if tight_breakout else rise_from_anchor_min
    )
    eff_high100_min = tight_high100_min if tight_breakout else high100_min

    rise_anchor = (close - anchor_close) / anchor_close
    seg_peak = float(np.max(high_arr[i_anchor : idx + 1]))
    rise_peak = (seg_peak - anchor_close) / anchor_close if anchor_close > 0 else 0.0
    rise_anchor = max(rise_anchor, rise_peak)
    if rise_anchor < eff_rise_min or rise_anchor > rise_from_anchor_max:
        return None

    if idx < breakout_lookback:
        return None
    prior_high = float(np.max(high_arr[idx - breakout_lookback : idx]))
    if prior_high <= 0:
        return None
    breakout_ratio = high / prior_high
    if breakout_ratio < breakout_min:
        return None
    breakout_pct = (high - prior_high) / prior_high * 100.0

    if idx < high100_lookback:
        return None
    prior_high100 = float(np.max(high_arr[idx - high100_lookback : idx]))
    if prior_high100 <= 0:
        return None
    high100_ratio = high / prior_high100
    if high100_ratio < eff_high100_min:
        return None

    if volume < vol_mult * max(
        float(rows[idx - 1].volume) if idx >= 1 else 0.0,
        float(np.mean([float(x.volume) for x in rows[idx - vol_ma : idx]])) if idx >= vol_ma else 0.0,
    ):
        return None
    if vol_ratio_max > 0 and vol_ratio > vol_ratio_max:
        return None

    o, c, h, l_ = float(r.open), float(r.close), float(r.high), float(r.low)
    rng_d = h - l_
    upper_ratio = (h - max(o, c)) / rng_d if rng_d > 0 else 0.0
    if upper_ratio_max > 0 and upper_ratio > upper_ratio_max:
        return None

    close_break60 = close / prior_high if prior_high > 0 else 0.0
    if close_break60_min > 0 and close_break60 < close_break60_min:
        return None

    if idx >= 6:
        pre_close = float(close_arr[idx - 1])
        base_close = float(close_arr[idx - 6])
        if base_close > 0:
            pre_rise5 = (pre_close - base_close) / base_close
            if pre_rise5 <= pre_rise5_min:
                return None
            if pre_rise5_max > 0 and pre_rise5 > pre_rise5_max:
                return None

    anchor_vr = _vol_ratio_at(rows, i_anchor, vol_ma)
    body_ratio = (c - o) / rng_d if rng_d > 0 else 0.0
    pct = float(getattr(r, "pct_chg", 0.0) or 0.0)
    pre_rise5_pct = 0.0
    if idx >= 6 and float(close_arr[idx - 6]) > 0:
        pre_rise5_pct = (
            (float(close_arr[idx - 1]) - float(close_arr[idx - 6]))
            / float(close_arr[idx - 6])
            * 100.0
        )

    return {
        "close": close,
        "anchor_date_idx": float(i_anchor),
        "anchor_close": anchor_close,
        "anchor_vol_ratio": anchor_vr,
        "phase_days": float(phase_days),
        "rise_from_anchor_pct": rise_anchor * 100.0,
        "consolid_amp_pct": consolid_amp * 100.0,
        "prior_high": prior_high,
        "breakout_ratio": breakout_ratio,
        "breakout_pct": breakout_pct,
        "prior_high100": prior_high100,
        "high100_ratio": high100_ratio,
        "vol_ratio": vol_ratio,
        "pct_chg": pct,
        "body_ratio": body_ratio,
        "upper_ratio": upper_ratio,
        "close_break60": close_break60,
        "pre_rise5_pct": pre_rise5_pct,
        "tight_breakout": float(1 if tight_breakout else 0),
    }


def _score_mode_mid_big_yang(
    rows: List[KlineRow],
    idx: int,
    ma10: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
    vol20: np.ndarray,
    code: str = "",
    name: str = "",
    breakdown: Optional[List[tuple]] = None,
    **kwargs,
) -> int:
    _ = (ma10, ma20, ma60, vol20, kwargs)
    det = _match_mode_mid_big_yang(rows, idx, code, name)
    if not det:
        return 0
    vr = float(det["vol_ratio"])
    pct = float(det["pct_chg"])
    brk = float(det["breakout_pct"])
    h100r = float(det["high100_ratio"])
    rise_a = float(det["rise_from_anchor_pct"])
    score = int(
        min(
            100,
            max(
                0,
                round(
                    45
                    + vr * 4
                    + pct * 1.0
                    + brk * 1.2
                    + (h100r - 1.0) * 80
                    + min(rise_a, 80) * 0.15
                ),
            ),
        )
    )
    if breakdown is not None:
        i_a = int(det["anchor_date_idx"])
        anchor_date = str(rows[i_a].date)[:10]
        breakdown.append(
            (
                f"锚点{anchor_date} 震仓{int(det['phase_days'])}日 "
                f"自锚点+{rise_a:.1f}% 突破+{brk:.1f}% 100日{h100r:.2f} "
                f"量比{vr:.2f} 涨{pct:.1f}%",
                0,
            )
        )
    return score


def _mode5_signals(
    rows: List[KlineRow],
    start_date: Optional[str],
    end_date: Optional[str],
    code: str,
    name: str,
    *,
    shrink_max_days: int = 5,
    half_year_bars: int = 120,
) -> List[int]:
    out: List[int] = []
    need = half_year_bars + 2
    if len(rows) < need:
        return out
    start_dt = _parse_date(start_date)
    end_dt = _parse_date(end_date)
    for s_idx in range(half_year_bars, len(rows)):
        if start_dt or end_dt:
            try:
                row_dt = datetime.strptime(rows[s_idx].date, "%Y-%m-%d").date()
            except Exception:
                continue
            if start_dt and row_dt < start_dt:
                continue
            if end_dt and row_dt > end_dt:
                continue
        if _mode5_anchor_detail(
            rows,
            s_idx,
            code,
            name,
            shrink_max_days=shrink_max_days,
            half_year_bars=half_year_bars,
        ):
            out.append(s_idx)
    return out


def _score_mode5(
    rows: List[KlineRow],
    idx: int,
    ma10: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
    vol20: np.ndarray,
    code: str = "",
    name: str = "",
    shrink_max_days: int = 5,
    half_year_bars: int = 120,
) -> int:
    det = _mode5_anchor_detail(
        rows,
        idx,
        code,
        name,
        shrink_max_days=shrink_max_days,
        half_year_bars=half_year_bars,
    )
    if det is None:
        return 0
    _T, _v_ref, ratio = det
    base = 75
    if ratio < 0.25:
        base += 10
    elif ratio < 0.35:
        base += 6
    elif ratio < 0.45:
        base += 3
    return int(min(100, base))


def _mode8_signals(
    rows: List[KlineRow],
    start_date: Optional[str],
    end_date: Optional[str],
) -> List[int]:
    """
    mode8 信号（大牛股买点）：在 mode3 基础上放宽 20 日涨幅、增加 60 日涨幅过滤。
    - 20日涨幅 ≤ 50%（mode3 为 25%）；
    - 买点前60日涨幅 -15% ≤ ret60 ≤ 50%（需至少 60 根 K 线）。
    其余与 mode3 一致：MA10>MA20>MA60、收盘≥MA20、volume≥vol20×1.2。见 docs/mode8模型说明.md
    """
    signals: List[int] = []
    if len(rows) < 60:
        return signals
    close = np.array([r.close for r in rows], dtype=float)
    volume = np.array([r.volume for r in rows], dtype=float)
    ma10 = _moving_mean(close, 10)
    ma20 = _moving_mean(close, 20)
    ma60 = _moving_mean(close, 60)
    vol20 = _moving_mean(volume, 20)
    start_dt = _parse_date(start_date)
    end_dt = _parse_date(end_date)

    for i in range(60, len(rows)):
        if start_dt or end_dt:
            try:
                row_dt = datetime.strptime(rows[i].date, "%Y-%m-%d").date()
            except Exception:
                continue
            if start_dt and row_dt < start_dt:
                continue
            if end_dt and row_dt > end_dt:
                continue
        if (
            np.isnan(ma10[i])
            or np.isnan(ma20[i])
            or np.isnan(ma60[i])
            or np.isnan(vol20[i])
        ):
            continue
        # mode8: 20日涨幅 ≤ 50%（mode3 为 25%）
        if i - 20 >= 0 and close[i - 20] > 0:
            ret20 = (close[i] - close[i - 20]) / close[i - 20] * 100
            if ret20 > 50:
                continue
        # mode8: 买点前60日涨幅 -15% ≤ ret60 ≤ 50%
        if close[i - 60] > 0:
            ret60 = (close[i] - close[i - 60]) / close[i - 60] * 100
            if ret60 < -15 or ret60 > 50:
                continue
        ma10_slope = ma10[i] - ma10[i - 3]
        ma20_slope = ma20[i] - ma20[i - 3]
        ma60_slope = ma60[i] - ma60[i - 3]
        if not (
            ma10[i] > ma20[i] > ma60[i]
            and ma10_slope > 0
            and ma20_slope > 0
            and ma60_slope > 0
        ):
            continue
        if close[i] < ma20[i]:
            continue
        if volume[i] < vol20[i] * 1.2:
            continue
        signals.append(i)
    return signals


def _mode10_signals(
    rows: List[KlineRow],
    start_date: Optional[str],
    end_date: Optional[str],
    conv_max: float = 1.0,
    ma30_turn_weeks: int = 5,
) -> List[int]:
    """
    mode10 信号：5 周内 MA30 拐头 + 当周 MA5>MA10>MA20 多头向上 + 买点前周线拟合 < conv_max（%）。
    返回满足条件的「当周最后交易日」在 rows 中的下标列表。
    """
    if not rows or len(rows) < 100:
        return []
    weekly_bars, last_indices = daily_to_weekly_with_last_index(rows)
    if len(weekly_bars) < 32:
        return []
    closes = np.array([w[4] for w in weekly_bars], dtype=float)
    ma5 = _moving_mean(closes, 5)
    ma10 = _moving_mean(closes, 10)
    ma20 = _moving_mean(closes, 20)
    ma30 = _moving_mean(closes, 30)
    conv = weekly_convergence_value_series(weekly_bars)
    if len(conv) == 0:
        return []
    signal_indices = []
    for i in range(30, len(weekly_bars)):
        if np.isnan(ma5[i]) or np.isnan(ma10[i]) or np.isnan(ma20[i]):
            continue
        if not (ma5[i] > ma10[i] > ma20[i]):
            continue
        if not _has_ma30_turn_in_weeks(weekly_bars, ma30, i, ma30_turn_weeks):
            continue
        conv_min = np.nanmin(conv[30:i]) if i > 30 else np.nan
        if np.isnan(conv_min) or conv_min >= conv_max:
            continue
        idx = last_indices[i]
        if idx >= len(rows):
            continue
        d = rows[idx].date
        if start_date and d < start_date:
            continue
        if end_date and d > end_date:
            continue
        signal_indices.append(idx)
    return sorted(signal_indices)


def _has_ma30_turn_in_weeks(weekly_bars: List[tuple], ma30: np.ndarray, signal_week_i: int, within_weeks: int) -> bool:
    """信号周 signal_week_i 的前 within_weeks 周内是否存在周线 MA30 由下转上的拐点。"""
    if signal_week_i < 32 or within_weeks <= 0:
        return False
    lo = max(31, signal_week_i - within_weeks)
    hi = signal_week_i
    for j in range(lo, hi):
        if np.isnan(ma30[j]) or np.isnan(ma30[j - 1]) or np.isnan(ma30[j - 2]):
            continue
        if ma30[j] > ma30[j - 1] and ma30[j - 1] < ma30[j - 2]:
            return True
    return False


def _nearest_ma30_turn_weeks_before(ma30: np.ndarray, signal_week_i: int, within_weeks: int) -> Optional[int]:
    """信号周前 within_weeks 周内，距离信号周最近的拐点周数（1=前1周，2=前2周…），无则 None。"""
    if signal_week_i < 32 or within_weeks <= 0:
        return None
    lo = max(31, signal_week_i - within_weeks)
    for j in range(signal_week_i - 1, lo - 1, -1):
        if np.isnan(ma30[j]) or np.isnan(ma30[j - 1]) or np.isnan(ma30[j - 2]):
            continue
        if ma30[j] > ma30[j - 1] and ma30[j - 1] < ma30[j - 2]:
            return signal_week_i - j
    return None


def _mode12_signals(
    rows: List[KlineRow],
    start_date: Optional[str],
    end_date: Optional[str],
    accel_th: float = 2.5,
    ma30_turn_weeks: int = 5,
) -> List[int]:
    """
    mode12 信号：mode10（周线 MA5 斜率突变）+ 5 周内存在周线 MA30 拐头向上。
    拐点 = MA30 由下转上（当周 > 上周 且 上周 < 上上周）。仅保留信号周前 ma30_turn_weeks 周内有拐点的信号。
    """
    if not rows or len(rows) < 100:
        return []
    weekly_bars, last_indices = daily_to_weekly_with_last_index(rows)
    if len(weekly_bars) < 32:
        return []
    closes = np.array([w[4] for w in weekly_bars], dtype=float)
    ma5 = _moving_mean(closes, 5)
    ma30 = _rolling_mean(closes, 30)
    slope = np.full_like(ma5, np.nan, dtype=float)
    for i in range(1, len(ma5)):
        if np.isnan(ma5[i]) or np.isnan(ma5[i - 1]) or ma5[i - 1] <= 0:
            continue
        slope[i] = (ma5[i] - ma5[i - 1]) / ma5[i - 1] * 100.0
    accel = np.full_like(slope, np.nan, dtype=float)
    for i in range(2, len(slope)):
        if np.isnan(slope[i]) or np.isnan(slope[i - 1]):
            continue
        accel[i] = slope[i] - slope[i - 1]
    signal_indices = []
    for i in range(2, len(accel)):
        if np.isnan(accel[i]) or accel[i] < accel_th:
            continue
        if not _has_ma30_turn_in_weeks(weekly_bars, ma30, i, ma30_turn_weeks):
            continue
        idx = last_indices[i]
        if idx >= len(rows):
            continue
        d = rows[idx].date
        if start_date and d < start_date:
            continue
        if end_date and d > end_date:
            continue
        signal_indices.append(idx)
    return sorted(signal_indices)


def _mode11_signals(
    rows: List[KlineRow],
    start_date: Optional[str],
    end_date: Optional[str],
    accel_th: float = 2.5,
    body_ratio_max: float = 0.35,
    vol_ratio_min: float = 1.5,
    vol_weeks: int = 20,
) -> List[int]:
    """
    mode11 信号：mode10（周线 MA5 斜率突变）+ 拐点形态过滤。
    - 当周 K 线小实体长影线：|close-open|/(high-low) < body_ratio_max；
    - 当周成交量放量：周量 >= vol_ratio_min * 过去 vol_weeks 周均量。
    返回满足条件的「当周最后交易日」在 rows 中的下标列表。
    """
    if not rows or len(rows) < 100:
        return []
    weekly_bars, last_indices = daily_to_weekly_with_volume_and_last_index(rows)
    if len(weekly_bars) < 18:
        return []
    closes = np.array([w[4] for w in weekly_bars], dtype=float)
    vols = np.array([w[5] for w in weekly_bars], dtype=float)
    ma5 = _moving_mean(closes, 5)
    slope = np.full_like(ma5, np.nan, dtype=float)
    for i in range(1, len(ma5)):
        if np.isnan(ma5[i]) or np.isnan(ma5[i - 1]) or ma5[i - 1] <= 0:
            continue
        slope[i] = (ma5[i] - ma5[i - 1]) / ma5[i - 1] * 100.0
    accel = np.full_like(slope, np.nan, dtype=float)
    for i in range(2, len(slope)):
        if np.isnan(slope[i]) or np.isnan(slope[i - 1]):
            continue
        accel[i] = slope[i] - slope[i - 1]
    signal_indices = []
    for i in range(2, len(accel)):
        if np.isnan(accel[i]) or accel[i] < accel_th:
            continue
        o, h, l, c = weekly_bars[i][1], weekly_bars[i][2], weekly_bars[i][3], weekly_bars[i][4]
        rng = h - l
        if rng > 0:
            body_ratio = abs(c - o) / rng
            if body_ratio >= body_ratio_max:
                continue
        else:
            continue
        lo = max(0, i - vol_weeks)
        vol_avg = np.mean(vols[lo:i]) if lo < i else vols[i]
        if vol_avg <= 0:
            continue
        if vols[i] < vol_ratio_min * vol_avg:
            continue
        idx = last_indices[i]
        if idx >= len(rows):
            continue
        d = rows[idx].date
        if start_date and d < start_date:
            continue
        if end_date and d > end_date:
            continue
        signal_indices.append(idx)
    return sorted(signal_indices)


def _limit_rate(code: str, name: str) -> float:
    """涨停/跌停幅度：ST 5%，科创/创业板 20%，其他 10%"""
    code = str(code or "")
    if _is_st(name or ""):
        return 0.05
    if code.startswith(("30", "301", "688")):
        return 0.20
    if code.startswith(("8", "9")):
        return 0.30
    return 0.10


def _has_limit_up_then_down(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    lookback: int = 5,
    min_consec_limit_up: int = 3,
) -> bool:
    """买点前 lookback 个交易日内：连续涨停(≥min_consec_limit_up天)后跌停 → True(需排除)"""
    if idx < lookback + 1:
        return False
    rate = _limit_rate(code, name)
    limit_up = (rate * 100) - 0.5
    limit_down = -(rate * 100) + 0.5
    conseq_limit_up = 0
    for j in range(lookback, 0, -1):
        i = idx - j
        p = rows[i].pct_chg
        if p >= limit_up:
            conseq_limit_up += 1
        elif p <= limit_down:
            if conseq_limit_up >= min_consec_limit_up:
                return True
            conseq_limit_up = 0
        else:
            conseq_limit_up = 0
    return False


def _has_limit_up_6d(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    lookback: int = 6,
) -> bool:
    """
    买点日前 lookback 个交易日内是否出现过涨停（仅判定有/无，不要求缩量）。
    涨停阈值与 _limit_rate 一致：按 ST / 创业板 / 科创板 / 主板 的涨停幅度计算。
    """
    if idx < 1:
        return False
    rate = _limit_rate(code, name)
    limit_up = (rate * 100) - 0.5
    start = max(1, idx - lookback)
    for i in range(start, idx):
        if rows[i].pct_chg >= limit_up:
            return True
    return False


def _has_limit_up_then_shrink_volume(
    rows: List[KlineRow],
    idx: int,
    code: str,
    name: str,
    lookback: int = 6,
    next_vol_max_mult: float = 1.8,
) -> bool:
    """
    买点前 lookback 个交易日内出现涨停，且涨停后1个交易日成交量 < 涨停日成交量 * next_vol_max_mult → True(加分特征)。
    仅用当日及历史数据；涨停阈值按代码板块与 ST 规则计算。
    """
    if idx < 2:
        return False
    rate = _limit_rate(code, name)
    limit_up = (rate * 100) - 0.5
    start = max(1, idx - lookback)
    for i in range(start, idx):
        if rows[i].pct_chg < limit_up:
            continue
        if i + 1 >= len(rows):
            continue
        v0 = rows[i].volume
        v1 = rows[i + 1].volume
        if v0 > 0 and v1 < v0 * next_vol_max_mult:
            return True
    return False


def _close_below_ma20_today(
    close: np.ndarray,
    ma20: np.ndarray,
    idx: int,
) -> bool:
    """当天收盘破 MA20 → True(需排除)"""
    if idx < 0 or idx >= len(close):
        return False
    return not np.isnan(ma20[idx]) and ma20[idx] > 0 and close[idx] < ma20[idx]


def _score_mode3(
    rows: List[KlineRow],
    idx: int,
    ma10: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
    vol20: np.ndarray,
) -> int:
    close_arr = np.array([r.close for r in rows], dtype=float)
    close = rows[idx].close
    volume = rows[idx].volume
    score = 40.0  # 基础分（原 60）

    ma20_now = ma20[idx]
    ma60_now = ma60[idx]
    ma10_now = ma10[idx]
    vol20_now = vol20[idx]

    if ma20_now > 0:
        gap = (ma10_now - ma20_now) / ma20_now
        if gap >= 0.02:
            score += 10
        elif gap >= 0.01:
            score += 6
        elif gap >= 0.005:
            score += 3

    if ma60_now > 0:
        gap = (ma20_now - ma60_now) / ma60_now
        if gap >= 0.02:
            score += 10
        elif gap >= 0.01:
            score += 6
        elif gap >= 0.005:
            score += 3

    if vol20_now > 0:
        vol_ratio = volume / vol20_now
        if vol_ratio >= 1.6:
            score += 15
        elif vol_ratio >= 1.4:
            score += 10
        elif vol_ratio >= 1.2:
            score += 6

    if ma20_now > 0:
        close_gap = (close - ma20_now) / ma20_now
        if close_gap >= 0.03:
            score += 5
        elif close_gap >= 0.01:
            score += 3

    # 近3日涨幅超过20%则降分
    if idx >= 3:
        base_close = rows[idx - 3].close
        if base_close > 0:
            ret3 = (close - base_close) / base_close * 100
            if ret3 > 20:
                score -= 10
            elif ret3 > 15:
                score -= 5

    # 5日线拐头向下：今日MA5低于昨日MA5则降分
    if idx >= 5:
        ma5 = _moving_mean(close_arr, 5)
        if not (np.isnan(ma5[idx]) or np.isnan(ma5[idx - 1])) and ma5[idx] < ma5[idx - 1]:
            score -= 5

    return int(max(0, round(score)))


def _score_mode9(
    rows: List[KlineRow],
    idx: int,
    ma10: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
    vol20: np.ndarray,
    code: str = "",
    name: str = "",
    breakdown: Optional[List[tuple]] = None,
    industry: str = "",
    hot_industries: Optional[Set[str]] = None,
    mode9_hot_industry_bonus: int = 0,
    hot_industry_counts: Optional[Dict[str, int]] = None,
    mode9_hot_industry_bonus_max: int = 12,
    industry_ndays_limit_total: Optional[int] = None,
    mode9_industry_ndays_penalty: int = 0,
    mode9_industry_ndays_bonus_per_unit: int = 5,
    mode9_industry_ndays_bonus_cap: int = 8,
) -> int:
    """
    mode9 评分：在 mode3 基础上微调，使选股更准。
    可选：信号日全市场涨停行业 TopN（与 limit_up_industry_top 一致）含本股行业时加分；
    若提供 hot_industry_counts，则该行业当日涨停家数越多，在封顶 mode9_hot_industry_bonus_max 内额外加分
    （资金向该板块集聚越强，排序越靠前）。
    可选 industry_ndays_limit_total：信号日前 N 个交易日内该行业涨停家次累计（非 None 时生效）；
    累计为 0 且配置了 penalty 则扣「资金关注度低」分，累计多则按 per_unit 加分封顶 cap。
    基于「满分100 表现最好5只 vs 最差5只」买点前特征对比：
    - 收盘距MA20 过远(>8%)略降分，偏好温和突破；
    - MA20-MA60 开口大(趋势强)额外加分；
    - 当日实体占比大(阳线实在)加分；
    - 60日涨幅适中(15%～40%)略加分；
    - MA5 斜率过于陡峭(5日内MA5涨幅>15%)降分，避免短期冲得过猛（如金时科技 17.44%）。
    - 量能放大太快（多看3～5日）：近3日均量/再前3日>2 或 当日量/5日前量>2.8 降分，可能快到顶。
    - 均线整齐度（参考明阳电路 vs 神开股份）：当日 MA5>MA10>MA20 加分；近5日内均线交叉次数多则降分。
    - 近20日前高（不含当日）：收盘略高于前高且突破幅度≤1%视为「当天刚刚突破前高」，+1（与「贴近前高蓄势」可叠加）。
    - 前期高点距信号日 21～59 日、收盘贴近该前高（0.95～1.03 倍）、近 5 日量能温和放大、当日量接近前 100 日（不含当日）最大量（0.9～1.5 倍）：+4（见下方 MODE9_PEAKDIST_VOL_BONUS）。
    """
    base = _score_mode3(rows, idx, ma10, ma20, ma60, vol20)
    close_arr = np.array([r.close for r in rows], dtype=float)
    volume = np.array([r.volume for r in rows], dtype=float)
    close = rows[idx].close
    ma20_now = ma20[idx]
    ma60_now = ma60[idx]
    if ma20_now <= 0:
        return base
    # 前一日收盘价跌破 MA10 且前一日最低价跌破 MA20：均线支撑弱，扣分（如中国电影 2月13日信号前一日）
    if idx >= 1:
        prev_close = rows[idx - 1].close
        prev_low = rows[idx - 1].low
        ma10_prev = ma10[idx - 1]
        ma20_prev = ma20[idx - 1]
        if (
            not (np.isnan(ma10_prev) or np.isnan(ma20_prev))
            and ma10_prev > 0
            and ma20_prev > 0
            and prev_close < ma10_prev
            and prev_low < ma20_prev
        ):
            base -= 4
            if breakdown is not None:
                breakdown.append(("前一日收盘破MA10且最低破MA20", -4))
    close_gap = (close - ma20_now) / ma20_now
    # 收盘距MA20 过远降分（最好组 10.34% vs 最差 12.53%，温和突破更优）
    if close_gap > 0.08:
        base -= 2
        if breakdown is not None:
            breakdown.append(("收盘距MA20过远(>8%)", -2))
    # 涨停后缩量：买点前6日内有涨停，且涨停次日量 < 涨停日量 * 1.8，加分（缩量不松、承接好）
    if code and _has_limit_up_then_shrink_volume(rows, idx, code, name, lookback=6, next_vol_max_mult=1.8):
        base += 2
        if breakdown is not None:
            breakdown.append(("涨停后缩量", 2))
    # MA5 斜率过于陡峭降分（最好5只 MA5斜率 7～14%，最差中金时科技 17.44%、亚康 16.32%）
    ma5 = None
    if idx >= 5:
        ma5 = _moving_mean(close_arr, 5)
        if not (np.isnan(ma5[idx]) or np.isnan(ma5[idx - 5]) or ma5[idx - 5] <= 0):
            ma5_slope_pct = (ma5[idx] - ma5[idx - 5]) / ma5[idx - 5] * 100
            if ma5_slope_pct > 15:
                base -= 2
                if breakdown is not None:
                    breakdown.append(("MA5斜率过陡(>15%)", -2))
        # 均线整齐度：当日 MA5>MA10>MA20 加分（明阳电路式完美多头）
        if ma5 is not None and not (np.isnan(ma5[idx]) or np.isnan(ma10[idx]) or np.isnan(ma20[idx])):
            if ma5[idx] > ma10[idx] > ma20[idx]:
                base += 2
                if breakdown is not None:
                    breakdown.append(("均线整齐MA5>MA10>MA20", 2))
        # 近5日内均线交叉次数多则降分（神开股份式乱序）
        if idx >= 6 and ma5 is not None:
            crosses = 0
            for i in range(idx - 4, idx + 1):
                if i <= 0:
                    continue
                if not (np.isnan(ma5[i]) or np.isnan(ma5[i - 1]) or np.isnan(ma10[i]) or np.isnan(ma10[i - 1])):
                    if (ma5[i] - ma10[i]) * (ma5[i - 1] - ma10[i - 1]) < 0:
                        crosses += 1
                if not (np.isnan(ma10[i]) or np.isnan(ma10[i - 1]) or np.isnan(ma20[i]) or np.isnan(ma20[i - 1])):
                    if (ma10[i] - ma20[i]) * (ma10[i - 1] - ma20[i - 1]) < 0:
                        crosses += 1
            if crosses >= 2:
                base -= 2
                if breakdown is not None:
                    breakdown.append(("近5日均线交叉多(>=2次)", -2))
        # MA5 与 MA10 粘连降分（如美利云 2月9日两线几乎贴在一起，趋势不清晰）
        if ma5 is not None and ma10[idx] > 0:
            # 当日粘连：|MA5-MA10|/MA10 < 1%
            gap_pct = abs(ma5[idx] - ma10[idx]) / ma10[idx]
            if gap_pct < 0.01:
                base -= 2
                if breakdown is not None:
                    breakdown.append(("MA5与MA10粘连(<1%)", -2))
            # 近5日内曾粘连（含当日）
            elif idx >= 5:
                for i in range(idx - 4, idx + 1):
                    if i < 0 or np.isnan(ma5[i]) or np.isnan(ma10[i]) or ma10[i] <= 0:
                        continue
                    if abs(ma5[i] - ma10[i]) / ma10[i] < 0.01:
                        base -= 2
                        if breakdown is not None:
                            breakdown.append(("近5日内MA5与MA10曾粘连", -2))
                        break
        # MA5 近期拐头向下、当日强行拐回降分（如兴民智通 2月12日 MA5 向下，2月13日拐回，形态不稳）
        if ma5 is not None and idx >= 3:
            today_up = not (np.isnan(ma5[idx]) or np.isnan(ma5[idx - 1])) and ma5[idx] > ma5[idx - 1]
            if today_up:
                # 昨日或前日 MA5 曾向下
                recent_down = False
                for i in range(1, 3):
                    if idx - i < 1:
                        break
                    if not (np.isnan(ma5[idx - i]) or np.isnan(ma5[idx - i - 1])) and ma5[idx - i] < ma5[idx - i - 1]:
                        recent_down = True
                        break
                if recent_down:
                    base -= 2
                    if breakdown is not None:
                        breakdown.append(("MA5近期拐头向下当日拐回", -2))
    # 当日收盘价低于 MA5 / 大阴线跌破 MA5：仅跌破幅度较大时扣分（轻微跌破不扣，避免误杀如招商轮船）
    if ma5 is not None and not np.isnan(ma5[idx]) and ma5[idx] > 0 and close < ma5[idx]:
        break_ma5_pct = (ma5[idx] - close) / ma5[idx] * 100  # 收盘低于 MA5 的幅度%
        open_ = rows[idx].open
        is_big_bear = (close < open_) and (getattr(rows[idx], "pct_chg", 0) <= -1.0)  # 阴线且跌幅≥1%
        # 仅当跌破超过 2% 才扣「收盘低于 MA5」；大阴线且跌破超过 2% 再加大扣分，超过 3% 更大扣分
        if break_ma5_pct >= 2.0:
            base -= 5  # 跌破 MA5 超 2 个点：扣分
            if breakdown is not None:
                breakdown.append(("当日收盘价跌破MA5超2%", -5))
        if is_big_bear and break_ma5_pct >= 3.0:
            base -= 15  # 大阴线且跌破 MA5 超 3 个点：大扣分
            if breakdown is not None:
                breakdown.append(("当日大阴线且跌破MA5超3%", -15))
        elif is_big_bear and break_ma5_pct >= 2.0:
            base -= 10  # 大阴线且跌破 MA5 超 2 个点
            if breakdown is not None:
                breakdown.append(("当日大阴线且跌破MA5超2%", -10))
    if ma60_now > 0:
        ma20_60_gap = (ma20[idx] - ma60_now) / ma60_now
        # 均线多头开口适中加分；开口过大（过度加速）不再额外加分，极端大时略降分
        if 0.03 <= ma20_60_gap <= 0.09:
            base += 2
            if breakdown is not None:
                breakdown.append(("MA20-MA60开口适中", 2))
        elif ma20_60_gap > 0.12:
            base -= 2
            if breakdown is not None:
                breakdown.append(("MA20-MA60开口过大(>12%)", -2))
    # 当日 K 线实体占比大加分（最好组 74.9% vs 最差 50%）
    rng = rows[idx].high - rows[idx].low
    if rng > 0:
        body = abs(close - rows[idx].open)
        if body / rng >= 0.6:
            base += 2
            if breakdown is not None:
                breakdown.append(("当日K线实体占比>=60%", 2))
    if idx >= 60 and close_arr[idx - 60] > 0:
        ret60 = (close - close_arr[idx - 60]) / close_arr[idx - 60] * 100
        # 60 日涨幅适中略加分；涨幅过大视为趋势已走出较长一段，适当降分
        if 15 <= ret60 <= 40:
            base += 1
            if breakdown is not None:
                breakdown.append(("60日涨幅15%~40%", 1))
        elif ret60 > 45:
            d = 2 if ret60 <= 55 else 4
            base -= d
            if breakdown is not None:
                breakdown.append((f"60日涨幅过大(>{45}%)", -d))
    # 突破质量：近20日（不含当日）最高价视为平台前高，当前收盘相对前高的位置
    if idx >= 21:
        high_arr = np.array([r.high for r in rows], dtype=float)
        prev_high_20 = float(np.nanmax(high_arr[idx - 20 : idx]))  # 前20根K线最高
        if prev_high_20 > 0:
            break_gap_pct = (close - prev_high_20) / prev_high_20 * 100
            if -3.0 <= break_gap_pct <= 1.0:
                base += 2  # 贴近前高或略低于前高，蓄势待发（如御银、金开新能）
                if breakdown is not None:
                    breakdown.append(("贴近前高蓄势", 2))
            elif 1.0 < break_gap_pct <= 6.0:
                base += 3  # 适度突破前高（如华盛昌、神马电力）
                if breakdown is not None:
                    breakdown.append(("适度突破前高", 3))
            elif break_gap_pct > 10.0:
                base -= 2  # 已远离前高，追高
                if breakdown is not None:
                    breakdown.append(("已远离前高追高", -2))
            # 当天刚刚突破前高：收盘略高于近20日前高（不含当日），幅度越小越「刚突破」，+1 便于同分区内排序靠前
            if 0 < break_gap_pct <= 1.0:
                base += 1
                if breakdown is not None:
                    breakdown.append(("当天刚刚突破前高(≤1%)", 1))
    # 前期高点（21～59 日前区间内最高价日，同价取最近一日）+ 收盘贴近前高 + 近 5 日温和放量 + 当日量贴近前 100 日最大量
    MODE9_PEAKDIST_VOL_BONUS = 4  # 多条件同时满足，与「适度突破前高+3」同级略高
    if idx >= 100:
        high_arr_pk = np.array([r.high for r in rows], dtype=float)
        w_lo, w_hi = idx - 59, idx - 21  # 距信号日 21～59 个交易日
        seg_h = high_arr_pk[w_lo : w_hi + 1]
        if seg_h.size > 0 and np.all(np.isfinite(seg_h)):
            # 区间内最高价；同价取距信号日最近的一日（右端优先）
            peak_idx = int(w_hi - int(np.argmax(seg_h[::-1])))
            days_pk = idx - peak_idx
            if 21 <= days_pk <= 59:
                peak_high = float(high_arr_pk[peak_idx])
                if peak_high > 0 and (0.95 * peak_high <= close <= 1.03 * peak_high):
                    hist_max_vol = float(np.max(volume[idx - 100 : idx]))
                    v_today = float(volume[idx])
                    if hist_max_vol > 0 and (0.9 * hist_max_vol <= v_today < 1.5 * hist_max_vol):
                        v_last5 = volume[idx - 4 : idx + 1]
                        v_prev5 = volume[idx - 9 : idx - 4]
                        if v_last5.size == 5 and v_prev5.size == 5 and np.all(v_last5 > 0) and np.all(v_prev5 > 0):
                            a5 = float(np.mean(v_last5))
                            a5p = float(np.mean(v_prev5))
                            gentle = 1.05 <= (a5 / a5p) <= 1.30
                            mx5, mn5 = float(np.max(v_last5)), float(np.min(v_last5))
                            not_spiky = (mn5 > 0) and ((mx5 / mn5) < 2.2)
                            if gentle and not_spiky:
                                base += MODE9_PEAKDIST_VOL_BONUS
                                if breakdown is not None:
                                    pxr = close / peak_high
                                    breakdown.append(
                                        (
                                            f"前高距{days_pk}日+收盘/前高{pxr:.2f}+近5日温和放量+量近100日高({v_today/hist_max_vol:.2f}倍)",
                                            MODE9_PEAKDIST_VOL_BONUS,
                                        )
                                    )
    # 量能放大太快（多看3～5日）：最好组 近3日/再前3日 1.27、最差组 2.13；当日/5日前 最好2.71、最差3.06
    if idx >= 6:
        vol_3d_recent = (volume[idx] + volume[idx - 1] + volume[idx - 2]) / 3.0
        vol_3d_older = (volume[idx - 3] + volume[idx - 4] + volume[idx - 5]) / 3.0
        if vol_3d_older > 0 and vol_3d_recent / vol_3d_older > 2.0:
            base -= 2  # 近3日量能相对再前3日陡升，可能快到顶
            if breakdown is not None:
                breakdown.append(("近3日量能/再前3日>2倍", -2))
    if idx >= 5 and volume[idx - 5] > 0:
        if volume[idx] / volume[idx - 5] > 2.8:
            base -= 2  # 5日内量能放大超过2.8倍，放大过快
            if breakdown is not None:
                breakdown.append(("当日量/5日前量>2.8倍", -2))
    # 买点前3日内爆量（如哈尔斯 2月12日量是2月11日的4倍以上）：前3日内任一天量>=前一日3倍则扣分；除非该日在近2个月最低价附近（底部放量可豁免）。按比例加重扣分。
    if idx >= 4:
        low_arr = np.array([r.low for r in rows], dtype=float)
        for i in range(idx - 3, idx):
            if i < 1 or volume[i - 1] <= 0:
                continue
            vol_ratio = volume[i] / volume[i - 1]
            if vol_ratio < 3.0:
                continue
            # 爆量日 i，是否在近2个月最低价附近（约40日）
            start = max(0, i - 39)
            min_low_40 = np.nanmin(low_arr[start : i + 1])
            if np.isnan(min_low_40) or min_low_40 <= 0:
                deduct = min(10, 4 + int((vol_ratio - 3) * 2))  # 3倍起扣4分，每多1倍多扣2分，上限10
                base -= deduct
                if breakdown is not None:
                    breakdown.append((f"买点前3日内爆量(约{vol_ratio:.1f}倍)", -deduct))
                break
            # 该日最低或收盘在最低价 5% 以内视为「最低价附近」
            near_bottom = (low_arr[i] <= min_low_40 * 1.05) or (close_arr[i] <= min_low_40 * 1.05)
            if not near_bottom:
                deduct = min(10, 4 + int((vol_ratio - 3) * 2))  # 3倍扣4分，4倍扣6分，4.67倍扣7分，上限10
                base -= deduct
                if breakdown is not None:
                    breakdown.append((f"买点前3日内爆量(约{vol_ratio:.1f}倍)", -deduct))
                break
    # 当日量/前一日量：小于2倍不扣（满分），>=2倍按比例扣分，比例越大扣越多
    if idx >= 1 and volume[idx - 1] > 0:
        vol_ratio_prev = volume[idx] / volume[idx - 1]
        if vol_ratio_prev > 2.0:
            deduct = min(10, max(2, int((vol_ratio_prev - 2) * 4)))
            base -= deduct
            if breakdown is not None:
                breakdown.append((f"当日量/前一日量(约{vol_ratio_prev:.1f}倍)", -deduct))
    # 信号日涨停家数前 N 行业与本股行业一致：基础分 + 按该行业当日涨停家数加成（有说服力地体现资金抱团）
    if hot_industries and mode9_hot_industry_bonus > 0:
        ind = (industry or "").strip()
        if ind and ind in hot_industries:
            base_pts = int(mode9_hot_industry_bonus)
            cap = max(base_pts, int(mode9_hot_industry_bonus_max))
            extra_pts = 0
            if hot_industry_counts:
                cnt = int(hot_industry_counts.get(ind, 0))
                room = max(0, cap - base_pts)
                extra_pts = min(room, max(0, (cnt - 1) // 2))
            total_hot = min(cap, base_pts + extra_pts)
            base += total_hot
            if breakdown is not None:
                nlu = int(hot_industry_counts.get(ind, 0)) if hot_industry_counts else 0
                label = "信号日涨停行业TopN"
                if nlu:
                    label += f"（当日该行业涨停{nlu}家）"
                breakdown.append((label, int(total_hot)))
    # 信号日前 N 个交易日：本行业涨停家次累计（资金是否持续涌入该板块）
    if industry_ndays_limit_total is not None:
        ind_nd = (industry or "").strip()
        if ind_nd:
            if industry_ndays_limit_total <= 0 and mode9_industry_ndays_penalty > 0:
                base -= int(mode9_industry_ndays_penalty)
                if breakdown is not None:
                    breakdown.append(
                        ("近N日行业涨停累计0（资金关注度低）", -int(mode9_industry_ndays_penalty))
                    )
            elif industry_ndays_limit_total > 0 and mode9_industry_ndays_bonus_per_unit > 0:
                bu = max(1, int(mode9_industry_ndays_bonus_per_unit))
                add = min(
                    int(mode9_industry_ndays_bonus_cap),
                    int(industry_ndays_limit_total) // bu,
                )
                if add > 0:
                    base += add
                    if breakdown is not None:
                        breakdown.append(
                            (
                                f"近N日行业涨停累计{int(industry_ndays_limit_total)}家次",
                                int(add),
                            )
                        )
    return int(max(0, base))  # 不封顶，允许超过 100


def _score_mode90(
    rows: List[KlineRow],
    idx: int,
    ma10: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
    vol20: np.ndarray,
    code: str = "",
    name: str = "",
    breakdown: Optional[List[tuple]] = None,
    macd_norm_factor: float = 1.0,
    mode90_macd_weight: float = 1.0,
    mode90_macd_max_bonus: float = 12.0,
    mode90_macd_s_scale: float = 0.12,
    industry: str = "",
    hot_industries: Optional[Set[str]] = None,
    mode9_hot_industry_bonus: int = 0,
    hot_industry_counts: Optional[Dict[str, int]] = None,
    mode9_hot_industry_bonus_max: int = 12,
    industry_ndays_limit_total: Optional[int] = None,
    mode9_industry_ndays_penalty: int = 0,
    mode9_industry_ndays_bonus_per_unit: int = 5,
    mode9_industry_ndays_bonus_cap: int = 8,
) -> int:
    """
    mode90 = mode9 评分 + 日线 MACD「贴 0 轴」加分。

    MACD 加分条件（同时满足才加分，否则 MACD 加分为 0）：
    - DIF_norm >= 0、DEA_norm >= 0、HIST_norm >= 0
      （HIST_norm = 2*(DIF-DEA)/denom，与常见 MACD 柱一致；HIST_norm>=0 等价于 DIF>=DEA）
    - 且信号日 DIF_norm、DEA_norm 相对前一日均上升（不允许回落或持平）。

    加分：s = DIF_norm + DEA_norm + HIST_norm；
    贴轴分 = max_bonus * max(0, 1 - s/s_scale) * weight；
    s=0 时满分，s>=s_scale 时 MACD 加分为 0。
    """
    base = _score_mode9(
        rows,
        idx,
        ma10,
        ma20,
        ma60,
        vol20,
        code,
        name,
        breakdown=breakdown,
        industry=industry,
        hot_industries=hot_industries,
        mode9_hot_industry_bonus=mode9_hot_industry_bonus,
        hot_industry_counts=hot_industry_counts,
        mode9_hot_industry_bonus_max=mode9_hot_industry_bonus_max,
        industry_ndays_limit_total=industry_ndays_limit_total,
        mode9_industry_ndays_penalty=mode9_industry_ndays_penalty,
        mode9_industry_ndays_bonus_per_unit=mode9_industry_ndays_bonus_per_unit,
        mode9_industry_ndays_bonus_cap=mode9_industry_ndays_bonus_cap,
    )

    close_arr = np.array([r.close for r in rows], dtype=float)
    dif_norm, dea_norm = _daily_macd_dif_dea(
        close_arr, 12, 26, 9, norm_factor=macd_norm_factor
    )

    if idx < 0 or idx >= len(close_arr):
        return base
    if np.isnan(dif_norm[idx]) or np.isnan(dea_norm[idx]):
        return base

    dn = float(dif_norm[idx])
    en = float(dea_norm[idx])
    hn = 2.0 * (dn - en)  # 柱归一化

    eps = 1e-12
    if dn < -eps or en < -eps or hn < -eps or dn < en - eps:
        return int(max(0, round(base)))

    # 上升趋势约束：当前 DIF/DEA 必须高于前一日（允许极小噪音）
    if idx == 0 or np.isnan(dif_norm[idx - 1]) or np.isnan(dea_norm[idx - 1]):
        return int(max(0, round(base)))
    dn_prev = float(dif_norm[idx - 1])
    en_prev = float(dea_norm[idx - 1])
    if not (dn > dn_prev + eps and en > en_prev + eps):
        return int(max(0, round(base)))

    s = dn + en + hn
    s_scale = float(mode90_macd_s_scale)
    if s_scale <= 0:
        prox = 0.0
    else:
        prox = max(0.0, min(1.0, 1.0 - s / s_scale))
    macd_points = float(mode90_macd_max_bonus) * prox * float(mode90_macd_weight)
    if breakdown is not None and macd_points > 0:
        breakdown.append(("MACD贴轴加分", round(macd_points, 2)))

    return int(max(0, round(base + macd_points)))


def _score_mode8(
    rows: List[KlineRow],
    idx: int,
    ma10: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
    vol20: np.ndarray,
    code: str = "",
    name: str = "",
) -> int:
    """
    mode8 评分（大牛股买点）：在 mode3 评分基础上，按买点前60日涨幅加分。
    - 0% ≤ ret60 ≤ 35%：+5 分；
    - -10% ≤ ret60 < 0%：+2 分。
    见 docs/mode8模型说明.md
    """
    base = _score_mode3(rows, idx, ma10, ma20, ma60, vol20)
    if idx < 60:
        return base
    close = np.array([r.close for r in rows], dtype=float)
    if close[idx - 60] <= 0:
        return base
    ret60 = (close[idx] - close[idx - 60]) / close[idx - 60] * 100
    if 0 <= ret60 <= 35:
        base += 5
    elif -10 <= ret60 < 0:
        base += 2
    return int(max(0, base))


def _score_mode10(
    rows: List[KlineRow],
    idx: int,
    ma10: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
    vol20: np.ndarray,
    code: str = "",
    name: str = "",
) -> int:
    """mode10 评分：以买点前周线拟合最小值为权重，拟合越小分越高。50 + (1 - conv_min)*50，conv_min 为百分比（<1），上限 100。"""
    if idx < 0 or idx >= len(rows):
        return 50
    sub = rows[: idx + 1]
    if len(sub) < 100:
        return 50
    weekly_bars, _ = daily_to_weekly_with_last_index(sub)
    if len(weekly_bars) < 32:
        return 50
    conv = weekly_convergence_value_series(weekly_bars)
    if len(conv) == 0:
        return 50
    target_date = rows[idx].date
    try:
        target_yr, target_wk = datetime.strptime(target_date, "%Y-%m-%d").date().isocalendar()[:2]
    except Exception:
        return 50
    week_keys = [w[0] for w in weekly_bars]
    if (target_yr, target_wk) not in week_keys:
        return 50
    wi = week_keys.index((target_yr, target_wk))
    if wi <= 30:
        return 50
    conv_min = float(np.nanmin(conv[30:wi]))
    if np.isnan(conv_min) or conv_min < 0:
        return 50
    # 拟合值 conv_min 已是百分比（1.0=1%），越小分越高：conv_min=0 -> 100, conv_min>=1 -> 50
    score = 50 + (1.0 - min(1.0, conv_min)) * 50
    return int(max(50, min(100, round(score))))


def _score_mode11(
    rows: List[KlineRow],
    idx: int,
    ma10: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
    vol20: np.ndarray,
    code: str = "",
    name: str = "",
) -> int:
    """mode11 评分：与 mode10 一致，以当周 MA5 斜率加速度 50 + min(50, accel*10)。"""
    return _score_mode10(rows, idx, ma10, ma20, ma60, vol20, code, name)


def _score_mode12(
    rows: List[KlineRow],
    idx: int,
    ma10: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
    vol20: np.ndarray,
    code: str = "",
    name: str = "",
) -> int:
    """mode12 评分：mode10 基础分（加速度）+ 拐点距离加权。拐点距信号 1～5 周分别加 10/8/6/4/2 分，总分上限 100。"""
    base = _score_mode10(rows, idx, ma10, ma20, ma60, vol20, code, name)
    if idx < 0 or idx >= len(rows):
        return base
    sub = rows[: idx + 1]
    if len(sub) < 100:
        return base
    weekly_bars, _ = daily_to_weekly_with_last_index(sub)
    if len(weekly_bars) < 32:
        return base
    closes = np.array([w[4] for w in weekly_bars], dtype=float)
    ma30 = _rolling_mean(closes, 30)
    target_date = rows[idx].date
    try:
        target_yr, target_wk = datetime.strptime(target_date, "%Y-%m-%d").date().isocalendar()[:2]
    except Exception:
        return base
    week_keys = [w[0] for w in weekly_bars]
    if (target_yr, target_wk) not in week_keys:
        return base
    wi = week_keys.index((target_yr, target_wk))
    dist = _nearest_ma30_turn_weeks_before(ma30, wi, 5)
    bonus = 0
    if dist is not None and 1 <= dist <= 5:
        bonus = [10, 8, 6, 4, 2][dist - 1]
    return min(100, base + bonus)


def _buy_point_score(
    rows: List[KlineRow],
    idx: int,
    ma10: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
    vol20: np.ndarray,
) -> int:
    """买点分值 0～100：衡量当日作为买入点的质量（放量、距MA20适中、均线多头、非追高等）。"""
    close = rows[idx].close
    volume = rows[idx].volume
    score = 50.0
    ma20_now = ma20[idx]
    ma60_now = ma60[idx]
    ma10_now = ma10[idx]
    vol20_now = vol20[idx]

    # 放量：量比越大买点越可靠
    if vol20_now > 0:
        vol_ratio = volume / vol20_now
        if vol_ratio >= 1.6:
            score += 15
        elif vol_ratio >= 1.4:
            score += 12
        elif vol_ratio >= 1.2:
            score += 8

    # 收盘相对 MA20：适中突破给高分，追高扣分
    if ma20_now > 0:
        close_gap = (close - ma20_now) / ma20_now
        if 0 <= close_gap <= 0.01:
            score += 15
        elif close_gap <= 0.03:
            score += 12
        elif close_gap <= 0.05:
            score += 5
        elif close_gap > 0.05:
            score -= 5

    # 均线多头排列
    if ma20_now > 0 and ma60_now > 0 and ma10_now > ma20_now and ma20_now > ma60_now:
        score += 10

    # 上影线适中（有试探但不过长）
    rng = rows[idx].high - rows[idx].low
    if rng > 0:
        upper = rows[idx].high - max(rows[idx].open, rows[idx].close)
        upper_ratio = upper / rng
        if 0.2 <= upper_ratio <= 0.5:
            score += 5

    # 近 3 日涨幅过大则扣分（追高）
    if idx >= 3 and rows[idx - 3].close > 0:
        ret3 = (close - rows[idx - 3].close) / rows[idx - 3].close * 100
        if ret3 > 20:
            score -= 15
        elif ret3 > 15:
            score -= 8

    return int(max(0, min(100, round(score))))


def mode3_sort_tuple(r: ScanResult, *, prefer_upper_shadow: bool = False) -> tuple:
    """与 `scan_with_mode3` 内 `_mode3_sort_key` 一致，供区间扫描脚本按日截取 topN 时复用。"""
    metrics = r.metrics or {}
    vol_ratio = float(metrics.get("vol_ratio", 0.0))
    ma20_gap = float(metrics.get("ma20_gap", 0.0))
    ma60_gap = float(metrics.get("ma60_gap", 0.0))
    close_gap = float(metrics.get("close_gap", 0.0))
    ret20_val = float(metrics.get("ret20", 0.0))
    ret5_val = float(metrics.get("ret5", 0.0))
    upper_score = float(metrics.get("upper_score", 0.0))
    buy_point_score = int(metrics.get("buy_point_score", 0))
    limitup_shrink_vol = int(metrics.get("limitup_shrink_vol", 0))
    has_limit_up_6d = int(metrics.get("has_limit_up_6d", 0))
    ir = metrics.get("industry_ret5")
    try:
        irf = float(ir) if ir is not None else float("nan")
    except (TypeError, ValueError):
        irf = float("nan")
    # 行业指数 5 日涨幅高者优先（同分时资金更可能集中在强势板块）
    industry_ret5_key = -irf if not math.isnan(irf) else 0.0
    tr5 = metrics.get("ths_flow_rank_5d")
    try:
        ths5_key = float(tr5) if tr5 is not None else 999.0
        if math.isnan(ths5_key):
            ths5_key = 999.0
    except (TypeError, ValueError):
        ths5_key = 999.0
    tm = metrics.get("ths_flow_momentum")
    try:
        tmf = float(tm) if tm is not None else float("nan")
        if math.isnan(tmf):
            tmf = float("nan")
    except (TypeError, ValueError):
        tmf = float("nan")
    # 同花顺 5 日行业资金排名越靠前越好；flow_momentum 越大表示相对 20 日榜更走强
    ths_mom_key = -tmf if not math.isnan(tmf) else 0.0
    if prefer_upper_shadow:
        return (
            -r.score,
            ret20_val,
            ret5_val,
            -buy_point_score,
            -limitup_shrink_vol,
            has_limit_up_6d,
            -upper_score,
            close_gap,
            -vol_ratio,
            -(ma20_gap + ma60_gap),
            industry_ret5_key,
            ths5_key,
            ths_mom_key,
            r.code,
        )
    return (
        -r.score,
        ret20_val,
        ret5_val,
        -buy_point_score,
        -limitup_shrink_vol,
        has_limit_up_6d,
        close_gap,
        -vol_ratio,
        -(ma20_gap + ma60_gap),
        industry_ret5_key,
        ths5_key,
        ths_mom_key,
        r.code,
    )


def scan_with_mode3(
    stock_list: List[StockItem],
    config: ScanConfig,
    cache_dir: str,
    progress_cb: Optional[Callable[[], None]] = None,
    local_only: bool = False,
    kline_loader: Optional[Callable[[StockItem], Optional[List[KlineRow]]]] = None,
    prefer_local: bool = False,
    cutoff_date: Optional[str] = None,
    start_date: Optional[str] = None,
    market_caps: Optional[Dict[str, float]] = None,
    avoid_big_candle: bool = False,
    big_candle_pct: float = 6.0,
    big_body_ratio: float = 0.6,
    prefer_upper_shadow: bool = False,
    require_upper_shadow: bool = False,
    upper_ratio_min: float = 0.30,
    upper_vol_min: float = 1.50,
    require_vol_ratio: bool = False,
    vol_ratio_min: float = 1.50,
    require_close_gap: bool = False,
    close_gap_max: float = 0.02,
    mode4_filters: bool = False,
    use_71x_standard: bool = False,
    use_mode8: bool = False,
    use_mode9: bool = False,
    use_mode90: bool = False,
    use_mode10: bool = False,
    use_mode11: bool = False,
    use_mode12: bool = False,
    use_mode18: bool = False,
    use_mode88: bool = False,
    use_mode5: bool = False,
    use_mode93: bool = False,
    use_mode_bottom_big_yang: bool = False,
    use_mode_platform_breakout_first_yang: bool = False,
    use_mode_mid_big_yang: bool = False,
    use_mode_bottom_support: bool = False,
    use_mode_final_shakeout: bool = False,
    use_mode98: bool = False,
    use_mode32: bool = False,
    use_mode33: bool = False,
    use_mode34: bool = False,
    use_mode35: bool = False,
    use_mode36: bool = False,
    use_mode37: bool = False,
    use_mode38: bool = False,
    use_mode39: bool = False,
    use_mode40: bool = False,
    use_mode41: bool = False,
    use_mode42: bool = False,
    use_mode43: bool = False,
    use_mode44: bool = False,
    use_mode45: bool = False,
    use_mode46: bool = False,
    sector_ak_cache_dir: Optional[str] = None,
    sector_fund_flow_max_points: int = 5,
    sector_fund_flow_yi_per_point: float = 3.0,
) -> List[ScanResult]:
    """use_mode5/8/9/90/10/11/12/18/88/93/底部大阳线/98/32；mode5 涨停缩量；mode98 日周月 KDJ；mode32 为实体首板后 3+2 整理。"""
    results: List[ScanResult] = []
    from .paths import GPT_DATA_DIR
    from .sector_trend import (
        concept_flow_best_rank_rolling,
        concept_rank_score_bonus,
        eastmoney_industry_flow_bonus,
        eastmoney_industry_flow_rank_today,
        load_stock_concepts,
        merge_ths_flow_features,
        metrics_for_signal,
        parse_ths_flow_net_yi,
        sector_fund_flow_score_delta,
    )

    sector_dir = sector_ak_cache_dir
    if sector_dir is None:
        _cand = os.path.join(GPT_DATA_DIR, "akshare_cache")
        _ind_dir = os.path.join(_cand, "industry")
        if os.path.isdir(_ind_dir):
            try:
                if any(name.endswith(".txt") for name in os.listdir(_ind_dir)):
                    sector_dir = _cand
            except OSError:
                pass
    sector_hist_mem: Dict[str, Optional[List[Dict[str, str]]]] = {}
    ths_features_data: Optional[Dict[str, Any]] = None
    if sector_dir:
        _ths_path = os.path.join(sector_dir, "sector_flow_ths_features.json")
        if os.path.isfile(_ths_path):
            try:
                with open(_ths_path, "r", encoding="utf-8") as _tf:
                    ths_features_data = json.load(_tf)
            except (OSError, json.JSONDecodeError):
                ths_features_data = None
    mode8_n_bars = getattr(config, "mode8_n_bars", 60)
    mode10_conv_max = getattr(config, "mode10_conv_max", 1.0)
    mode10_ma30_turn_weeks = getattr(config, "mode10_ma30_turn_weeks", 5)
    mode11_accel_th = getattr(config, "mode11_accel_th", 2.5)
    mode11_body_ratio_max = getattr(config, "mode11_body_ratio_max", 0.35)
    mode11_vol_ratio_min = getattr(config, "mode11_vol_ratio_min", 1.5)
    mode11_vol_weeks = getattr(config, "mode11_vol_weeks", 20)
    mode12_accel_th = getattr(config, "mode12_accel_th", 2.5)
    mode12_ma30_turn_weeks = getattr(config, "mode12_ma30_turn_weeks", 5)
    if use_mode12:
        signal_fn = lambda rows, start, end: _mode12_signals(
            rows, start, end,
            accel_th=mode12_accel_th,
            ma30_turn_weeks=mode12_ma30_turn_weeks,
        )
        score_fn = _score_mode12
        mode_label = "mode12"
    elif use_mode11:
        signal_fn = lambda rows, start, end: _mode11_signals(
            rows, start, end,
            accel_th=mode11_accel_th,
            body_ratio_max=mode11_body_ratio_max,
            vol_ratio_min=mode11_vol_ratio_min,
            vol_weeks=mode11_vol_weeks,
        )
        score_fn = _score_mode11
        mode_label = "mode11"
    elif use_mode10:
        signal_fn = lambda rows, start, end: _mode10_signals(
            rows, start, end,
            conv_max=mode10_conv_max,
            ma30_turn_weeks=mode10_ma30_turn_weeks,
        )
        score_fn = _score_mode10
        mode_label = "mode10"
    elif use_mode18:
        signal_fn = _mode18_signals
        score_fn = _score_mode18
        mode_label = "mode18"
    elif use_mode88:
        mode88_d_min = getattr(config, "mode88_d_min", 0.03)
        mode88_d_max = getattr(config, "mode88_d_max", 0.15)
        mode88_r_min = getattr(config, "mode88_r_min", 0.03)
        mode88_acc_L = getattr(config, "mode88_acc_L", 8)
        mode88_acc_R = getattr(config, "mode88_acc_R", 20)
        mode88_A_min = getattr(config, "mode88_A_min", 15.0)
        mode88_A_max = getattr(config, "mode88_A_max", 55.0)
        mode88_epsilon = getattr(config, "mode88_epsilon", 0.02)
        mode88_wash_L = getattr(config, "mode88_wash_L", 2)
        mode88_wash_R = getattr(config, "mode88_wash_R", 10)
        mode88_R_rise = getattr(config, "mode88_R_rise", 8.0)
        mode88_D_pull = getattr(config, "mode88_D_pull", 3.0)
        mode88_K_vol = getattr(config, "mode88_K_vol", 1.0)
        signal_fn = lambda rows, start, end: _mode88_signals(
            rows, start, end,
            d_min=mode88_d_min, d_max=mode88_d_max, r_min=mode88_r_min,
            acc_L=mode88_acc_L, acc_R=mode88_acc_R,
            A_min=mode88_A_min, A_max=mode88_A_max, epsilon=mode88_epsilon,
            wash_L=mode88_wash_L, wash_R=mode88_wash_R,
            R_rise=mode88_R_rise, D_pull=mode88_D_pull, K_vol=mode88_K_vol,
        )
        score_fn = _score_mode88
        mode_label = "mode88"
    elif use_mode5:
        # mode5 的 signals 需要 code/name，因此在循环里逐只调用 _mode5_signals
        signal_fn = _mode3_signals
        score_fn = _score_mode5
        mode_label = "mode5"
    elif use_mode93:
        # mode93 的 signals 同样需要 code/name，因此在循环里逐只调用 _mode93_anchor_detail
        signal_fn = _mode3_signals
        score_fn = _score_mode93
        mode_label = "mode93"
    elif use_mode_bottom_big_yang:
        signal_fn = _mode3_signals
        score_fn = _score_mode_bottom_big_yang
        mode_label = "mode底部大阳线"
    elif use_mode_platform_breakout_first_yang:
        signal_fn = _mode3_signals
        score_fn = _score_mode_platform_breakout_first_yang
        mode_label = "mode平台突破首阳"
    elif use_mode_mid_big_yang:
        signal_fn = _mode3_signals
        score_fn = _score_mode_mid_big_yang
        mode_label = "mode中位大阳线"
    elif use_mode_bottom_support:
        signal_fn = _mode3_signals
        score_fn = _score_mode_bottom_support
        mode_label = "mode底部支撑"
    elif use_mode_final_shakeout:
        signal_fn = _mode3_signals
        score_fn = _score_mode_final_shakeout
        mode_label = "mode最后震仓"
    elif use_mode98:
        thr = float(getattr(config, "mode98_kdj_threshold", 20.0))
        n_k = int(getattr(config, "mode98_kdj_n", 9) or 9)
        m1_k = int(getattr(config, "mode98_kdj_m1", 3) or 3)
        m2_k = int(getattr(config, "mode98_kdj_m2", 3) or 3)
        signal_fn = lambda rows, s, e: _mode98_signals(
            rows, s, e, threshold=thr, n=n_k, m1=m1_k, m2=m2_k
        )

        def _score_mode98_bound(
            rows,
            idx,
            ma10,
            ma20,
            ma60,
            vol20,
            code="",
            name="",
            breakdown=None,
        ):
            return _score_mode98(
                rows,
                idx,
                ma10,
                ma20,
                ma60,
                vol20,
                code=code,
                name=name,
                breakdown=breakdown,
                threshold=thr,
                n=n_k,
                m1=m1_k,
                m2=m2_k,
            )

        score_fn = _score_mode98_bound
        mode_label = "mode98"
    elif use_mode38:
        signal_fn = _mode3_signals
        score_fn = _score_mode38
        mode_label = "mode38"
    elif use_mode39:
        signal_fn = _mode3_signals
        score_fn = _score_mode39
        mode_label = "mode39"
    elif use_mode40:
        signal_fn = _mode3_signals
        score_fn = _score_mode40
        mode_label = "mode40"
    elif use_mode41:
        signal_fn = _mode3_signals
        score_fn = _score_mode41
        mode_label = "mode41"
    elif use_mode42:
        signal_fn = _mode3_signals
        score_fn = _score_mode42
        mode_label = "mode42"
    elif use_mode43:
        signal_fn = _mode3_signals
        score_fn = _score_mode43
        mode_label = "mode43"
    elif use_mode44:
        signal_fn = _mode3_signals
        score_fn = _score_mode44
        mode_label = "mode44"
    elif use_mode45:
        signal_fn = _mode3_signals
        score_fn = _score_mode45
        mode_label = "mode45"
    elif use_mode46:
        signal_fn = _mode3_signals
        score_fn = _score_mode46
        mode_label = "mode46"
    elif use_mode37:
        signal_fn = _mode3_signals
        score_fn = _score_mode37
        mode_label = "mode37"
    elif use_mode36:
        signal_fn = _mode3_signals
        score_fn = _score_mode36
        mode_label = "mode36"
    elif use_mode35:
        signal_fn = _mode3_signals
        score_fn = _score_mode35
        mode_label = "mode35"
    elif use_mode34:
        signal_fn = _mode3_signals
        score_fn = _score_mode34
        mode_label = "mode34"
    elif use_mode33:
        signal_fn = _mode3_signals
        score_fn = _score_mode33
        mode_label = "mode33"
    elif use_mode32:
        signal_fn = _mode3_signals
        score_fn = _score_mode32
        mode_label = "mode32"
    elif use_mode90:
        signal_fn = _mode9_signals
        macd_norm_factor = getattr(config, "macd_norm_factor", 1.0)
        mode90_macd_weight = getattr(config, "mode90_macd_weight", 1.0)
        mode90_macd_max_bonus = getattr(config, "mode90_macd_max_bonus", 12.0)
        mode90_macd_s_scale = getattr(config, "mode90_macd_s_scale", 0.12)

        def _score_mode90_fn(
            rows,
            idx,
            ma10,
            ma20,
            ma60,
            vol20,
            code="",
            name="",
            breakdown=None,
            industry: str = "",
            hot_industries: Optional[Set[str]] = None,
            mode9_hot_industry_bonus: int = 0,
            hot_industry_counts: Optional[Dict[str, int]] = None,
            mode9_hot_industry_bonus_max: int = 12,
            industry_ndays_limit_total: Optional[int] = None,
            mode9_industry_ndays_penalty: int = 0,
            mode9_industry_ndays_bonus_per_unit: int = 5,
            mode9_industry_ndays_bonus_cap: int = 8,
        ) -> int:
            return _score_mode90(
                rows,
                idx,
                ma10,
                ma20,
                ma60,
                vol20,
                code=code,
                name=name,
                breakdown=breakdown,
                macd_norm_factor=macd_norm_factor,
                mode90_macd_weight=mode90_macd_weight,
                mode90_macd_max_bonus=mode90_macd_max_bonus,
                mode90_macd_s_scale=mode90_macd_s_scale,
                industry=industry,
                hot_industries=hot_industries,
                mode9_hot_industry_bonus=mode9_hot_industry_bonus,
                hot_industry_counts=hot_industry_counts,
                mode9_hot_industry_bonus_max=mode9_hot_industry_bonus_max,
                industry_ndays_limit_total=industry_ndays_limit_total,
                mode9_industry_ndays_penalty=mode9_industry_ndays_penalty,
                mode9_industry_ndays_bonus_per_unit=mode9_industry_ndays_bonus_per_unit,
                mode9_industry_ndays_bonus_cap=mode9_industry_ndays_bonus_cap,
            )

        score_fn = _score_mode90_fn
        mode_label = "mode90"
    else:
        # mode3 / mode8 / mode9 为三套独立模型：信号上 mode8 与 mode3/mode9 不同，评分上三者均不同。见 docs/mode3_mode8_mode9_三者区别.md
        signal_fn = (
            _mode8_signals
            if use_mode8
            else (_mode9_signals if use_mode9 else _mode3_signals)
        )
        score_fn = _score_mode8 if use_mode8 else (_score_mode9 if use_mode9 else _score_mode3)
        mode_label = "mode8" if use_mode8 else ("mode9" if use_mode9 else ("mode4" if mode4_filters else "mode3"))
    end_date = cutoff_date

    hot_cache: Dict[str, Tuple[Set[str], Dict[str, int]]] = {}
    hot_bonus = int(getattr(config, "mode9_hot_industry_bonus", 0) or 0)
    hot_top_n = max(1, int(getattr(config, "mode9_hot_industry_top_n", 5) or 5))
    hot_bonus_max = max(
        hot_bonus,
        int(getattr(config, "mode9_hot_industry_bonus_max", 12) or 12),
    )
    ndays_n = int(getattr(config, "mode9_industry_limit_ndays", 0) or 0)
    ndays_pen = int(getattr(config, "mode9_industry_ndays_penalty", 3) or 0)
    ndays_unit = int(getattr(config, "mode9_industry_ndays_bonus_per_unit", 5) or 5)
    ndays_cap_cfg = int(getattr(config, "mode9_industry_ndays_bonus_cap", 8) or 8)
    ndays_cache: Dict[str, Tuple[Dict[str, int], bool]] = {}
    em_top_n = int(getattr(config, "em_industry_flow_top_n", 10) or 10)
    em_bonus = int(getattr(config, "em_industry_flow_bonus", 3) or 0)

    mode33_kw: Dict[str, Any] = {}
    if use_mode33:
        mode33_kw = dict(
            anchor_lookback=int(getattr(config, "mode33_anchor_lookback", 300) or 300),
            anchor_body_min=float(getattr(config, "mode33_anchor_body_min", 0.32) or 0.32),
            anchor_vol_mult=float(getattr(config, "mode33_anchor_vol_mult", 1.20) or 1.20),
            break_tol=float(getattr(config, "mode33_break_tol", 0.005) or 0.005),
            shakeout_days_min=int(getattr(config, "mode33_shakeout_days_min", 55) or 55),
            shakeout_days_max=int(getattr(config, "mode33_shakeout_days_max", 165) or 165),
            sideways_range_pct=float(
                getattr(config, "mode33_sideways_range_pct", 0.62) or 0.62
            ),
            trial_after_anchor_min=int(
                getattr(config, "mode33_trial_after_anchor_min", 35) or 35
            ),
            trial_box_pct=float(getattr(config, "mode33_trial_box_pct", 0.10) or 0.10),
            require_mid_rebound=bool(
                getattr(config, "mode33_require_mid_rebound", True)
            ),
            mid_rebound_min_shake_frac=float(
                getattr(config, "mode33_mid_rebound_min_shake_frac", 0.28) or 0.28
            ),
            mid_rebound_min_rise_pct=float(
                getattr(config, "mode33_mid_rebound_min_rise_pct", 0.12) or 0.12
            ),
            mid_surge_pct_min=float(
                getattr(config, "mode33_mid_surge_pct_min", 5.5) or 5.5
            ),
            mid_surge_vol_mult=float(
                getattr(config, "mode33_mid_surge_vol_mult", 1.25) or 1.25
            ),
            mid_min_days_before_trial=int(
                getattr(config, "mode33_mid_min_days_before_trial", 8) or 8
            ),
            mid_pullback_min_pct=float(
                getattr(config, "mode33_mid_pullback_min_pct", 0.06) or 0.06
            ),
            final_day_min=int(getattr(config, "mode33_final_day_min", 6) or 6),
            final_day_max=int(getattr(config, "mode33_final_day_max", 20) or 20),
            final_vol_mult=float(getattr(config, "mode33_final_vol_mult", 1.50) or 1.50),
            box_end_vol_min=float(getattr(config, "mode33_box_end_vol_min", 0.60) or 0.60),
            box_end_vol_max=float(getattr(config, "mode33_box_end_vol_max", 1.40) or 1.40),
            box_end_day_max=int(getattr(config, "mode33_box_end_day_max", 12) or 12),
            final_pct_max=float(getattr(config, "mode33_final_pct_max", 5.0) or 5.0),
            final_body_max=float(getattr(config, "mode33_final_body_max", 0.95) or 0.95),
            vol_ma=int(getattr(config, "mode33_vol_ma", 20) or 20),
            ma_slope_days=int(getattr(config, "mode33_ma_slope_days", 5) or 5),
            ma120_slope_min_pct=float(
                getattr(config, "mode33_ma120_slope_min_pct", 0.01) or 0.01
            ),
            ma120_slope_max_pct=float(
                getattr(config, "mode33_ma120_slope_max_pct", 0.55) or 0.55
            ),
            ma250_slope_min_pct=float(
                getattr(config, "mode33_ma250_slope_min_pct", 0.01) or 0.01
            ),
            ma250_slope_max_pct=float(
                getattr(config, "mode33_ma250_slope_max_pct", 1.05) or 1.05
            ),
        )

    mode34_kw: Dict[str, Any] = {}
    if use_mode34:
        mode34_kw = _mode34_kw_from_config(config)

    mode35_kw: Dict[str, Any] = {}
    if use_mode35:
        mode35_kw = _mode35_kw_from_config(config)

    mode36_kw: Dict[str, Any] = {}
    if use_mode36:
        mode36_kw = _mode36_kw_from_config(config)

    mode38_kw: Dict[str, Any] = {}
    if use_mode38:
        mode38_kw = _mode38_kw_from_config(config)
    mode39_kw: Dict[str, Any] = {}
    if use_mode39:
        mode39_kw = _mode39_kw_from_config(config)
    mode40_kw: Dict[str, Any] = {}
    if use_mode40:
        mode40_kw = _mode40_kw_from_config(config)
    mode41_kw: Dict[str, Any] = {}
    if use_mode41:
        mode41_kw = _mode41_kw_from_config(config)
    mode42_kw: Dict[str, Any] = {}
    if use_mode42:
        mode42_kw = _mode42_kw_from_config(config)
    mode43_kw: Dict[str, Any] = {}
    if use_mode43:
        mode43_kw = _mode43_kw_from_config(config)
    mode44_kw: Dict[str, Any] = {}
    if use_mode44:
        mode44_kw = _mode44_kw_from_config(config)
    mode45_kw: Dict[str, Any] = {}
    if use_mode45:
        mode45_kw = _mode45_kw_from_config(config)
    mode46_kw: Dict[str, Any] = {}
    if use_mode46:
        mode46_kw = _mode46_kw_from_config(config)

    mode37_kw: Dict[str, Any] = {}
    if use_mode37:
        mode37_kw = _mode37_kw_from_config(config)

    for item in stock_list:
        if _is_st(item.name or ""):
            continue
        if progress_cb:
            progress_cb()
        cap_value = None
        if config.max_market_cap and market_caps is not None:
            cap_value = market_caps.get(_normalize_code(item.code))
            if cap_value is None:
                continue
            if cap_value > config.max_market_cap:
                continue
        try:
            if kline_loader:
                rows = kline_loader(item)
            else:
                from .eastmoney import get_kline_cached

                rows = get_kline_cached(
                    item.secid,
                    cache_dir=cache_dir,
                    count=max(260, config.year_lookback + 5),
                    max_age_days=config.cache_days,
                    pause=0.0,
                    local_only=local_only,
                    prefer_local=prefer_local,
                )
        except Exception:
            rows = None
        min_rows = max(80, mode8_n_bars) if use_mode8 else (
            100
            if (use_mode10 or use_mode11 or use_mode12)
            else (
                260
                if use_mode88
                else (
                    200
                    if (
                        use_mode18
                        or use_mode98
                        or use_mode32
                        or use_mode33
                        or use_mode34
                        or use_mode35
                        or use_mode36
                        or use_mode37
                        or use_mode38
                        or use_mode39
                        or use_mode40
                        or use_mode41
                        or use_mode42
                        or use_mode43
                        or use_mode44
                        or use_mode45
                        or use_mode46
                    )
                    else (
                        max(130, int(getattr(config, "mode5_half_year_bars", 120)) + 5)
                        if use_mode5
                        else (
                            max(160, int(getattr(config, "mode93_low_window", 120)) + 10)
                            if use_mode93
                            else (
                                max(
                                    70,
                                    int(getattr(config, "modebbd_low_lookback", 60))
                                    + int(getattr(config, "modebbd_vol_ma", 20))
                                    + 5,
                                )
                                if use_mode_bottom_big_yang
                                else (
                                    max(
                                        120,
                                        int(getattr(config, "modepbs_phase_days_max", 95))
                                        + int(getattr(config, "modepbs_high100_lookback", 100))
                                        + 5,
                                    )
                                    if use_mode_platform_breakout_first_yang
                                    else (
                                        max(
                                            120,
                                            int(getattr(config, "mode_mby_anchor_days_max", 200))
                                            + int(getattr(config, "mode_mby_high100_lookback", 100))
                                            + 5,
                                        )
                                        if use_mode_mid_big_yang
                                        else (
                                            max(
                                                220,
                                                int(getattr(config, "mode_mbs_anchor_days_max", 200))
                                                + int(getattr(config, "mode_mbs_low_lookback", 60))
                                                + 5,
                                            )
                                            if use_mode_bottom_support
                                            else (
                                                max(
                                                    200,
                                                    int(getattr(config, "mode_mfs_phase_days_max", 90))
                                                    + int(getattr(config, "mode_mfs_phase_low_lookback", 90))
                                                    + 5,
                                                )
                                                if use_mode_final_shakeout
                                                else 80
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
        if not rows or len(rows) < min_rows:
            continue

        if end_date:
            filtered = []
            end_dt = _parse_date(end_date)
            scan_end_dt = end_dt
            # mode39 等信号日 T+1 企稳确认，截止日当日也需多读几天 K 线
            if end_dt and (use_mode39 or use_mode38 or use_mode40 or use_mode41 or use_mode42 or use_mode43 or use_mode44 or use_mode45 or use_mode46):
                scan_end_dt = end_dt + timedelta(days=8)
            if end_dt:
                for row in rows:
                    try:
                        row_dt = datetime.strptime(row.date, "%Y-%m-%d").date()
                    except Exception:
                        continue
                    if row_dt <= scan_end_dt:
                        filtered.append(row)
                rows = filtered

        if len(rows) < config.year_lookback + 5:
            continue

        # 标准71倍模型（与脚本一致）不做一年高低价比过滤；否则排除 1年最高/最低>=4 倍
        if not use_71x_standard and len(rows) >= config.year_lookback:
            window = rows[-config.year_lookback:]
            max_high = max(r.high for r in window)
            min_low = min(r.low for r in window)
            if min_low > 0 and max_high / min_low >= config.year_high_low_ratio_limit:
                continue

        if use_mode5:
            m5_shrink_d = max(3, int(getattr(config, "mode5_shrink_max_days", 5) or 5))
            m5_hb = max(60, int(getattr(config, "mode5_half_year_bars", 120) or 120))
            signals = _mode5_signals(
                rows,
                start_date,
                end_date,
                item.code,
                item.name,
                shrink_max_days=m5_shrink_d,
                half_year_bars=m5_hb,
            )
        elif use_mode38:
            st = str(start_date).strip()[:10] if start_date else ""
            ed = str(end_date).strip()[:10] if end_date else ""
            start_i = max(260, int(mode38_kw.get("phase_lookback", 120) or 120) + 10)
            signals = []
            for i in range(start_i, len(rows)):
                d = str(rows[i].date)[:10]
                if st and d < st:
                    continue
                if ed and d > ed:
                    continue
                if _mode38_signal_at(rows, i, item.code, item.name, **mode38_kw):
                    signals.append(i)
        elif use_mode39:
            st = str(start_date).strip()[:10] if start_date else ""
            ed = str(end_date).strip()[:10] if end_date else ""
            start_i = max(
                160,
                int(mode39_kw.get("anchor_lookback_max", 120) or 120)
                + int(mode39_kw.get("ma45_period", 45) or 45)
                + int(mode39_kw.get("ma45_slope_days", 10) or 10)
                + 5,
            )
            signals = []
            for i in range(start_i, len(rows)):
                d = str(rows[i].date)[:10]
                if st and d < st:
                    continue
                if ed and d > ed:
                    continue
                if _mode39_signal_at(rows, i, item.code, item.name, **mode39_kw):
                    signals.append(i)
        elif use_mode40:
            st = str(start_date).strip()[:10] if start_date else ""
            ed = str(end_date).strip()[:10] if end_date else ""
            start_i = max(
                160,
                int(mode40_kw.get("peak_lookback", 60) or 60)
                + int(mode40_kw.get("ma_period", 60) or 60)
                + int(mode40_kw.get("pullback_days_max", 10) or 10)
                + int(mode40_kw.get("rebound_window", 3) or 3)
                + 5,
            )
            signals = []
            for i in range(start_i, len(rows)):
                d = str(rows[i].date)[:10]
                if st and d < st:
                    continue
                if ed and d > ed:
                    continue
                if _mode40_signal_at(rows, i, item.code, item.name, **mode40_kw):
                    signals.append(i)
        elif use_mode41:
            st = str(start_date).strip()[:10] if start_date else ""
            ed = str(end_date).strip()[:10] if end_date else ""
            start_i = max(
                320,
                int(mode41_kw.get("phase_lookback_weeks", 48) or 48) * 5
                + int(mode41_kw.get("peak_lookback_weeks", 20) or 20) * 5
                + 20,
            )
            signals = []
            for i in range(start_i, len(rows)):
                d = str(rows[i].date)[:10]
                if st and d < st:
                    continue
                if ed and d > ed:
                    continue
                if _mode41_signal_at(rows, i, item.code, item.name, **mode41_kw):
                    signals.append(i)
        elif use_mode42:
            st = str(start_date).strip()[:10] if start_date else ""
            ed = str(end_date).strip()[:10] if end_date else ""
            start_i = max(
                280,
                int(mode42_kw.get("phase_lookback_weeks", 48) or 48) * 5
                + int(mode42_kw.get("peak_lookback_weeks", 20) or 20) * 5
                + 20,
            )
            signals = []
            for i in range(start_i, len(rows)):
                d = str(rows[i].date)[:10]
                if st and d < st:
                    continue
                if ed and d > ed:
                    continue
                if _mode42_signal_at(rows, i, item.code, item.name, **mode42_kw):
                    signals.append(i)
        elif use_mode43:
            st = str(start_date).strip()[:10] if start_date else ""
            ed = str(end_date).strip()[:10] if end_date else ""
            start_i = max(
                280,
                int(mode43_kw.get("min_weeks_history", 16) or 16) * 5 + 20,
            )
            signals = []
            for i in range(start_i, len(rows)):
                d = str(rows[i].date)[:10]
                if st and d < st:
                    continue
                if ed and d > ed:
                    continue
                if _mode43_signal_at(rows, i, item.code, item.name, **mode43_kw):
                    signals.append(i)
        elif use_mode44:
            st = str(start_date).strip()[:10] if start_date else ""
            ed = str(end_date).strip()[:10] if end_date else ""
            streak = int(mode44_kw.get("streak_days", 3) or 3)
            ma120_n = int(mode44_kw.get("ma120_period", 120) or 120)
            start_i = max(80, streak + 5, ma120_n)
            signals = []
            for i in range(start_i, len(rows)):
                d = str(rows[i].date)[:10]
                if st and d < st:
                    continue
                if ed and d > ed:
                    continue
                if _mode44_signal_at(rows, i, item.code, item.name, **mode44_kw):
                    signals.append(i)
        elif use_mode45:
            st = str(start_date).strip()[:10] if start_date else ""
            ed = str(end_date).strip()[:10] if end_date else ""
            max_g = int(mode45_kw.get("max_grind_days", 10) or 10)
            high_lb = int(mode45_kw.get("high_lookback", 60) or 60)
            start_i = max(high_lb + max_g + 5, 130)
            signals = []
            for i in range(start_i, len(rows)):
                d = str(rows[i].date)[:10]
                if st and d < st:
                    continue
                if ed and d > ed:
                    continue
                if _mode45_signal_at(rows, i, item.code, item.name, **mode45_kw):
                    signals.append(i)
        elif use_mode46:
            st = str(start_date).strip()[:10] if start_date else ""
            ed = str(end_date).strip()[:10] if end_date else ""
            peak_lb = int(mode46_kw.get("peak_lookback", 100) or 100)
            start_i = max(peak_lb + 15, 120)
            signals = []
            for i in range(start_i, len(rows)):
                d = str(rows[i].date)[:10]
                if st and d < st:
                    continue
                if ed and d > ed:
                    continue
                if _mode46_signal_at(rows, i, item.code, item.name, **mode46_kw):
                    signals.append(i)
        elif use_mode37:
            st = str(start_date).strip()[:10] if start_date else ""
            ed = str(end_date).strip()[:10] if end_date else ""
            start_i = max(
                80,
                int(mode37_kw.get("vol_ma", 20) or 20) + 5,
            )
            signals = []
            for i in range(start_i, len(rows)):
                d = str(rows[i].date)[:10]
                if st and d < st:
                    continue
                if ed and d > ed:
                    continue
                if _mode37_signal_at(rows, i, item.code, item.name, **mode37_kw):
                    signals.append(i)
        elif use_mode36:
            st = str(start_date).strip()[:10] if start_date else ""
            ed = str(end_date).strip()[:10] if end_date else ""
            periods = mode36_kw.get("ma_periods") or [5, 10, 20, 30, 60, 120]
            start_i = max(130, max(int(p) for p in periods) + 5)
            signals = []
            for i in range(start_i, len(rows)):
                d = str(rows[i].date)[:10]
                if st and d < st:
                    continue
                if ed and d > ed:
                    continue
                if _mode36_signal_at(rows, i, item.code, item.name, **mode36_kw):
                    signals.append(i)
        elif use_mode35:
            st = str(start_date).strip()[:10] if start_date else ""
            ed = str(end_date).strip()[:10] if end_date else ""
            signals = []
            start_i = max(
                200,
                int(mode35_kw.get("anchor_lookback", 180))
                + int(mode35_kw.get("min_under_days", 40))
                + 15,
            )
            for i in range(start_i, len(rows)):
                d = str(rows[i].date)[:10]
                if st and d < st:
                    continue
                if ed and d > ed:
                    continue
                if _mode35_signal_at(rows, i, item.code, item.name, **mode35_kw):
                    signals.append(i)
        elif use_mode34:
            st = str(start_date).strip()[:10] if start_date else ""
            ed = str(end_date).strip()[:10] if end_date else ""
            signals = []
            start_i = 90
            for i in range(start_i, len(rows)):
                d = str(rows[i].date)[:10]
                if st and d < st:
                    continue
                if ed and d > ed:
                    continue
                if _mode34_signal_at(rows, i, item.code, item.name, **mode34_kw):
                    signals.append(i)
        elif use_mode33:
            msd = int(mode33_kw.get("ma_slope_days", 5) or 5)
            st = str(start_date).strip()[:10] if start_date else ""
            ed = str(end_date).strip()[:10] if end_date else ""
            signals = []
            start_i = max(
                int(mode33_kw.get("anchor_lookback", 120))
                + int(mode33_kw.get("trial_after_anchor_min", 40))
                + int(mode33_kw.get("final_day_max", 15))
                + 6,
                250 + msd,
            )
            for i in range(start_i, len(rows)):
                d = str(rows[i].date)[:10]
                if st and d < st:
                    continue
                if ed and d > ed:
                    continue
                if _mode33_signal_at(rows, i, item.code, item.name, **mode33_kw):
                    signals.append(i)
        elif use_mode32:
            L = int(getattr(config, "mode32_sideways_days", 60) or 60)
            sr = float(getattr(config, "mode32_sideways_range_pct", 0.44) or 0.44)
            d1b = float(getattr(config, "mode32_day1_body_max", 0.50) or 0.50)
            d1v = float(getattr(config, "mode32_day1_vol_vs_limit_min", 1.0) or 1.0)
            nh = float(getattr(config, "mode32_near_high_pct", 0.028) or 0.028)
            d23 = float(getattr(config, "mode32_days23_low_min_frac", 0.97) or 0.97)
            d45b = float(getattr(config, "mode32_day45_body_max", 0.95) or 0.95)
            v43 = float(getattr(config, "mode32_vol_day43_vs_day3_max", 1.20) or 1.20)
            v54 = float(getattr(config, "mode32_vol_day5_vs_day4_max", 1.08) or 1.08)
            v45 = float(getattr(config, "mode32_vol_day45_vs_day1_max", 0.72) or 0.72)
            midm = float(getattr(config, "mode32_min_close_vs_mid", 1.0) or 1.0)
            msd = int(getattr(config, "mode32_ma_slope_days", 5) or 5)
            s120min = float(getattr(config, "mode32_ma120_slope_min_pct", 0.01) or 0.01)
            s120max = float(getattr(config, "mode32_ma120_slope_max_pct", 0.55) or 0.55)
            s250min = float(getattr(config, "mode32_ma250_slope_min_pct", 0.01) or 0.01)
            s250max = float(getattr(config, "mode32_ma250_slope_max_pct", 1.05) or 1.05)
            st = str(start_date).strip()[:10] if start_date else ""
            ed = str(end_date).strip()[:10] if end_date else ""
            signals = []
            start_i = max(L + 6, 250 + msd, 7)
            for i in range(start_i, len(rows)):
                d = str(rows[i].date)[:10]
                if st and d < st:
                    continue
                if ed and d > ed:
                    continue
                if _mode32_signal_at(
                    rows,
                    i,
                    item.code,
                    item.name,
                    sideways_days=L,
                    sideways_range_pct=sr,
                    day1_body_max=d1b,
                    day1_vol_vs_limit_min=d1v,
                    near_high_pct=nh,
                    days23_low_min_frac=d23,
                    day45_body_max=d45b,
                    vol_day43_vs_day3_max=v43,
                    vol_day5_vs_day4_max=v54,
                    vol_day45_vs_day1_max=v45,
                    min_close_vs_mid=midm,
                    ma_slope_days=msd,
                    ma120_slope_min_pct=s120min,
                    ma120_slope_max_pct=s120max,
                    ma250_slope_min_pct=s250min,
                    ma250_slope_max_pct=s250max,
                ):
                    signals.append(i)
        elif use_mode93:
            # mode93: 逐点判定（需要 code/name + 参数）
            m93_lookback = int(getattr(config, "mode93_lookback_days", 20) or 20)
            m93_low_win = int(getattr(config, "mode93_low_window", 120) or 120)
            m93_low_recent = int(getattr(config, "mode93_low_recent_days", 3) or 3)
            m93_vol_mult = float(getattr(config, "mode93_vol_mult", 3.0) or 3.0)
            m93_pb_min = float(getattr(config, "mode93_pullback_min", 0.99) or 0.99)
            m93_pb_max = float(getattr(config, "mode93_pullback_max", 1.02) or 1.02)
            m93_pb_days = int(getattr(config, "mode93_pullback_max_days", 20) or 20)
            st = str(start_date).strip()[:10] if start_date else ""
            ed = str(end_date).strip()[:10] if end_date else ""
            signals = []
            for i in range(max(m93_low_win + 5, 5), len(rows)):
                d = str(rows[i].date)[:10]
                if st and d < st:
                    continue
                if ed and d > ed:
                    continue
                if _mode93_anchor_detail(
                    rows,
                    i,
                    item.code,
                    item.name,
                    lookback_days=m93_lookback,
                    low_window=m93_low_win,
                    low_recent_days=m93_low_recent,
                    vol_mult=m93_vol_mult,
                    pullback_min=m93_pb_min,
                    pullback_max=m93_pb_max,
                    pullback_max_days=m93_pb_days,
                ):
                    signals.append(i)
        elif use_mode_bottom_big_yang:
            mbbd_low = int(getattr(config, "modebbd_low_lookback", 60) or 60)
            mbbd_pos = float(getattr(config, "modebbd_bottom_pos_max", 0.50) or 0.50)
            mbbd_pct = float(getattr(config, "modebbd_big_pct_min", 5.0) or 5.0)
            mbbd_body = float(getattr(config, "modebbd_body_ratio_min", 0.55) or 0.55)
            mbbd_vm = float(getattr(config, "modebbd_vol_mult", 2.0) or 2.0)
            mbbd_vma = int(getattr(config, "modebbd_vol_ma", 20) or 20)
            mbbd_sd = int(getattr(config, "modebbd_sudden_days", 5) or 5)
            mbbd_pvr = float(getattr(config, "modebbd_prior_vol_ratio_max", 0.65) or 0.65)
            need_i = max(mbbd_low + 1, mbbd_vma + 1, mbbd_sd + 2)
            st = str(start_date).strip()[:10] if start_date else ""
            ed = str(end_date).strip()[:10] if end_date else ""
            signals = []
            for i in range(need_i, len(rows)):
                d = str(rows[i].date)[:10]
                if st and d < st:
                    continue
                if ed and d > ed:
                    continue
                if _match_mode_bottom_big_yang(
                    rows,
                    i,
                    item.code,
                    item.name,
                    low_lookback=mbbd_low,
                    bottom_pos_max=mbbd_pos,
                    big_pct_min=mbbd_pct,
                    body_ratio_min=mbbd_body,
                    vol_mult=mbbd_vm,
                    vol_ma=mbbd_vma,
                    sudden_days=mbbd_sd,
                    prior_vol_ratio_max=mbbd_pvr,
                ):
                    signals.append(i)
        elif use_mode_platform_breakout_first_yang:
            mpbs_pmin = int(getattr(config, "modepbs_phase_days_min", 45) or 45)
            mpbs_pmax = int(getattr(config, "modepbs_phase_days_max", 95) or 95)
            mpbs_rmin = float(getattr(config, "modepbs_rise_from_low_min", 0.20) or 0.20)
            mpbs_rmax = float(getattr(config, "modepbs_rise_from_low_max", 0.55) or 0.55)
            mpbs_cd = int(getattr(config, "modepbs_consolid_days", 20) or 20)
            mpbs_ca = float(getattr(config, "modepbs_consolid_amp_max", 0.30) or 0.30)
            mpbs_bl = int(getattr(config, "modepbs_breakout_lookback", 60) or 60)
            mpbs_bn = float(getattr(config, "modepbs_breakout_near_min", 0.93) or 0.93)
            mpbs_pct = float(getattr(config, "modepbs_big_pct_min", 7.0) or 7.0)
            mpbs_pct_main = float(getattr(config, "modepbs_big_pct_min_main", 4.5) or 4.5)
            mpbs_body = float(getattr(config, "modepbs_body_ratio_min", 0.55) or 0.55)
            mpbs_vm = float(getattr(config, "modepbs_vol_mult", 1.25) or 1.25)
            mpbs_vma = int(getattr(config, "modepbs_vol_ma", 20) or 20)
            mpbs_gap = int(getattr(config, "modepbs_big_yang_gap", 15) or 15)
            mpbs_gbn = float(getattr(config, "modepbs_gap_breakout_near_min", 0.93) or 0.93)
            mpbs_h100 = int(getattr(config, "modepbs_high100_lookback", 100) or 100)
            mpbs_h100n = float(getattr(config, "modepbs_high100_near_min", 0.93) or 0.93)
            mpbs_vmax = float(getattr(config, "modepbs_vol_ratio_max", 4.0) or 4.0)
            mpbs_vext = float(getattr(config, "modepbs_vol_ratio_extended_max", 6.5) or 6.5)
            mpbs_vhw = int(getattr(config, "modepbs_vol_high100_wash_min", 3) or 3)
            mpbs_umax = float(getattr(config, "modepbs_upper_ratio_max", 0.35) or 0.35)
            mpbs_uext = float(getattr(config, "modepbs_upper_ratio_extended_max", 0.30) or 0.30)
            mpbs_uhv = float(getattr(config, "modepbs_upper_high100_vol_min", 4.0) or 4.0)
            mpbs_wcm = int(getattr(config, "modepbs_wash_close_min_cnt", 2) or 2)
            mpbs_wc60 = float(getattr(config, "modepbs_wash_close60_min", 0.98) or 0.98)
            mpbs_pr5 = float(getattr(config, "modepbs_pre_rise5_min", -0.05))
            mpbs_pr5max = float(getattr(config, "modepbs_pre_rise5_max", 0.10) or 0.10)
            mpbs_hrda = float(
                getattr(config, "modepbs_high_rise_wash_drop_rise_above", 0.38) or 0.38
            )
            mpbs_hrdm = float(getattr(config, "modepbs_high_rise_wash_drop_min", 0.08) or 0.08)
            mpbs_wcsmax = float(getattr(config, "modepbs_weekly_conv_sig_max", 15.0) or 15.0)
            mpbs_wcimin = float(getattr(config, "modepbs_weekly_conv_improve_min", -1.5) or -1.5)
            need_i = max(
                mpbs_pmax + 1,
                mpbs_bl + 1,
                mpbs_h100 + 1,
                mpbs_cd + 1,
                mpbs_vma + 1,
                mpbs_gap + 2,
            )
            st = str(start_date).strip()[:10] if start_date else ""
            ed = str(end_date).strip()[:10] if end_date else ""
            signals = []
            for i in range(need_i, len(rows)):
                d = str(rows[i].date)[:10]
                if st and d < st:
                    continue
                if ed and d > ed:
                    continue
                if _match_mode_platform_breakout_first_yang(
                    rows,
                    i,
                    item.code,
                    item.name,
                    phase_days_min=mpbs_pmin,
                    phase_days_max=mpbs_pmax,
                    rise_from_low_min=mpbs_rmin,
                    rise_from_low_max=mpbs_rmax,
                    consolid_days=mpbs_cd,
                    consolid_amp_max=mpbs_ca,
                    breakout_lookback=mpbs_bl,
                    breakout_near_min=mpbs_bn,
                    big_pct_min=mpbs_pct,
                    big_pct_min_main=mpbs_pct_main,
                    body_ratio_min=mpbs_body,
                    vol_mult=mpbs_vm,
                    vol_ma=mpbs_vma,
                    big_yang_gap=mpbs_gap,
                    gap_breakout_near_min=mpbs_gbn,
                    high100_lookback=mpbs_h100,
                    high100_near_min=mpbs_h100n,
                    vol_ratio_max=mpbs_vmax,
                    vol_ratio_extended_max=mpbs_vext,
                    vol_high100_wash_min=mpbs_vhw,
                    upper_ratio_max=mpbs_umax,
                    upper_ratio_extended_max=mpbs_uext,
                    upper_high100_vol_min=mpbs_uhv,
                    wash_close_min_cnt=mpbs_wcm,
                    wash_close60_min=mpbs_wc60,
                    pre_rise5_min=mpbs_pr5,
                    pre_rise5_max=mpbs_pr5max,
                    high_rise_wash_drop_rise_above=mpbs_hrda,
                    high_rise_wash_drop_min=mpbs_hrdm,
                    weekly_conv_sig_max=mpbs_wcsmax,
                    weekly_conv_improve_min=mpbs_wcimin,
                ):
                    signals.append(i)
        elif use_mode_mid_big_yang:
            mby_amin = int(getattr(config, "mode_mby_anchor_days_min", 30) or 30)
            mby_amax = int(getattr(config, "mode_mby_anchor_days_max", 200) or 200)
            mby_avm = float(getattr(config, "mode_mby_anchor_vol_mult", 1.5) or 1.5)
            mby_rmin = float(getattr(config, "mode_mby_rise_from_anchor_min", 0.20) or 0.20)
            mby_rmax = float(getattr(config, "mode_mby_rise_from_anchor_max", 1.20) or 1.20)
            mby_cd = int(getattr(config, "mode_mby_consolid_days", 20) or 20)
            mby_ca = float(getattr(config, "mode_mby_consolid_amp_max", 0.35) or 0.35)
            mby_bl = int(getattr(config, "mode_mby_breakout_lookback", 60) or 60)
            mby_bmin = float(getattr(config, "mode_mby_breakout_min", 1.0) or 1.0)
            mby_h100 = int(getattr(config, "mode_mby_high100_lookback", 100) or 100)
            mby_h100m = float(getattr(config, "mode_mby_high100_min", 1.0) or 1.0)
            mby_tc_amp = float(getattr(config, "mode_mby_tight_consolid_amp_max", 0.15) or 0.15)
            mby_tc_vol = float(getattr(config, "mode_mby_tight_vol_ratio_min", 1.8) or 1.8)
            mby_tc_rmin = float(getattr(config, "mode_mby_tight_rise_from_anchor_min", 0.10) or 0.10)
            mby_tc_h100 = float(getattr(config, "mode_mby_tight_high100_min", 0.985) or 0.985)
            mby_pct = float(getattr(config, "mode_mby_big_pct_min", 7.0) or 7.0)
            mby_pct_main = float(getattr(config, "mode_mby_big_pct_min_main", 4.5) or 4.5)
            mby_body = float(getattr(config, "mode_mby_body_ratio_min", 0.55) or 0.55)
            mby_vm = float(getattr(config, "mode_mby_vol_mult", 1.25) or 1.25)
            mby_vma = int(getattr(config, "mode_mby_vol_ma", 20) or 20)
            mby_vmax = float(getattr(config, "mode_mby_vol_ratio_max", 5.0) or 5.0)
            mby_umax = float(getattr(config, "mode_mby_upper_ratio_max", 0.40) or 0.40)
            mby_cb60 = float(getattr(config, "mode_mby_close_break60_min", 1.0) or 1.0)
            mby_pr5 = float(getattr(config, "mode_mby_pre_rise5_min", -0.05))
            mby_pr5max = float(getattr(config, "mode_mby_pre_rise5_max", 0.15))
            need_i = max(
                mby_amax + 1,
                mby_bl + 1,
                mby_h100 + 1,
                mby_cd + 1,
                mby_vma + 1,
            )
            st = str(start_date).strip()[:10] if start_date else ""
            ed = str(end_date).strip()[:10] if end_date else ""
            signals = []
            for i in range(need_i, len(rows)):
                d = str(rows[i].date)[:10]
                if st and d < st:
                    continue
                if ed and d > ed:
                    continue
                if _match_mode_mid_big_yang(
                    rows,
                    i,
                    item.code,
                    item.name,
                    anchor_days_min=mby_amin,
                    anchor_days_max=mby_amax,
                    anchor_vol_mult=mby_avm,
                    rise_from_anchor_min=mby_rmin,
                    rise_from_anchor_max=mby_rmax,
                    consolid_days=mby_cd,
                    consolid_amp_max=mby_ca,
                    breakout_lookback=mby_bl,
                    breakout_min=mby_bmin,
                    high100_lookback=mby_h100,
                    high100_min=mby_h100m,
                    tight_consolid_amp_max=mby_tc_amp,
                    tight_vol_ratio_min=mby_tc_vol,
                    tight_rise_from_anchor_min=mby_tc_rmin,
                    tight_high100_min=mby_tc_h100,
                    big_pct_min=mby_pct,
                    big_pct_min_main=mby_pct_main,
                    body_ratio_min=mby_body,
                    vol_mult=mby_vm,
                    vol_ma=mby_vma,
                    vol_ratio_max=mby_vmax,
                    upper_ratio_max=mby_umax,
                    close_break60_min=mby_cb60,
                    pre_rise5_min=mby_pr5,
                    pre_rise5_max=mby_pr5max,
                ):
                    signals.append(i)
        elif use_mode_bottom_support:
            mbs_amin = int(getattr(config, "mode_mbs_anchor_days_min", 30) or 30)
            mbs_amax = int(getattr(config, "mode_mbs_anchor_days_max", 200) or 200)
            mbs_ll = int(getattr(config, "mode_mbs_low_lookback", 60) or 60)
            mbs_pos = float(getattr(config, "mode_mbs_bottom_pos_max", 0.50) or 0.50)
            mbs_avm = float(getattr(config, "mode_mbs_anchor_vol_mult", 2.0) or 2.0)
            mbs_vma = int(getattr(config, "mode_mbs_anchor_vol_ma", 20) or 20)
            mbs_pct = float(getattr(config, "mode_mbs_big_pct_min", 5.0) or 5.0)
            mbs_body = float(getattr(config, "mode_mbs_body_ratio_min", 0.55) or 0.55)
            mbs_rally = float(getattr(config, "mode_mbs_min_rally_pct", 0.15) or 0.15)
            mbs_near = float(getattr(config, "mode_mbs_support_near_max", 0.15) or 0.15)
            mbs_brk = float(getattr(config, "mode_mbs_support_break_min", 0.97) or 0.97)
            mbs_test = float(getattr(config, "mode_mbs_test_tol", 0.15) or 0.15)
            mbs_mint = int(getattr(config, "mode_mbs_min_support_tests", 1) or 1)
            mbs_bounce = int(getattr(config, "mode_mbs_bounce_days", 5) or 5)
            mbs_wvm = float(getattr(config, "mode_mbs_weekly_vol_mult", 1.5) or 1.5)
            need_i = max(mbs_amax + 1, mbs_ll + 1, mbs_vma + 1, mbs_bounce + 2)
            st = str(start_date).strip()[:10] if start_date else ""
            ed = str(end_date).strip()[:10] if end_date else ""
            signals = []
            for i in range(need_i, len(rows)):
                d = str(rows[i].date)[:10]
                if st and d < st:
                    continue
                if ed and d > ed:
                    continue
                if _match_mode_bottom_support(
                    rows,
                    i,
                    item.code,
                    item.name,
                    anchor_days_min=mbs_amin,
                    anchor_days_max=mbs_amax,
                    low_lookback=mbs_ll,
                    bottom_pos_max=mbs_pos,
                    anchor_vol_mult=mbs_avm,
                    anchor_vol_ma=mbs_vma,
                    big_pct_min=mbs_pct,
                    body_ratio_min=mbs_body,
                    min_rally_pct=mbs_rally,
                    support_near_max=mbs_near,
                    support_break_min=mbs_brk,
                    test_tol=mbs_test,
                    min_support_tests=mbs_mint,
                    bounce_days=mbs_bounce,
                    weekly_vol_mult=mbs_wvm,
                ):
                    signals.append(i)
        elif use_mode_final_shakeout:
            mfs_pmin = int(getattr(config, "mode_mfs_phase_days_min", 30) or 30)
            mfs_pmax = int(getattr(config, "mode_mfs_phase_days_max", 90) or 90)
            mfs_avm = float(getattr(config, "mode_mfs_anchor_vol_mult", 1.5) or 1.5)
            mfs_rally = float(getattr(config, "mode_mfs_min_rally_pct", 0.10) or 0.10)
            mfs_cd = int(getattr(config, "mode_mfs_consolid_days", 20) or 20)
            mfs_ca = float(getattr(config, "mode_mfs_consolid_amp_max", 0.15) or 0.15)
            mfs_pl = int(getattr(config, "mode_mfs_peak_lookback", 15) or 15)
            mfs_smin = int(getattr(config, "mode_mfs_shakeout_days_min", 3) or 3)
            mfs_smax = int(getattr(config, "mode_mfs_shakeout_days_max", 7) or 7)
            mfs_drop_min = float(getattr(config, "mode_mfs_shakeout_drop_min", 0.10) or 0.10)
            mfs_drop_max = float(getattr(config, "mode_mfs_shakeout_drop_max", 0.22) or 0.22)
            mfs_pll = int(getattr(config, "mode_mfs_phase_low_lookback", 90) or 90)
            mfs_plb = float(getattr(config, "mode_mfs_phase_low_break_min", 0.95) or 0.95)
            mfs_svmn = float(getattr(config, "mode_mfs_shakeout_vol_min", 0.6) or 0.6)
            mfs_svmx = float(getattr(config, "mode_mfs_shakeout_vol_max", 2.5) or 2.5)
            mfs_ma60d = int(getattr(config, "mode_mfs_ma60_slope_days", 20) or 20)
            mfs_rev_pct = float(getattr(config, "mode_mfs_reversal_pct_min", 8.0) or 8.0)
            mfs_rev_vr = float(getattr(config, "mode_mfs_reversal_vol_min", 1.5) or 1.5)
            mfs_rev_tol = float(getattr(config, "mode_mfs_reversal_low_tol", 0.05) or 0.05)
            mfs_brk_pct = float(getattr(config, "mode_mfs_breakout_pct_min", 15.0) or 15.0)
            mfs_brk_main = float(getattr(config, "mode_mfs_breakout_pct_min_main", 9.0) or 9.0)
            mfs_brk_vr = float(getattr(config, "mode_mfs_breakout_vol_min", 3.0) or 3.0)
            mfs_body = float(getattr(config, "mode_mfs_body_ratio_min", 0.55) or 0.55)
            need_i = max(mfs_pmax + 1, mfs_pll + 1, mfs_pl + mfs_smax + 5, mfs_ma60d + 60)
            st = str(start_date).strip()[:10] if start_date else ""
            ed = str(end_date).strip()[:10] if end_date else ""
            signals = []
            for i in range(need_i, len(rows)):
                d = str(rows[i].date)[:10]
                if st and d < st:
                    continue
                if ed and d > ed:
                    continue
                if _match_mode_final_shakeout(
                    rows,
                    i,
                    item.code,
                    item.name,
                    phase_days_min=mfs_pmin,
                    phase_days_max=mfs_pmax,
                    anchor_vol_mult=mfs_avm,
                    min_rally_pct=mfs_rally,
                    consolid_days=mfs_cd,
                    consolid_amp_max=mfs_ca,
                    peak_lookback=mfs_pl,
                    shakeout_days_min=mfs_smin,
                    shakeout_days_max=mfs_smax,
                    shakeout_drop_min=mfs_drop_min,
                    shakeout_drop_max=mfs_drop_max,
                    phase_low_lookback=mfs_pll,
                    phase_low_break_min=mfs_plb,
                    shakeout_vol_min=mfs_svmn,
                    shakeout_vol_max=mfs_svmx,
                    ma60_slope_days=mfs_ma60d,
                    reversal_pct_min=mfs_rev_pct,
                    reversal_vol_min=mfs_rev_vr,
                    reversal_low_tol=mfs_rev_tol,
                    breakout_pct_min=mfs_brk_pct,
                    breakout_pct_min_main=mfs_brk_main,
                    breakout_vol_min=mfs_brk_vr,
                    body_ratio_min=mfs_body,
                ):
                    signals.append(i)
        else:
            signals = signal_fn(rows, start_date, end_date)
        if cutoff_date and not start_date:
            signals = [s for s in signals if rows[s].date == cutoff_date]
        if not start_date and not cutoff_date and signals:
            signals = [signals[-1]]
        if not signals:
            continue

        close = np.array([r.close for r in rows], dtype=float)
        volume = np.array([r.volume for r in rows], dtype=float)
        ma10 = _moving_mean(close, 10)
        ma20 = _moving_mean(close, 20)
        ma60 = _moving_mean(close, 60)
        vol20 = _moving_mean(volume, 20)
        ret20 = np.full_like(close, np.nan, dtype=float)
        if len(close) > 20:
            for i in range(20, len(close)):
                base = close[i - 20]
                if base > 0:
                    ret20[i] = (close[i] - base) / base * 100

        for idx in signals:
            if np.isnan(ma20[idx]) or np.isnan(ma60[idx]) or np.isnan(vol20[idx]):
                continue

            if avoid_big_candle:
                o = rows[idx].open
                c = rows[idx].close
                h = rows[idx].high
                l = rows[idx].low
                rng = h - l
                body = abs(c - o)
                body_ratio = body / rng if rng > 0 else 0.0
                is_big_bull = (
                    c > o
                    and rows[idx].pct_chg >= big_candle_pct
                    and body_ratio >= big_body_ratio
                )
                if is_big_bull:
                    continue

            if mode4_filters:
                # 放宽：仅排除连续3天涨停后跌停（原2天过严），移除当天破MA20（mode3已要求close>=ma20）
                if _has_limit_up_then_down(rows, idx, item.code, item.name, lookback=5, min_consec_limit_up=3):
                    continue

            industry_nm = ""
            hot_set: Optional[Set[str]] = None
            hot_counts: Optional[Dict[str, int]] = None
            ndays_total: Optional[int] = None
            if (use_mode9 or use_mode90) and sector_dir:
                from .limit_up_industry_top import (
                    industry_limit_up_counts_for_date,
                    industry_limit_up_sum_ndays,
                    load_stock_industry_name,
                )

                industry_nm = load_stock_industry_name(sector_dir, item.code)
                sig_date = rows[idx].date
                if hot_bonus > 0:
                    if sig_date not in hot_cache:
                        try:
                            counts = industry_limit_up_counts_for_date(
                                sig_date,
                                kline_dir=cache_dir,
                                ak_base=sector_dir,
                                stock_list_csv=os.path.join(GPT_DATA_DIR, "stock_list.csv"),
                            )
                            if not counts:
                                hot_cache[sig_date] = (set(), {})
                            else:
                                ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
                                hot_cache[sig_date] = (
                                    {name for name, _ in ranked[:hot_top_n]},
                                    counts,
                                )
                        except Exception:
                            hot_cache[sig_date] = (set(), {})
                    hot_set, hot_counts = hot_cache[sig_date]
                if ndays_n > 0:
                    if sig_date not in ndays_cache:
                        try:
                            m, valid = industry_limit_up_sum_ndays(
                                sig_date,
                                ndays_n,
                                kline_dir=cache_dir,
                                ak_base=sector_dir,
                                stock_list_csv=os.path.join(GPT_DATA_DIR, "stock_list.csv"),
                            )
                            ndays_cache[sig_date] = (m, valid)
                        except Exception:
                            ndays_cache[sig_date] = ({}, False)
                    nd_map, nd_ok = ndays_cache[sig_date]
                    if nd_ok and (industry_nm or "").strip():
                        ndays_total = int(nd_map.get((industry_nm or "").strip(), 0))

            if use_mode90:
                score = score_fn(
                    rows,
                    idx,
                    ma10,
                    ma20,
                    ma60,
                    vol20,
                    item.code,
                    item.name,
                    None,
                    industry=industry_nm,
                    hot_industries=hot_set,
                    mode9_hot_industry_bonus=hot_bonus,
                    hot_industry_counts=hot_counts,
                    mode9_hot_industry_bonus_max=hot_bonus_max,
                    industry_ndays_limit_total=ndays_total,
                    mode9_industry_ndays_penalty=ndays_pen,
                    mode9_industry_ndays_bonus_per_unit=ndays_unit,
                    mode9_industry_ndays_bonus_cap=ndays_cap_cfg,
                )
            elif use_mode5:
                score = _score_mode5(
                    rows,
                    idx,
                    ma10,
                    ma20,
                    ma60,
                    vol20,
                    item.code,
                    item.name,
                    int(getattr(config, "mode5_shrink_max_days", 5) or 5),
                    int(getattr(config, "mode5_half_year_bars", 120) or 120),
                )
            elif (
                (use_mode9 and score_fn is _score_mode9)
                or (use_mode8 and score_fn is _score_mode8)
                or (use_mode10 and score_fn is _score_mode10)
                or (use_mode11 and score_fn is _score_mode11)
                or (use_mode12 and score_fn is _score_mode12)
                or (use_mode18 and score_fn is _score_mode18)
                or use_mode98
                or use_mode32
                or use_mode33
                or use_mode34
                or use_mode35
                or use_mode36
                or use_mode37
                or use_mode38
                or use_mode39
                or use_mode40
                or use_mode41
                or use_mode42
                or use_mode43
                or use_mode44
                or use_mode45
                or use_mode46
                or (use_mode88 and score_fn is _score_mode88)
        or (use_mode93 and score_fn is _score_mode93)
                or use_mode_bottom_big_yang
                or use_mode_platform_breakout_first_yang
                or use_mode_mid_big_yang
                or use_mode_bottom_support
                or use_mode_final_shakeout
            ):
                if use_mode9 and score_fn is _score_mode9:
                    score = score_fn(
                        rows,
                        idx,
                        ma10,
                        ma20,
                        ma60,
                        vol20,
                        item.code,
                        item.name,
                        None,
                        industry=industry_nm,
                        hot_industries=hot_set,
                        mode9_hot_industry_bonus=hot_bonus,
                        hot_industry_counts=hot_counts,
                        mode9_hot_industry_bonus_max=hot_bonus_max,
                        industry_ndays_limit_total=ndays_total,
                        mode9_industry_ndays_penalty=ndays_pen,
                        mode9_industry_ndays_bonus_per_unit=ndays_unit,
                        mode9_industry_ndays_bonus_cap=ndays_cap_cfg,
                    )
                elif use_mode33:
                    score = _score_mode33(
                        rows,
                        idx,
                        ma10,
                        ma20,
                        ma60,
                        vol20,
                        item.code,
                        item.name,
                        None,
                        **mode33_kw,
                    )
                elif use_mode34:
                    score = _score_mode34(
                        rows,
                        idx,
                        ma10,
                        ma20,
                        ma60,
                        vol20,
                        item.code,
                        item.name,
                        None,
                        **mode34_kw,
                    )
                elif use_mode35:
                    score = _score_mode35(
                        rows,
                        idx,
                        ma10,
                        ma20,
                        ma60,
                        vol20,
                        item.code,
                        item.name,
                        None,
                        **mode35_kw,
                    )
                elif use_mode38:
                    score = _score_mode38(
                        rows,
                        idx,
                        ma10,
                        ma20,
                        ma60,
                        vol20,
                        item.code,
                        item.name,
                        None,
                        **mode38_kw,
                    )
                elif use_mode39:
                    score = _score_mode39(
                        rows,
                        idx,
                        ma10,
                        ma20,
                        ma60,
                        vol20,
                        item.code,
                        item.name,
                        None,
                        **mode39_kw,
                    )
                elif use_mode40:
                    score = _score_mode40(
                        rows,
                        idx,
                        ma10,
                        ma20,
                        ma60,
                        vol20,
                        item.code,
                        item.name,
                        None,
                        **mode40_kw,
                    )
                elif use_mode41:
                    score = _score_mode41(
                        rows,
                        idx,
                        ma10,
                        ma20,
                        ma60,
                        vol20,
                        item.code,
                        item.name,
                        None,
                        **mode41_kw,
                    )
                elif use_mode42:
                    score = _score_mode42(
                        rows,
                        idx,
                        ma10,
                        ma20,
                        ma60,
                        vol20,
                        item.code,
                        item.name,
                        None,
                        **mode42_kw,
                    )
                elif use_mode43:
                    score = _score_mode43(
                        rows,
                        idx,
                        ma10,
                        ma20,
                        ma60,
                        vol20,
                        item.code,
                        item.name,
                        None,
                        **mode43_kw,
                    )
                elif use_mode44:
                    score = _score_mode44(
                        rows,
                        idx,
                        ma10,
                        ma20,
                        ma60,
                        vol20,
                        item.code,
                        item.name,
                        None,
                        **mode44_kw,
                    )
                elif use_mode45:
                    score = _score_mode45(
                        rows,
                        idx,
                        ma10,
                        ma20,
                        ma60,
                        vol20,
                        item.code,
                        item.name,
                        None,
                        **mode45_kw,
                    )
                elif use_mode46:
                    score = _score_mode46(
                        rows,
                        idx,
                        ma10,
                        ma20,
                        ma60,
                        vol20,
                        item.code,
                        item.name,
                        None,
                        **mode46_kw,
                    )
                elif use_mode37:
                    score = _score_mode37(
                        rows,
                        idx,
                        ma10,
                        ma20,
                        ma60,
                        vol20,
                        item.code,
                        item.name,
                        None,
                        **mode37_kw,
                    )
                elif use_mode36:
                    score = _score_mode36(
                        rows,
                        idx,
                        ma10,
                        ma20,
                        ma60,
                        vol20,
                        item.code,
                        item.name,
                        None,
                        **mode36_kw,
                    )
                else:
                    score = score_fn(
                        rows, idx, ma10, ma20, ma60, vol20, item.code, item.name
                    )
            else:
                score = score_fn(rows, idx, ma10, ma20, ma60, vol20)

            sector_sm: Dict[str, Any] = {}
            if sector_dir:
                sector_sm = metrics_for_signal(
                    item.code, rows[idx].date, sector_dir, sector_hist_mem
                )
                merge_ths_flow_features(
                    sector_sm, rows[idx].date, ths_features_data
                )
                # 板块热度：仅通过「信号日涨停行业 TopN + 家数」等（见 _score_mode9 与下方 reasons），
                # 不再按行业指数涨跌幅对总分加分（避免与「只统计涨停个数」策略重复）。

                # 同花顺行业净额：净流入加分、净流出减分（须 trade_date 与信号日对齐的 ths 特征）
                if (
                    sector_fund_flow_max_points > 0
                    and sector_fund_flow_yi_per_point > 0
                    and (
                        use_mode90
                        or (use_mode9 and score_fn is _score_mode9)
                    )
                ):
                    net_raw = sector_sm.get("ths_flow_net_1d")
                    if net_raw is None:
                        net_raw = sector_sm.get("ths_flow_net_5d")
                    net_yi = parse_ths_flow_net_yi(net_raw)
                    fd = sector_fund_flow_score_delta(
                        net_yi,
                        yi_per_point=sector_fund_flow_yi_per_point,
                        max_abs_points=sector_fund_flow_max_points,
                    )
                    if fd != 0 and net_yi is not None:
                        score = min(100, int(score) + fd)
                        sector_sm["sector_fund_flow_net_yi"] = net_yi
                        sector_sm["sector_fund_flow_score_delta"] = fd

                # 概念板块资金（东财 push2 快照 → 近5/10天滚动最好排名）
                if use_mode90 or (use_mode9 and score_fn is _score_mode9):
                    concepts = load_stock_concepts(item.code, sector_dir)
                    if concepts:
                        r5 = concept_flow_best_rank_rolling(
                            sector_dir, rows[idx].date, concepts, window_days=5
                        )
                        r10 = concept_flow_best_rank_rolling(
                            sector_dir, rows[idx].date, concepts, window_days=10
                        )
                        sector_sm["concepts"] = concepts[:12]
                        sector_sm["concept_flow_best_rank_5d"] = r5
                        sector_sm["concept_flow_best_rank_10d"] = r10
                        rb = r10 if r10 is not None else r5
                        cb = concept_rank_score_bonus(rb)
                        if cb:
                            score = min(100, int(score) + int(cb))
                            sector_sm["concept_flow_score_bonus"] = int(cb)

                # 东财当日行业资金TopN（需 scripts/fetch_board_flow_top10_em.py 预先落盘）
                if em_bonus and em_top_n > 0 and (use_mode90 or (use_mode9 and score_fn is _score_mode9)):
                    ind = sector_sm.get("industry")
                    rk_em = eastmoney_industry_flow_rank_today(
                        sector_dir, rows[idx].date, str(ind) if ind else None, top_n=em_top_n
                    )
                    if rk_em is not None:
                        sector_sm["em_industry_flow_rank"] = int(rk_em)
                        b = eastmoney_industry_flow_bonus(rk_em, bonus=em_bonus)
                        if b:
                            score = min(100, int(score) + int(b))
                            sector_sm["em_industry_flow_bonus"] = int(b)

            if score < config.min_score:
                continue

            signal_date = rows[idx].date
            mode34_prebuy: Optional[Dict[str, Any]] = None
            mode34_watch: Optional[Dict[str, Any]] = None
            if use_mode34:
                from app.mode34_bottom_break_pullback import (
                    match_mode34_prebuy_signal,
                    match_mode34_watchlist,
                )

                mode34_prebuy = match_mode34_prebuy_signal(
                    rows, idx, item.code, item.name, **mode34_kw
                )
                if not mode34_prebuy:
                    mode34_watch = match_mode34_watchlist(
                        rows, idx, item.code, item.name, **mode34_kw
                    )

            if use_mode34 and mode34_prebuy:
                buy_idx = idx
                buy_date = str(mode34_prebuy.get("exec_buy_date") or signal_date)[:10]
                buy_point_score = int(
                    mode34_prebuy.get("advice_score")
                    or mode34_prebuy.get("mode34_score")
                    or 0
                )
            elif use_mode34 and mode34_watch:
                buy_idx = idx
                buy_date = str(mode34_watch.get("exec_buy_date") or signal_date)[:10]
                buy_point_score = int(mode34_watch.get("watch_score") or 0)
            elif use_mode35:
                buy_idx = idx
                buy_date = str(signal_date)[:10]
                buy_point_score = min(100, int(score))
            else:
                buy_point_score = _buy_point_score(rows, idx, ma10, ma20, ma60, vol20)
                buy_idx = min(idx + 1, len(rows) - 1)
                buy_date = rows[buy_idx].date

            vol_ratio = volume[idx] / vol20[idx] if vol20[idx] > 0 else 0.0
            ma20_now = ma20[idx]
            ma60_now = ma60[idx]
            ma10_now = ma10[idx]
            ma20_gap = (ma10_now - ma20_now) / ma20_now if ma20_now > 0 else 0.0
            ma60_gap = (ma20_now - ma60_now) / ma60_now if ma60_now > 0 else 0.0
            close_gap = abs(close[idx] - ma20_now) / ma20_now if ma20_now > 0 else 0.0
            ret20_val = ret20[idx] if not np.isnan(ret20[idx]) else 0.0
            if idx >= 5 and close[idx - 5] > 0:
                ret5_val = (close[idx] - close[idx - 5]) / close[idx - 5] * 100.0
            else:
                ret5_val = 0.0
            o = rows[idx].open
            c = rows[idx].close
            h = rows[idx].high
            l = rows[idx].low
            rng = h - l
            upper = h - max(o, c)
            upper_ratio = upper / rng if rng > 0 else 0.0
            upper_score = upper_ratio * vol_ratio
            if require_upper_shadow:
                if upper_ratio < upper_ratio_min or vol_ratio < upper_vol_min:
                    continue
            if require_vol_ratio and vol_ratio < vol_ratio_min:
                continue
            if require_close_gap and close_gap > close_gap_max:
                continue
            reasons = [
                f"启动点 {mode_label}",
                f"信号日 {signal_date}",
                (
                    f"买入日 {buy_date} (T+1 开盘)"
                    if not (use_mode34 or use_mode35)
                    else f"买点日 {buy_date}"
                ),
                f"放量 {vol_ratio:.2f}x",
                f"MA10-20 {ma20_gap:.2%}",
                f"MA20-60 {ma60_gap:.2%}",
                f"距MA20 {close_gap:.2%}",
                f"20日涨幅 {ret20_val:.2f}%",
                f"5日涨幅 {ret5_val:.2f}%",
                f"上影占比 {upper_ratio:.2%}",
            ]
            if use_mode34 and mode34_prebuy:
                reasons[2] = (
                    f"买点日 {buy_date} 盘中突破昨高≥{mode34_prebuy.get('buy_trigger_above', '—')}"
                )
                reasons.append(f"预案 {mode34_prebuy.get('advice', '')}")
                if mode34_prebuy.get("watch_date"):
                    reasons.append(f"观察日 {mode34_prebuy['watch_date']}")
            elif use_mode34 and mode34_watch:
                reasons[1] = f"观察入池 {signal_date}"
                psd = mode34_watch.get("planned_signal_date", "")
                reasons[2] = (
                    f"预案信号日 {psd}，买点日 {buy_date} 盘中突破昨高试仓"
                    if psd
                    else f"买点日 {buy_date} 盘中突破昨高试仓"
                )
            elif use_mode35:
                reasons[2] = f"A类突破日 {buy_date} 放量破前高试仓"
            elif use_mode38:
                reasons.append(
                    f"大牛股回踩MA{int(_mode38_metrics(rows, idx, item.code, item.name, **mode38_kw).get('support_ma', 0) or 0)}关键位"
                )
            elif use_mode39:
                m39 = _mode39_metrics(rows, idx, item.code, item.name, **mode39_kw)
                style = m39.get("signal_style", "")
                label = "锚点回踩小阳" if style == "near_anchor" else "长下影探底"
                slope = m39.get("ma45_slope_pct")
                slope_s = f"{float(slope):.2f}%" if slope is not None else "—"
                reasons.append(
                    f"大阳锚点{label} MA45向上({slope_s}) 买点日{m39.get('exec_buy_date', buy_date)}开盘"
                )
            elif use_mode40:
                m40 = _mode40_metrics(rows, idx, item.code, item.name, **mode40_kw)
                reasons.append(
                    f"新高{m40.get('peak_date', '')}后回调{m40.get('pullback_days', '—')}日"
                    f"踩MA60({float(m40.get('ma_touch_dist_pct', 0) or 0):.1f}%)"
                    f"回升 买点日{m40.get('exec_buy_date', buy_date)}开盘"
                )
            elif use_mode41:
                m41 = _mode41_metrics(rows, idx, item.code, item.name, **mode41_kw)
                reasons.append(
                    f"周线回踩周MA{int(m41.get('support_ma', 0) or 0)}关键位"
                    f"量能近5周低(+{float(m41.get('vol_vs_min_pct', 0) or 0):.1f}%)"
                    f" 买点日{m41.get('exec_buy_date', buy_date)}开盘"
                )
            elif use_mode42:
                m42 = _mode42_metrics(rows, idx, item.code, item.name, **mode42_kw)
                reasons.append(
                    f"阴转阳回踩周MA{int(m42.get('support_ma', 0) or 0)}"
                    f"({m42.get('probe_week_date', '')}探底)"
                    f"量+{float(m42.get('vol_vs_min_pct', 0) or 0):.1f}%"
                    f"周涨{float(m42.get('week_chg_pct', 0) or 0):.1f}%"
                    f" 买点日{m42.get('exec_buy_date', buy_date)}开盘"
                )
            elif use_mode43:
                m43 = _mode43_metrics(rows, idx, item.code, item.name, **mode43_kw)
                reasons.append(
                    f"爆量洗盘周(量{float(m43.get('vol_vs_ma5', 0) or 0):.1f}x5周均)"
                    f"振幅{float(m43.get('amplitude_pct', 0) or 0):.0f}%"
                    f"前4周+{float(m43.get('prior_4w_gain_pct', 0) or 0):.0f}%"
                    f" 买点日{m43.get('exec_buy_date', buy_date)}开盘"
                )
            elif use_mode44:
                m44 = _mode44_metrics(rows, idx, item.code, item.name, **mode44_kw)
                reasons.append(
                    f"三连阴量增(量{float(m44.get('vol_ramp_total', 0) or 0):.2f}x)"
                    f"跌{float(m44.get('cum_drop_pct', 0) or 0):.1f}%"
                    f" 买点日{m44.get('exec_buy_date', buy_date)}开盘"
                )
            elif use_mode45:
                m45 = _mode45_metrics(rows, idx, item.code, item.name, **mode45_kw)
                reasons.append(
                    f"涨停新高后缓升(启动{m45.get('launch_date', '')}"
                    f"量{float(m45.get('launch_vol_ratio', 0) or 0):.1f}x"
                    f"回撤{float(m45.get('grind_close_pullback_pct', 0) or 0):.1f}%)"
                    f" 买点日{m45.get('exec_buy_date', buy_date)}开盘"
                )
            elif use_mode46:
                m46 = _mode46_metrics(rows, idx, item.code, item.name, **mode46_kw)
                reasons.append(
                    f"前高附近二次攻击(前高{m46.get('peak_date', '')}"
                    f"高{float(m46.get('prior_high', 0) or 0):.2f}"
                    f"距高{float(m46.get('high_dist_pct', 0) or 0):.1f}%"
                    f"回撤{float(m46.get('pullback_pct', 0) or 0):.1f}%)"
                    f" 突破试仓>{float(m46.get('buy_trigger_above', 0) or 0):.2f}"
                )
            elif use_mode37:
                reasons.append("回踩向上跳空缺口支撑区")
            elif use_mode36:
                reasons.append("一阳穿越多条均线（开盘在下、收盘在上）")
            if sector_sm.get("sub_industry"):
                reasons.append(f"细分行业 {sector_sm['sub_industry']}")
            if sector_sm.get("industry"):
                ir5v = sector_sm.get("industry_ret5")
                ir5s = f"{float(ir5v):.1f}%" if ir5v is not None else "—"
                reasons.append(
                    f"行业 {sector_sm['industry']} 板块指数5日 {ir5s}"
                )
                ir10, ir20 = sector_sm.get("industry_ret10"), sector_sm.get("industry_ret20")
                if ir10 is not None or ir20 is not None:
                    t10 = f"{float(ir10):.1f}%" if ir10 is not None else "—"
                    t20 = f"{float(ir20):.1f}%" if ir20 is not None else "—"
                    reasons.append(f"行业指数 10日{t10} 20日{t20}")
            rk = sector_sm.get("sector_flow_rank")
            if rk is not None:
                reasons.append(f"行业净流入排行 约第{rk}名（快照）")
            if sector_sm.get("em_industry_flow_rank") is not None:
                reasons.append(
                    f"东财行业资金Top{em_top_n} 命中第{int(sector_sm['em_industry_flow_rank'])}名 评分{int(sector_sm.get('em_industry_flow_bonus') or 0):+d}"
                )
            if sector_sm.get("concept_flow_best_rank_10d") is not None or sector_sm.get("concept_flow_best_rank_5d") is not None:
                r10 = sector_sm.get("concept_flow_best_rank_10d")
                r5 = sector_sm.get("concept_flow_best_rank_5d")
                if r10 is not None:
                    reasons.append(f"概念资金10日滚动最好排名 第{int(r10)}名")
                elif r5 is not None:
                    reasons.append(f"概念资金5日滚动最好排名 第{int(r5)}名")
            if sector_sm.get("concept_flow_score_bonus"):
                reasons.append(f"概念资金加分 {int(sector_sm['concept_flow_score_bonus']):+d}")
            if sector_sm.get("ths_flow_rank_5d") is not None:
                t5 = sector_sm["ths_flow_rank_5d"]
                reasons.append(f"同花顺行业资金5日榜 第{t5}名")
            if sector_sm.get("ths_flow_rank_1d") is not None:
                reasons.append(
                    f"同花顺行业资金即时榜 第{sector_sm['ths_flow_rank_1d']}名"
                )
            if sector_sm.get("ths_flow_momentum") is not None:
                reasons.append(
                    f"行业资金相对走强(20日名次-5日名次差) {sector_sm['ths_flow_momentum']}"
                )
            if sector_sm.get("sector_fund_flow_score_delta"):
                ny = sector_sm.get("sector_fund_flow_net_yi")
                fd = sector_sm.get("sector_fund_flow_score_delta")
                ny_s = f"{float(ny):+.2f}" if ny is not None else "—"
                reasons.append(
                    f"板块资金净额约{ny_s}亿 → 评分{int(fd):+d}（每{sector_fund_flow_yi_per_point:g}亿约1分，上限±{sector_fund_flow_max_points}）"
                )
            if (
                hot_bonus > 0
                and (use_mode9 or use_mode90)
                and industry_nm
                and hot_set
                and industry_nm.strip() in hot_set
            ):
                nlu_r = int((hot_counts or {}).get(industry_nm.strip(), 0))
                reasons.append(
                    f"信号日涨停行业Top{hot_top_n} 含「{industry_nm.strip()}」"
                    f"（当日该行业涨停{nlu_r}家，资金抱团加分）"
                )
            if (
                ndays_n > 0
                and (use_mode9 or use_mode90)
                and industry_nm
                and ndays_total is not None
            ):
                reasons.append(
                    f"近{ndays_n}个交易日本行业涨停累计{int(ndays_total)}家次"
                )

            m_extra: Dict[str, Any] = {
                "signal_date": signal_date,
                "buy_date": buy_date,
                "vol_ratio": float(vol_ratio),
                "ma20_gap": float(ma20_gap),
                "ma60_gap": float(ma60_gap),
                "close_gap": float(close_gap),
                "ret20": float(ret20_val),
                "ret5": float(ret5_val),
                "upper_ratio": float(upper_ratio),
                "upper_score": float(upper_score),
                "market_cap": float(cap_value) if cap_value is not None else None,
                "buy_point_score": int(buy_point_score),
                "limitup_shrink_vol": int(
                    _has_limit_up_then_shrink_volume(rows, idx, item.code, item.name, lookback=6, next_vol_max_mult=1.8)
                ),
                "has_limit_up_6d": int(
                    _has_limit_up_6d(rows, idx, item.code, item.name, lookback=6)
                ),
                "mode9_hot_industry_bonus_applied": int(
                    bool(
                        hot_bonus > 0
                        and (use_mode9 or use_mode90)
                        and industry_nm
                        and hot_set
                        and industry_nm.strip() in hot_set
                    )
                ),
                "hot_industry_limit_up_count": (
                    int((hot_counts or {}).get((industry_nm or "").strip(), 0))
                    if ((use_mode9 or use_mode90) and hot_counts)
                    else 0
                ),
                "industry_ndays_limit_up_total": (
                    ndays_total
                    if (use_mode9 or use_mode90) and ndays_n > 0
                    else None
                ),
            }
            if use_mode98:
                _nk = int(getattr(config, "mode98_kdj_n", 9) or 9)
                _m1k = int(getattr(config, "mode98_kdj_m1", 3) or 3)
                _m2k = int(getattr(config, "mode98_kdj_m2", 3) or 3)
                m_extra.update(_mode98_kdj_metrics(rows, idx, _nk, _m1k, _m2k))
            if use_mode32:
                m_extra.update(_mode32_metrics(rows, idx))
            if use_mode33:
                m_extra.update(
                    _mode33_metrics(rows, idx, item.code, item.name, **mode33_kw)
                )
            if use_mode34:
                m_extra.update(
                    _mode34_metrics(rows, idx, item.code, item.name, **mode34_kw)
                )
                m_extra["signal_date"] = str(signal_date)[:10]
                if mode34_prebuy:
                    m_extra["event_type"] = "信号"
                    m_extra["buy_mode"] = "intraday"
                elif mode34_watch:
                    m_extra["event_type"] = "观察"
                    m_extra["buy_mode"] = "watch"
                    m_extra["planned_signal_date"] = mode34_watch.get(
                        "planned_signal_date", ""
                    )
            if use_mode35:
                m_extra.update(
                    _mode35_metrics(rows, idx, item.code, item.name, **mode35_kw)
                )
                m_extra["signal_date"] = str(signal_date)[:10]
                m_extra["event_type"] = "突破"
                m_extra["buy_mode"] = "breakout_a"
            if use_mode38:
                m_extra.update(
                    _mode38_metrics(rows, idx, item.code, item.name, **mode38_kw)
                )
                m_extra["signal_date"] = str(signal_date)[:10]
                m_extra["event_type"] = "关键位回踩"
            if use_mode39:
                m_extra.update(
                    _mode39_metrics(rows, idx, item.code, item.name, **mode39_kw)
                )
                m_extra["signal_date"] = str(signal_date)[:10]
                st = m_extra.get("signal_style", "")
                m_extra["event_type"] = (
                    "锚点回踩" if st == "near_anchor" else "长下影探底"
                )
                m_extra["buy_mode"] = "next_open"
            if use_mode40:
                m_extra.update(
                    _mode40_metrics(rows, idx, item.code, item.name, **mode40_kw)
                )
                m_extra["signal_date"] = str(signal_date)[:10]
                m_extra["event_type"] = "新高回踩MA60"
                m_extra["buy_mode"] = "next_open"
            if use_mode41:
                m_extra.update(
                    _mode41_metrics(rows, idx, item.code, item.name, **mode41_kw)
                )
                m_extra["signal_date"] = str(signal_date)[:10]
                m_extra["event_type"] = "周线关键位回踩"
                m_extra["buy_mode"] = "next_open"
            if use_mode42:
                m_extra.update(
                    _mode42_metrics(rows, idx, item.code, item.name, **mode42_kw)
                )
                m_extra["signal_date"] = str(signal_date)[:10]
                m_extra["event_type"] = "阴转阳回踩"
                m_extra["buy_mode"] = "next_open"
            if use_mode43:
                m_extra.update(
                    _mode43_metrics(rows, idx, item.code, item.name, **mode43_kw)
                )
                m_extra["signal_date"] = str(signal_date)[:10]
                m_extra["event_type"] = "爆量洗盘周"
                m_extra["buy_mode"] = "next_open"
            if use_mode44:
                m_extra.update(
                    _mode44_metrics(rows, idx, item.code, item.name, **mode44_kw)
                )
                m_extra["signal_date"] = str(signal_date)[:10]
                m_extra["event_type"] = "三连阴量增背离"
                m_extra["buy_mode"] = "next_open"
            if use_mode45:
                m_extra.update(
                    _mode45_metrics(rows, idx, item.code, item.name, **mode45_kw)
                )
                m_extra["signal_date"] = str(signal_date)[:10]
                m_extra["event_type"] = "涨停新高后缓升"
                m_extra["buy_mode"] = "next_open"
            if use_mode46:
                m_extra.update(
                    _mode46_metrics(rows, idx, item.code, item.name, **mode46_kw)
                )
                m_extra["signal_date"] = str(signal_date)[:10]
                m_extra["event_type"] = "前高附近二次攻击"
                m_extra["buy_mode"] = "break_prior_high"
            if use_mode37:
                m_extra.update(
                    _mode37_metrics(rows, idx, item.code, item.name, **mode37_kw)
                )
                m_extra["signal_date"] = str(signal_date)[:10]
                st = m_extra.get("signal_type", "support")
                m_extra["event_type"] = (
                    "强反包" if st == "strong_bounce" else ("反包" if st == "bounce" else "回踩支撑")
                )
            if use_mode36:
                m_extra.update(
                    _mode36_metrics(rows, idx, item.code, item.name, **mode36_kw)
                )
                m_extra["signal_date"] = str(signal_date)[:10]
                m_extra["event_type"] = "一阳穿线"
            for k in (
                "industry",
                "sub_industry",
                "industry_ret5",
                "industry_ret10",
                "industry_ret20",
                "sector_flow_rank",
                "concepts",
                "concept_flow_best_rank_5d",
                "concept_flow_best_rank_10d",
                "concept_flow_score_bonus",
                "em_industry_flow_rank",
                "em_industry_flow_bonus",
                "ths_flow_rank_1d",
                "ths_flow_rank_5d",
                "ths_flow_rank_10d",
                "ths_flow_rank_20d",
                "ths_flow_momentum",
                "ths_flow_net_1d",
                "ths_flow_net_5d",
                "sector_fund_flow_net_yi",
                "sector_fund_flow_score_delta",
            ):
                if k in sector_sm and sector_sm[k] is not None:
                    m_extra[k] = sector_sm[k]

            results.append(
                ScanResult(
                    code=item.code,
                    name=item.name,
                    score=int(score),
                    latest_close=float(rows[-1].close),
                    change_pct=float(rows[-1].pct_chg),
                    reasons=reasons,
                    metrics=m_extra,
                )
            )

    def _mode3_sort_key(r: ScanResult):
        return mode3_sort_tuple(r, prefer_upper_shadow=prefer_upper_shadow)

    # 先按评分与买点/涨停特征排序
    results.sort(key=_mode3_sort_key)

    # 同一代码、同一信号日只保留排序最优的一条（避免区间扫描重复入表）
    deduped: List[ScanResult] = []
    best_by_key: Dict[tuple, ScanResult] = {}
    key_order: List[tuple] = []
    for r in results:
        sig = str((r.metrics or {}).get("signal_date") or "")[:10]
        key = (r.code, sig) if sig else (r.code,)
        if key not in best_by_key:
            key_order.append(key)
            best_by_key[key] = r
        elif _mode3_sort_key(r) < _mode3_sort_key(best_by_key[key]):
            best_by_key[key] = r
    results = [best_by_key[k] for k in key_order]

    # 计算每个代码在本次扫描区间内的最早信号日，用于前端展示「最早出现日期」
    first_dates: Dict[str, str] = {}
    for r in results:
        metrics = r.metrics or {}
        sig = str(metrics.get("signal_date") or "").strip()
        if not sig:
            continue
        code = r.code
        if code not in first_dates or sig < first_dates[code]:
            first_dates[code] = sig
    for r in results:
        if not first_dates:
            break
        metrics = r.metrics or {}
        code = r.code
        if code in first_dates:
            metrics["first_signal_date"] = first_dates[code]
            r.metrics = metrics

    # 一段时间选股：每日取分数最高的 max_results 只（与前端「输出数量」一致）
    if start_date:
        grouped: Dict[str, List[ScanResult]] = {}
        for r in results:
            sig = (r.metrics or {}).get("signal_date")
            if sig:
                grouped.setdefault(str(sig), []).append(r)
        out: List[ScanResult] = []
        for day in sorted(grouped.keys()):
            group = grouped[day]
            group.sort(key=_mode3_sort_key)
            out.extend(group[: config.max_results])
        return out[: config.max_results]

    return results[: config.max_results]


def _flatten_result_metrics(row: Dict[str, object]) -> Dict[str, object]:
    """JSON 结果顶层补齐 metrics 字段，兼容旧版 latest.json。"""
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    for k in (
        "signal_date",
        "first_signal_date",
        "buy_point_score",
        "has_limit_up_6d",
        "event_type",
        "anchor_date",
    ):
        if row.get(k) in (None, "") and metrics.get(k) not in (None, ""):
            row[k] = metrics[k]
    return row


def serialize_results(results: List[ScanResult]) -> List[Dict[str, object]]:
    rows = [
        {
            **asdict(r),
            "reasons": ", ".join(r.reasons),
            "signal_date": (r.metrics or {}).get("signal_date"),
            "buy_point_score": (r.metrics or {}).get("buy_point_score"),
            "first_signal_date": (r.metrics or {}).get("first_signal_date"),
            "has_limit_up_6d": (r.metrics or {}).get("has_limit_up_6d"),
            "event_type": (r.metrics or {}).get("event_type"),
            "anchor_date": (r.metrics or {}).get("anchor_date"),
        }
        for r in results
    ]
    return [_flatten_result_metrics(row) for row in rows]
