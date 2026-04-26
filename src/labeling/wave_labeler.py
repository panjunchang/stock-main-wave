"""主升浪区间标注模块。

标注规则（同时满足以下条件，则标注为主升浪，label=1）：
  1. 均线多头排列：MA5 > MA10 > MA20 > MA60
  2. 收盘价在 MA60 之上
  3. MACD HIST > 0（金叉区域）
  4. 20 日涨幅 >= 15%
  5. 量比 >= 1.5（成交量放大）
"""

from __future__ import annotations

import pandas as pd


def label_main_wave(
    df: pd.DataFrame,
    ma_full_bull: bool = True,
    macd_positive: bool = True,
    min_ret_20d: float = 0.15,
    min_vol_ratio: float = 1.5,
) -> pd.Series:
    """根据规则标注主升浪区间。

    参数
    ----
    df : pd.DataFrame
        已包含技术指标特征的 DataFrame。
    ma_full_bull : bool
        是否要求均线完全多头排列（MA5>MA10>MA20>MA60 且收盘>MA60）。
    macd_positive : bool
        是否要求 MACD HIST > 0。
    min_ret_20d : float
        20 日最低涨幅阈值（默认 0.15，即 15%）。
    min_vol_ratio : float
        最低量比阈值（默认 1.5）。

    返回
    ----
    pd.Series
        与 df 对齐的标注序列，1=主升浪，0=非主升浪。
    """
    required = {"ma5", "ma10", "ma20", "ma60", "macd_hist", "ret_20d", "vol_ratio", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame 缺少必要特征列: {missing}")

    cond = pd.Series(True, index=df.index)

    if ma_full_bull:
        cond &= (
            (df["ma5"] > df["ma10"])
            & (df["ma10"] > df["ma20"])
            & (df["ma20"] > df["ma60"])
            & (df["close"] > df["ma60"])
        )

    if macd_positive:
        cond &= df["macd_hist"] > 0

    cond &= df["ret_20d"] >= min_ret_20d
    cond &= df["vol_ratio"] >= min_vol_ratio

    return cond.astype(int).rename("label")
