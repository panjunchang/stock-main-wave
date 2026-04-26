"""技术指标特征工程模块。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """为 OHLCV DataFrame 添加技术指标特征列。

    输入 DataFrame 必须包含 open、high、low、close、volume 列。
    本函数不修改原始数据，而是返回一个新的 DataFrame。

    参数
    ----
    df : pd.DataFrame
        OHLCV 数据。

    返回
    ----
    pd.DataFrame
        添加了技术指标的新 DataFrame（已删除含 NaN 的前若干行）。
    """
    df = df.copy()

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # --- 移动均线 ---
    for window in [5, 10, 20, 60]:
        df[f"ma{window}"] = close.rolling(window).mean()

    # --- 均线多头排列得分（0-4，每满足一项 +1）---
    df["ma_bull_score"] = (
        (df["ma5"] > df["ma10"]).astype(int)
        + (df["ma10"] > df["ma20"]).astype(int)
        + (df["ma20"] > df["ma60"]).astype(int)
        + (close > df["ma60"]).astype(int)
    )

    # --- MACD ---
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd_dif"] = ema12 - ema26
    df["macd_dea"] = df["macd_dif"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = 2 * (df["macd_dif"] - df["macd_dea"])

    # --- RSI ---
    for period in [6, 12, 24]:
        df[f"rsi{period}"] = _rsi(close, period)

    # --- 布林带 ---
    df["boll_mid"] = close.rolling(20).mean()
    # 布林带使用总体标准差（ddof=0），与主流交易软件（如通达信、东方财富）的计算方式一致
    boll_std = close.rolling(20).std(ddof=0)
    df["boll_upper"] = df["boll_mid"] + 2 * boll_std
    df["boll_lower"] = df["boll_mid"] - 2 * boll_std
    boll_width = df["boll_upper"] - df["boll_lower"]
    df["boll_width"] = boll_width
    df["boll_pct"] = (close - df["boll_lower"]) / boll_width.replace(0, np.nan)

    # --- 量比（当日量 / 过去 5 日均量）---
    df["vol_ratio"] = volume / volume.rolling(5).mean()

    # --- 涨跌幅 ---
    df["ret_1d"] = close.pct_change(1)
    df["ret_5d"] = close.pct_change(5)
    df["ret_20d"] = close.pct_change(20)

    # --- 振幅 ---
    df["amplitude"] = (high - low) / close.shift(1)

    # --- 删除由滚动窗口产生的 NaN 行（最多 60 行）---
    df.dropna(inplace=True)

    return df


def _rsi(close: pd.Series, period: int) -> pd.Series:
    """计算相对强弱指数（RSI）。"""
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


# 特征列（供模型训练使用）
FEATURE_COLUMNS = [
    "ma5", "ma10", "ma20", "ma60",
    "ma_bull_score",
    "macd_dif", "macd_dea", "macd_hist",
    "rsi6", "rsi12", "rsi24",
    "boll_mid", "boll_upper", "boll_lower", "boll_width", "boll_pct",
    "vol_ratio",
    "ret_1d", "ret_5d", "ret_20d",
    "amplitude",
]
