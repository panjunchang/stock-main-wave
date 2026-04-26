"""数据加载与校验模块。"""

from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = {"date", "open", "high", "low", "close", "volume"}


def load_csv(path: str) -> pd.DataFrame:
    """从 CSV 文件加载 OHLCV 数据。

    参数
    ----
    path : str
        CSV 文件路径，必须包含 date、open、high、low、close、volume 列。

    返回
    ----
    pd.DataFrame
        以 date 为索引、按日期升序排列的 DataFrame。
    """
    df = pd.read_csv(path, parse_dates=["date"])
    validate_ohlcv(df)
    df = df.sort_values("date").reset_index(drop=True)
    df.set_index("date", inplace=True)
    return df


def validate_ohlcv(df: pd.DataFrame) -> None:
    """校验 DataFrame 是否包含所有必要列，并检查数值有效性。

    参数
    ----
    df : pd.DataFrame
        待校验的数据帧。

    异常
    ----
    ValueError
        当缺少必要列或数据包含无效值（负值、OHLC 逻辑错误）时抛出。
    """
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"缺少必要列: {missing}")

    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if (df[col] < 0).any():
            raise ValueError(f"列 '{col}' 包含负值")

    if (df["high"] < df["low"]).any():
        raise ValueError("存在 high < low 的行")

    if (df["high"] < df["close"]).any():
        raise ValueError("存在 high < close 的行")

    if (df["low"] > df["close"]).any():
        raise ValueError("存在 low > close 的行")
