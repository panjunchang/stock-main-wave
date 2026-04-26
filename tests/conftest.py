"""辅助函数：生成用于测试的合成 OHLCV 数据。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_ohlcv(
    n: int = 200,
    start: str = "2020-01-01",
    trend: float = 0.002,
    seed: int = 42,
) -> pd.DataFrame:
    """生成带趋势的合成 OHLCV DataFrame（不含日期索引）。

    参数
    ----
    n : int
        行数。
    start : str
        起始日期。
    trend : float
        每日价格趋势漂移（正值=上涨趋势）。
    seed : int
        随机种子。
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=n, freq="B")

    price = 10.0
    closes = []
    for _ in range(n):
        price *= 1 + trend + rng.normal(0, 0.01)
        price = max(price, 0.01)
        closes.append(price)

    closes = np.array(closes)
    highs = closes * (1 + rng.uniform(0, 0.02, n))
    lows = closes * (1 - rng.uniform(0, 0.02, n))
    opens = lows + rng.uniform(0, 1, n) * (highs - lows)
    volumes = rng.uniform(1e6, 5e6, n)

    return pd.DataFrame(
        {
            "date": dates,
            "open": np.round(opens, 2),
            "high": np.round(highs, 2),
            "low": np.round(lows, 2),
            "close": np.round(closes, 2),
            "volume": np.round(volumes, 0),
        }
    )
