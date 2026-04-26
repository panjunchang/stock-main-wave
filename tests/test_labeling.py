"""标注模块测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import add_technical_indicators
from src.labeling.wave_labeler import label_main_wave
from tests.conftest import make_ohlcv


class TestLabelMainWave:
    def _processed_df(self, n: int = 300, trend: float = 0.003):
        df = make_ohlcv(n=n, trend=trend)
        df.set_index("date", inplace=True)
        return add_technical_indicators(df)

    def test_returns_series_with_binary_values(self):
        df = self._processed_df()
        labels = label_main_wave(df)
        assert isinstance(labels, pd.Series)
        assert set(labels.unique()).issubset({0, 1})

    def test_labels_aligned_with_df_index(self):
        df = self._processed_df()
        labels = label_main_wave(df)
        assert list(labels.index) == list(df.index)

    def test_label_name(self):
        df = self._processed_df()
        labels = label_main_wave(df)
        assert labels.name == "label"

    def test_missing_column_raises(self):
        df = self._processed_df().drop(columns=["macd_hist"])
        with pytest.raises(ValueError, match="缺少必要特征列"):
            label_main_wave(df)

    def test_strong_uptrend_produces_positive_labels(self):
        # 极强上涨趋势（每日 +2%）应产生部分主升浪标签
        df = self._processed_df(n=400, trend=0.02)
        labels = label_main_wave(df, min_ret_20d=0.05, min_vol_ratio=0.5)
        assert labels.sum() > 0, "强上涨趋势应产生主升浪标签"

    def test_flat_market_produces_no_labels(self):
        # 横盘（无涨幅）不应产生主升浪标签，20 日涨幅接近 0 不满足 15% 阈值
        df = self._processed_df(n=300, trend=0.0)
        labels = label_main_wave(df, min_ret_20d=0.15, min_vol_ratio=1.5)
        assert labels.mean() < 0.05

    def test_custom_thresholds(self):
        df = self._processed_df(n=400, trend=0.01)
        labels_strict = label_main_wave(df, min_ret_20d=0.30, min_vol_ratio=3.0)
        labels_loose = label_main_wave(df, min_ret_20d=0.01, min_vol_ratio=0.1)
        assert labels_strict.sum() <= labels_loose.sum()
