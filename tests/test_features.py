"""特征工程模块测试。"""

from __future__ import annotations

import pytest

from src.features.engineering import FEATURE_COLUMNS, add_technical_indicators
from tests.conftest import make_ohlcv


class TestAddTechnicalIndicators:
    def _base_df(self, n: int = 200, trend: float = 0.002):
        df = make_ohlcv(n=n, trend=trend)
        df.set_index("date", inplace=True)
        return df

    def test_output_contains_all_feature_columns(self):
        df = add_technical_indicators(self._base_df())
        for col in FEATURE_COLUMNS:
            assert col in df.columns, f"缺少特征列: {col}"

    def test_no_nan_in_feature_columns_after_processing(self):
        df = add_technical_indicators(self._base_df())
        assert not df[FEATURE_COLUMNS].isnull().any().any(), "特征列中存在 NaN"

    def test_output_rows_less_than_input_rows(self):
        raw = self._base_df(n=200)
        processed = add_technical_indicators(raw)
        assert len(processed) < len(raw), "处理后应删除含 NaN 的前几行"

    def test_does_not_modify_original_dataframe(self):
        raw = self._base_df()
        original_cols = set(raw.columns)
        add_technical_indicators(raw)
        assert set(raw.columns) == original_cols, "不应修改原始 DataFrame"

    def test_ma_columns_are_rolling_means(self):
        df = self._base_df(n=200)
        processed = add_technical_indicators(df)
        # MA5 应约等于过去 5 日 close 均值（由于 dropna 偏移，取一行验证）
        idx = processed.index[10]
        close_slice = df.loc[:idx, "close"].tail(5)
        assert abs(processed.loc[idx, "ma5"] - close_slice.mean()) < 1e-6

    def test_vol_ratio_positive(self):
        df = add_technical_indicators(self._base_df())
        assert (df["vol_ratio"] > 0).all(), "量比应全为正值"

    def test_boll_upper_greater_than_lower(self):
        df = add_technical_indicators(self._base_df())
        assert (df["boll_upper"] >= df["boll_lower"]).all()

    def test_rsi_bounded(self):
        df = add_technical_indicators(self._base_df())
        for col in ["rsi6", "rsi12", "rsi24"]:
            assert df[col].between(0, 100).all(), f"{col} 超出 [0, 100] 范围"
