"""数据加载模块测试。"""

from __future__ import annotations

import io
import textwrap

import pandas as pd
import pytest

from src.data.loader import load_csv, validate_ohlcv


VALID_CSV = textwrap.dedent(
    """\
    date,open,high,low,close,volume
    2024-01-02,10.0,10.5,9.8,10.3,1000000
    2024-01-03,10.3,11.0,10.1,10.8,1200000
    2024-01-04,10.8,11.2,10.6,11.0,900000
    """
)


class TestValidateOHLCV:
    def _df(self, csv: str) -> pd.DataFrame:
        return pd.read_csv(io.StringIO(csv))

    def test_valid_data_passes(self):
        df = self._df(VALID_CSV)
        validate_ohlcv(df)  # should not raise

    def test_missing_column_raises(self):
        df = self._df(VALID_CSV).drop(columns=["volume"])
        with pytest.raises(ValueError, match="缺少必要列"):
            validate_ohlcv(df)

    def test_negative_close_raises(self):
        df = self._df(VALID_CSV)
        df.loc[0, "close"] = -1.0
        with pytest.raises(ValueError, match="负值"):
            validate_ohlcv(df)

    def test_high_less_than_low_raises(self):
        df = self._df(VALID_CSV)
        df.loc[0, "high"] = 9.0
        df.loc[0, "low"] = 9.5
        with pytest.raises(ValueError, match="high < low"):
            validate_ohlcv(df)

    def test_high_less_than_close_raises(self):
        df = self._df(VALID_CSV)
        df.loc[0, "high"] = df.loc[0, "close"] - 0.1
        with pytest.raises(ValueError, match="high < close"):
            validate_ohlcv(df)

    def test_low_greater_than_close_raises(self):
        df = self._df(VALID_CSV)
        df.loc[0, "low"] = df.loc[0, "close"] + 0.1
        with pytest.raises(ValueError, match="low > close"):
            validate_ohlcv(df)


class TestLoadCSV:
    def test_load_csv_returns_sorted_date_index(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(VALID_CSV)
        df = load_csv(str(csv_file))
        assert df.index.name == "date"
        assert list(df.index) == sorted(df.index)

    def test_load_csv_missing_file_raises(self, tmp_path):
        with pytest.raises(Exception):
            load_csv(str(tmp_path / "nonexistent.csv"))
