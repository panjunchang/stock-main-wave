"""模型训练与推理测试。"""

from __future__ import annotations

import os
import pickle

import numpy as np
import pytest

from src.features.engineering import add_technical_indicators
from src.labeling.wave_labeler import label_main_wave
from src.models.classifier import MainWaveClassifier
from src.utils.metrics import evaluate_classification
from tests.conftest import make_ohlcv


def _build_dataset(n: int = 400, trend: float = 0.01, seed: int = 0):
    df = make_ohlcv(n=n, trend=trend, seed=seed)
    df.set_index("date", inplace=True)
    df = add_technical_indicators(df)
    labels = label_main_wave(df, min_ret_20d=0.05, min_vol_ratio=0.5)
    return df, labels


class TestMainWaveClassifier:
    def test_fit_returns_metrics_dict(self):
        df, labels = _build_dataset()
        clf = MainWaveClassifier()
        metrics = clf.fit(df, labels)
        assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1"}
        for v in metrics.values():
            assert 0.0 <= v <= 1.0

    def test_predict_returns_binary_array(self):
        df, labels = _build_dataset()
        clf = MainWaveClassifier()
        clf.fit(df, labels)
        preds = clf.predict(df)
        assert set(np.unique(preds)).issubset({0, 1})
        assert len(preds) == len(df)

    def test_predict_proba_returns_valid_probabilities(self):
        df, labels = _build_dataset()
        clf = MainWaveClassifier()
        clf.fit(df, labels)
        proba = clf.predict_proba(df)
        assert proba.shape == (len(df),)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_predict_before_fit_raises(self):
        clf = MainWaveClassifier()
        df, _ = _build_dataset()
        with pytest.raises(RuntimeError, match="尚未训练"):
            clf.predict(df)

    def test_feature_importances_sums_to_one(self):
        df, labels = _build_dataset()
        clf = MainWaveClassifier()
        clf.fit(df, labels)
        fi = clf.feature_importances()
        assert abs(fi.sum() - 1.0) < 1e-6

    def test_save_and_load_roundtrip(self, tmp_path):
        df, labels = _build_dataset()
        clf = MainWaveClassifier()
        clf.fit(df, labels)
        model_path = str(tmp_path / "model.pkl")
        clf.save(model_path)
        assert os.path.exists(model_path)

        clf2 = MainWaveClassifier.load(model_path)
        preds1 = clf.predict(df)
        preds2 = clf2.predict(df)
        np.testing.assert_array_equal(preds1, preds2)

    def test_save_creates_parent_dirs(self, tmp_path):
        df, labels = _build_dataset()
        clf = MainWaveClassifier()
        clf.fit(df, labels)
        model_path = str(tmp_path / "nested" / "dir" / "model.pkl")
        clf.save(model_path)
        assert os.path.exists(model_path)


class TestEvaluateClassification:
    def test_perfect_prediction(self):
        y = np.array([0, 1, 0, 1, 1])
        metrics = evaluate_classification(y, y)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1"] == 1.0

    def test_all_wrong_prediction(self):
        y_true = np.array([1, 1, 1])
        y_pred = np.array([0, 0, 0])
        metrics = evaluate_classification(y_true, y_pred)
        assert metrics["recall"] == 0.0

    def test_metric_values_are_bounded(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, size=100)
        y_pred = rng.integers(0, 2, size=100)
        metrics = evaluate_classification(y_true, y_pred)
        for v in metrics.values():
            assert 0.0 <= v <= 1.0
