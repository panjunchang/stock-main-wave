"""模型训练脚本。

用法
----
    python scripts/train.py --input data/sample.csv --output models/rf_model.pkl

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 使项目根目录可被 import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_csv
from src.features.engineering import add_technical_indicators
from src.labeling.wave_labeler import label_main_wave
from src.models.classifier import MainWaveClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练主升浪识别模型")
    parser.add_argument("--input", required=True, help="输入 CSV 数据文件路径")
    parser.add_argument("--output", default="models/rf_model.pkl", help="模型保存路径")
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="验证集比例（默认 0.2）"
    )
    parser.add_argument(
        "--min-ret-20d", type=float, default=0.15, help="20 日最低涨幅阈值（默认 0.15）"
    )
    parser.add_argument(
        "--min-vol-ratio", type=float, default=1.5, help="最低量比阈值（默认 1.5）"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[1/4] 加载数据: {args.input}")
    df = load_csv(args.input)
    print(f"      共 {len(df)} 行数据，时间范围: {df.index.min()} ~ {df.index.max()}")

    print("[2/4] 计算技术指标特征...")
    df = add_technical_indicators(df)
    print(f"      特征计算完成，有效数据 {len(df)} 行")

    print("[3/4] 标注主升浪区间...")
    labels = label_main_wave(
        df,
        min_ret_20d=args.min_ret_20d,
        min_vol_ratio=args.min_vol_ratio,
    )
    pos_count = int(labels.sum())
    print(
        f"      主升浪样本: {pos_count} 行 ({pos_count / len(labels) * 100:.1f}%)，"
        f"非主升浪: {len(labels) - pos_count} 行"
    )

    if pos_count == 0:
        print("警告：未找到任何主升浪样本，请调整标注参数后重试。")
        sys.exit(1)

    print("[4/4] 训练随机森林分类器...")
    clf = MainWaveClassifier()
    metrics = clf.fit(df, labels, test_size=args.test_size)

    print("\n===== 验证集评估结果 =====")
    for k, v in metrics.items():
        print(f"  {k:>12}: {v:.4f}")

    clf.save(args.output)
    print(f"\n模型已保存至: {args.output}")

    print("\n===== 特征重要性（Top 10）=====")
    fi = clf.feature_importances().head(10)
    for feat, imp in fi.items():
        print(f"  {feat:<20}: {imp:.4f}")


if __name__ == "__main__":
    main()
