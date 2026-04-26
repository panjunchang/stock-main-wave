"""推理预测脚本。

用法
----
    python scripts/predict.py --model models/rf_model.pkl --input data/new_data.csv [--output predictions.csv]

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_csv
from src.features.engineering import add_technical_indicators
from src.models.classifier import MainWaveClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用已训练模型进行主升浪预测")
    parser.add_argument("--model", required=True, help="已保存的模型文件路径（.pkl）")
    parser.add_argument("--input", required=True, help="输入 CSV 数据文件路径")
    parser.add_argument(
        "--output", default=None, help="预测结果保存路径（可选，默认打印到控制台）"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[1/3] 加载模型: {args.model}")
    clf = MainWaveClassifier.load(args.model)

    print(f"[2/3] 加载并预处理数据: {args.input}")
    df = load_csv(args.input)
    df = add_technical_indicators(df)
    print(f"      有效数据 {len(df)} 行")

    print("[3/3] 运行预测...")
    df["pred_label"] = clf.predict(df)
    df["pred_proba"] = clf.predict_proba(df)

    result = df[["close", "pred_label", "pred_proba"]].copy()
    wave_count = int(result["pred_label"].sum())
    print(
        f"      预测主升浪: {wave_count} 行 ({wave_count / len(result) * 100:.1f}%)"
    )

    if args.output:
        result.to_csv(args.output)
        print(f"预测结果已保存至: {args.output}")
    else:
        print("\n===== 预测结果（最后 20 行）=====")
        print(result.tail(20).to_string())


if __name__ == "__main__":
    main()
