# stock-main-wave-ml — 主升浪 ML 识别模型

一个基于机器学习的 A 股"主升浪"自动识别系统。通过提取技术指标特征，训练分类模型，帮助判断股票当前是否处于主升浪阶段。

## 项目简介

**主升浪**（Main Upward Wave）是艾略特波浪理论中上涨行情的核心阶段，具有：

- 成交量显著放大
- 均线多头排列（MA5 > MA10 > MA20 > MA60）
- MACD 金叉并持续扩张
- 价格相对高位突破

本项目将这些特征量化，训练随机森林分类器识别主升浪区间。

## 项目结构

```
├── src/
│   ├── data/           # 数据加载与预处理
│   ├── features/       # 技术指标特征工程
│   ├── labeling/       # 主升浪区间标注
│   ├── models/         # 机器学习模型
│   └── utils/          # 评估指标工具
├── tests/              # 单元测试
├── scripts/
│   ├── train.py        # 模型训练脚本
│   └── predict.py      # 推理预测脚本
├── data/               # 示例数据目录
└── requirements.txt
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 训练模型

```bash
python scripts/train.py --input data/sample.csv --output models/rf_model.pkl
```

### 运行推理

```bash
python scripts/predict.py --model models/rf_model.pkl --input data/new_data.csv
```

### 运行测试

```bash
pytest tests/ -v
```

## 数据格式

输入 CSV 文件需包含以下列：

| 列名   | 说明         |
|--------|--------------|
| date   | 日期         |
| open   | 开盘价       |
| high   | 最高价       |
| low    | 最低价       |
| close  | 收盘价       |
| volume | 成交量       |

## 特征工程

- 移动均线：MA5、MA10、MA20、MA60
- MACD（DIF、DEA、HIST）
- RSI（6、12、24）
- 布林带（上轨、中轨、下轨）
- 量比、涨跌幅、振幅
- 均线多头排列得分

## 标注规则

满足以下全部条件，则标注为主升浪（label=1）：

1. MA5 > MA10 > MA20 > MA60（均线多头排列）
2. 收盘价在 MA60 之上
3. MACD HIST > 0（金叉区域）
4. 20 日涨幅 ≥ 15%
5. 量比 ≥ 1.5（成交量放大）

## 许可证

MIT
