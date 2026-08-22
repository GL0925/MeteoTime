# MeteoTime

[English](README.md) | [中文](README_CN.md)

MeteoTime 是面向气象小时序列的自回归时序预测模型，执行单变量概率预测并输出分位数。

## 模型规格

- 约 32M 参数的 Decoder-only Transformer
- 最大输入 2048 小时，Patch 大小 32
- 单次前向传播预测未来 64 小时
- 输出 9 个分位数：`0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95`
- Pre-RMSNorm、RoPE、QK-Norm、SwiGLU、因果 SDPA
- BF16 混合精度、六卡 DDP 训练

## 项目结构

```
MeteoTime/
├── config_data.py         # 数据路径、长度、源权重
├── config_model.py        # 模型结构、分位数
├── config_train.py        # 优化器、批量大小、训练默认值
├── config_eval.py         # 评测协议设置
├── train_meteotime.py     # 训练入口
├── evaluate_all.py        # 统一评测（MeteoTime、TimesFM、Chronos2、基线）
├── run.sh                 # 启动六卡训练
├── tensorboard.sh         # 启动 TensorBoard 服务
├── scripts_data/          # 数据管道（预处理、加载、拼接）
│   ├── preprocess.py      # 构建数据源清单
│   ├── mixture_dataset.py # 训练时随机窗口采样
│   ├── sources/           # 数据源适配器（ERA5、BTS 等）
│   └── validate.py        # 验证清单和采样
├── models/
│   └── meteotime.py       # MeteoTime 模型定义
├── checkpoints/           # 模型检查点（best.pt）
└── runs/                  # TensorBoard 日志
```

## 数据集

### 训练数据

训练数据存储在 `<DATA_ROOT>/MeteoTime_train_data`。从 [lotsa_data](https://huggingface.co/datasets/Salesforce/lotsa_data)（公共时序数据仓库）下载：

```
MeteoTime_train_data/
├── era5_1989/
│   └── data-00000-of-00096.arrow  # HuggingFace Arrow 格式
├── era5_1990/
├── ...
└── bts_flights_weather/
    └── airport_hourly_weather.parquet
```

### 评测数据

评测数据存储在 `<DATA_ROOT>/MeteoTime_benchmark`。从 [lotsa_data](https://huggingface.co/datasets/Salesforce/lotsa_data) 下载：

```
MeteoTime_benchmark/
├── weather/
│   ├── test.parquet
│   └── scaler_params.json
├── DSN/
├── XMD/
├── XCS/
├── SSN/
└── wtbdata_245days.csv
```

海洋观测站（DSN/XMD/XCS/SSN）也可从 [NMDIS](https://mds.nmdis.org.cn/) 下载。

### 处理产物

处理后的数据存储在 `<DATA_ROOT>/MeteoTime_data_artifacts`。此目录在预处理期间自动生成：

```
MeteoTime_data_artifacts/
├── meta/
│   ├── era5/
│   │   ├── manifest.parquet
│   │   ├── files.json
│   │   └── report.json
│   ├── bts_airport_weather/
│   │   ├── manifest.parquet
│   │   ├── files.json
│   │   └── report.json
│   └── ...
└── processed/
    └── bts_airport_weather/
        └── data-00000-of-00001.arrow
```

注意：LOTSA 的 ERA5 数据缺失部分气压变量，无法完整覆盖 `pressure` 目标类别。
需要将补充的 ERA5 气压数据放在
`<DATA_ROOT>/MeteoTime_train_data/era5_pressure_1989_2018/`，并单独执行：

```bash
python -m scripts_data.preprocess --source era5_pressure
```

生成的 `era5_pressure` 数据源会自动归入 `era5` 统计组，并参与气压类别训练。
如果原始 NetCDF 已经转换为 Arrow 文件，预处理会直接复用现有 Arrow 文件。

### 路径配置

如需使用不同路径，请修改以下文件：

- **config_data.py**：`raw_root`（训练数据路径）、`artifact_root`（处理产物路径）
- **config_eval.py**：`weather_dir`、`marine_root`、`wind_benchmark_path`（评测数据路径）

将 `<DATA_ROOT>` 替换为您的实际数据目录（如 `/home/amax/SSD2/GL` 或 `/data`）。

### 预训练模型

下载对比模型到 `models/` 目录：

```
MeteoTime/
├── models/
│   ├── timesfm2.5/
│   │   └── model.safetensors  # Google TimesFM 2.5
│   └── chronos2/
│       └── model.safetensors  # Amazon Chronos 2
```

下载地址：
- TimesFM 2.5: [ModelScope](https://www.modelscope.cn/models/google/timesfm-2.5-200m-pytorch)
- Chronos 2: [ModelScope](https://www.modelscope.cn/models/amazon/chronos-2)

## 快速开始

### 1. 预处理数据

首次运行或数据变更后，构建所有数据源清单：

```bash
python -m scripts_data.preprocess --source all
```

使用补充气压数据时，如果首次预处理没有包含该数据源，还需要额外执行上面的
`--source era5_pressure` 命令。

### 2. 训练

启动六卡 DDP 训练：

```bash
bash run.sh
```

训练自动读取 `config_data.py`、`config_model.py`、`config_train.py`。默认配置：50 个 Epoch，每卡批量 256，余弦学习率加预热。

### 3. 监控

在另一个终端启动 TensorBoard：

```bash
bash tensorboard.sh
```

通过 VS Code 端口转发访问（默认端口 6006）。

### 4. 评测

对所有基准数据集执行统一评测：

```bash
python evaluate_all.py
```

评测 MeteoTime、TimesFM 2.5、Chronos 2、持续性基线和 24 小时季节性朴素基线。输出 MAE、RMSE、MASE 和 P05-P95 覆盖率到 `results.txt`。

## 配置说明

所有参数通过 `config_*.py` 文件配置（无命令行参数）：

- **config_data.py**：上下文长度（128/256/512/1024）、预测长度（64）、源权重（ERA5: 70%，BTS: 30%）
- **config_model.py**：隐藏维度、注意力头数、Patch 大小、分位数层级
- **config_train.py**：学习率、批量大小、Epoch 数、验证间隔
- **config_eval.py**：上下文长度（512）、预测窗口（48 小时）、批量大小

## 评测协议

- 固定预测起点，预测未来 48 小时
- 上下文长度：128、256、512、1024 小时
- 数据集：Jena 气象、4 个海洋观测站（DSN/XMD/XCS/SSN）、真实风电场风机
- 指标：MAE、RMSE、MASE（相对 24 小时季节性朴素基线）、P05-P95 覆盖率
