# 🌤️ MeteoTime

> **Lightweight autoregressive model for zero-shot meteorological time-series forecasting**

[English](README.md) | [中文](README_CN.md)

## ✨ Highlights

| | |
|---|---|
| 🧩 **Lightweight** | Only **~33.2M** parameters — orders of magnitude smaller than time-series forecasting foundation models |
| 🔮 **Long-Range Context** | Supports up to **1024 time steps** of context for zero-shot generalization |
| 📊 **Top-Tier Performance** | Dominates across **3 benchmarks**, **8 tasks**, **32 comparison experiments** |

### 🏆 Benchmark Results

**Input Length = 512 (3 benchmarks × 8 tasks)**

| Rank | Count | Share |
|------|-------|-------|
| 🥇 Best | **5 / 8** | 62.5% |
| 🥈 Runner-up | **2 / 8** | 25% |

**Full Sweep: 128 / 256 / 512 / 1024 × 8 tasks = 32 comparisons**

| Rank | Count | Share |
|------|-------|-------|
| 🥇 Best | **18 / 32** | 56.3% |
| 🥈 Runner-up | **6 / 32** | 18.8% |

> MeteoTime achieves competitive or superior results to Google TimesFM 2.5 and Amazon Chronos 2 while being **6× smaller** than TimesFM-200M.

## Architecture

- ~33.2M parameter Decoder-only Transformer
- Max input: 1024 hours, Patch size: 32
- Single forward pass predicts 64 hours ahead
- 9 quantile outputs: `0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95`
- Pre-RMSNorm, RoPE, QK-Norm, SwiGLU, causal SDPA
- BF16 mixed precision, 6-card DDP training

## Project Structure

```
MeteoTime/
├── config_data.py         # Data paths, lengths, source weights
├── config_model.py        # Model architecture, quantiles
├── config_train.py        # Optimizer, batch size, training defaults
├── config_eval.py         # Evaluation protocol settings
├── train_meteotime.py     # Training entry point
├── evaluate_all.py        # Unified evaluation (MeteoTime, TimesFM, Chronos2, baselines)
├── run.sh                 # Launch 6-card training
├── tensorboard.sh         # Start TensorBoard server
├── scripts_data/          # Data pipeline (preprocessing, loading, collation)
│   ├── preprocess.py      # Build source manifests
│   ├── mixture_dataset.py # Random window sampling for training
│   ├── sources/           # Data source adapters (ERA5, BTS, etc.)
│   └── validate.py        # Validate manifests and sampling
├── models/
│   └── meteotime.py       # MeteoTime model definition
├── checkpoints/           # Model checkpoints (best.pt)
└──  runs/                  # TensorBoard logs
```

## Datasets

### Training Data

Training data is stored at `<DATA_ROOT>/MeteoTime_train_data`. Download from [lotsa_data](https://huggingface.co/datasets/Salesforce/lotsa_data) (public time-series repository):

```
MeteoTime_train_data/
├── era5_1989/
│   └── data-00000-of-00096.arrow  # HuggingFace Arrow format
├── era5_1990/
├── ...
└── bts_flights_weather/
    └── airport_hourly_weather.parquet
```

### Benchmark Data

Benchmark data is stored at `<DATA_ROOT>/MeteoTime_benchmark`. Download from [lotsa_data](https://huggingface.co/datasets/Salesforce/lotsa_data):

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

Marine stations (DSN/XMD/XCS/SSN) can also be downloaded from [NMDIS](https://mds.nmdis.org.cn/).

### Processed Artifacts

Processed data is stored at `<DATA_ROOT>/MeteoTime_data_artifacts`. This directory is auto-generated during preprocessing:

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

LOTSA ERA5 does not contain the complete pressure variables needed by the
`pressure` target category. Prepare the supplementary ERA5 pressure data in
`<DATA_ROOT>/MeteoTime_train_data/era5_pressure_1989_2018/` and build its
manifest explicitly:

```bash
python -m scripts_data.preprocess --source era5_pressure
```

The generated `era5_pressure` source is automatically aliased to the `era5`
source group and contributes to pressure training. If the original NetCDF
files have already been converted to Arrow files, preprocessing reuses those
Arrow files directly.

### Path Configuration

To use different paths, modify these files:

- **config_data.py**: `raw_root` (training data path), `artifact_root` (processed artifacts path)
- **config_eval.py**: `weather_dir`, `marine_root`, `wind_benchmark_path` (benchmark paths)

Replace `<DATA_ROOT>` with your actual data directory (e.g., `/home/amax/SSD2/GL` or `/data`).

### Pre-trained Models

Download comparison models to `models/` directory:

```
MeteoTime/
├── models/
│   ├── timesfm2.5/
│   │   └── model.safetensors  # Google TimesFM 2.5
│   └── chronos2/
│       └── model.safetensors  # Amazon Chronos 2
```

Download from:
- TimesFM 2.5: [ModelScope](https://www.modelscope.cn/models/google/timesfm-2.5-200m-pytorch)
- Chronos 2: [ModelScope](https://www.modelscope.cn/models/amazon/chronos-2)

## Quick Start

### 1. Preprocess Data

Build manifests for all sources (first run or after data changes):

```bash
python -m scripts_data.preprocess --source all
```

When using the supplementary pressure data, also run the explicit
`--source era5_pressure` command above if it was not included in the initial
preprocessing run.

### 2. Train

Launch 6-card DDP training:

```bash
bash run.sh
```

Training reads `config_data.py`, `config_model.py`, `config_train.py` automatically. Default: 50 epochs, batch size 256 per card, cosine learning rate with warmup.

### 3. Monitor

Start TensorBoard in a separate terminal:

```bash
bash tensorboard.sh
```

Access via VS Code port forwarding (default port 6006).

### 4. Evaluate

Run unified evaluation against all benchmarks:

```bash
python evaluate_all.py
```

Evaluates MeteoTime, TimesFM 2.5, Chronos 2, persistence baseline, and 24h seasonal naive. Outputs MAE, RMSE, MASE, and P05-P95 coverage to `results.txt`.

## Configuration

All parameters are configured via `config_*.py` files (no CLI arguments):

- **config_data.py**: Context lengths (128/256/512/1024), prediction length (64), source weights (ERA5: 70%, BTS: 30%)
- **config_model.py**: Hidden dim, attention heads, patch size, quantile levels
- **config_train.py**: Learning rate, batch size, epochs, validation interval
- **config_eval.py**: Context length (512), prediction horizon (48h), batch sizes

## Evaluation Protocol

- Fixed prediction origin, predict 48 hours ahead
- Context lengths: 128, 256, 512, 1024 hours
- Datasets: Jena weather, 4 marine stations (DSN/XMD/XCS/SSN), real wind farm turbines
- Metrics: MAE, RMSE, MASE (vs 24h seasonal naive), P05-P95 coverage
