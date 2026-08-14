# STPVE: Sales Trend Prediction in Livestream E-Commerce

This repository provides the implementation of **STPVE**, a multi-task deep learning framework for sales trend prediction in livestream e-commerce.

The repository also contains baseline implementations, including RNN, GRU, LSTM, MLP, Transformer, BERT+Transformer, Linear Regression, and LightGBM. Baselines are run through `run_baselines.py` and are **not** executed by the demo script.

---

## Project Structure

```text
.
├── README.md
├── requirements.txt
├── run_stpve.py                     # Full cross-validation for STPVE
├── run_demo.py                      # STPVE only, Fold 1 only
├── run_baselines.py                 # Full/selected baseline experiments
├── data/                            # Small synthetic dataset for demo execution
└── STPVE/
    ├── __init__.py
    ├── config.py                    # Configuration dataclasses and feature definitions
    ├── configs/
    │   ├── default.yaml             # Configuration for STPVE
    │   └── baseline.yaml            # Configuration for baseline models
    ├── data/
    │   ├── dataset.py
    │   ├── preprocessing.py
    │   ├── multi_feature_map_builder.py
    │   ├── text_builder.py
    │   └── hetero_graph_builder.py
    ├── models/
    │   ├── model.py                 # Main STPVE model
    │   ├── behavior_module.py
    │   ├── content_module.py
    │   ├── product_module.py
    │   └── fusion_module.py
    ├── baseline_models/
    │   └── baseline_models.py
    ├── training/
    │   ├── trainer.py               # STPVE trainer
    │   ├── baseline_trainer.py      # Baseline trainer
    │   └── metrics.py
    └── utils.py
```

---

## Installation

Create a virtual environment and install the required packages:

```bash
pip install -r requirements.txt
```

PyTorch and PyTorch Geometric may require installation commands matching the local CUDA version.

---

## Configuration

Configuration files are located in:

```text
STPVE/configs/
```

The training configuration contains the following flag:

```yaml
training:
  demo_mode: 0
```

`demo_mode` has the following meaning:

- `0`: standard mode. The trainer can run all configured cross-validation folds.
- `1`: demo mode. The STPVE trainer stops after completing Fold 1.

The YAML files use `demo_mode: 0` by default. Normally, users do not need to edit this value manually: `run_stpve.py` forces standard mode and `run_demo.py` forces demo mode.

---

## Running the Full STPVE Experiment

To train STPVE from raw data using all configured cross-validation folds:

```bash
python run_stpve.py --config STPVE/configs/default.yaml
```

To reuse previously processed data:

```bash
python run_stpve.py --config STPVE/configs/default.yaml --use-preprocessed
```

`run_stpve.py` explicitly sets:

```python
config.training.demo_mode = 0
```

The full STPVE run writes epoch-level results and the best-`sequence_R2` result for each fold to:

```text
result/stpve_epoch_result.xlsx
result/stpve_best_epoch_result.xlsx
```

---

## Running the Demo

The demo is intentionally restricted to the **main STPVE model only**. A small synthetic dataset for the demo should be placed in the repository-level `data/` directory (i.e., alongside `STPVE/`, `run_demo.py`, and `README.md`). The corresponding file paths should match those specified in `STPVE/configs/default.yaml`.

The synthetic dataset is provided solely to make the released pipeline executable without exposing the confidential empirical data.

The demo does the following:

1. Loads the STPVE configuration from `STPVE/configs/default.yaml`.
2. Sets `config.training.demo_mode = 1` automatically.
3. Runs **only Fold 1** of STPVE.
4. Does **not** import, initialize, or train any baseline model.
5. Runs all configured epochs within Fold 1.
6. Selects the Fold-1 epoch with the highest `sequence_R2`.
7. Saves only that best result and removes the `epoch` column from the demo Excel file.

Run the demo from raw data with:

```bash
python run_demo.py
```

Or reuse preprocessed data with:

```bash
python run_demo.py --use-preprocessed
```

The output is:

```text
result/demo_result.xlsx
```

The console will show:

```text
Demo mode: STPVE only | Fold 1 only | Baselines disabled
```

**Important:** `run_demo.py` never runs baseline models. Use `run_baselines.py` separately when baseline results are needed.

### Note on Demo Results

The demo is intended as a lightweight verification of the released implementation and execution pipeline rather than as a reproduction of the empirical performance reported in the paper. The STPVE architecture and its number of trainable parameters are unchanged from the full implementation. However, the demo uses only a small synthetic sample and therefore contains substantially fewer observations than the full empirical dataset. The synthetic sample is constructed for code demonstration and is not intended to reproduce all statistical properties or predictive relationships of the original Douyin data.

Consequently, predictive metrics obtained from the demo may differ substantially from those reported in the paper. In particular, STPVE is not necessarily expected to outperform simpler baseline models in this reduced-data setting, because relative model performance can depend on sample size, data distribution, and the information contained in the observations. Demo results should therefore be interpreted as verification that the end-to-end workflow—including data loading, model construction, training, evaluation, and result export—executes successfully, rather than as a replication of the full empirical comparison.

---

## Running Baselines

To train all baseline models:

```bash
python run_baselines.py --config STPVE/configs/baseline.yaml
```

To train selected baseline models:

```bash
python run_baselines.py --config STPVE/configs/baseline.yaml --baseline LSTM GRU Transformer LightGBM
```

To reuse previously processed data:

```bash
python run_baselines.py --config STPVE/configs/baseline.yaml --use-preprocessed
```

`run_baselines.py` explicitly sets:

```python
config.training.demo_mode = 0
```

Baseline results are written separately to:

```text
result/baseline_epoch_result.xlsx
result/baseline_best_epoch_result.xlsx
```

---

## Entry-Point Summary

- `python run_demo.py` -> **STPVE only, Fold 1 only, no baselines**.
- `python run_stpve.py` -> **full STPVE cross-validation experiment**.
- `python run_baselines.py` -> **baseline experiments only**.
---

## Data Availability

The empirical data used in this study were obtained from Douyin and a collaborating merchant and contain proprietary platform and merchant information. Because of confidentiality and data-use restrictions, the underlying empirical dataset cannot be made publicly available.

To support code verification and end-to-end execution, the released repository includes a small synthetic dataset in the repository-level `data/` directory. The synthetic dataset contains no real user-, merchant-, or transaction-level observations from the empirical Douyin dataset. It is constructed solely to reproduce the input structure required by the code and to allow the STPVE demo pipeline to run from data loading through model training, evaluation, and result export.

The synthetic dataset is not intended to reproduce the statistical properties, predictive relationships, or empirical performance of the confidential full dataset. Accordingly, results obtained from the demo should not be interpreted as a replication of the performance reported in the paper or as a benchmark comparison among models. The empirical results reported in the manuscript are based on the confidential full dataset and the complete experimental procedure described in the paper.

For the released repository, place the synthetic demo files under the root-level `data/` directory and ensure that the corresponding paths in `STPVE/configs/default.yaml` point to those files using repository-relative paths. Then run `python run_demo.py` from the repository root to execute the demo.

