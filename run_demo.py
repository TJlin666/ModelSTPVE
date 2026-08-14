from __future__ import annotations

import argparse
import os

import pandas as pd
import torch

from STPVE.config import ExperimentConfig
from STPVE.data.preprocessing import load_preprocessed_data, prepare_data
from STPVE.training.trainer import ExperimentTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run an STPVE demo on Fold 1 only; baseline models are not run."
    )
    parser.add_argument(
        "--config",
        default="STPVE/configs/default.yaml",
        help="Path to the STPVE YAML config file.",
    )
    parser.add_argument(
        "--use-preprocessed",
        action="store_true",
        help="Load preprocessed data instead of rebuilding it from raw data.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)

    # Demo mode is controlled by the entry script rather than the YAML default.
    # In demo mode ExperimentTrainer stops immediately after Fold 1.
    config.training.demo_mode = 1

    output_path = config.paths.demo_output_path
    if os.path.exists(output_path):
        os.remove(output_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Demo mode: STPVE only | Fold 1 only | Baselines disabled")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    if args.use_preprocessed:
        data, graph_dict, vocabulary = load_preprocessed_data(config, device)
    else:
        data, graph_dict, vocabulary = prepare_data(config, device)

    trainer = ExperimentTrainer(config=config, device=device)
    results = trainer.train(
        data=data,
        graph_dict=graph_dict,
        vocabulary=vocabulary,
        model_name="STPVE",
        save_results=False,
    )

    frame = pd.DataFrame(results)
    if frame.empty:
        raise RuntimeError("The demo run produced no evaluation results.")

    valid = frame.dropna(subset=["sequence_R2"])
    if valid.empty:
        best = frame.iloc[[0]].copy()
    else:
        best_idx = valid["sequence_R2"].idxmax()
        best = frame.loc[[best_idx]].copy()

    # Epoch is used only to select the best row and is omitted from demo_result.xlsx.
    best = best.drop(columns=["epoch"], errors="ignore").reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    best.to_excel(output_path, sheet_name="Demo_Result", index=False)
    print(f"Saved STPVE Fold-1 demo result: {output_path}")


if __name__ == "__main__":
    main()
