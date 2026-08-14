from __future__ import annotations

import argparse
import os

import torch

from STPVE.config import ExperimentConfig
from STPVE.data.preprocessing import load_preprocessed_data, prepare_data
from STPVE.training.trainer import ExperimentTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train the live sales forecasting model.")
    parser.add_argument("--config", default="STPVE/configs/default.yaml", help="Path to YAML config file.")
    parser.add_argument("--use-preprocessed", action="store_true",
                        help="Load preprocessed tensors and graphs instead of rebuilding them.")
    parser.add_argument("--model-name", default="STPVE", help="Name written to the evaluation file.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    if args.use_preprocessed:
        data, graph_dict, vocabulary = load_preprocessed_data(config, device)
    else:
        data, graph_dict, vocabulary = prepare_data(config, device)

    # run_stpve.py always performs the standard cross-validation run.
    # run_demo.py is the only entry point that enables demo mode.
    config.training.demo_mode = 0
    trainer = ExperimentTrainer(config=config, device=device)
    output_paths = [
        config.paths.stpve_epoch_output_path,
        config.paths.stpve_best_epoch_output_path,
    ]
    for output_path in output_paths:
        if os.path.exists(output_path):
            os.remove(output_path)
    trainer.train(data=data, graph_dict=graph_dict, vocabulary=vocabulary, model_name=args.model_name)


if __name__ == "__main__":
    main()
