from __future__ import annotations
import ast
import pickle
from pathlib import Path
from typing import Dict

import pandas as pd
import torch

try:
    from pympler import asizeof
except Exception:
    class _SizeOfFallback:
        @staticmethod
        def asizeof(obj):
            return 0


    asizeof = _SizeOfFallback()
from tqdm import tqdm

from STPVE.config import ExperimentConfig
from STPVE.data.hetero_graph_builder import build_dynamic_graphs, load_product_attribute_index
from STPVE.data.multi_feature_map_builder import build_multi_category_feature_map
from STPVE.data.text_builder import (
    build_character_vocabulary,
    get_comment_indices_for_sample,
    load_pickle,
    save_pickle,
)
from STPVE.training.metrics import compute_slope, slope_to_class
from STPVE.utils import floor_to_window


def _graph_lookup(row: pd.Series, graph_dict: Dict) -> str | None:
    key = f"{row['live_room_id']}_{pd.Timestamp(row['time']).isoformat()}"
    graph = graph_dict.get(key)
    if graph is None:
        return None
    return key


def prepare_data(config: ExperimentConfig, device: torch.device):
    behavior_data = pd.read_csv(config.paths.behavior_data_path, encoding="utf-8", index_col=0)
    data = behavior_data.copy()
    data["time"] = pd.to_datetime(data["time"])
    target = config.data.target_column

    data["time_window"] = data["time"].apply(lambda value: floor_to_window(value, config.data.graph_time_window_size))
    grouped_samples = []
    for _, group in tqdm(data.groupby("live_room_id"), desc="Creating sequence samples"):
        processed = build_multi_category_feature_map(
            group=group,
            history_length=config.data.history_length,
            prediction_length=config.data.prediction_length,
            target_column=target,
        )
        if not processed.empty:
            grouped_samples.append(processed)
    if not grouped_samples:
        raise ValueError("No valid sequence samples were created. Check history/prediction lengths and input data.")
    data = pd.concat(grouped_samples, ignore_index=True)

    attribute_index = load_product_attribute_index(config.paths.product_attribute_path)
    graph_dict = build_dynamic_graphs(data, config.paths.product_data_path, attribute_index,
                                      config.data.graph_time_window_size)
    print(f"graph_dict memory footprint: {asizeof.asizeof(graph_dict) / 1024 / 1024:.2f} MB")

    vocabulary, data = build_character_vocabulary(data, config.paths.comment_data_path)
    comment_indices, comment_texts = [], []
    for _, row in tqdm(data.iterrows(), total=len(data), desc="Creating comment tensors"):
        indices, text = get_comment_indices_for_sample(row, data, vocabulary, config.data.max_comment_len)
        comment_indices.append(indices)
        comment_texts.append(text)
    data["comment_indices"] = comment_indices
    data["comment_text"] = comment_texts

    # Keep comment_text in the processed dataset. LDA topic/aspect indices are now
    # fitted inside each cross-validation fold using only the training split, then
    # applied to both train and test splits for that fold.

    data["graph_key"] = data.apply(lambda row: _graph_lookup(row, graph_dict), axis=1)
    missing_graph_count = data["graph_key"].isna().sum()
    if missing_graph_count > 0:
        print(f"Dropping {missing_graph_count} samples without matched product graphs.")
        data = data.dropna(subset=["graph_key"]).reset_index(drop=True)
    data["class_label"] = data["sequence_label"].apply(lambda seq: slope_to_class(compute_slope(seq)))
    Path(config.paths.processed_data_path).parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(config.paths.processed_data_path, index=False)
    with open(config.paths.graph_dict_path, "wb") as file:
        pickle.dump(graph_dict, file)
    save_pickle(vocabulary, config.paths.vocab_path)
    for key in graph_dict:
        graph_dict[key] = graph_dict[key].to(device)
    return data, graph_dict, vocabulary


def load_preprocessed_data(config: ExperimentConfig, device: torch.device):
    data = pd.read_csv(config.paths.processed_data_path)
    for col in ["behavior", "sequence_label", "comment_indices"]:
        if col in data.columns:
            data[col] = data[col].apply(ast.literal_eval)
    vocabulary = load_pickle(config.paths.vocab_path)
    graph_dict = load_pickle(config.paths.graph_dict_path)
    for key in graph_dict:
        graph_dict[key] = graph_dict[key].to(device)

    return data, graph_dict, vocabulary
