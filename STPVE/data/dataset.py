from __future__ import annotations

import ast
from typing import Any, Dict

import pandas as pd
import torch
from torch.utils.data import Dataset


def parse_list_column(value: Any):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return ast.literal_eval(value)
    return value


class ForecastingDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, use_graph: bool = True):
        self.data = dataframe.reset_index(drop=True)
        self.use_graph = use_graph

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.data.iloc[index]
        if "behavior" not in sample.index:
            raise KeyError("The processed dataset must contain a behavior column.")
        item: Dict[str, Any] = {
            "behavior": torch.tensor(parse_list_column(sample["behavior"]), dtype=torch.float32),
            "comment_indices": torch.tensor(parse_list_column(sample["comment_indices"]), dtype=torch.long),
            "topic_indices": torch.tensor(parse_list_column(sample["topic_indices"]), dtype=torch.long),
            "sequence_label": torch.tensor(parse_list_column(sample["sequence_label"]), dtype=torch.float32),
            "class_label": torch.tensor(int(sample["class_label"]), dtype=torch.long),
            "comment_text": str(sample.get("comment_text", "")),
            "graph_key": sample["graph_key"],
        }

        return item
