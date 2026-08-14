from __future__ import annotations
import ast
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm

from STPVE.config import PRODUCT_ATTRIBUTE_COLUMNS, PRODUCT_FEATURES
from STPVE.utils import floor_to_window


@dataclass
class ProductAttributeIndex:
    value_to_id: Dict[str, int]
    id_to_value: List[Tuple[str, str]]
    product_to_attr_nodes: Dict[str, List[int]]
    num_attr_nodes: int


@dataclass
class ProductFeatureNormalizer:
    minimum: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, product_lists: List[dict], feature_names: List[str]) -> "ProductFeatureNormalizer":
        values = []
        for products in product_lists:
            if not isinstance(products, dict):
                continue
            for product_info in products.values():
                values.append([float(product_info.get(feature, 0.0) or 0.0) for feature in feature_names])
        if not values:
            feature_count = len(feature_names)
            return cls(minimum=np.zeros(feature_count, dtype=np.float32),
                       scale=np.ones(feature_count, dtype=np.float32))
        matrix = np.asarray(values, dtype=np.float32)
        minimum = np.nanmin(matrix, axis=0)
        maximum = np.nanmax(matrix, axis=0)
        scale = maximum - minimum
        scale[scale < 1e-8] = 1.0
        return cls(minimum=minimum.astype(np.float32), scale=scale.astype(np.float32))

    def transform(self, feature_values: List[float]) -> List[float]:
        values = np.asarray(feature_values, dtype=np.float32)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        normalized = (values - self.minimum) / self.scale
        return np.clip(normalized, 0.0, 1.0).astype(np.float32).tolist()


def load_product_attribute_index(product_attribute_path: str) -> ProductAttributeIndex:
    attributes = pd.read_csv(product_attribute_path)
    value_to_id: Dict[str, int] = {}
    id_to_value: List[Tuple[str, str]] = []
    current_id = 0
    for column in PRODUCT_ATTRIBUTE_COLUMNS:
        if column not in attributes.columns:
            continue
        for value in attributes[column].dropna().unique():
            key = f"{column}::{value}"
            value_to_id[key] = current_id
            id_to_value.append((column, value))
            current_id += 1
    value_to_id["unknown"] = current_id
    id_to_value.append(("unknown", "unknown"))
    current_id += 1

    product_to_attr_nodes: Dict[str, List[int]] = {}
    for _, row in attributes.iterrows():
        product_id = str(row["product_id"])
        attr_ids = []
        for column in PRODUCT_ATTRIBUTE_COLUMNS:
            value = row[column] if column in row else np.nan
            key = "unknown" if pd.isna(value) else f"{column}::{value}"
            attr_ids.append(value_to_id.get(key, value_to_id["unknown"]))
        product_to_attr_nodes[product_id] = list(set(attr_ids))
    return ProductAttributeIndex(value_to_id, id_to_value, product_to_attr_nodes, current_id)


def build_graph_from_precomputed(
        current_minute: pd.Timestamp,
        window_info_list: List[Tuple[pd.Timestamp, set, dict]],
        product_id_to_node: Dict[str, int],
        attribute_index: ProductAttributeIndex,
        num_feature_nodes: int,
        window_minutes: int,
        feature_normalizer: ProductFeatureNormalizer,
) -> HeteroData | None:
    current_window_start = floor_to_window(current_minute, window_minutes)
    current_window_idx = -1
    for i, (w_start, _, _) in enumerate(window_info_list):
        if w_start == current_window_start:
            current_window_idx = i
            break
    if current_window_idx == -1:
        return None
    prefix_info = window_info_list[: current_window_idx + 1]
    num_time_nodes = len(prefix_info)
    graph = HeteroData()

    timestamps = np.asarray([window_start.timestamp() for window_start, _, _ in prefix_info])
    timestamps_norm = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min() + 1e-8)
    graph["explain"].x = torch.tensor(timestamps_norm, dtype=torch.float).view(-1, 1)
    graph["product"].x = torch.zeros(len(product_id_to_node), 1)
    graph["attribute"].x = torch.eye(attribute_index.num_attr_nodes)
    graph["feature"].x = torch.eye(num_feature_nodes)

    time_product_src, time_product_dst, time_product_attr = [], [], []
    for time_idx, (_, explained_products, _) in enumerate(prefix_info):
        for product_id in explained_products:
            product_key = str(product_id)
            if product_key in product_id_to_node:
                time_product_src.append(time_idx)
                time_product_dst.append(product_id_to_node[product_key])
                time_product_attr.append(1.0)
    if time_product_src:
        edge_index = torch.tensor([time_product_src, time_product_dst], dtype=torch.long)
        graph["explain", "to", "product"].edge_index = edge_index
        graph["explain", "to", "product"].edge_attr = torch.tensor(time_product_attr, dtype=torch.float).view(-1, 1)
        graph["product", "to", "explain"].edge_index = edge_index[[1, 0]]

    if num_time_nodes > 1:
        graph["explain", "to", "explain"].edge_index = torch.tensor(
            [list(range(num_time_nodes - 1)), list(range(1, num_time_nodes))], dtype=torch.long
        )

    product_attr_src, product_attr_dst = [], []
    for product_id, product_node in product_id_to_node.items():
        for attr_id in attribute_index.product_to_attr_nodes.get(str(product_id), []):
            product_attr_src.append(product_node)
            product_attr_dst.append(attr_id)
    if product_attr_src:
        edge_index = torch.tensor([product_attr_src, product_attr_dst], dtype=torch.long)
        graph["product", "to", "attribute"].edge_index = edge_index
        graph["attribute", "to", "product"].edge_index = edge_index[[1, 0]]

    _, _, last_products_dict = prefix_info[-1]
    minute_feature_dict = {}
    for product_info in last_products_dict.values():
        product_id = str(product_info.get("product_id"))
        raw_values = [float(product_info.get(feature, 0.0) or 0.0) for feature in PRODUCT_FEATURES]
        minute_feature_dict[product_id] = raw_values

    product_feature_src, product_feature_dst = [], []
    normalized_feature_weights, raw_feature_weights = [], []
    for product_id, product_node in product_id_to_node.items():
        raw_feature_values = minute_feature_dict.get(str(product_id), [0.0] * num_feature_nodes)
        normalized_feature_values = feature_normalizer.transform(raw_feature_values)
        for feature_id, (normalized_value, raw_value) in enumerate(zip(normalized_feature_values, raw_feature_values)):
            product_feature_src.append(product_node)
            product_feature_dst.append(feature_id)
            normalized_feature_weights.append(normalized_value)
            raw_feature_weights.append(raw_value)
    if product_feature_src:
        edge_index = torch.tensor([product_feature_src, product_feature_dst], dtype=torch.long)
        edge_attr = torch.tensor(normalized_feature_weights, dtype=torch.float).view(-1, 1)
        raw_edge_attr = torch.tensor(raw_feature_weights, dtype=torch.float).view(-1, 1)
        graph["product", "to", "feature"].edge_index = edge_index
        graph["product", "to", "feature"].edge_attr = edge_attr
        graph["product", "to", "feature"].raw_edge_attr = raw_edge_attr
        graph["feature", "to", "product"].edge_index = edge_index[[1, 0]]
        graph["feature", "to", "product"].edge_attr = edge_attr
        graph["feature", "to", "product"].raw_edge_attr = raw_edge_attr

    graph.current_time_idx = current_window_idx
    return graph


def build_dynamic_graphs(
        data: pd.DataFrame,
        product_data_path: str,
        attribute_index: ProductAttributeIndex,
        window_minutes: int,
) -> Dict[str, HeteroData]:
    product_data = pd.read_csv(product_data_path, index_col=0)
    product_data["product_list"] = product_data["product_list"].apply(ast.literal_eval)
    product_data["time"] = pd.to_datetime(product_data["time"])
    product_data = product_data.sort_values("time")
    product_data["time_window"] = product_data["time"].apply(lambda value: floor_to_window(value, window_minutes))

    room_to_times = data.groupby("live_room_id")["time"].unique().to_dict()
    graph_dict: Dict[str, HeteroData] = {}

    for live_room_id, times in tqdm(room_to_times.items(), desc="Building dynamic graphs"):
        live_df = product_data[product_data["live_room_id"] == live_room_id].copy()
        if live_df.empty:
            continue

        window_info_list = []
        for window_start in sorted(live_df["time_window"].unique()):
            window_data = live_df[live_df["time_window"] == window_start].sort_values("time")
            explained_products = set()

            for _, row in window_data.iterrows():
                for product_info in row["product_list"].values():
                    if product_info.get("explaining", False):
                        explained_products.add(str(product_info.get("product_id")))

            window_info_list.append(
                (window_start, explained_products, window_data.iloc[-1]["product_list"])
            )

        for minute in times:
            current_window_start = floor_to_window(pd.Timestamp(minute), window_minutes)

            current_window_idx = -1
            for i, (window_start, _, _) in enumerate(window_info_list):
                if window_start == current_window_start:
                    current_window_idx = i
                    break

            if current_window_idx == -1:
                continue

            prefix_info = window_info_list[: current_window_idx + 1]

            available_products = set()
            for _, _, products_dict in prefix_info:
                for product_info in products_dict.values():
                    product_id = product_info.get("product_id")
                    if product_id is not None:
                        available_products.add(str(product_id))

            if not available_products:
                continue

            product_ids = sorted(available_products)
            product_id_to_node = {product_id: idx for idx, product_id in enumerate(product_ids)}

            # Fit the product-feature normalizer using only information available
            # up to the current prediction window.
            prefix_product_lists = [products_dict for _, _, products_dict in prefix_info]
            feature_normalizer = ProductFeatureNormalizer.fit(prefix_product_lists, PRODUCT_FEATURES)

            graph = build_graph_from_precomputed(
                current_minute=pd.Timestamp(minute),
                window_info_list=window_info_list,
                product_id_to_node=product_id_to_node,
                attribute_index=attribute_index,
                num_feature_nodes=len(PRODUCT_FEATURES),
                window_minutes=window_minutes,
                feature_normalizer=feature_normalizer,
            )

            if graph is not None:
                key = f"{live_room_id}_{pd.Timestamp(minute).isoformat()}"
                graph_dict[key] = graph

    return graph_dict
