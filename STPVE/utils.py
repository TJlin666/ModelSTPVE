from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd
import torch

IDENTIFIER_COLUMNS = {
    "time", "time_window", "compass_id", "year_month", "live_room_id", "product_list",
    "live_comment", "comment_indices", "topic_indices", "label", "graph_key", "time_node_idx",
    "slope_label",
}


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def floor_to_window(timestamp: pd.Timestamp, window_minutes: int) -> pd.Timestamp:
    timestamp = pd.Timestamp(timestamp)
    total_minutes = timestamp.hour * 60 + timestamp.minute
    floored_minutes = total_minutes - total_minutes % window_minutes
    return timestamp.replace(hour=floored_minutes // 60, minute=floored_minutes % 60, second=0, microsecond=0)


def worker_init_fn(worker_id: int) -> None:
    torch.manual_seed(worker_id + torch.initial_seed())


def _reshape_history_matrix(values: np.ndarray):
    T, C = values.shape  # T = time_step, C = channels
    F = len(values[0, 0])
    result = np.zeros((C, T, F), dtype=float)

    # Pad data
    for i in range(T):
        for j in range(C):
            result[j, i, :] = values[i, j]

    return result


def count_traditional_model_parameters(model) -> int:
    """Count model complexity for traditional baselines.

    For LightGBM models, this counts total tree nodes.
    For linear/logistic models, this counts learned coefficients and intercepts.
    For sklearn Pipelines, it counts the final estimator.
    For MultiOutputRegressor-like wrappers, it sums over all fitted estimators.
    """
    # sklearn Pipeline: use the final estimator
    if hasattr(model, "steps"):
        model = model.steps[-1][1]

    # MultiOutputRegressor or similar wrappers
    if hasattr(model, "estimators_"):
        return sum(count_traditional_model_parameters(estimator) for estimator in model.estimators_)

    # LightGBM sklearn API: LGBMRegressor / LGBMClassifier
    if hasattr(model, "booster_"):
        booster_dump = model.booster_.dump_model()

        def count_tree_nodes(tree_node):
            if "left_child" not in tree_node and "right_child" not in tree_node:
                return 1
            return (
                    1
                    + count_tree_nodes(tree_node["left_child"])
                    + count_tree_nodes(tree_node["right_child"])
            )

        total_nodes = 0
        for tree_info in booster_dump["tree_info"]:
            total_nodes += count_tree_nodes(tree_info["tree_structure"])

        return total_nodes

    # LinearRegression / LogisticRegression / Ridge / Lasso etc.
    total_params = 0

    if hasattr(model, "coef_"):
        total_params += model.coef_.size

    if hasattr(model, "intercept_"):
        # intercept_ may be scalar or array
        if hasattr(model.intercept_, "size"):
            total_params += model.intercept_.size
        else:
            total_params += 1

    return int(total_params)
