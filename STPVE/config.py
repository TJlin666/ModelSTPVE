from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml

PRODUCT_FEATURES: List[str] = [
    "pay_combo_cnt", "unpay_cnt", "refund_rate",
    "create_cnt", "pay_cnt", "product_click_pay_ucnt_ratio", "product_click_cnt_5min",
    "pay_amt_5min", "campaign_stock_cnt", "avg_max_pay_amt_min", "product_show_ucnt",
    "stock_cnt", "product_show_click_ucnt_ratio", "create_pay_ucnt_ratio",
    "product_show_pay_ucnt_ratio", "pay_amt", "gpm", "market_price", "product_click_ucnt",
    "room_cart_num", "explaining",
]

PRODUCT_ATTRIBUTE_COLUMNS: List[str] = [
    "Product Information Type",
    "Consumer Involvement Level",
    "Utility Structure",
    "Dominant Perceived Risk Type",
    "Return Risk Level",
    "Standardization Degree",
    "Fulfillment and Logistics Attribute",
    "Visual Verifiability",
    "Product Form",
    "Product Life Cycle Stage",
    "Durability",
    "Configuration Decision Complexity",
    "Bundle Complexity",
    "Social and Emotional Attribute",
    "Seasonality and Time Sensitivity",
    "Repurchase Frequency",
    "Trust Source",
    "Sensory Dependence Degree",
    "Perishability",
    "Commitment Structure",
    "Search Intensity",
    "Consumer Goods Category",
    "Price Transparency",
    "Fit Uncertainty",
    "Using Feedback Delay",
    "Using Learning Cost",
    "Severity Consequences",
    "Complement Dependence",
    "Lock-in Effect",
    "Demonstration Dependence",
    "Intended User",
    "Decision Deferrability",
    "Safety and Compliance Sensitivity",
    "Information Presentation Format",
    "Decision Inertia Degree",
    "Substitutes Density",
    "Brand Awareness"
]


@dataclass
class PathsConfig:
    baseline_epoch_output_path: str = "result/baseline_epoch_result.xlsx"
    behavior_data_path: str = "data/data_matched_result.csv"
    comment_data_path: str = "data/comment_data.csv"
    stpve_epoch_output_path: str = "result/stpve_epoch_result.xlsx"
    stpve_best_epoch_output_path: str = "result/stpve_best_epoch_result.xlsx"
    baseline_best_epoch_output_path: str = "result/baseline_best_epoch_result.xlsx"
    demo_output_path: str = "result/demo_result.xlsx"
    graph_dict_path: str = "data/graph_dict.pkl"
    lda_vectorizer_path: str = "data/lda_vectorizer.pkl"
    lda_model_path: str = "data/lda_model.pkl"
    product_attribute_path: str = "data/all_product.csv"
    product_data_path: str = "data/product_data.csv"
    processed_data_path: str = "data/new_data_with_topic.csv"
    vocab_path: str = "data/vocab.pkl"


@dataclass
class DataConfig:
    channels: int = 3
    data_width: int = 9
    graph_time_window_size: int = 5
    history_length: int = 5
    lda_num_topics: int = 20
    lda_max_features: int = 2000
    lda_top_topics: int = 5
    max_comment_len: int = 200
    prediction_length: int = 5
    target_column: str = "ConvOrder_s"


@dataclass
class TrainingConfig:
    batch_size: int = 512
    learning_rate: float = 0.0001
    loss_weights: Dict[str, float] = field(default_factory=lambda: {"sequence": 1.0, "class": 10.0})
    num_epochs: int = 50
    n_splits: int = 10
    random_state: int = 42
    demo_mode: int = 0
    lgb_n_estimators: int = 200
    lgb_learning_rate: float = 0.05
    lgb_num_leaves: int = 31


@dataclass
class ModelConfig:
    bert_model_name: str = "hfl/chinese-bert-wwm-ext"
    baseline_input_dim: int = 256
    baseline_hidden_size: int = 128
    baseline_num_layers: int = 4
    behavior_kernel_size: int = 5
    content_kernel_num: int = 100
    content_kernel_sizes: List[int] = field(default_factory=lambda: [3, 4, 5])
    hidden_size_lstm: int = 128
    module_out_dim: int = 256
    num_layers_lstm: int = 1
    num_heads_graph: int = 4
    num_heads_lstm: int = 4
    topic_embed_dim: int = 32
    topic_kernel_size: int = 3
    vocab_embed_dim: int = 256


@dataclass
class ExperimentConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as file:
            payload: Dict[str, Any] = yaml.safe_load(file) or {}
        return cls(
            paths=PathsConfig(**payload.get("paths", {})),
            data=DataConfig(**payload.get("data", {})),
            training=TrainingConfig(**payload.get("training", {})),
            model=ModelConfig(**payload.get("model", {})),
        )
