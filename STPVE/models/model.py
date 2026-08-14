from __future__ import annotations

import torch
import torch.nn as nn

from STPVE.config import PRODUCT_FEATURES
from STPVE.models.behavior_module import VEBehaviorExtractionModule
from STPVE.models.content_module import VEContentExtractionModule
from STPVE.models.product_module import VEProductExtractionModule
from STPVE.models.fusion_module import VETripleFusionModule


class STPVE(nn.Module):
    """End-to-end model matching the paper flowchart.

    Step 2 contains three extraction modules: viewer engagement behavior,
    viewer engagement content, and product information. Step 3 fuses the three
    feature vectors. Step 4 produces sequence prediction and trend classification.
    """

    def __init__(
            self,
            cnn_input_size: list[int],
            content_kernel_sizes: list[int],
            hidden_size_lstm: int,
            module_out_dim: int,
            num_attr_nodes: int,
            prediction_length: int,
            behavior_kernel_size: int = 5,
            content_kernel_num: int = 100,
            num_feature_nodes: int = len(PRODUCT_FEATURES),
            num_heads_graph: int = 4,
            num_heads_lstm: int = 4,
            num_layers_lstm: int = 1,
            num_topics: int = 20,
            topic_embed_dim: int = 32,
            topic_kernel_size: int = 3,
            vocab_embed_dim: int = 256,
            vocab_size: int = 5000,
    ):
        super().__init__()

        self.behavior_extraction = VEBehaviorExtractionModule(
            behavior_kernel_size=behavior_kernel_size,
            input_size=cnn_input_size,
            module_out_dim=module_out_dim,
        )
        self.content_extraction = VEContentExtractionModule(
            content_kernel_num=content_kernel_num,
            content_kernel_sizes=content_kernel_sizes,
            module_out_dim=module_out_dim,
            num_topics=num_topics,
            topic_embed_dim=topic_embed_dim,
            topic_kernel_size=topic_kernel_size,
            vocab_embed_dim=vocab_embed_dim,
            vocab_size=vocab_size
        )

        self.product_extraction = VEProductExtractionModule(
            module_out_dim=module_out_dim,
            num_attr_nodes=num_attr_nodes,
            num_feature_nodes=num_feature_nodes,
            num_heads_graph=num_heads_graph
        )
        self.fusion = VETripleFusionModule(
            feature_dim=module_out_dim,
            hidden_size_lstm=hidden_size_lstm,
            num_layers_lstm=num_layers_lstm,
            num_heads_lstm=num_heads_lstm,
            prediction_length=prediction_length,
        )

    def forward(self, comment_indices, topic_indices, behavior, graph_dict, graph_keys):
        behavior_features, categorical_attention, spatial_attention = self.behavior_extraction(behavior)
        content_features = self.content_extraction(comment_indices, topic_indices)
        product_features = self.product_extraction(graph_dict, graph_keys)
        sequence_output, trend_output = self.fusion(behavior_features, content_features, product_features)
        return sequence_output, trend_output, categorical_attention, spatial_attention
