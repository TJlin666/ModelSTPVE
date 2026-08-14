from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv, Linear, global_mean_pool
from torch_geometric.data import Batch
from STPVE.config import PRODUCT_FEATURES


class HeteroGraphExtractor(nn.Module):
    def __init__(self, hidden_dim: int, num_attr_nodes: int, num_feature_nodes: int, num_heads_graph: int = 4):
        super().__init__()
        self.attr_projection = Linear(num_attr_nodes, hidden_dim)
        self.product_projection = Linear(1, hidden_dim)
        self.time_projection = Linear(1, hidden_dim)
        self.feature_projection = Linear(num_feature_nodes, hidden_dim)
        self.conv1 = self._build_conv(hidden_dim, num_heads_graph)
        self.conv2 = self._build_conv(hidden_dim, num_heads_graph)

    @staticmethod
    def _build_conv(hidden_dim: int, num_heads_graph: int = 4) -> HeteroConv:
        return HeteroConv({
            ("attribute", "to", "product"): GATConv(
                hidden_dim, hidden_dim, heads=num_heads_graph, concat=False, add_self_loops=False
            ),
            ("product", "to", "attribute"): GATConv(
                hidden_dim, hidden_dim, heads=num_heads_graph, concat=False, add_self_loops=False
            ),
            ("explain", "to", "product"): GATConv(
                hidden_dim, hidden_dim, heads=num_heads_graph, concat=False, add_self_loops=False
            ),
            ("product", "to", "explain"): GATConv(
                hidden_dim, hidden_dim, heads=num_heads_graph, concat=False, add_self_loops=False
            ),
            ("product", "to", "feature"): GATConv(
                hidden_dim, hidden_dim, heads=num_heads_graph, edge_dim=1,
                concat=False, add_self_loops=False
            ),
            ("feature", "to", "product"): GATConv(
                hidden_dim, hidden_dim, heads=num_heads_graph, edge_dim=1,
                concat=False, add_self_loops=False
            ),
            ("explain", "to", "explain"): GATConv(
                hidden_dim, hidden_dim, heads=num_heads_graph, concat=False, add_self_loops=False
            ),
        }, aggr="sum")

    def forward(self, graph):
        x_dict = {
            "attribute": self.attr_projection(graph["attribute"].x),
            "product": self.product_projection(graph["product"].x),
            "explain": self.time_projection(graph["explain"].x),
            "feature": self.feature_projection(graph["feature"].x),
        }

        edge_attr_dict = self._build_edge_attr_dict(graph)

        x_dict = self._apply_layer(
            self.conv1,
            x_dict,
            graph.edge_index_dict,
            edge_attr_dict=edge_attr_dict,
        )

        x_dict = self._apply_layer(
            self.conv2,
            x_dict,
            graph.edge_index_dict,
            edge_attr_dict=edge_attr_dict,
        )

        return x_dict

    @staticmethod
    def _build_edge_attr_dict(graph):
        edge_attr_dict = {}

        for edge_type in [
            ("product", "to", "feature"),
            ("feature", "to", "product"),
        ]:
            if edge_type in graph.edge_types and hasattr(graph[edge_type], "edge_attr"):
                edge_attr_dict[edge_type] = graph[edge_type].edge_attr

        return edge_attr_dict

    @staticmethod
    def _apply_layer(conv: HeteroConv, x_dict, edge_index_dict, edge_attr_dict=None):
        if edge_attr_dict:
            updated = conv(
                x_dict,
                edge_index_dict,
                edge_attr_dict=edge_attr_dict,
            )
        else:
            updated = conv(
                x_dict,
                edge_index_dict,
            )

        out_dict = {}
        for node_type, original_value in x_dict.items():
            if node_type in updated:
                out = updated[node_type] + original_value
            else:
                out = original_value
            out_dict[node_type] = F.relu(out)

        return out_dict


class VEProductExtractionModule(nn.Module):
    """Extract FV^Product from dynamic heterogeneous product graphs."""

    def __init__(
            self,
            num_attr_nodes: int,
            module_out_dim: int = 256,
            num_feature_nodes: int = len(PRODUCT_FEATURES),
            num_heads_graph: int = 4,

    ):
        super().__init__()
        self.output_dim = module_out_dim
        self.graph_encoder = HeteroGraphExtractor(
            hidden_dim=module_out_dim,
            num_attr_nodes=num_attr_nodes,
            num_feature_nodes=num_feature_nodes,
            num_heads_graph=num_heads_graph
        )
        self.graph_readout = nn.Sequential(
            nn.Linear(module_out_dim * 4, module_out_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    def forward(self, graph_dict: dict, graph_keys: Sequence[str]) -> torch.Tensor:
        graphs = [graph_dict[key] for key in graph_keys]
        batch = Batch.from_data_list(graphs)

        device = next(self.parameters()).device
        batch = batch.to(device)

        x_dict = self.graph_encoder(batch)
        graph_features = self._pool_graph_representation(x_dict, batch)
        return graph_features

    def _pool_graph_representation(self, x_dict: dict, batch: Batch) -> torch.Tensor:
        def batched_mean_pool(node_type: str) -> torch.Tensor:
            emb = x_dict[node_type]
            node_batch = batch[node_type].batch
            return global_mean_pool(emb, node_batch)

        attribute_feat = batched_mean_pool("attribute")
        product_feat = batched_mean_pool("product")
        time_feat = batched_mean_pool("explain")
        feature_feat = batched_mean_pool("feature")

        full_graph_feature = torch.cat([
            attribute_feat,
            product_feat,
            time_feat,
            feature_feat
        ], dim=-1)

        return self.graph_readout(full_graph_feature)
