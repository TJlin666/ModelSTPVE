from __future__ import annotations

from typing import Iterable, List, Optional

import torch
import torch.nn as nn

from STPVE.config import PRODUCT_FEATURES


class BaselineBehaviorSequenceEncoder(nn.Module):
    """Encode the preprocessed multi-category live-room behavior sequence for baseline models.

    Expected input format is (batch, history_length, channels, data_width). If a tensor
    arrives as (batch, channels, history_length, data_width), it is detected and
    converted before encoding.
    """

    def __init__(self, baseline_input_dim: int, data_width: int, channels: int, history_length: int,
                 baseline_hidden_size: int = 128):
        super().__init__()
        self.channels = channels
        self.history_length = history_length
        self.data_width = data_width
        flat_dim = channels * data_width
        self.feature_batch_norm = nn.BatchNorm1d(flat_dim, affine=False)
        self.temporal_encoder = nn.LSTM(flat_dim, baseline_hidden_size, batch_first=True)
        self.projection = nn.Linear(baseline_hidden_size, baseline_input_dim)

    def _to_time_major(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"behavior must be 4D, got shape={tuple(x.shape)}")
        if x.shape[1] == self.channels and x.shape[2] == self.history_length:
            x = x.permute(0, 2, 1, 3)
        return x

    def forward(self, behavior: torch.Tensor) -> torch.Tensor:
        x = self._to_time_major(behavior)
        batch_size, history_length, channels, data_width = x.shape
        flat_dim = channels * data_width
        x = x.reshape(batch_size, history_length, flat_dim)

        # [B, T, 27] -> [B*T, 27]. BatchNorm1d therefore calculates one
        # mean/variance pair for each of the 27 features over B*T values.
        x = x.reshape(batch_size * history_length, flat_dim)
        x = self.feature_batch_norm(x)
        x = x.reshape(batch_size, history_length, flat_dim)

        output, _ = self.temporal_encoder(x)
        return self.projection(output[:, -1, :])


class MeanPooledCommentEncoder(nn.Module):
    """Simple comment encoder: token indices -> embedding -> mask-aware mean pooling."""

    def __init__(self, baseline_input_dim: int, vocab_size: int):
        super().__init__()
        self.padding_idx = 0
        self.embedding = nn.Embedding(vocab_size, baseline_input_dim, padding_idx=self.padding_idx)
        self.projection = nn.Linear(baseline_input_dim, baseline_input_dim)

    def forward(self, comment_indices: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(comment_indices)
        mask = (comment_indices != self.padding_idx).float().unsqueeze(-1)
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.projection(pooled)


class OptionalBertCommentEncoder(nn.Module):
    """Pretrained Chinese BERT comment encoder with a safe fallback.

    The module first tries to load a pretrained Chinese BERT model. If BERT cannot
    be loaded or raw comment texts are not available during forward propagation,
    it falls back to a mean-pooled embedding encoder.
    """

    def __init__(
            self,
            baseline_input_dim: int,
            vocab_size: int,
            bert_model_name: str = "hfl/chinese-bert-wwm-ext",

    ):
        super().__init__()

        self.uses_bert = False
        self.tokenizer = None
        self.bert = None
        self.bert_projection = None

        self.fallback = MeanPooledCommentEncoder(
            vocab_size=vocab_size,
            baseline_input_dim=baseline_input_dim,
        )

        try:
            from transformers import AutoModel, AutoTokenizer  # type: ignore

            self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
            self.bert = AutoModel.from_pretrained(bert_model_name)

            bert_hidden_size = int(getattr(self.bert.config, "hidden_size"))
            self.bert_projection = nn.Linear(bert_hidden_size, baseline_input_dim)

            for parameter in self.bert.parameters():
                parameter.requires_grad = False

            self.uses_bert = True
            print(f"Loaded pretrained BERT model for baseline: {bert_model_name}")

        except Exception as exc:
            print(f"BERT baseline fallback enabled because BERT could not be loaded: {exc}")

    def forward(
            self,
            comment_indices: torch.Tensor,
            comment_texts: Optional[List[str]] = None,
    ) -> torch.Tensor:
        if (
                not self.uses_bert
                or self.tokenizer is None
                or self.bert is None
                or self.bert_projection is None
                or comment_texts is None
        ):
            return self.fallback(comment_indices)

        encoded = self.tokenizer(
            list(comment_texts),
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=128
        )

        encoded = {
            key: value.to(comment_indices.device)
            for key, value in encoded.items()
        }

        self.bert.eval()

        with torch.no_grad():
            bert_outputs = self.bert(**encoded)

        cls_vector = bert_outputs.last_hidden_state[:, 0, :]

        return self.bert_projection(cls_vector)


class ProductFeatureEncoder(nn.Module):
    """Encode mean raw product-feature values extracted from product-feature edges."""

    def __init__(self, baseline_input_dim: int):
        super().__init__()
        self.product_dim = len(PRODUCT_FEATURES)
        self.layer_norm = nn.LayerNorm(self.product_dim)
        self.projection = nn.Linear(self.product_dim, baseline_input_dim)

    def forward(self, product_features: torch.Tensor) -> torch.Tensor:
        return self.projection(self.layer_norm(product_features))


def extract_product_feature_vector(graph, product_dim: int = len(PRODUCT_FEATURES),
                                   device: Optional[torch.device] = None) -> torch.Tensor:
    """Return mean raw product-feature values when raw_edge_attr is available.

    The graph stores normalized edge_attr for graph attention and raw_edge_attr for
    baseline feature extraction. If raw_edge_attr is unavailable, this function
    falls back to edge_attr for compatibility with older graphs.
    """

    if device is None:
        device = graph["feature"].x.device if hasattr(graph["feature"], "x") else torch.device("cpu")
    vector = torch.zeros(product_dim, dtype=torch.float32, device=device)
    if ("product", "to", "feature") not in graph.edge_types:
        return vector
    edge_store = graph["product", "to", "feature"]
    if not hasattr(edge_store, "edge_index"):
        return vector
    if hasattr(edge_store, "raw_edge_attr"):
        edge_values = edge_store.raw_edge_attr
    elif hasattr(edge_store, "edge_attr"):
        edge_values = edge_store.edge_attr
    else:
        return vector
    feature_ids = edge_store.edge_index[1].to(device)
    values = edge_values.view(-1).to(device=device, dtype=torch.float32)
    counts = torch.zeros(product_dim, dtype=torch.float32, device=device)
    vector.scatter_add_(0, feature_ids, values)
    counts.scatter_add_(0, feature_ids, torch.ones_like(values))
    return vector / counts.clamp_min(1.0)


def extract_product_feature_batch(graph_dict, graph_keys: Iterable[str], device: torch.device,
                                  product_dim: int = len(PRODUCT_FEATURES)) -> torch.Tensor:
    vectors = []
    for key in graph_keys:
        graph = graph_dict.get(key)
        if graph is None:
            vectors.append(torch.zeros(product_dim, dtype=torch.float32, device=device))
        else:
            vectors.append(extract_product_feature_vector(graph, product_dim=product_dim, device=device))
    return torch.stack(vectors, dim=0)


class NeuralBaselineModel(nn.Module):
    """Unified neural baseline for LSTM, GRU, RNN, Transformer, MLP, and BERT+Transformer."""

    def __init__(
            self,
            baseline_type: str,
            baseline_num_layers: int,
            channels: int,
            data_width: int,
            history_length: int,
            prediction_length: int,
            vocab_size: int,
            baseline_hidden_size: int = 256,
            baseline_input_dim: int = 256,
            bert_model_name: str = "hfl/chinese-bert-wwm-ext",
            dropout: float = 0.1,

    ):
        super().__init__()
        self.baseline_type = baseline_type.replace("-", "_")
        self.behavior_encoder = BaselineBehaviorSequenceEncoder(
            baseline_hidden_size=baseline_hidden_size,
            baseline_input_dim=baseline_input_dim,
            channels=channels,
            data_width=data_width,
            history_length=history_length,

        )
        if self.baseline_type == "Bert+Transformer":
            self.comment_encoder = OptionalBertCommentEncoder(
                baseline_input_dim=baseline_input_dim,
                bert_model_name=bert_model_name,
                vocab_size=vocab_size,
            )
        else:
            self.comment_encoder = MeanPooledCommentEncoder(
                vocab_size=vocab_size,
                baseline_input_dim=baseline_input_dim,
            )
        self.product_encoder = ProductFeatureEncoder(baseline_input_dim=baseline_input_dim)

        if self.baseline_type == "LSTM":
            self.sequence_model = nn.LSTM(baseline_input_dim, baseline_hidden_size, num_layers=baseline_num_layers,
                                          batch_first=True,
                                          dropout=dropout if baseline_num_layers > 1 else 0.0)
        elif self.baseline_type == "GRU":
            self.sequence_model = nn.GRU(baseline_input_dim, baseline_hidden_size, num_layers=baseline_num_layers,
                                         batch_first=True,
                                         dropout=dropout if baseline_num_layers > 1 else 0.0)
        elif self.baseline_type == "RNN":
            self.sequence_model = nn.RNN(baseline_input_dim, baseline_hidden_size, num_layers=baseline_num_layers,
                                         batch_first=True,
                                         nonlinearity="tanh", dropout=dropout if baseline_num_layers > 1 else 0.0)
        elif self.baseline_type in {"Transformer", "Bert+Transformer"}:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=baseline_input_dim,
                nhead=4,
                dropout=dropout,
                batch_first=True,
            )
            self.position_embedding = nn.Parameter(torch.randn(1, 3, baseline_input_dim))
            self.sequence_model = nn.TransformerEncoder(encoder_layer, num_layers=baseline_num_layers)
            self.transformer_projection = nn.Linear(baseline_input_dim, baseline_hidden_size)
        elif self.baseline_type == "MLP":
            self.sequence_model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(3 * baseline_input_dim, baseline_hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(baseline_hidden_size, baseline_hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(baseline_hidden_size, baseline_hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(baseline_hidden_size, baseline_hidden_size),
                nn.ReLU(),
            )

        else:
            raise ValueError(f"Unsupported neural baseline_type: {baseline_type}")

        self.ffn_sequence = nn.Linear(baseline_hidden_size, prediction_length)
        self.ffn_classification = nn.Linear(baseline_hidden_size, 3)

    def _fuse_modalities(self, behavior: torch.Tensor, content: torch.Tensor,
                         product: torch.Tensor) -> torch.Tensor:
        return torch.stack([behavior, content, product], dim=1)

    def _encode_sequence(self, modality_sequence: torch.Tensor) -> torch.Tensor:
        if self.baseline_type in {"LSTM", "GRU", "RNN"}:
            output, _ = self.sequence_model(modality_sequence)
            return output[:, -1, :]
        if self.baseline_type in {"Transformer", "Bert+Transformer"}:
            output = self.sequence_model(modality_sequence + self.position_embedding)
            return self.transformer_projection(output[:, -1, :])
        return self.sequence_model(modality_sequence)

    def forward(self, behavior, comment_indices,product_features,  comment_texts: Optional[List[str]] = None):
        behavior_vector = self.behavior_encoder(behavior)
        if isinstance(self.comment_encoder, OptionalBertCommentEncoder):
            content_vector = self.comment_encoder(comment_indices, comment_texts=comment_texts)
        else:
            content_vector = self.comment_encoder(comment_indices)
        product_vector = self.product_encoder(product_features)
        modality_sequence = self._fuse_modalities(behavior_vector, content_vector, product_vector)
        representation = self._encode_sequence(modality_sequence)
        return self.ffn_sequence(representation), self.ffn_classification(representation)
