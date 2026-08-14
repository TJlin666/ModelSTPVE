from __future__ import annotations

import torch
import torch.nn as nn


class VETripleFusionModule(nn.Module):
    """Fuse FV^Behavior, FV^Content, and FV^Product and produce two task outputs."""

    def __init__(
            self,
            feature_dim: int,
            hidden_size_lstm: int,
            prediction_length: int,
            num_layers_lstm: int = 1,
            num_heads_lstm: int = 4,
    ):
        super().__init__()
        self.prediction_length = prediction_length
        self.lstm = nn.LSTM(feature_dim, hidden_size_lstm, num_layers_lstm, batch_first=True)
        self.attention = StepMultiHeadAttention(hidden_size_lstm, num_heads_lstm, prediction_length)
        self.ffn_sequence = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size_lstm, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
            for _ in range(prediction_length)
        ])
        self.ffn_classification = nn.Sequential(nn.Linear(hidden_size_lstm, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(
            self,
            behavior_features: torch.Tensor,
            content_features: torch.Tensor,
            product_features: torch.Tensor,
    ):
        multisource_sequence = torch.stack([behavior_features, content_features, product_features], dim=1)
        lstm_output, _ = self.lstm(multisource_sequence)
        attention_outputs = self.attention(lstm_output)
        global_feature = torch.stack(attention_outputs, dim=1).mean(dim=1)
        sequence_output = torch.cat([head(attention_outputs[i]) for i, head in enumerate(self.ffn_sequence)], dim=1)
        class_output = self.ffn_classification(global_feature)
        return sequence_output, class_output


class StepMultiHeadAttention(nn.Module):
    def __init__(self, hidden_size_lstm: int, num_heads_lstm: int, prediction_length: int):
        super().__init__()
        if hidden_size_lstm % num_heads_lstm != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")
        self.hidden_size = hidden_size_lstm
        self.num_heads = num_heads_lstm
        self.head_dim = hidden_size_lstm // num_heads_lstm
        self.prediction_length = prediction_length
        self.query_layers = nn.ModuleList([nn.Linear(hidden_size_lstm, hidden_size_lstm) for _ in range(prediction_length)])
        self.key_layers = nn.ModuleList([nn.Linear(hidden_size_lstm, hidden_size_lstm) for _ in range(prediction_length)])
        self.value_layers = nn.ModuleList([nn.Linear(hidden_size_lstm, hidden_size_lstm) for _ in range(prediction_length)])

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        outputs = []
        scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32, device=x.device))
        for step in range(self.prediction_length):
            q = self.query_layers[step](x).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            k = self.key_layers[step](x).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v = self.value_layers[step](x).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            attention_scores = torch.matmul(q, k.permute(0, 1, 3, 2)) / scale
            attention_weights = torch.softmax(attention_scores, dim=-1)
            attended = torch.matmul(attention_weights, v)
            attended = attended.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.hidden_size)
            outputs.append(attended[:, -1, :])
        return outputs
