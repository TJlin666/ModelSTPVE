from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopicGatedContentCNN(nn.Module):
    """Character-level CNN with LDA-topic gated feature modulation."""

    def __init__(
            self,
            topic_kernel_size: int,
            vocab_size: int,
            content_kernel_num: int = 100,
            content_kernel_sizes=None,
            num_topics: int = 20,
            topic_embed_dim: int = 32,
            vocab_embed_dim: int = 256,
    ):
        super().__init__()
        if content_kernel_sizes is None:
            content_kernel_sizes = [3, 4, 5]
        self.word_embedding = nn.Embedding(vocab_size, vocab_embed_dim, padding_idx=0)
        self.topic_embedding = nn.Embedding(num_topics, topic_embed_dim)

        self.content_convs = nn.ModuleList([
            nn.Conv1d(vocab_embed_dim, content_kernel_num, kernel_size)
            for kernel_size in content_kernel_sizes
        ])

        self.gate_convs = nn.ModuleList([
            nn.Conv1d(vocab_embed_dim, content_kernel_num, kernel_size)
            for kernel_size in content_kernel_sizes
        ])

        self.topic_convs = nn.ModuleList([
            nn.Conv1d(topic_embed_dim, content_kernel_num, kernel_size=topic_kernel_size)
        ])

        self.topic_projection = nn.Linear(content_kernel_num, content_kernel_num)
        self.dropout = nn.Dropout(0.2)

    def forward(self, comment_indices: torch.Tensor, topic_indices: torch.Tensor):
        comment_embedding = self.word_embedding(comment_indices)
        topic_embedding = self.topic_embedding(topic_indices)

        topic_features = [
            F.relu(conv(topic_embedding.transpose(1, 2)))
            for conv in self.topic_convs
        ]
        topic_features = [
            F.max_pool1d(feature, feature.size(2)).squeeze(2)
            for feature in topic_features
        ]
        topic_features = torch.cat(topic_features, dim=1)

        content_features = [
            torch.tanh(conv(comment_embedding.transpose(1, 2)))
            for conv in self.content_convs
        ]

        gate_bias = self.topic_projection(topic_features).unsqueeze(2)

        gate_features = [
            F.relu(conv(comment_embedding.transpose(1, 2)) + gate_bias)
            for conv in self.gate_convs
        ]

        gated_features = [
            content * gate
            for content, gate in zip(content_features, gate_features)
        ]

        pooled_features = [
            F.max_pool1d(feature, feature.size(2)).squeeze(2)
            for feature in gated_features
        ]

        representation = self.dropout(torch.cat(pooled_features, dim=1))

        return representation


class CommentEmbeddingModule(nn.Module):
    def __init__(
            self,
            topic_kernel_size: int,
            vocab_size: int,
            content_kernel_num: int = 100,
            content_kernel_sizes=None,
            module_out_dim: int = 256,
            num_topics: int = 20,
            topic_embed_dim: int = 32,
            vocab_embed_dim: int = 256,

    ):
        super().__init__()

        if content_kernel_sizes is None:
            content_kernel_sizes = [3, 4, 5]
        self.encoder = TopicGatedContentCNN(
            content_kernel_num=content_kernel_num,
            content_kernel_sizes=content_kernel_sizes,
            num_topics=num_topics,
            topic_embed_dim=topic_embed_dim,
            topic_kernel_size=topic_kernel_size,
            vocab_embed_dim=vocab_embed_dim,
            vocab_size=vocab_size,
        )

        self.projection = nn.Linear(len(content_kernel_sizes) * content_kernel_num, module_out_dim)

    def forward(
            self,
            comment_indices: torch.Tensor,
            topic_indices: torch.Tensor,
    ) -> torch.Tensor:
        representation = self.encoder(comment_indices, topic_indices)
        return self.projection(representation)


class VEContentExtractionModule(nn.Module):
    """Extract FV^Content from viewer-generated comment/content tokens."""

    def __init__(
            self,
            content_kernel_sizes: list,
            topic_kernel_size: int,
            vocab_size: int,
            module_out_dim: int = 256,
            num_topics: int = 20,
            vocab_embed_dim: int = 256,
            content_kernel_num: int = 100,
            topic_embed_dim: int = 32,
    ):
        super().__init__()
        self.encoder = CommentEmbeddingModule(
            content_kernel_num=content_kernel_num,
            content_kernel_sizes=content_kernel_sizes,
            module_out_dim=module_out_dim,
            num_topics=num_topics,
            topic_embed_dim=topic_embed_dim,
            topic_kernel_size=topic_kernel_size,
            vocab_embed_dim=vocab_embed_dim,
            vocab_size=vocab_size,
        )

    def forward(self, comment_indices: torch.Tensor, topic_indices: torch.Tensor) -> torch.Tensor:
        return self.encoder(comment_indices, topic_indices)
