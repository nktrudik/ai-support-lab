import torch
from torch import nn


class TicketTextClassifier(nn.Module):
    def __init__(self, vocabulary_size: int, classes: int, embedding_dim: int = 48) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, classes)
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # [B, L] -> [B, L, D]. Padding не должен уменьшать среднее коротких
        # обращений, поэтому делим на число реальных токенов, а не на MAX_LENGTH.
        embeddings = self.embedding(token_ids)
        mask = (token_ids != 0).unsqueeze(-1)
        pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        # Возвращаем logits [B, C]. CrossEntropyLoss сама применяет log-softmax;
        # предварительный softmax здесь ухудшил бы численную устойчивость.
        return self.head(pooled)
