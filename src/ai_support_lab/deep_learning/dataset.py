import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from ai_support_lab.classical_ml.dataset import load_dataset
from ai_support_lab.tickets.enums import Category

LABELS = [category.value for category in Category]
MAX_LENGTH = 64


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.casefold())


def build_vocabulary(texts: list[str]) -> dict[str, int]:
    counts = Counter(token for text in texts for token in tokenize(text))
    # Стабильная сортировка делает индексы одинаковыми при одинаковом train split.
    return {
        "<pad>": 0,
        "<unk>": 1,
        **{token: index + 2 for index, token in enumerate(sorted(counts))},
    }


def encode(text: str, vocabulary: dict[str, int]) -> torch.Tensor:
    ids = [vocabulary.get(token, 1) for token in tokenize(text)][:MAX_LENGTH]
    return torch.tensor(ids + [0] * (MAX_LENGTH - len(ids)), dtype=torch.long)


class TicketDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, texts: list[str], labels: list[str], vocabulary: dict[str, int]) -> None:
        self.tokens = torch.stack([encode(text, vocabulary) for text in texts])
        self.labels = torch.tensor([LABELS.index(label) for label in labels], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.tokens[index], self.labels[index]


@dataclass(frozen=True)
class TrainingData:
    train: DataLoader
    validation: DataLoader
    vocabulary: dict[str, int]


def prepare_data(path: Path, batch_size: int = 32, seed: int = 42) -> TrainingData:
    frame = load_dataset(path)
    train, validation = train_test_split(
        frame, test_size=0.2, stratify=frame["category"], random_state=seed
    )
    vocabulary = build_vocabulary(train["text"].tolist())
    training = TicketDataset(train["text"].tolist(), train["category"].tolist(), vocabulary)
    validating = TicketDataset(
        validation["text"].tolist(), validation["category"].tolist(), vocabulary
    )
    generator = torch.Generator().manual_seed(seed)
    return TrainingData(
        DataLoader(
            training, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0
        ),
        DataLoader(validating, batch_size=batch_size, shuffle=False, num_workers=0),
        vocabulary,
    )
