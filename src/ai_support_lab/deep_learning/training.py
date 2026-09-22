import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from ai_support_lab.deep_learning.dataset import LABELS, prepare_data
from ai_support_lab.deep_learning.model import TicketTextClassifier


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.inference_mode():
        for tokens, targets in loader:
            tokens, targets = tokens.to(device), targets.to(device)
            logits = model(tokens)
            total_loss += float(criterion(logits, targets))
            correct += int((logits.argmax(dim=1) == targets).sum())
            count += len(targets)
    return {"loss": total_loss / count, "accuracy": correct / count}


def save_inference_checkpoint(model: nn.Module, vocabulary: dict[str, int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # state_dict переносим на CPU: артефакт можно читать на машине без CUDA.
    # Словарь и порядок классов — часть модели, без них веса не воспроизводят inference.
    torch.save(
        {
            "version": 1,
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "vocabulary": vocabulary,
            "labels": LABELS,
        },
        path,
    )


def train(
    path: Path, checkpoint: Path, epochs: int = 8, device_name: str = "cpu"
) -> dict[str, Any]:
    torch.manual_seed(42)
    torch.set_num_threads(2)
    device = torch.device(device_name)
    data = prepare_data(path)
    model = TicketTextClassifier(len(data.vocabulary), len(LABELS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    best_loss = float("inf")
    history = []
    for epoch in range(epochs):
        model.train()
        total_loss, count = 0.0, 0
        for tokens, targets in data.train:
            tokens, targets = tokens.to(device), targets.to(device)
            # .grad накапливается по умолчанию. Обнуляем его перед новым batch;
            # set_to_none уменьшает лишние записи в память и явно отделяет шаги.
            optimizer.zero_grad(set_to_none=True)
            logits = model(tokens)
            loss = criterion(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(targets)
            count += len(targets)
        validation = evaluate(model, data.validation, device)
        history.append({"epoch": epoch + 1, "train_loss": total_loss / count, **validation})
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            save_inference_checkpoint(model, data.vocabulary, checkpoint)
            # Отдельный resumable checkpoint хранит optimizer и epoch; inference
            # артефакт выше намеренно компактнее и не нужен для продолжения training.
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "rng_state": torch.get_rng_state(),
                },
                checkpoint.with_suffix(".resume.pt"),
            )
    report = {"history": history, "device": str(device), "best_validation_loss": best_loss}
    checkpoint.with_suffix(".metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
