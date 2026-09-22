from pathlib import Path

import torch

from ai_support_lab.deep_learning.dataset import encode
from ai_support_lab.deep_learning.model import TicketTextClassifier
from ai_support_lab.tickets.enums import Category
from ai_support_lab.tickets.exceptions import ModelUnavailable
from ai_support_lab.tickets.schemas import ClassificationResponse


class TorchTicketClassifier:
    def __init__(self, path: Path) -> None:
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if checkpoint["version"] != 1:
                raise ValueError("Unsupported checkpoint version")
            self.vocabulary = checkpoint["vocabulary"]
            self.labels = checkpoint["labels"]
            self.model = TicketTextClassifier(len(self.vocabulary), len(self.labels))
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.eval()
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            raise ModelUnavailable(f"Train the torch model first: {path}") from exc

    @torch.inference_mode()
    def predict(self, text: str) -> ClassificationResponse:
        tokens = encode(text, self.vocabulary).unsqueeze(0)
        # На границе NumPy требуется CPU tensor без графа autograd.
        probabilities = self.model(tokens).softmax(dim=-1)[0].cpu().numpy()
        index = int(probabilities.argmax())
        return ClassificationResponse(
            category=Category(self.labels[index]),
            score=float(probabilities[index]),
            model="torch-embedding-v1",
            probabilities=dict(zip(self.labels, map(float, probabilities), strict=True)),
        )
