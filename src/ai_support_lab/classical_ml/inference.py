from pathlib import Path
from typing import Protocol

import joblib

from ai_support_lab.classical_ml.features import top_predictions
from ai_support_lab.tickets.enums import Category
from ai_support_lab.tickets.exceptions import ModelUnavailable
from ai_support_lab.tickets.schemas import ClassificationResponse


class TicketClassifier(Protocol):
    def predict(self, text: str) -> ClassificationResponse: ...


class SklearnTicketClassifier:
    def __init__(self, artifact_path: Path) -> None:
        try:
            # joblib использует pickle: загружаем только локально обученный,
            # доверенный артефакт. Пользователь API не может передать путь к файлу.
            artifact = joblib.load(artifact_path)
            if artifact["version"] != 1:
                raise ValueError("Unsupported model artifact version")
            self.pipeline = artifact["pipeline"]
        except (OSError, ValueError, KeyError) as exc:
            raise ModelUnavailable(f"Train the sklearn model first: {artifact_path}") from exc

    def predict(self, text: str) -> ClassificationResponse:
        probabilities = self.pipeline.predict_proba([text])
        indices, scores = top_predictions(probabilities)
        labels = self.pipeline.classes_
        return ClassificationResponse(
            category=Category(labels[indices[0]]),
            score=float(scores[0]),
            model="sklearn-tfidf-v1",
            probabilities={
                str(label): float(p) for label, p in zip(labels, probabilities[0], strict=True)
            },
        )
