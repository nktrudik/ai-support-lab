from pathlib import Path

import numpy as np
import pytest
import torch

from ai_support_lab.classical_ml.inference import SklearnTicketClassifier
from ai_support_lab.classical_ml.pipeline import build_pipeline
from ai_support_lab.deep_learning.dataset import encode
from ai_support_lab.deep_learning.inference import TorchTicketClassifier
from ai_support_lab.deep_learning.model import TicketTextClassifier
from ai_support_lab.tickets.enums import Category
from ai_support_lab.tickets.exceptions import ModelUnavailable


@pytest.mark.parametrize("backend", ["sklearn", "torch"])
def test_trained_inference(model_paths: tuple[Path, Path, Path], backend: str) -> None:
    model = (
        SklearnTicketClassifier(model_paths[0])
        if backend == "sklearn"
        else TorchTicketClassifier(model_paths[1])
    )
    prediction = model.predict("login password rejected authentication token expired")
    assert prediction.category == Category.AUTHENTICATION
    assert prediction.score > 0.5
    assert sum(prediction.probabilities.values()) == pytest.approx(1.0, abs=1e-6)


def test_estimator_fit_transform_and_transform_share_feature_space() -> None:
    pipeline = build_pipeline()
    vectorizer = pipeline.named_steps["tfidf"]
    train = vectorizer.fit_transform(["login password", "invoice refund"])
    test = vectorizer.transform(["unseen words login"])
    assert train.shape[1] == test.shape[1]
    assert test.nnz == 1


def test_torch_padding_backward_and_numpy_bridge() -> None:
    model = TicketTextClassifier(5, 7)
    model.eval()
    tokens = torch.tensor([[2, 3, 0], [2, 3, 0]], dtype=torch.long)
    logits = model(tokens)
    assert logits.shape == (2, 7)
    torch.testing.assert_close(logits[0], logits[1])
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
    loss.backward()
    assert model.embedding.weight.grad is not None
    assert model.embedding.weight.grad[0].abs().sum().item() == 0
    np.testing.assert_equal(logits.detach().numpy().shape, (2, 7))
    assert encode("unknown", {"<pad>": 0, "<unk>": 1})[0].item() == 1


@pytest.mark.parametrize("model", [SklearnTicketClassifier, TorchTicketClassifier])
def test_missing_artifact_is_explicit(tmp_path: Path, model: type) -> None:
    with pytest.raises(ModelUnavailable):
        model(tmp_path / "missing")
