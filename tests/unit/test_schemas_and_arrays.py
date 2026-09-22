import numpy as np
import pytest
from pydantic import ValidationError

from ai_support_lab.classical_ml.dataset import dataset_report, generate_dataset
from ai_support_lab.classical_ml.features import row_softmax, top_predictions
from ai_support_lab.config import Settings
from ai_support_lab.tickets.schemas import Customer, TicketCreate, TicketUpdate


@pytest.mark.parametrize("payload", [{}, {"title": None}, {"title": "a"}, {"unexpected": 1}])
def test_patch_rejects_ambiguous_or_invalid_fields(payload: dict) -> None:
    with pytest.raises(ValidationError):
        TicketUpdate.model_validate(payload)


def test_schema_normalizes_product_and_serializes_nested_customer() -> None:
    ticket = TicketCreate(
        title="Valid title",
        description="Valid description",
        product=" Cloud ",
        customer=Customer(name="Example"),
    )
    assert ticket.product == "cloud"
    assert ticket.model_dump()["customer"]["tier"] == "free"


def test_stable_softmax_shapes_and_advanced_indexing() -> None:
    logits = np.array([[1000.0, 1001.0, 1002.0], [1.0, -4.0, 0.0]])
    probabilities = row_softmax(logits)
    indices, scores = top_predictions(probabilities)
    assert probabilities.shape == (2, 3)
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(2))
    np.testing.assert_array_equal(indices, [2, 0])
    np.testing.assert_allclose(scores, probabilities.max(axis=1))


def test_shape_validation() -> None:
    with pytest.raises(ValueError):
        row_softmax(np.array([1.0, 2.0]))


def test_dataset_determinism_and_vectorized_features() -> None:
    frame = generate_dataset(140)
    assert frame.equals(generate_dataset(140))
    assert len(frame["category"].unique()) == 7
    hours = frame["resolution_hours"].to_numpy()
    assert hours.shape == (140,)
    assert (hours > 0).all()
    assert len(dataset_report(frame)["by_category"]) == 7


def test_remote_configuration_requires_url() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, llm_backend="vllm")
