import numpy as np
from numpy.typing import NDArray


def row_softmax(logits: NDArray[np.floating]) -> NDArray[np.float64]:
    if logits.ndim != 2 or logits.shape[1] == 0 or not np.isfinite(logits).all():
        raise ValueError("Expected finite logits with shape [batch, classes]")
    values = logits.astype(np.float64)
    # keepdims оставляет форму [B, 1]. Broadcasting вычитает максимум отдельно
    # для каждой строки и предотвращает переполнение exp при больших logits.
    exponentials = np.exp(values - values.max(axis=1, keepdims=True))
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def top_predictions(
    probabilities: NDArray[np.floating],
) -> tuple[NDArray[np.int64], NDArray[np.floating]]:
    if probabilities.ndim != 2 or probabilities.shape[1] == 0:
        raise ValueError("Expected probabilities with shape [batch, classes]")
    indices = probabilities.argmax(axis=1)
    return indices, probabilities[np.arange(len(indices)), indices]
