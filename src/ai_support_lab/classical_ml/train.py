import json
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from ai_support_lab.classical_ml.dataset import load_dataset
from ai_support_lab.classical_ml.pipeline import build_pipeline


def train(data_path: Path, artifact_path: Path) -> dict[str, object]:
    frame = load_dataset(data_path)
    train_rows, test_rows = train_test_split(
        frame, test_size=0.2, stratify=frame["category"], random_state=42
    )
    model = build_pipeline()
    model.fit(train_rows["text"], train_rows["category"])
    predicted = model.predict(test_rows["text"])
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_rows["category"],
        predicted,
        average="macro",
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(test_rows["category"], predicted)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "labels": model.classes_.tolist(),
        "confusion_matrix": confusion_matrix(
            test_rows["category"], predicted, labels=model.classes_
        ).tolist(),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"version": 1, "pipeline": model}, artifact_path)
    artifact_path.with_suffix(".metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics
