from pathlib import Path

import numpy as np
import pandas as pd

from ai_support_lab.tickets.enums import Category

# Несколько формулировок на класс дают модели воспроизводимый сигнал; вариации
# продуктов и срочности не являются прямой копией целевой метки category.
PHRASES = {
    "billing": ["invoice charged twice", "refund payment missing", "subscription billing amount"],
    "authentication": [
        "login password rejected",
        "authentication token expired",
        "two factor sign in failed",
    ],
    "performance": [
        "slow response latency",
        "dashboard takes minutes",
        "request timeout under load",
    ],
    "integration": [
        "webhook delivery failed",
        "api integration disconnected",
        "connector sync stopped",
    ],
    "account": ["change account owner", "update profile email", "workspace membership settings"],
    "bug": [
        "application crash exception",
        "screen error reproducible",
        "button broken after upgrade",
    ],
    "feature_request": [
        "please add export feature",
        "request new dashboard option",
        "suggest custom report support",
    ],
}


def generate_dataset(rows: int = 1400, seed: int = 42) -> pd.DataFrame:
    if rows < 70:
        raise ValueError("Use at least 70 rows to support stratified splits")
    rng = np.random.default_rng(seed)
    categories = np.resize(np.array([c.value for c in Category]), rows)
    rng.shuffle(categories)
    products = rng.choice(["cloud", "desk", "analytics"], size=rows)
    tiers = rng.choice(["free", "pro", "enterprise"], size=rows, p=[0.3, 0.5, 0.2])
    urgent = rng.random(rows) < 0.18
    phrases = [str(rng.choice(PHRASES[str(category)])) for category in categories]
    titles = np.array(phrases)
    descriptions = np.array(
        [
            f"{phrase}. Observed in {product}; {rng.choice(['since yesterday', 'after deployment', 'for our team'])}."
            for phrase, product in zip(phrases, products, strict=True)
        ],
        dtype=object,
    )
    descriptions[urgent] += " Production outage; all users blocked."
    priority = np.full(rows, "medium", dtype=object)
    priority[categories == "feature_request"] = "low"
    priority[tiers == "enterprise"] = "high"
    priority[urgent] = "critical"
    # Булевы маски задают понятную бизнес-зависимость. Broadcasting умножает
    # вектор базового времени на коэффициенты обслуживания, без Python-цикла.
    hours = rng.lognormal(mean=2.0, sigma=0.4, size=rows)
    hours *= np.where(tiers == "enterprise", 0.6, 1.0)
    hours *= np.where(urgent, 0.4, 1.0)
    return pd.DataFrame(
        {
            "ticket_id": np.arange(1, rows + 1),
            "title": titles,
            "description": descriptions,
            "product": products,
            "customer_tier": tiers,
            "country": rng.choice(["DE", "FR", "GB"], size=rows),
            "category": categories,
            "priority": priority,
            "resolution_hours": hours.round(2),
        }
    )


def load_dataset(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path).dropna(subset=["title", "category"]).copy()
    frame["description"] = frame["description"].fillna("")
    # .apply уместен для небольшой пользовательской нормализации строк; числовую
    # обработку оставляем векторной, чтобы не терять преимущества pandas/NumPy.
    frame["title"] = frame["title"].apply(lambda text: " ".join(str(text).split()))
    frame["text"] = frame["title"] + " " + frame["description"]
    return frame


def dataset_report(frame: pd.DataFrame) -> dict[str, object]:
    summary = (
        frame.groupby("category")
        .agg(count=("ticket_id", "count"), mean_hours=("resolution_hours", "mean"))
        .reset_index()
    )
    counts = frame.loc[frame["customer_tier"] == "enterprise"].groupby("category").size()
    enterprise = counts.rename("enterprise_count").reset_index()
    summary = summary.merge(enterprise, on="category", how="left").fillna({"enterprise_count": 0})
    summary = summary.sort_values("mean_hours", ascending=False)
    return {
        "by_category": summary.to_dict(orient="records"),
        "priorities": frame["priority"].value_counts().to_dict(),
        "slowest_category": str(summary.iloc[0]["category"]),
    }
