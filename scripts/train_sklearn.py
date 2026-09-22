import argparse
import json
from pathlib import Path

from ai_support_lab.classical_ml.train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/tickets.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/sklearn.joblib"))
    args = parser.parse_args()
    print(json.dumps(train(args.data, args.output), indent=2))
