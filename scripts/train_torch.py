import argparse
import json
from pathlib import Path

from ai_support_lab.deep_learning.training import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/tickets.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/torch.pt"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(train(args.data, args.output, args.epochs, args.device), indent=2))
