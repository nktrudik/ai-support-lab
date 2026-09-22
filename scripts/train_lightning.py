import argparse
from pathlib import Path

from ai_support_lab.deep_learning.lightning_module import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/tickets.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/lightning.pt"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--accelerator", default="cpu")
    args = parser.parse_args()
    print(train(args.data, args.output, args.epochs, args.accelerator))
