import argparse
import json
from pathlib import Path

from ai_support_lab.classical_ml.dataset import dataset_report, generate_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1400)
    parser.add_argument("--output", type=Path, default=Path("data/tickets.csv"))
    args = parser.parse_args()
    frame = generate_dataset(args.rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(json.dumps(dataset_report(frame), indent=2))
