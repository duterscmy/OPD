#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from opd.config import load_config
from opd.data import load_training_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview mapped OPD training examples")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--num-examples", type=int, default=3)
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    dataset = load_training_dataset(cfg)
    print(f"rows={len(dataset)} columns={dataset.column_names}")
    for i in range(min(args.num_examples, len(dataset))):
        print("=" * 100)
        print(json.dumps(dataset[i], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
