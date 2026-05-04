"""
Find all checkpoints for a given sweep ID, formatted as wandb_id/step.

Usage:
    python -m scripts.find_sweep_checkpoints <sweep_id> [--ckpt-root <path>] [--filter key=value ...]

Outputs one ckpt per line in the format: wandb_id/step
"""
import os
import json
import argparse
from typing import Any

DEFAULT_CKPT_ROOT = "/home/l/leog/links/scratch/nanochat/logs/outputs"
MISSING = object()


def parse_filters(raw_filters: list[str]) -> list[tuple[str, Any]]:
    parsed: list[tuple[str, Any]] = []
    for raw in raw_filters:
        if "=" not in raw:
            raise ValueError(f"Invalid filter {raw!r}. Expected key=value.")
        key, value_str = raw.split("=", 1)
        if not key:
            raise ValueError(f"Invalid filter {raw!r}. Key cannot be empty.")
        try:
            value: Any = json.loads(value_str)
        except json.JSONDecodeError:
            value = value_str
        parsed.append((key, value))
    return parsed


def get_nested_value(data: dict[str, Any], dotted_key: str) -> Any:
    cur: Any = data
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return MISSING
        cur = cur[part]
    return cur


def main():
    parser = argparse.ArgumentParser(description="Find all checkpoints for a sweep")
    parser.add_argument("sweep_id", help="Value of log.sweep_id to search for")
    parser.add_argument("--ckpt-root", default=DEFAULT_CKPT_ROOT, help="Root directory for checkpoints")
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional run filter on config.json fields (supports dotted keys, e.g. model.n_layer=16)",
    )
    args = parser.parse_args()

    try:
        filters = parse_filters(args.filter)
    except ValueError as e:
        parser.error(str(e))

    if not os.path.isdir(args.ckpt_root):
        print(f"ckpt_root not found: {args.ckpt_root}", flush=True)
        return

    for run_id in sorted(os.listdir(args.ckpt_root)):
        config_path = os.path.join(args.ckpt_root, run_id, "config.json")
        if not os.path.isfile(config_path):
            continue
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        if config.get("log", {}).get("sweep_id") != args.sweep_id:
            continue
        if any(get_nested_value(config, key) != value for key, value in filters):
            continue

        run_dir = os.path.join(args.ckpt_root, run_id)
        step_dirs = [
            d for d in os.listdir(run_dir)
            if os.path.isdir(os.path.join(run_dir, d)) and d.isdigit()
        ]
        for step in sorted(int(d) for d in step_dirs):
            print(f"{run_id}/{step:06d}")


if __name__ == "__main__":
    main()
