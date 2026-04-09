"""
Find all checkpoints for a given sweep ID, formatted as wandb_id/step.

Usage:
    python -m scripts.find_sweep_checkpoints <sweep_id> [--ckpt-root <path>]

Outputs one ckpt per line in the format: wandb_id/step
"""
import os
import json
import argparse

DEFAULT_CKPT_ROOT = "/home/l/leog/links/scratch/nanochat/logs/outputs"


def main():
    parser = argparse.ArgumentParser(description="Find all checkpoints for a sweep")
    parser.add_argument("sweep_id", help="Value of log.sweep_id to search for")
    parser.add_argument("--ckpt-root", default=DEFAULT_CKPT_ROOT, help="Root directory for checkpoints")
    args = parser.parse_args()

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

        run_dir = os.path.join(args.ckpt_root, run_id)
        step_dirs = [
            d for d in os.listdir(run_dir)
            if os.path.isdir(os.path.join(run_dir, d)) and d.isdigit()
        ]
        for step in sorted(int(d) for d in step_dirs):
            print(f"{run_id}/{step:06d}")


if __name__ == "__main__":
    main()
