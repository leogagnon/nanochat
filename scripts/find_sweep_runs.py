"""
Find all wandb run IDs that belong to a given sweep ID.

Usage:
    python -m scripts.find_sweep_runs <sweep_id>
    python -m scripts.find_sweep_runs scaling_law_sanity
"""
import os
import json
import argparse

OUTPUTS_DIR = os.path.join("logs", "outputs")


def main():
    parser = argparse.ArgumentParser(description="Find run IDs by sweep ID")
    parser.add_argument("sweep_id", help="Value of log.sweep_id to search for")
    args = parser.parse_args()

    if not os.path.isdir(OUTPUTS_DIR):
        print(f"No outputs directory found at {OUTPUTS_DIR}")
        return

    matches = []
    for run_id in sorted(os.listdir(OUTPUTS_DIR)):
        config_path = os.path.join(OUTPUTS_DIR, run_id, "config.json")
        if not os.path.isfile(config_path):
            continue
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        if config.get("log", {}).get("sweep_id") == args.sweep_id:
            matches.append(run_id)

    for run_id in matches:
        print(run_id)

    if not matches:
        print(f"No runs found with sweep_id={args.sweep_id!r}")


if __name__ == "__main__":
    main()
