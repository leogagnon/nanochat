"""
Push full_evals JSON results to wandb for any steps not already logged.

For each run in ckpt_root (optionally filtered by sweep_id), this script:
  1. Skips checkpoints that already have a .wandb_done marker.
  2. Queries the wandb API to check if eval data exists at that step.
  3. Pushes missing results and writes the .wandb_done marker.

Usage:
    python -m scripts.push_evals_to_wandb
    python -m scripts.push_evals_to_wandb --sweep-id scaling_law_sanity
    python -m scripts.push_evals_to_wandb --run-id abc123
    python -m scripts.push_evals_to_wandb --ckpt-root /path/to/outputs
"""

import os
import re
import json
import argparse

DEFAULT_CKPT_ROOT = "/home/l/leog/links/scratch/nanochat/logs/outputs"


def get_pushed_steps(api, entity, project, run_id):
    """Return the set of steps that already have eval/core_metric logged in wandb."""
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        rows = run.history(keys=["step", "eval/core_metric"], pandas=False)
        return {int(row["step"]) for row in rows if "eval/core_metric" in row and "step" in row}
    except Exception as e:
        print(f"  Warning: could not fetch history for {run_id}: {e}")
        return set()


def push_eval(run_output_dir, run_id, step, eval_data, meta, wandb_project, wandb_entity):
    import wandb
    wandb_run = wandb.init(
        id=run_id,
        resume="must",
        project=wandb_project,
        entity=wandb_entity,
    )
    wandb_run.define_metric("eval/*", step_metric="step")
    log_data = {
        "step": step,
        "total_flops": meta.get("total_flops"),
        "total_tokens": meta.get("total_tokens"),
        "eval/core_metric": eval_data["core_metric"],
        "eval/bpb_val": eval_data["bpb"]["val"],
        "eval/bpb_train": eval_data["bpb"]["train"],
    }
    for task, score in eval_data["results"].items():
        log_data[f"eval/results/{task}"] = score
    for task, score in eval_data["centered_results"].items():
        log_data[f"eval/centered_results/{task}"] = score
    wandb_run.log(log_data)
    wandb_run.finish()


def process_run(run_id, run_dir, api):
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.isfile(config_path):
        return

    with open(config_path) as f:
        config = json.load(f)
    log_cfg = config.get("log", {})
    wandb_project = log_cfg.get("wandb_project", "nanochat")
    wandb_entity = log_cfg.get("wandb_entity", None)

    evals_dir = os.path.join(run_dir, "full_evals")
    if not os.path.isdir(evals_dir):
        return

    json_files = sorted(
        f for f in os.listdir(evals_dir)
        if re.match(r"step_\d+\.json$", f)
    )
    if not json_files:
        return

    # Find which steps already need no action (have .wandb_done marker)
    pending = []
    for fname in json_files:
        marker = os.path.join(evals_dir, fname.replace(".json", ".wandb_done"))
        if not os.path.isfile(marker):
            step = int(re.search(r"\d+", fname).group())
            pending.append((step, fname, marker))

    if not pending:
        print(f"  {run_id}: all steps already pushed, skipping")
        return

    # Query wandb once per run to get already-pushed steps
    print(f"  {run_id}: checking wandb for {len(pending)} pending step(s)...")
    pushed_steps = get_pushed_steps(api, wandb_entity, wandb_project, run_id)

    for step, fname, marker in pending:
        eval_path = os.path.join(evals_dir, fname)
        if step in pushed_steps:
            print(f"    step {step}: already in wandb, stamping marker")
            open(marker, "w").close()
            continue

        print(f"    step {step}: pushing to wandb...", end=" ", flush=True)
        with open(eval_path) as f:
            eval_data = json.load(f)
        meta_path = os.path.join(run_dir, f"{step:06d}", "meta.json")
        meta = json.load(open(meta_path)) if os.path.isfile(meta_path) else {}
        try:
            push_eval(run_dir, run_id, step, eval_data, meta, wandb_project, wandb_entity)
            open(marker, "w").close()
            print("done")
        except Exception as e:
            print(f"FAILED: {e}")


def main():
    parser = argparse.ArgumentParser(description="Push eval results to wandb")
    parser.add_argument("--sweep-id", default=None, help="Only process runs with this sweep_id")
    parser.add_argument("--run-id", default=None, help="Only process this specific run")
    parser.add_argument("--ckpt-root", default=DEFAULT_CKPT_ROOT)
    args = parser.parse_args()

    import wandb
    api = wandb.Api()

    run_ids = sorted(os.listdir(args.ckpt_root))

    if args.run_id:
        run_ids = [r for r in run_ids if r == args.run_id]

    if args.sweep_id:
        filtered = []
        for run_id in run_ids:
            config_path = os.path.join(args.ckpt_root, run_id, "config.json")
            if not os.path.isfile(config_path):
                continue
            with open(config_path) as f:
                config = json.load(f)
            if config.get("log", {}).get("sweep_id") == args.sweep_id:
                filtered.append(run_id)
        run_ids = filtered

    print(f"Processing {len(run_ids)} run(s) in {args.ckpt_root}")
    for run_id in run_ids:
        run_dir = os.path.join(args.ckpt_root, run_id)
        if not os.path.isdir(run_dir):
            continue
        process_run(run_id, run_dir, api)


if __name__ == "__main__":
    main()
