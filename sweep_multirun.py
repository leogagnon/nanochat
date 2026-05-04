"""
Materialize a sweep (grid + per-run random samples), then launch one Hydra multirun.

This mirrors the intent of:
  python train.py --multirun hydra/sweeper/params=sweep
but guarantees independent random optimizer sampling for each run.
"""

from __future__ import annotations

import argparse
import itertools
import math
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
import os
import yaml


DEFAULT_GRID_AXES: dict[str, list[Any]] = {
    "horizon.target_flops": [1e18, 2.15e18, 4.64e18, 1e19],
    "model.depth": [10, 12, 14, 16, 18, 20],
}

DEFAULT_FIXED_VALUES: dict[str, Any] = {
    "horizon.target_param_data_ratio": -1,
    "eval.core_metric_every": -1,
    "eval.core_metric_max_per_task": -1,
    "eval.sample_every": -1,
    "log.save_every": -1,
    "log.num_checkpoints": 4,
}

# kind, lo, hi
DEFAULT_RANDOM_SPECS: dict[str, tuple[str, float, float]] = {
    "optim.embedding_lr": ("uniform", 0.1, 1.0),
    "optim.unembedding_lr": ("uniform", 0.004, 0.04),
    "optim.weight_decay": ("uniform", 0.05, 0.5),
    "optim.matrix_lr": ("uniform", 0.005, 0.05),
    "optim.scalar_lr": ("uniform", 0.1, 1.0),
    "optim.warmdown_ratio": ("uniform", 0.2, 0.85),
    "optim.final_lr_frac": ("uniform", 0.005, 0.15),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute sweep runs and launch a single Hydra multirun.")
    p.add_argument("--out-dir", default=os.environ["SCRATCH"], help="Directory for generated config groups.")
    p.add_argument("--group-name", default="precomputed_run", help="Hydra config group name to generate.")
    p.add_argument("--repeats", type=int, default=2, help="Independent random trials per grid point.")
    p.add_argument("--seed", type=int, default=0, help="Optional RNG seed for reproducible sampling.")
    p.add_argument("--sweep-id", default="sweep_2", help="Value to write to log.sweep_id.")
    p.add_argument("--wandb-project", default="nanochat", help="Value to write to log.wandb_project.")
    p.add_argument("--print-only", action="store_true", help="Only print planned runs, do not launch train.py.")
    p.add_argument(
        "--keep-generated-configs",
        action="store_true",
        help="Do not delete generated precomputed configs after launch (legacy flag).",
    )
    p.add_argument(
        "--cleanup-generated-configs",
        action="store_true",
        help=(
            "Delete generated precomputed configs after train.py exits. "
            "Unsafe for asynchronous launchers like submitit/tamia because jobs may start after submission."
        ),
    )
    p.add_argument(
        "extra_overrides",
        nargs="*",
        help="Extra Hydra overrides forwarded to train.py (e.g. hydra.launcher.array_parallelism=64).",
    )
    return p.parse_args()


def set_nested(d: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur = d
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def sample_value(spec: tuple[str, float, float], rng: random.Random) -> Any:
    kind, lo, hi = spec
    if kind == "uniform":
        return rng.uniform(lo, hi)
    if kind == "loguniform":
        return math.exp(rng.uniform(math.log(lo), math.log(hi)))
    if kind == "randint":
        return rng.randint(int(lo), int(hi))
    raise ValueError(f"Unsupported random spec kind: {kind}")


def _launcher_override(extra_overrides: list[str]) -> str | None:
    for ov in extra_overrides:
        if ov.startswith("hydra/launcher="):
            return ov.split("=", 1)[1]
    return None


def main() -> None:
    args = parse_args()
    if args.repeats <= 0:
        raise ValueError(f"--repeats must be > 0, got {args.repeats}")

    repo_root = Path(__file__).resolve().parent
    rng = random.Random(args.seed)

    grid_keys = list(DEFAULT_GRID_AXES.keys())
    combos = list(itertools.product(*(DEFAULT_GRID_AXES[k] for k in grid_keys)))

    generated_root = (repo_root / args.out_dir / "hardcoded_sweep" / args.group_name).resolve()
    if generated_root.parent.exists():
        shutil.rmtree(generated_root.parent)
    generated_root.mkdir(parents=True, exist_ok=True)

    run_names: list[str] = []
    run_summaries: list[dict[str, Any]] = []
    run_idx = 0
    for combo in combos:
        for _ in range(args.repeats):
            run_cfg: dict[str, Any] = {}
            for k, v in DEFAULT_FIXED_VALUES.items():
                set_nested(run_cfg, k, v)
            set_nested(run_cfg, "log.sweep_id", args.sweep_id)
            set_nested(run_cfg, "log.wandb_project", args.wandb_project)
            for k, v in zip(grid_keys, combo):
                set_nested(run_cfg, k, v)
            for k, spec in DEFAULT_RANDOM_SPECS.items():
                set_nested(run_cfg, k, sample_value(spec, rng))

            name = f"run_{run_idx:04d}"
            run_names.append(name)
            summary: dict[str, Any] = {
                "log.sweep_id": args.sweep_id,
                "log.wandb_project": args.wandb_project,
            }
            for k, v in DEFAULT_FIXED_VALUES.items():
                summary[k] = v
            for k, v in zip(grid_keys, combo):
                summary[k] = v
            for k in DEFAULT_RANDOM_SPECS.keys():
                # Value already written to run_cfg
                cur = run_cfg
                for part in k.split("."):
                    cur = cur[part]
                summary[k] = cur
            run_summaries.append(summary)
            rendered = "# @package _global_\n" + yaml.safe_dump(run_cfg, sort_keys=False)
            (generated_root / f"{name}.yaml").write_text(rendered, encoding="utf-8")
            run_idx += 1

    print(f"[sweep_multirun] grid_axes={{{', '.join(f'{k}:{len(v)}' for k, v in DEFAULT_GRID_AXES.items())}}}")
    print(f"[sweep_multirun] repeats_per_grid_point={args.repeats}")
    print(f"[sweep_multirun] random_keys={list(DEFAULT_RANDOM_SPECS.keys())}")
    print(f"[sweep_multirun] generated_runs={len(run_names)}")
    print(f"[sweep_multirun] generated_group_dir={generated_root}")
    print("[sweep_multirun] runs:")
    ordered_keys = [
        "horizon.target_flops",
        "model.depth",
        "horizon.target_param_data_ratio",
        "eval.core_metric_every",
        "eval.core_metric_max_per_task",
        "eval.sample_every",
        "log.save_every",
        "log.num_checkpoints",
        "optim.embedding_lr",
        "optim.unembedding_lr",
        "optim.weight_decay",
        "optim.matrix_lr",
        "optim.scalar_lr",
        "optim.warmup_steps",
        "optim.warmdown_ratio",
        "optim.final_lr_frac",
        "log.sweep_id",
        "log.wandb_project",
    ]
    for name, summary in zip(run_names, run_summaries):
        parts = [f"{k}={summary[k]}" for k in ordered_keys if k in summary]
        print(f"  {name}: " + " ".join(parts))

    if args.print_only:
        return

    launcher = _launcher_override(args.extra_overrides)
    using_local_launcher = launcher == "basic"
    should_cleanup = False
    if args.keep_generated_configs:
        should_cleanup = False
    elif args.cleanup_generated_configs:
        should_cleanup = True
    else:
        # Default to preserving configs unless we're explicitly using Hydra's local launcher.
        # For submitit/tamia, train.py returns after submission and jobs may still need these files.
        should_cleanup = using_local_launcher

    cmd = [
        sys.executable,
        "train.py",
        "--multirun",
        "--config-dir",
        str(generated_root.parent),
        f"+{args.group_name}={','.join(run_names)}",
        *args.extra_overrides,
    ]
    print("[sweep_multirun] launching:", " ".join(cmd))
    if not should_cleanup:
        print(f"[sweep_multirun] keeping generated configs: {generated_root.parent}")
    try:
        subprocess.run(cmd, check=True, cwd=repo_root)
    finally:
        if should_cleanup and generated_root.parent.exists():
            shutil.rmtree(generated_root.parent, ignore_errors=True)
            print(f"[sweep_multirun] cleaned generated configs: {generated_root.parent}")


if __name__ == "__main__":
    main()
