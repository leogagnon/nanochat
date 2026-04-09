"""
Evaluate a checkpoint produced by train.py.

From the root directory, run as:
    python eval.py ckpt=<wandb_id>/<step>
    python eval.py ckpt=<wandb_id>/<step> max_per_task=500

The checkpoint is looked up at <ckpt_root>/<wandb_id>/<step>/.
Results are written to <ckpt_root>/<wandb_id>/full_evals/step_XXXXX.json.
A <ckpt_root>/<wandb_id>/full_evals/step_XXXXX.wandb_done marker is created
after the results are successfully pushed to wandb.
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["NANOCHAT_BASE_DIR"] = os.path.join(os.getcwd(), "cache")
import json

from dataclasses import dataclass
from omegaconf import MISSING, DictConfig, OmegaConf

import hydra
from hydra.core.config_store import ConfigStore


@dataclass
class EvalConfig:
    ckpt: str = MISSING            # format: wandb_id/step (e.g. abc123/001000)
    ckpt_root: str = "/home/l/leog/links/scratch/nanochat/logs/outputs"
    max_per_task: int = -1          # -1 = all examples (full eval)
    eval_tokens: int = 80 * 524288  # tokens per split for bpb eval
    device_batch_size: int = 32
    device_type: str = ""           # "" = autodetect


def _register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="eval_schema", node=EvalConfig)


_register_configs()


def _push_to_wandb(run_output_dir, run_id, step, eval_data, meta):
    """Push eval results from a JSON dict to the pre-training wandb run."""
    import wandb
    config_path = os.path.join(run_output_dir, "config.json")
    if not os.path.isfile(config_path):
        print(f"No config.json found at {config_path}, skipping wandb push")
        return False
    with open(config_path) as f:
        run_config = json.load(f)
    log_cfg = run_config.get("log", {})
    wandb_project = log_cfg.get("wandb_project", "nanochat")
    wandb_entity = log_cfg.get("wandb_entity", None)

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
    return True


@hydra.main(version_base=None, config_path="configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    c: EvalConfig = OmegaConf.to_object(cfg)  # type: ignore[assignment]

    from nanochat.common import print_banner
    print_banner()

    # Parse ckpt: "wandb_id/step"
    ckpt_parts = c.ckpt.split("/")
    if len(ckpt_parts) != 2:
        print(f"Invalid ckpt format: {c.ckpt!r}. Expected wandb_id/step (e.g. abc123/001000)")
        return
    run_id, step_str = ckpt_parts
    step = int(step_str)

    run_output_dir = os.path.join(c.ckpt_root, run_id)
    if not os.path.isdir(run_output_dir):
        print(f"Skipping {c.ckpt}: run directory not found at {run_output_dir}")
        return

    checkpoint_dir = os.path.join(run_output_dir, f"{step:06d}")
    if not os.path.isdir(checkpoint_dir):
        print(f"Skipping {c.ckpt}: checkpoint directory not found at {checkpoint_dir}")
        return

    evals_dir = os.path.join(run_output_dir, "full_evals")
    eval_path = os.path.join(evals_dir, f"step_{step:05d}.json")
    wandb_marker = os.path.join(evals_dir, f"step_{step:05d}.wandb_done")

    # If eval JSON exists but wandb push is missing, push without re-evaluating.
    if os.path.isfile(eval_path) and not os.path.isfile(wandb_marker):
        print(f"{c.ckpt}: eval exists, pushing missing wandb results")
        with open(eval_path) as f:
            eval_data = json.load(f)
        meta_path = os.path.join(checkpoint_dir, "meta.json")
        meta = json.load(open(meta_path)) if os.path.isfile(meta_path) else {}
        try:
            _push_to_wandb(run_output_dir, run_id, step, eval_data, meta)
            open(wandb_marker, "w").close()
            print(f"Wandb push done for {c.ckpt}")
        except Exception as e:
            print(f"Wandb push failed for {c.ckpt}: {e}")
        return

    if os.path.isfile(eval_path):
        print(f"Skipping {c.ckpt}: already evaluated and pushed to wandb")
        return

    # -------------------------------------------------------------------------
    # Full evaluation

    from nanochat.common import compute_init, compute_cleanup, print0, autodetect_device_type
    from nanochat.tokenizer import get_token_bytes
    from nanochat.checkpoint_manager import build_model
    from nanochat.loss_eval import evaluate_bpb
    from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
    from nanochat.engine import Engine
    from scripts.base_eval import evaluate_core

    device_type = autodetect_device_type() if c.device_type == "" else c.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    master_process = ddp_rank == 0

    print0(f"Evaluating {c.ckpt}")

    model, tokenizer, meta = build_model(run_output_dir, step, device, phase="eval")
    token_bytes = get_token_bytes(device=device)
    sequence_len = meta["model_config"]["sequence_len"]

    print0(f"Model loaded: {meta['model_config']}")

    # -------------------------------------------------------------------------
    # CORE evaluation

    print0("\n" + "=" * 80)
    print0("CORE Evaluation")
    print0("=" * 80)
    core_results = evaluate_core(model, tokenizer, device, max_per_task=c.max_per_task)
    print0(f"CORE metric: {core_results['core_metric']:.4f}")

    # -------------------------------------------------------------------------
    # BPB evaluation

    print0("\n" + "=" * 80)
    print0("BPB Evaluation")
    print0("=" * 80)
    tokens_per_step = c.device_batch_size * sequence_len * ddp_world_size
    eval_tokens = (c.eval_tokens // tokens_per_step) * tokens_per_step
    steps = eval_tokens // tokens_per_step

    bpb_results = {}
    for split in ["train", "val"]:
        loader = tokenizing_distributed_data_loader_bos_bestfit(
            tokenizer, c.device_batch_size, sequence_len, split, device=device,
        )
        bpb = evaluate_bpb(model, loader, steps, token_bytes)
        bpb_results[split] = bpb
        print0(f"{split} bpb: {bpb:.6f}")

    # -------------------------------------------------------------------------
    # Sampling

    print0("\n" + "=" * 80)
    print0("Samples")
    print0("=" * 80)
    samples = []
    if master_process:
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(model, tokenizer)
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            sample_str = tokenizer.decode(sample[0])
            print0(sample_str)
            samples.append({"prompt": prompt, "completion": sample_str})

    # -------------------------------------------------------------------------
    # Save results and push to wandb

    if master_process:
        os.makedirs(evals_dir, exist_ok=True)
        eval_data = {
            "run_id": run_id,
            "step": step,
            "max_per_task": c.max_per_task,
            "core_metric": core_results["core_metric"],
            "results": core_results["results"],
            "centered_results": core_results["centered_results"],
            "bpb": bpb_results,
            "samples": samples,
        }
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(eval_data, f, indent=2)
        print0(f"\nResults written to: {eval_path}")

        try:
            _push_to_wandb(run_output_dir, run_id, step, eval_data, meta)
            open(wandb_marker, "w").close()
            print0(f"Eval results logged to wandb run {run_id}")
        except Exception as e:
            print0(f"Wandb push failed for {c.ckpt}: {e}")

    compute_cleanup()


if __name__ == "__main__":
    main()
