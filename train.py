"""
Train base model with Hydra + structured config.

From the root directory, run as:
    python train.py
    python train.py model=tiny  # CPU-friendly
    python train.py model=small log.run=my_run
    torchrun --nproc_per_node=8 train.py model=base log.run=my_run

Override any field inline:
    python train.py model.depth=12 optim.device_batch_size=16
    python train.py horizon.num_iterations=100 eval.core_metric_every=-1

Equivalent to the old argparse invocation:
    python -m scripts.base_train --depth=4 --max-seq-len=512 --device-batch-size=1 \\
        --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --num-iterations=20
becomes:
    python train.py model=tiny optim.device_batch_size=1 eval.eval_tokens=512 \\
        eval.core_metric_every=-1 optim.total_batch_size=512 horizon.num_iterations=20
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["NANOCHAT_BASE_DIR"] = os.path.join(os.getcwd(), "cache")
import gc
import json
import time
import math
from dataclasses import asdict
from contextlib import contextmanager

from dataclasses import dataclass, field
from typing import Any, Optional

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

import hydra_plugins.lazy_sweeper  # registers hydra/sweeper=lazy_basic

OmegaConf.register_new_resolver("uniform", lambda lo, hi: __import__("random").uniform(float(lo), float(hi)))
OmegaConf.register_new_resolver("randint", lambda lo, hi: __import__("random").randint(int(lo), int(hi)))




@dataclass
class ModelConfig:
    depth: int = 20
    aspect_ratio: int = 64
    head_dim: int = 128
    max_seq_len: int = 2048
    window_pattern: str = "SSSL"


@dataclass
class HorizonConfig:
    """Training horizon: only one of num_iterations / target_flops / target_tokens / target_param_data_ratio / target_time_minutes is used (in that order)."""
    num_iterations: int = -1
    target_flops: float = -1.0
    target_tokens: int = -1
    target_param_data_ratio: float = 12.0
    target_time_minutes: float = -1.0


@dataclass
class OptimConfig:
    device_batch_size: int = 32
    total_batch_size: int = -1  # -1 = auto-compute optimal
    embedding_lr: float = 0.3
    unembedding_lr: float = 0.008
    weight_decay: float = 0.28
    matrix_lr: float = 0.02
    scalar_lr: float = 0.5
    warmup_steps: int = 40
    warmdown_ratio: float = 0.65
    final_lr_frac: float = 0.05
    resume_from_step: int = -1  # -1 = fresh run


@dataclass
class Fp8Config:
    enabled: bool = False
    recipe: str = "tensorwise"  # tensorwise | rowwise


@dataclass
class EvalConfig:
    eval_every: int = 250
    eval_tokens: int = 80 * 524288  # 41_943_040
    core_metric_every: int = 2000
    core_metric_max_per_task: int = 500
    sample_every: int = 2000


@dataclass
class LogConfig:
    save_every: int = -1        # -1 = only save at end
    num_checkpoints: int = -1  # -1 = disabled; N = save at N equal intervals including the last step
    model_tag: Optional[str] = None  # override checkpoint directory name
    run: Optional[str] = None  # wandb run name; "dummy" disables wandb, "" enables with auto-generated name
    resume_run_id: Optional[str] = None  # resume an existing wandb run by ID (also resumes its output directory)
    wandb_project: str = "nanochat"  # wandb project name
    wandb_entity: Optional[str] = None  # wandb entity/team name
    sweep_id: Optional[str] = None
    output_dir: str = "logs/outputs"  # base directory for run outputs and checkpoints

@dataclass
class RuntimeConfig:
    device_type: str = ""  # "" = autodetect, or cuda | cpu | mps


@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    horizon: HorizonConfig = field(default_factory=HorizonConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    fp8: Fp8Config = field(default_factory=Fp8Config)
    eval: EvalConfig = field(default_factory=EvalConfig)
    log: LogConfig = field(default_factory=LogConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def _register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="train_schema", node=TrainConfig)


_register_configs()


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    # If resuming an existing run, replace cfg with the saved config.
    # Only resume_run_id and optim.resume_from_step are taken from the current invocation.
    resume_run_id = OmegaConf.select(cfg, "log.resume_run_id")
    if resume_run_id:
        run_output_dir = os.path.join(OmegaConf.select(cfg, "log.output_dir", default="logs/outputs"), resume_run_id)
        config_path = os.path.join(run_output_dir, "config.json")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"No config.json found for resume_run_id={resume_run_id} at {config_path}")
        with open(config_path) as f:
            saved_config = json.load(f)
        saved_config.pop("computed", None)
        resume_from_step = OmegaConf.select(cfg, "optim.resume_from_step", default=-1)
        # Merge saved config on top of the already-structured cfg (saved values override train.yaml defaults)
        cfg = OmegaConf.merge(cfg, OmegaConf.create(saved_config))
        OmegaConf.update(cfg, "log.resume_run_id", resume_run_id)

        # Auto-detect last checkpoint step if resume_from_step not explicitly provided
        if resume_from_step == -1:
            step_dirs = [d for d in os.listdir(run_output_dir) if d.isdigit()]
            if step_dirs:
                resume_from_step = max(int(d) for d in step_dirs)
                print(f"Auto-detected resume_from_step={resume_from_step} for run {resume_run_id}")
        OmegaConf.update(cfg, "optim.resume_from_step", resume_from_step)

    # Convert to structured config for type safety and attribute access
    c: TrainConfig = OmegaConf.to_object(cfg)  # type: ignore[assignment]

    import wandb
    import torch
    import torch.distributed as dist

    from nanochat.gpt import GPT, GPTConfig, Linear
    from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit
    from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type, get_peak_flops, COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, is_ddp_initialized
    from nanochat.tokenizer import get_tokenizer, get_token_bytes
    from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
    from nanochat.loss_eval import evaluate_bpb
    from nanochat.engine import Engine
    from nanochat.flash_attention import HAS_FA3
    from scripts.base_eval import evaluate_core
    print_banner()

    # keep a plain dict for wandb / checkpoint logging (mirrors old user_config)
    user_config = OmegaConf.to_container(cfg, resolve=True)

    # -------------------------------------------------------------------------
    # Runtime / distributed setup

    device_type = autodetect_device_type() if c.runtime.device_type == "" else c.runtime.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    master_process = ddp_rank == 0
    synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
    get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

    if device_type == "cuda":
        gpu_device_name = torch.cuda.get_device_name(0)
        gpu_peak_flops = get_peak_flops(gpu_device_name)
        print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
    else:
        gpu_peak_flops = float('inf')
    print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")

    use_dummy_wandb = c.log.run == "dummy" or not master_process
    if use_dummy_wandb:
        wandb_run = DummyWandb()
    elif c.log.resume_run_id is not None:
        wandb_run = wandb.init(
            id=c.log.resume_run_id,
            resume="must",
            project=c.log.wandb_project,
            entity=c.log.wandb_entity,
        )
    else:
        wandb_run = wandb.init(
            project=c.log.wandb_project,
            entity=c.log.wandb_entity,
            name=c.log.run,
            config=user_config,
        )
    
    # Extract run ID for checkpoint directory naming, broadcast from rank 0 to all ranks
    if master_process:
        run_id = "DUMMY" if use_dummy_wandb else wandb_run.id
    else:
        run_id = None
    if is_ddp_initialized():
        run_id_list = [run_id]
        dist.broadcast_object_list(run_id_list, src=0)
        run_id = run_id_list[0]
    print0(f"Run ID: {run_id}")

    from nanochat.flash_attention import USE_FA3
    using_fa3 = USE_FA3
    if using_fa3:
        print0("✓ Using Flash Attention 3 (Hopper GPU detected), efficient, new and awesome.")
    else:
        print0("!" * 80)
        if HAS_FA3 and COMPUTE_DTYPE != torch.bfloat16:
            print0(f"WARNING: Flash Attention 3 only supports bf16, but COMPUTE_DTYPE={COMPUTE_DTYPE}. Using PyTorch SDPA fallback")
        else:
            print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback")
        print0("WARNING: Training will be less efficient without FA3")
        if c.model.window_pattern != "L":
            print0(f"WARNING: SDPA has no support for sliding window attention (window_pattern='{c.model.window_pattern}'). Your GPU utilization will be terrible.")
            print0("WARNING: Recommend using model.window_pattern=L for full context attention without alternating sliding window patterns.")
        print0("!" * 80)

    # -------------------------------------------------------------------------
    # Tokenizer

    tokenizer = get_tokenizer()
    token_bytes = get_token_bytes(device=device)
    vocab_size = tokenizer.get_vocab_size()
    print0(f"Vocab size: {vocab_size:,}")

    # -------------------------------------------------------------------------
    # Model

    def build_model_meta(depth):
        base_dim = depth * c.model.aspect_ratio
        model_dim = ((base_dim + c.model.head_dim - 1) // c.model.head_dim) * c.model.head_dim
        num_heads = model_dim // c.model.head_dim
        config = GPTConfig(
            sequence_len=c.model.max_seq_len, vocab_size=vocab_size,
            n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
            window_pattern=c.model.window_pattern,
        )
        with torch.device("meta"):
            model_meta = GPT(config)
        return model_meta

    model = build_model_meta(c.model.depth)
    model_config = model.config
    model_config_kwargs = asdict(model_config)
    print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")
    model.to_empty(device=device)
    model.init_weights()

    run_output_dir = os.path.join(c.log.output_dir, run_id)
    checkpoint_dir = run_output_dir
    if master_process:
        os.makedirs(run_output_dir, exist_ok=True)
        with open(os.path.join(run_output_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(user_config, f, indent=2)
    print0(f"Run output directory: {run_output_dir}")
    
    resuming = c.optim.resume_from_step != -1
    if resuming:
        print0(f"Resuming optimization from step {c.optim.resume_from_step}")
        model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, c.optim.resume_from_step, device, load_optimizer=True, rank=ddp_rank)
        model.load_state_dict(model_data, strict=True, assign=True)
        del model_data

    # -------------------------------------------------------------------------
    # FP8

    if c.fp8.enabled:
        if device_type != "cuda":
            print0("Warning: FP8 training requires CUDA, ignoring fp8.enabled")
        else:
            from nanochat.fp8 import Float8LinearConfig, convert_to_float8_training
            import torch.nn as nn

            def fp8_module_filter(mod: nn.Module, fqn: str) -> bool:
                if not isinstance(mod, nn.Linear):
                    return False
                if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                    return False
                if min(mod.in_features, mod.out_features) < 128:
                    return False
                return True

            fp8_config = Float8LinearConfig.from_recipe_name(c.fp8.recipe)
            num_linear = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
            convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_module_filter)
            num_fp8 = sum(1 for m in model.modules() if 'Float8' in type(m).__name__)
            num_skipped = num_linear - num_fp8
            print0(f"✓ FP8 training enabled ({c.fp8.recipe} scaling) - converted {num_fp8}/{num_linear} linear layers, skipped {num_skipped} (too small)")

    @contextmanager
    def disable_fp8(model):
        """Temporarily swap Float8Linear modules with nn.Linear for BF16 evaluation."""
        import torch.nn as nn

        fp8_locations = []
        for name, module in model.named_modules():
            if 'Float8' in type(module).__name__:
                if '.' in name:
                    parent_name, attr_name = name.rsplit('.', 1)
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model
                    attr_name = name
                fp8_locations.append((parent, attr_name, module))

        if not fp8_locations:
            yield
            return

        for parent, attr_name, fp8_module in fp8_locations:
            linear = Linear(
                fp8_module.in_features, fp8_module.out_features,
                bias=fp8_module.bias is not None,
                device="meta",
                dtype=fp8_module.weight.dtype,
            )
            linear.weight = fp8_module.weight
            if fp8_module.bias is not None:
                linear.bias = fp8_module.bias
            setattr(parent, attr_name, linear)

        try:
            yield
        finally:
            for parent, attr_name, fp8_module in fp8_locations:
                setattr(parent, attr_name, fp8_module)

    # -------------------------------------------------------------------------
    # Compile

    orig_model = model
    model = torch.compile(model, dynamic=False)

    # -------------------------------------------------------------------------
    # Scaling laws

    param_counts = model.num_scaling_params()
    print0(f"Parameter counts:")
    for key, value in param_counts.items():
        print0(f"{key:24s}: {value:,}")
    num_params = param_counts['total']
    num_flops_per_token = model.estimate_flops()
    print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

    def get_scaling_params(m):
        params_counts = m.num_scaling_params()
        return params_counts['transformer_matrices'] + params_counts['lm_head']

    num_scaling_params = get_scaling_params(model)
    target_tokens = int(c.horizon.target_param_data_ratio * num_scaling_params)

    d12_ref = build_model_meta(12)
    D_REF = c.horizon.target_param_data_ratio * get_scaling_params(d12_ref)
    B_REF = 2**19

    total_batch_size = c.optim.total_batch_size
    if total_batch_size == -1:
        batch_size_ratio = target_tokens / D_REF
        predicted_batch_size = B_REF * batch_size_ratio ** 0.383
        total_batch_size = 2 ** round(math.log2(predicted_batch_size))
        print0(f"Auto-computed optimal batch size: {total_batch_size:,} tokens")

    batch_lr_scale = 1.0
    batch_ratio = total_batch_size / B_REF
    if batch_ratio != 1.0:
        batch_lr_scale = batch_ratio ** 0.5
        print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,} (reference: {B_REF:,})")

    weight_decay_scaled = c.optim.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
    if weight_decay_scaled != c.optim.weight_decay:
        print0(f"Scaling weight decay from {c.optim.weight_decay:.6f} to {weight_decay_scaled:.6f} for depth {c.model.depth}")

    # -------------------------------------------------------------------------
    # Optimizer

    optimizer = model.setup_optimizer(
        unembedding_lr=c.optim.unembedding_lr * batch_lr_scale,
        embedding_lr=c.optim.embedding_lr * batch_lr_scale,
        scalar_lr=c.optim.scalar_lr * batch_lr_scale,
        matrix_lr=c.optim.matrix_lr * batch_lr_scale,
        weight_decay=weight_decay_scaled,
    )

    if resuming:
        optimizer.load_state_dict(optimizer_data)
        del optimizer_data

    scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
    if scaler is not None:
        print0("GradScaler enabled for fp16 training")

    # -------------------------------------------------------------------------
    # DataLoaders

    dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
    train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tokenizer, c.optim.device_batch_size, c.model.max_seq_len,
        split="train", device=device, resume_state_dict=dataloader_resume_state_dict,
    )
    build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, c.optim.device_batch_size, c.model.max_seq_len, split="val", device=device,
    )
    x, y, dataloader_state_dict = next(train_loader)

    # -------------------------------------------------------------------------
    # Training horizon

    assert c.horizon.num_iterations > 0 or c.horizon.target_tokens > 0 or c.horizon.target_param_data_ratio > 0 or c.horizon.target_flops > 0 or c.horizon.target_time_minutes > 0
    if c.horizon.num_iterations > 0:
        num_iterations = c.horizon.num_iterations
        print0(f"Using user-provided number of iterations: {num_iterations:,}")
    elif c.horizon.target_flops > 0:
        num_iterations = round(c.horizon.target_flops / (num_flops_per_token * total_batch_size))
        print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
    elif c.horizon.target_tokens > 0:
        num_iterations = c.horizon.target_tokens // total_batch_size
        print0(f"Calculated number of iterations from target tokens: {num_iterations:,}")
    elif c.horizon.target_param_data_ratio > 0:
        num_iterations = target_tokens // total_batch_size
        print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
    elif c.horizon.target_time_minutes > 0:
        num_iterations = 10_000_000  # placeholder; recomputed after warmup once throughput is known
        print0(f"Target time: {c.horizon.target_time_minutes}min — num_iterations will be set after warmup")
    else:
        raise ValueError("No training horizon specified")

    total_tokens = total_batch_size * num_iterations
    print0(f"Total number of training tokens: {total_tokens:,}")

    # Pre-compute checkpoint steps for num_checkpoints mode
    checkpoint_steps = set()
    if c.log.num_checkpoints > 0:
        checkpoint_steps = {round(num_iterations * i / c.log.num_checkpoints) for i in range(1, c.log.num_checkpoints + 1)}
        print0(f"Checkpoint steps (num_checkpoints={c.log.num_checkpoints}): {sorted(checkpoint_steps)}")
    print0(f"Tokens : Scaling params ratio: {total_batch_size * num_iterations / num_scaling_params:.2f}")
    print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

    # Log computed (non-config) values to wandb config and config.json
    computed_summary = {
        "num_params": num_params,
        "flops_per_token": num_flops_per_token,
        "num_iterations": num_iterations,
        "total_tokens": total_tokens,
        "total_batch_size": total_batch_size,
        "ddp_world_size": ddp_world_size,
        "tokens_per_scaling_param": total_batch_size * num_iterations / num_scaling_params,
    }
    if master_process:
        wandb_run.config.update({"computed": computed_summary}, allow_val_change=True)
        config_path = os.path.join(run_output_dir, "config.json")
        with open(config_path) as f:
            stored_config = json.load(f)
        stored_config["computed"] = computed_summary
        with open(config_path, "w") as f:
            json.dump(stored_config, f, indent=2)

    # -------------------------------------------------------------------------
    # Schedulers

    def get_lr_multiplier(it):
        warmup_iters = c.optim.warmup_steps
        warmdown_iters = round(c.optim.warmdown_ratio * num_iterations)
        if it < warmup_iters:
            return (it + 1) / warmup_iters
        elif it <= num_iterations - warmdown_iters:
            return 1.0
        else:
            progress = (num_iterations - it) / warmdown_iters
            return progress * 1.0 + (1 - progress) * c.optim.final_lr_frac

    def get_muon_momentum(it):
        warmdown_iters = round(c.optim.warmdown_ratio * num_iterations)
        warmdown_start = num_iterations - warmdown_iters
        if it < 400:
            frac = it / 400
            return (1 - frac) * 0.85 + frac * 0.97
        elif it >= warmdown_start:
            progress = (it - warmdown_start) / warmdown_iters
            return 0.97 * (1 - progress) + 0.90 * progress
        else:
            return 0.97

    def get_weight_decay(it):
        return weight_decay_scaled * 0.5 * (1 + math.cos(math.pi * it / num_iterations))

    # -------------------------------------------------------------------------
    # Loop state

    if not resuming:
        step = 0
        val_bpb = None
        min_val_bpb = float("inf")
        smooth_train_loss = 0
        total_training_time = 0
    else:
        step = meta_data["step"]
        loop_state = meta_data["loop_state"]
        val_bpb = meta_data["val_bpb"]
        min_val_bpb = loop_state["min_val_bpb"]
        smooth_train_loss = loop_state["smooth_train_loss"]
        total_training_time = loop_state["total_training_time"]

    tokens_per_fwdbwd = c.optim.device_batch_size * c.model.max_seq_len
    world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
    assert total_batch_size % world_tokens_per_fwdbwd == 0
    grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
    print0(f"Tokens / micro-batch / rank: {c.optim.device_batch_size} x {c.model.max_seq_len} = {tokens_per_fwdbwd:,}")
    print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
    print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

    # -------------------------------------------------------------------------
    # Training loop

    results = {}
    while True:
        last_step = step == num_iterations
        flops_so_far = num_flops_per_token * total_batch_size * step

        if c.eval.eval_every > 0 and (last_step or step % c.eval.eval_every == 0):
            model.eval()
            val_loader = build_val_loader()
            eval_steps = c.eval.eval_tokens // (c.optim.device_batch_size * c.model.max_seq_len * ddp_world_size)
            with disable_fp8(model):
                val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
            print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
            if val_bpb < min_val_bpb:
                min_val_bpb = val_bpb
            wandb_run.log({
                "step": step,
                "total_training_flops": flops_so_far,
                "total_tokens": total_batch_size * step,
                "total_training_time": total_training_time,
                "val/bpb": val_bpb,
            })
            model.train()

        results = {}
        if c.eval.core_metric_every > 0 and (last_step or (step > 0 and step % c.eval.core_metric_every == 0)):
            model.eval()
            with disable_fp8(orig_model):
                results = evaluate_core(orig_model, tokenizer, device, max_per_task=c.eval.core_metric_max_per_task)
            print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
            wandb_run.log({
                "step": step,
                "total_training_flops": flops_so_far,
                "total_tokens": total_batch_size * step,
                "total_training_time": total_training_time,
                "core_metric": results["core_metric"],
                "centered_results": results["centered_results"],
            })
            if master_process:
                evals_dir = os.path.join(run_output_dir, "evals")
                os.makedirs(evals_dir, exist_ok=True)
                eval_path = os.path.join(evals_dir, f"step_{step:05d}.json")
                with open(eval_path, "w", encoding="utf-8") as f:
                    json.dump({"step": step, **results}, f, indent=2)
            model.train()

        if c.eval.sample_every > 0 and master_process and (last_step or (step > 0 and step % c.eval.sample_every == 0)):
            model.eval()
            prompts = [
                "The capital of France is",
                "The chemical symbol of gold is",
                "If yesterday was Friday, then tomorrow will be",
                "The opposite of hot is",
                "The planets of the solar system are:",
                "My favorite color is",
                "If 5*x + 3 = 13, then x is",
            ]
            engine = Engine(orig_model, tokenizer)
            for prompt in prompts:
                tokens = tokenizer(prompt, prepend="<|bos|>")
                with disable_fp8(orig_model):
                    sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
                print0(tokenizer.decode(sample[0]))
            model.train()

        is_checkpoint_step = last_step or (step > 0 and step != c.optim.resume_from_step and (
            (c.log.save_every > 0 and step % c.log.save_every == 0) or
            (step in checkpoint_steps)
        ))
        if is_checkpoint_step:
            save_checkpoint(
                checkpoint_dir,
                step,
                orig_model.state_dict(),
                optimizer.state_dict(),
                {
                    "step": step,
                    "total_flops": flops_so_far,
                    "total_tokens": total_batch_size * step,
                    "val_bpb": val_bpb,
                    "model_config": model_config_kwargs,
                    "user_config": user_config,
                    "device_batch_size": c.optim.device_batch_size,
                    "max_seq_len": c.model.max_seq_len,
                    "total_batch_size": total_batch_size,
                    "dataloader_state_dict": dataloader_state_dict,
                    "loop_state": {
                        "min_val_bpb": min_val_bpb,
                        "smooth_train_loss": smooth_train_loss,
                        "total_training_time": total_training_time,
                    },
                },
                rank=ddp_rank,
            )

        if last_step:
            if step > 0:  # at least one training step happened
                wandb_run.log({
                    "step": step,
                    "total_training_flops": flops_so_far,
                    "total_tokens": total_batch_size * step,
                    "total_training_time": total_training_time,
                    "train/loss": debiased_smooth_loss,
                    "train/lrm": lrm,
                    "train/dt": dt,
                    "train/tok_per_sec": tok_per_sec,
                    "train/mfu": mfu,
                    "train/epoch": epoch,
                })
            break

        synchronize()
        t0 = time.time()
        for micro_step in range(grad_accum_steps):
            loss = model(x, y)
            train_loss = loss.detach()
            loss = loss / grad_accum_steps
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            x, y, dataloader_state_dict = next(train_loader)

        lrm = get_lr_multiplier(step)
        muon_momentum = get_muon_momentum(step)
        muon_weight_decay = get_weight_decay(step)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group['kind'] == 'muon':
                group["momentum"] = muon_momentum
                group["weight_decay"] = muon_weight_decay

        if scaler is not None:
            scaler.unscale_(optimizer)
            if is_ddp_initialized():
                for v in scaler._found_inf_per_device(optimizer).values():
                    dist.all_reduce(v, op=dist.ReduceOp.MAX)
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        model.zero_grad(set_to_none=True)
        train_loss_f = train_loss.item()
        synchronize()
        t1 = time.time()
        dt = t1 - t0

        ema_beta = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
        debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
        pct_done = 100 * step / num_iterations
        tok_per_sec = int(total_batch_size / dt)
        flops_per_sec = num_flops_per_token * total_batch_size / dt
        mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
        if step > 10:
            total_training_time += dt
        steps_done = step - 10
        if steps_done > 0:
            avg_time_per_step = total_training_time / steps_done
            if c.horizon.target_time_minutes > 0 and steps_done == 20:
                num_iterations = step + int(c.horizon.target_time_minutes * 60 / avg_time_per_step)
                print0(f"Recomputed num_iterations={num_iterations:,} based on {c.horizon.target_time_minutes}min target ({avg_time_per_step*1000:.1f}ms/step)")
            remaining_steps = num_iterations - step
            eta_seconds = remaining_steps * avg_time_per_step
            eta_str = f" | eta: {eta_seconds/60:.1f}m"
        else:
            eta_str = ""
        epoch = f"{dataloader_state_dict['epoch']} pq: {dataloader_state_dict['pq_idx']} rg: {dataloader_state_dict['rg_idx']}"
        print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f} | epoch: {epoch} | total time: {total_training_time/60:.2f}m{eta_str}")
        if step % 100 == 0 or is_checkpoint_step:
            wandb_run.log({
                "step": step,
                "total_training_flops": flops_so_far,
                "total_tokens": total_batch_size * step,
                "total_training_time": total_training_time,
                "train/loss": debiased_smooth_loss,
                "train/lrm": lrm,
                "train/dt": dt,
                "train/tok_per_sec": tok_per_sec,
                "train/mfu": mfu,
                "train/epoch": epoch,
            })

        first_step_of_run = (step == 0) or (resuming and step == c.optim.resume_from_step)
        step += 1

        if first_step_of_run:
            gc.collect()
            gc.freeze()
            gc.disable()
        elif step % 5000 == 0:
            gc.collect()

    # -------------------------------------------------------------------------
    # End of training

    if is_ddp_initialized():
        dist.barrier()  # ensure all ranks finish before any rank starts teardown

    print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
    print0(f"Total training time: {total_training_time/60:.2f}m")
    if val_bpb is not None:
        print0(f"Minimum validation bpb: {min_val_bpb:.6f}")

    from nanochat.report import get_report
    get_report(report_dir=run_output_dir).log(section="Base model training", data=[
        user_config,
        {
            "Number of parameters": num_params,
            "Number of FLOPs per token": f"{num_flops_per_token:e}",
            "Calculated number of iterations": num_iterations,
            "Number of training tokens": total_tokens,
            "Tokens : Scaling params ratio": total_batch_size * num_iterations / num_scaling_params,
            "DDP world size": ddp_world_size,
            "warmup_steps": c.optim.warmup_steps,
            "warmdown_ratio": c.optim.warmdown_ratio,
            "final_lr_frac": c.optim.final_lr_frac,
        },
        {
            "Minimum validation bpb": min_val_bpb if val_bpb is not None else None,
            "Final validation bpb": val_bpb,
            "CORE metric estimate": results.get("core_metric", None),
            "MFU %": f"{mfu:.2f}%",
            "Total training flops": f"{flops_so_far:e}",
            "Total training time": f"{total_training_time/60:.2f}m",
            "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
        },
    ])

    # Update config.json and wandb summary with end-of-training results
    if master_process:
        results_summary = {
            "min_val_bpb": min_val_bpb if val_bpb is not None else None,
            "final_val_bpb": val_bpb,
            "core_metric": results.get("core_metric", None),
            "mfu_pct": mfu,
            "total_flops": flops_so_far,
            "total_training_time_min": total_training_time / 60,
            "peak_memory_mib": get_max_memory() / 1024 / 1024,
        }
        config_path = os.path.join(run_output_dir, "config.json")
        with open(config_path) as f:
            stored_config = json.load(f)
        stored_config.setdefault("computed", {}).update(results_summary)
        with open(config_path, "w") as f:
            json.dump(stored_config, f, indent=2)
        wandb_run.summary.update(results_summary)

    wandb_run.finish()
    compute_cleanup()


if __name__ == "__main__":
    main()
