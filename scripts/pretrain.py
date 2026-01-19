"""Small-scale pretraining script with sequence packing and document masking.

Usage:
    uv run python scripts/pretrain.py

Features:
- Streams from HuggingFace datasets
- Packs sequences to 8k tokens with document masking (cu_seqlens)
- Uses custom MoE model or HuggingFace model
- Supports gradient accumulation, checkpointing, wandb logging
"""

import time
from dataclasses import dataclass
from typing import Any

import datasets
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from transformers import AutoTokenizer

from nanomoe.data import PackedPretrainDataset, PretrainBatch, create_document_mask
from nanomoe.model import MoEConfig, create_model
from nanomoe.train import (
    Checkpointer,
    WSDConfig,
    WSDScheduler,
    setup_logging,
)


@dataclass
class TrainConfig:
    # Model
    model_preset: str = "small"  # "tiny", "small", "medium", "large", or "custom"
    num_experts: int | None = None  # Override num_experts if set
    num_experts_per_tok: int | None = None  # Override if set

    # Data
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str = "sample-10BT"
    tokenizer_name: str = "Qwen/Qwen2.5-0.5B"
    pack_size: int = 8192

    # Training
    batch_size: int = 1  # Micro batch size (packed sequences)
    gradient_accumulation: int = 8
    max_steps: int = 1000
    max_tokens: int | None = None  # Stop after this many tokens (overrides max_steps)

    # Optimizer
    peak_lr: float = 1e-4
    floor_lr: float = 1e-6
    warmup_steps: int = 100
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0

    # Logging & checkpointing
    log_every: int = 10
    checkpoint_every: int = 500
    checkpoint_dir: str = "checkpoints/pretrain"
    wandb_project: str | None = None
    wandb_name: str | None = None

    # System
    seed: int = 42
    dtype: str = "bfloat16"  # "float32", "float16", "bfloat16"
    compile_model: bool = False  # Use torch.compile


def compute_loss(
    model: Any,  # Can be MoETransformer or torch.compile'd version
    batch: PretrainBatch,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, dict]:
    """Compute cross-entropy loss with document masking."""
    # Move batch to device
    batch = batch.to(device)

    # Create document-aware attention mask
    attention_mask = create_document_mask(batch.cu_seqlens, dtype=dtype)

    # Forward pass
    # Reshape for batch dim
    input_ids = batch.input_ids.unsqueeze(0)  # [1, seq_len]
    position_ids = batch.position_ids.unsqueeze(0)  # [1, seq_len]

    outputs = model(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )

    logits = outputs["logits"][0]  # [seq_len, vocab]
    aux_loss = outputs["aux_loss"]

    # Shift for next-token prediction
    shift_logits = logits[:-1]
    shift_labels = batch.labels[:-1]
    shift_mask = batch.loss_mask[:-1]

    # Compute loss only on masked positions
    loss = F.cross_entropy(shift_logits, shift_labels, reduction="none")
    masked_loss = (loss * shift_mask).sum() / shift_mask.sum().clamp(min=1)

    # Add auxiliary loss
    total_loss = masked_loss + aux_loss

    metrics = {
        "loss": masked_loss.item(),
        "aux_loss": aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss,
        "num_tokens": shift_mask.sum().item(),
        "num_docs": batch.num_docs,
    }

    return total_loss, metrics


def main():
    cfg = TrainConfig()

    # Set seed
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # Device and dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[cfg.dtype]

    print(f"Device: {device}, dtype: {dtype}")

    # Create model config
    model_config = getattr(MoEConfig, cfg.model_preset)()
    if cfg.num_experts is not None:
        model_config.num_experts = cfg.num_experts
    if cfg.num_experts_per_tok is not None:
        model_config.num_experts_per_tok = cfg.num_experts_per_tok

    print(f"Model config: {cfg.model_preset}")
    print(f"  num_experts: {model_config.num_experts}")
    print(f"  num_experts_per_tok: {model_config.num_experts_per_tok}")
    print(f"  hidden_size: {model_config.hidden_size}")
    print(f"  num_layers: {model_config.num_layers}")
    print(f"  Total params: ~{model_config.num_total_params / 1e6:.1f}M")
    print(f"  Active params: ~{model_config.num_active_params / 1e6:.1f}M")

    # Load tokenizer and update vocab size
    print(f"Loading tokenizer: {cfg.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    model_config.vocab_size = len(tokenizer)

    # Create model
    print("Creating model...")
    model = create_model(model_config)
    model = model.to(device=device, dtype=dtype)
    model.train()

    # Keep reference to original model for checkpointing
    model_for_checkpoint = model

    if cfg.compile_model and torch.cuda.is_available():
        print("Compiling model with torch.compile...")
        model.compile()

    print(f"Model parameters: {model.num_parameters() / 1e6:.2f}M")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.peak_lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    # LR scheduler
    lr_config = WSDConfig(
        peak_lr=cfg.peak_lr,
        floor_lr=cfg.floor_lr,
        warmup_steps=cfg.warmup_steps,
        sustain_tokens=0,
        decay_tokens=cfg.max_tokens or cfg.max_steps * cfg.pack_size * cfg.batch_size * cfg.gradient_accumulation,
    )
    scheduler = WSDScheduler(optimizer, lr_config)

    # Checkpointing
    checkpointer = Checkpointer(cfg.checkpoint_dir, keep_last=3)

    # Logging
    logger = setup_logging(
        log_dir=cfg.checkpoint_dir,
        wandb_project=cfg.wandb_project,
        wandb_name=cfg.wandb_name,
        console=True,
        console_every=cfg.log_every,
        config=cfg,
    )

    # Load dataset
    print(f"Loading dataset: {cfg.dataset_name}/{cfg.dataset_config}")
    hf_dataset = datasets.load_dataset(cfg.dataset_name, cfg.dataset_config, streaming=True, split="train")

    # Create packed dataset with prefetching
    dataset = PackedPretrainDataset(
        hf_dataset=hf_dataset,
        tokenizer=tokenizer,
        pack_size=cfg.pack_size,
        prefetch_batches=4,
        shuffle_buffer=10_000,
        seed=cfg.seed,
    )

    # Training state
    step = 0
    tokens_seen = 0
    accumulated_loss = 0.0
    accumulated_aux_loss = 0.0
    accumulated_tokens = 0
    start_time = time.time()

    # Resume from checkpoint if exists
    resume_step, resume_tokens = checkpointer.load(model_for_checkpoint, optimizer, scheduler)
    if resume_step > 0:
        step = resume_step
        tokens_seen = resume_tokens
        print(f"Resumed from step {step}, tokens_seen {tokens_seen}")

    # Gradient scaler for mixed precision
    scaler = GradScaler() if cfg.dtype == "float16" else None

    print(f"Starting training for {cfg.max_steps} steps...")
    print(f"Pack size: {cfg.pack_size}, batch size: {cfg.batch_size}, grad accum: {cfg.gradient_accumulation}")

    try:
        batch_iter = iter(dataset)
        micro_batches = []

        while step < cfg.max_steps:
            if cfg.max_tokens and tokens_seen >= cfg.max_tokens:
                print(f"Reached max_tokens {cfg.max_tokens}")
                break

            # Collect micro batches for gradient accumulation
            while len(micro_batches) < cfg.gradient_accumulation:
                try:
                    batch = next(batch_iter)
                    micro_batches.append(batch)
                except StopIteration:
                    # Restart iterator
                    batch_iter = iter(dataset)

            # Gradient accumulation
            optimizer.zero_grad()
            step_loss = 0.0
            step_aux_loss = 0.0
            step_tokens = 0

            for batch in micro_batches[: cfg.gradient_accumulation]:
                # Forward + backward
                if scaler:
                    with autocast(dtype=torch.float16):
                        loss, metrics = compute_loss(model, batch, device, dtype)
                    scaler.scale(loss / cfg.gradient_accumulation).backward()
                else:
                    with torch.autocast(device_type="cuda", dtype=dtype, enabled=dtype != torch.float32):
                        loss, metrics = compute_loss(model, batch, device, dtype)
                    (loss / cfg.gradient_accumulation).backward()

                step_loss += metrics["loss"]
                step_aux_loss += metrics["aux_loss"]
                step_tokens += metrics["num_tokens"]

            micro_batches = micro_batches[cfg.gradient_accumulation :]

            # Gradient clipping and optimizer step
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()

            # Update LR
            tokens_seen += step_tokens
            lr = scheduler.step(step=step, tokens_seen=tokens_seen)

            step += 1
            accumulated_loss += step_loss / cfg.gradient_accumulation
            accumulated_aux_loss += step_aux_loss / cfg.gradient_accumulation
            accumulated_tokens += step_tokens

            # Logging
            if step % cfg.log_every == 0:
                elapsed = time.time() - start_time
                tokens_per_sec = accumulated_tokens / elapsed if elapsed > 0 else 0

                log_metrics = {
                    "train/loss": accumulated_loss / cfg.log_every,
                    "train/aux_loss": accumulated_aux_loss / cfg.log_every,
                    "train/lr": lr,
                    "train/tokens_seen": tokens_seen,
                    "train/step": step,
                    "perf/tokens_per_sec": tokens_per_sec,
                    "perf/elapsed_time": elapsed,
                }
                logger.log_metrics(log_metrics, step=step)

                accumulated_loss = 0.0
                accumulated_aux_loss = 0.0
                accumulated_tokens = 0
                start_time = time.time()

            # Checkpointing
            if step % cfg.checkpoint_every == 0:
                checkpointer.save(
                    step=step,
                    model=model_for_checkpoint,
                    optimizer=optimizer,
                    tokens_seen=tokens_seen,
                    scheduler=scheduler,
                )
                print(f"Saved checkpoint at step {step}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        dataset.stop()

    # Final checkpoint
    checkpointer.save(
        step=step,
        model=model_for_checkpoint,
        optimizer=optimizer,
        tokens_seen=tokens_seen,
        scheduler=scheduler,
    )
    checkpointer.wait()

    logger.close()
    print(f"Training complete. Final step: {step}, tokens_seen: {tokens_seen}")


if __name__ == "__main__":
    main()
