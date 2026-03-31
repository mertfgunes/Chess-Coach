from __future__ import annotations

import os
import time
import random
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from move_vocab import MoveVocab
from dataset import ChessDataset
from model import PolicyCNN


@dataclass
class TrainConfig:
    data_dir: str = "data"
    runs_dir: str = "runs"
    checkpoints_dir: str = "checkpoints"

    # Vocab
    vocab_path: str = "data/move_vocab.txt"
    max_moves: int = 5000
    min_freq: int = 2

    # Dataset sampling
    max_files: int | None = None
    max_games_per_file: int | None = None
    max_positions: int | None = 100_000

    skip_unk: bool = True

    # Train split
    #Train split
    val_ratio: float = 0.1

    #Training
    batch_size: int = 32
    num_workers: int = 0
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0

    # early stoping and lr schedule
    early_stopping_patience: int = 4
    lr_scheduler_patience: int = 2
    lr_scheduler_factor: float = 0.5
    min_lr: float = 1e-6

    # Model
    channels: int = 128
    dropout: float = 0.1

    # Reproducibility
    seed: int = 42

    # Logging
    log_every_steps: int = 50


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


@torch.no_grad()
def accuracy_topk(logits: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
    k = min(k, logits.size(1))
    topk = logits.topk(k, dim=1).indices
    targets = targets.view(-1, 1)
    return (topk == targets).any(dim=1).float().mean().item()


def ensure_dirs(cfg: TrainConfig):
    os.makedirs(cfg.runs_dir, exist_ok=True)
    os.makedirs(cfg.checkpoints_dir, exist_ok=True)


def load_or_build_vocab(cfg: TrainConfig) -> MoveVocab:
    if os.path.exists(cfg.vocab_path):
        vocab = MoveVocab.load(cfg.vocab_path)
        print(f"[Vocab] Loaded from {cfg.vocab_path} (size={len(vocab)})")
        return vocab

    print("[Vocab] Building vocab from PGNs...")
    vocab = MoveVocab.build_from_pgns(
        data_dir=cfg.data_dir,
        max_moves=cfg.max_moves,
        min_freq=cfg.min_freq,
        max_files=cfg.max_files,
        max_games_per_file=cfg.max_games_per_file,
        max_positions=cfg.max_positions,
        verbose=True,
    )
    vocab.save(cfg.vocab_path)
    print(f"[Vocab] Saved to {cfg.vocab_path}")
    return vocab


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_val_acc1: float,
    cfg: TrainConfig,
):
    checkpoint = {
        "epoch": epoch,
        "best_val_acc1": best_val_acc1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "config": asdict(cfg),
    }
    torch.save(checkpoint, path)


def main():
    print(">>>DEBUG: INSIDE MAIN()")

    cfg = TrainConfig()
    ensure_dirs(cfg)
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    #1-Vocabulary
    vocab = load_or_build_vocab(cfg)
    vocab_size = len(vocab)

    # 2 - Dataset
    dataset = ChessDataset(
        vocab=vocab,
        data_dir=cfg.data_dir,
        max_files=cfg.max_files,
        max_games_per_file=cfg.max_games_per_file,
        max_positions=cfg.max_positions,
        skip_unk=cfg.skip_unk,
        verbose=True,
    )

    if len(dataset) < 100:
        raise RuntimeError("Dataset too small. Increase max_positions/max_files or check your PGNs.")

    # 3 - Train/val split
    val_len = int(len(dataset) * cfg.val_ratio)
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print(f"[Split] train={len(train_ds):,} val={len(val_ds):,}")

    # 4 - Model
    model = PolicyCNN(
        vocab_size=vocab_size,
        channels=cfg.channels,
        dropout=cfg.dropout,
    ).to(device)

    # 5 - Loss + optimizer + scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.lr_scheduler_factor,
        patience=cfg.lr_scheduler_patience,
        min_lr=cfg.min_lr,
    )

    # 6 - TensorBoard
    run_name = time.strftime("chess_%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=os.path.join(cfg.runs_dir, run_name))
    writer.add_text("debug", "training started", 0)
    writer.flush()

    best_val_acc1 = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0

    best_model_path = os.path.join(cfg.checkpoints_dir, "best_model.pt")
    last_model_path = os.path.join(cfg.checkpoints_dir, "last_model.pt")

    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()

        train_loss_sum = 0.0
        train_correct_top1_sum = 0.0
        train_correct_top5_sum = 0.0
        train_sample_count = 0

        for step, (x, extras, y) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            extras = extras.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x, extras)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()

            if cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)

            optimizer.step()

            batch_size = y.size(0)
            acc1 = accuracy_top1(logits, y)
            acc5 = accuracy_topk(logits, y, k=5)

            train_loss_sum += loss.item() * batch_size
            train_correct_top1_sum += acc1 * batch_size
            train_correct_top5_sum += acc5 * batch_size
            train_sample_count += batch_size
            global_step += 1

            if step % cfg.log_every_steps == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"[Epoch {epoch}/{cfg.epochs}] "
                    f"Step {step}/{len(train_loader)} "
                    f"Loss: {loss.item():.4f} "
                    f"Acc@1: {acc1:.3f} "
                    f"Acc@5: {acc5:.3f} "
                    f"LR: {current_lr:.6f}"
                )

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_correct_top1_sum = 0.0
        val_correct_top5_sum = 0.0
        val_sample_count = 0

        with torch.no_grad():
            for x, extras, y in val_loader:
                x = x.to(device, non_blocking=True)
                extras = extras.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                logits = model(x, extras)
                loss = criterion(logits, y)

                batch_size = y.size(0)
                acc1 = accuracy_top1(logits, y)
                acc5 = accuracy_topk(logits, y, k=5)

                val_loss_sum += loss.item() * batch_size
                val_correct_top1_sum += acc1 * batch_size
                val_correct_top5_sum += acc5 * batch_size
                val_sample_count += batch_size

        train_loss_avg = train_loss_sum / max(train_sample_count, 1)
        train_acc1_avg = train_correct_top1_sum / max(train_sample_count, 1)
        train_acc5_avg = train_correct_top5_sum / max(train_sample_count, 1)

        val_loss = val_loss_sum / max(val_sample_count, 1)
        val_acc1 = val_correct_top1_sum / max(val_sample_count, 1)
        val_acc5 = val_correct_top5_sum / max(val_sample_count, 1)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"[Epoch {epoch}] "
            f"Train Loss: {train_loss_avg:.4f} "
            f"Train Acc@1: {train_acc1_avg:.3f} "
            f"Train Acc@5: {train_acc5_avg:.3f} | "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc@1: {val_acc1:.3f} "
            f"Val Acc@5: {val_acc5:.3f} "
            f"LR: {current_lr:.6f}"
        )

        # TensorBoard
        writer.add_scalar("train/loss", train_loss_avg, epoch)
        writer.add_scalar("train/acc_top1", train_acc1_avg, epoch)
        writer.add_scalar("train/acc_top5", train_acc5_avg, epoch)

        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/acc_top1", val_acc1, epoch)
        writer.add_scalar("val/acc_top5", val_acc5, epoch)
        writer.add_scalar("train/lr", current_lr, epoch)
        writer.flush()

        # Always keep the last model
        save_checkpoint(
            path=last_model_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_val_acc1=best_val_acc1,
            cfg=cfg,
        )

        # Save best model
        if val_acc1 > best_val_acc1:
            best_val_acc1 = val_acc1
            best_epoch = epoch
            epochs_without_improvement = 0

            save_checkpoint(
                path=best_model_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_acc1=best_val_acc1,
                cfg=cfg,
            )

            print(
                f"[Checkpoint] New best model saved to {best_model_path} "
                f"(epoch={epoch}, val_acc={val_acc1:.3f})"
            )
        else:
            epochs_without_improvement += 1
            print(
                f"[EarlyStopping] No improvement for {epochs_without_improvement} "
                f"epoch(s). Best epoch so far: {best_epoch} "
                f"(val_acc={best_val_acc1:.3f})"
            )

        # Step scheduler after validation
        scheduler.step(val_loss)

        # Early stopping
        if epochs_without_improvement >= cfg.early_stopping_patience:
            print(
                f"[EarlyStopping] Stopping early at epoch {epoch}. "
                f"Best epoch: {best_epoch} | Best val acc: {best_val_acc1:.3f}"
            )
            break

    writer.close()
    print(f"Best epoch: {best_epoch} | Best val acc: {best_val_acc1:.3f}")
    print(f"Best model: {best_model_path}")
    print(f"Last model: {last_model_path}")
    print("Training done.")


if __name__ == "__main__":
    print(">>>DEBUG: CALLING MAIN()")
    main()