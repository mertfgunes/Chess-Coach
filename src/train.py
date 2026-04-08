from __future__ import annotations

import os
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

    vocab_path: str = "data/move_vocab.txt"
    max_moves: int = 5000
    min_freq: int = 2

    max_files: int | None = None
    max_games_per_file: int | None = None
    max_positions: int | None = 100_000

    skip_unk: bool = True
    val_ratio: float = 0.1

    batch_size: int = 32
    num_workers: int = 0
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0

    channels: int = 128
    dropout: float = 0.1

    seed: int = 42
    log_every_steps: int = 50

    value_loss_weight: float = 0.5


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def ensure_dirs(cfg: TrainConfig):
    os.makedirs(cfg.runs_dir, exist_ok=True)
    os.makedirs(cfg.checkpoints_dir, exist_ok=True)


def load_or_build_vocab(cfg: TrainConfig) -> MoveVocab:
    if os.path.exists(cfg.vocab_path):
        vocab = MoveVocab.load(cfg.vocab_path)
        print(f"[Vocab] Loaded from {cfg.vocab_path} (size={len(vocab)})")
        return vocab

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
    print(f"[Vocab] Built and saved to {cfg.vocab_path} (size={len(vocab)})")
    return vocab


def save_checkpoint(path, model, optimizer, epoch, best_val_loss, cfg):
    checkpoint = {
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(cfg),
    }
    torch.save(checkpoint, path)


def main():
    cfg = TrainConfig()
    ensure_dirs(cfg)
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    vocab = load_or_build_vocab(cfg)
    vocab_size = len(vocab)

    dataset = ChessDataset(
        vocab=vocab,
        data_dir=cfg.data_dir,
        max_files=cfg.max_files,
        max_games_per_file=cfg.max_games_per_file,
        max_positions=cfg.max_positions,
        skip_unk=cfg.skip_unk,
    )

    if len(dataset) < 2:
        raise ValueError("Dataset too small. Need at least 2 samples.")

    val_len = max(1, int(len(dataset) * cfg.val_ratio))
    train_len = len(dataset) - val_len

    if train_len <= 0:
        raise ValueError("Training split became empty. Reduce val_ratio.")

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
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    model = PolicyCNN(
        vocab_size=vocab_size,
        channels=cfg.channels,
        dropout=cfg.dropout,
    ).to(device)

    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    writer = SummaryWriter(log_dir=cfg.runs_dir)

    best_val_loss = float("inf")
    best_model_path = os.path.join(cfg.checkpoints_dir, "best_model.pt")

    for epoch in range(cfg.epochs):
        model.train()

        train_loss_sum = 0.0
        train_policy_loss_sum = 0.0
        train_value_loss_sum = 0.0
        train_acc_sum = 0.0
        train_batches = 0

        for step, (x, extras, policy_y, value_y) in enumerate(train_loader):
            x = x.to(device)
            extras = extras.to(device)
            policy_y = policy_y.to(device)
            value_y = value_y.to(device)

            policy_logits, value_pred = model(x, extras)

            policy_loss = policy_criterion(policy_logits, policy_y)
            value_loss = value_criterion(value_pred, value_y)
            loss = policy_loss + cfg.value_loss_weight * value_loss

            optimizer.zero_grad()
            loss.backward()

            if cfg.grad_clip_norm is not None and cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)

            optimizer.step()

            batch_acc = accuracy_top1(policy_logits, policy_y)

            train_loss_sum += loss.item()
            train_policy_loss_sum += policy_loss.item()
            train_value_loss_sum += value_loss.item()
            train_acc_sum += batch_acc
            train_batches += 1

            if step % cfg.log_every_steps == 0:
                print(
                    f"[Epoch {epoch} | Step {step}] "
                    f"Loss={loss.item():.4f} "
                    f"Policy={policy_loss.item():.4f} "
                    f"Value={value_loss.item():.4f} "
                    f"Acc={batch_acc:.3f}"
                )

        avg_train_loss = train_loss_sum / train_batches
        avg_train_policy_loss = train_policy_loss_sum / train_batches
        avg_train_value_loss = train_value_loss_sum / train_batches
        avg_train_acc = train_acc_sum / train_batches

        model.eval()

        val_loss_sum = 0.0
        val_policy_loss_sum = 0.0
        val_value_loss_sum = 0.0
        val_acc_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for x, extras, policy_y, value_y in val_loader:
                x = x.to(device)
                extras = extras.to(device)
                policy_y = policy_y.to(device)
                value_y = value_y.to(device)

                policy_logits, value_pred = model(x, extras)

                policy_loss = policy_criterion(policy_logits, policy_y)
                value_loss = value_criterion(value_pred, value_y)
                loss = policy_loss + cfg.value_loss_weight * value_loss

                batch_acc = accuracy_top1(policy_logits, policy_y)

                val_loss_sum += loss.item()
                val_policy_loss_sum += policy_loss.item()
                val_value_loss_sum += value_loss.item()
                val_acc_sum += batch_acc
                val_batches += 1

        # FIX: average validation loss instead of printing raw sum
        avg_val_loss = val_loss_sum / val_batches
        avg_val_policy_loss = val_policy_loss_sum / val_batches
        avg_val_value_loss = val_value_loss_sum / val_batches
        avg_val_acc = val_acc_sum / val_batches

        print(
            f"[Epoch {epoch}] "
            f"Train Loss={avg_train_loss:.4f} "
            f"(Policy={avg_train_policy_loss:.4f}, Value={avg_train_value_loss:.4f}, Acc={avg_train_acc:.3f}) | "
            f"Val Loss={avg_val_loss:.4f} "
            f"(Policy={avg_val_policy_loss:.4f}, Value={avg_val_value_loss:.4f}, Acc={avg_val_acc:.3f})"
        )

        writer.add_scalar("train/loss_total", avg_train_loss, epoch)
        writer.add_scalar("train/loss_policy", avg_train_policy_loss, epoch)
        writer.add_scalar("train/loss_value", avg_train_value_loss, epoch)
        writer.add_scalar("train/acc_top1", avg_train_acc, epoch)

        writer.add_scalar("val/loss_total", avg_val_loss, epoch)
        writer.add_scalar("val/loss_policy", avg_val_policy_loss, epoch)
        writer.add_scalar("val/loss_value", avg_val_value_loss, epoch)
        writer.add_scalar("val/acc_top1", avg_val_acc, epoch)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(
                best_model_path,
                model,
                optimizer,
                epoch,
                best_val_loss,
                cfg,
            )
            print(f"[Checkpoint] Saved best model to {best_model_path}")

    writer.close()
    print("Training done.")


if __name__ == "__main__":
    main()