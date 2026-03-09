from __future__ import annotations

import os
import time
import random
from dataclasses import dataclass

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

    #Vocab
    vocab_path: str = "data/move_vocab.txt"
    max_moves: int = 5000
    min_freq: int = 2

    #Dataset sampling
    max_files: int | None = 2
    max_games_per_file: int | None = 200
    max_positions: int | None = 100_000

    skip_unk: bool = True

    #Train split
    val_ratio: float = 0.1

    #Training
    batch_size: int = 64
    num_workers: int = 0
    epochs: int = 15
    lr: float = 1e-3
    weight_decay: float = 1e-4

    #Model
    channels: int = 128
    dropout: float = 0.1

    #Reproducibility
    seed: int = 42

    #Logging
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

    #2-dataset
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

    #3-train split
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

    #4-model
    model = PolicyCNN(
        vocab_size=vocab_size,
        channels=cfg.channels,
        dropout=cfg.dropout,
    ).to(device)

    #5-Loss + optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    #6-tensorboard
    run_name = time.strftime("chess_%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=os.path.join(cfg.runs_dir, run_name))

    #Force one event so TensorBoard is never empty
    writer.add_text("debug", "training started", 0)
    writer.flush()

    best_val_acc1 = float("-inf")
    best_epoch = 0
    best_model_path = os.path.join(cfg.checkpoints_dir, "best_model.pt")
    last_model_path = os.path.join(cfg.checkpoints_dir, "last_model.pt")

    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_acc1 = 0.0
        epoch_acc5 = 0.0

        for step, (x, extras, y) in enumerate(train_loader, start=1):
            x = x.to(device)
            extras = extras.to(device)
            y = y.to(device)

            logits = model(x, extras)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc1 = accuracy_top1(logits, y)
            acc5 = accuracy_topk(logits, y, k=5)

            epoch_loss += loss.item()
            epoch_acc1 += acc1
            epoch_acc5 += acc5
            global_step += 1

        #validation
        model.eval()
        val_loss = 0.0
        val_acc1 = 0.0
        val_acc5 = 0.0
        n_batches = 0

        with torch.no_grad():
            for x, extras, y in val_loader:
                x = x.to(device)
                extras = extras.to(device)
                y = y.to(device)

                logits = model(x, extras)
                loss = criterion(logits, y)

                val_loss += loss.item()
                val_acc1 += accuracy_top1(logits, y)
                val_acc5 += accuracy_topk(logits, y, k=5)
                n_batches += 1

        val_loss /= max(n_batches, 1)
        val_acc1 /= max(n_batches, 1)
        val_acc5 /= max(n_batches, 1)

        train_loss_avg = epoch_loss / len(train_loader)
        train_acc1_avg = epoch_acc1 / len(train_loader)
        train_acc5_avg = epoch_acc5 / len(train_loader)

        print(
            f"[Epoch {epoch}] "
            f"Train Loss: {train_loss_avg:.4f} "
            f"Train Acc@1: {train_acc1_avg:.3f} "
            f"Train Acc@5: {train_acc5_avg:.3f} | "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc@1: {val_acc1:.3f} "
            f"Val Acc@5: {val_acc5:.3f}"
        )

        #tesnorboard scalars
        writer.add_scalar("train/loss", train_loss_avg, epoch)
        writer.add_scalar("train/acc_top1", train_acc1_avg, epoch)
        writer.add_scalar("train/acc_top5", train_acc5_avg, epoch)

        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/acc_top1", val_acc1, epoch)
        writer.add_scalar("val/acc_top5", val_acc5, epoch)
        writer.flush()

        #always keeping the last recent model
        torch.save(model.state_dict(), last_model_path)

        #updating the best checkpoint only when validation improves
        #so we always have the best epoch choosen. 

        #in some cases sometimes (for example in the last run) the epoch 12 gave better result than 15.
        if val_acc1 > best_val_acc1:
            best_val_acc1 = val_acc1
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
            print(
                f"[Checkpoint] New best model saved to {best_model_path} "
                f"(epoch={epoch}, val_acc={val_acc1:.3f})"
            )

    writer.close()
    print(f"Best epoch: {best_epoch} | Best val acc: {best_val_acc1:.3f}")
    print(f"Best model: {best_model_path}")
    print(f"Last model: {last_model_path}")
    print("Training done.")


if __name__ == "__main__":
    print(">>>DEBUG: CALLING MAIN()")
    main()