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
    max_files: int | None = 1
    max_games_per_file: int | None = 200
    max_positions: int | None = 50_000

    skip_unk: bool = True

    #Train split
    val_ratio: float = 0.1

    #Training
    batch_size: int = 64
    num_workers: int = 0
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4

    #Model
    channels: int = 64
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
        """
        logits: B,C
        targets: B
        returns accuracy in 0,1
        """
        preds = logits.argmax(dim=1)
        return (preds == targets).float().mean().item()


    @torch.no_grad()
    def accuracy_topk(logits: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
        """
        Top-k accuracy: correct if target is in top k predicted classes.
        """
        topk = logits.topk(k, dim=1).indices  # (B, k)
        targets = targets.view(-1, 1)         # (B, 1)
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
        cfg = TrainConfig()
        ensure_dirs(cfg)
        set_seed(cfg.seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("[Device]", device)

        #1-Vocabulary

        #the vocab maps moves to integer classlabel used by NN.
        vocab = load_or_build_vocab(cfg)
        vocab_size = len(vocab)


        #2-dataset

        #read pgn, convert board to tensor, convert move to integer class.
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


        #3-train

        #reserving a portion of the data(10% rn). to evulatue the general performance. 
        val_len = int(len(dataset) * cfg.val_ratio)
        train_len = len(dataset) - val_len
        train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(cfg.seed))

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

        #this inits the CNN policy.
        model = PolicyCNN(
            vocab_size=vocab_size,
            channels=cfg.channels,
            dropout=cfg.dropout
        ).to(device)


        #5-crossentropyloss, internally applies the softmax. compares predicted class dist. to actaul class dist.. 

        criterion = nn.CrossEntropyLoss()


        #adamw => also learnt this week in the ML class. 
        #adaptive gradient optimizer
        #weight decay helps prevent overfitting
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )


        #6-tensorboard implementation to track progress
        run_name = time.strftime("chess_%Y%m%d_%H%M%S")
        writer = SummaryWriter(log_dir=os.path.join(cfg.runs_dir, run_name))

        global_step = 0


        #training loop/phase

        for epoch in range(1, cfg.epochs + 1):

        #Switch model to training mode
            model.train()

            epoch_loss = 0.0
            epoch_acc1 = 0.0

            for step, (x, y) in enumerate(train_loader, start=1):

                # Move data to gpu
                x = x.to(device)
                y = y.to(device)

                #forwardpass

                #predicts
                logits = model(x)

                #calculate loss
                loss = criterion(logits, y)

                #Backwardpass
                optimizer.zero_grad()
                loss.backward()      #compute gradients
                optimizer.step()     #update weights

                #Track metrics
                acc1 = accuracy_top1(logits, y)

                epoch_loss += loss.item()
                epoch_acc1 += acc1

                global_step += 1

            #validation phase


            model.eval()

            val_loss = 0.0
            val_acc1 = 0.0
            n_batches = 0

            with torch.no_grad():  # disables gradient tracking
                for x, y in val_loader:
                    x = x.to(device)
                    y = y.to(device)

                    logits = model(x)
                    loss = criterion(logits, y)

                    val_loss += loss.item()
                    val_acc1 += accuracy_top1(logits, y)
                    n_batches += 1

            val_loss /= max(n_batches, 1)
            val_acc1 /= max(n_batches, 1)

            print(
                f"[Epoch {epoch}] "
                f"Train Loss: {epoch_loss/len(train_loader):.4f} "
                f"Train Acc: {epoch_acc1/len(train_loader):.3f} | "
                f"Val Loss: {val_loss:.4f} "
                f"Val Acc: {val_acc1:.3f}"
            )

            # Save checkpoint after each epoch
            torch.save(
                model.state_dict(),
                os.path.join(cfg.checkpoints_dir, f"{run_name}_epoch{epoch}.pt"),
            )

        writer.close()
        print("Training complete.")












