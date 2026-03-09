from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple

import torch
from torch.utils.data import Dataset

from pgn_loader import PGNLoader
from encoding import board_to_tensor, board_extras
from move_vocab import MoveVocab


@dataclass
class DatasetStats:
    total_pairs_seen: int
    total_samples_kept: int
    total_unk_skipped: int


class ChessDataset(Dataset):
    """
    Supervised dataset:
      X = encoded board tensor (12, 8, 8)
      extras = extra board-state features (6,)
      y = move class id (int)

    This implementation preloads samples into memory for simplicity.
    """

    def __init__(
        self,
        vocab: MoveVocab,
        data_dir: str = "data",
        *,
        max_files: Optional[int] = None,
        max_games_per_file: Optional[int] = None,
        max_positions: Optional[int] = None,
        skip_unk: bool = True,
        dtype: torch.dtype = torch.float32,
        verbose: bool = True,
    ):
        self.vocab = vocab
        self.dtype = dtype
        self.samples: List[Tuple[torch.Tensor, torch.Tensor, int]] = []

        loader = PGNLoader(data_dir=data_dir)

        total_seen = 0
        kept = 0
        unk_skipped = 0

        for board, move in loader.generate_position_move_pairs(
            max_files=max_files,
            max_games_per_file=max_games_per_file,
            max_positions=max_positions,
        ):
            total_seen += 1
            y = vocab.encode(move)

            # UNK index is vocab.stoi["<UNK>"] which is defined as 1
            if skip_unk and y == vocab.stoi[vocab.UNK]:
                unk_skipped += 1
                continue

            x = board_to_tensor(board, dtype=dtype)
            extras = board_extras(board, dtype=dtype)

            self.samples.append((x, extras, y))
            kept += 1

        self.stats = DatasetStats(
            total_pairs_seen=total_seen,
            total_samples_kept=kept,
            total_unk_skipped=unk_skipped,
        )

        if verbose:
            print("[ChessDataset] Built dataset")
            print(f"  total pairs seen: {self.stats.total_pairs_seen:,}")
            print(f"  samples kept:     {self.stats.total_samples_kept:,}")
            print(f"  UNK skipped:      {self.stats.total_unk_skipped:,}")
            if self.stats.total_pairs_seen > 0:
                pct = (self.stats.total_samples_kept / self.stats.total_pairs_seen) * 100
                print(f"  kept ratio:       {pct:.2f}%")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, extras, y = self.samples[idx]
        # CrossEntropyLoss expects y as a LongTensor
        return x, extras, torch.tensor(y, dtype=torch.long)