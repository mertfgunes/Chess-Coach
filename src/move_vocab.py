from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from typing import Dict, List, Iterable, Optional, Tuple

import chess
from pgn_loader import PGNLoader


@dataclass
class MoveVocab:
    """
    Maps UCI move strings <-> integer class IDs.

    Special tokens:
      <PAD> : 0  
      <UNK> : 1  unknown moves
    """
    stoi: Dict[str, int]
    itos: List[str]

    PAD: str = "<PAD>"
    UNK: str = "<UNK>"

    @classmethod
    def build_from_pgns(
        cls,
        data_dir: str = "data",
        max_moves: int = 5000,
        min_freq: int = 1,
        verbose: bool = True,
        
        #this ones are for fast iterative to see if it actually works.
        max_files: int | None = None,
        max_games_per_file: int | None = None,
        max_positions: int | None = 200_000,
        progress_every: int = 50_000,
    ) -> "MoveVocab":
        """
        Build a vocabulary from all PGN files in data_dir.

        Args:
            max_moves: keep top N most common moves 
            min_freq: ignore moves that occur < min_freq: so regardless of the move being good or bad, if it is played only 1 time. it is not important
            (it can not be really a good move if it is less than min_freq logcically.)
        """
        loader = PGNLoader(data_dir=data_dir)
        counter = Counter()

        seen = 0
        for _board, move in loader.generate_position_move_pairs(
            max_files=max_files,
            max_games_per_file=max_games_per_file,
            max_positions=max_positions,
        ):
            counter[move.uci()] += 1
            seen += 1

            if verbose and progress_every and seen % progress_every == 0:
                print(f"[MoveVocab] Scanned {seen:,} positions... (unique moves so far: {len(counter):,})")



        #filter and take most common
        items = [(m, c) for (m, c) in counter.items() if c >= min_freq]
        items.sort(key=lambda x: x[1], reverse=True)
        items = items[:max_moves]

        #itos list
        itos = [cls.PAD, cls.UNK] + [m for (m, _c) in items]
        stoi = {m: i for i, m in enumerate(itos)}

        if verbose:
            total_unique = len(counter)
            kept = len(items)
            total_counts = sum(counter.values())
            kept_counts = sum(c for (_m, c) in items)

            coverage = (kept_counts / total_counts) if total_counts > 0 else 0.0
            print(f"[MoveVocab] Unique moves in PGNs: {total_unique}")
            print(f"[MoveVocab] Kept moves: {kept} (max_moves={max_moves}, min_freq={min_freq})")
            print(f"[MoveVocab] Coverage of kept moves: {coverage:.2%}")
            print(f"[MoveVocab] Vocab size (incl specials): {len(itos)}")

        return cls(stoi=stoi, itos=itos)

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, move: chess.Move | str) -> int:
        #Convert chess.Move or UCI string to class index.
        
        uci = move.uci() if isinstance(move, chess.Move) else str(move)
        return self.stoi.get(uci, self.stoi[self.UNK])

    def decode(self, idx: int) -> str:
        #Convert class index back to UCI string.
        if idx < 0 or idx >= len(self.itos):
            return self.UNK
        return self.itos[idx]

    def is_known(self, move: chess.Move | str) -> bool:
        #True if move is not UNK
        uci = move.uci() if isinstance(move, chess.Move) else str(move)
        return uci in self.stoi and uci not in (self.PAD, self.UNK)

    def save(self, path: str) -> None:
        #Save vocab to a simple text file
        with open(path, "w", encoding="utf-8") as f:
            for token in self.itos:
                f.write(token + "\n")

    @classmethod
    def load(cls, path: str) -> "MoveVocab":
        #Load vocab from a text file saved
        with open(path, "r", encoding="utf-8") as f:
            itos = [line.strip() for line in f if line.strip()]
        stoi = {m: i for i, m in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)