from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
import chess


@dataclass
class EvaluationBreakdown:
    material: float = 0.0
    mobility: float = 0.0
    center_control: float = 0.0
    king_safety: float = 0.0
    development: float = 0.0
    pawn_structure: float = 0.0
    piece_safety: float = 0.0
    
    @property
    def total(self) -> float:
        return (
            self.material
            + self.mobility
            + self.center_control
            + self.king_safety
            + self.development
            + self.pawn_structure
            + self.piece_safety
        )


@dataclass
class MoveSuggestion:
    move: chess.Move
    san: str
    score: float
    explanation: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class PositionAnalysis:
    fen: str
    side_to_move: str
    score: float
    winner_hint: str
    breakdown: EvaluationBreakdown
    summary: str = ""
    top_moves: List[MoveSuggestion] = field(default_factory=list)