from __future__ import annotations

import chess

from coach_models import PositionAnalysis
from coach_evaluation import evaluate_position, winner_hint_from_score


class ChessCoachService:
    def analyze_position(self, board: chess.Board) -> PositionAnalysis:
        breakdown = evaluate_position(board)
        score = breakdown.total
        side_to_move = "white" if board.turn == chess.WHITE else "black"

        summary = self._build_summary(score, breakdown)

        return PositionAnalysis(
            fen=board.fen(),
            side_to_move=side_to_move,
            score=score,
            winner_hint=winner_hint_from_score(score),
            breakdown=breakdown,
            summary=summary,
            top_moves=[],
        )

    def _build_summary(self, score: float, breakdown) -> str:
        parts = [winner_hint_from_score(score)]

        if breakdown.material > 0.3:
            parts.append("White has a material edge")
        elif breakdown.material < -0.3:
            parts.append("Black has a material edge")

        if breakdown.center_control > 0.2:
            parts.append("White controls the center better")
        elif breakdown.center_control < -0.2:
            parts.append("Black controls the center better")

        if breakdown.king_safety > 0.2:
            parts.append("Black's king looks less safe")
        elif breakdown.king_safety < -0.2:
            parts.append("White's king looks less safe")

        if len(parts) == 1:
            parts.append("Neither side has a big strategic advantage yet")

        return ". ".join(parts) + "."