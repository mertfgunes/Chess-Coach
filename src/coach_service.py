from __future__ import annotations

import chess

from coach_models import PositionAnalysis
from coach_evaluation import evaluate_position, winner_hint_from_score
from coach_explainer import explain_move
from coach_move_selector import get_top_moves


class ChessCoachService:
    def analyze_position(self, board: chess.Board, model=None, vocab=None) -> PositionAnalysis:
        breakdown = evaluate_position(board)
        score = round(breakdown.total, 2)
        side_to_move = "white" if board.turn == chess.WHITE else "black"

        summary = self._build_summary(score, breakdown)

        top_moves = get_top_moves(model, vocab, board, top_n=3)
        for move_data in top_moves:
            move_data.explanation = explain_move(move_data)

        return PositionAnalysis(
            fen=board.fen(),
            side_to_move=side_to_move,
            score=score,
            winner_hint=winner_hint_from_score(score),
            breakdown=breakdown,
            summary=summary,
            top_moves=top_moves,
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