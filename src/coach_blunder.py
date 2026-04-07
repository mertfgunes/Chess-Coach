from __future__ import annotations

import chess


def classify_move_loss(loss: float) -> str:
    if loss < 0.30:
        return "good"
    if loss < 0.80:
        return "inaccuracy"
    if loss < 1.80:
        return "mistake"
    return "blunder"


def explain_bad_move(board_before: chess.Board, played_move: chess.Move, best_move: chess.Move) -> str:
    reasons = []

    if board_before.is_capture(best_move) and not board_before.is_capture(played_move):
        reasons.append("you missed a useful capture")

    if board_before.gives_check(best_move) and not board_before.gives_check(played_move):
        reasons.append("you missed a forcing check")

    if not reasons:
        reasons.append("the move makes your position less effective than the best option")

    return reasons[0].capitalize() + "."