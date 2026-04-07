from __future__ import annotations

import chess
from coach_models import EvaluationBreakdown
from coach_tactics import immediate_material_threat

PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.2,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}

CENTER_SQUARES = [chess.D4, chess.E4, chess.D5, chess.E5]


def evaluate_material(board: chess.Board) -> float:
    score = 0.0
    for _, piece in board.piece_map().items():
        value = PIECE_VALUES[piece.piece_type]
        if piece.color == chess.WHITE:
            score += value
        else:
            score -= value
    return score


def evaluate_mobility(board: chess.Board) -> float:
    temp = board.copy(stack=False)

    temp.turn = chess.WHITE
    white_moves = temp.legal_moves.count()

    temp.turn = chess.BLACK
    black_moves = temp.legal_moves.count()

    return (white_moves - black_moves) * 0.03


def evaluate_center_control(board: chess.Board) -> float:
    score = 0.0
    for sq in CENTER_SQUARES:
        if board.is_attacked_by(chess.WHITE, sq):
            score += 0.10
        if board.is_attacked_by(chess.BLACK, sq):
            score -= 0.10
    return score


def evaluate_king_safety(board: chess.Board) -> float:
    score = 0.0

    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)

    if white_king is not None:
        white_attackers = len(board.attackers(chess.BLACK, white_king))
        score -= white_attackers * 0.15

    if black_king is not None:
        black_attackers = len(board.attackers(chess.WHITE, black_king))
        score += black_attackers * 0.15

    return score


def evaluate_development(board: chess.Board) -> float:
    score = 0.0

    white_home = [chess.B1, chess.G1, chess.C1, chess.F1]
    black_home = [chess.B8, chess.G8, chess.C8, chess.F8]

    for sq in white_home:
        if board.piece_at(sq) is None:
            score += 0.10

    for sq in black_home:
        if board.piece_at(sq) is None:
            score -= 0.10

    return score


def evaluate_pawn_structure(board: chess.Board) -> float:
    score = 0.0

    for color, sign in [(chess.WHITE, 1), (chess.BLACK, -1)]:
        pawns = list(board.pieces(chess.PAWN, color))
        files = [chess.square_file(sq) for sq in pawns]

        for file_idx in range(8):
            count = files.count(file_idx)
            if count > 1:
                score -= sign * (count - 1) * 0.15

        for sq in pawns:
            file_idx = chess.square_file(sq)
            has_left = (file_idx - 1) in files
            has_right = (file_idx + 1) in files
            if not has_left and not has_right:
                score -= sign * 0.10

    return score


def evaluate_position(board: chess.Board) -> EvaluationBreakdown:
    return EvaluationBreakdown(
        material=evaluate_material(board),
        mobility=evaluate_mobility(board),
        center_control=evaluate_center_control(board),
        king_safety=evaluate_king_safety(board),
        development=evaluate_development(board),
        pawn_structure=evaluate_pawn_structure(board),
        piece_safety=evaluate_piece_safety(board),
    )


def winner_hint_from_score(score: float) -> str:
    if score > 1.5:
        return "White is clearly better"
    if score > 0.4:
        return "White is slightly better"
    if score < -1.5:
        return "Black is clearly better"
    if score < -0.4:
        return "Black is slightly better"
    return "The position is roughly equal"

def evaluate_piece_safety(board: chess.Board) -> float:
    white_threat = immediate_material_threat(board, chess.WHITE)
    black_threat = immediate_material_threat(board, chess.BLACK)

    # if White has hanging pieces, that's bad for White
    # if Black has hanging pieces, that's good for White
    return (black_threat - white_threat) * 0.35