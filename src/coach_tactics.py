from __future__ import annotations

import chess

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100,
}


def get_piece_value(piece: chess.Piece | None) -> int:
    if piece is None:
        return 0
    return PIECE_VALUES.get(piece.piece_type, 0)


def attackers_count(board: chess.Board, square: int, color: chess.Color) -> int:
    return len(board.attackers(color, square))


def defenders_count(board: chess.Board, square: int, color: chess.Color) -> int:
    return len(board.attackers(color, square))


def _least_valuable_attacker(
    board: chess.Board,
    square: int,
    color: chess.Color,
) -> tuple[int | None, int]:
    min_val = 999
    min_sq = None
    for atk_sq in board.attackers(color, square):
        piece = board.piece_at(atk_sq)
        if piece is None:
            continue
        val = PIECE_VALUES.get(piece.piece_type, 0)
        if val < min_val:
            min_val = val
            min_sq = atk_sq
    return min_sq, min_val


def _legal_captures_to_square(
    board: chess.Board,
    square: int,
    color: chess.Color,
) -> list[chess.Move]:
    original_turn = board.turn
    board.turn = color
    try:
        return [
            move
            for move in board.legal_moves
            if move.to_square == square
            and board.piece_at(move.from_square) is not None
        ]
    finally:
        board.turn = original_turn


def _best_capture_sequence_gain(
    board: chess.Board,
    square: int,
    color: chess.Color,
    target_value: int,
    depth: int = 0,
) -> int:
    """
    Best net gain for `color` if it chooses to capture on `square`.

    The side can decline the exchange, so the returned gain is never negative.
    """
    if depth >= 12:
        return 0

    best_gain = 0
    captures = _legal_captures_to_square(board, square, color)
    captures.sort(key=lambda move: get_piece_value(board.piece_at(move.from_square)))

    for capture in captures:
        attacker = board.piece_at(capture.from_square)
        if attacker is None:
            continue

        attacker_value = get_piece_value(attacker)
        sim = board.copy(stack=False)
        sim.push(capture)

        reply_gain = _best_capture_sequence_gain(
            sim,
            square,
            not color,
            attacker_value,
            depth + 1,
        )
        gain = target_value - reply_gain

        if gain > best_gain:
            best_gain = gain

    return best_gain


def static_exchange_evaluation(board: chess.Board, move: chess.Move) -> int:
    """
    Estimate the net material gain for the side making `move`.

    Positive = capturing side wins material.
    Negative = capturing side loses material.
    Zero = equal exchange or non-capture.

    Legal recaptures are generated through python-chess, so pins and king
    safety are respected.
    """
    target_square = move.to_square

    captured_piece = board.piece_at(target_square)
    if captured_piece is None:
        if board.is_en_passant(move):
            captured_value = PIECE_VALUES[chess.PAWN]
        else:
            return 0
    else:
        captured_value = PIECE_VALUES.get(captured_piece.piece_type, 0)

    attacker_piece = board.piece_at(move.from_square)
    if attacker_piece is None:
        return 0

    if move not in board.legal_moves:
        return 0

    attacker_value = get_piece_value(attacker_piece)
    sim = board.copy(stack=False)
    sim.push(move)

    opponent_gain = _best_capture_sequence_gain(
        sim,
        target_square,
        sim.turn,
        attacker_value,
    )
    return captured_value - opponent_gain


def hangs_piece_after_move(board: chess.Board, move: chess.Move) -> bool:
    board_copy = board.copy(stack=False)
    board_copy.push(move)

    moved_piece = board_copy.piece_at(move.to_square)
    if moved_piece is None:
        return False

    opponent = board_copy.turn

    if not board_copy.is_attacked_by(opponent, move.to_square):
        return False

    atk_sq, _ = _least_valuable_attacker(board_copy, move.to_square, opponent)
    if atk_sq is None:
        return False

    recapture = chess.Move(atk_sq, move.to_square)
    see_result = static_exchange_evaluation(board_copy, recapture)
    return see_result >= 0


def find_hanging_pieces(board: chess.Board, color: chess.Color) -> list[tuple[int, int]]:
    """
    Returns pieces of `color` that the opponent can capture for net gain.
    """
    result = []
    opponent = not color

    for square, piece in board.piece_map().items():
        if piece.color != color:
            continue
        if not board.is_attacked_by(opponent, square):
            continue

        best_gain = _best_capture_sequence_gain(
            board,
            square,
            opponent,
            get_piece_value(piece),
        )
        if best_gain > 0:
            result.append((square, get_piece_value(piece)))

    return result


def immediate_material_threat(board: chess.Board, color: chess.Color) -> int:
    hanging = find_hanging_pieces(board, color)
    if not hanging:
        return 0
    return max(value for _, value in hanging)


def hanging_material_after_move(board: chess.Board, move: chess.Move) -> int:
    """
    Most valuable newly hanging piece the mover leaves after making `move`.
    Existing hanging pieces are ignored to avoid double penalties.
    """
    mover_color = board.turn
    hanging_before = {sq for sq, _ in find_hanging_pieces(board, mover_color)}

    board_copy = board.copy(stack=False)
    board_copy.push(move)

    hanging_after = find_hanging_pieces(board_copy, mover_color)
    new_hanging = [(sq, val) for sq, val in hanging_after if sq not in hanging_before]

    if not new_hanging:
        return 0
    return max(val for _, val in new_hanging)


def leaves_piece_hanging_after_move(board: chess.Board, move: chess.Move) -> bool:
    return hanging_material_after_move(board, move) > 0


def is_likely_recaptured(board: chess.Board, move: chess.Move) -> bool:
    board_copy = board.copy(stack=False)
    board_copy.push(move)
    opponent = board_copy.turn
    return board_copy.is_attacked_by(opponent, move.to_square)
