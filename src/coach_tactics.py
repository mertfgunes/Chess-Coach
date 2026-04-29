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


# ---------------------------------------------------------------------------
# Static Exchange Evaluation (SEE)
# ---------------------------------------------------------------------------
# Returns the material gain/loss (in pawn units) for a capture on `square`
# by `side`, assuming both sides recapture with their least-valuable pieces.
# Positive = the capturing side wins material.
# Negative = the capturing side loses material.
# ---------------------------------------------------------------------------

def _least_valuable_attacker(
    board: chess.Board, square: int, color: chess.Color
) -> tuple[int | None, int]:
    """
    Find the least-valuable piece of `color` that attacks `square`.
    Returns (from_square, piece_value) or (None, 0) if no attacker exists.
    """
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


def static_exchange_evaluation(board: chess.Board, move: chess.Move) -> int:
    """
    Estimate the net material gain for the side making `move`.

    Positive  → capturing side comes out ahead (or even).
    Negative  → capturing side loses material.
    Zero      → even exchange or no capture.

    The implementation simulates the full recapture sequence using each
    side's least-valuable attacker, consistent with standard SEE algorithms.
    """
    target_square = move.to_square

    # Value of the piece being captured (0 if not a capture)
    captured_piece = board.piece_at(target_square)
    if captured_piece is None:
        if board.is_en_passant(move):
            captured_value = PIECE_VALUES[chess.PAWN]
        else:
            return 0  # not a capture
    else:
        captured_value = PIECE_VALUES.get(captured_piece.piece_type, 0)

    # Value of the piece doing the capturing
    attacker_piece = board.piece_at(move.from_square)
    if attacker_piece is None:
        return 0
    attacker_value = PIECE_VALUES.get(attacker_piece.piece_type, 0)

    # gains[0] = what the capturing side gains on the first capture
    gains: list[int] = [captured_value]

    # Simulate the board after the first capture
    sim = board.copy(stack=False)
    sim.push(move)

    # Alternate sides recapturing with least-valuable piece
    current_value_on_square = attacker_value
    side = sim.turn  # opponent goes next

    for _ in range(30):  # safety cap – a recapture chain can't exceed 30 plies
        atk_sq, atk_val = _least_valuable_attacker(sim, target_square, side)
        if atk_sq is None:
            break  # no more attackers for this side

        # This side captures; they gain whatever is currently on the square
        gains.append(current_value_on_square)
        current_value_on_square = atk_val

        # Push the recapture
        recap_move = chess.Move(atk_sq, target_square)
        if recap_move in sim.legal_moves:
            sim.push(recap_move)
        else:
            # Handle promotion or special cases gracefully
            break

        side = sim.turn

    # Minimax back through the gains array
    # Each side will only recapture if it improves their result
    while len(gains) > 1:
        gains[-2] = max(-gains[-1], gains[-2])
        gains.pop()

    return gains[0]


# ---------------------------------------------------------------------------
# Hanging-piece helpers
# ---------------------------------------------------------------------------

def hangs_piece_after_move(board: chess.Board, move: chess.Move) -> bool:
    """
    Returns True if the piece that just moved ends up on an undefended square
    attacked by a cheaper enemy piece (i.e. it would lose material if taken).
    Uses SEE so it correctly accounts for recaptures.
    """
    board_copy = board.copy(stack=False)
    board_copy.push(move)

    moved_piece = board_copy.piece_at(move.to_square)
    if moved_piece is None:
        return False

    opponent = board_copy.turn  # side to move after the push = opponent

    if not board_copy.is_attacked_by(opponent, move.to_square):
        return False

    # Build a hypothetical recapture move and run SEE from opponent's POV
    atk_sq, _ = _least_valuable_attacker(board_copy, move.to_square, opponent)
    if atk_sq is None:
        return False

    recapture = chess.Move(atk_sq, move.to_square)
    see_result = static_exchange_evaluation(board_copy, recapture)
    # If SEE ≥ 0 for the opponent, the opponent benefits → piece hangs
    return see_result >= 0


def find_hanging_pieces(board: chess.Board, color: chess.Color) -> list[tuple[int, int]]:
    """
    Returns a list of (square, piece_value) for pieces of `color` that are
    hanging — i.e. the opponent can capture them with a net material gain.
    """
    result = []
    opponent = not color

    for square, piece in board.piece_map().items():
        if piece.color != color:
            continue
        if not board.is_attacked_by(opponent, square):
            continue

        atk_sq, _ = _least_valuable_attacker(board, square, opponent)
        if atk_sq is None:
            continue

        capture = chess.Move(atk_sq, square)
        if static_exchange_evaluation(board, capture) > 0:
            result.append((square, PIECE_VALUES.get(piece.piece_type, 0)))

    return result


def immediate_material_threat(board: chess.Board, color: chess.Color) -> int:
    hanging = find_hanging_pieces(board, color)
    if not hanging:
        return 0
    return max(value for _, value in hanging)


def hanging_material_after_move(board: chess.Board, move: chess.Move) -> int:
    """
    Returns the value of the most valuable piece the mover leaves hanging
    AFTER making `move`, excluding pieces that were already hanging BEFORE
    the move (to avoid double-penalising pre-existing problems).
    """
    mover_color = board.turn

    # Pieces already hanging before the move
    hanging_before = {sq for sq, _ in find_hanging_pieces(board, mover_color)}

    board_copy = board.copy(stack=False)
    board_copy.push(move)

    hanging_after = find_hanging_pieces(board_copy, mover_color)

    # Only count pieces newly hanging (not already hanging before)
    new_hanging = [(sq, val) for sq, val in hanging_after if sq not in hanging_before]

    if not new_hanging:
        return 0
    return max(val for _, val in new_hanging)


def leaves_piece_hanging_after_move(board: chess.Board, move: chess.Move) -> bool:
    return hanging_material_after_move(board, move) > 0


def is_likely_recaptured(board: chess.Board, move: chess.Move) -> bool:
    """
    Returns True if the opponent has at least one attacker on the destination
    square after the move.  Used as a lightweight signal (not a full SEE).
    """
    board_copy = board.copy(stack=False)
    board_copy.push(move)
    opponent = board_copy.turn
    return board_copy.is_attacked_by(opponent, move.to_square)
