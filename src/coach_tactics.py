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


def is_square_overloaded(board: chess.Board, square: int, color: chess.Color) -> bool:
    enemy = not color
    defenders = defenders_count(board, square, color)
    attackers = attackers_count(board, square, enemy)
    return attackers > defenders


def hangs_piece_after_move(board: chess.Board, move: chess.Move) -> bool:
    board_copy = board.copy(stack=False)
    board_copy.push(move)

    moved_piece = board_copy.piece_at(move.to_square)
    if moved_piece is None:
        return False

    color = moved_piece.color
    enemy = not color

    attacked = board_copy.is_attacked_by(enemy, move.to_square)
    defended = board_copy.is_attacked_by(color, move.to_square)

    if attacked and not defended:
        return True

    if attacked and defended:
        moved_value = get_piece_value(moved_piece)
        enemy_attackers = list(board_copy.attackers(enemy, move.to_square))
        if enemy_attackers:
            cheapest_enemy = min(
                get_piece_value(board_copy.piece_at(sq))
                for sq in enemy_attackers
                if board_copy.piece_at(sq)
            )
            if cheapest_enemy < moved_value:
                return True

    return False


def find_hanging_pieces(board: chess.Board, color: chess.Color) -> list[tuple[int, int]]:
    result = []

    for square, piece in board.piece_map().items():
        if piece.color != color:
            continue

        enemy = not color
        attacked = board.is_attacked_by(enemy, square)
        defended = board.is_attacked_by(color, square)

        if attacked and not defended:
            result.append((square, get_piece_value(piece)))

    return result


def immediate_material_threat(board: chess.Board, color: chess.Color) -> int:
    hanging = find_hanging_pieces(board, color)
    if not hanging:
        return 0
    return max(value for _, value in hanging)


def hanging_material_after_move(board: chess.Board, move: chess.Move) -> int:
    board_copy = board.copy(stack=False)
    board_copy.push(move)

    mover_color = not board_copy.turn
    hanging = find_hanging_pieces(board_copy, mover_color)
    if not hanging:
        return 0

    return max(value for _, value in hanging)


def leaves_piece_hanging_after_move(board: chess.Board, move: chess.Move) -> bool:
    return hanging_material_after_move(board, move) > 0
