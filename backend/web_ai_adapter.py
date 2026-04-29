import random
import chess


PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


def get_position_evaluation(board: chess.Board) -> int:
    score = 0

    for piece_type, value in PIECE_VALUES.items():
        score += len(board.pieces(piece_type, chess.WHITE)) * value
        score -= len(board.pieces(piece_type, chess.BLACK)) * value

    return score


def get_ai_move(board: chess.Board, difficulty: str = "medium"):
    legal_moves = list(board.legal_moves)

    if not legal_moves:
        return None

    if difficulty == "easy":
        return random.choice(legal_moves)

    best_move = None
    best_score = None

    for move in legal_moves:
        board.push(move)
        score = get_position_evaluation(board)
        board.pop()

        # If it is White to move, White wants higher score.
        # If it is Black to move, Black wants lower score.
        if board.turn == chess.WHITE:
            adjusted_score = score
        else:
            adjusted_score = -score

        if best_score is None or adjusted_score > best_score:
            best_score = adjusted_score
            best_move = move

    if difficulty == "medium" and random.random() < 0.25:
        return random.choice(legal_moves)

    return best_move