from __future__ import annotations

import os
import random

import chess
import torch
import torch.nn as nn

from move_vocab import MoveVocab
from encoding import board_to_tensor
from model import PolicyCNN
from train import TrainConfig


cfg = TrainConfig()

MODEL_PATH = os.path.join(cfg.checkpoints_dir, "best_model.pt")
VOCAB_PATH = cfg.vocab_path
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TOP_K = 1
TEMPERATURE = 0.8

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100,
}


def board_to_extras(board: chess.Board) -> torch.Tensor:
    return torch.tensor(
        [
            1.0 if board.turn == chess.WHITE else 0.0,
            1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0,
            1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0,
            1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0,
            1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0,
            min(board.halfmove_clock, 100) / 100.0,
        ],
        dtype=torch.float32,
    )


def load_model():
    if not os.path.exists(VOCAB_PATH):
        raise FileNotFoundError(f"Vocab file not found: {VOCAB_PATH}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Checkpoint file not found: {MODEL_PATH}")

    vocab = MoveVocab.load(VOCAB_PATH)

    model = PolicyCNN(
        vocab_size=len(vocab),
        channels=cfg.channels,
        dropout=cfg.dropout,
    ).to(DEVICE)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    return model, vocab


def _captured_piece_type(board: chess.Board, move: chess.Move) -> int | None:
    captured_piece = board.piece_at(move.to_square)

    if captured_piece is not None:
        return captured_piece.piece_type

    if board.is_en_passant(move):
        return chess.PAWN

    return None


def _is_move_safe_enough(board: chess.Board, move: chess.Move) -> bool:
    board_copy = board.copy(stack=False)
    board_copy.push(move)

    moved_piece = board_copy.piece_at(move.to_square)
    if moved_piece is None:
        return False

    opponent_color = board_copy.turn
    mover_color = not opponent_color

    attacked = board_copy.is_attacked_by(opponent_color, move.to_square)
    defended = board_copy.is_attacked_by(mover_color, move.to_square)

    if attacked and not defended:
        return False

    return True


def _move_safety_penalty(board: chess.Board, move: chess.Move) -> float:
    board_copy = board.copy(stack=False)
    board_copy.push(move)

    moved_piece = board_copy.piece_at(move.to_square)
    if moved_piece is None:
        return -100.0

    opponent_color = board_copy.turn
    mover_color = not opponent_color

    attacked = board_copy.is_attacked_by(opponent_color, move.to_square)
    defended = board_copy.is_attacked_by(mover_color, move.to_square)

    if not attacked:
        return 0.0

    piece_value = PIECE_VALUES.get(moved_piece.piece_type, 1)

    if attacked and not defended:
        return -(piece_value * 2.5)

    enemy_attackers = list(board_copy.attackers(opponent_color, move.to_square))
    if enemy_attackers:
        cheapest_enemy = min(
            PIECE_VALUES.get(board_copy.piece_at(sq).piece_type, 1)
            for sq in enemy_attackers
            if board_copy.piece_at(sq) is not None
        )
        if cheapest_enemy < piece_value:
            return -(piece_value * 1.2)

    return -0.2


def tactical_capture_score(board: chess.Board, move: chess.Move) -> float:
    if not board.is_capture(move):
        return float("-inf")

    attacker = board.piece_at(move.from_square)
    if attacker is None:
        return float("-inf")

    captured_type = _captured_piece_type(board, move)
    if captured_type is None:
        return float("-inf")

    attacker_value = PIECE_VALUES.get(attacker.piece_type, 0)
    captured_value = PIECE_VALUES.get(captured_type, 0)

    safe_after_capture = _is_move_safe_enough(board, move)

    score = 0.0
    score += captured_value * 20
    score += (captured_value - attacker_value) * 8

    if safe_after_capture:
        score += 25
    else:
        score -= attacker_value * 6

    if captured_type == chess.QUEEN:
        score += 120
    elif captured_type == chess.ROOK:
        score += 50
    elif captured_type in (chess.BISHOP, chess.KNIGHT):
        score += 20

    return score


def find_best_tactical_move(board: chess.Board) -> chess.Move | None:
    legal_moves = list(board.legal_moves)
    capture_moves = [move for move in legal_moves if board.is_capture(move)]

    if not capture_moves:
        return None

    scored_moves = [(move, tactical_capture_score(board, move)) for move in capture_moves]
    scored_moves.sort(key=lambda x: x[1], reverse=True)

    best_move, best_score = scored_moves[0]

    if best_score >= 35:
        return best_move

    return None


@torch.no_grad()
def predict_legal_move(
    model: nn.Module,
    vocab: MoveVocab,
    board: chess.Board,
    top_k: int = TOP_K,
    temperature: float = TEMPERATURE,
) -> chess.Move:
    tactical_move = find_best_tactical_move(board)
    if tactical_move is not None:
        return tactical_move

    x = board_to_tensor(board).unsqueeze(0).to(DEVICE)
    extras = board_to_extras(board).unsqueeze(0).to(DEVICE)

    logits = model(x, extras).squeeze(0)

    legal_moves = list(board.legal_moves)
    scored_legal_moves: list[tuple[chess.Move, float]] = []

    unk_idx = vocab.stoi[vocab.UNK]

    for move in legal_moves:
        move_idx = vocab.encode(move)
        if move_idx == unk_idx:
            continue

        score = float(logits[move_idx].item())
        score += _move_safety_penalty(board, move)

        if board.gives_check(move):
            score += 0.5

        if board.is_capture(move):
            score += 0.3

        scored_legal_moves.append((move, score))

    if not scored_legal_moves:
        return random.choice(legal_moves)

    scored_legal_moves.sort(key=lambda x: x[1], reverse=True)

    k = max(1, min(top_k, len(scored_legal_moves)))
    candidates = scored_legal_moves[:k]

    if k == 1:
        return candidates[0][0]

    scores = torch.tensor([score for _, score in candidates], dtype=torch.float32)
    temperature = max(temperature, 1e-6)
    probs = torch.softmax(scores / temperature, dim=0).tolist()

    moves = [move for move, _ in candidates]
    return random.choices(moves, weights=probs, k=1)[0]


def ask_user_color() -> chess.Color:
    while True:
        choice = input("Play as white or black? [w/b]: ").strip().lower()
        if choice == "w":
            return chess.WHITE
        if choice == "b":
            return chess.BLACK
        print("Please type 'w' or 'b'.")


def ask_user_move(board: chess.Board) -> chess.Move:
    while True:
        user_input = input("Your move (UCI like e2e4, or 'quit'): ").strip().lower()

        if user_input in {"quit", "exit"}:
            raise SystemExit("Game ended.")

        try:
            move = chess.Move.from_uci(user_input)
        except ValueError:
            print("Invalid move format. Use UCI format like e2e4 or g1f3.")
            continue

        if move in board.legal_moves:
            return move

        print("Illegal move. Try again.")


def print_game_result(board: chess.Board):
    print("\nFinal board:")
    print(board)
    print()

    if board.is_checkmate():
        winner = "Black" if board.turn == chess.WHITE else "White"
        print(f"Checkmate. {winner} wins.")
    elif board.is_stalemate():
        print("Draw by stalemate.")
    elif board.is_insufficient_material():
        print("Draw by insufficient material.")
    elif board.is_seventyfive_moves():
        print("Draw by 75-move rule.")
    elif board.is_fivefold_repetition():
        print("Draw by fivefold repetition.")
    else:
        print("Game over.")


def main():
    model, vocab = load_model()
    board = chess.Board()

    human_color = ask_user_color()

    print("\nGame start.")
    print(f"AI settings: TOP_K={TOP_K}, TEMPERATURE={TEMPERATURE}")
    print("Board:")
    print(board)
    print()

    while not board.is_game_over():
        print("-" * 50)
        print(board)
        print()

        if board.turn == human_color:
            move = ask_user_move(board)
            board.push(move)
        else:
            ai_move = predict_legal_move(model, vocab, board)
            print(f"AI plays: {ai_move.uci()}")
            board.push(ai_move)

        if board.is_check():
            print("Check!")

    print_game_result(board)


if __name__ == "__main__":
    main()