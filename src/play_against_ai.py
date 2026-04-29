from __future__ import annotations

import os
import random

import chess
import torch
import torch.nn as nn

from coach_tactics import (
    hanging_material_after_move,
    static_exchange_evaluation,
)
from move_vocab import MoveVocab
from encoding import board_to_tensor
from model import PolicyCNN
from train import TrainConfig


cfg = TrainConfig()

MODEL_PATH = os.path.join(cfg.checkpoints_dir, "best_model.pt")
VOCAB_PATH = cfg.vocab_path
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Smaller top-k and lower temperature = stronger / less random.
TOP_K = 4
TEMPERATURE = 0.55

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100,
}


def board_to_extras(board: chess.Board) -> torch.Tensor:
    """
    Must match training-time extra features exactly.

    [0] side_to_move, 1 if white, 0 if black
    [1] white_can_castle_k
    [2] white_can_castle_q
    [3] black_can_castle_k
    [4] black_can_castle_q
    [5] en_passant_file normalized to [0..1], or 0 if none
    """
    side = 1.0 if board.turn == chess.WHITE else 0.0
    wck = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    wcq = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    bck = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    bcq = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

    ep = board.ep_square
    ep_file = chess.square_file(ep) / 7.0 if ep is not None else 0.0

    return torch.tensor([side, wck, wcq, bck, bcq, ep_file], dtype=torch.float32)


def load_model():
    """
    Loads the trained policy/value model and move vocabulary.
    """
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

    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model, vocab


# ---------------------------------------------------------------------------
# Light tactical safety helpers
# ---------------------------------------------------------------------------

def _see_score(board: chess.Board, move: chess.Move) -> int:
    """
    Static Exchange Evaluation wrapper.

    Positive = moving side wins material.
    Negative = moving side loses material.
    Zero = equal exchange or non-capture.
    """
    return static_exchange_evaluation(board, move)


def _is_move_safe_enough(board: chess.Board, move: chess.Move) -> bool:
    """
    Soft safety check.

    This should not replace the neural model.
    It only checks whether a move is obviously unsafe.
    """
    if board.is_capture(move):
        return _see_score(board, move) >= 0

    hanging_value = hanging_material_after_move(board, move)
    return hanging_value == 0


def _move_safety_adjustment(board: chess.Board, move: chess.Move) -> float:
    """
    Small adjustment added to the neural model's policy score.

    Important:
    This must stay mild. If this is too strong, the hand-written chess logic
    starts overpowering the trained model and the AI can become worse.
    """
    adjustment = 0.0

    if board.is_capture(move):
        see = _see_score(board, move)

        if see < 0:
            # Losing capture.
            # Mild penalty only.
            adjustment += float(see) * 0.5

        elif see > 0:
            # Winning capture.
            # Small reward, capped.
            adjustment += min(float(see) * 0.10, 0.5)

    else:
        hanging_value = hanging_material_after_move(board, move)

        if hanging_value > 0:
            # Quiet move leaves a new piece hanging.
            # Mild penalty only.
            adjustment -= float(hanging_value) * 0.7

    if board.gives_check(move):
        if _is_move_safe_enough(board, move):
            # Tiny check bonus.
            # Checks should not dominate the model.
            adjustment += 0.03
        else:
            # Unsafe checks should be discouraged.
            adjustment -= 0.25

    return adjustment


def opening_move_bonus(board: chess.Board, move: chess.Move) -> float:
    """
    Small opening-principle bonus.

    This does not force an opening book.
    It only nudges the AI toward normal chess development in the opening.
    """
    if board.fullmove_number > 10:
        return 0.0

    piece = board.piece_at(move.from_square)

    if piece is None:
        return 0.0

    bonus = 0.0
    uci = move.uci()

    # Common strong opening pawn moves.
    good_pawn_moves = {
        "e2e4", "d2d4", "c2c4",
        "e7e5", "d7d5", "c7c5",
    }

    # Common natural knight development.
    good_knight_moves = {
        "g1f3", "b1c3",
        "g8f6", "b8c6",
    }

    # Encourage central pawn control.
    if piece.piece_type == chess.PAWN:
        if uci in good_pawn_moves:
            bonus += 0.25

        if move.to_square in {chess.D4, chess.E4, chess.D5, chess.E5}:
            bonus += 0.15

    # Encourage developing knights.
    if piece.piece_type == chess.KNIGHT:
        if uci in good_knight_moves:
            bonus += 0.25
        else:
            from_rank = chess.square_rank(move.from_square)

            if piece.color == chess.WHITE and from_rank == 0:
                bonus += 0.12

            if piece.color == chess.BLACK and from_rank == 7:
                bonus += 0.12

    # Encourage developing bishops from their starting rank.
    if piece.piece_type == chess.BISHOP:
        from_rank = chess.square_rank(move.from_square)

        if piece.color == chess.WHITE and from_rank == 0:
            bonus += 0.18

        if piece.color == chess.BLACK and from_rank == 7:
            bonus += 0.18

    # Discourage early queen adventures.
    if piece.piece_type == chess.QUEEN and board.fullmove_number <= 6:
        bonus -= 0.35

    # Encourage castling.
    if board.is_castling(move):
        bonus += 0.45

    # Slightly discourage moving the same piece many times in the opening.
    if board.fullmove_number <= 8 and len(board.move_stack) >= 2:
        last_move = board.peek()
        if last_move.to_square == move.from_square:
            bonus -= 0.10

    return bonus


# ---------------------------------------------------------------------------
# Model scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_board_value(model: nn.Module, board: chess.Board) -> float:
    """
    Returns value in White-centric form:
    +1 means White is better.
    -1 means Black is better.

    This is kept for compatibility, but predict_legal_move does not use
    heavy value-head search anymore.
    """
    x = board_to_tensor(board).unsqueeze(0).to(DEVICE)
    extras = board_to_extras(board).unsqueeze(0).to(DEVICE)

    _policy_logits, value_pred = model(x, extras)

    return float(value_pred.item())


@torch.no_grad()
def get_policy_scored_legal_moves(
    model: nn.Module,
    vocab: MoveVocab,
    board: chess.Board,
) -> list[tuple[chess.Move, float]]:
    """
    Scores legal moves using:

    neural policy score
    + light tactical safety adjustment
    + small opening-principle bonus

    Higher score = better move according to the combined score.
    """
    legal_moves = list(board.legal_moves)

    if not legal_moves:
        return []

    x = board_to_tensor(board).unsqueeze(0).to(DEVICE)
    extras = board_to_extras(board).unsqueeze(0).to(DEVICE)

    policy_logits, _value_pred = model(x, extras)
    logits = policy_logits.squeeze(0)

    unk_idx = vocab.stoi[vocab.UNK]
    scored_moves: list[tuple[chess.Move, float]] = []

    for move in legal_moves:
        move_idx = vocab.encode(move)

        if move_idx == unk_idx:
            continue

        model_score = float(logits[move_idx].item())
        safety_adjustment = _move_safety_adjustment(board, move)
        opening_adjustment = opening_move_bonus(board, move)

        final_score = model_score + safety_adjustment + opening_adjustment

        scored_moves.append((move, final_score))

    if not scored_moves:
        # If every move is unknown to the vocab, fall back safely.
        return [(move, 0.0) for move in legal_moves]

    scored_moves.sort(key=lambda item: item[1], reverse=True)
    return scored_moves


def value_from_side_to_move_perspective(
    board_before_move: chess.Board,
    white_value: float,
) -> float:
    """
    Converts White-centric value into the perspective of the side to move.
    Kept for compatibility with older code.
    """
    return white_value if board_before_move.turn == chess.WHITE else -white_value


# ---------------------------------------------------------------------------
# Main move prediction
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_legal_move(
    model: nn.Module,
    vocab: MoveVocab,
    board: chess.Board,
    top_k: int = TOP_K,
    temperature: float = TEMPERATURE,
) -> chess.Move:
    """
    Predicts one legal AI move.

    New approach:
    - Model remains the main decision-maker.
    - SEE only gives light safety adjustments.
    - Opening bonus gives small early-game guidance.
    - No hard tactical override.
    - No heavy opponent-reply search.
    - No strong value-head weighting.
    """
    legal_moves = list(board.legal_moves)

    if not legal_moves:
        raise ValueError("No legal moves available.")

    scored_moves = get_policy_scored_legal_moves(model, vocab, board)

    if not scored_moves:
        return random.choice(legal_moves)

    k = max(1, min(top_k, len(scored_moves)))
    candidates = scored_moves[:k]

    if len(candidates) == 1:
        return candidates[0][0]

    best_score = candidates[0][1]
    second_score = candidates[1][1]

    # If the best move is clearly better, play it.
    if best_score - second_score >= 1.25:
        return candidates[0][0]

    # Otherwise sample between the top candidate moves.
    # This gives variety without making the bot fully random.
    scores = torch.tensor([score for _, score in candidates], dtype=torch.float32)

    safe_temperature = max(float(temperature), 0.01)
    probs = torch.softmax(scores / safe_temperature, dim=0)

    chosen_index = torch.multinomial(probs, num_samples=1).item()

    return candidates[chosen_index][0]


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

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
        user_input = input("Your move, UCI like e2e4, or 'quit': ").strip().lower()

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
    print(board)

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