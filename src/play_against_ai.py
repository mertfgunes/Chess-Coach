from __future__ import annotations

import os
import random
from pathlib import Path

import chess
import torch
import torch.nn as nn

from coach_evaluation import evaluate_position
from coach_tactics import (
    immediate_material_threat,
    hanging_material_after_move,
    static_exchange_evaluation,
)
from move_vocab import MoveVocab
from encoding import board_to_tensor
from model import PolicyCNN


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "checkpoints" / "best_model.pt"
VOCAB_PATH = ROOT_DIR / "data" / "move_vocab.txt"
DEFAULT_CHANNELS = 128
DEFAULT_DROPOUT = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Wider candidate lists let the model generate ideas while the chess logic
# rejects obvious tactics that fail on the next move.
TOP_K = 12
TEMPERATURE = 0.35
CANDIDATE_POOL = 24

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
    vocab_path = Path(os.environ.get("CHESS_COACH_VOCAB_PATH", VOCAB_PATH))
    model_path = Path(os.environ.get("CHESS_COACH_MODEL_PATH", MODEL_PATH))

    if not vocab_path.is_absolute():
        vocab_path = ROOT_DIR / vocab_path
    if not model_path.is_absolute():
        model_path = ROOT_DIR / model_path

    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocab file not found: {vocab_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {model_path}")

    vocab = MoveVocab.load(str(vocab_path))
    checkpoint = torch.load(model_path, map_location=DEVICE)
    checkpoint_cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    channels = int(checkpoint_cfg.get("channels", DEFAULT_CHANNELS))
    dropout = float(checkpoint_cfg.get("dropout", DEFAULT_DROPOUT))

    model = PolicyCNN(
        vocab_size=len(vocab),
        channels=channels,
        dropout=dropout,
    ).to(DEVICE)

    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint does not match the current model architecture. "
            f"Loaded config channels={channels}, dropout={dropout}, "
            f"vocab_size={len(vocab)} from {model_path}."
        ) from exc

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


def immediate_blunder_loss(board: chess.Board, move: chess.Move) -> int:
    """
    Pawn-unit estimate of material the opponent can win right after this move.

    This is intentionally a blunt final gate, not a full engine. It catches the
    expensive one-ply mistakes that make the bot drop bishops, rooks, or queens.
    """
    if move not in board.legal_moves:
        return PIECE_VALUES[chess.QUEEN]

    if board.is_capture(move):
        see = static_exchange_evaluation(board, move)
        if see < 0:
            return abs(see)

    mover = board.turn
    existing_threat = immediate_material_threat(board, mover)
    new_hanging = hanging_material_after_move(board, move)

    board_after = board.copy(stack=False)
    board_after.push(move)

    if board_after.is_checkmate():
        return 0

    remaining_threat = immediate_material_threat(board_after, mover)
    unresolved_threat = remaining_threat if remaining_threat >= existing_threat else 0

    return max(new_hanging, unresolved_threat)


def is_candidate_blunder(board: chess.Board, move: chess.Move) -> bool:
    loss = immediate_blunder_loss(board, move)

    if loss < PIECE_VALUES[chess.BISHOP]:
        return False

    return True


def filter_blunder_candidates(
    board: chess.Board,
    scored_moves: list[tuple[chess.Move, float]],
) -> list[tuple[chess.Move, float]]:
    safe_moves = [
        (move, score)
        for move, score in scored_moves
        if not is_candidate_blunder(board, move)
    ]

    return safe_moves or scored_moves


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


def heuristic_score_for_color(board: chess.Board, color: chess.Color) -> float:
    """
    Position score in pawn units from one side's perspective.

    Mate scores are intentionally much larger than normal eval terms so the
    selector never ignores checkmate or walks into it for a material nibble.
    """
    if board.is_checkmate():
        return -100.0 if board.turn == color else 100.0

    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0

    white_score = evaluate_position(board).total
    return white_score if color == chess.WHITE else -white_score


def opponent_reply_penalty(
    board_after_move: chess.Board,
    mover_color: chess.Color,
    reply_limit: int = 12,
) -> float:
    """
    Estimate how much the opponent can damage the move on the next ply.

    This is the key strength upgrade: captures and checks are no longer judged
    only by their immediate appearance. If the opponent has a strong recapture
    or tactic, this penalty pulls the candidate down.
    """
    if board_after_move.is_game_over():
        return 0.0

    current_score = heuristic_score_for_color(board_after_move, mover_color)
    replies = list(board_after_move.legal_moves)

    if not replies:
        return 0.0

    def reply_priority(reply: chess.Move) -> float:
        priority = 0.0
        if board_after_move.is_capture(reply):
            priority += 4.0 + max(0, static_exchange_evaluation(board_after_move, reply))
        if board_after_move.gives_check(reply):
            priority += 1.0
        return priority

    replies.sort(key=reply_priority, reverse=True)
    worst_drop = 0.0

    for reply in replies[:reply_limit]:
        reply_board = board_after_move.copy(stack=False)
        reply_board.push(reply)
        reply_score = heuristic_score_for_color(reply_board, mover_color)
        worst_drop = max(worst_drop, current_score - reply_score)

    return worst_drop


def search_adjustment_for_move(
    board: chess.Board,
    move: chess.Move,
    reply_limit: int,
) -> float:
    mover_color = board.turn
    board_after = board.copy(stack=False)
    board_after.push(move)

    if board_after.is_checkmate():
        return 100.0

    position_score = heuristic_score_for_color(board_after, mover_color)
    reply_penalty = opponent_reply_penalty(
        board_after,
        mover_color,
        reply_limit=reply_limit,
    )

    return position_score - (0.85 * reply_penalty)


def difficulty_settings(difficulty: str | None) -> dict[str, float | int | bool]:
    level = (difficulty or "medium").lower()

    if level == "easy":
        return {
            "top_k": 8,
            "candidate_pool": 14,
            "temperature": 0.85,
            "reply_limit": 5,
            "search_weight": 0.35,
            "random_chance": 0.18,
            "deterministic": False,
        }

    if level == "hard":
        return {
            "top_k": 1,
            "candidate_pool": 32,
            "temperature": 0.05,
            "reply_limit": 18,
            "search_weight": 1.15,
            "random_chance": 0.0,
            "deterministic": True,
        }

    return {
        "top_k": TOP_K,
        "candidate_pool": CANDIDATE_POOL,
        "temperature": TEMPERATURE,
        "reply_limit": 12,
        "search_weight": 0.75,
        "random_chance": 0.04,
        "deterministic": False,
    }


def apply_opening_variety(
    board: chess.Board,
    settings: dict[str, float | int | bool],
) -> dict[str, float | int | bool]:
    """
    Let the AI choose among similarly valued opening moves, while still keeping
    deterministic precision once a move is clearly stronger.
    """
    if board.fullmove_number > 7:
        return settings

    varied = dict(settings)
    varied["top_k"] = max(int(varied["top_k"]), 4)
    return varied


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
    difficulty: str | None = "medium",
) -> chess.Move:
    """
    Predicts one legal AI move.

    The model generates candidate moves; python-chess filters legality; then a
    shallow reply check and exchange safety rerank the strongest candidates.
    """
    legal_moves = list(board.legal_moves)

    if not legal_moves:
        raise ValueError("No legal moves available.")

    settings = apply_opening_variety(board, difficulty_settings(difficulty))
    if top_k == TOP_K:
        top_k = int(settings["top_k"])
    if temperature == TEMPERATURE:
        temperature = float(settings["temperature"])

    scored_moves = get_policy_scored_legal_moves(model, vocab, board)

    if not scored_moves:
        return random.choice(legal_moves)

    pool_size = max(1, min(int(settings["candidate_pool"]), len(scored_moves)))
    search_weight = float(settings["search_weight"])
    reply_limit = int(settings["reply_limit"])
    reranked: list[tuple[chess.Move, float]] = []

    for move, policy_score in scored_moves[:pool_size]:
        search_score = search_adjustment_for_move(
            board,
            move,
            reply_limit=reply_limit,
        )
        final_score = policy_score + (search_weight * search_score)
        reranked.append((move, final_score))

    reranked.sort(key=lambda item: item[1], reverse=True)
    reranked = filter_blunder_candidates(board, reranked)

    if random.random() < float(settings["random_chance"]):
        weaker_pool = reranked[: max(1, min(6, len(reranked)))]
        return random.choice(weaker_pool)[0]

    k = max(1, min(top_k, len(reranked)))
    candidates = reranked[:k]

    if len(candidates) == 1 or bool(settings["deterministic"]):
        return candidates[0][0]

    best_score = candidates[0][1]
    second_score = candidates[1][1]

    if board.fullmove_number <= 7:
        close_candidates = [
            move
            for move, score in candidates
            if best_score - score <= 0.12
        ]
        if len(close_candidates) > 1:
            return random.choice(close_candidates)

    # If the best move is clearly better, play it.
    if best_score - second_score >= 0.85:
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
