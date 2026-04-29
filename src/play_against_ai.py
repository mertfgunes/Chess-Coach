from __future__ import annotations

import os
import random

import chess
import torch
import torch.nn as nn

from coach_tactics import (
    hanging_material_after_move,
    is_likely_recaptured,
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

TOP_K = 5
REPLY_TOP_K = 3
TEMPERATURE = 0.8
VALUE_WEIGHT = 3.0

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

    [0] side_to_move (1 if white, 0 if black)
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
# SEE-based safety helpers
# ---------------------------------------------------------------------------

def _see_score(board: chess.Board, move: chess.Move) -> int:
    """
    Wrapper around SEE.  For non-captures returns 0.
    Positive = the moving side gains material.
    """
    return static_exchange_evaluation(board, move)


def _is_move_safe_enough(board: chess.Board, move: chess.Move) -> bool:
    """
    A move is considered 'safe enough' when:
      - If it is a capture: SEE >= 0  (we don't lose material in the exchange)
      - If it is not a capture: the moved piece does not end up newly hanging
    """
    if board.is_capture(move):
        return _see_score(board, move) >= 0

    # Non-capture: check if the piece lands on a square where opponent wins
    board_copy = board.copy(stack=False)
    board_copy.push(move)
    opponent = board_copy.turn
    if not board_copy.is_attacked_by(opponent, move.to_square):
        return True

    # Simulate an opponent capture from the new position
    from coach_tactics import _least_valuable_attacker
    atk_sq, _ = _least_valuable_attacker(board_copy, move.to_square, opponent)
    if atk_sq is None:
        return True

    recapture = chess.Move(atk_sq, move.to_square)
    return static_exchange_evaluation(board_copy, recapture) < 0  # opponent loses


def _move_safety_penalty(board: chess.Board, move: chess.Move) -> float:
    """
    Returns a penalty (negative float, in policy-logit units) for moves
    that lose material.  Uses SEE for captures and hanging detection for
    quiet moves, so the penalty is proportional to how much material is lost.
    """
    if board.is_capture(move):
        see = _see_score(board, move)
        if see < 0:
            # Losing capture — penalty proportional to material lost
            return float(see) * 1.5
        return 0.0  # winning or even capture — no penalty

    # Quiet move: penalise if it leaves a piece newly hanging
    hanging_value = hanging_material_after_move(board, move)
    if hanging_value > 0:
        return -float(hanging_value) * 2.0

    return 0.0


# ---------------------------------------------------------------------------
# Tactical override: find a clearly winning capture
# ---------------------------------------------------------------------------

def _tactical_capture_score(board: chess.Board, move: chess.Move) -> float:
    """
    Score a capture move for the tactical override.
    Based purely on SEE so we don't double-count policy scores.
    Returns -inf for non-captures.
    """
    if not board.is_capture(move):
        return float("-inf")

    see = _see_score(board, move)
    if see <= 0:
        return float("-inf")  # even or losing capture — skip tactical override

    # Scale by material gain; add small bonus for higher-value captures
    captured = board.piece_at(move.to_square)
    captured_val = PIECE_VALUES.get(captured.piece_type, 0) if captured else PIECE_VALUES[chess.PAWN]

    return float(see * 10 + captured_val)


def find_best_tactical_move(board: chess.Board) -> chess.Move | None:
    """
    Returns a clearly winning capture (SEE > 0) if one exists, else None.
    We only override the neural network if we are unambiguously winning
    material — avoids the old bug where a losing queen capture scored +300.
    """
    legal_moves = list(board.legal_moves)
    capture_moves = [m for m in legal_moves if board.is_capture(m)]
    if not capture_moves:
        return None

    scored = [
        (m, _tactical_capture_score(board, m))
        for m in capture_moves
    ]
    scored = [(m, s) for m, s in scored if s > float("-inf")]
    if not scored:
        return None

    scored.sort(key=lambda x: x[1], reverse=True)
    best_move, best_score = scored[0]

    # Require a meaningful material gain (at least 1 pawn equivalent via SEE)
    if _see_score(board, best_move) >= 1:
        return best_move

    return None


# ---------------------------------------------------------------------------
# Policy scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_board_value(model: nn.Module, board: chess.Board) -> float:
    """
    Returns value in White-centric form:
      +1 White is winning, -1 Black is winning.
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
    Score legal moves using policy head + SEE-based safety adjustments.
    Returns (move, score) sorted descending (higher = better for side to move).
    """
    x = board_to_tensor(board).unsqueeze(0).to(DEVICE)
    extras = board_to_extras(board).unsqueeze(0).to(DEVICE)

    policy_logits, _value_pred = model(x, extras)
    logits = policy_logits.squeeze(0)

    legal_moves = list(board.legal_moves)
    unk_idx = vocab.stoi[vocab.UNK]
    scored_moves: list[tuple[chess.Move, float]] = []

    for move in legal_moves:
        move_idx = vocab.encode(move)
        if move_idx == unk_idx:
            continue

        score = float(logits[move_idx].item())

        # --- Safety penalty (SEE-based) ---
        score += _move_safety_penalty(board, move)

        # --- Check bonus: tiny and only when the check is also SEE-safe ---
        # A check is only a bonus if:
        #   (a) it is a safe capture (SEE >= 0), or
        #   (b) it is a quiet move that doesn't leave us hanging
        # We intentionally keep this bonus small so the model's own
        # evaluation drives move choice, not a hard-coded check bonus.
        if board.gives_check(move):
            if _is_move_safe_enough(board, move):
                score += 0.05   # tiny nudge — check is safe but not inherently great
            else:
                score -= 0.5    # unsafe check is actively penalised

        # --- Capture bonus proportional to SEE gain ---
        # Replaces the old flat +0.25 bonus that didn't account for recaptures.
        if board.is_capture(move):
            see = _see_score(board, move)
            if see > 0:
                # Winning capture — small bonus scaled by gain (capped to avoid over-weighting)
                score += min(see * 0.15, 1.0)
            elif see < 0:
                # Already penalised by _move_safety_penalty; no further bonus
                pass
            # see == 0: even exchange — no bonus, no penalty

        scored_moves.append((move, score))

    if not scored_moves and legal_moves:
        return [(m, 0.0) for m in legal_moves]

    scored_moves.sort(key=lambda x: x[1], reverse=True)
    return scored_moves


def value_from_side_to_move_perspective(
    board_before_move: chess.Board, white_value: float
) -> float:
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
    # Tactical override: only when SEE confirms a clear material gain
    tactical_move = find_best_tactical_move(board)
    if tactical_move is not None:
        return tactical_move

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        raise ValueError("No legal moves available.")

    # Step 1: shortlist candidate moves via policy + safety adjustments
    scored_moves = get_policy_scored_legal_moves(model, vocab, board)
    if not scored_moves:
        return random.choice(legal_moves)

    k = max(1, min(top_k, len(scored_moves)))
    candidates = scored_moves[:k]

    best_move = None
    best_score = float("-inf")

    # Step 2: for each candidate, simulate opponent best replies
    for ai_move, ai_policy_score in candidates:
        board_after_ai = board.copy(stack=False)
        board_after_ai.push(ai_move)

        # Immediate checkmate is always best
        if board_after_ai.is_checkmate():
            return ai_move

        # Terminal position (stalemate, draw, etc.)
        if board_after_ai.is_game_over():
            white_value = evaluate_board_value(model, board_after_ai)
            move_score = value_from_side_to_move_perspective(board, white_value)
            move_score += 0.25 * ai_policy_score
            if move_score > best_score:
                best_score = move_score
                best_move = ai_move
            continue

        # Step 3: opponent best-reply shortlist
        reply_scored_moves = get_policy_scored_legal_moves(model, vocab, board_after_ai)

        if not reply_scored_moves:
            white_value = evaluate_board_value(model, board_after_ai)
            move_score = value_from_side_to_move_perspective(board, white_value)
            move_score += 0.25 * ai_policy_score
            if move_score > best_score:
                best_score = move_score
                best_move = ai_move
            continue

        reply_k = max(1, min(REPLY_TOP_K, len(reply_scored_moves)))
        reply_candidates = reply_scored_moves[:reply_k]

        # Pessimistic assumption: opponent picks the reply that hurts us most
        worst_reply_score = float("inf")

        for opp_move, _opp_policy_score in reply_candidates:
            board_after_reply = board_after_ai.copy(stack=False)
            board_after_reply.push(opp_move)

            white_value = evaluate_board_value(model, board_after_reply)
            ai_perspective = value_from_side_to_move_perspective(board, white_value)

            if ai_perspective < worst_reply_score:
                worst_reply_score = ai_perspective

        combined_score = (VALUE_WEIGHT * worst_reply_score) + (0.25 * ai_policy_score)

        if combined_score > best_score:
            best_score = combined_score
            best_move = ai_move

    return best_move if best_move is not None else random.choice(legal_moves)


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
    print(
        f"AI settings: TOP_K={TOP_K}, REPLY_TOP_K={REPLY_TOP_K}, "
        f"TEMPERATURE={TEMPERATURE}, VALUE_WEIGHT={VALUE_WEIGHT}"
    )
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
