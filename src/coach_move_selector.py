from __future__ import annotations

from typing import List

import chess
import torch

from coach_models import MoveSuggestion
from coach_evaluation import evaluate_position
from encoding import board_to_tensor
from play_against_ai import board_to_extras, DEVICE, search_adjustment_for_move
from coach_tactics import (
    hangs_piece_after_move,
    hanging_material_after_move,
    leaves_piece_hanging_after_move,
    static_exchange_evaluation,
)


@torch.no_grad()
def get_top_moves(
    model,
    vocab,
    board: chess.Board,
    top_n: int = 3,
) -> List[MoveSuggestion]:
    """
    Returns top coach move suggestions.

    This version:
    - keeps the model policy as an important signal
    - fixes Black perspective by converting evaluation to side-to-move view
    - uses SEE for captures
    - uses softer penalties so the coach does not overreact
    - gives better tags and explanations
    """
    if model is None or vocab is None:
        return []

    x = board_to_tensor(board).unsqueeze(0).to(DEVICE)
    extras = board_to_extras(board).unsqueeze(0).to(DEVICE)

    policy_logits, _ = model(x, extras)
    logits = policy_logits.squeeze(0)

    legal_moves = list(board.legal_moves)
    unk_idx = vocab.stoi[vocab.UNK]

    scored_moves = []

    for move in legal_moves:
        idx = vocab.encode(move)

        if idx == unk_idx:
            continue

        policy_score = float(logits[idx].item())

        board_copy = board.copy(stack=False)
        san = board.san(move)
        board_copy.push(move)

        # evaluate_position returns White-centric score.
        # Convert to side-to-move perspective so higher is better for both colors.
        raw_eval = evaluate_position(board_copy).total
        eval_score = raw_eval if board.turn == chess.WHITE else -raw_eval

        tactical_adjustment = 0.0

        # SEE-based capture adjustment.
        if board.is_capture(move):
            see = static_exchange_evaluation(board, move)

            if see < 0:
                # Losing capture.
                # Softer than Claude's version so it does not overpower everything.
                tactical_adjustment += float(see) * 0.6

            elif see > 0:
                # Winning capture.
                # Small reward, capped.
                tactical_adjustment += min(float(see) * 0.10, 0.5)

        # Newly hanging material penalty.
        hanging_value = hanging_material_after_move(board, move)

        if hanging_value > 0:
            tactical_adjustment -= 0.8 * float(hanging_value)

        # Tiny check bonus only if the move is not unsafe.
        if board.gives_check(move):
            if hanging_value == 0:
                tactical_adjustment += 0.05
            else:
                tactical_adjustment -= 0.25

        # Small opening-principle bonus for the coach ranking too.
        tactical_adjustment += opening_coach_bonus(board, move)

        search_score = search_adjustment_for_move(board, move, reply_limit=12)
        final_score = eval_score + tactical_adjustment + (0.65 * search_score)

        scored_moves.append(
            {
                "move": move,
                "san": san,
                "policy_score": policy_score,
                "eval_score": eval_score,
                "tactical_adjustment": tactical_adjustment,
                "final_score": final_score,
            }
        )

    if not scored_moves:
        return []

    # Since eval_score is now side-to-move perspective, higher is always better.
    scored_moves.sort(
        key=lambda item: (item["final_score"], item["policy_score"]),
        reverse=True,
    )

    result: List[MoveSuggestion] = []

    for item in scored_moves[:top_n]:
        move = item["move"]
        san = item["san"]
        final_score = item["final_score"]

        tags = build_move_tags(board, move)
        explanation = build_move_explanation(board, move, tags)

        result.append(
            MoveSuggestion(
                move=move,
                san=san,
                score=round(final_score, 2),
                tags=tags,
                explanation=explanation,
            )
        )

    return result


def opening_coach_bonus(board: chess.Board, move: chess.Move) -> float:
    """
    Small opening-principle bonus for coach move ranking.

    This keeps early suggestions more human-like without forcing an opening book.
    """
    if board.fullmove_number > 10:
        return 0.0

    piece = board.piece_at(move.from_square)

    if piece is None:
        return 0.0

    bonus = 0.0
    uci = move.uci()

    good_pawn_moves = {
        "e2e4", "d2d4", "c2c4",
        "e7e5", "d7d5", "c7c5",
    }

    good_knight_moves = {
        "g1f3", "b1c3",
        "g8f6", "b8c6",
    }

    if piece.piece_type == chess.PAWN:
        if uci in good_pawn_moves:
            bonus += 0.20

        if move.to_square in {chess.D4, chess.E4, chess.D5, chess.E5}:
            bonus += 0.10

    if piece.piece_type == chess.KNIGHT:
        if uci in good_knight_moves:
            bonus += 0.20
        else:
            from_rank = chess.square_rank(move.from_square)

            if piece.color == chess.WHITE and from_rank == 0:
                bonus += 0.10

            if piece.color == chess.BLACK and from_rank == 7:
                bonus += 0.10

    if piece.piece_type == chess.BISHOP:
        from_rank = chess.square_rank(move.from_square)

        if piece.color == chess.WHITE and from_rank == 0:
            bonus += 0.14

        if piece.color == chess.BLACK and from_rank == 7:
            bonus += 0.14

    if piece.piece_type == chess.QUEEN and board.fullmove_number <= 6:
        bonus -= 0.30

    if board.is_castling(move):
        bonus += 0.35

    return bonus


def build_move_tags(board: chess.Board, move: chess.Move) -> list[str]:
    """
    Creates simple tags for UI display.
    """
    tags: list[str] = []

    if board.is_capture(move):
        see = static_exchange_evaluation(board, move)

        if see > 0:
            tags.append("winning capture")
        elif see == 0:
            tags.append("even capture")
        else:
            tags.append("losing capture")

    if board.gives_check(move):
        tags.append("check")

    if move.promotion:
        tags.append("promotion")

    if board.is_castling(move):
        tags.append("castle")

    if hangs_piece_after_move(board, move) or leaves_piece_hanging_after_move(board, move):
        tags.append("unsafe")

    piece = board.piece_at(move.from_square)

    if piece is not None and board.fullmove_number <= 10:
        if piece.piece_type in {chess.KNIGHT, chess.BISHOP}:
            from_rank = chess.square_rank(move.from_square)

            if piece.color == chess.WHITE and from_rank == 0:
                tags.append("development")

            if piece.color == chess.BLACK and from_rank == 7:
                tags.append("development")

    return tags


def build_move_explanation(
    board: chess.Board,
    move: chess.Move,
    tags: list[str],
) -> str:
    
    #creating a short explanation for the move suggestion.
    
    piece = board.piece_at(move.from_square)
    piece_name = piece.symbol().upper() if piece else "Piece"
    square_to = chess.square_name(move.to_square)

    if "winning capture" in tags:
        return f"{board.san(move)} wins material based on the exchange sequence."

    if "losing capture" in tags:
        return f"{board.san(move)} is tactical but the exchange may lose material."

    if "castle" in tags:
        return f"{board.san(move)} improves king safety and connects the rooks."

    if "development" in tags:
        return f"{board.san(move)} develops a piece toward a more active square."

    if "check" in tags and "unsafe" not in tags:
        return f"{board.san(move)} gives check while keeping the move tactically safe."

    if "unsafe" in tags:
        return f"{board.san(move)} has some tactical risk, so check the follow-up carefully."

    if move.promotion:
        return f"{board.san(move)} promotes a pawn and creates a major material advantage."

    return f"{board.san(move)} improves the position and keeps {piece_name} active on {square_to}."
