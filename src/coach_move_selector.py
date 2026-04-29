from __future__ import annotations

from typing import List
import chess
import torch

from coach_models import MoveSuggestion
from coach_evaluation import evaluate_position
from encoding import board_to_tensor
from play_against_ai import board_to_extras, DEVICE
from coach_tactics import (
    hangs_piece_after_move,
    hanging_material_after_move,
    leaves_piece_hanging_after_move,
    static_exchange_evaluation,
)


@torch.no_grad()
def get_top_moves(
    model, vocab, board: chess.Board, top_n: int = 3
) -> List[MoveSuggestion]:
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

        # evaluate_position returns a White-centric score; we convert it
        # to the perspective of the side to move so that sorting is correct
        # for both White AND Black.
        raw_eval = evaluate_position(board_copy).total
        eval_score = raw_eval if board.turn == chess.WHITE else -raw_eval

        tactical_penalty = 0.0

        # SEE-based capture penalty/bonus — replaces the old flat bonuses
        if board.is_capture(move):
            see = static_exchange_evaluation(board, move)
            if see < 0:
                # Losing capture: penalise proportionally
                tactical_penalty += see * 1.5
            elif see > 0:
                # Winning capture: small reward
                tactical_penalty += min(see * 0.15, 1.0)
            # Even capture (see == 0): no adjustment

        # Hanging penalty only for *newly* hanging pieces (SEE-aware)
        hanging_value = hanging_material_after_move(board, move)
        if hanging_value > 0:
            tactical_penalty -= 2.0 * hanging_value

        final_score = eval_score + tactical_penalty

        scored_moves.append((move, san, policy_score, final_score))

    # Sort descending by (final_score, policy_score) for BOTH colours.
    # eval_score is already in side-to-move perspective, so higher is always
    # better regardless of colour — no more ascending sort for Black.
    scored_moves.sort(key=lambda x: (x[3], x[2]), reverse=True)

    result: List[MoveSuggestion] = []

    for move, san, _, final_score in scored_moves[:top_n]:
        tags = []

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

        if hangs_piece_after_move(board, move) or leaves_piece_hanging_after_move(board, move):
            tags.append("unsafe")

        result.append(
            MoveSuggestion(
                move=move,
                san=san,
                score=round(final_score, 2),
                tags=tags,
            )
        )

    return result
