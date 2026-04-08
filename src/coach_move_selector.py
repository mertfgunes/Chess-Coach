from __future__ import annotations

from typing import List
import chess
import torch

from coach_models import MoveSuggestion
from coach_evaluation import evaluate_position
from encoding import board_to_tensor
from play_against_ai import board_to_extras, DEVICE
from coach_tactics import hangs_piece_after_move


@torch.no_grad()
def get_top_moves(model, vocab, board: chess.Board, top_n: int = 3) -> List[MoveSuggestion]:
    if model is None or vocab is None:
        return []

    x = board_to_tensor(board).unsqueeze(0).to(DEVICE)
    extras = board_to_extras(board).unsqueeze(0).to(DEVICE)

    policy_logits, _value_pred = model(x, extras)
    logits = policy_logits.squeeze(0)

    legal_moves = list(board.legal_moves)
    scored_moves = []

    unk_idx = vocab.stoi[vocab.UNK]

    for move in legal_moves:
        idx = vocab.encode(move)
        if idx == unk_idx:
            continue

        policy_score = float(logits[idx].item())

        board_copy = board.copy(stack=False)
        san = board.san(move)
        board_copy.push(move)

        eval_score = evaluate_position(board_copy).total

        tactical_penalty = 0.0
        if hangs_piece_after_move(board, move):
            tactical_penalty -= 2.5

        final_score = eval_score + tactical_penalty
        scored_moves.append((move, san, policy_score, final_score))

    if not scored_moves:
        fallback = []
        for move in legal_moves[:top_n]:
            board_copy = board.copy(stack=False)
            san = board.san(move)
            board_copy.push(move)
            eval_score = evaluate_position(board_copy).total

            fallback.append(
                MoveSuggestion(
                    move=move,
                    san=san,
                    score=round(eval_score, 2),
                    explanation="Legal fallback move.",
                    tags=[],
                )
            )
        return fallback

    if board.turn == chess.WHITE:
        scored_moves.sort(key=lambda item: (item[3], item[2]), reverse=True)
    else:
        scored_moves.sort(key=lambda item: (item[3], item[2]))

    result: List[MoveSuggestion] = []

    for move, san, _policy_score, final_score in scored_moves[:top_n]:
        tags = []

        if board.is_capture(move):
            tags.append("capture")
        if board.gives_check(move):
            tags.append("check")
        if move.promotion is not None:
            tags.append("promotion")
        if hangs_piece_after_move(board, move):
            tags.append("unsafe")

        result.append(
            MoveSuggestion(
                move=move,
                san=san,
                score=round(final_score, 2),
                explanation="",
                tags=tags,
            )
        )

    return result