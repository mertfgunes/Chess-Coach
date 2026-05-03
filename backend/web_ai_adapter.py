from __future__ import annotations

import os
import sys
import random
import traceback
from typing import Optional, Dict, Any

import chess

# Paths
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BACKEND_DIR)
SRC_DIR = os.path.join(ROOT_DIR, "src")

os.chdir(ROOT_DIR)

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
# Fallback values if real AI cannot load
PIECE_VALUES_CP = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

REAL_AI_AVAILABLE = False
REAL_EVAL_AVAILABLE = False
REAL_COACH_AVAILABLE = False

_real_ai_error: Optional[str] = None
_real_eval_error: Optional[str] = None
_real_coach_error: Optional[str] = None

_model = None
_vocab = None
_coach_service = None

try:
    from play_against_ai import load_model, predict_legal_move

    REAL_AI_AVAILABLE = True
except Exception as e:
    _real_ai_error = str(e)
    print(f"[Web AI] Could not import real AI: {e}")

try:
    from coach_tactics import hanging_material_after_move, static_exchange_evaluation
except Exception:
    def hanging_material_after_move(board: chess.Board, move: chess.Move) -> int:
        return 0

    def static_exchange_evaluation(board: chess.Board, move: chess.Move) -> int:
        captured = board.piece_at(move.to_square)
        if captured is None:
            return 0
        return PIECE_VALUES_CP.get(captured.piece_type, 0) // 100

try:
    from coach_evaluation import evaluate_position as real_evaluate_position

    REAL_EVAL_AVAILABLE = True
except Exception as e:
    _real_eval_error = str(e)
    print(f"[Web AI] Could not import real evaluation: {e}")

try:
    from coach_service import ChessCoachService

    REAL_COACH_AVAILABLE = True
except Exception as e:
    _real_coach_error = str(e)
    print(f"[Web AI] Could not import coach service: {e}")


def get_ai_status() -> Dict[str, Any]:
    return {
        "real_ai_available": REAL_AI_AVAILABLE,
        "real_eval_available": REAL_EVAL_AVAILABLE,
        "real_coach_available": REAL_COACH_AVAILABLE,
        "model_loaded": _model is not None,
        "vocab_loaded": _vocab is not None,
        "ai_error": _real_ai_error,
        "eval_error": _real_eval_error,
        "coach_error": _real_coach_error,
    }


def get_model_and_vocab():
    """
    Loads your trained model once, then reuses it.
    Your play_against_ai.load_model() returns (model, vocab).
    """
    global _model, _vocab, _real_ai_error

    if not REAL_AI_AVAILABLE:
        return None, None

    if _model is not None and _vocab is not None:
        return _model, _vocab

    try:
        loaded = load_model()

        if isinstance(loaded, tuple) and len(loaded) == 2:
            _model, _vocab = loaded
        else:
            _model = loaded
            _vocab = None

        return _model, _vocab

    except Exception as e:
        _real_ai_error = str(e)
        print(f"[Web AI] Could not load model: {e}")
        traceback.print_exc()
        return None, None


def get_coach_service():
    global _coach_service, _real_coach_error

    if not REAL_COACH_AVAILABLE:
        return None

    if _coach_service is not None:
        return _coach_service

    try:
        _coach_service = ChessCoachService()
        return _coach_service
    except Exception as e:
        _real_coach_error = str(e)
        print(f"[Web AI] Could not create coach service: {e}")
        traceback.print_exc()
        return None


def fallback_material_evaluation(board: chess.Board) -> int:
    """
    Positive = White is better.
    Negative = Black is better.
    """
    score = 0

    for piece_type, value in PIECE_VALUES_CP.items():
        score += len(board.pieces(piece_type, chess.WHITE)) * value
        score -= len(board.pieces(piece_type, chess.BLACK)) * value

    return score


def get_position_evaluation(board: chess.Board) -> float:
    """
    Uses your real coach_evaluation.py if possible.
    Falls back to simple material evaluation if not.
    """
    if REAL_EVAL_AVAILABLE:
        try:
            result = real_evaluate_position(board)

            # Your board_ui.py uses evaluate_position(board).total
            if hasattr(result, "total"):
                return float(result.total)

            if isinstance(result, (int, float)):
                return float(result)

            return float(fallback_material_evaluation(board))

        except Exception as e:
            print(f"[Web AI] Real evaluation failed, using fallback: {e}")
            traceback.print_exc()

    return float(fallback_material_evaluation(board))


def fallback_ai_move(board: chess.Board, difficulty: str = "medium") -> Optional[chess.Move]:
    legal_moves = list(board.legal_moves)

    if not legal_moves:
        return None

    if difficulty == "easy":
        return random.choice(legal_moves)

    best_move = None
    best_score = None

    for move in legal_moves:
        board_copy = board.copy(stack=False)
        board_copy.push(move)

        score = fallback_material_evaluation(board_copy)

        # If White is choosing, higher is better.
        # If Black is choosing, lower is better.
        adjusted_score = score if board.turn == chess.WHITE else -score

        if board.is_capture(move):
            adjusted_score += static_exchange_evaluation(board, move) * 90

        hanging_value = hanging_material_after_move(board, move)
        adjusted_score -= hanging_value * 120

        if board.gives_check(move):
            adjusted_score += 5

        if best_score is None or adjusted_score > best_score:
            best_score = adjusted_score
            best_move = move

    if difficulty == "medium" and random.random() < 0.20:
        return random.choice(legal_moves)

    return best_move


def get_ai_move(board: chess.Board, difficulty: str = "medium") -> Optional[chess.Move]:
    """
    First tries your trained model.
    If that fails, uses fallback legal move selector.
    """
    model, vocab = get_model_and_vocab()

    if model is not None and vocab is not None:
        try:
            move = predict_legal_move(model, vocab, board, difficulty=difficulty)

            if isinstance(move, str):
                move = chess.Move.from_uci(move)

            if isinstance(move, chess.Move) and move in board.legal_moves:
                return move

            print(f"[Web AI] Real AI returned invalid/illegal move: {move}")

        except Exception as e:
            print(f"[Web AI] Real AI prediction failed, using fallback: {e}")
            traceback.print_exc()

    return fallback_ai_move(board, difficulty)


def move_to_safe_san(board: chess.Board, move: Optional[chess.Move]) -> Optional[str]:
    if move is None:
        return None

    try:
        return board.san(move)
    except Exception:
        return move.uci()


def basic_position_message(board: chess.Board, evaluation: float) -> str:
    if board.is_checkmate():
        winner = "Black" if board.turn == chess.WHITE else "White"
        return f"Checkmate. {winner} wins."

    if board.is_stalemate():
        return "The game is drawn by stalemate."

    if board.is_insufficient_material():
        return "The game is drawn by insufficient material."

    if board.is_check():
        return "Your king is in check. You must respond immediately."

    if evaluation > 250:
        return "White is clearly better. Look for active moves and avoid unnecessary trades."
    if evaluation > 80:
        return "White is slightly better. Improve your pieces and keep control."
    if evaluation < -250:
        return "Black is clearly better. White needs defensive accuracy."
    if evaluation < -80:
        return "Black is slightly better. White should avoid weakening the position."

    return "The position is close to equal. Focus on piece activity, king safety, and avoiding blunders."


def get_coach_advice(board: chess.Board, difficulty: str = "medium") -> Dict[str, Any]:
    """
    Gives web-friendly coach advice.
    Uses real coach service if possible, otherwise gives simple advice.
    """
    evaluation = get_position_evaluation(board)
    suggested_move = get_ai_move(board, difficulty)
    suggested_move_uci = suggested_move.uci() if suggested_move else None
    suggested_move_san = move_to_safe_san(board, suggested_move)

    # Try your real ChessCoachService if available
    coach = get_coach_service()
    model, vocab = get_model_and_vocab()

    real_summary = None
    real_explanation = None

    if coach is not None:
        try:
            analysis = coach.analyze_position(board, model, vocab)

            if hasattr(analysis, "summary"):
                real_summary = analysis.summary

            if getattr(analysis, "top_moves", None):
                top = analysis.top_moves[0]

                if hasattr(top, "move"):
                    suggested_move = top.move
                    suggested_move_uci = suggested_move.uci()
                    suggested_move_san = move_to_safe_san(board, suggested_move)

                if hasattr(top, "explanation"):
                    real_explanation = top.explanation

        except Exception as e:
            print(f"[Web AI] Real coach failed, using fallback advice: {e}")
            traceback.print_exc()

    message_parts = []

    if suggested_move_san:
        message_parts.append(f"Suggested move: {suggested_move_san}.")

    if real_summary:
        message_parts.append(real_summary)
    else:
        message_parts.append(basic_position_message(board, evaluation))

    if real_explanation:
        message_parts.append(real_explanation)

    return {
        "suggested_move": suggested_move_uci,
        "suggested_move_san": suggested_move_san,
        "evaluation": evaluation,
        "message": " ".join(message_parts),
        "ai_status": get_ai_status(),
    }
