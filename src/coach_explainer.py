from __future__ import annotations

from coach_models import MoveSuggestion, EvaluationBreakdown


def explain_move(move: MoveSuggestion) -> str:
    reasons = []

    if "capture" in move.tags:
        reasons.append("it wins or contests material")
    if "check" in move.tags:
        reasons.append("it gives check and creates pressure")
    if "promotion" in move.tags:
        reasons.append("it creates a promotion")

    if not reasons:
        reasons.append("it improves the position in a simple and safe way")

    return f"{move.san} is strong because " + ", ".join(reasons) + "."


def explain_breakdown(b: EvaluationBreakdown) -> str:
    parts = []

    if b.material > 0.3:
        parts.append("White has more material")
    elif b.material < -0.3:
        parts.append("Black has more material")

    if b.center_control > 0.2:
        parts.append("White controls the center better")
    elif b.center_control < -0.2:
        parts.append("Black controls the center better")

    if b.king_safety > 0.2:
        parts.append("Black's king is under more pressure")
    elif b.king_safety < -0.2:
        parts.append("White's king is under more pressure")

    if not parts:
        return "The position is balanced without one clear strategic edge."

    return ". ".join(parts) + "."