from __future__ import annotations

from coach_models import MoveSuggestion, EvaluationBreakdown


def explain_move(move: MoveSuggestion) -> str:
    reasons = []

    if "winning capture" in move.tags:
        reasons.append("the exchange sequence wins material")
    elif "even capture" in move.tags:
        reasons.append("the capture does not lose material after recaptures")
    elif "losing capture" in move.tags:
        reasons.append("it is forcing, but the recapture sequence needs caution")

    if "development" in move.tags:
        reasons.append("it improves piece activity")
    if "castle" in move.tags:
        reasons.append("it improves king safety and connects the rooks")
    if "check" in move.tags and "unsafe" not in move.tags:
        reasons.append("the check is useful without leaving material loose")
    elif "check" in move.tags:
        reasons.append("the check comes with tactical risk")
    if "promotion" in move.tags:
        reasons.append("it creates a decisive promotion threat")
    if "unsafe" in move.tags and "check" not in move.tags:
        reasons.append("there is a tactical drawback to calculate")

    if not reasons:
        reasons.append("it improves the position without creating an obvious tactical problem")

    return f"{move.san} is recommended because " + ", ".join(reasons) + "."


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
