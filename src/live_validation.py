from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import chess
import chess.engine


@dataclass
class LiveValidationResult:
    your_eval: Optional[float] = None          # pawn units, white POV
    sf_eval: Optional[float] = None            # pawn units, white POV
    abs_error: Optional[float] = None
    same_sign: Optional[bool] = None
    depth_scores: Optional[List[Optional[float]]] = None
    spread: Optional[float] = None
    stable: Optional[bool] = None
    error_text: Optional[str] = None


def score_to_white_pawns(score: chess.engine.PovScore) -> Optional[float]:
    
    #Convert engine score to pawn units from White's perspective.
    # + white better
    # - black better
    
    if score is None:
        return None

    white_score = score.white()

    mate = white_score.mate()
    if mate is not None:
        return 100.0 if mate > 0 else -100.0

    cp = white_score.score()
    if cp is None:
        return None

    return cp / 100.0


def sign(x: Optional[float], eps: float = 1e-6) -> Optional[int]:
    if x is None:
        return None
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


class LiveValidator:
    
    #compares evaluator against Stockfish for the current board position.
    

    def __init__(
        self,
        stockfish_path: str,
        your_eval_fn: Callable[[chess.Board], float],
        reference_depth: int = 12,
        stability_depths: Optional[List[int]] = None,
        stability_threshold: float = 0.75,
    ) -> None:
        self.stockfish_path = stockfish_path
        self.your_eval_fn = your_eval_fn
        self.reference_depth = reference_depth
        self.stability_depths = stability_depths or [4, 8, 12]
        self.stability_threshold = stability_threshold

        self.engine: Optional[chess.engine.SimpleEngine] = None
        self.last_fen: Optional[str] = None
        self.last_result: LiveValidationResult = LiveValidationResult()

    def start(self) -> None:
        if self.engine is None:
            command = [self.stockfish_path]
            self.engine = chess.engine.SimpleEngine.popen_uci(command)

    def close(self) -> None:
        if self.engine is not None:
            try:
                self.engine.quit()
            except Exception:
                pass
            self.engine = None

    def _eval_stockfish(self, board: chess.Board, depth: int) -> Optional[float]:
        if self.engine is None:
            return None
        info = self.engine.analyse(board, chess.engine.Limit(depth=depth))
        return score_to_white_pawns(info.get("score"))

    def update(self, board: chess.Board, force: bool = False) -> LiveValidationResult:
        """
        Recompute only if the position changed, unless force=True.
        """
        fen = board.fen()
        if not force and fen == self.last_fen:
            return self.last_result

        result = LiveValidationResult(depth_scores=[])

        try:
            result.your_eval = float(self.your_eval_fn(board.copy(stack=False)))
        except Exception as exc:
            result.error_text = f"Your eval failed: {exc}"

        try:
            if self.engine is None:
                self.start()

            result.sf_eval = self._eval_stockfish(board, self.reference_depth)

            depth_scores: List[Optional[float]] = []
            for d in self.stability_depths:
                try:
                    depth_scores.append(self._eval_stockfish(board, d))
                except Exception:
                    depth_scores.append(None)

            result.depth_scores = depth_scores

            usable = [x for x in depth_scores if x is not None]
            if usable:
                result.spread = max(usable) - min(usable)
                result.stable = result.spread <= self.stability_threshold

        except Exception as exc:
            if result.error_text:
                result.error_text += f" | Stockfish failed: {exc}"
            else:
                result.error_text = f"Stockfish failed: {exc}"

        if result.your_eval is not None and result.sf_eval is not None:
            result.abs_error = abs(result.your_eval - result.sf_eval)
            result.same_sign = sign(result.your_eval) == sign(result.sf_eval)

        self.last_fen = fen
        self.last_result = result
        return result