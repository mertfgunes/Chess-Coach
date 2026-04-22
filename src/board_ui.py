from __future__ import annotations

import os
import shutil
import sys
import traceback
from typing import Dict, List, Optional, Tuple

import chess
import pygame

from play_against_ai import load_model, predict_legal_move
from coach_service import ChessCoachService
from coach_evaluation import evaluate_position
from coach_blunder import classify_move_loss, explain_bad_move
from live_validation import LiveValidator


BOARD_SIZE = 8
DEFAULT_SQUARE_SIZE = 90
MIN_SQUARE_SIZE = 64
MAX_SQUARE_SIZE = 104

LEFT_PANEL_WIDTH = 250
RIGHT_PANEL_WIDTH = 360
BOTTOM_PANEL_HEIGHT = 150

WINDOW_WIDTH = LEFT_PANEL_WIDTH + BOARD_SIZE * DEFAULT_SQUARE_SIZE + RIGHT_PANEL_WIDTH
WINDOW_HEIGHT = BOARD_SIZE * DEFAULT_SQUARE_SIZE + BOTTOM_PANEL_HEIGHT
FPS = 60

LIGHT_SQUARE = (240, 217, 181)
DARK_SQUARE = (181, 136, 99)
SELECTED_COLOR = (255, 255, 0, 110)
LAST_MOVE_COLOR = (80, 160, 255, 90)
MOVE_DOT_COLOR = (80, 200, 120)
CAPTURE_RING_COLOR = (220, 80, 80)
HINT_FROM_COLOR = (90, 170, 255, 90)
HINT_TO_COLOR = (100, 220, 140, 110)
WARNING_SQUARE_COLOR = (255, 110, 110, 95)

APP_BG = (25, 27, 31)
PANEL_BG = (35, 35, 35)
CARD_BG = (44, 47, 54)
CARD_BG_2 = (52, 56, 64)
BORDER = (84, 91, 102)
TEXT_COLOR = (245, 245, 245)
SUBTLE_TEXT = (205, 209, 216)
ACCENT = (100, 180, 255)
GOOD = (96, 200, 120)
WARN = (255, 190, 95)
BAD = (255, 120, 120)
BUTTON_COLOR = (70, 70, 70)
BUTTON_HOVER = (95, 95, 95)

ASSETS_DIR = "assets/pieces"

PIECE_IMAGE_NAMES = {
    "P": "whitepawn.png",
    "N": "whiteknight.png",
    "B": "whitebishop.png",
    "R": "whiterook.png",
    "Q": "whitequeen.png",
    "K": "whiteking.png",
    "p": "blackpawn.png",
    "n": "blackknight.png",
    "b": "blackbishop.png",
    "r": "blackrook.png",
    "q": "blackqueen.png",
    "k": "blackking.png",
}

FALLBACK_UNICODE = {
    "P": "♙",
    "N": "♘",
    "B": "♗",
    "R": "♖",
    "Q": "♕",
    "K": "♔",
    "p": "♟",
    "n": "♞",
    "b": "♝",
    "r": "♜",
    "q": "♛",
    "k": "♚",
}


class ChessUI:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Chess Coach - Play Against AI")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()

        self.title_font = pygame.font.SysFont("Segoe UI", 28, bold=True)
        self.text_font = pygame.font.SysFont("Segoe UI", 22)
        self.small_font = pygame.font.SysFont("Segoe UI", 18)
        self.tiny_font = pygame.font.SysFont("Segoe UI", 15)
        self.piece_font = pygame.font.SysFont("Segoe UI Symbol", 56)

        self.window_width = WINDOW_WIDTH
        self.window_height = WINDOW_HEIGHT
        self.square_size = DEFAULT_SQUARE_SIZE
        self.board_pixel_size = self.square_size * BOARD_SIZE
        self.board_rect = pygame.Rect(LEFT_PANEL_WIDTH, 0, self.board_pixel_size, self.board_pixel_size)

        self.model = None
        self.vocab = None
        self.model_loaded = False
        self.model_error: Optional[str] = None

        self.board = chess.Board()
        self.human_color: Optional[chess.Color] = None
        self.ai_color: Optional[chess.Color] = None

        self.selected_square: Optional[int] = None
        self.legal_targets: List[int] = []
        self.last_move: Optional[chess.Move] = None

        self.awaiting_promotion_from: Optional[int] = None
        self.awaiting_promotion_to: Optional[int] = None

        self.ai_thinking = False
        self.game_over_message = ""

        self.coach = ChessCoachService()
        self.latest_analysis = self.coach.analyze_position(self.board, self.model, self.vocab)
        self.latest_move_feedback: Optional[str] = None
        self.mode_label = "Coach"

        self.validation_result = None
        self.validator_error: Optional[str] = None
        self.stockfish_path = self.find_stockfish_path()
        self.live_validator: Optional[LiveValidator] = None

        self.board_flipped = False
        self.move_list_scroll = 0

        self.cached_image_size = self.square_size
        self.piece_images = self.load_piece_images()

        self.recompute_layout(self.window_width, self.window_height)

        self.load_ai_once()
        self.setup_live_validator()
        self.refresh_validation(force=True)

    def recompute_layout(self, width: int, height: int):
        self.window_width = max(980, width)
        self.window_height = max(760, height)

        available_board_w = self.window_width - LEFT_PANEL_WIDTH - RIGHT_PANEL_WIDTH
        available_board_h = self.window_height - BOTTOM_PANEL_HEIGHT
        self.square_size = max(
            MIN_SQUARE_SIZE,
            min(MAX_SQUARE_SIZE, available_board_w // BOARD_SIZE, available_board_h // BOARD_SIZE),
        )
        self.board_pixel_size = self.square_size * BOARD_SIZE

        board_x = LEFT_PANEL_WIDTH + max(0, (available_board_w - self.board_pixel_size) // 2)
        board_y = max(0, (available_board_h - self.board_pixel_size) // 2)

        self.left_panel_rect = pygame.Rect(0, 0, LEFT_PANEL_WIDTH, self.window_height)
        self.right_panel_rect = pygame.Rect(self.window_width - RIGHT_PANEL_WIDTH, 0, RIGHT_PANEL_WIDTH, self.window_height)
        self.board_rect = pygame.Rect(board_x, board_y, self.board_pixel_size, self.board_pixel_size)
        self.bottom_rect = pygame.Rect(board_x, self.window_height - BOTTOM_PANEL_HEIGHT, self.board_pixel_size, BOTTOM_PANEL_HEIGHT)

        left_inner = 14
        right_inner = 16

        self.move_history_rect = pygame.Rect(
            self.left_panel_rect.x + left_inner,
            72,
            self.left_panel_rect.width - 2 * left_inner,
            self.window_height - 230,
        )
        self.left_status_rect = pygame.Rect(
            self.left_panel_rect.x + left_inner,
            self.window_height - 145,
            self.left_panel_rect.width - 2 * left_inner,
            120,
        )

        card_width = self.right_panel_rect.width - 2 * right_inner
        y = 18
        self.status_card = pygame.Rect(self.right_panel_rect.x + right_inner, y, card_width, 110)
        y += 124
        self.bestmove_card = pygame.Rect(self.right_panel_rect.x + right_inner, y, card_width, 128)
        y += 142
        self.warning_card = pygame.Rect(self.right_panel_rect.x + right_inner, y, card_width, 112)
        y += 126
        self.feedback_card = pygame.Rect(self.right_panel_rect.x + right_inner, y, card_width, 138)
        y += 152
        remaining = self.window_height - y - 18
        self.validation_card = pygame.Rect(self.right_panel_rect.x + right_inner, y, card_width, max(120, remaining))

        button_y = self.bottom_rect.y + 78
        button_w = max(112, (self.bottom_rect.width - 5 * 12 - 24) // 5)
        button_h = 42
        start_x = self.bottom_rect.x + 12
        self.white_button = pygame.Rect(start_x + 0 * (button_w + 12), button_y, button_w, button_h)
        self.black_button = pygame.Rect(start_x + 1 * (button_w + 12), button_y, button_w, button_h)
        self.restart_button = pygame.Rect(start_x + 2 * (button_w + 12), button_y, button_w, button_h)
        self.flip_button = pygame.Rect(start_x + 3 * (button_w + 12), button_y, button_w, button_h)
        self.hint_button = pygame.Rect(start_x + 4 * (button_w + 12), button_y, button_w, button_h)

        if self.cached_image_size != self.square_size:
            self.cached_image_size = self.square_size
            self.piece_images = self.load_piece_images()

    def draw_text(self, text: str, font: pygame.font.Font, color, x: int, y: int, surface: Optional[pygame.Surface] = None, center: bool = False):
        target = surface or self.screen
        img = font.render(text, True, color)
        rect = img.get_rect(center=(x, y)) if center else img.get_rect(topleft=(x, y))
        target.blit(img, rect)
        return rect

    def wrap_text(self, text: str, max_width: int, font: Optional[pygame.font.Font] = None) -> List[str]:
        active_font = font or self.small_font
        words = text.split()
        if not words:
            return [""]
        lines: List[str] = []
        current = words[0]
        for word in words[1:]:
            trial = f"{current} {word}"
            if active_font.size(trial)[0] <= max_width:
                current = trial
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines

    def draw_multiline(self, text: str, rect: pygame.Rect, font: Optional[pygame.font.Font] = None, color=SUBTLE_TEXT, line_gap: int = 4, max_lines: Optional[int] = None, surface: Optional[pygame.Surface] = None) -> int:
        active_font = font or self.small_font
        target = surface or self.screen
        lines = self.wrap_text(text, rect.width, active_font)
        if max_lines is not None:
            lines = lines[:max_lines]
        y = rect.y
        for line in lines:
            target.blit(active_font.render(line, True, color), (rect.x, y))
            y += active_font.get_height() + line_gap
        return y

    def draw_card(self, rect: pygame.Rect, title: str, accent_color):
        pygame.draw.rect(self.screen, CARD_BG, rect, border_radius=14)
        pygame.draw.rect(self.screen, BORDER, rect, 1, border_radius=14)
        pygame.draw.rect(self.screen, accent_color, (rect.x, rect.y, rect.width, 5), border_radius=14)
        self.draw_text(title, self.text_font, TEXT_COLOR, rect.x + 14, rect.y + 10)
        return pygame.Rect(rect.x + 14, rect.y + 42, rect.width - 28, rect.height - 54)

    def piece_value(self, piece_type: int) -> int:
        return {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0,
        }.get(piece_type, 0)

    def detect_hanging_square(self) -> Optional[int]:
        side = self.board.turn
        best_sq = None
        best_val = -1
        for square, piece in self.board.piece_map().items():
            if piece.color != side:
                continue
            attackers = self.board.attackers(not side, square)
            defenders = self.board.attackers(side, square)
            if attackers and not defenders:
                val = self.piece_value(piece.piece_type)
                if val > best_val:
                    best_val = val
                    best_sq = square
        return best_sq

    def get_best_move(self) -> Optional[chess.Move]:
        if self.latest_analysis and getattr(self.latest_analysis, "top_moves", None):
            return self.latest_analysis.top_moves[0].move
        return None

    def get_best_move_explanation(self) -> str:
        if self.latest_analysis and getattr(self.latest_analysis, "top_moves", None):
            return self.latest_analysis.top_moves[0].explanation
        return "No move suggestion available yet."

    def get_status_lines(self) -> List[str]:
        turn_text = "Choose a side first" if self.human_color is None else ("White to move" if self.board.turn == chess.WHITE else "Black to move")
        if self.board.is_game_over():
            turn_text = "Game finished"
        model_text = "Model loaded" if self.model_loaded else "Model unavailable"
        ai_text = "AI thinking..." if self.ai_thinking else "AI ready"
        return [turn_text, model_text, ai_text]

    def get_objective_text(self) -> str:
        if self.board.is_check():
            return "Your king is in check. Respond immediately."
        hanging = self.detect_hanging_square()
        if hanging is not None:
            return f"Your piece on {chess.square_name(hanging)} is hanging."
        if self.latest_analysis is not None:
            return self.latest_analysis.summary
        return "Develop pieces and keep your king safe."

    def find_stockfish_path(self) -> Optional[str]:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(base_dir)

        env_path = os.environ.get("STOCKFISH_PATH")
        if env_path and os.path.exists(env_path):
            return env_path

        which_path = shutil.which("stockfish")
        if which_path:
            return which_path

        which_path = shutil.which("stockfish.exe")
        if which_path:
            return which_path

        candidates = [
            os.path.join(project_root, "engines", "stockfish.exe"),
            os.path.join(project_root, "engines", "stockfish"),
            os.path.join(project_root, "bin", "stockfish.exe"),
            os.path.join(project_root, "bin", "stockfish"),
            r"C:\stockfish\stockfish.exe",
            r"C:\Program Files\Stockfish\stockfish.exe",
        ]

        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    def get_model_eval_for_validation(self, board: chess.Board) -> float:
        breakdown = evaluate_position(board)
        return float(breakdown.total)

    def setup_live_validator(self):
        if not self.stockfish_path:
            self.validator_error = "Stockfish not found. Set STOCKFISH_PATH or install Stockfish."
            return

        try:
            self.live_validator = LiveValidator(
                stockfish_path=self.stockfish_path,
                your_eval_fn=self.get_model_eval_for_validation,
                reference_depth=12,
                stability_depths=[4, 8, 12],
                stability_threshold=0.75,
            )
            self.live_validator.start()
            self.validator_error = None
        except Exception as e:
            self.live_validator = None
            self.validator_error = f"Stockfish failed to start: {e}"
            traceback.print_exc()

    def refresh_validation(self, force: bool = False):
        if self.live_validator is None:
            self.validation_result = None
            return

        try:
            self.validation_result = self.live_validator.update(self.board, force=force)
            if self.validation_result and self.validation_result.error_text:
                self.validator_error = self.validation_result.error_text
            else:
                self.validator_error = None
        except Exception as e:
            self.validation_result = None
            self.validator_error = f"Validator update failed: {e}"
            traceback.print_exc()

    def close(self):
        if self.live_validator is not None:
            try:
                self.live_validator.close()
            except Exception:
                pass
            self.live_validator = None

    def load_ai_once(self):
        try:
            loaded = load_model()
            if isinstance(loaded, tuple) and len(loaded) == 2:
                self.model, self.vocab = loaded
            else:
                self.model, self.vocab = loaded, None
            self.model_loaded = True
            self.model_error = None
            self.latest_analysis = self.coach.analyze_position(self.board, self.model, self.vocab)
        except Exception as e:
            self.model_loaded = False
            self.model_error = str(e)
            traceback.print_exc()

    def update_coach_analysis(self):
        self.latest_analysis = self.coach.analyze_position(self.board, self.model, self.vocab)

    def make_ai_move(self):
        if not self.model_loaded or self.ai_thinking or self.human_color is None:
            return

        self.ai_thinking = True
        self.draw()
        pygame.display.flip()

        try:
            move = predict_legal_move(self.model, self.vocab, self.board)
            if move is None:
                self.model_error = "predict_legal_move returned None"
            elif isinstance(move, str):
                move = chess.Move.from_uci(move)
            elif not isinstance(move, chess.Move):
                self.model_error = f"predict_legal_move returned invalid type: {type(move)}"
                move = None

            if move is not None and move in self.board.legal_moves:
                self.board.push(move)
                self.last_move = move
                self.clear_selection()
                self.update_coach_analysis()
                self.refresh_validation(force=True)
            elif move is not None:
                self.model_error = f"Illegal AI move returned: {move.uci()}"
        except Exception as e:
            self.model_error = str(e)
            traceback.print_exc()

        self.ai_thinking = False

    def reset_game(self):
        self.board = chess.Board()
        self.selected_square = None
        self.legal_targets = []
        self.last_move = None
        self.awaiting_promotion_from = None
        self.awaiting_promotion_to = None
        self.ai_thinking = False
        self.game_over_message = ""
        self.latest_move_feedback = None
        self.move_list_scroll = 0
        self.update_coach_analysis()
        self.refresh_validation(force=True)

    def choose_side(self, color: chess.Color):
        self.human_color = color
        self.ai_color = not color
        self.board_flipped = color == chess.BLACK
        self.reset_game()

    def clear_selection(self):
        self.selected_square = None
        self.legal_targets = []

    def update_game_over_message(self):
        if not self.board.is_game_over():
            self.game_over_message = ""
            return

        if self.board.is_checkmate():
            winner = "Black" if self.board.turn == chess.WHITE else "White"
            self.game_over_message = f"Checkmate - {winner} wins."
        elif self.board.is_stalemate():
            self.game_over_message = "Draw by stalemate."
        elif self.board.is_insufficient_material():
            self.game_over_message = "Draw by insufficient material."
        elif self.board.is_seventyfive_moves():
            self.game_over_message = "Draw by 75-move rule."
        elif self.board.is_fivefold_repetition():
            self.game_over_message = "Draw by repetition."
        else:
            self.game_over_message = "Game over."

    def screen_to_square(self, pos: Tuple[int, int]) -> Optional[int]:
        x, y = pos
        if not self.board_rect.collidepoint(x, y):
            return None

        local_x = x - self.board_rect.x
        local_y = y - self.board_rect.y
        file = local_x // self.square_size
        rank = 7 - (local_y // self.square_size)

        if self.board_flipped:
            file = 7 - file
            rank = 7 - rank

        return chess.square(file, rank)

    def square_to_screen(self, square: int) -> Tuple[int, int]:
        file = chess.square_file(square)
        rank = chess.square_rank(square)

        if self.board_flipped:
            file = 7 - file
            rank = 7 - rank

        x = self.board_rect.x + file * self.square_size
        y = self.board_rect.y + (7 - rank) * self.square_size
        return x, y

    def handle_board_click(self, mouse_pos: Tuple[int, int]):
        if self.human_color is None or self.board.turn != self.human_color or self.board.is_game_over():
            return

        clicked_square = self.screen_to_square(mouse_pos)
        if clicked_square is None:
            return

        piece = self.board.piece_at(clicked_square)

        if self.selected_square is not None:
            if self.try_make_human_move(self.selected_square, clicked_square):
                return

        if piece and piece.color == self.human_color:
            self.selected_square = clicked_square
            self.legal_targets = [move.to_square for move in self.board.legal_moves if move.from_square == clicked_square]
        else:
            self.clear_selection()

    def try_make_human_move(self, from_sq: int, to_sq: int) -> bool:
        legal_moves = [m for m in self.board.legal_moves if m.from_square == from_sq and m.to_square == to_sq]

        if not legal_moves:
            return False

        promotion_moves = [m for m in legal_moves if m.promotion is not None]
        if promotion_moves:
            self.awaiting_promotion_from = from_sq
            self.awaiting_promotion_to = to_sq
            return True

        move = legal_moves[0]
        board_before = self.board.copy(stack=False)

        self.board.push(move)
        self.last_move = move
        self.clear_selection()
        self.update_coach_analysis()
        self.update_move_feedback(board_before, move)
        self.refresh_validation(force=True)
        return True

    def update_move_feedback(self, board_before: chess.Board, played_move: chess.Move):
        if self.model is None:
            self.latest_move_feedback = None
            return

        best_analysis = self.coach.analyze_position(board_before, self.model, self.vocab)
        if not getattr(best_analysis, "top_moves", None):
            self.latest_move_feedback = None
            return

        best_move = best_analysis.top_moves[0].move

        played_board = board_before.copy(stack=False)
        played_board.push(played_move)
        played_score = evaluate_position(played_board).total

        best_board = board_before.copy(stack=False)
        best_board.push(best_move)
        best_score = evaluate_position(best_board).total

        if board_before.turn == chess.WHITE:
            loss = best_score - played_score
        else:
            loss = played_score - best_score

        classification = classify_move_loss(loss)

        if played_move == best_move:
            self.latest_move_feedback = "Good move. You played the best move in the position."
            return

        reason = explain_bad_move(board_before, played_move, best_move)
        played_san = board_before.san(played_move)
        best_san = board_before.san(best_move)
        self.latest_move_feedback = f"Your move {played_san} was {classification}. Best move was {best_san}. {reason}"

    def get_promotion_option_rects(self) -> List[Tuple[int, pygame.Rect]]:
        panel_w, panel_h = 360, 150
        panel_x = self.window_width // 2 - panel_w // 2
        panel_y = self.window_height // 2 - panel_h // 2

        options = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]

        box_w = 72
        box_h = 72
        gap = 12
        start_x = panel_x + 18
        y = panel_y + 54

        rects: List[Tuple[int, pygame.Rect]] = []
        for i, piece_type in enumerate(options):
            rect = pygame.Rect(start_x + i * (box_w + gap), y, box_w, box_h)
            rects.append((piece_type, rect))
        return rects

    def handle_promotion_click(self, mouse_pos: Tuple[int, int]):
        if self.awaiting_promotion_from is None or self.awaiting_promotion_to is None:
            return

        board_before = self.board.copy(stack=False)

        for piece_type, rect in self.get_promotion_option_rects():
            if rect.collidepoint(mouse_pos):
                move = chess.Move(self.awaiting_promotion_from, self.awaiting_promotion_to, promotion=piece_type)
                if move in self.board.legal_moves:
                    self.board.push(move)
                    self.last_move = move
                    self.update_coach_analysis()
                    self.update_move_feedback(board_before, move)
                    self.refresh_validation(force=True)

                self.awaiting_promotion_from = None
                self.awaiting_promotion_to = None
                self.clear_selection()
                return

    def clamp_move_list_scroll(self, total_rows: int):
        max_scroll = max(0, total_rows - self.visible_move_rows())
        self.move_list_scroll = max(0, min(self.move_list_scroll, max_scroll))

    def visible_move_rows(self) -> int:
        return max(4, (self.move_history_rect.height - 22) // 24)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                pygame.quit()
                sys.exit()

            if event.type == pygame.VIDEORESIZE:
                self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                self.recompute_layout(event.w, event.h)
                continue

            if event.type == pygame.MOUSEWHEEL:
                mouse_pos = pygame.mouse.get_pos()
                if self.move_history_rect.collidepoint(mouse_pos):
                    rows = (len(self.board.move_stack) + 1) // 2
                    self.move_list_scroll -= event.y
                    self.clamp_move_list_scroll(rows)
                continue

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.reset_game()
                elif event.key == pygame.K_f:
                    self.board_flipped = not self.board_flipped
                elif event.key == pygame.K_ESCAPE:
                    self.clear_selection()
                elif event.key == pygame.K_h:
                    self.mode_label = "Hint" if self.mode_label != "Hint" else "Coach"

            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos

                if event.button == 4:
                    if self.move_history_rect.collidepoint(mouse_pos):
                        rows = (len(self.board.move_stack) + 1) // 2
                        self.move_list_scroll -= 1
                        self.clamp_move_list_scroll(rows)
                    continue

                if event.button == 5:
                    if self.move_history_rect.collidepoint(mouse_pos):
                        rows = (len(self.board.move_stack) + 1) // 2
                        self.move_list_scroll += 1
                        self.clamp_move_list_scroll(rows)
                    continue

                if event.button != 1:
                    continue

                if self.awaiting_promotion_from is not None:
                    self.handle_promotion_click(mouse_pos)
                    continue

                if self.white_button.collidepoint(mouse_pos):
                    self.choose_side(chess.WHITE)
                    continue

                if self.black_button.collidepoint(mouse_pos):
                    self.choose_side(chess.BLACK)
                    continue

                if self.restart_button.collidepoint(mouse_pos):
                    self.reset_game()
                    continue

                if self.flip_button.collidepoint(mouse_pos):
                    self.board_flipped = not self.board_flipped
                    continue

                if self.hint_button.collidepoint(mouse_pos):
                    self.mode_label = "Hint" if self.mode_label != "Hint" else "Coach"
                    continue

                if self.board_rect.collidepoint(mouse_pos):
                    self.handle_board_click(mouse_pos)

    def load_piece_images(self) -> Dict[str, pygame.Surface]:
        images: Dict[str, pygame.Surface] = {}
        target = max(28, self.square_size - 12)

        for symbol, filename in PIECE_IMAGE_NAMES.items():
            path = os.path.join(ASSETS_DIR, filename)
            if os.path.exists(path):
                try:
                    img = pygame.image.load(path).convert_alpha()
                    img = pygame.transform.smoothscale(img, (target, target))
                    images[symbol] = img
                except Exception as e:
                    print(f"WARNING: failed to load piece image {path}: {e}")
        return images

    def draw_button(self, rect: pygame.Rect, text: str):
        mouse_pos = pygame.mouse.get_pos()
        color = BUTTON_HOVER if rect.collidepoint(mouse_pos) else BUTTON_COLOR
        pygame.draw.rect(self.screen, color, rect, border_radius=10)
        pygame.draw.rect(self.screen, ACCENT, rect, 2, border_radius=10)
        self.draw_text(text, self.small_font, TEXT_COLOR, rect.centerx, rect.centery, center=True)

    def draw_board(self):
        pygame.draw.rect(self.screen, PANEL_BG, self.board_rect.inflate(8, 8), border_radius=10)

        for rank in range(8):
            for file in range(8):
                x = self.board_rect.x + file * self.square_size
                y = self.board_rect.y + rank * self.square_size
                color = LIGHT_SQUARE if (file + rank) % 2 == 0 else DARK_SQUARE
                pygame.draw.rect(self.screen, color, (x, y, self.square_size, self.square_size))

        if self.last_move:
            for sq in [self.last_move.from_square, self.last_move.to_square]:
                x, y = self.square_to_screen(sq)
                overlay = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
                overlay.fill(LAST_MOVE_COLOR)
                self.screen.blit(overlay, (x, y))

        best_move = self.get_best_move()
        if best_move and self.mode_label in ("Coach", "Hint"):
            for sq, color in [(best_move.from_square, HINT_FROM_COLOR), (best_move.to_square, HINT_TO_COLOR)]:
                x, y = self.square_to_screen(sq)
                overlay = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
                overlay.fill(color)
                self.screen.blit(overlay, (x, y))

        hanging_sq = self.detect_hanging_square()
        if hanging_sq is not None and self.mode_label == "Coach":
            x, y = self.square_to_screen(hanging_sq)
            overlay = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
            overlay.fill(WARNING_SQUARE_COLOR)
            self.screen.blit(overlay, (x, y))

        if self.selected_square is not None:
            x, y = self.square_to_screen(self.selected_square)
            overlay = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
            overlay.fill(SELECTED_COLOR)
            self.screen.blit(overlay, (x, y))

        for target in self.legal_targets:
            x, y = self.square_to_screen(target)
            center = (x + self.square_size // 2, y + self.square_size // 2)
            if self.board.piece_at(target):
                pygame.draw.circle(self.screen, CAPTURE_RING_COLOR, center, max(18, self.square_size // 3), 4)
            else:
                pygame.draw.circle(self.screen, MOVE_DOT_COLOR, center, max(8, self.square_size // 8))

        for file in range(8):
            file_label = chr(ord("a") + (7 - file if self.board_flipped else file))
            self.draw_text(file_label, self.tiny_font, (30, 30, 30), self.board_rect.x + file * self.square_size + self.square_size - 14, self.board_rect.bottom - 18)

        for rank in range(8):
            rank_label = str(rank + 1 if self.board_flipped else 8 - rank)
            self.draw_text(rank_label, self.tiny_font, (30, 30, 30), self.board_rect.x + 4, self.board_rect.y + rank * self.square_size + 4)

        pygame.draw.rect(self.screen, BORDER, self.board_rect, 2)

    def draw_pieces(self):
        for square, piece in self.board.piece_map().items():
            x, y = self.square_to_screen(square)
            symbol = piece.symbol()

            if symbol in self.piece_images:
                margin = (self.square_size - self.piece_images[symbol].get_width()) // 2
                self.screen.blit(self.piece_images[symbol], (x + margin, y + margin))
            else:
                size = max(34, int(self.square_size * 0.58))
                fallback_font = pygame.font.SysFont("Segoe UI Symbol", size)
                text = fallback_font.render(FALLBACK_UNICODE[symbol], True, (20, 20, 20) if piece.color == chess.BLACK else (250, 250, 250))
                shadow = fallback_font.render(FALLBACK_UNICODE[symbol], True, (40, 40, 40))
                self.screen.blit(shadow, (x + int(self.square_size * 0.18) + 2, y + int(self.square_size * 0.08) + 2))
                self.screen.blit(text, (x + int(self.square_size * 0.18), y + int(self.square_size * 0.08)))

    def draw_move_history(self):
        pygame.draw.rect(self.screen, CARD_BG, self.move_history_rect, border_radius=14)
        pygame.draw.rect(self.screen, BORDER, self.move_history_rect, 1, border_radius=14)
        self.draw_text("Moves", self.text_font, TEXT_COLOR, self.move_history_rect.x + 12, self.move_history_rect.y + 10)

        rows: List[str] = []
        move_stack = list(self.board.move_stack)
        for i in range(0, len(move_stack), 2):
            move_no = i // 2 + 1
            white_move = move_stack[i].uci()
            black_move = move_stack[i + 1].uci() if i + 1 < len(move_stack) else ""
            rows.append(f"{move_no:>2}. {white_move:<7} {black_move}")

        self.clamp_move_list_scroll(len(rows))
        visible_rows = self.visible_move_rows()
        start = self.move_list_scroll
        end = min(len(rows), start + visible_rows)

        y = self.move_history_rect.y + 42
        for row in rows[start:end]:
            self.draw_text(row, self.small_font, SUBTLE_TEXT, self.move_history_rect.x + 12, y)
            y += 24

        if len(rows) > visible_rows:
            track = pygame.Rect(self.move_history_rect.right - 8, self.move_history_rect.y + 42, 4, self.move_history_rect.height - 54)
            pygame.draw.rect(self.screen, (80, 80, 80), track, border_radius=3)
            max_scroll = max(1, len(rows) - visible_rows)
            thumb_h = max(28, int(track.height * visible_rows / max(visible_rows, len(rows))))
            thumb_y = track.y + int((track.height - thumb_h) * (self.move_list_scroll / max_scroll))
            pygame.draw.rect(self.screen, ACCENT, (track.x, thumb_y, track.width, thumb_h), border_radius=3)

    def draw_left_status(self):
        pygame.draw.rect(self.screen, CARD_BG, self.left_status_rect, border_radius=14)
        pygame.draw.rect(self.screen, BORDER, self.left_status_rect, 1, border_radius=14)
        self.draw_text("Status", self.text_font, TEXT_COLOR, self.left_status_rect.x + 12, self.left_status_rect.y + 10)

        y = self.left_status_rect.y + 42
        if self.human_color is not None:
            self.draw_text(f"You: {'White' if self.human_color == chess.WHITE else 'Black'}", self.small_font, SUBTLE_TEXT, self.left_status_rect.x + 12, y)
            y += 22
            self.draw_text(f"AI: {'White' if self.ai_color == chess.WHITE else 'Black'}", self.small_font, SUBTLE_TEXT, self.left_status_rect.x + 12, y)
            y += 22
        self.draw_text(f"Mode: {self.mode_label}", self.small_font, SUBTLE_TEXT, self.left_status_rect.x + 12, y)
        y += 22
        self.draw_text(self.game_over_message if self.game_over_message else "Game in progress", self.small_font, WARN if self.game_over_message else SUBTLE_TEXT, self.left_status_rect.x + 12, y)

    def draw_validation_section(self, rect: pygame.Rect):
        content = self.draw_card(rect, "Live Validation", ACCENT)
        y = content.y

        if self.validator_error:
            self.draw_multiline(f"Validator: {self.validator_error}", pygame.Rect(content.x, y, content.width, content.height), color=BAD, max_lines=6)
            return

        if self.validation_result is None:
            self.draw_text("No validation yet.", self.small_font, SUBTLE_TEXT, content.x, y)
            return

        result = self.validation_result
        your_eval = "None" if result.your_eval is None else f"{result.your_eval:+.2f}"
        sf_eval = "None" if result.sf_eval is None else f"{result.sf_eval:+.2f}"
        abs_error = "None" if result.abs_error is None else f"{result.abs_error:.2f}"
        same_sign = "None" if result.same_sign is None else ("Yes" if result.same_sign else "No")
        spread = "None" if result.spread is None else f"{result.spread:.2f}"
        stable = "None" if result.stable is None else ("Yes" if result.stable else "No")

        lines = [
            f"Your eval: {your_eval}",
            f"Stockfish: {sf_eval}",
            f"Abs error: {abs_error}",
            f"Same side better: {same_sign}",
            f"Depth spread: {spread}",
            f"Stable: {stable}",
        ]
        for line in lines:
            self.draw_text(line, self.small_font, SUBTLE_TEXT, content.x, y)
            y += 20

        if result.depth_scores:
            parts = []
            for depth, value in zip([4, 8, 12], result.depth_scores):
                parts.append(f"d{depth}=" + ("None" if value is None else f"{value:+.2f}"))
            self.draw_multiline("Depths: " + ", ".join(parts), pygame.Rect(content.x, y + 4, content.width, content.bottom - y), color=(200, 200, 255), max_lines=4)

    def draw_side_panel(self):
        pygame.draw.rect(self.screen, PANEL_BG, self.left_panel_rect)
        pygame.draw.rect(self.screen, PANEL_BG, self.right_panel_rect)

        self.draw_text("Chess Coach", self.title_font, TEXT_COLOR, 16, 16)
        self.draw_text("Play • Learn • Validate", self.small_font, ACCENT, 16, 46)

        self.draw_move_history()
        self.draw_left_status()

        content = self.draw_card(self.status_card, "Position Status", ACCENT)
        y = content.y
        for line in self.get_status_lines():
            self.draw_text(line, self.small_font, SUBTLE_TEXT, content.x, y)
            y += 22

        content = self.draw_card(self.bestmove_card, "Best Move", GOOD)
        best_move = self.get_best_move()
        if best_move is None:
            self.draw_text("No suggestion available", self.small_font, SUBTLE_TEXT, content.x, content.y)
        else:
            self.draw_text(self.board.san(best_move), self.title_font, TEXT_COLOR, content.x, content.y - 4)
            self.draw_multiline(self.get_best_move_explanation(), pygame.Rect(content.x, content.y + 36, content.width, content.height - 40), color=SUBTLE_TEXT, max_lines=4)

        content = self.draw_card(self.warning_card, "Immediate Focus", WARN if not self.board.is_check() else BAD)
        self.draw_multiline(self.get_objective_text(), pygame.Rect(content.x, content.y, content.width, content.height), color=TEXT_COLOR, max_lines=4)

        content = self.draw_card(self.feedback_card, "Last Move Feedback", BAD if self.latest_move_feedback else ACCENT)
        feedback = self.latest_move_feedback or "Make a move to receive coaching feedback."
        self.draw_multiline(feedback, pygame.Rect(content.x, content.y, content.width, content.height), color=(255, 220, 180) if self.latest_move_feedback else SUBTLE_TEXT, max_lines=6)

        self.draw_validation_section(self.validation_card)

    def draw_bottom_bar(self):
        pygame.draw.rect(self.screen, CARD_BG_2, self.bottom_rect, border_radius=14)
        pygame.draw.rect(self.screen, BORDER, self.bottom_rect, 1, border_radius=14)

        coach_line = self.get_objective_text()
        self.draw_text("Coach", self.text_font, TEXT_COLOR, self.bottom_rect.x + 14, self.bottom_rect.y + 10)
        self.draw_multiline(coach_line, pygame.Rect(self.bottom_rect.x + 14, self.bottom_rect.y + 36, self.bottom_rect.width - 28, 36), color=SUBTLE_TEXT, max_lines=2)

        self.draw_button(self.white_button, "Play White")
        self.draw_button(self.black_button, "Play Black")
        self.draw_button(self.restart_button, "Restart (R)")
        self.draw_button(self.flip_button, "Flip (F)")
        self.draw_button(self.hint_button, "Hint (H)")

    def draw_promotion_dialog(self):
        overlay = pygame.Surface((self.window_width, self.window_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 140))
        self.screen.blit(overlay, (0, 0))

        panel_w, panel_h = 360, 150
        panel_x = self.window_width // 2 - panel_w // 2
        panel_y = self.window_height // 2 - panel_h // 2
        panel = pygame.Rect(panel_x, panel_y, panel_w, panel_h)

        pygame.draw.rect(self.screen, (50, 50, 50), panel, border_radius=12)
        pygame.draw.rect(self.screen, ACCENT, panel, 2, border_radius=12)

        self.draw_text("Choose promotion", self.text_font, TEXT_COLOR, panel_x + 96, panel_y + 14)

        for piece_type, rect in self.get_promotion_option_rects():
            color = BUTTON_HOVER if rect.collidepoint(pygame.mouse.get_pos()) else BUTTON_COLOR
            pygame.draw.rect(self.screen, color, rect, border_radius=8)
            pygame.draw.rect(self.screen, ACCENT, rect, 2, border_radius=8)

            symbol = {chess.QUEEN: "Q", chess.ROOK: "R", chess.BISHOP: "B", chess.KNIGHT: "N"}[piece_type]
            if self.human_color == chess.BLACK:
                symbol = symbol.lower()

            if symbol in self.piece_images:
                img = pygame.transform.smoothscale(self.piece_images[symbol], (56, 56))
                img_rect = img.get_rect(center=rect.center)
                self.screen.blit(img, img_rect)
            else:
                fallback_font = pygame.font.SysFont("Segoe UI Symbol", 44)
                txt = fallback_font.render(FALLBACK_UNICODE[symbol], True, TEXT_COLOR)
                txt_rect = txt.get_rect(center=rect.center)
                self.screen.blit(txt, txt_rect)

    def draw(self):
        self.screen.fill(APP_BG)
        self.draw_board()
        self.draw_pieces()
        self.draw_side_panel()
        self.draw_bottom_bar()

        if self.awaiting_promotion_from is not None:
            self.draw_promotion_dialog()

    def run(self):
        try:
            while True:
                self.handle_events()

                if (
                    self.human_color is not None
                    and not self.board.is_game_over()
                    and self.awaiting_promotion_from is None
                    and self.board.turn == self.ai_color
                ):
                    self.make_ai_move()

                self.update_game_over_message()
                self.refresh_validation(force=False)
                self.draw()

                pygame.display.flip()
                self.clock.tick(FPS)
        finally:
            self.close()
            pygame.quit()


def main():
    app = ChessUI()
    app.run()


if __name__ == "__main__":
    main()
