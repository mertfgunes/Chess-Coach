from __future__ import annotations

import os
import sys
import traceback
from typing import Dict, List, Optional, Tuple

import chess
import pygame

from play_against_ai import load_model, predict_legal_move
from coach_service import ChessCoachService
from coach_evaluation import evaluate_position
from coach_blunder import classify_move_loss, explain_bad_move


BOARD_SIZE = 8
SQUARE_SIZE = 96
SIDE_PANEL_WIDTH = 300
WINDOW_WIDTH = BOARD_SIZE * SQUARE_SIZE + SIDE_PANEL_WIDTH
WINDOW_HEIGHT = BOARD_SIZE * SQUARE_SIZE
FPS = 60

LIGHT_SQUARE = (240, 217, 181)
DARK_SQUARE = (181, 136, 99)
SELECTED_COLOR = (255, 255, 0, 110)
LAST_MOVE_COLOR = (80, 160, 255, 90)
MOVE_DOT_COLOR = (80, 200, 120)
CAPTURE_RING_COLOR = (220, 80, 80)

PANEL_BG = (35, 35, 35)
TEXT_COLOR = (245, 245, 245)
ACCENT = (100, 180, 255)
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
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()

        self.title_font = pygame.font.SysFont("Segoe UI", 28)
        self.text_font = pygame.font.SysFont("Segoe UI", 22)
        self.small_font = pygame.font.SysFont("Segoe UI", 18)
        self.piece_font = pygame.font.SysFont("Segoe UI Symbol", 56)

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

        self.piece_images = self.load_piece_images()

        panel_x = BOARD_SIZE * SQUARE_SIZE + 20
        self.white_button = pygame.Rect(panel_x, 120, 260, 50)
        self.black_button = pygame.Rect(panel_x, 190, 260, 50)
        self.restart_button = pygame.Rect(panel_x, 300, 260, 50)
        self.flip_button = pygame.Rect(panel_x, 370, 260, 50)

        self.board_flipped = False

        self.load_ai_once()

    def load_ai_once(self):
        try:
            print("DEBUG: load_ai_once() called")
            self.model, self.vocab = load_model()
            self.model_loaded = True
            self.model_error = None
            self.latest_analysis = self.coach.analyze_position(self.board, self.model, self.vocab)
            print("DEBUG: model loaded successfully")
            print("DEBUG: vocab loaded =", self.vocab is not None)
        except Exception as e:
            self.model_loaded = False
            self.model_error = str(e)
            print("ERROR in load_ai_once():", e)
            traceback.print_exc()

    def update_coach_analysis(self):
        self.latest_analysis = self.coach.analyze_position(self.board, self.model, self.vocab)

    def make_ai_move(self):
        if not self.model_loaded:
            print("DEBUG: AI move skipped - model not loaded")
            return

        if self.ai_thinking:
            print("DEBUG: AI move skipped - already thinking")
            return

        if self.human_color is None:
            print("DEBUG: AI move skipped - human side not chosen yet")
            return

        self.ai_thinking = True
        self.draw()
        pygame.display.flip()

        try:
            print("DEBUG: make_ai_move() called")
            print("DEBUG: board FEN =", self.board.fen())
            print("DEBUG: turn =", "white" if self.board.turn == chess.WHITE else "black")
            print("DEBUG: legal move count =", self.board.legal_moves.count())

            move = predict_legal_move(self.model, self.vocab, self.board)
            print("DEBUG: predict_legal_move returned:", move)

            if move is None:
                self.model_error = "predict_legal_move returned None"
                print("ERROR: predict_legal_move returned None")

            elif not isinstance(move, chess.Move):
                self.model_error = f"predict_legal_move returned non-chess.Move: {move}"
                print("ERROR: move is not a chess.Move:", type(move), move)

            elif move in self.board.legal_moves:
                print("DEBUG: AI move is legal, pushing move:", move.uci())
                self.board.push(move)
                self.last_move = move
                self.clear_selection()
                self.update_coach_analysis()
            else:
                self.model_error = f"Illegal AI move returned: {move.uci()}"
                print("ERROR: Illegal AI move returned:", move.uci())
                print("DEBUG: legal moves are:", [m.uci() for m in self.board.legal_moves])

        except Exception as e:
            self.model_error = str(e)
            print("ERROR in make_ai_move():", e)
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
        self.update_coach_analysis()

    def choose_side(self, color: chess.Color):
        print("DEBUG: choose_side() called with", "white" if color == chess.WHITE else "black")
        self.human_color = color
        self.ai_color = not color
        self.board_flipped = color == chess.BLACK
        self.reset_game()
        print(
            "DEBUG: side chosen | human =",
            "white" if self.human_color == chess.WHITE else "black",
            "| ai =",
            "white" if self.ai_color == chess.WHITE else "black",
        )

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
        if x < 0 or x >= BOARD_SIZE * SQUARE_SIZE or y < 0 or y >= BOARD_SIZE * SQUARE_SIZE:
            return None

        file = x // SQUARE_SIZE
        rank = 7 - (y // SQUARE_SIZE)

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

        x = file * SQUARE_SIZE
        y = (7 - rank) * SQUARE_SIZE
        return x, y

    def handle_board_click(self, mouse_pos: Tuple[int, int]):
        if self.human_color is None:
            print("DEBUG: board click ignored - choose side first")
            return
        if self.board.turn != self.human_color:
            print("DEBUG: board click ignored - not human turn")
            return
        if self.board.is_game_over():
            print("DEBUG: board click ignored - game over")
            return

        clicked_square = self.screen_to_square(mouse_pos)
        if clicked_square is None:
            return

        piece = self.board.piece_at(clicked_square)

        if self.selected_square is not None:
            if self.try_make_human_move(self.selected_square, clicked_square):
                print("DEBUG: human move made")
                return

        if piece and piece.color == self.human_color:
            self.selected_square = clicked_square
            self.legal_targets = [
                move.to_square
                for move in self.board.legal_moves
                if move.from_square == clicked_square
            ]
            print(
                "DEBUG: selected square =",
                chess.square_name(clicked_square),
                "| legal targets =",
                [chess.square_name(sq) for sq in self.legal_targets],
            )
        else:
            self.clear_selection()

    def try_make_human_move(self, from_sq: int, to_sq: int) -> bool:
        legal_moves = [
            m for m in self.board.legal_moves
            if m.from_square == from_sq and m.to_square == to_sq
        ]

        if not legal_moves:
            print(
                "DEBUG: illegal human move attempt:",
                chess.square_name(from_sq),
                "->",
                chess.square_name(to_sq),
            )
            return False

        promotion_moves = [m for m in legal_moves if m.promotion is not None]
        if promotion_moves:
            self.awaiting_promotion_from = from_sq
            self.awaiting_promotion_to = to_sq
            print("DEBUG: promotion pending")
            return True

        move = legal_moves[0]
        board_before = self.board.copy(stack=False)

        print("DEBUG: human move pushed:", move.uci())
        self.board.push(move)
        self.last_move = move
        self.clear_selection()
        self.update_coach_analysis()
        self.update_move_feedback(board_before, move)
        return True

    def update_move_feedback(self, board_before: chess.Board, played_move: chess.Move):
        if self.model is None or self.vocab is None:
            self.latest_move_feedback = None
            return

        best_analysis = self.coach.analyze_position(board_before, self.model, self.vocab)
        if not best_analysis.top_moves:
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
            self.latest_move_feedback = "Your move was: good. You played the best move."
            return

        reason = explain_bad_move(board_before, played_move, best_move)
        played_san = board_before.san(played_move)
        best_san = board_before.san(best_move)
        self.latest_move_feedback = (
            f"Your move {played_san} was: {classification}. "
            f"Best move was {best_san}. {reason}"
        )

    def get_promotion_option_rects(self) -> List[Tuple[int, pygame.Rect]]:
        panel_w, panel_h = 340, 140
        panel_x = WINDOW_WIDTH // 2 - panel_w // 2
        panel_y = WINDOW_HEIGHT // 2 - panel_h // 2

        options = [
            chess.QUEEN,
            chess.ROOK,
            chess.BISHOP,
            chess.KNIGHT,
        ]

        box_w = 70
        box_h = 70
        gap = 10
        start_x = panel_x + 15
        y = panel_y + 50

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
                move = chess.Move(
                    self.awaiting_promotion_from,
                    self.awaiting_promotion_to,
                    promotion=piece_type,
                )

                if move in self.board.legal_moves:
                    print("DEBUG: promotion move pushed:", move.uci())
                    self.board.push(move)
                    self.last_move = move
                    self.update_coach_analysis()
                    self.update_move_feedback(board_before, move)
                else:
                    print("ERROR: promotion move not legal:", move.uci())

                self.awaiting_promotion_from = None
                self.awaiting_promotion_to = None
                self.clear_selection()
                return

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    print("DEBUG: restart key pressed")
                    self.reset_game()
                elif event.key == pygame.K_f:
                    print("DEBUG: flip key pressed")
                    self.board_flipped = not self.board_flipped
                elif event.key == pygame.K_ESCAPE:
                    print("DEBUG: clear selection")
                    self.clear_selection()

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_pos = event.pos

                if self.white_button.collidepoint(mouse_pos):
                    self.choose_side(chess.WHITE)
                    continue

                if self.black_button.collidepoint(mouse_pos):
                    self.choose_side(chess.BLACK)
                    continue

                if self.restart_button.collidepoint(mouse_pos):
                    print("DEBUG: restart button clicked")
                    self.reset_game()
                    continue

                if self.flip_button.collidepoint(mouse_pos):
                    print("DEBUG: flip button clicked")
                    self.board_flipped = not self.board_flipped
                    continue

                if self.awaiting_promotion_from is not None:
                    self.handle_promotion_click(mouse_pos)
                    continue

                if mouse_pos[0] < BOARD_SIZE * SQUARE_SIZE:
                    self.handle_board_click(mouse_pos)

    def load_piece_images(self) -> Dict[str, pygame.Surface]:
        images: Dict[str, pygame.Surface] = {}

        for symbol, filename in PIECE_IMAGE_NAMES.items():
            path = os.path.join(ASSETS_DIR, filename)
            if os.path.exists(path):
                try:
                    img = pygame.image.load(path).convert_alpha()
                    img = pygame.transform.smoothscale(img, (SQUARE_SIZE - 12, SQUARE_SIZE - 12))
                    images[symbol] = img
                except Exception as e:
                    print(f"WARNING: failed to load piece image {path}: {e}")

        return images

    def draw_board(self):
        for rank in range(8):
            for file in range(8):
                x = file * SQUARE_SIZE
                y = rank * SQUARE_SIZE
                color = LIGHT_SQUARE if (file + rank) % 2 == 0 else DARK_SQUARE
                pygame.draw.rect(self.screen, color, (x, y, SQUARE_SIZE, SQUARE_SIZE))

        if self.last_move:
            for sq in [self.last_move.from_square, self.last_move.to_square]:
                x, y = self.square_to_screen(sq)
                overlay = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                overlay.fill(LAST_MOVE_COLOR)
                self.screen.blit(overlay, (x, y))

        if self.selected_square is not None:
            x, y = self.square_to_screen(self.selected_square)
            overlay = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
            overlay.fill(SELECTED_COLOR)
            self.screen.blit(overlay, (x, y))

        for target in self.legal_targets:
            x, y = self.square_to_screen(target)
            center = (x + SQUARE_SIZE // 2, y + SQUARE_SIZE // 2)

            if self.board.piece_at(target):
                pygame.draw.circle(self.screen, CAPTURE_RING_COLOR, center, 26, 5)
            else:
                pygame.draw.circle(self.screen, MOVE_DOT_COLOR, center, 12)

        for file in range(8):
            file_label = chr(ord("a") + (7 - file if self.board_flipped else file))
            text = self.small_font.render(file_label, True, (20, 20, 20))
            self.screen.blit(text, (file * SQUARE_SIZE + 4, WINDOW_HEIGHT - 22))

        for rank in range(8):
            rank_label = str(rank + 1 if self.board_flipped else 8 - rank)
            text = self.small_font.render(rank_label, True, (20, 20, 20))
            self.screen.blit(text, (4, rank * SQUARE_SIZE + 4))

    def draw_pieces(self):
        for square, piece in self.board.piece_map().items():
            x, y = self.square_to_screen(square)
            symbol = piece.symbol()

            if symbol in self.piece_images:
                self.screen.blit(self.piece_images[symbol], (x + 6, y + 6))
            else:
                text = self.piece_font.render(
                    FALLBACK_UNICODE[symbol],
                    True,
                    (20, 20, 20) if piece.color == chess.BLACK else (250, 250, 250),
                )
                shadow = self.piece_font.render(FALLBACK_UNICODE[symbol], True, (40, 40, 40))
                self.screen.blit(shadow, (x + 18, y + 14))
                self.screen.blit(text, (x + 15, y + 10))

    def draw_button(self, rect: pygame.Rect, text: str):
        mouse_pos = pygame.mouse.get_pos()
        color = BUTTON_HOVER if rect.collidepoint(mouse_pos) else BUTTON_COLOR
        pygame.draw.rect(self.screen, color, rect, border_radius=10)
        pygame.draw.rect(self.screen, ACCENT, rect, 2, border_radius=10)

        label = self.text_font.render(text, True, TEXT_COLOR)
        label_rect = label.get_rect(center=rect.center)
        self.screen.blit(label, label_rect)

    def wrap_text(self, text: str, max_width: int) -> List[str]:
        words = text.split()
        lines: List[str] = []
        current = ""

        for word in words:
            trial = word if not current else f"{current} {word}"
            if self.small_font.size(trial)[0] <= max_width:
                current = trial
            else:
                if current:
                    lines.append(current)
                current = word

        if current:
            lines.append(current)

        return lines

    def draw_side_panel(self):
        panel_x = BOARD_SIZE * SQUARE_SIZE
        pygame.draw.rect(self.screen, PANEL_BG, (panel_x, 0, SIDE_PANEL_WIDTH, WINDOW_HEIGHT))

        self.screen.blit(self.title_font.render("Chess Coach", True, TEXT_COLOR), (panel_x + 20, 20))
        self.screen.blit(self.text_font.render("Play vs AI", True, ACCENT), (panel_x + 20, 60))

        status_text = "Loaded" if self.model_loaded else "Failed"
        self.screen.blit(
            self.small_font.render(f"Model: {status_text}", True, TEXT_COLOR),
            (panel_x + 20, 90),
        )

        self.draw_button(self.white_button, "Play as White")
        self.draw_button(self.black_button, "Play as Black")
        self.draw_button(self.restart_button, "Restart (R)")
        self.draw_button(self.flip_button, "Flip Board (F)")

        y = 450
        turn_text = (
            "Choose a side"
            if self.human_color is None
            else ("White to move" if self.board.turn == chess.WHITE else "Black to move")
        )
        if self.board.is_game_over():
            turn_text = "Game finished"

        self.screen.blit(self.text_font.render(turn_text, True, TEXT_COLOR), (panel_x + 20, y))
        y += 35

        if self.human_color is not None:
            your_side = "White" if self.human_color == chess.WHITE else "Black"
            ai_side = "Black" if self.ai_color == chess.BLACK else "White"
            self.screen.blit(self.small_font.render(f"You: {your_side}", True, TEXT_COLOR), (panel_x + 20, y))
            y += 25
            self.screen.blit(self.small_font.render(f"AI: {ai_side}", True, TEXT_COLOR), (panel_x + 20, y))
            y += 25

        ai_status = "Thinking..." if self.ai_thinking else "Ready"
        self.screen.blit(self.small_font.render(f"Status: {ai_status}", True, TEXT_COLOR), (panel_x + 20, y))
        y += 35

        controls = [
            "Controls:",
            "Click piece to select",
            "Click target square to move",
            "R = restart",
            "F = flip board",
            "Esc = clear selection",
        ]

        for i, line in enumerate(controls):
            font = self.text_font if i == 0 else self.small_font
            color = TEXT_COLOR if i == 0 else (220, 220, 220)
            self.screen.blit(font.render(line, True, color), (panel_x + 20, y))
            y += 28

        if self.latest_analysis is not None:
            y += 10
            self.screen.blit(
                self.text_font.render("Coach", True, ACCENT),
                (panel_x + 20, y),
            )
            y += 30

            coach_lines = [
                self.latest_analysis.winner_hint,
                self.latest_analysis.summary,
                f"Score: {self.latest_analysis.score:.2f}",
                f"Material: {self.latest_analysis.breakdown.material:.2f}",
                f"Mobility: {self.latest_analysis.breakdown.mobility:.2f}",
                f"Center: {self.latest_analysis.breakdown.center_control:.2f}",
                f"King safety: {self.latest_analysis.breakdown.king_safety:.2f}",
            ]

            for line in coach_lines:
                for wrapped in self.wrap_text(line, 260):
                    self.screen.blit(
                        self.small_font.render(wrapped, True, (220, 220, 220)),
                        (panel_x + 20, y),
                    )
                    y += 20
                y += 2

            if self.latest_analysis.top_moves:
                y += 8
                self.screen.blit(
                    self.text_font.render("Top moves", True, ACCENT),
                    (panel_x + 20, y),
                )
                y += 28

                for i, move_info in enumerate(self.latest_analysis.top_moves[:3], start=1):
                    self.screen.blit(
                        self.small_font.render(f"{i}. {move_info.san} ({move_info.score:.2f})", True, TEXT_COLOR),
                        (panel_x + 20, y),
                    )
                    y += 20

                    for line in self.wrap_text(move_info.explanation, 250)[:2]:
                        self.screen.blit(
                            self.small_font.render(line, True, (200, 200, 255)),
                            (panel_x + 30, y),
                        )
                        y += 18
                    y += 4

        if self.latest_move_feedback:
            y += 10
            self.screen.blit(
                self.text_font.render("Last move", True, ACCENT),
                (panel_x + 20, y),
            )
            y += 28

            for line in self.wrap_text(self.latest_move_feedback, 260)[:6]:
                self.screen.blit(
                    self.small_font.render(line, True, (255, 210, 150)),
                    (panel_x + 20, y),
                )
                y += 20

        if self.model_error:
            y += 15
            for line in self.wrap_text(f"Error: {self.model_error}", 260)[:8]:
                self.screen.blit(
                    self.small_font.render(line, True, (255, 170, 170)),
                    (panel_x + 20, y),
                )
                y += 22

        if self.game_over_message:
            y += 15
            for line in self.wrap_text(self.game_over_message, 260):
                self.screen.blit(self.text_font.render(line, True, (255, 220, 120)), (panel_x + 20, y))
                y += 28

    def draw_promotion_dialog(self):
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 140))
        self.screen.blit(overlay, (0, 0))

        panel_w, panel_h = 340, 140
        panel_x = WINDOW_WIDTH // 2 - panel_w // 2
        panel_y = WINDOW_HEIGHT // 2 - panel_h // 2
        panel = pygame.Rect(panel_x, panel_y, panel_w, panel_h)

        pygame.draw.rect(self.screen, (50, 50, 50), panel, border_radius=12)
        pygame.draw.rect(self.screen, ACCENT, panel, 2, border_radius=12)

        title = self.text_font.render("Choose promotion", True, TEXT_COLOR)
        self.screen.blit(title, (panel_x + 90, panel_y + 15))

        for piece_type, rect in self.get_promotion_option_rects():
            color = BUTTON_HOVER if rect.collidepoint(pygame.mouse.get_pos()) else BUTTON_COLOR
            pygame.draw.rect(self.screen, color, rect, border_radius=8)
            pygame.draw.rect(self.screen, ACCENT, rect, 2, border_radius=8)

            symbol = {
                chess.QUEEN: "Q",
                chess.ROOK: "R",
                chess.BISHOP: "B",
                chess.KNIGHT: "N",
            }[piece_type]

            if self.human_color == chess.BLACK:
                symbol = symbol.lower()

            if symbol in self.piece_images:
                img = pygame.transform.smoothscale(self.piece_images[symbol], (56, 56))
                img_rect = img.get_rect(center=rect.center)
                self.screen.blit(img, img_rect)
            else:
                txt = self.piece_font.render(FALLBACK_UNICODE[symbol], True, TEXT_COLOR)
                txt_rect = txt.get_rect(center=rect.center)
                self.screen.blit(txt, txt_rect)

    def draw(self):
        self.screen.fill((0, 0, 0))
        self.draw_board()
        self.draw_pieces()
        self.draw_side_panel()

        if self.awaiting_promotion_from is not None:
            self.draw_promotion_dialog()

    def run(self):
        while True:
            self.handle_events()

            if (
                self.human_color is not None
                and not self.board.is_game_over()
                and self.awaiting_promotion_from is None
                and self.board.turn == self.ai_color
            ):
                print(
                    "DEBUG: AI turn condition met | human_color =",
                    self.human_color,
                    "| ai_color =",
                    self.ai_color,
                    "| board.turn =",
                    self.board.turn,
                )
                self.make_ai_move()

            self.update_game_over_message()
            self.draw()

            pygame.display.flip()
            self.clock.tick(FPS)


def main():
    app = ChessUI()
    app.run()


if __name__ == "__main__":
    main()