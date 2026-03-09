# src/pgn_loader.py

from __future__ import annotations

import os
import chess
import chess.pgn


class PGNLoader:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir

    def _safe_int(self, value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _game_passes_rating_filter(
        self,
        game,
        *,
        min_white_elo: int | None = None,
        min_black_elo: int | None = None,
        min_avg_elo: int | None = None,
    ) -> bool:
        #return 1 if the game passes rating-based filtering.
        
        white_elo = self._safe_int(game.headers.get("WhiteElo"))
        black_elo = self._safe_int(game.headers.get("BlackElo"))

        #if a filter is requested but Elo is missing, reject the game
        if min_white_elo is not None:
            if white_elo is None or white_elo < min_white_elo:
                return False

        if min_black_elo is not None:
            if black_elo is None or black_elo < min_black_elo:
                return False

        if min_avg_elo is not None:
            if white_elo is None or black_elo is None:
                return False
            avg_elo = (white_elo + black_elo) / 2.0
            if avg_elo < min_avg_elo:
                return False

        return True

    def load_games(
        self,
        *,
        max_files=None,
        max_games_per_file=None,
        min_white_elo: int | None = None,
        min_black_elo: int | None = None,
        min_avg_elo: int | None = None,
    ):
        
        #generator that yields (filename, game) for PGN games that pass filters.
        
        files = [f for f in os.listdir(self.data_dir) if f.lower().endswith(".pgn")]
        files.sort()

        if max_files is not None:
            files = files[:max_files]

        for filename in files:
            path = os.path.join(self.data_dir, filename)

            #errors="replace" prevents crashes on weird characters
            with open(path, "r", encoding="utf-8", errors="replace") as pgn_file:
                game_count = 0

                while True:
                    if max_games_per_file is not None and game_count >= max_games_per_file:
                        break

                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        break

                    if not self._game_passes_rating_filter(
                        game,
                        min_white_elo=min_white_elo,
                        min_black_elo=min_black_elo,
                        min_avg_elo=min_avg_elo,
                    ):
                        continue

                    game_count += 1
                    yield filename, game

    def generate_position_move_pairs(
        self,
        *,
        max_files=None,
        max_games_per_file=None,
        max_positions=None,
        min_white_elo: int | None = None,
        min_black_elo: int | None = None,
        min_avg_elo: int | None = None,
    ):
        
        #Generator yielding (board, move) pairs from games that pass filters.
        
        yielded = 0

        for filename, game in self.load_games(
            max_files=max_files,
            max_games_per_file=max_games_per_file,
            min_white_elo=min_white_elo,
            min_black_elo=min_black_elo,
            min_avg_elo=min_avg_elo,
        ):
            board = game.board()

            try:
                for move in game.mainline_moves():
                    if move not in board.legal_moves:
                        break

                    yield board.copy(), move
                    board.push(move)

                    yielded += 1
                    if max_positions is not None and yielded >= max_positions:
                        return

            except Exception:
                # corrupted game, skip
                continue