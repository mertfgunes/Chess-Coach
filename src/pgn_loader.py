# src/pgn_loader.py

import os
import chess
import chess.pgn


class PGNLoader:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir

    def load_games(self, *, max_files=None, max_games_per_file=None):
        #Generator that yields chess.pgn.Game objects
        files = [f for f in os.listdir(self.data_dir) if f.lower().endswith(".pgn")]
        files.sort()

        if max_files is not None:
            files = files[:max_files]

        for filename in files:
            path = os.path.join(self.data_dir, filename)

            # errors="replace" prevents crashes on weird characters
            with open(path, "r", encoding="utf-8", errors="replace") as pgn_file:
                game_count = 0
                while True:
                    if max_games_per_file is not None and game_count >= max_games_per_file:
                        break

                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        break

                    game_count += 1
                    yield filename, game

    def generate_position_move_pairs(
        self,
        *,
        max_files=None,
        max_games_per_file=None,
        max_positions=None,
    ):
        #Generator yielding (board, move) pairs
        yielded = 0

        for filename, game in self.load_games(max_files=max_files, max_games_per_file=max_games_per_file):
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