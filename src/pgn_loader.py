import os
import chess
import chess.pgn

class PGNLoader:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir

    def load_games(self):
        #Generator that yields chess.pgn.Game objects
       
        for filename in os.listdir(self.data_dir):
            if not filename.lower().endswith(".pgn"):
                continue

            path = os.path.join(self.data_dir, filename)
            with open(path, "r", encoding="utf-8", errors="replace") as pgn_file:
                while True:
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        break
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
                continue