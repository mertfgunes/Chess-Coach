import os
import chess
import chess.pgn

class PGNLoader:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir

    def load_games(self):
        #Generator that yields chess.pgn.Game objects
       
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".pgn"):
                path = os.path.join(self.data_dir, filename)
                with open(path, encoding="utf-8") as pgn_file:
                    while True:
                        game = chess.pgn.read_game(pgn_file)
                        if game is None:
                            break
                        yield game

    def generate_position_move_pairs(self):
        #Generator yielding (board, move) pairs
        
        for game in self.load_games():
            board = game.board()

            for move in game.mainline_moves():
                yield board.copy(), move
                board.push(move)