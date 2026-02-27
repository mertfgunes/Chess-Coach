from move_vocab import MoveVocab
from pgn_loader import PGNLoader

vocab = MoveVocab.build_from_pgns(
    data_dir="data",
    max_moves=5000,
    min_freq=2,
    max_files=1,              #only first PGN file
    max_games_per_file=200,   #only first 200 games
    max_positions=100_000,    #only first 100k positions
)
vocab.save("data/move_vocab.txt")

print("Vocab size:", len(vocab))
print("Index of e2e4:", vocab.encode("e2e4"))
print("Decode 2:", vocab.decode(2))

# Test a few moves from PGNs
loader = PGNLoader("data")
for i, (_board, move) in enumerate(loader.generate_position_move_pairs()):
    idx = vocab.encode(move)
    print(move.uci(), "->", idx, "(known)" if idx != 1 else "(UNK)")
    if i == 10:
        break