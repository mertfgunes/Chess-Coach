from torch.utils.data import DataLoader
from move_vocab import MoveVocab
from dataset import ChessDataset

vocab = MoveVocab.build_from_pgns(
    data_dir="data",
    max_moves=5000,
    min_freq=2,
    max_files=1,
    max_games_per_file=200,
    max_positions=100_000,
)

dataset = ChessDataset(
    vocab=vocab,
    data_dir="data",
    max_files=1,
    max_games_per_file=200,
    max_positions=50_000,
    skip_unk=True,
)

loader = DataLoader(dataset, batch_size=32, shuffle=True)

X, extras, policy_y, value_y = next(iter(loader))

print("Batch X:", X.shape)              # (32, 12, 8, 8)
print("Batch extras:", extras.shape)   # (32, 6)
print("Policy y:", policy_y.shape)     # (32,)
print("Value y:", value_y.shape)       # (32,)