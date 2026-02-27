from torch.utils.data import DataLoader
from move_vocab import MoveVocab
from dataset import ChessDataset

#build vocab
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

X, y = next(iter(loader))
print("Batch X:", X.shape)  #expected 32 12 8 8
print("Batch y:", y.shape)  #expected 32
print("y min/max:", y.min().item(), y.max().item())