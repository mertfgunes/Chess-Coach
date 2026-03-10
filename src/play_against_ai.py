from __future__ import annotations

import random

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F

from move_vocab import MoveVocab
from encoding import board_to_tensor


MODEL_PATH = "checkpoints/best_model.pt"
VOCAB_PATH = "data/move_vocab.txt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#ai behaviour settings
TOP_K = 1          #best K legal moves (it was 3 now made it 3 to see its best performance.)
TEMPERATURE = 0.8  #lower = more deterministic, higher = more random
MODEL_CHANNELS = 128
MODEL_DROPOUT = 0.1


class PolicyCNNLegacy(nn.Module):
    """
    Matches the checkpoint structure:
    conv1/bn1, conv2/bn2, conv3/bn3, global_pool, dropout, fc
    and fc input size = 128 (no extras concatenated).
    """

    def __init__(self, vocab_size: int, channels: int = 128, dropout: float = 0.1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=12, out_channels=channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(channels)

        self.dropout = nn.Dropout(dropout)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Important: checkpoint expects 128 here, not 134
        self.fc = nn.Linear(channels, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.dropout(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        logits = self.fc(x)
        return logits


def load_model():
    vocab = MoveVocab.load(VOCAB_PATH)

    model = PolicyCNNLegacy(
        vocab_size=len(vocab),
        channels=MODEL_CHANNELS,
        dropout=MODEL_DROPOUT,
    )

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model, vocab


@torch.no_grad()
def predict_legal_move(
    model: nn.Module,
    vocab: MoveVocab,
    board: chess.Board,
    top_k: int = TOP_K,
    temperature: float = TEMPERATURE,
) -> chess.Move:
    x = board_to_tensor(board).unsqueeze(0).to(DEVICE)
    logits = model(x).squeeze(0)

    legal_moves = list(board.legal_moves)
    scored_legal_moves: list[tuple[chess.Move, float]] = []

    unk_idx = vocab.stoi[vocab.UNK]

    for move in legal_moves:
        move_idx = vocab.encode(move)
        if move_idx == unk_idx:
            continue

        score = logits[move_idx].item()
        scored_legal_moves.append((move, score))

    #fallback if all legal moves are unknown to vocab
    if not scored_legal_moves:
        return random.choice(legal_moves)

    #sort best to worst
    scored_legal_moves.sort(key=lambda x: x[1], reverse=True)

    #keep only top-k
    k = max(1, min(top_k, len(scored_legal_moves)))
    candidates = scored_legal_moves[:k]


    #if top_k == 1, deterministic best move
    if k == 1:
        return candidates[0][0]


    #temperature-scaled softmax over candidate scores
    scores = torch.tensor([score for _, score in candidates], dtype=torch.float32)
    
    #prevent divide-by-zero or weird values
    temperature = max(temperature, 1e-6)
    probs = torch.softmax(scores / temperature, dim=0).tolist()

    moves = [move for move, _ in candidates]
    return random.choices(moves, weights=probs, k=1)[0]


def ask_user_color() -> chess.Color:
    while True:
        choice = input("Play as white or black? [w/b]: ").strip().lower()
        if choice == "w":
            return chess.WHITE
        if choice == "b":
            return chess.BLACK
        print("Please type 'w' or 'b'.")


def ask_user_move(board: chess.Board) -> chess.Move:
    while True:
        user_input = input("Your move (UCI like e2e4, or 'quit'): ").strip().lower()

        if user_input in {"quit", "exit"}:
            raise SystemExit("Game ended.")

        try:
            move = chess.Move.from_uci(user_input)
        except ValueError:
            print("Invalid move format. Use UCI format like e2e4 or g1f3.")
            continue

        if move in board.legal_moves:
            return move

        print("Illegal move. Try again.")


def print_game_result(board: chess.Board):
    print("\nFinal board:")
    print(board)
    print()

    if board.is_checkmate():
        winner = "Black" if board.turn == chess.WHITE else "White"
        print(f"Checkmate. {winner} wins.")
    elif board.is_stalemate():
        print("Draw by stalemate.")
    elif board.is_insufficient_material():
        print("Draw by insufficient material.")
    elif board.is_seventyfive_moves():
        print("Draw by 75-move rule.")
    elif board.is_fivefold_repetition():
        print("Draw by fivefold repetition.")
    else:
        print("Game over.")


def main():
    model, vocab = load_model()
    board = chess.Board()

    human_color = ask_user_color()

    print("\nGame start.")
    print(f"AI settings: TOP_K={TOP_K}, TEMPERATURE={TEMPERATURE}")
    print("Board:")
    print(board)
    print()

    while not board.is_game_over():
        print("-" * 50)
        print(board)
        print()

        if board.turn == human_color:
            move = ask_user_move(board)
            board.push(move)
        else:
            ai_move = predict_legal_move(model, vocab, board)
            print(f"AI plays: {ai_move.uci()}")
            board.push(ai_move)

        if board.is_check():
            print("Check!")

    print_game_result(board)


if __name__ == "__main__":
    main()