from __future__ import annotations

import random

import torch
import chess

from model import PolicyCNN
from move_vocab import MoveVocab
from encoding import board_to_tensor


MODEL_PATH = "checkpoints/best_model.pt"
VOCAB_PATH = "data/move_vocab.txt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# AI behavior settings
TOP_K = 1          # consider the best K legal moves (it was 3 now made it 3 to see its best performance.)
TEMPERATURE = 0.8  # lower = more deterministic, higher = more random


def load_model():
    vocab = MoveVocab.load(VOCAB_PATH)

    model = PolicyCNN(vocab_size=len(vocab))
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model, vocab


@torch.no_grad()
def predict_legal_move(
    model: PolicyCNN,
    vocab: MoveVocab,
    board: chess.Board,
    top_k: int = TOP_K,
    temperature: float = TEMPERATURE,
) -> chess.Move:

    
    #Predict a legal move using legal move masking + top-k sampling.

    #Run the model to get logits for all vocab moves
    #Keep only legal moves known by the vocab
    #Take the top-k legal moves
    #Sample among them with temperature-scaled softmax

    #This makes play less repetitive than always picking argmax.


    x = board_to_tensor(board).unsqueeze(0).to(DEVICE)  # (1, 12, 8, 8)
    logits = model(x).squeeze(0)  # (vocab_size,)

    legal_moves = list(board.legal_moves)
    scored_legal_moves: list[tuple[chess.Move, float]] = []

    for move in legal_moves:
        move_idx = vocab.encode(move)

        #skip unknown moves
        if move_idx == vocab.stoi[vocab.UNK]:
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
    chosen_move = random.choices(moves, weights=probs, k=1)[0]
    return chosen_move


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