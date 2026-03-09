from __future__ import annotations

import torch
import chess

from model import PolicyCNN
from move_vocab import MoveVocab
from encoding import board_to_tensor


MODEL_PATH = "checkpoints/best_model.pt"
VOCAB_PATH = "data/move_vocab.txt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    vocab = MoveVocab.load(VOCAB_PATH)

    model = PolicyCNN(vocab_size=len(vocab))
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model, vocab


@torch.no_grad()
def predict_legal_move(model: PolicyCNN, vocab: MoveVocab, board: chess.Board) -> chess.Move:
    #predict the best legal move
    #only moves that are legal is considered
    #rest is just oignored
    x = board_to_tensor(board).unsqueeze(0).to(DEVICE)  # (1, 12, 8, 8)
    logits = model(x).squeeze(0)  # (vocab_size,)

    legal_moves = list(board.legal_moves)

    best_move = None
    best_score = float("-inf")

    for move in legal_moves:
        move_idx = vocab.encode(move)

        #skip moves unknown to vocab
        if move_idx == vocab.stoi[vocab.UNK]:
            continue

        score = logits[move_idx].item()
        if score > best_score:
            best_score = score
            best_move = move

    #fallback if all legal moves are UNK
    if best_move is None:
        best_move = legal_moves[0]

    return best_move


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
    ai_color = not human_color

    print("\nGame start.")
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