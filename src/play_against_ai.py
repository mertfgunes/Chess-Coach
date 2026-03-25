from __future__ import annotations

import random

import chess
import torch
import torch.nn as nn

from move_vocab import MoveVocab
from encoding import board_to_tensor
from model import PolicyCNN


MODEL_PATH = "checkpoints/best_model.pt"
VOCAB_PATH = "data/move_vocab.txt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#ai behaviour settings
TOP_K = 1          #best K legal moves (it was 3 now made it 3 to see its best performance.)
TEMPERATURE = 0.8  #lower = more deterministic, higher = more random
MODEL_CHANNELS = 128
MODEL_DROPOUT = 0.1


#implementing this because the AI can not understand the problem with taking the free pieces.
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100,
}


def board_to_extras(board: chess.Board) -> torch.Tensor:
    # These extra features must match training-time extras.
    # 6 features are used here because your checkpoint shape shows 128 + 6 = 134.
    extras = torch.tensor(
        [
            1.0 if board.turn == chess.WHITE else 0.0,
            1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0,
            1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0,
            1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0,
            1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0,
            min(board.halfmove_clock, 100) / 100.0,
        ],
        dtype=torch.float32,
    )
    return extras


def load_model():
    model_path = "checkpoints/best_model.pt"
    vocab_path = "data/move_vocab.txt"

    vocab = MoveVocab.load(vocab_path)

    model = PolicyCNN(
        vocab_size=len(vocab),
        channels=MODEL_CHANNELS,
        dropout=MODEL_DROPOUT,
    ).to(DEVICE)

    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    return model, vocab


def _captured_piece_type(board: chess.Board, move: chess.Move) -> int | None:
    captured_piece = board.piece_at(move.to_square)

    if captured_piece is not None:
        return captured_piece.piece_type

    if board.is_en_passant(move):
        return chess.PAWN

    return None


def _is_move_safe_enough(board: chess.Board, move: chess.Move) -> bool:
    """
    Plays the move on a copy of the board and checks whether the moved piece
    sits on a square attacked by the opponent afterwards.
    """
    board_copy = board.copy(stack=False)
    board_copy.push(move)

    moved_piece = board_copy.piece_at(move.to_square)
    if moved_piece is None:
        return False

    return not board_copy.is_attacked_by(board_copy.turn, move.to_square)


def tactical_capture_score(board: chess.Board, move: chess.Move) -> float:
    """
    Scores capture moves with a stronger tactical heuristic.
    Higher is better.
    """
    if not board.is_capture(move):
        return float("-inf")

    attacker = board.piece_at(move.from_square)
    if attacker is None:
        return float("-inf")

    captured_type = _captured_piece_type(board, move)
    if captured_type is None:
        return float("-inf")

    attacker_value = PIECE_VALUES.get(attacker.piece_type, 0)
    captured_value = PIECE_VALUES.get(captured_type, 0)

    safe_after_capture = _is_move_safe_enough(board, move)

    score = 0.0

    #base priority: winning bigger material is good
    score += captured_value * 20

    #prefer using lower-value attacker to take higher-value victim
    score += (captured_value - attacker_value) * 8

    #strong bonus for truly safe captures
    if safe_after_capture:
        score += 25
    else:
        #penalize unsafe captures, but still allow huge wins like free queen captures
        score -= attacker_value * 6

    #extra bonuses for high-value targets
    if captured_type == chess.QUEEN:
        score += 120
    elif captured_type == chess.ROOK:
        score += 50
    elif captured_type in (chess.BISHOP, chess.KNIGHT):
        score += 20

    return score


def find_best_tactical_move(board: chess.Board) -> chess.Move | None:
    """
    Finds a clearly strong tactical capture if one exists.
    Returns None if there is no obvious tactical override.
    """
    legal_moves = list(board.legal_moves)
    capture_moves = [move for move in legal_moves if board.is_capture(move)]

    if not capture_moves:
        return None

    scored_moves = [(move, tactical_capture_score(board, move)) for move in capture_moves]
    scored_moves.sort(key=lambda x: x[1], reverse=True)

    best_move, best_score = scored_moves[0]

    #only override the neural model if the tactic is clearly attractive.
    #this is tuned to grab hanging queens/rooks much more reliably.
    if best_score >= 35:
        return best_move

    return None


@torch.no_grad()
def predict_legal_move(
    model: nn.Module,
    vocab: MoveVocab,
    board: chess.Board,
    top_k: int = TOP_K,
    temperature: float = TEMPERATURE,
) -> chess.Move:
    #1st tactical override for obvious captures / hanging pieces
    tactical_move = find_best_tactical_move(board)
    if tactical_move is not None:
        return tactical_move

    x = board_to_tensor(board).unsqueeze(0).to(DEVICE)
    extras = board_to_extras(board).unsqueeze(0).to(DEVICE)

    logits = model(x, extras).squeeze(0)

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