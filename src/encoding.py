from __future__ import annotations

import numpy as np
import chess
import torch

"""
encoding is basically when i look into a board i can understand it
quueen king pawn etc. but when a NN looks at it just is meaningless.


it only understand numbers,vectors and tensors so i have to translate the board into a numeric rep.
so now it is 12 * 8 * 8 tensor 

12 for type of pieces for black and white
and 8 * 8 represents the board
"""



# ordered from 0-11 (first half is white second half is black)
PIECE_TO_PLANE = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


def board_to_tensor(board: chess.Board, *, dtype=torch.float32) -> torch.Tensor:
    """
    Encoding a python-chess Board into a (12, 8, 8) tensor.

    Basically:
    Rank 8 at row 0, rank 1 at row 7.
    - File 'a' at col 0, file 'h' at col 7.
    - 1.0 indicates presence of a piece.
    """
    x = np.zeros((12, 8, 8), dtype=np.float32)

    for square, piece in board.piece_map().items():
        #square: 0..63 (a1=0, h8=63)
        #Convert to (row, col) with row 0 = rank 8
        row = 7 - chess.square_rank(square)
        col = chess.square_file(square)

        base_plane = PIECE_TO_PLANE[piece.piece_type]
        plane = base_plane if piece.color == chess.WHITE else base_plane + 6
        x[plane, row, col] = 1.0

    return torch.tensor(x, dtype=dtype)


def board_extras(board: chess.Board, *, dtype=torch.float32) -> torch.Tensor:
    """
    this might have been added as extra features (small vector) later.

    Returns shape: (6,)

    [0] side_to_move (1 if white, 0 if black)
    [1] white_can_castle_k
    [2] white_can_castle_q
    [3] black_can_castle_k
    [4] black_can_castle_q

    forced move :D
    [5] en_passant_file (0..7) normalized to [0..1], or 0 if none
    """
    side = 1.0 if board.turn == chess.WHITE else 0.0
    wck = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    wcq = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    bck = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    bcq = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

    ep = board.ep_square
    if ep is None:
        ep_file = 0.0
    else:
        ep_file = chess.square_file(ep) / 7.0  #normalize

    return torch.tensor([side, wck, wcq, bck, bcq, ep_file], dtype=dtype)


def move_to_uci(move: chess.Move) -> str:
    #target label helper: return UCI string like 'e2e4' or 'e7e8q'."""
    return move.uci()


def uci_to_move(uci: str) -> chess.Move:
    #inverse helper
    return chess.Move.from_uci(uci)