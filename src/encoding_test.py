import chess
from encoding import board_to_tensor, board_extras

board = chess.Board()
x = board_to_tensor(board)
e = board_extras(board)

print("board tensor shape:", x.shape)   #should be torch.Size([12, 8, 8])
print("extras shape:", e.shape)         #should be torch.Size([6])
print("sum pieces:", x.sum().item())    #should be 32 at start position