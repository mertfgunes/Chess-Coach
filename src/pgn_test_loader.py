from pgn_loader import PGNLoader

loader = PGNLoader()

count = 0
for board, move in loader.generate_position_move_pairs():
    print(board)
    print("Move:", move)
    print("-" * 40)
    count += 1

    if count == 5:
        break