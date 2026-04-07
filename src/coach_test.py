import chess
from coach_service import ChessCoachService

board = chess.Board()
coach = ChessCoachService()

analysis = coach.analyze_position(board)

print("FEN:", analysis.fen)
print("Side to move:", analysis.side_to_move)
print("Score:", analysis.score)
print("Winner hint:", analysis.winner_hint)
print("Summary:", analysis.summary)
print("Breakdown:", analysis.breakdown)