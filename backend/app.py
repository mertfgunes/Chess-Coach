import os
import sys

from flask import Flask, request, jsonify
from flask_cors import CORS
import chess

# Allow backend/app.py to import files from src/
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
sys.path.append(SRC_DIR)

from web_ai_adapter import get_ai_move, get_position_evaluation


app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Chess Coach API is running"
    })


@app.route("/new-game", methods=["GET"])
def new_game():
    board = chess.Board()
    return jsonify({
        "fen": board.fen()
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    fen = data.get("fen")
    difficulty = data.get("difficulty", "medium")

    if not fen:
        return jsonify({"error": "Missing FEN"}), 400

    try:
        board = chess.Board(fen)
    except ValueError:
        return jsonify({"error": "Invalid FEN"}), 400

    if board.is_game_over():
        return jsonify({
            "game_over": True,
            "result": board.result()
        })

    try:
        move = get_ai_move(board, difficulty)
    except Exception as e:
        return jsonify({
            "error": f"AI failed: {str(e)}"
        }), 500

    if move is None:
        return jsonify({"error": "AI could not find a legal move"}), 500

    if isinstance(move, str):
        move = chess.Move.from_uci(move)

    if move not in board.legal_moves:
        return jsonify({
            "error": f"AI returned illegal move: {move.uci()}"
        }), 500

    board.push(move)

    return jsonify({
        "move": move.uci(),
        "fen_after": board.fen(),
        "game_over": board.is_game_over(),
        "result": board.result() if board.is_game_over() else None
    })


@app.route("/evaluate", methods=["POST"])
def evaluate():
    data = request.get_json()
    fen = data.get("fen")

    if not fen:
        return jsonify({"error": "Missing FEN"}), 400

    try:
        board = chess.Board(fen)
    except ValueError:
        return jsonify({"error": "Invalid FEN"}), 400

    try:
        evaluation = get_position_evaluation(board)
    except Exception as e:
        return jsonify({
            "error": f"Evaluation failed: {str(e)}"
        }), 500

    return jsonify({
        "evaluation": evaluation
    })


if __name__ == "__main__":
    app.run(debug=True)