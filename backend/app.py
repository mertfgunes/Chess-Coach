import os
import sys
import traceback

from flask import Flask, request, jsonify
from flask_cors import CORS
import chess

# Project paths
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BACKEND_DIR)
SRC_DIR = os.path.join(ROOT_DIR, "src")

if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from web_ai_adapter import (
    get_ai_move,
    get_position_evaluation,
    get_coach_advice,
    get_ai_status,
)

app = Flask(__name__)
CORS(app)


def get_json_data():
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


def board_from_fen(fen: str):
    if not fen:
        return None, (jsonify({"error": "Missing FEN"}), 400)

    try:
        return chess.Board(fen), None
    except ValueError:
        return None, (jsonify({"error": "Invalid FEN"}), 400)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Chess Coach API is running",
        "ai_status": get_ai_status(ensure_loaded=True),
    })


@app.route("/status", methods=["GET"])
def status():
    return jsonify(get_ai_status(ensure_loaded=True))


@app.route("/new-game", methods=["GET"])
def new_game():
    board = chess.Board()
    return jsonify({
        "fen": board.fen(),
        "turn": "white" if board.turn == chess.WHITE else "black",
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = get_json_data()
    fen = data.get("fen")
    difficulty = data.get("difficulty", "medium")

    board, error_response = board_from_fen(fen)
    if error_response:
        return error_response

    if board.is_game_over():
        return jsonify({
            "game_over": True,
            "result": board.result(),
            "message": "Game is already over.",
        })

    try:
        move = get_ai_move(board, difficulty)
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": f"AI failed: {str(e)}"
        }), 500

    if move is None:
        return jsonify({
            "error": "AI could not find a legal move."
        }), 500

    if isinstance(move, str):
        try:
            move = chess.Move.from_uci(move)
        except ValueError:
            return jsonify({
                "error": f"AI returned invalid move string: {move}"
            }), 500

    if not isinstance(move, chess.Move):
        return jsonify({
            "error": f"AI returned invalid move type: {type(move)}"
        }), 500

    if move not in board.legal_moves:
        return jsonify({
            "error": f"AI returned illegal move: {move.uci()}"
        }), 500

    try:
        move_san = board.san(move)
    except Exception:
        move_san = move.uci()

    board.push(move)

    response_data = {
        "move": move.uci(),
        "move_san": move_san,
        "fen_after": board.fen(),
        "turn_after": "white" if board.turn == chess.WHITE else "black",
        "game_over": board.is_game_over(),
        "result": board.result() if board.is_game_over() else None,
        "ai_status": get_ai_status(),
    }

    if board.is_game_over():
        response_data.update(get_coach_advice(board, difficulty))

    return jsonify(response_data)


@app.route("/evaluate", methods=["POST"])
def evaluate():
    data = get_json_data()
    fen = data.get("fen")

    board, error_response = board_from_fen(fen)
    if error_response:
        return error_response

    try:
        evaluation = get_position_evaluation(board)
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": f"Evaluation failed: {str(e)}"
        }), 500

    return jsonify({
        "evaluation": evaluation,
        "meaning": "Positive means White is better. Negative means Black is better.",
        "ai_status": get_ai_status(),
    })


@app.route("/coach", methods=["POST"])
def coach():
    data = get_json_data()
    fen = data.get("fen")
    difficulty = data.get("difficulty", "medium")

    board, error_response = board_from_fen(fen)
    if error_response:
        return error_response

    try:
        advice = get_coach_advice(board, difficulty)
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": f"Coach failed: {str(e)}"
        }), 500

    return jsonify(advice)


@app.route("/legal-moves", methods=["POST"])
def legal_moves():
    data = get_json_data()
    fen = data.get("fen")
    square = data.get("square")

    board, error_response = board_from_fen(fen)
    if error_response:
        return error_response

    moves = []
    for move in board.legal_moves:
        if square is None or chess.square_name(move.from_square) == square:
            moves.append({
                "uci": move.uci(),
                "from": chess.square_name(move.from_square),
                "to": chess.square_name(move.to_square),
            })

    return jsonify({
        "legal_moves": moves
    })


if __name__ == "__main__":
    app.run(debug=True)
