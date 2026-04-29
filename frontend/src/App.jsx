import { useState } from "react";
import { Chess } from "chess.js";
import "./App.css";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:5000";

const pieceSymbols = {
  p: "♟",
  r: "♜",
  n: "♞",
  b: "♝",
  q: "♛",
  k: "♚",
  P: "♙",
  R: "♖",
  N: "♘",
  B: "♗",
  Q: "♕",
  K: "♔",
};

function App() {
  const [game, setGame] = useState(() => new Chess());
  const [selectedSquare, setSelectedSquare] = useState(null);
  const [status, setStatus] = useState("Your move.");
  const [difficulty, setDifficulty] = useState("medium");
  const [evaluation, setEvaluation] = useState(null);
  const [coachMessage, setCoachMessage] = useState("No coach advice yet.");
  const [lastAiMove, setLastAiMove] = useState(null);
  const [aiStatus, setAiStatus] = useState(null);

  // New loading states
  const [isAiThinking, setIsAiThinking] = useState(false);
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [isCoachThinking, setIsCoachThinking] = useState(false);
  const [isCheckingStatus, setIsCheckingStatus] = useState(false);

  const files = ["a", "b", "c", "d", "e", "f", "g", "h"];
  const ranks = ["8", "7", "6", "5", "4", "3", "2", "1"];

  const isBusy =
    isAiThinking || isEvaluating || isCoachThinking || isCheckingStatus;

  function updateAiStatus(data) {
    if (data && data.ai_status) {
      setAiStatus(data.ai_status);
    }
  }

  function handleSquareClick(square) {
    if (isBusy) {
      setStatus("Please wait until the current action finishes.");
      return;
    }

    const gameCopy = new Chess(game.fen());

    if (gameCopy.isGameOver()) {
      setStatus("Game is over. Press Reset to start again.");
      return;
    }

    if (!selectedSquare) {
      const piece = gameCopy.get(square);

      if (!piece) {
        return;
      }

      if (piece.color !== gameCopy.turn()) {
        setStatus("Select the side whose turn it is.");
        return;
      }

      setSelectedSquare(square);
      setStatus(`Selected ${square}. Choose a target square.`);
      return;
    }

    const move = gameCopy.move({
      from: selectedSquare,
      to: square,
      promotion: "q",
    });

    if (move === null) {
      setSelectedSquare(null);
      setStatus("Illegal move.");
      return;
    }

    setGame(gameCopy);
    setSelectedSquare(null);
    setLastAiMove(null);
    setStatus(`You played ${move.san}. Click Ask AI Move.`);
  }

  async function askAiMove() {
    if (isBusy) {
      return;
    }

    if (game.isGameOver()) {
      setStatus("Game is already over. Press Reset to start again.");
      return;
    }

    setIsAiThinking(true);
    setStatus("AI is thinking...");

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          fen: game.fen(),
          difficulty,
        }),
      });

      const data = await response.json();
      updateAiStatus(data);

      if (data.error) {
        setStatus(data.error);
        return;
      }

      if (data.game_over && !data.move) {
        setStatus(`Game over: ${data.result}`);
        return;
      }

      const gameCopy = new Chess(data.fen_after);
      setGame(gameCopy);
      setSelectedSquare(null);
      setLastAiMove(data.move_san || data.move);

      if (data.game_over) {
        setStatus(
          `AI played ${data.move_san || data.move}. Game over: ${data.result}`
        );
      } else {
        setStatus(`AI played ${data.move_san || data.move}. Your move.`);
      }
    } catch {
      setStatus(
        "Could not connect to backend. Make sure python backend/app.py is running."
      );
    } finally {
      setIsAiThinking(false);
    }
  }

  async function evaluatePosition() {
    if (isBusy) {
      return;
    }

    setIsEvaluating(true);
    setStatus("Evaluating...");

    try {
      const response = await fetch(`${API_URL}/evaluate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          fen: game.fen(),
        }),
      });

      const data = await response.json();
      updateAiStatus(data);

      if (data.error) {
        setStatus(data.error);
        return;
      }

      setEvaluation(data.evaluation);
      setStatus("Evaluation updated.");
    } catch {
      setStatus(
        "Could not connect to backend. Make sure python backend/app.py is running."
      );
    } finally {
      setIsEvaluating(false);
    }
  }

  async function getCoachAdvice() {
    if (isBusy) {
      return;
    }

    setIsCoachThinking(true);
    setStatus("Coach is thinking...");

    try {
      const response = await fetch(`${API_URL}/coach`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          fen: game.fen(),
          difficulty,
        }),
      });

      const data = await response.json();
      updateAiStatus(data);

      if (data.error) {
        setStatus(data.error);
        return;
      }

      setEvaluation(data.evaluation);
      setCoachMessage(data.message);
      setStatus("Coach advice updated.");
    } catch {
      setStatus(
        "Could not connect to backend. Make sure python backend/app.py is running."
      );
    } finally {
      setIsCoachThinking(false);
    }
  }

  async function checkBackendStatus() {
    if (isBusy) {
      return;
    }

    setIsCheckingStatus(true);
    setStatus("Checking backend...");

    try {
      const response = await fetch(`${API_URL}/status`);
      const data = await response.json();

      setAiStatus(data);
      setStatus("Backend status updated.");
    } catch {
      setStatus("Backend is not reachable.");
    } finally {
      setIsCheckingStatus(false);
    }
  }

  function resetGame() {
    if (isBusy) {
      setStatus("Please wait until the current action finishes.");
      return;
    }

    setGame(new Chess());
    setSelectedSquare(null);
    setStatus("New game started.");
    setEvaluation(null);
    setCoachMessage("No coach advice yet.");
    setLastAiMove(null);
  }

  function renderBoard() {
    return ranks.map((rank, rankIndex) =>
      files.map((file, fileIndex) => {
        const square = `${file}${rank}`;
        const piece = game.get(square);
        const isLight = (rankIndex + fileIndex) % 2 === 0;
        const isSelected = selectedSquare === square;

        return (
          <button
            key={square}
            type="button"
            className={`square ${isLight ? "light" : "dark"} ${
              isSelected ? "selected" : ""
            } ${isBusy ? "board-disabled" : ""}`}
            onClick={() => handleSquareClick(square)}
            title={square}
            disabled={isBusy}
          >
            <span className={piece?.color === "w" ? "white-piece" : "black-piece"}>
              {piece
                ? pieceSymbols[
                    piece.color === "w" ? piece.type.toUpperCase() : piece.type
                  ]
                : ""}
            </span>
          </button>
        );
      })
    );
  }

  function getTurnText() {
    return game.turn() === "w" ? "White to move" : "Black to move";
  }

  function getGameStateText() {
    if (game.isCheckmate()) {
      return "Checkmate";
    }

    if (game.isStalemate()) {
      return "Stalemate";
    }

    if (game.isDraw()) {
      return "Draw";
    }

    if (game.inCheck()) {
      return `${getTurnText()} - check`;
    }

    return getTurnText();
  }

  return (
    <div className="app">
      <div className="container">
        <section className="board-section">
          <div className="header-row">
            <div>
              <h1>Chess Coach</h1>
              <p>Click a piece, then click the target square.</p>
            </div>

            <div className="turn-pill">{isBusy ? "Working..." : getGameStateText()}</div>
          </div>

          <div className="custom-board">{renderBoard()}</div>
        </section>

        <aside className="panel">
          <h2>Coach Panel</h2>

          <label htmlFor="difficulty">Difficulty</label>
          <select
            id="difficulty"
            value={difficulty}
            onChange={(e) => setDifficulty(e.target.value)}
            disabled={isBusy}
          >
            <option value="easy">Easy</option>
            <option value="medium">Medium</option>
            <option value="hard">Hard</option>
          </select>

          <button type="button" onClick={askAiMove} disabled={isBusy}>
            {isAiThinking ? "AI Thinking..." : "Ask AI Move"}
          </button>

          <button type="button" onClick={evaluatePosition} disabled={isBusy}>
            {isEvaluating ? "Evaluating..." : "Evaluate Position"}
          </button>

          <button type="button" onClick={getCoachAdvice} disabled={isBusy}>
            {isCoachThinking ? "Coach Thinking..." : "Get Coach Advice"}
          </button>

          <button type="button" onClick={checkBackendStatus} disabled={isBusy}>
            {isCheckingStatus ? "Checking..." : "Check Backend Status"}
          </button>

          <button
            type="button"
            className="secondary"
            onClick={resetGame}
            disabled={isBusy}
          >
            Reset
          </button>

          <div className="box">
            <h3>Status</h3>
            <p>{status}</p>
          </div>

          <div className="box">
            <h3>Last AI Move</h3>
            <p>{lastAiMove || "No AI move yet."}</p>
          </div>

          <div className="box">
            <h3>Evaluation</h3>
            <p>
              {evaluation === null
                ? "No evaluation yet."
                : `${Number(evaluation).toFixed(2)} ${
                    Math.abs(evaluation) > 20 ? "points" : "score"
                  }`}
            </p>
          </div>

          <div className="box coach-box">
            <h3>Coach Advice</h3>
            <p>{coachMessage}</p>
          </div>

          <div className="box">
            <h3>AI Status</h3>
            {aiStatus ? (
              <ul className="status-list">
                <li>
                  Real AI: {aiStatus.real_ai_available ? "Available" : "Unavailable"}
                </li>
                <li>Model loaded: {aiStatus.model_loaded ? "Yes" : "No"}</li>
                <li>
                  Real eval:{" "}
                  {aiStatus.real_eval_available ? "Available" : "Unavailable"}
                </li>
                {aiStatus.ai_error ? (
                  <li className="error-text">{aiStatus.ai_error}</li>
                ) : null}
              </ul>
            ) : (
              <p>Not checked yet.</p>
            )}
          </div>

          <div className="box">
            <h3>FEN</h3>
            <p className="fen">{game.fen()}</p>
          </div>
        </aside>
      </div>
    </div>
  );
}

export default App;