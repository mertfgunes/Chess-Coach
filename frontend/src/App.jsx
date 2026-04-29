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

  const files = ["a", "b", "c", "d", "e", "f", "g", "h"];
  const ranks = ["8", "7", "6", "5", "4", "3", "2", "1"];

  function handleSquareClick(square) {
    const gameCopy = new Chess(game.fen());

    if (!selectedSquare) {
      const piece = gameCopy.get(square);

      if (!piece) {
        return;
      }

      if (piece.color !== gameCopy.turn()) {
        setStatus("Select your own piece.");
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
    setStatus("You moved. Click Ask AI Move.");
  }

  async function askAiMove() {
    setStatus("AI is thinking...");

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          fen: game.fen(),
          difficulty: difficulty,
        }),
      });

      const data = await response.json();

      if (data.error) {
        setStatus(data.error);
        return;
      }

      if (data.game_over) {
        setStatus(`Game over: ${data.result}`);
        return;
      }

      const gameCopy = new Chess(data.fen_after);
      setGame(gameCopy);
      setSelectedSquare(null);
      setStatus(`AI played: ${data.move}`);
    } catch {
      setStatus("Could not connect to backend.");
    }
  }

  async function evaluatePosition() {
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

      if (data.error) {
        setStatus(data.error);
        return;
      }

      setEvaluation(data.evaluation);
      setStatus("Evaluation updated.");
    } catch {
      setStatus("Could not connect to backend.");
    }
  }

  function resetGame() {
    setGame(new Chess());
    setSelectedSquare(null);
    setStatus("New game started.");
    setEvaluation(null);
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
            }`}
            onClick={() => handleSquareClick(square)}
          >
            <span className={piece?.color === "w" ? "white-piece" : "black-piece"}>
              {piece ? pieceSymbols[piece.color === "w" ? piece.type.toUpperCase() : piece.type] : ""}
            </span>
          </button>
        );
      })
    );
  }

  return (
    <div className="app">
      <div className="container">
        <section className="board-section">
          <h1>Chess Coach</h1>
          <p>Click a piece, then click the target square.</p>

          <div className="custom-board">{renderBoard()}</div>
        </section>

        <aside className="panel">
          <h2>Coach Panel</h2>

          <label htmlFor="difficulty">Difficulty</label>
          <select
            id="difficulty"
            value={difficulty}
            onChange={(e) => setDifficulty(e.target.value)}
          >
            <option value="easy">Easy</option>
            <option value="medium">Medium</option>
            <option value="hard">Hard</option>
          </select>

          <button type="button" onClick={askAiMove}>
            Ask AI Move
          </button>

          <button type="button" onClick={evaluatePosition}>
            Evaluate Position
          </button>

          <button type="button" className="secondary" onClick={resetGame}>
            Reset
          </button>

          <div className="box">
            <h3>Status</h3>
            <p>{status}</p>
          </div>

          <div className="box">
            <h3>Evaluation</h3>
            <p>
              {evaluation === null
                ? "No evaluation yet."
                : `${evaluation} centipawns`}
            </p>
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