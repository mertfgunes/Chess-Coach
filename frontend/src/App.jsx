import { useEffect, useMemo, useState } from "react";
import { Chess } from "chess.js";
import "./App.css";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:5000";

const pieceSymbols = {
  p: "\u265F",
  r: "\u265C",
  n: "\u265E",
  b: "\u265D",
  q: "\u265B",
  k: "\u265A",
  P: "\u2659",
  R: "\u2656",
  N: "\u2658",
  B: "\u2657",
  Q: "\u2655",
  K: "\u2654",
};

const files = ["a", "b", "c", "d", "e", "f", "g", "h"];
const ranks = ["8", "7", "6", "5", "4", "3", "2", "1"];

const difficultyCopy = {
  easy: "Relaxed",
  medium: "Balanced",
  hard: "Sharp",
};

function formatEvaluation(evaluation) {
  if (evaluation === null || Number.isNaN(Number(evaluation))) {
    return "No eval";
  }

  const value = Number(evaluation);
  if (Math.abs(value) >= 90) {
    return value > 0 ? "White mating" : "Black mating";
  }

  const prefix = value > 0 ? "+" : "";
  return `${prefix}${value.toFixed(2)}`;
}

function evaluationLabel(evaluation) {
  if (evaluation === null || Number.isNaN(Number(evaluation))) {
    return "Run an evaluation when you want a read on the position.";
  }

  const value = Number(evaluation);
  if (value > 1.5) return "White has a serious edge.";
  if (value > 0.4) return "White is a little better.";
  if (value < -1.5) return "Black has a serious edge.";
  if (value < -0.4) return "Black is a little better.";
  return "The position is close to equal.";
}

function App() {
  const [game, setGame] = useState(() => new Chess());
  const [selectedSquare, setSelectedSquare] = useState(null);
  const [status, setStatus] = useState("White to move.");
  const [difficulty, setDifficulty] = useState("hard");
  const [evaluation, setEvaluation] = useState(null);
  const [coachMessage, setCoachMessage] = useState(
    "Play a move, then ask for advice when you want a coaching note."
  );
  const [coachInsight, setCoachInsight] = useState({
    title: "Coach Insight",
    summary: "Ask the coach for a focused plan in the current position.",
    explanation: "",
    points: ["Look for forcing moves, loose pieces, and king safety."],
  });
  const [lastAiMove, setLastAiMove] = useState(null);
  const [lastMove, setLastMove] = useState(null);
  const [moveHistory, setMoveHistory] = useState([]);
  const [aiStatus, setAiStatus] = useState(null);
  const [autoReply, setAutoReply] = useState(true);
  const [endGameModal, setEndGameModal] = useState(null);

  const [isAiThinking, setIsAiThinking] = useState(false);
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [isCoachThinking, setIsCoachThinking] = useState(false);
  const [isCheckingStatus, setIsCheckingStatus] = useState(false);

  const isBusy =
    isAiThinking || isEvaluating || isCoachThinking || isCheckingStatus;

  const legalTargets = useMemo(() => {
    if (!selectedSquare) return new Set();
    return new Set(
      game.moves({ square: selectedSquare, verbose: true }).map((move) => move.to)
    );
  }, [game, selectedSquare]);

  useEffect(() => {
    refreshBackendStatus({ quiet: true });
  }, []);

  function updateAiStatus(data) {
    if (data?.ai_status) {
      setAiStatus(data.ai_status);
    }
  }

  function appendMove(actor, move) {
    setMoveHistory((history) => [
      ...history,
      {
        actor,
        san: move.san || move.move_san || move.uci || move.move,
        uci: move.uci || move.move,
      },
    ]);
  }

  function applyCoachData(data) {
    if (!data) return;

    setEvaluation(data.evaluation);
    setCoachMessage(data.message || "");
    setCoachInsight({
      title: data.coach_title || "Coach Insight",
      summary: data.coach_summary || data.message || "No summary available.",
      explanation: data.coach_explanation || "",
      points: Array.isArray(data.coach_points) ? data.coach_points : [],
    });
  }

  function buildEndGameModalData(finalGame, data = {}) {
    const result = finalGame.result();
    let heading = "Game over";
    let tone = "draw";

    if (finalGame.isCheckmate()) {
      const winner = finalGame.turn() === "w" ? "Black" : "White";
      heading = winner === "White" ? "You won by checkmate" : "You lost by checkmate";
      tone = winner === "White" ? "win" : "loss";
    } else if (finalGame.isDraw()) {
      heading = "Draw";
      tone = "draw";
    }

    return {
      heading,
      tone,
      result,
      title: data.coach_title || heading,
      summary: data.coach_summary || data.message || "The game has ended.",
      explanation: data.coach_explanation || "",
      points: Array.isArray(data.coach_points) ? data.coach_points : [],
    };
  }

  function handleSquareClick(square) {
    if (isBusy) {
      setStatus("Let the current analysis finish first.");
      return;
    }

    const gameCopy = new Chess(game.fen());

    if (gameCopy.isGameOver()) {
      setStatus("Game is over. Reset to start a new one.");
      return;
    }

    if (gameCopy.turn() !== "w") {
      setStatus("Black is the AI side. Ask for the AI move.");
      return;
    }

    if (!selectedSquare) {
      const piece = gameCopy.get(square);
      if (!piece || piece.color !== "w") {
        setStatus("Choose one of your white pieces.");
        return;
      }

      setSelectedSquare(square);
      setStatus(`Selected ${square}.`);
      return;
    }

    if (selectedSquare === square) {
      setSelectedSquare(null);
      setStatus("Selection cleared.");
      return;
    }

    const targetPiece = gameCopy.get(square);
    if (targetPiece?.color === "w") {
      setSelectedSquare(square);
      setStatus(`Selected ${square}.`);
      return;
    }

    let move = null;
    try {
      move = gameCopy.move({
        from: selectedSquare,
        to: square,
        promotion: "q",
      });
    } catch {
      move = null;
    }

    if (move === null) {
      setSelectedSquare(null);
      setStatus("That move is not legal.");
      return;
    }

    setGame(gameCopy);
    setSelectedSquare(null);
    setLastMove({ from: move.from, to: move.to });
    setLastAiMove(null);
    appendMove("You", move);

    if (gameCopy.isGameOver()) {
      setStatus(`You played ${move.san}. Game over: ${gameCopy.result()}`);
      setEndGameModal(buildEndGameModalData(gameCopy));
      getCoachAdvice(gameCopy, { allowBusy: true, finalGame: true });
      return;
    }

    setStatus(autoReply ? `You played ${move.san}. AI is thinking.` : `You played ${move.san}.`);

    if (autoReply) {
      askAiMove(gameCopy);
    }
  }

  async function askAiMove(sourceGame = game) {
    if (isBusy && sourceGame === game) {
      return;
    }

    if (sourceGame.isGameOver()) {
      setStatus("Game is already over. Reset to start again.");
      return;
    }

    setIsAiThinking(true);
    setStatus(`${difficultyCopy[difficulty]} AI is calculating...`);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          fen: sourceGame.fen(),
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
      setLastMove({
        from: data.move?.slice(0, 2),
        to: data.move?.slice(2, 4),
      });
      appendMove("AI", {
        san: data.move_san || data.move,
        uci: data.move,
      });

      if (data.game_over) {
        setStatus(`AI played ${data.move_san || data.move}. Game over: ${data.result}`);
        applyCoachData(data);
        setEndGameModal(buildEndGameModalData(gameCopy, data));
      } else {
        setStatus(`AI played ${data.move_san || data.move}. Your move.`);
      }
    } catch {
      setStatus("Backend is not reachable. Start the Flask server first.");
    } finally {
      setIsAiThinking(false);
    }
  }

  async function evaluatePosition() {
    if (isBusy) return;

    setIsEvaluating(true);
    setStatus("Evaluating the position...");

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
      setStatus("Backend is not reachable. Start the Flask server first.");
    } finally {
      setIsEvaluating(false);
    }
  }

  async function getCoachAdvice(sourceGame = game, options = {}) {
    if (isBusy && !options.allowBusy) return;

    setIsCoachThinking(true);
    setStatus(options.finalGame ? "Explaining the final position..." : "Preparing coach advice...");

    try {
      const response = await fetch(`${API_URL}/coach`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          fen: sourceGame.fen(),
          difficulty,
        }),
      });

      const data = await response.json();
      updateAiStatus(data);

      if (data.error) {
        setStatus(data.error);
        return;
      }

      applyCoachData(data);
      if (options.finalGame && sourceGame.isGameOver()) {
        setEndGameModal(buildEndGameModalData(sourceGame, data));
      }
      setStatus(options.finalGame ? "Final explanation ready." : "Coach advice updated.");
    } catch {
      setStatus("Backend is not reachable. Start the Flask server first.");
    } finally {
      setIsCoachThinking(false);
    }
  }

  async function refreshBackendStatus(options = {}) {
    setIsCheckingStatus(true);
    if (!options.quiet) {
      setStatus("Checking engine status...");
    }

    try {
      const response = await fetch(`${API_URL}/status`, {
        cache: "no-store",
      });
      const data = await response.json();

      setAiStatus(data);
      if (!options.quiet) {
        setStatus("Engine status updated.");
      }
    } catch {
      if (!options.quiet) {
        setStatus("Backend is not reachable.");
      }
    } finally {
      setIsCheckingStatus(false);
    }
  }

  async function checkBackendStatus() {
    if (isBusy) return;
    await refreshBackendStatus();
  }

  function resetGame() {
    if (isBusy) {
      setStatus("Let the current action finish first.");
      return;
    }

    setGame(new Chess());
    setSelectedSquare(null);
    setStatus("New game started. White to move.");
    setEvaluation(null);
    setCoachMessage("Play a move, then ask for advice when you want a coaching note.");
    setCoachInsight({
      title: "Coach Insight",
      summary: "Ask the coach for a focused plan in the current position.",
      explanation: "",
      points: ["Look for forcing moves, loose pieces, and king safety."],
    });
    setLastAiMove(null);
    setLastMove(null);
    setMoveHistory([]);
    setEndGameModal(null);
  }

  function renderBoard() {
    return ranks.map((rank, rankIndex) =>
      files.map((file, fileIndex) => {
        const square = `${file}${rank}`;
        const piece = game.get(square);
        const isLight = (rankIndex + fileIndex) % 2 === 0;
        const isSelected = selectedSquare === square;
        const isTarget = legalTargets.has(square);
        const isLastMove = lastMove?.from === square || lastMove?.to === square;

        return (
          <button
            key={square}
            type="button"
            className={`square ${isLight ? "light" : "dark"} ${
              isSelected ? "selected" : ""
            } ${isTarget ? "target" : ""} ${isLastMove ? "last-move" : ""}`}
            onClick={() => handleSquareClick(square)}
            title={square}
            disabled={isBusy}
          >
            <span className="coord file-coord">{rankIndex === 7 ? file : ""}</span>
            <span className="coord rank-coord">{fileIndex === 0 ? rank : ""}</span>
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

  function getGameStateText() {
    if (game.isCheckmate()) return "Checkmate";
    if (game.isStalemate()) return "Stalemate";
    if (game.isDraw()) return "Draw";
    if (game.inCheck()) return `${game.turn() === "w" ? "White" : "Black"} in check`;
    return game.turn() === "w" ? "White to move" : "AI to move";
  }

  return (
    <div className="app">
      {endGameModal ? (
        <div className="endgame-backdrop" role="dialog" aria-modal="true">
          <section className={`endgame-modal ${endGameModal.tone}`}>
            <div className="endgame-kicker">Final position</div>
            <h2>{endGameModal.heading}</h2>
            <div className="endgame-result">{endGameModal.result}</div>
            <h3>{endGameModal.title}</h3>
            <p>{endGameModal.summary}</p>
            {endGameModal.explanation ? <p>{endGameModal.explanation}</p> : null}
            {endGameModal.points.length ? (
              <ul>
                {endGameModal.points.map((point) => (
                  <li key={point}>{point}</li>
                ))}
              </ul>
            ) : null}
            <div className="endgame-actions">
              <button type="button" onClick={() => setEndGameModal(null)}>
                Review board
              </button>
              <button type="button" onClick={resetGame}>
                New game
              </button>
            </div>
          </section>
        </div>
      ) : null}
      <div className="shell">
        <main className="play-area">
          <div className="topbar">
            <div>
              <h1>Chess Coach</h1>
              <p className="subtitle">Play White against the trained AI.</p>
            </div>
            <div className="state-stack">
              <span className="state-pill">{isBusy ? "Thinking" : getGameStateText()}</span>
              <span className="eval-pill">{formatEvaluation(evaluation)}</span>
            </div>
          </div>

          <div className="board-wrap">
            <div className="custom-board">{renderBoard()}</div>
          </div>

          <section className="coach-stage">
            <div className="coach-stage-header">
              <span>Coach</span>
              <button type="button" onClick={() => getCoachAdvice()} disabled={isBusy}>
                {isCoachThinking ? "Analyzing..." : "Explain this position"}
              </button>
            </div>
            <h2>{coachInsight.title}</h2>
            <p>{coachInsight.summary}</p>
            {coachInsight.explanation ? (
              <p className="coach-explanation">{coachInsight.explanation}</p>
            ) : null}
            {coachInsight.points.length ? (
              <ul className="coach-points">
                {coachInsight.points.map((point) => (
                  <li key={point}>{point}</li>
                ))}
              </ul>
            ) : null}
          </section>

          <div className="table-strip">
            <div>
              <span>Last AI move</span>
              <strong>{lastAiMove || "None"}</strong>
            </div>
            <div>
              <span>Position</span>
              <strong>{evaluationLabel(evaluation)}</strong>
            </div>
          </div>
        </main>

        <aside className="panel">
          <section className="control-block">
            <div className="panel-heading">
              <h2>Game Controls</h2>
              <span className={aiStatus?.model_loaded ? "engine-ok" : "engine-idle"}>
                {aiStatus?.model_loaded ? "Model ready" : "Status unknown"}
              </span>
            </div>

            <div className="segmented" aria-label="Difficulty">
              {["easy", "medium", "hard"].map((level) => (
                <button
                  key={level}
                  type="button"
                  className={difficulty === level ? "active" : ""}
                  onClick={() => setDifficulty(level)}
                  disabled={isBusy}
                >
                  {level}
                </button>
              ))}
            </div>

            <label className="toggle-row">
              <input
                type="checkbox"
                checked={autoReply}
                onChange={(event) => setAutoReply(event.target.checked)}
                disabled={isBusy}
              />
              <span>AI replies automatically</span>
            </label>

            <div className="button-grid">
              <button type="button" onClick={() => askAiMove()} disabled={isBusy}>
                {isAiThinking ? "Thinking..." : "AI Move"}
              </button>
              <button type="button" onClick={evaluatePosition} disabled={isBusy}>
                {isEvaluating ? "Reading..." : "Evaluate"}
              </button>
              <button type="button" onClick={checkBackendStatus} disabled={isBusy}>
                {isCheckingStatus ? "Checking..." : "Status"}
              </button>
              <button type="button" onClick={resetGame} disabled={isBusy}>
                Reset game
              </button>
            </div>

            <button
              type="button"
              className="coach-button"
              onClick={() => getCoachAdvice()}
              disabled={isBusy}
            >
              {isCoachThinking ? "Coach is analyzing..." : "Get coach explanation"}
            </button>
          </section>

          <section className="info-block">
            <h3>Status</h3>
            <p>{status}</p>
          </section>

          <section className="info-block">
            <h3>Move List</h3>
            {moveHistory.length ? (
              <ol className="move-list">
                {moveHistory.map((move, index) => (
                  <li key={`${move.uci}-${index}`}>
                    <span>{move.actor}</span>
                    <strong>{move.san}</strong>
                  </li>
                ))}
              </ol>
            ) : (
              <p>No moves yet.</p>
            )}
          </section>

          <section className="info-block">
            <h3>Engine</h3>
            {aiStatus ? (
              <dl className="engine-list">
                <div>
                  <dt>AI</dt>
                  <dd>{aiStatus.real_ai_available ? "Available" : "Unavailable"}</dd>
                </div>
                <div>
                  <dt>Model</dt>
                  <dd>{aiStatus.model_loaded ? "Loaded" : "Not loaded"}</dd>
                </div>
                <div>
                  <dt>Coach</dt>
                  <dd>{aiStatus.real_coach_available ? "Available" : "Unavailable"}</dd>
                </div>
              </dl>
            ) : (
              <p>Check status to confirm the model is loaded.</p>
            )}
          </section>

          <section className="info-block">
            <h3>FEN</h3>
            <p className="fen">{game.fen()}</p>
          </section>
        </aside>
      </div>
    </div>
  );
}

export default App;
