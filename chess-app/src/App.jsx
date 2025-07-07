import React, { useState } from 'react';
import { Chess } from 'chess.js';
import { Chessboard } from 'react-chessboard';
import axios from 'axios';

function App() {
  const [game, setGame] = useState(new Chess());
  const [suggestedMove, setSuggestedMove] = useState(null);

  const makeMove = ({ from, to }) => {
    const result = game.move({
      from,
      to,
      promotion: 'q', // always promote to queen for simplicity
    });

    if (result) {
      setGame(new Chess(game.fen()));
      setSuggestedMove(null);
      return true;
    }
    return false;
  };

  const onDrop = async (sourceSquare, targetSquare) => {
    const success = makeMove({ from: sourceSquare, to: targetSquare });
    if (success) {
      await getSuggestedMove();
    }
  };

  const getSuggestedMove = async () => {
    try {
      const response = await axios.post('http://127.0.0.1:8000/predict-move', {
        fen: game.fen(),
      });
      setSuggestedMove(response.data.best_move);
    } catch (error) {
      console.error('Failed to fetch move', error);
    }
  };

  const getHighlightSquares = () => {
    if (!suggestedMove) return {};
    const from = suggestedMove.slice(0, 2);
    const to = suggestedMove.slice(2, 4);
    return {
      [from]: { background: 'radial-gradient(circle, #ff0 40%, transparent 45%)' },
      [to]: { background: 'radial-gradient(circle, #0f0 40%, transparent 45%)' },
    };
  };

  return (
    <div style={{ padding: 20 }}>
      <h2>Chess Move Recommender</h2>
      <Chessboard
        position={game.fen()}
        onPieceDrop={onDrop}
        boardWidth={400}
        customSquareStyles={getHighlightSquares()}
      />
      <br />
      <button onClick={getSuggestedMove}>Suggest Move</button>
      {suggestedMove && (
        <p>
          💡 Suggested Move: <strong>{suggestedMove}</strong>
        </p>
      )}
    </div>
  );
}

export default App;
