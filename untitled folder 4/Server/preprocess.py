# Save this as preprocess.py or similar

import chess
import chess.pgn
import numpy as np

piece_to_channel = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5
}

def board_to_tensor(board):
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_type = piece.piece_type
            color = int(piece.color)
            channel = piece_to_channel[piece_type] + (6 * color)
            row = 7 - (square // 8)
            col = square % 8
            tensor[channel, row, col] = 1
    return tensor

def generate_move_dict():
    import chess
    moves = set()
    board = chess.Board()
    for from_sq in chess.SQUARES:
        for to_sq in chess.SQUARES:
            for promo in [None, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                move = chess.Move(from_sq, to_sq, promotion=promo)
                if move in board.legal_moves:
                    moves.add(move.uci())
    return sorted(list(moves))

def pgn_to_dataset(pgn_path, move_to_index, max_games=100):
    X = []
    y = []
    count_games = 0
    with open(pgn_path) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None or count_games >= max_games:
                break
            board = game.board()
            for move in game.mainline_moves():
                X.append(board_to_tensor(board))
                move_uci = move.uci()
                if move_uci in move_to_index:
                    y.append(move_to_index[move_uci])
                else:
                    y.append(-1)  # skip unknown moves
                board.push(move)
            count_games += 1
    # Filter invalid moves
    X = np.array([x for i,x in enumerate(X) if y[i] != -1])
    y = np.array([label for label in y if label != -1])
    return X, y

if __name__ == "__main__":
    pgn_file_path = "lichess_sample.pgn"  # your downloaded file
    move_list = generate_move_dict()
    move_to_index = {m: i for i, m in enumerate(move_list)}

    X, y = pgn_to_dataset(pgn_file_path, move_to_index, max_games=100)
    print(f"Extracted {X.shape[0]} positions and {y.shape[0]} moves.")

    np.save("X.npy", X)
    np.save("y.npy", y)
