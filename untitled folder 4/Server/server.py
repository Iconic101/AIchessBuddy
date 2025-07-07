from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np
import chess
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev; restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Piece to channel mapping (same as preprocessing)
piece_to_channel = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5
}

# Load move dictionary created during preprocessing
def generate_move_dict():
    moves = set()
    board = chess.Board()
    for from_sq in chess.SQUARES:
        for to_sq in chess.SQUARES:
            for promo in [None, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                move = chess.Move(from_sq, to_sq, promotion=promo)
                if move in board.legal_moves:
                    moves.add(move.uci())
    return sorted(list(moves))

move_list = generate_move_dict()
index_to_move = {i: m for i, m in enumerate(move_list)}

# Model definition (same as training)
class ChessPolicyNet(torch.nn.Module):
    def __init__(self, num_moves):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(12, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Flatten()
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(128 * 8 * 8, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, num_moves)
        )
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# Load the trained model
model = ChessPolicyNet(num_moves=len(move_list))
model.load_state_dict(torch.load("chess_policy_model.pth", map_location=torch.device('cpu')))
model.eval()

def board_to_tensor_from_fen(fen: str):
    board = chess.Board(fen)
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            channel = piece_to_channel[piece.piece_type] + (6 if piece.color else 0)
            row = 7 - (square // 8)
            col = square % 8
            tensor[channel, row, col] = 1
    return tensor

class BoardState(BaseModel):
    fen: str

@app.post("/predict-move")
def predict_move(state: BoardState):
    input_tensor = board_to_tensor_from_fen(state.fen)
    input_tensor = torch.tensor(input_tensor).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        logits = model(input_tensor)
        predicted_index = torch.argmax(logits, dim=1).item()
    predicted_move = index_to_move[predicted_index]
    return {"best_move": predicted_move}
