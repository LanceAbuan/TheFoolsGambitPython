"""Convert chess board positions to tensor representations for neural network input.

Board representation follows AlphaZero-style: 8x8x12 piece planes + auxiliary channels.

Total channels: 12 (piece planes) + 4 (auxiliary) = 16 channels
- 6 piece types x 2 colors = 12 piece planes
- 1 side-to-move plane
- 1 castling availability plane  
- 1 en passant plane
- 1 halfmove clock plane (normalized)
"""
import chess
import numpy as np


PIECE_ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
NUM_PIECE_CHANNELS = 12  # 6 types x 2 colors
NUM_AUX_CHANNELS = 4
NUM_CHANNELS = NUM_PIECE_CHANNELS + NUM_AUX_CHANNELS  # 16


def board_to_tensor(board):
    """Convert a chess.Board to a numpy tensor of shape (16, 8, 8).
    
    Returns:
        numpy.ndarray: Tensor of shape (16, 8, 8) with float32 values.
    """
    tensor = np.zeros((NUM_CHANNELS, 8, 8), dtype=np.float32)
    
    for pt_idx, piece_type in enumerate(PIECE_ORDER):
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece and piece.piece_type == piece_type:
                rank, file = chess.square_rank(sq), chess.square_file(sq)
                if piece.color == chess.WHITE:
                    tensor[pt_idx, rank, file] = 1.0
                else:
                    tensor[pt_idx + 6, rank, file] = 1.0
    
    side_to_move = 1.0 if board.turn == chess.WHITE else -1.0
    tensor[12, :, :] = side_to_move
    
    _encode_castling(tensor, board)
    _encode_en_passant(tensor, board)
    _encode_halfmove(tensor, board)
    
    return tensor


def _encode_castling(tensor, board):
    """Encode castling rights as a single channel with values in [-1, 1]."""
    castling = 0.0
    if board.has_knight_promotion():
        castling += 0.0
    
    if board.castling_rights:
        if board.has_queenside_castling_rights(chess.WHITE):
            castling += 0.25
        if board.has_kingside_castling_rights(chess.WHITE):
            castling += 0.25
        if board.has_queenside_castling_rights(chess.BLACK):
            castling += 0.25
        if board.has_kingside_castling_rights(chess.BLACK):
            castling += 0.25
    
    tensor[13, :, :] = castling


def _encode_en_passant(tensor, board):
    """Encode en passant target square."""
    if board.ep_square is not None:
        rank, file = chess.square_rank(board.ep_square), chess.square_file(board.ep_square)
        tensor[14, rank, file] = 1.0


def _encode_halfmove(tensor, board):
    """Encode halfmove clock normalized to [0, 1]."""
    normalized = min(board.halfmove_clock / 100.0, 1.0)
    tensor[15, :, :] = normalized


def legal_moves_mask(board):
    """Create a mask of legal moves as (1, 8, 8) binary tensor.
    
    Returns:
        numpy.ndarray: Binary mask where 1 = legal move target square.
    """
    mask = np.zeros((1, 8, 8), dtype=np.float32)
    for move in board.legal_moves:
        rank, file = chess.square_rank(move.to_square), chess.square_file(move.to_square)
        mask[0, rank, file] = 1.0
    return mask


def move_to_idx(move):
    """Convert a chess.Move to a linear index for policy head output.
    
    Uses from_sq * 64 + to_sq encoding (4096 possible moves).
    """
    return move.from_square * 64 + move.to_square


def idx_to_move(idx):
    """Convert a linear index back to a chess.Move."""
    from_sq = idx // 64
    to_sq = idx % 64
    return chess.Move(from_sq, to_sq)


NUM_POSSIBLE_MOVES = 4096  # 64 from squares x 64 to squares
