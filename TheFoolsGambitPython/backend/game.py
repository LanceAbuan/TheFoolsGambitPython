import chess


class ChessGame:
    def __init__(self):
        self.board = chess.Board()
        self.move_history = []

    def make_move(self, move_uci: str) -> dict:
        move = chess.Move.from_uci(move_uci)
        if move not in self.board.legal_moves:
            return {"success": False, "error": "Illegal move"}

        self.board.push(move)
        self.move_history.append(move_uci)
        return self._build_response(True, move_uci)

    def undo_move(self) -> dict:
        if not self.move_history:
            return {"success": False, "error": "No moves to undo"}

        self.board.pop()
        self.move_history.pop()
        return self._build_response(True, None)

    def get_state(self) -> dict:
        return self._build_response(True, None)

    def _build_response(self, success: bool, last_move: str = None) -> dict:
        is_over = self.board.is_game_over()
        result_msg = None

        if is_over:
            if self.board.is_checkmate():
                winner = "Black" if self.board.turn == chess.WHITE else "White"
                result_msg = f"{winner} wins by checkmate"
            elif self.board.is_stalemate():
                result_msg = "Draw by stalemate"
            elif self.board.is_insufficient_material():
                result_msg = "Draw by insufficient material"
            elif self.board.is_repetition():
                result_msg = "Draw by threefold repetition"
            elif self.board.is_fifty_moves():
                result_msg = "Draw by fifty moves"
            else:
                result_msg = self.board.result()

        return {
            "success": success,
            "fen": self.board.fen(),
            "legal_moves": [m.uci() for m in self.board.legal_moves],
            "turn": "white" if self.board.turn == chess.WHITE else "black",
            "game_over": is_over,
            "is_check": self.board.is_check(),
            "result": result_msg,
            "move": last_move,
            "move_count": len(self.move_history),
        }

    def get_pgn(self) -> str:
        return self.board.pgn()
