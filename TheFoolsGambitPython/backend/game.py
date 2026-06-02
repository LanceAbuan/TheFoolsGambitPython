import chess
from ai import ai


class GameManager:
    def __init__(self):
        self.board = chess.Board()
        self.pgn_moves = []

    def new_game(self):
        self.board.reset()
        self.pgn_moves = []
        return self._state()

    def make_move(self, uci_str):
        move = chess.Move.from_uci(uci_str)
        if move not in self.board.legal_moves:
            return {"error": "Illegal move", "status": "active"}

        self.pgn_moves.append(self.board.san(move))
        self.board.push(move)
        return self._state()

    def ai_move(self):
        uci = ai.generate(self.board)
        if not uci:
            return self._state()
        move = chess.Move.from_uci(uci)
        self.pgn_moves.append(self.board.san(move))
        self.board.push(move)
        return self._state()

    def undo(self):
        if self.board.peek():
            self.board.pop()
            if self.pgn_moves:
                self.pgn_moves.pop()
        return self._state()

    def _state(self):
        status = "active"
        result = None
        if self.board.is_checkmate():
            status = "checkmate"
            winner = "Black" if self.board.turn == chess.WHITE else "White"
            result = f"Checkmate! {winner} wins."
        elif self.board.is_stalemate():
            status = "stalemate"
            result = "Stalemate. Draw."
        elif self.board.is_insufficient_material():
            status = "draw"
            result = "Draw — insufficient material."
        elif self.board.is_repetition():
            status = "draw"
            result = "Draw — repetition."
        elif self.board.halfmove_clock >= 100:
            status = "draw"
            result = "Draw — 50-move rule."

        return {
            "fen": self.board.fen(),
            "legal": [m.uci() for m in self.board.legal_moves],
            "turn": "white" if self.board.turn == chess.WHITE else "black",
            "pgn": " ".join(self.pgn_moves),
            "status": status,
            "result": result,
            "in_check": self.board.is_check(),
        }

    def state(self):
        return self._state()


game_manager = GameManager()
