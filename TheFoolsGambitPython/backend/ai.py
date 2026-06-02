from typing import Optional
import chess
import random
import chess.pgn


class AIMoveGenerator:
    """
    Chess AI move generator.

    STUB: Currently makes random legal moves.
    Replace the generate() method with your trained model inference.

    To integrate your own model:

        class MyAI:
            def __init__(self):
                self.model = load_your_model()

            def generate(self, board: chess.Board) -> str:
                fen = board.fen()
                best_move = self.model.predict(fen)
                return best_move.uci()

    """

    def generate(self, board: chess.Board) -> Optional[str]:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        # TODO: Replace with your trained AI model
        # return your_model.predict(board)

        return random.choice(legal_moves).uci()


ai = AIMoveGenerator()
