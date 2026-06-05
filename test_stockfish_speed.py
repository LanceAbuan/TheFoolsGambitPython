import sys
import os
import chess
from training.stockfish_engine import StockfishPlayer
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def test():
    print("[TEST] Initializing StockfishPlayer...")
    sf = StockfishPlayer(depth=12, threads=2, hash_mb=256)
    board = chess.Board()
    print("[TEST] Starting evaluate_legal_moves_batch...")
    start = time.time()
    res = sf.evaluate_legal_moves_batch(board)
    end = time.time()
    print(f"[TEST] Done in {end - start:.2f} seconds. Result size: {len(res)}")

if __name__ == "__main__":
    test()
