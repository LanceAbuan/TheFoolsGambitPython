import torch
import numpy as np
import chess
import time
import logging
from training.model import ChessNet, NUM_POSSIBLE_MOVES
from training.selfplay import MCTS, BatchEvaluator

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    model = ChessNet().to(device)
    model.eval()

    # Initialize MCTS with BatchEvaluator
    # We use a large batch size to test the batching efficiency
    batch_size = 128
    mcts = MCTS(
        model=model,
        stockfish=None,
        cpuct=1.0,
        noise_epsilon=0.25,
        noise_alpha=0.03,
        batch_size=batch_size,
        use_stockfish=False
    )

    board = chess.Board()
    
    # Warm up
    print("Warming up...")
    for _ in range(5):
        mcts.search(board, num_simulations=10)

    # Benchmark
    num_sims = 200
    print(f"Benchmarking {num_sims} simulations...")
    
    start_time = time.time()
    visit_counts, nn_value = mcts.search(board, num_simulations=num_sims)
    end_time = time.time()

    total_time = end_time - start_time
    avg_time_per_sim = total_time / num_sims

    print(f"Total time for {num_sims} simulations: {total_time:.4f}s")
    print(f"Average time per simulation: {avg_time_per_sim*1000:.4f}ms")
    
    # Test throughput (simulations per second)
    sims_per_sec = num_sims / total_time
    print(f"Throughput: {sims_per_sec:.2f} sims/sec")

if __name__ == "__main__":
    benchmark()
