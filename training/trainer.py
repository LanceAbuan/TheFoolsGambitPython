"""Training loop with Hugging Face persistence.

Auto-pushes model checkpoints to HF Hub. Model persists if local machine goes down.
"""
import os
import json
import time
import shutil
import threading
import torch
import torch.optim as optim
torch.backends.cudnn.benchmark = True

import numpy as np
from collections import deque
from datetime import datetime

from .model import ChessNet
from .tensorize import board_to_tensor, move_to_idx, NUM_POSSIBLE_MOVES
from .selfplay import SelfPlayGame
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

HF_REPO = os.environ.get('HF_REPO', 'LanceAbuan/chess-alpha-zero')
HF_TOKEN = os.environ.get('HF_TOKEN', '')
HF_PUSH_INTERVAL = 50
LOCAL_MODEL_DIR = os.environ.get('MODEL_DIR', '/tmp/chess-models')
MAX_BUFFER_SIZE = 100000
BATCH_SIZE = 64
MIN_BATCH_SIZE = 16
CALIBRATION_GAMES = 10
CALIBRATION_INTERVAL = 50
POLICY_WEIGHT = 1.0
VALUE_WEIGHT = 1.0
L2_REG = 1e-4
LEARNING_RATE = 1e-3


def push_to_hf(model_path, metadata=None):
    if not HF_TOKEN:
        log.info("[HF] No token set - skipping push")
        return False
    try:
        from huggingface_hub import HfApi, upload_file
        api = HfApi(token=HF_TOKEN)
        try:
            api.create_repo(repo_id=HF_REPO, repo_type="model", exist_ok=True)
        except Exception:
            pass
        upload_file(
            path_or_fileobj=model_path,
            path_in_repo="checkpoint.pt",
            repo_id=HF_REPO,
            token=HF_TOKEN
        )
        if metadata:
            meta_path = os.path.join(LOCAL_MODEL_DIR, 'metadata.json')
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            upload_file(
                path_or_fileobj=meta_path,
                path_in_repo="metadata.json",
                repo_id=HF_REPO,
                token=HF_TOKEN
            )
            os.remove(meta_path)
        log.info(f"[HF] Pushed to {HF_REPO}")
        return True
    except Exception as e:
        log.error(f"[HF] Push failed: {e}")
        return False


def download_from_hf():
    if not HF_TOKEN:
        return False
    try:
        from huggingface_hub import hf_hub_download
        os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
        local_path = hf_hub_download(
            repo_id=HF_REPO,
            filename="checkpoint.pt",
            token=HF_TOKEN
        )
        shutil.copy2(local_path, os.path.join(LOCAL_MODEL_DIR, 'checkpoint.pt'))
        log.info(f"[HF] Downloaded from {HF_REPO}")
        return True
    except Exception as e:
        log.error(f"[HF] Download failed: {e}")
        return False


class TrainingBuffer:
    def __init__(self, max_size=MAX_BUFFER_SIZE):
        self.buffer = deque(maxlen=max_size)
        self._lock = threading.Lock()

    def add(self, examples):
        with self._lock:
            self.buffer.extend(examples)

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return None
        # Adaptive: use whatever is available, up to batch_size
        actual_size = min(len(self.buffer), batch_size)
        indices = np.random.choice(len(self.buffer), actual_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        tensors = torch.stack([torch.FloatTensor(ex['board_tensor']) for ex in batch])
        policies = torch.stack([torch.FloatTensor(ex['policy']) for ex in batch])
        values = torch.FloatTensor([ex['value'] for ex in batch])
        return tensors, policies, values

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()


class Trainer:
    def __init__(self, num_residual_blocks=2, residual_filters=32, stockfish=None):
        """Initialize trainer.

        Args:
            stockfish: Optional shared StockfishPlayer instance. If None,
                       creates its own (not recommended — prefer shared instance).
        """
        os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
        # Force CPU if CUDA_VISIBLE_DEVICES is empty or CUDA unavailable
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'default')
        if cuda_visible == '' or not torch.cuda.is_available():
            self.device = torch.device('cpu')
            self.model = ChessNet(num_residual_blocks, residual_filters).to(self.device)
            log.info(f'[TRAINER] Running on CPU (CUDA disabled or unavailable)')
        else:
            try:
                self.device = torch.device('cuda')
                self.model = ChessNet(num_residual_blocks, residual_filters).to(self.device)
                with torch.no_grad():
                    test_input = torch.randn(1, 8, 8, 16).to(self.device)
                    self.model(test_input)
                log.info(f'[TRAINER] Model on CUDA')
            except RuntimeError:
                self.device = torch.device('cpu')
                self.model = ChessNet(num_residual_blocks, residual_filters).to(self.device)
                log.info(f'[TRAINER] CUDA OOM, falling back to CPU')
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.buffer = TrainingBuffer()

        # Use provided stockfish or create fallback
        self.stockfish = stockfish
        if stockfish is None:
            try:
                from .stockfish_engine import StockfishPlayer
                self.stockfish = StockfishPlayer(depth=11)
                log.info(f'[TRAINER] Stockfish initialized (depth=10, own instance)')
            except Exception as e:
                log.info(f'[TRAINER] Stockfish unavailable: {e}')
                self.stockfish = None
        else:
            log.info(f'[TRAINER] Using shared Stockfish instance')

        self.selfplay = SelfPlayGame(self.model, stockfish=self.stockfish)
        self.step = 0
        self.games_played = 0
        self.status = "idle"
        self.loss = 0.0
        self.policy_loss = 0.0
        self.value_loss = 0.0
        self._sf_results = []
        self.started_at = time.time()
        self.last_update = time.time()
        self.running = False
        self.last_hf_push = 0
        self.last_game_pgn = ""
        self.last_game_result = ""
        self.last_game_moves = []

        self._load_checkpoint()
        checkpoint_path = os.path.join(LOCAL_MODEL_DIR, 'checkpoint.pt')
        if not os.path.exists(checkpoint_path) and HF_TOKEN:
            log.info("[HF] No local checkpoint - downloading from HF...")
            download_from_hf()
            self._load_checkpoint()
        if HF_TOKEN:
            log.info(f"[HF] Connected to repo: {HF_REPO}")

    def _load_checkpoint(self):
        checkpoint_path = os.path.join(LOCAL_MODEL_DIR, 'checkpoint.pt')
        if os.path.exists(checkpoint_path):
            data = torch.load(checkpoint_path, map_location='cpu')
            self.model.load_state_dict(data['model_state'])
            self.step = data.get('step', 0)
            self.games_played = data.get('games_played', 0)
            log.info(f"[Checkpoint] Loaded step={self.step}, games={self.games_played}")

    def save_checkpoint(self):
        checkpoint_path = os.path.join(LOCAL_MODEL_DIR, 'checkpoint.pt')
        torch.save({
            'model_state': self.model.state_dict(),
            'step': self.step,
            'games_played': self.games_played,
            'buffer_size': len(self.buffer),
            'loss': self.loss,
        }, checkpoint_path)

    def push_checkpoint(self):
        checkpoint_path = os.path.join(LOCAL_MODEL_DIR, 'checkpoint.pt')
        if not os.path.exists(checkpoint_path):
            self.save_checkpoint()
        metadata = {
            'step': self.step,
            'games_played': self.games_played,
            'buffer_size': len(self.buffer),
            'loss': self.loss,
            'policy_loss': self.policy_loss,
            'value_loss': self.value_loss,
            'last_game_pgn': self.last_game_pgn[:500],
            'last_game_result': self.last_game_result,
            'pushed_at': datetime.utcnow().isoformat(),
            'model_version': 'v1',
        }
        return push_to_hf(checkpoint_path, metadata)

    def play_game(self, on_move=None):
        self.status = "playing"
        self.last_update = time.time()
        game_data = self.selfplay.play(on_move=on_move)
        examples = game_data['examples']
        self.buffer.add(examples)
        self.games_played += 1
        self.last_game_pgn = game_data.get('pgn', '')
        self.last_game_result = game_data.get('result', '*')
        self.last_game_moves = game_data.get('moves', [])
        return {
            'games_played': self.games_played,
            'buffer_size': len(self.buffer),
            'examples_collected': len(examples),
            'pgn': self.last_game_pgn[:200],
            'result': self.last_game_result,
        }

    def train_step(self):
        self.status = "training"
        self.last_update = time.time()
        batch = self.buffer.sample(BATCH_SIZE)
        if batch is None:
            self.status = "idle"
            return None
        
        # Adaptive batch: if buffer is small, use minimum batch threshold
        actual_batch_size = len(batch[0])
        if actual_batch_size < MIN_BATCH_SIZE:
            self.status = "idle"
            log.info(f'[TRAINER] Buffer too small ({actual_batch_size} < {MIN_BATCH_SIZE}), skipping step')
            return None
        tensors, policies, values = batch
        tensors = tensors.to(self.device)
        policies = policies.to(self.device)
        values = values.to(self.device)
        self.model.train()
        self.optimizer.zero_grad()
        pred_policy, pred_value = self.model(tensors)
        p_loss = torch.nn.functional.cross_entropy(pred_policy, policies)
        v_loss = torch.nn.functional.mse_loss(pred_value.squeeze(), values)
        l2 = sum(p.pow(2).sum() for p in self.model.parameters()) * L2_REG
        loss = POLICY_WEIGHT * p_loss + VALUE_WEIGHT * v_loss + l2
        loss.backward()
        self.optimizer.step()
        self.step += 1
        self.loss = loss.item()
        self.policy_loss = p_loss.item()
        self.value_loss = v_loss.item()

        # Run ELO calibration games against Stockfish every N training steps
        if self.step % CALIBRATION_INTERVAL == 0:
            self.status = "stockfish"
            self._calibrate_elo()

        if self.step % 100 == 0:
            self.save_checkpoint()
        if self.step > 0 and (self.step - self.last_hf_push) >= HF_PUSH_INTERVAL:
            self.save_checkpoint()
            self.push_checkpoint()
            self.last_hf_push = self.step
        return {
            'step': self.step,
            'loss': self.loss,
            'policy_loss': self.policy_loss,
            'value_loss': self.value_loss,
            'buffer_size': len(self.buffer)
        }

    def _calibrate_elo(self):
        """Play calibration games vs Stockfish and record results."""
        import chess
        from .tensorize import board_to_tensor, move_to_idx


        if self.stockfish is None:
            return

        self.model.eval()
        results = []
        for _ in range(CALIBRATION_GAMES):
            board = chess.Board()
            for _ in range(200):
                if board.is_game_over():
                    break
                if board.turn == chess.WHITE:
                    # AI moves using the neural network
                    legal = list(board.legal_moves)
                    if not legal:
                        break
                    with torch.no_grad():
                        tensor = torch.FloatTensor(board_to_tensor(board)).to(self.device)
                        pred_policy, _ = self.model(tensor.unsqueeze(0))
                    scores = {}
                    for m in legal:
                        idx = move_to_idx(m)
                        scores[m] = pred_policy[0][idx].item()
                    move = max(scores, key=scores.get)
                else:
                    # Stockfish moves
                    uci = self.stockfish.get_move(board)
                    if uci in ('0-1', 'resign', None, ''):
                        break
                    move = chess.Move.from_uci(uci)
                board.push(move)

            results.append(self._game_result(board))
        self._sf_results.extend(results)
        # Keep only last N calibration games to avoid unbounded growth
        max_history = 50
        if len(self._sf_results) > max_history:
            self._sf_results = self._sf_results[-max_history:]

    def _game_result(self, board):
        """Returns 1.0 for AI win, 0.5 for draw, 0.0 for AI loss."""
        result = board.result()
        if result == '1/2-1/2':
            return 0.5
        if result == '1-0':
            # White won — AI is white, so AI wins
            return 1.0
        # 0-1: Black (Stockfish) won — AI loses
        return 0.0

    def get_status(self):
        model_path = os.path.join(LOCAL_MODEL_DIR, 'checkpoint.pt')
        model_size = os.path.getsize(model_path) if os.path.exists(model_path) else 0
        elo = self.estimate_elo()
        return {
            'status': self.status,
            'step': self.step,
            'games_played': self.games_played,
            'buffer_size': len(self.buffer),
            'loss': self.loss,
            'policy_loss': self.policy_loss,
            'value_loss': self.value_loss,
            'learning_rate': LEARNING_RATE,
            'model_size': model_size,
            'estimated_elo': elo,
            'started_at': self.started_at,
            'last_update': self.last_update,
            'hf_repo': HF_REPO if HF_TOKEN else None,
            'last_game_pgn': self.last_game_pgn[:200],
            'last_game_result': self.last_game_result,
            'last_game_moves': self.last_game_moves[:20],
            'stockfish_available': self.stockfish is not None,
            'sf_blend_ratio': 0.6,
            'sf_calibration_results': self._sf_results[-10:],
        }

    def estimate_elo(self):
        """Estimate network ELO against a Stockfish baseline.

        Calibrated via the Stockfish calibration games. Base ELO is 200.
        Each win adds +20, draw +10, loss +0. Divides by games played to
        get an average score, then maps to ELO.
        """
        base_elo = 200
        if self.games_played == 0:
            return base_elo
        if not self._sf_results:
            return base_elo

        total = sum(self._sf_results[-CALIBRATION_GAMES:])
        count = len(self._sf_results[-CALIBRATION_GAMES:])
        win_rate = total / max(count, 1)
        # Map win_rate [0,1] -> [-200, +400] ELO offset
        elo_offset = int(win_rate * 400 - 200)
        return max(base_elo, base_elo + elo_offset)
