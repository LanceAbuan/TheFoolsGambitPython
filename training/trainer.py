"""Training loop with Hugging Face persistence.

Auto-pushes model checkpoints to HF Hub. Model persists if local machine goes down.
"""
import os
import json
import time
import shutil
import torch
import torch.optim as optim
import numpy as np
from collections import deque
from datetime import datetime

from .model import ChessNet
from .tensorize import board_to_tensor, move_to_idx, NUM_POSSIBLE_MOVES
from .selfplay import SelfPlayGame
from .stockfish_engine import StockfishPlayer

HF_REPO = os.environ.get('HF_REPO', 'LanceAbuan/chess-alpha-zero')
HF_TOKEN = os.environ.get('HF_TOKEN', '')
HF_PUSH_INTERVAL = 50
LOCAL_MODEL_DIR = os.environ.get('MODEL_DIR', '/tmp/chess-models')
MAX_BUFFER_SIZE = 100000
BATCH_SIZE = 64
LEARNING_RATE = 0.001
POLICY_WEIGHT = 1.0
VALUE_WEIGHT = 1.0
L2_REG = 1e-4


def push_to_hf(model_path, metadata=None):
    if not HF_TOKEN:
        print("[HF] No token set - skipping push")
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
        print(f"[HF] Pushed to {HF_REPO}")
        return True
    except Exception as e:
        print(f"[HF] Push failed: {e}")
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
        print(f"[HF] Downloaded from {HF_REPO}")
        return True
    except Exception as e:
        print(f"[HF] Download failed: {e}")
        return False


class TrainingBuffer:
    def __init__(self, max_size=MAX_BUFFER_SIZE):
        self.buffer = deque(maxlen=max_size)

    def add(self, examples):
        self.buffer.extend(examples)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
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
    def __init__(self, num_residual_blocks=2, residual_filters=32):
        os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
        try:
            self.device = torch.device('cuda')
            self.model = ChessNet(num_residual_blocks, residual_filters).to(self.device)
            with torch.no_grad():
                test_input = torch.randn(1, 8, 8, 16).to(self.device)
                self.model(test_input)
            print(f'[TRAINER] Model on CUDA', flush=True)
        except RuntimeError:
            self.device = torch.device('cpu')
            self.model = ChessNet(num_residual_blocks, residual_filters).to(self.device)
            print(f'[TRAINER] CUDA OOM, falling back to CPU', flush=True)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.buffer = TrainingBuffer()

        self.stockfish = None
        try:
            self.stockfish = StockfishPlayer(depth=10)
            print(f'[TRAINER] Stockfish initialized (depth=10)', flush=True)
        except Exception as e:
            print(f'[TRAINER] Stockfish unavailable: {e}', flush=True)

        self.selfplay = SelfPlayGame(self.model, stockfish=self.stockfish)
        self.step = 0
        self.games_played = 0
        self.status = "idle"
        self.loss = 0.0
        self.policy_loss = 0.0
        self.value_loss = 0.0
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
            print("[HF] No local checkpoint - downloading from HF...")
            download_from_hf()
            self._load_checkpoint()
        if HF_TOKEN:
            print(f"[HF] Connected to repo: {HF_REPO}")

    def _load_checkpoint(self):
        checkpoint_path = os.path.join(LOCAL_MODEL_DIR, 'checkpoint.pt')
        if os.path.exists(checkpoint_path):
            data = torch.load(checkpoint_path, map_location='cpu')
            self.model.load_state_dict(data['model_state'])
            self.step = data.get('step', 0)
            self.games_played = data.get('games_played', 0)
            print(f"[Checkpoint] Loaded step={self.step}, games={self.games_played}")

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
        }

    def estimate_elo(self):
        import math
        base_elo = 200
        if self.games_played == 0:
            return base_elo
        games_factor = min(math.log2(max(self.games_played, 1)) * 80, 400)
        steps_factor = min(math.log2(max(self.step, 1)) * 40, 300)
        loss_factor = 0
        if self.loss and self.loss < 1.0:
            loss_factor = (1 - self.loss) * 200
        return int(base_elo + games_factor + steps_factor + loss_factor)
