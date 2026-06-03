"""Training loop with data management and progress tracking.

The trainer:
1. Manages a replay buffer of self-play training examples
2. Runs training steps with mini-batch SGD/Adam
3. Saves/checkpoints the model periodically
4. Tracks training metrics: loss, game results, etc.
5. Exposes status via a simple status dict for the API

Data flows:
- Self-play generates examples -> replay buffer
- Trainer samples batches from buffer -> updates model
- Model improvements feed back into self-play

Loss function:
- Policy loss: cross-entropy between predicted and target policy
- Value loss: MSE between predicted and target value
- Total loss: policy_loss + value_loss + l2_reg
"""
import os
import json
import time
import torch
import torch.optim as optim
import numpy as np
from collections import deque

from .model import ChessNet
from .tensorize import board_to_tensor, move_to_idx, NUM_POSSIBLE_MOVES
from .selfplay import SelfPlayGame

MODEL_DIR = os.environ.get('MODEL_DIR', '/tmp/chess-models')
CHECKPOINT_INTERVAL = 100  # Steps between checkpoints
MAX_BUFFER_SIZE = 100000  # Max training examples in buffer
BATCH_SIZE = 64
LEARNING_RATE = 0.001
POLICY_WEIGHT = 1.0
VALUE_WEIGHT = 1.0
L2_REG = 1e-4


class TrainingBuffer:
    """Stores self-play training examples for batched training."""
    
    def __init__(self, max_size=MAX_BUFFER_SIZE):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, examples):
        """Add a batch of training examples."""
        self.buffer.extend(examples)
    
    def sample(self, batch_size):
        """Sample a random batch from the buffer."""
        if len(self.buffer) < batch_size:
            return None
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        tensors = torch.stack([
            torch.FloatTensor(ex['board_tensor']) for ex in batch
        ])
        policies = torch.stack([
            torch.FloatTensor(ex['policy']) for ex in batch
        ])
        values = torch.FloatTensor([ex['value'] for ex in batch])
        
        return tensors, policies, values
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        self.buffer.clear()


class Trainer:
    """Manages the complete training loop.
    
    Status dict (exposed to API):
    {
        "status": "idle" | "playing" | "training",
        "step": int,           # total training steps
        "games_played": int,   # self-play games completed
        "buffer_size": int,    # examples in replay buffer
        "loss": float,         # current training loss
        "policy_loss": float,  # policy head loss
        "value_loss": float,   # value head loss
        "learning_rate": float,
        "model_size": int,     # model file size in bytes
        "started_at": float,   # unix timestamp
        "last_update": float,  # unix timestamp
    }
    """
    
    def __init__(self, num_residual_blocks=2, residual_filters=32):
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        self.model = ChessNet(num_residual_blocks, residual_filters)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.buffer = TrainingBuffer()
        self.selfplay = SelfPlayGame(self.model)
        
        self.step = 0
        self.games_played = 0
        self.status = "idle"
        self.loss = 0.0
        self.policy_loss = 0.0
        self.value_loss = 0.0
        self.started_at = time.time()
        self.last_update = time.time()
        self.running = False
        
        self._load_checkpoint()
    
    def _load_checkpoint(self):
        """Load model from disk if available."""
        checkpoint_path = os.path.join(MODEL_DIR, 'checkpoint.pt')
        if os.path.exists(checkpoint_path):
            data = torch.load(checkpoint_path)
            self.model.load_state_dict(data['model_state'])
            self.step = data.get('step', 0)
            self.games_played = data.get('games_played', 0)
    
    def save_checkpoint(self):
        """Save model and training state to disk."""
        checkpoint_path = os.path.join(MODEL_DIR, 'checkpoint.pt')
        torch.save({
            'model_state': self.model.state_dict(),
            'step': self.step,
            'games_played': self.games_played,
            'buffer_size': len(self.buffer),
        }, checkpoint_path)
    
    def save_model(self, filename='model.pt'):
        """Save just the model weights."""
        path = os.path.join(MODEL_DIR, filename)
        torch.save(self.model.state_dict(), path)
        return path
    
    def play_game(self):
        """Play a single self-play game and add examples to buffer."""
        self.status = "playing"
        self.last_update = time.time()
        
        examples = self.selfplay.play()
        self.buffer.add(examples)
        self.games_played += 1
        
        return {
            'games_played': self.games_played,
            'buffer_size': len(self.buffer),
            'examples_collected': len(examples)
        }
    
    def train_step(self):
        """Run a single training step (one batch)."""
        self.status = "training"
        self.last_update = time.time()
        
        batch = self.buffer.sample(BATCH_SIZE)
        if batch is None:
            self.status = "idle"
            return None
        
        tensors, policies, values = batch
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
        
        if self.step % CHECKPOINT_INTERVAL == 0:
            self.save_checkpoint()
        
        return {
            'step': self.step,
            'loss': self.loss,
            'policy_loss': self.policy_loss,
            'value_loss': self.value_loss,
            'buffer_size': len(self.buffer)
        }
    
    def get_status(self):
        """Get current training status dict."""
        model_path = os.path.join(MODEL_DIR, 'checkpoint.pt')
        model_size = os.path.getsize(model_path) if os.path.exists(model_path) else 0
        
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
            'started_at': self.started_at,
            'last_update': self.last_update,
        }
