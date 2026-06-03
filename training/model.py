"""Neural network model for chess evaluation.

Architecture inspired by AlphaZero:
- Input: 16-channel board tensor (8x8x16)
- Residual tower: N residual blocks with batch norm
- Policy head: conv -> softmax over legal moves
- Value head: conv -> fc -> tanh (position evaluation)

The model outputs:
- policy_logits: raw logits over 4096 possible moves (before softmax)
- value: scalar in [-1, 1] representing win probability for side to move
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_CHANNELS = 16
NUM_RESIDUAL_BLOCKS = 2
RESIDUAL_FILTERS = 32
POLICY_HEAD_CHANNELS = 4
VALUE_HEAD_CHANNELS = 32
NUM_POSSIBLE_MOVES = 4096


class ResidualBlock(nn.Module):
    """Residual block with batch normalization and ReLU."""
    
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


class ChessNet(nn.Module):
    """AlphaZero-style chess neural network.
    
    Args:
        num_residual_blocks: Number of residual blocks in the tower.
        residual_filters: Number of filters in each residual block.
    """
    
    def __init__(self, num_residual_blocks=NUM_RESIDUAL_BLOCKS, residual_filters=RESIDUAL_FILTERS):
        super().__init__()
        self.residual_filters = residual_filters
        
        self.input_conv = nn.Conv2d(NUM_CHANNELS, residual_filters, 3, padding=1)
        self.input_bn = nn.BatchNorm2d(residual_filters)
        
        self.residual_tower = nn.Sequential(*[
            ResidualBlock(residual_filters) for _ in range(num_residual_blocks)
        ])
        
        self.policy_conv = nn.Conv2d(residual_filters, POLICY_HEAD_CHANNELS, 1)
        self.policy_bn = nn.BatchNorm2d(POLICY_HEAD_CHANNELS)
        self.policy_fc = nn.Linear(POLICY_HEAD_CHANNELS * 64, NUM_POSSIBLE_MOVES)
        
        self.value_conv = nn.Conv2d(residual_filters, VALUE_HEAD_CHANNELS, 1)
        self.value_bn = nn.BatchNorm2d(VALUE_HEAD_CHANNELS)
        self.value_fc1 = nn.Linear(VALUE_HEAD_CHANNELS * 64, VALUE_HEAD_CHANNELS)
        self.value_fc2 = nn.Linear(VALUE_HEAD_CHANNELS, 1)
    
    def forward(self, x):
        x = x.to(self.input_conv.weight.device)
        x = x.permute(0, 3, 1, 2).contiguous()
        
        out = F.relu(self.input_bn(self.input_conv(x)))
        out = self.residual_tower(out)
        
        policy = F.relu(self.policy_bn(self.policy_conv(out)))
        policy = policy.reshape(-1, POLICY_HEAD_CHANNELS * 64)
        policy = self.policy_fc(policy)
        
        value = F.relu(self.value_bn(self.value_conv(out)))
        value = value.reshape(-1, VALUE_HEAD_CHANNELS * 64)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value
    
    def evaluate(self, board_tensor, legal_mask=None):
        """Evaluate a single board position.
        
        Args:
            board_tensor: numpy array of shape (16, 8, 8)
            legal_mask: optional tensor of shape (NUM_POSSIBLE_MOVES,) with 1 for legal moves
        
        Returns:
            tuple: (policy_probs, value) where policy_probs is over legal moves,
                   value is in [-1, 1]
        """
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(board_tensor).unsqueeze(0)
            device = next(self.parameters()).device
            x = x.to(device)
            policy_logits, value = self(x)
            
            if legal_mask is not None:
                mask = legal_mask.unsqueeze(0).float()
                policy_logits = policy_logits + (1 - mask) * -1e10
                policy_logits = policy_logits.masked_fill(~mask.bool(), float('-inf'))
            
            policy_probs = F.softmax(policy_logits, dim=-1)
            
            return policy_probs.squeeze(0), value.squeeze(0).item()
