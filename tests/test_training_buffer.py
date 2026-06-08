"""Tests for TrainingBuffer — pure Python collections, no GPU needed."""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.trainer import TrainingBuffer


class TestTrainingBuffer:
    def setup_method(self):
        self.buffer = TrainingBuffer(max_size=100)

    def _make_example(self, value=1.0):
        """Create a dummy training example."""
        board_tensor = np.random.randn(8, 8, 16).astype(np.float32)
        policy = np.zeros(4096, dtype=np.float32)
        policy[0] = 1.0  # One-hot for simplicity
        return {
            "board_tensor": board_tensor,
            "policy": policy,
            "value": value,
        }

    def test_initial_state(self):
        assert len(self.buffer.buffer) == 0

    def test_add_single_example(self):
        ex = self._make_example()
        self.buffer.add([ex])
        assert len(self.buffer.buffer) == 1

    def test_add_batch(self):
        examples = [self._make_example() for _ in range(10)]
        self.buffer.add(examples)
        assert len(self.buffer.buffer) == 10

    def test_add_multiple_batches(self):
        self.buffer.add([self._make_example() for _ in range(10)])
        self.buffer.add([self._make_example() for _ in range(20)])
        assert len(self.buffer.buffer) == 30

    def test_max_size_enforced(self):
        self.buffer.add([self._make_example() for _ in range(110)])
        assert len(self.buffer.buffer) == 100

    def test_oldest_entries_dropped(self):
        """When max_size is exceeded, oldest entries are dropped (deque behavior)."""
        first = self._make_example(value=1.0)
        self.buffer.add([first])
        # Fill up to max
        self.buffer.add([self._make_example(value=0.5) for _ in range(99)])
        # Add one more — should push out the first
        self.buffer.add([self._make_example(value=-1.0)])
        assert len(self.buffer.buffer) == 100
        # First example had value=1.0, remaining have 0.5 or -1.0
        values = [ex["value"] for ex in self.buffer.buffer]
        assert 1.0 not in values

    def test_sample_returns_correct_batch_size(self):
        examples = [self._make_example() for _ in range(64)]
        self.buffer.add(examples)
        batch = self.buffer.sample(64)
        assert batch is not None
        tensors, policies, values = batch
        assert tensors.shape[0] == 64
        assert policies.shape[0] == 64
        assert len(values) == 64

    def test_sample_returns_none_when_empty(self):
        batch = self.buffer.sample(64)
        assert batch is None

    def test_sample_adaptive_returns_partial_batch(self):
        """When buffer < batch_size, sample returns whatever is available."""
        examples = [self._make_example() for _ in range(10)]
        self.buffer.add(examples)
        batch = self.buffer.sample(64)
        # Should NOT return None anymore — returns partial batch
        assert batch is not None
        tensors, policies, values = batch
        assert tensors.shape[0] == 10  # returns what's available

    def test_sample_respects_batch_size_when_sufficient(self):
        examples = [self._make_example() for _ in range(64)]
        self.buffer.add(examples)
        batch = self.buffer.sample(64)
        tensors, policies, values = batch
        assert tensors.shape[0] == 64

    def test_sample_values_correct_for_win(self):
        examples = [self._make_example(value=1.0) for _ in range(64)]
        self.buffer.add(examples)
        batch = self.buffer.sample(64)
        tensors, policies, values = batch
        assert all(v == 1.0 for v in values.tolist())

    def test_sample_values_correct_for_loss(self):
        examples = [self._make_example(value=-1.0) for _ in range(64)]
        self.buffer.add(examples)
        batch = self.buffer.sample(64)
        _, _, values = batch
        assert all(v == -1.0 for v in values.tolist())

    def test_clear(self):
        self.buffer.add([self._make_example() for _ in range(50)])
        self.buffer.clear()
        assert len(self.buffer.buffer) == 0

    def test_sample_smaller_than_buffer(self):
        examples = [self._make_example() for _ in range(100)]
        self.buffer.add(examples)
        batch = self.buffer.sample(32)
        tensors, policies, values = batch
        assert tensors.shape[0] == 32

    def test_default_max_size(self):
        default_buffer = TrainingBuffer()
        assert default_buffer.buffer.maxlen == 100000
