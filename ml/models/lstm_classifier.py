"""
LSTM/GRU temporal behaviour classifier.

Architecture:
  Input:  (batch, seq_len, input_size)
            where seq_len = frame_sample_rate × window_seconds (e.g. 8fps × 10s = 80 frames)
            and   input_size = 51 (17 keypoints × 3: x, y, confidence)

  Encoder: 2-layer bidirectional GRU → captures both past and future context within window
  Head:    Attention pooling → dense → sigmoid output

  Output: scalar confidence score in [0, 1] per behaviour class
          (multi-label: one score per MalpracticeClass)

Design choices:
  - Bidirectional GRU: faster than LSTM, comparable accuracy on short sequences
  - Attention pooling: learns which frames within the 10s window are most discriminative
  - Dropout (0.3): regularisation for limited training data scenarios
  - Sigmoid output: independent confidence per class (not softmax) — allows multi-malpractice
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Number of output behaviour classes (matches MalpracticeClass enum)
NUM_CLASSES = 8
# COCO full keypoint vector size
INPUT_SIZE = 51   # 17 kp × 3 (x, y, conf)
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.3


class AttentionPooling(nn.Module):
    """
    Learns a scalar weight for each time step, then computes a weighted sum.
    Allows the model to attend to frames where anomalous motion occurs.
    """
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.attention = nn.Linear(hidden_size * 2, 1)  # ×2 for bidirectional

    def forward(self, gru_out: torch.Tensor) -> torch.Tensor:
        # gru_out: (batch, seq_len, hidden*2)
        scores = self.attention(gru_out)        # (batch, seq_len, 1)
        weights = torch.softmax(scores, dim=1)  # (batch, seq_len, 1)
        pooled = (weights * gru_out).sum(dim=1)  # (batch, hidden*2)
        return pooled


class MalpracticeLSTM(nn.Module):
    """
    Bidirectional GRU + Attention classifier for temporal malpractice behaviour.
    """
    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        num_classes: int = NUM_CLASSES,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()

        self.input_norm = nn.LayerNorm(input_size)

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.attention = AttentionPooling(hidden_size)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size) keypoint sequence tensor.
        Returns:
            (batch, num_classes) confidence scores in [0, 1].
        """
        x = self.input_norm(x)
        gru_out, _ = self.gru(x)           # (batch, seq_len, hidden*2)
        pooled = self.attention(gru_out)   # (batch, hidden*2)
        return self.classifier(pooled)     # (batch, num_classes)


class LSTMClassifier:
    """
    Inference wrapper around MalpracticeLSTM.
    Manages:
      - Per-track sliding window of keypoint sequences
      - Model loading from .pt checkpoint
      - Batch inference for multiple tracked persons
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        sequence_length: int = 80,   # 8fps × 10s
        num_classes: int = NUM_CLASSES,
    ) -> None:
        self.model_path = model_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.sequence_length = sequence_length
        self.num_classes = num_classes

        # Per-track sliding window: track_id → deque of keypoint vectors
        from collections import deque
        self._windows: dict[int, deque] = {}

        self._model = MalpracticeLSTM(num_classes=num_classes)
        self._load_weights()
        self._model.eval()
        self._model.to(self.device)

        logger.info(
            "LSTMClassifier loaded on %s, seq_len=%d", self.device, sequence_length
        )

    def _load_weights(self) -> None:
        path = Path(self.model_path)
        if not path.exists():
            logger.warning(
                "LSTM weights not found at %s — using random weights (scaffold mode). "
                "Train the model with ml/training/train_lstm.py before deployment.",
                path,
            )
            return
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self._model.load_state_dict(state_dict)
        logger.info("LSTM weights loaded from %s", path)

    def update_window(self, track_id: int, keypoint_vector: np.ndarray) -> None:
        """
        Add a new frame's keypoint vector to the sliding window for a tracked person.

        Args:
            track_id: Unique ByteTrack ID for the person.
            keypoint_vector: shape (51,) — full 17-keypoint × 3 vector.
        """
        from collections import deque
        if track_id not in self._windows:
            self._windows[track_id] = deque(maxlen=self.sequence_length)
        self._windows[track_id].append(keypoint_vector)

    def classify(self, track_id: int) -> np.ndarray | None:
        """
        Run inference for a tracked person's current window.

        Returns:
            Array of shape (num_classes,) with confidence scores, or None
            if insufficient frames have been accumulated.
        """
        window = self._windows.get(track_id)
        if window is None or len(window) < self.sequence_length // 2:
            return None   # not enough data yet

        # Pad to sequence_length if needed
        frames = list(window)
        if len(frames) < self.sequence_length:
            pad = [np.zeros(INPUT_SIZE, dtype=np.float32)] * (self.sequence_length - len(frames))
            frames = pad + frames

        tensor = torch.tensor(
            np.stack(frames, axis=0), dtype=torch.float32
        ).unsqueeze(0).to(self.device)  # (1, seq_len, input_size)

        with torch.no_grad():
            scores = self._model(tensor)  # (1, num_classes)

        return scores.squeeze(0).cpu().numpy()

    def classify_all(self) -> dict[int, np.ndarray]:
        """Run inference for all active track windows. Returns {track_id: scores}."""
        results = {}
        for track_id in list(self._windows.keys()):
            scores = self.classify(track_id)
            if scores is not None:
                results[track_id] = scores
        return results

    def remove_track(self, track_id: int) -> None:
        """Clean up window for a track that's no longer visible."""
        self._windows.pop(track_id, None)

    def active_tracks(self) -> list[int]:
        return list(self._windows.keys())
