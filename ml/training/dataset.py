"""
PyTorch Dataset for malpractice keypoint sequences.

Expected directory layout:
    data_dir/
        sequences/
            <sample_id>.npy          # shape: (any_length, 51) float32
        labels/
            <sample_id>.json         # {"labels": [0,1,0,...], "class_name": "USING_MOBILE_PHONE"}
        manifest.csv                 # sample_id, split (optional, for pre-defined splits)

Augmentation:
  - Random horizontal flip (mirrors keypoints across x=0.5)
  - Random temporal crop/shift within sequence
  - Gaussian noise on keypoint positions (σ=0.01)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

NUM_CLASSES = 8
INPUT_SIZE = 51   # 17 × 3


class MalpracticeDataset(Dataset):
    """
    Loads pre-extracted keypoint sequences and their multi-label targets.
    """

    def __init__(
        self,
        data_dir: str,
        sequence_length: int = 80,
        augment: bool = True,
        gaussian_noise_std: float = 0.01,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.seq_len = sequence_length
        self.augment = augment
        self.noise_std = gaussian_noise_std

        self.seq_dir = self.data_dir / "sequences"
        self.lbl_dir = self.data_dir / "labels"

        self.samples: List[Tuple[Path, Path]] = []
        self._discover_samples()

    def _discover_samples(self) -> None:
        if not self.seq_dir.exists():
            raise FileNotFoundError(f"Sequences directory not found: {self.seq_dir}")
        if not self.lbl_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {self.lbl_dir}")

        for seq_file in sorted(self.seq_dir.glob("*.npy")):
            lbl_file = self.lbl_dir / (seq_file.stem + ".json")
            if lbl_file.exists():
                self.samples.append((seq_file, lbl_file))
            else:
                logger.warning("No label file for %s — skipping", seq_file.name)

        logger.info("Discovered %d labelled sequences in %s", len(self.samples), self.data_dir)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_path, lbl_path = self.samples[idx]

        # Load sequence (T, 51)
        seq = np.load(str(seq_path)).astype(np.float32)

        # Load label
        with open(lbl_path) as f:
            label_data = json.load(f)
        labels = np.array(label_data["labels"], dtype=np.float32)
        assert len(labels) == NUM_CLASSES, f"Expected {NUM_CLASSES} labels, got {len(labels)}"

        # Resize to fixed sequence length
        seq = self._resize_sequence(seq, self.seq_len)

        # Augmentation
        if self.augment:
            seq = self._augment(seq, labels)

        return torch.from_numpy(seq), torch.from_numpy(labels)

    def _resize_sequence(self, seq: np.ndarray, target_len: int) -> np.ndarray:
        """Pad with zeros or truncate to target_len."""
        T, D = seq.shape
        if T == target_len:
            return seq
        elif T > target_len:
            # Random crop during training for augmentation
            start = np.random.randint(0, T - target_len) if self.augment else 0
            return seq[start:start + target_len]
        else:
            pad = np.zeros((target_len - T, D), dtype=np.float32)
            return np.concatenate([pad, seq], axis=0)  # left-pad

    def _augment(self, seq: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Apply random augmentations to a keypoint sequence."""
        # 1) Horizontal flip (mirror x coords)
        if np.random.random() < 0.5:
            seq = seq.copy()
            # Every 3rd column starting at 0 is x-coordinate
            seq[:, 0::3] = 1.0 - seq[:, 0::3]

        # 2) Gaussian noise on x, y positions
        if np.random.random() < 0.5:
            noise = np.random.normal(0, self.noise_std, seq.shape).astype(np.float32)
            # Only add noise to x,y, not confidence columns
            noise[:, 2::3] = 0.0
            seq = np.clip(seq + noise, 0.0, 1.0)

        # 3) Random temporal jitter (±5 frames shift)
        if np.random.random() < 0.3:
            shift = np.random.randint(-5, 6)
            if shift > 0:
                seq = np.concatenate([seq[shift:], np.zeros((shift, seq.shape[1]), dtype=np.float32)], axis=0)
            elif shift < 0:
                seq = np.concatenate([np.zeros((-shift, seq.shape[1]), dtype=np.float32), seq[:shift]], axis=0)

        return seq


def collate_fn(batch: list) -> Tuple[torch.Tensor, torch.Tensor]:
    """Default collate (all sequences same length after resize)."""
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.stack(ys)
