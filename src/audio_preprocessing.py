"""
audio_preprocessing.py — Audio preprocessing pipeline using Librosa.

Implements the audio feature extraction pipeline required by the assignment:
  - Load audio from WAV/MP3/WebM files
  - Compute Mel-Frequency Cepstral Coefficients (MFCCs) using Librosa
  - Normalise features per utterance
  - Pad/truncate sequences to a fixed length for batch processing
  - Build a PyTorch Dataset and DataLoader for batched training

Why MFCCs?
  MFCCs capture the spectral envelope of speech in a compact representation
  that correlates well with human perception of phonemes. They are the
  standard feature for classical ASR pipelines and are required explicitly
  by the assignment's preprocessing section.

Note: In this project Whisper handles live transcription — it operates on
  raw log-mel spectrograms internally. This module provides MFCCs for
  the data exploration / preprocessing demonstration required by the
  assignment's Section 3 (Preprocessing, 15%) and can be used to train
  a lightweight intent classifier directly on audio features if needed.

Usage:
    python src/audio_preprocessing.py           # runs demo
    from src.audio_preprocessing import extract_mfcc, AudioDataset
"""

from __future__ import annotations

import os
import sys
import numpy as np
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader

try:
    import librosa
    import librosa.display
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Default MFCC hyperparameters ─────────────────────────────────────────────
# These match values commonly used in lightweight speech processing pipelines.

DEFAULT_SAMPLE_RATE   = 16_000   # Hz — Whisper operates at 16 kHz
DEFAULT_N_MFCC        = 40       # number of MFCC coefficients (standard range 13–40)
DEFAULT_N_FFT         = 512      # FFT window length in samples (~32 ms at 16 kHz)
DEFAULT_HOP_LENGTH    = 160      # hop between windows in samples (10 ms at 16 kHz)
DEFAULT_N_MELS        = 80       # mel filterbanks used before MFCC compression
DEFAULT_MAX_LENGTH_S  = 10.0     # seconds — clips longer than this are truncated
DEFAULT_PAD_MODE      = "zero"   # 'zero' | 'edge' — how to pad short clips


# ── Core feature extraction ───────────────────────────────────────────────────

def load_audio(path: str, target_sr: int = DEFAULT_SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """
    Load an audio file and resample to target_sr.

    Librosa loads audio as a 32-bit float numpy array normalised to [-1, 1].
    Stereo signals are mixed down to mono automatically.

    Args:
        path      : File path (.wav, .mp3, .webm, .ogg, etc.)
        target_sr : Target sample rate in Hz. Default 16 kHz (Whisper standard).

    Returns:
        (waveform, sample_rate) — waveform shape is (n_samples,).

    Raises:
        ImportError : if librosa is not installed.
        FileNotFoundError : if the file does not exist.
    """
    _require_librosa()
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    return y, sr


def extract_mfcc(
    waveform: np.ndarray,
    sample_rate: int  = DEFAULT_SAMPLE_RATE,
    n_mfcc: int       = DEFAULT_N_MFCC,
    n_fft: int        = DEFAULT_N_FFT,
    hop_length: int   = DEFAULT_HOP_LENGTH,
    n_mels: int       = DEFAULT_N_MELS,
) -> np.ndarray:
    """
    Compute MFCCs from a waveform array.

    Pipeline:
        1. Short-time Fourier transform (STFT) with Hann window.
        2. Power spectrogram → Mel filterbank (n_mels bins).
        3. Log compression → log-mel spectrogram.
        4. Discrete Cosine Transform → n_mfcc coefficients per frame.

    The result has shape (n_mfcc, n_frames) where n_frames ≈ len(waveform) / hop_length.

    Args:
        waveform    : 1-D float32 array, values in [-1, 1].
        sample_rate : Hz.
        n_mfcc      : Number of cepstral coefficients to return.
        n_fft       : FFT window size in samples.
        hop_length  : Hop size in samples.
        n_mels      : Number of mel filterbank channels.

    Returns:
        2-D numpy array of shape (n_mfcc, n_frames), dtype float32.
    """
    _require_librosa()
    mfcc = librosa.feature.mfcc(
        y          = waveform.astype(np.float32),
        sr         = sample_rate,
        n_mfcc     = n_mfcc,
        n_fft      = n_fft,
        hop_length = hop_length,
        n_mels     = n_mels,
        window     = "hann",
    )
    return mfcc.astype(np.float32)   # shape: (n_mfcc, n_frames)


def normalise_mfcc(mfcc: np.ndarray) -> np.ndarray:
    """
    Per-utterance mean-variance normalisation (MVN) of MFCC features.

    Each of the n_mfcc coefficient dimensions is shifted to zero mean and
    scaled to unit variance independently. This removes recording-condition
    bias (microphone gain, room acoustics) and makes the features more
    consistent across different speakers and devices.

    Args:
        mfcc : Shape (n_mfcc, n_frames).

    Returns:
        Normalised array of the same shape.
    """
    mean = mfcc.mean(axis=1, keepdims=True)   # shape (n_mfcc, 1)
    std  = mfcc.std(axis=1,  keepdims=True) + 1e-8
    return (mfcc - mean) / std


def pad_or_truncate(
    mfcc: np.ndarray,
    max_length_s: float  = DEFAULT_MAX_LENGTH_S,
    sample_rate: int     = DEFAULT_SAMPLE_RATE,
    hop_length: int      = DEFAULT_HOP_LENGTH,
    pad_mode: str        = DEFAULT_PAD_MODE,
) -> np.ndarray:
    """
    Pad or truncate an MFCC array to a fixed number of frames.

    Fixed-length sequences are required for batch processing: PyTorch tensors
    in a batch must all have the same shape. We compute the target number of
    frames from the maximum clip duration.

    Args:
        mfcc         : Shape (n_mfcc, n_frames).
        max_length_s : Maximum clip duration in seconds.
        sample_rate  : Sample rate of the original waveform.
        hop_length   : Hop length used during MFCC extraction.
        pad_mode     : 'zero' for zero-padding, 'edge' for edge replication.

    Returns:
        Array of shape (n_mfcc, target_frames).
    """
    target_frames = int(np.ceil(max_length_s * sample_rate / hop_length))
    n_mfcc, n_frames = mfcc.shape

    if n_frames >= target_frames:
        return mfcc[:, :target_frames]

    pad_width = target_frames - n_frames
    if pad_mode == "edge":
        pad = np.repeat(mfcc[:, -1:], pad_width, axis=1)
    else:
        pad = np.zeros((n_mfcc, pad_width), dtype=mfcc.dtype)

    return np.concatenate([mfcc, pad], axis=1)


def preprocess_audio_file(
    path: str,
    n_mfcc: int          = DEFAULT_N_MFCC,
    max_length_s: float  = DEFAULT_MAX_LENGTH_S,
    normalise: bool      = True,
) -> np.ndarray:
    """
    End-to-end preprocessing for a single audio file.

    Steps: load → extract MFCCs → normalise → pad/truncate.

    Args:
        path         : Path to audio file.
        n_mfcc       : Number of MFCC coefficients.
        max_length_s : Maximum duration in seconds.
        normalise    : Apply mean-variance normalisation if True.

    Returns:
        Fixed-length MFCC array of shape (n_mfcc, target_frames).
    """
    waveform, sr = load_audio(path)
    mfcc = extract_mfcc(waveform, sample_rate=sr, n_mfcc=n_mfcc)
    if normalise:
        mfcc = normalise_mfcc(mfcc)
    mfcc = pad_or_truncate(mfcc, max_length_s=max_length_s, sample_rate=sr)
    return mfcc


# ── Dataset and DataLoader ────────────────────────────────────────────────────

class AudioDataset(Dataset):
    """
    PyTorch Dataset for campus voice query audio files.

    Expected directory structure:
        data/audio/
            find_location/
                query_001.wav
                query_002.wav
            ask_hours/
                query_001.wav
            ...

    Each subdirectory name is used as the intent label. The label is mapped
    to an integer using the order in which directories appear (sorted).

    Args:
        audio_root   : Path to top-level audio directory.
        intent_labels: List of valid intent strings (for stable label→id mapping).
                       If None, derived from directory names.
        n_mfcc       : Number of MFCC coefficients.
        max_length_s : Maximum clip length in seconds.
        normalise    : Apply MVN normalisation.
    """

    SUPPORTED_EXTS = {".wav", ".mp3", ".ogg", ".webm", ".flac"}

    def __init__(
        self,
        audio_root: str,
        intent_labels: Optional[list[str]] = None,
        n_mfcc: int = DEFAULT_N_MFCC,
        max_length_s: float = DEFAULT_MAX_LENGTH_S,
        normalise: bool = True,
    ):
        _require_librosa()
        self.audio_root   = Path(audio_root)
        self.n_mfcc       = n_mfcc
        self.max_length_s = max_length_s
        self.normalise    = normalise

        # Build label mapping
        if intent_labels:
            self.label2id = {l: i for i, l in enumerate(intent_labels)}
        else:
            dirs = sorted(p.name for p in self.audio_root.iterdir() if p.is_dir())
            self.label2id = {d: i for i, d in enumerate(dirs)}
        self.id2label = {i: l for l, i in self.label2id.items()}

        # Collect (file_path, label_id) pairs
        file_records: list[tuple[Path, int]] = []
        if self.audio_root.exists():
            for intent_dir in sorted(self.audio_root.iterdir()):
                if not intent_dir.is_dir():
                    continue
                label_id = self.label2id.get(intent_dir.name, -1)
                for f in sorted(intent_dir.iterdir()):
                    if f.suffix.lower() in self.SUPPORTED_EXTS:
                        file_records.append((f, label_id))

        # Pre-compute all MFCC tensors at init time so __getitem__ is O(1)
        # (avoids repeated disk I/O and librosa overhead during training batches)
        self.paths: list[str] = []
        self.labels: list[int] = []
        self.tensors: list[torch.Tensor] = []
        for path, label_id in file_records:
            mfcc = preprocess_audio_file(
                str(path),
                n_mfcc=self.n_mfcc,
                max_length_s=self.max_length_s,
                normalise=self.normalise,
            )
            self.tensors.append(torch.from_numpy(mfcc).unsqueeze(0))  # (1, n_mfcc, frames)
            self.labels.append(label_id)
            self.paths.append(str(path))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "features": self.tensors[idx],
            "label":    torch.tensor(self.labels[idx], dtype=torch.long),
            "path":     self.paths[idx],
        }


def build_audio_dataloaders(
    audio_root: str,
    intent_labels: Optional[list[str]] = None,
    batch_size: int = 16,
    train_fraction: float = 0.8,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train/val DataLoaders for audio files.

    Returns:
        (train_loader, val_loader)
    """
    dataset = AudioDataset(audio_root, intent_labels=intent_labels)
    n = len(dataset)
    if n == 0:
        raise ValueError(
            f"No audio files found in {audio_root}. "
            "Add .wav files under data/audio/<intent_name>/ to use this pipeline."
        )
    n_train = int(n * train_fraction)
    from torch.utils.data import Subset, random_split
    train_set, val_set = random_split(dataset, [n_train, n - n_train],
                                      generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


# ── Helper ────────────────────────────────────────────────────────────────────

def _require_librosa():
    if not LIBROSA_AVAILABLE:
        raise ImportError(
            "librosa is required for audio preprocessing. "
            "Install it with:  pip install librosa"
        )


# ── Demo ──────────────────────────────────────────────────────────────────────

def demo():
    """
    Demonstrate the MFCC pipeline with a synthetic sine-wave signal.

    Simulates a 3-second 440 Hz tone (like a single note), extracts MFCCs,
    normalises, and pads it — exactly as would happen with a real voice query.
    """
    _require_librosa()
    print("=" * 60)
    print("Audio Preprocessing Pipeline (MFCCs) — Demo")
    print("=" * 60)

    # Generate synthetic audio: 3-second 440 Hz sine wave at 16 kHz
    sr       = DEFAULT_SAMPLE_RATE
    duration = 3.0   # seconds
    t        = np.linspace(0, duration, int(sr * duration), endpoint=False)
    waveform = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    print(f"\n[Input waveform]")
    print(f"  Sample rate   : {sr} Hz")
    print(f"  Duration      : {duration:.1f} s  ({len(waveform):,} samples)")
    print(f"  Amplitude     : {waveform.min():.4f} to {waveform.max():.4f}")

    # Step 1: Extract MFCCs
    mfcc_raw = extract_mfcc(waveform, sample_rate=sr)
    print(f"\n[Step 1: MFCC extraction]")
    print(f"  Shape  : {mfcc_raw.shape}  (n_mfcc × n_frames)")
    print(f"  n_mfcc : {DEFAULT_N_MFCC}")
    print(f"  n_fft  : {DEFAULT_N_FFT}  ({DEFAULT_N_FFT/sr*1000:.1f} ms window)")
    print(f"  hop    : {DEFAULT_HOP_LENGTH}  ({DEFAULT_HOP_LENGTH/sr*1000:.1f} ms step)")
    print(f"  Value range: {mfcc_raw.min():.2f} to {mfcc_raw.max():.2f}")

    # Step 2: Normalise
    mfcc_norm = normalise_mfcc(mfcc_raw)
    print(f"\n[Step 2: Mean-Variance Normalisation]")
    print(f"  After MVN: mean ≈ {mfcc_norm.mean():.4f}, std ≈ {mfcc_norm.std():.4f}")

    # Step 3: Pad/truncate
    mfcc_fixed = pad_or_truncate(mfcc_norm, max_length_s=DEFAULT_MAX_LENGTH_S, sample_rate=sr)
    target_frames = int(np.ceil(DEFAULT_MAX_LENGTH_S * sr / DEFAULT_HOP_LENGTH))
    print(f"\n[Step 3: Pad / Truncate to fixed length]")
    print(f"  Max duration   : {DEFAULT_MAX_LENGTH_S} s → {target_frames} frames")
    print(f"  Input  shape   : {mfcc_norm.shape}")
    print(f"  Output shape   : {mfcc_fixed.shape}")
    action = "truncated" if mfcc_raw.shape[1] >= target_frames else "zero-padded"
    print(f"  Action         : {action}")

    # Step 4: Tensor conversion
    tensor = torch.from_numpy(mfcc_fixed).unsqueeze(0)  # (1, n_mfcc, frames)
    print(f"\n[Step 4: Tensor conversion]")
    print(f"  Tensor shape : {tensor.shape}  (C × n_mfcc × frames) — ready for CNN/RNN")
    print(f"  dtype        : {tensor.dtype}")
    print(f"  Memory       : {tensor.numel() * 4 / 1024:.1f} KB")

    # Simulate a batch
    batch = tensor.unsqueeze(0).repeat(8, 1, 1, 1)  # N=8
    print(f"\n[Batch simulation (N=8)]")
    print(f"  Batch shape  : {batch.shape}  (N × C × n_mfcc × frames)")

    print("\nDemo complete.")
    print("For real audio, call preprocess_audio_file('path/to/query.wav').")
    print("Supported formats: WAV, MP3, OGG, WebM, FLAC (via librosa/soundfile).")


if __name__ == "__main__":
    demo()
