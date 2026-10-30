"""Dataset, collate function and train/validation split for Pop2Piano training.

Each item is one beat-aligned segment of ``n_bars * 4`` beat steps. Besides the
segment's audio, the dataset returns ``context_seconds`` of neighbouring audio
(limited to what the chunk file contains), the segment's position inside that
window, and its beat grid, which the model uses for the beat-position embedding.
"""

from __future__ import annotations

import os
import random

import librosa
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from midiaudiopair import MidiAudioPair
from midi_tokenizer import MidiTokenizer
from transformer_wrapper import get_encoder_config


class Pop2PianoDataset(Dataset):
    def __init__(
        self,
        meta_files,
        config,
        composer_value,
        n_bars=None,
        max_target_length=None,
        stride_fraction=1.0,
        min_segment_notes=1,
    ):
        """
        stride_fraction  : segment stride as a fraction of the segment length.
                           1.0 = disjoint segments, 0.5 = 50 % overlap (more
                           varied beat phases; use it for training only).
        min_segment_notes: segments with fewer notes are skipped.
        """
        self.config = config
        encoder_cfg = get_encoder_config(config)
        self.sample_rate = int(encoder_cfg.sample_rate)
        self.context_seconds = float(encoder_cfg.context_seconds)

        self.n_bars = config.dataset.n_bars if n_bars is None else n_bars
        self.steps_per_segment = self.n_bars * 4
        self.stride = max(1, int(round(self.steps_per_segment * stride_fraction)))
        self.max_target_length = (
            config.dataset.target_length
            if max_target_length is None
            else max_target_length
        )
        self.tokenizer = MidiTokenizer(config.tokenizer)
        self.composer_value = composer_value
        self.samples = []  # (MidiAudioPair, start step, beatstep array)

        for meta_file in meta_files:
            sample = MidiAudioPair(meta_file)
            if not sample.is_valid:
                continue
            # Chunks written by the cleaner carry no ``eval.kept`` flag, so a
            # missing flag means "kept". Only an explicit False is skipped.
            if sample.is_clean_chunk and not bool(
                getattr(sample.yaml.eval, "kept", True)
            ):
                continue
            required = (
                getattr(sample, "song", None),
                getattr(sample, "beatstep", None),
                getattr(sample, "notes", None),
            )
            if not all(path is not None and os.path.exists(path) for path in required):
                continue

            beatstep = np.load(sample.beatstep)
            notes = np.load(sample.notes)
            if notes.ndim != 2 or notes.shape[0] == 0:
                continue
            for start in range(0, len(beatstep) - self.steps_per_segment, self.stride):
                end = start + self.steps_per_segment
                n_notes = int(np.sum((notes[:, 0] < end) & (notes[:, 1] > start)))
                if n_notes >= min_segment_notes:
                    self.samples.append((sample, start, beatstep))

        if not self.samples:
            raise ValueError(
                "No valid training segments found. Run pop_align.py, "
                "bpm_quantize.py and the cleaner first."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample, start, beatstep = self.samples[index]
        notes = np.load(sample.notes)

        end = start + self.steps_per_segment
        start_time = float(beatstep[start])
        end_time = float(beatstep[end])

        # Segment plus context. Truncated at the file start (offset clamped to
        # 0) and at the file end (librosa returns what exists).
        win_start = max(0.0, start_time - self.context_seconds)
        win_end = end_time + self.context_seconds
        audio, _ = librosa.load(
            sample.song,
            sr=self.sample_rate,
            mono=True,
            offset=win_start,
            duration=win_end - win_start,
        )
        if audio.size == 0:
            audio = np.zeros(int(0.1 * self.sample_rate), dtype=np.float32)

        segment_notes = notes[(notes[:, 0] < end) & (notes[:, 1] > start)].copy()
        segment_notes[:, 0] = np.maximum(segment_notes[:, 0], start) - start
        segment_notes[:, 1] = np.minimum(segment_notes[:, 1], end) - start
        tokens = self.tokenizer.notes_to_relative_tokens(
            segment_notes,
            offset_idx=0,
            add_eos=True,
        )
        tokens = tokens[: self.max_target_length]

        beat_grid = beatstep[start : end + 1] - start_time
        return {
            "audio": torch.from_numpy(audio.astype(np.float32)),
            "offset": float(start_time - win_start),
            "duration": float(end_time - start_time),
            "beat_grid": torch.from_numpy(beat_grid.astype(np.float32)),
            "labels": torch.tensor(tokens, dtype=torch.long),
            "composer": torch.tensor(self.composer_value, dtype=torch.long),
        }


def collate_batch(items):
    audio = pad_sequence(
        [item["audio"] for item in items], batch_first=True, padding_value=0.0
    )
    labels = pad_sequence(
        [item["labels"] for item in items], batch_first=True, padding_value=-100
    )
    return {
        "audio": audio,
        "audio_lengths": torch.tensor(
            [item["audio"].shape[0] for item in items], dtype=torch.long
        ),
        "offsets": torch.tensor([item["offset"] for item in items], dtype=torch.float32),
        "durations": torch.tensor(
            [item["duration"] for item in items], dtype=torch.float32
        ),
        "beat_grids": torch.stack([item["beat_grid"] for item in items]),
        "labels": labels,
        "composer": torch.stack([item["composer"] for item in items]),
    }


def song_group_key(meta_file) -> str:
    """Identifier of the *source song* a sample was made from.

    Cleaner chunks point to their source through ``eval.source_song_ytid``;
    plain samples use ``song.ytid``.
    """
    meta = OmegaConf.load(meta_file)
    evaluation = meta.get("eval")
    if evaluation is not None and evaluation.get("source_song_ytid") is not None:
        return str(evaluation.source_song_ytid)
    return str(meta.song.ytid)


def split_by_song(meta_files, validation_fraction, seed):
    """Split files so that all chunks of one song land on the same side.

    Chunks of the same song overlap in content (and share the same audio), so
    splitting by file would leak training data into validation.
    """
    groups = {}
    for meta_file in meta_files:
        groups.setdefault(song_group_key(meta_file), []).append(meta_file)

    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n_val = max(1, int(round(len(keys) * validation_fraction))) if len(keys) > 1 else 0
    val_keys = set(keys[:n_val])

    train_files = [f for k in keys if k not in val_keys for f in groups[k]]
    val_files = [f for k in keys if k in val_keys for f in groups[k]]
    if not val_files:  # single-song dataset: fall back so validation can run
        val_files = train_files[:1]
    return train_files, val_files