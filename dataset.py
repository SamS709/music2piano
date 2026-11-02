"""Dataset, collate function and train/validation split for Pop2Piano training.

Training segments are sampled **inside the good regions** written by
``cleaner.py`` (``<song>.clean.json``). Audio is read from the original song, so
each segment gets its full ``context_seconds`` of neighbouring audio, exactly as
at inference (only the very start/end of a song is truncated).

A segment is ``n_bars * 4`` beat steps. Besides its audio, an item carries the
segment's position inside the audio window and its beat grid, which the model
uses for the beat-position embedding.

A segment *owns* the notes whose onset lies inside it; their durations are not
clipped at the segment border (see ``midi_tokenizer.py``).

At start-up every segment is tokenized once. Segments whose target is longer
than ``max_target_length`` are dropped (cutting them would remove the EOS and
the end of the music), and the share of discarded segments is printed.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import librosa
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

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
        min_region_score=0.0,
    ):
        """
        stride_fraction  : segment stride as a fraction of the segment length.
                           1.0 = disjoint segments, 0.5 = 50 % overlap (more
                           varied beat phases; use it for training only).
        min_segment_notes: segments with fewer note onsets are skipped.
        min_region_score : skip regions whose mean cleaner score is lower.

        ``sample_weights`` (one value per segment, the score of its region) can
        be given to a ``WeightedRandomSampler`` to favour well-aligned regions.
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

        self.songs = []  # dicts: song_path, beatstep
        self.samples = []  # (song index, start step)
        self.targets = []  # token array of each kept segment
        self.sample_weights = []

        n_no_manifest = n_regions = n_discarded = 0
        token_lengths = []  # of every tokenized segment, kept or not
        for meta_file in meta_files:
            meta_file = Path(meta_file)
            source = OmegaConf.load(meta_file)
            song_id = str(source.song.ytid)
            source_dir = meta_file.parent / str(source.piano.ytid)
            song_path = source_dir / f"{song_id}.pitchshift.wav"
            beatstep_path = source_dir / f"{song_id}.beatstep.npy"
            notes_path = source_dir / f"{song_id}.notes.npy"
            manifest_path = source_dir / f"{song_id}.clean.json"

            if not manifest_path.exists():
                n_no_manifest += 1
                continue
            if not all(p.exists() for p in (song_path, beatstep_path, notes_path)):
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            regions = [
                r
                for r in manifest.get("regions", [])
                if float(r.get("score", 1.0)) >= min_region_score
            ]
            if not regions:
                continue
            beatstep = np.load(beatstep_path)
            notes = np.load(notes_path)
            if notes.ndim != 2 or notes.shape[0] == 0:
                continue

            song_index = len(self.songs)
            self.songs.append({"song_path": str(song_path), "beatstep": beatstep})
            note_starts = np.sort(notes[:, 0])
            for region in regions:
                n_regions += 1
                first = int(region["start_step"])
                last = min(int(region["end_step"]), len(beatstep) - 1)
                # Segments lie entirely inside the region.
                starts = np.arange(first, last - self.steps_per_segment + 1, self.stride)
                if starts.size == 0:
                    continue
                ends = starts + self.steps_per_segment
                # note onsets inside [start, end)
                n_notes = np.searchsorted(note_starts, ends, side="left") - np.searchsorted(
                    note_starts, starts, side="left"
                )
                weight = max(float(region.get("score", 1.0)), 1e-3)
                for start in starts[n_notes >= min_segment_notes]:
                    start = int(start)
                    tokens = self.tokenizer.segment_to_tokens(
                        notes, start, start + self.steps_per_segment, add_eos=True
                    )
                    token_lengths.append(len(tokens))
                    if len(tokens) > self.max_target_length:
                        n_discarded += 1
                        continue
                    self.samples.append((song_index, start))
                    self.targets.append(np.asarray(tokens, dtype=np.int32))
                    self.sample_weights.append(weight)

        n_total = len(token_lengths)
        if n_total == 0:
            raise ValueError(
                "No training segments found. Run cleaner.py first (it writes a "
                f"<song>.clean.json per song); {n_no_manifest} songs had no manifest."
            )
        if not self.samples:
            raise ValueError(
                f"All {n_total} segments have more than max_target_length="
                f"{self.max_target_length} tokens. Raise dataset.target_length or "
                "use a smaller n_bars."
            )
        p50, p90, p99 = np.percentile(token_lengths, [50, 90, 99])
        print(
            f"Pop2PianoDataset: {len(self.songs)} songs, {n_regions} regions, "
            f"{len(self.samples)} segments kept ({n_no_manifest} songs without manifest)\n"
            f"  discarded {n_discarded}/{n_total} segments "
            f"({100.0 * n_discarded / n_total:.2f} %) with more than "
            f"{self.max_target_length} target tokens\n"
            f"  target length: p50 {p50:.0f} / p90 {p90:.0f} / p99 {p99:.0f} / "
            f"max {max(token_lengths)}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        song_index, start = self.samples[index]
        song = self.songs[song_index]
        beatstep = song["beatstep"]

        end = start + self.steps_per_segment
        start_time = float(beatstep[start])
        end_time = float(beatstep[end])

        # Segment plus context, read from the full song. Truncated only at the
        # song start (offset clamped to 0) and end (librosa returns what exists).
        win_start = max(0.0, start_time - self.context_seconds)
        win_end = end_time + self.context_seconds
        audio, _ = librosa.load(
            song["song_path"],
            sr=self.sample_rate,
            mono=True,
            offset=win_start,
            duration=win_end - win_start,
        )
        if audio.size == 0:
            audio = np.zeros(int(0.1 * self.sample_rate), dtype=np.float32)

        beat_grid = beatstep[start : end + 1] - start_time
        return {
            "audio": torch.from_numpy(audio.astype(np.float32)),
            "offset": float(start_time - win_start),
            "duration": float(end_time - start_time),
            "beat_grid": torch.from_numpy(beat_grid.astype(np.float32)),
            "labels": torch.tensor(self.targets[index], dtype=torch.long),
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
    """Identifier of the song a sample was made from (``song.ytid``)."""
    meta = OmegaConf.load(meta_file)
    evaluation = meta.get("eval")
    if evaluation is not None and evaluation.get("source_song_ytid") is not None:
        return str(evaluation.source_song_ytid)  # chunks from the old cleaner
    return str(meta.song.ytid)


def split_by_song(meta_files, validation_fraction, seed):
    """Split files so that all samples of one song land on the same side.

    Different piano covers of the same song share its audio, so splitting by
    file would leak training data into validation.
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