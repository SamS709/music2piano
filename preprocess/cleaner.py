"""Score songs, write good-region manifests, and help calibrate the thresholds.

The input must already have been processed by ``pop_align.py``,
``bpm_quantize.py`` and ``split_spleeter.py``. The original song assets stay in
place; two files are added next to them:

    <song>.scores.npz   per-beat-step metrics for the whole song (slow to compute)
    <song>.clean.json   good regions derived from those scores (fast to recompute)

Typical workflow
----------------
1. Score every song and write manifests with the default thresholds::

       python cleaner.py data --workers 4

2. Export random clips (song + rendered piano) together with their metrics::

       python cleaner.py data --regions-only --random 60

   Listen to them and fill the ``label`` column of ``random_samples/samples.csv``
   with 1 (piano and song are aligned) or 0 (misaligned / unusable).

3. See which metrics and thresholds separate your labels::

       python cleaner.py --analyze random_samples/samples.csv

4. Re-derive the regions with new thresholds, without recomputing metrics::

       python cleaner.py data --regions-only --min-chroma-similarity 0.45
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from types import SimpleNamespace

import librosa
import numpy as np
import pretty_midi
import soundfile as sf
from omegaconf import OmegaConf
from scipy.ndimage import maximum_filter1d, median_filter
from scipy.stats import rankdata

sys.path.append(str(Path(__file__).resolve().parents[1]))
from evaluate.midi_melody_accuracy import get_highest_pitches_from_piano_roll
from data_roots import dataset_roots


DEFAULT_SR = 44100
DEFAULT_HOP_LENGTH = 1024
ONSET_TOLERANCE_FRAMES = 3
CHORD_TOLERANCE_FRAMES = 3
N_NULL_SHUFFLES = 8
SCORES_VERSION = 1
MANIFEST_VERSION = 2

# Metrics stored for every beat step (each one is computed on a window centred
# on the step, see --window-seconds).
METRIC_NAMES = (
    "music_chroma_similarity",
    "music_chroma_null_similarity",
    "music_chroma_alignment_margin",
    "music_chroma_coverage",
    "activity_iou",
    "onset_f1",
    "note_density_ratio",
    "vocal_chord_support",
    "vocal_chord_support_null",
    "vocal_chord_support_margin",
    "vocal_voiced_coverage",
    "vocal_melody_pitch_accuracy",
    "vocal_melody_chroma_accuracy",
    "midi_note_count",
    "midi_onset_count",
)


@dataclass
class Thresholds:
    """Everything that decides which steps / regions are kept.

    Changing these never requires recomputing metrics (use --regions-only).
    """

    # Hard gates: a step must pass them to get a non-zero score.
    min_chroma_similarity: float = 0.35
    min_alignment_margin: float = 0.05
    min_activity_iou: float = 0.2
    min_onset_f1: float = 0.0
    min_vocal_support: float = 0.35
    min_vocal_coverage: float = 0.20
    min_vocal_margin: float = -0.05
    min_notes: int = 12
    min_note_onsets: int = 4
    require: str = "either"  # which view must pass: music | vocals | either | both
    # Region extraction (applied to the 0-1 step score).
    enter_threshold: float = 0.35
    exit_threshold: float = 0.28
    smooth_steps: int = 5
    min_region_seconds: float = 4.0
    merge_gap_steps: int = 2


THRESHOLD_HELP = {
    "min_chroma_similarity": "min song-chroma vs MIDI-chroma cosine similarity",
    "min_alignment_margin": "min similarity gain over the time-shuffled baseline",
    "min_activity_iou": "min overlap between audio-active and MIDI-active frames",
    "min_onset_f1": "min audio/MIDI onset F1 (0 = disabled)",
    "min_vocal_support": "min fraction of vocal notes contained in the MIDI chords",
    "min_vocal_coverage": "min fraction of frames with a detected vocal pitch",
    "min_vocal_margin": "min vocal support gain over the shuffled baseline",
    "min_notes": "min MIDI notes in the scoring window",
    "min_note_onsets": "min distinct MIDI onset steps in the scoring window",
    "require": "which view must pass: music, vocals, either, or both",
    "enter_threshold": "a region starts when the smoothed score reaches this",
    "exit_threshold": "a region ends when the smoothed score falls below this",
    "smooth_steps": "median-filter width (beat steps) applied to the score",
    "min_region_seconds": "drop regions shorter than this",
    "merge_gap_steps": "merge regions separated by at most this many steps",
}


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def song_paths(meta_file: Path) -> SimpleNamespace:
    source = OmegaConf.load(meta_file)
    piano_id = str(source.piano.ytid)
    song_id = str(source.song.ytid)
    source_dir = meta_file.parent / piano_id
    return SimpleNamespace(
        piano_id=piano_id,
        song_id=song_id,
        source_dir=source_dir,
        song_path=source_dir / f"{song_id}.pitchshift.wav",
        midi_path=source_dir / f"{song_id}.qmidi.mid",
        vocal_path=source_dir / f"{song_id}.vocals.mp3",
        notes_path=source_dir / f"{song_id}.notes.npy",
        beatstep_path=source_dir / f"{song_id}.beatstep.npy",
        scores_path=source_dir / f"{song_id}.scores.npz",
        manifest_path=source_dir / f"{song_id}.clean.json",
    )


# --------------------------------------------------------------------------- #
# Window metrics (all take frame indices into the song-level feature arrays)
# --------------------------------------------------------------------------- #
def _midi_chroma(piano_roll: np.ndarray) -> np.ndarray:
    padded = np.zeros((132, piano_roll.shape[1]), dtype=np.float32)
    padded[: piano_roll.shape[0]] = piano_roll
    return padded.reshape(11, 12, -1).sum(axis=0)


def _fraction_matched(a: np.ndarray, b: np.ndarray, tolerance: int) -> float:
    """Fraction of the (sorted) events in ``a`` with an event of ``b`` nearby."""
    if a.size == 0 or b.size == 0:
        return 0.0
    position = np.searchsorted(b, a)
    left = np.abs(a - b[np.clip(position - 1, 0, b.size - 1)])
    right = np.abs(a - b[np.clip(position, 0, b.size - 1)])
    return float(np.mean(np.minimum(left, right) <= tolerance))


def _music_chroma_metrics(f, start: int, end: int) -> dict[str, float]:
    """Synchronised chroma similarity against a time-shuffled null baseline."""
    zero = {
        "music_chroma_similarity": 0.0,
        "music_chroma_null_similarity": 0.0,
        "music_chroma_alignment_margin": 0.0,
        "music_chroma_coverage": 0.0,
    }
    audio = f.song_chroma[:, start:end]
    midi = f.midi_chroma[:, start:end]
    length = min(audio.shape[1], midi.shape[1])
    if length == 0:
        return zero
    audio, midi = audio[:, :length], midi[:, :length]
    audio_norm = f.audio_norm[start : start + length]
    midi_norm = f.midi_norm[start : start + length]
    audio_active = audio_norm > max(1e-6, np.percentile(audio_norm, 20))
    active = audio_active & (midi_norm > 1e-6)
    if not np.any(active):
        return zero

    idx = np.flatnonzero(active)
    dot = np.einsum("ij,ij->j", audio[:, idx], midi[:, idx])
    similarity = float(
        np.mean(np.clip(dot / (audio_norm[idx] * midi_norm[idx]), 0.0, 1.0))
    )

    rng = np.random.default_rng(0)
    perms = np.stack([rng.permutation(length) for _ in range(N_NULL_SHUFFLES)])
    shuffled = midi[:, perms[:, idx]]  # (12, shuffles, active frames)
    shuffled_norm = np.linalg.norm(shuffled, axis=0)
    shuffled_dot = np.einsum("ik,isk->sk", audio[:, idx], shuffled)
    valid = shuffled_norm > 1e-6
    denominator = np.where(valid, audio_norm[idx][None] * shuffled_norm, 1.0)
    shuffled_similarity = np.where(
        valid, np.clip(shuffled_dot / denominator, 0.0, 1.0), 0.0
    )
    counts = valid.sum(axis=1)
    keep = counts > 0
    null = (
        float(np.mean(shuffled_similarity.sum(axis=1)[keep] / counts[keep]))
        if np.any(keep)
        else 0.0
    )
    return {
        "music_chroma_similarity": similarity,
        "music_chroma_null_similarity": null,
        "music_chroma_alignment_margin": similarity - null,
        "music_chroma_coverage": float(np.mean(active)),
    }


def _vocal_support_metrics(f, start: int, end: int) -> dict[str, float]:
    """Does the active MIDI chord contain the vocal pitch class (+-3 frames)?"""
    zero = {
        "vocal_chord_support": 0.0,
        "vocal_chord_support_null": 0.0,
        "vocal_chord_support_margin": 0.0,
        "vocal_voiced_coverage": 0.0,
    }
    voiced = f.voiced[start:end]
    midi = f.midi_bool[:, start:end]
    length = min(voiced.size, midi.shape[1])
    if length == 0 or not np.any(voiced[:length]):
        return zero
    voiced, midi = voiced[:length], midi[:, :length]
    idx = np.flatnonzero(voiced)
    pitch_classes = f.f0_pc[start : start + length][idx]
    size = 2 * CHORD_TOLERANCE_FRAMES + 1

    dilated = maximum_filter1d(midi, size=size, axis=1, mode="constant")
    support = float(np.mean(dilated[pitch_classes, idx]))

    rng = np.random.default_rng(0)
    perms = np.stack([rng.permutation(length) for _ in range(N_NULL_SHUFFLES)])
    shuffled = maximum_filter1d(midi[:, perms], size=size, axis=2, mode="constant")
    null = float(
        np.mean(
            shuffled[
                pitch_classes[:, None],
                np.arange(N_NULL_SHUFFLES)[None, :],
                idx[:, None],
            ]
        )
    )
    return {
        "vocal_chord_support": support,
        "vocal_chord_support_null": null,
        "vocal_chord_support_margin": support - null,
        "vocal_voiced_coverage": float(np.mean(voiced)),
    }


def _vocal_melody_metrics(f, start: int, end: int) -> dict[str, float]:
    """Vocal pitch vs the highest MIDI pitch, frames voiced in both."""
    ref = f.f0_midi[start:end]
    est = f.midi_pitch[start:end]
    size = min(ref.size, est.size)
    ref, est = ref[:size], est[:size]
    both = np.isfinite(ref) & np.isfinite(est)
    if not np.any(both):
        return {"vocal_melody_pitch_accuracy": 0.0, "vocal_melody_chroma_accuracy": 0.0}
    error = np.abs(ref[both] - est[both])
    chroma_error = np.minimum(error % 12.0, 12.0 - error % 12.0)
    return {
        "vocal_melody_pitch_accuracy": float(np.mean(error < 0.5)),
        "vocal_melody_chroma_accuracy": float(np.mean(chroma_error < 0.5)),
    }


def _activity_metrics(f, start: int, end: int) -> dict[str, float]:
    rms = f.rms[start:end]
    midi_active = f.midi_active[start:end]
    size = min(rms.size, midi_active.size)
    if size == 0:
        return {"activity_iou": 0.0}
    rms, midi_active = rms[:size], midi_active[:size]
    audio_active = rms > max(1e-6, np.percentile(rms, 20))
    intersection = np.sum(audio_active & midi_active)
    union = np.sum(audio_active | midi_active)
    return {"activity_iou": float(intersection / union) if union else 0.0}


def _onset_metrics(f, start: int, end: int) -> dict[str, float]:
    """Audio onsets vs MIDI note onsets (from notes.npy), +-3 frames."""
    audio = f.audio_onsets[(f.audio_onsets >= start) & (f.audio_onsets < end)]
    midi = f.midi_onsets[(f.midi_onsets >= start) & (f.midi_onsets < end)]
    if audio.size == 0 and midi.size == 0:
        return {"onset_f1": 1.0, "note_density_ratio": 1.0}
    density = min(audio.size, midi.size) / max(audio.size, midi.size)
    precision = _fraction_matched(audio, midi, ONSET_TOLERANCE_FRAMES)
    recall = _fraction_matched(midi, audio, ONSET_TOLERANCE_FRAMES)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"onset_f1": float(f1), "note_density_ratio": float(density)}


# --------------------------------------------------------------------------- #
# Stage 1: per-step metrics for a whole song (slow)
# --------------------------------------------------------------------------- #
def compute_song_metrics(paths, window_seconds: float, stride: int):
    song_audio, _ = librosa.load(paths.song_path, sr=DEFAULT_SR, mono=True)
    has_vocals = paths.vocal_path.exists()
    metric_audio = (
        librosa.load(paths.vocal_path, sr=DEFAULT_SR, mono=True)[0]
        if has_vocals
        else song_audio
    )
    f0 = librosa.pyin(
        metric_audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C6"),
        sr=DEFAULT_SR,
        hop_length=DEFAULT_HOP_LENGTH,
    )[0]
    song_chroma = librosa.feature.chroma_stft(
        y=song_audio, sr=DEFAULT_SR, hop_length=DEFAULT_HOP_LENGTH
    )
    rms = librosa.feature.rms(
        y=song_audio, frame_length=2048, hop_length=DEFAULT_HOP_LENGTH
    )[0]
    pm = pretty_midi.PrettyMIDI(str(paths.midi_path))

    frame_count = min(len(f0), song_chroma.shape[1], len(rms))
    frame_times = np.arange(frame_count) * DEFAULT_HOP_LENGTH / DEFAULT_SR
    piano_roll = pm.get_piano_roll(
        fs=DEFAULT_SR / DEFAULT_HOP_LENGTH, times=frame_times
    )
    f0 = f0[:frame_count]
    midi_chroma = _midi_chroma(piano_roll)
    midi_pitch = get_highest_pitches_from_piano_roll(piano_roll).astype(float)
    midi_pitch[midi_pitch == 0] = np.nan

    beatstep = np.load(paths.beatstep_path)
    notes = np.load(paths.notes_path)
    if notes.ndim != 2:
        notes = np.zeros((0, 4))

    voiced = np.isfinite(f0)
    f0_midi = np.full(frame_count, np.nan)
    f0_midi[voiced] = librosa.hz_to_midi(f0[voiced])
    f0_pc = np.full(frame_count, -1, dtype=int)
    f0_pc[voiced] = np.rint(f0_midi[voiced]).astype(int) % 12

    audio_onsets = librosa.onset.onset_detect(
        y=song_audio, sr=DEFAULT_SR, hop_length=DEFAULT_HOP_LENGTH, backtrack=False
    )
    note_steps = np.clip(notes[:, 0].astype(int), 0, len(beatstep) - 1)
    midi_onsets = np.unique(
        np.rint(beatstep[note_steps] * DEFAULT_SR / DEFAULT_HOP_LENGTH).astype(int)
    )

    audio_chroma = song_chroma[:, :frame_count]
    features = SimpleNamespace(
        song_chroma=audio_chroma,
        audio_norm=np.linalg.norm(audio_chroma, axis=0),
        midi_chroma=midi_chroma,
        midi_norm=np.linalg.norm(midi_chroma, axis=0),
        midi_bool=(midi_chroma > 1e-6).astype(np.uint8),
        midi_active=midi_chroma.max(axis=0) > 1e-6,
        rms=rms[:frame_count],
        voiced=voiced,
        f0_pc=f0_pc,
        f0_midi=f0_midi,
        midi_pitch=midi_pitch,
        audio_onsets=audio_onsets[audio_onsets < frame_count],
        midi_onsets=midi_onsets,
    )

    duration = min(float(beatstep[-1]), len(song_audio) / DEFAULT_SR)
    step_count = max(0, int(np.searchsorted(beatstep, duration, side="right")) - 1)
    if step_count < 1:
        return None
    window_steps = max(2, int(np.searchsorted(beatstep, window_seconds, side="left")))

    half = max(1, window_steps // 2)
    starts = np.arange(0, step_count, stride)
    out = {name: np.zeros(len(starts), dtype=np.float32) for name in METRIC_NAMES}
    for k, step in enumerate(starts):
        metric_start = max(0, step - half)
        metric_end = min(step_count, step + half)
        if metric_end <= metric_start:
            continue
        start_frame = int(
            np.searchsorted(frame_times, beatstep[metric_start], side="left")
        )
        end_frame = int(np.searchsorted(frame_times, beatstep[metric_end], side="left"))
        local = {}
        local.update(_music_chroma_metrics(features, start_frame, end_frame))
        local.update(_vocal_support_metrics(features, start_frame, end_frame))
        local.update(_vocal_melody_metrics(features, start_frame, end_frame))
        local.update(_activity_metrics(features, start_frame, end_frame))
        local.update(_onset_metrics(features, start_frame, end_frame))
        overlap = (notes[:, 0] < metric_end) & (notes[:, 1] > metric_start)
        local["midi_note_count"] = int(overlap.sum())
        local["midi_onset_count"] = int(np.unique(notes[overlap, 0].astype(int)).size)
        for name in METRIC_NAMES:
            out[name][k] = local[name]

    expand = np.minimum(np.arange(step_count) // stride, len(starts) - 1)
    metrics = {name: out[name][expand] for name in METRIC_NAMES}
    meta = {
        "version": SCORES_VERSION,
        "step_count": int(step_count),
        "window_seconds": float(window_seconds),
        "score_stride": int(stride),
        "has_vocals": bool(has_vocals),
    }
    return metrics, meta


def save_scores(path: Path, metrics: dict, meta: dict) -> None:
    np.savez_compressed(
        path,
        _meta=np.array(json.dumps(meta)),
        **{name: metrics[name].astype(np.float32) for name in METRIC_NAMES},
    )


def load_scores(path: Path):
    with np.load(path) as data:
        meta = json.loads(str(data["_meta"]))
        metrics = {name: data[name] for name in METRIC_NAMES}
    return metrics, meta


# --------------------------------------------------------------------------- #
# Stage 2: thresholds -> step scores -> regions (fast)
# --------------------------------------------------------------------------- #
def _unit(x):
    return np.clip(x, 0.0, 1.0)


def step_scores(m: dict, th: Thresholds):
    """Return (score in [0, 1], gate-passed mask) for every beat step."""
    margin = m["music_chroma_alignment_margin"]
    music_score = np.mean(
        [
            _unit(m["music_chroma_similarity"]),
            _unit(m["music_chroma_coverage"]),
            _unit(m["activity_iou"]),
            _unit(m["onset_f1"]),
            _unit(margin / max(th.min_alignment_margin, 1e-6)),
        ],
        axis=0,
    )
    if th.min_vocal_margin > 0:
        vocal_margin_term = _unit(m["vocal_chord_support_margin"] / th.min_vocal_margin)
    else:
        vocal_margin_term = _unit(m["vocal_chord_support"])
    vocal_score = np.mean(
        [
            _unit(m["vocal_chord_support"]),
            _unit(m["vocal_voiced_coverage"]),
            vocal_margin_term,
        ],
        axis=0,
    )

    music_ok = (
        (m["music_chroma_similarity"] >= th.min_chroma_similarity)
        & (margin >= th.min_alignment_margin)
        & (m["activity_iou"] >= th.min_activity_iou)
        & (m["onset_f1"] >= th.min_onset_f1)
    )
    vocal_ok = (
        (m["vocal_chord_support"] >= th.min_vocal_support)
        & (m["vocal_voiced_coverage"] >= th.min_vocal_coverage)
        & (m["vocal_chord_support_margin"] >= th.min_vocal_margin)
    )
    notes_ok = (m["midi_note_count"] >= th.min_notes) & (
        m["midi_onset_count"] >= th.min_note_onsets
    )

    if th.require == "music":
        ok, score = music_ok, music_score
    elif th.require == "vocals":
        ok, score = vocal_ok, vocal_score
    elif th.require == "both":
        ok, score = music_ok & vocal_ok, np.minimum(music_score, vocal_score)
    else:  # either
        ok = music_ok | vocal_ok
        score = np.maximum(
            np.where(music_ok, music_score, 0.0), np.where(vocal_ok, vocal_score, 0.0)
        )
    ok = ok & notes_ok
    return np.where(ok, score, 0.0).astype(np.float32), ok


def extract_regions(score: np.ndarray, beatstep: np.ndarray, th: Thresholds):
    """Hysteresis regions ``(start_step, end_step)`` (end exclusive).

    ``beatstep`` must have ``len(score) + 1`` entries.
    """
    width = max(1, int(th.smooth_steps))
    width += width % 2 == 0
    smoothed = median_filter(score, size=width, mode="nearest")

    raw, start = [], None
    for index, value in enumerate(smoothed):
        if start is None:
            if value >= th.enter_threshold:
                start = index
        elif value < th.exit_threshold:
            raw.append((start, index))
            start = None
    if start is not None:
        raw.append((start, len(smoothed)))

    merged = []
    for begin, end in raw:
        if merged and begin - merged[-1][1] <= th.merge_gap_steps:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((begin, end))
    return [
        (begin, end)
        for begin, end in merged
        if beatstep[end] - beatstep[begin] >= th.min_region_seconds
    ]


# --------------------------------------------------------------------------- #
# Per-song driver
# --------------------------------------------------------------------------- #
def _process_song(meta_file, th, window_seconds, stride, overwrite, regions_only):
    paths = song_paths(meta_file)
    needed = (
        (paths.beatstep_path,)
        if regions_only
        else (paths.song_path, paths.midi_path, paths.notes_path, paths.beatstep_path)
    )
    if not all(path.exists() for path in needed):
        print(f"skip {meta_file}: missing preprocessed file")
        return None

    metrics = meta = None
    computed = False
    if paths.scores_path.exists() and not overwrite:
        metrics, meta = load_scores(paths.scores_path)
        if not regions_only and (
            meta["window_seconds"] != window_seconds or meta["score_stride"] != stride
        ):
            metrics = None  # scored with other settings: recompute
    if metrics is None:
        if regions_only:
            print(f"skip {meta_file}: no scores yet (run without --regions-only)")
            return None
        result = compute_song_metrics(paths, window_seconds, stride)
        if result is None:
            print(f"skip {meta_file}: no usable beat steps")
            return None
        metrics, meta = result
        save_scores(paths.scores_path, metrics, meta)
        computed = True

    beatstep = np.load(paths.beatstep_path)
    step_count = meta["step_count"]
    score, _ = step_scores(metrics, th)
    regions = extract_regions(score, beatstep[: step_count + 1], th)

    manifest = {
        "version": MANIFEST_VERSION,
        "source_metadata": meta_file.name,
        "song_id": paths.song_id,
        "has_vocals": meta["has_vocals"],
        "score_step_count": step_count,
        "window_seconds": meta["window_seconds"],
        "thresholds": asdict(th),
        "regions": [
            {
                "start_step": int(begin),
                "end_step": int(end),
                "start_seconds": float(beatstep[begin]),
                "end_seconds": float(beatstep[end]),
                "score": float(np.mean(score[begin:end])),
            }
            for begin, end in regions
        ],
    }
    paths.manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "song": paths.song_id,
        "regions": len(regions),
        "good_seconds": float(sum(beatstep[e] - beatstep[b] for b, e in regions)),
        "total_seconds": float(beatstep[step_count] - beatstep[0]),
        "computed": computed,
    }


def process_song(job):
    meta_file, th_dict, window_seconds, stride, overwrite, regions_only = job
    try:
        return _process_song(
            Path(meta_file),
            Thresholds(**th_dict),
            window_seconds,
            stride,
            overwrite,
            regions_only,
        )
    except Exception as error:  # keep going: one bad song must not stop a batch
        traceback.print_exc()
        print(f"skip {meta_file}: {type(error).__name__}: {error}")
        return None


def run_pipeline(roots, th: Thresholds, args) -> None:
    jobs = [
        (
            meta_file,
            asdict(th),
            args.window_seconds,
            args.score_stride,
            args.overwrite,
            args.regions_only,
        )
        for root in roots
        for meta_file in sorted(root.glob("*.yaml"))
    ]
    if args.workers > 1:
        pool = ProcessPoolExecutor(max_workers=args.workers)
        results = pool.map(process_song, jobs)
    else:
        pool = None
        results = map(process_song, jobs)

    done = []
    for k, result in enumerate(results, 1):
        if result is None:
            continue
        done.append(result)
        print(
            f"[{k}/{len(jobs)}] {result['song']}: {result['regions']} regions, "
            f"{result['good_seconds']:.0f}/{result['total_seconds']:.0f} s kept"
            + (" (scored)" if result["computed"] else "")
        )
    if pool is not None:
        pool.shutdown()

    good = sum(r["good_seconds"] for r in done)
    total = sum(r["total_seconds"] for r in done)
    print(
        f"{len(done)}/{len(jobs)} songs processed | "
        f"{sum(r['regions'] for r in done)} regions | "
        f"{good / 60:.1f} of {total / 60:.1f} minutes kept "
        f"({100 * good / max(total, 1e-9):.0f} %)"
    )


# --------------------------------------------------------------------------- #
# Random clips for listening tests
# --------------------------------------------------------------------------- #
def _clip_midi(pm: pretty_midi.PrettyMIDI, start: float, end: float):
    """Return a MIDI copy clipped to [start, end), with time reset to zero."""
    clipped = copy.deepcopy(pm)
    for instrument in clipped.instruments:
        notes = []
        for note in instrument.notes:
            if note.end <= start or note.start >= end:
                continue
            note.start = max(note.start, start) - start
            note.end = min(note.end, end) - start
            if note.end > note.start:
                notes.append(note)
        instrument.notes = notes
        instrument.control_changes = [
            event for event in instrument.control_changes if start <= event.time < end
        ]
        for event in instrument.control_changes:
            event.time -= start
    return clipped


def export_random(roots, count, mode, seconds, out_dir: Path, th: Thresholds, seed):
    from utils.dsp import get_stereo

    songs = []
    for root in roots:
        for meta_file in sorted(root.glob("*.yaml")):
            paths = song_paths(meta_file)
            if not (paths.scores_path.exists() and paths.beatstep_path.exists()):
                continue
            metrics, meta = load_scores(paths.scores_path)
            beatstep = np.load(paths.beatstep_path)
            step_count = meta["step_count"]
            score, ok = step_scores(metrics, th)
            in_region = np.zeros(step_count, dtype=bool)
            for begin, end in extract_regions(score, beatstep[: step_count + 1], th):
                in_region[begin:end] = True
            window = max(2, int(np.searchsorted(beatstep, seconds, side="left")))
            if step_count - window + 1 < 1:
                continue
            songs.append(
                {
                    "paths": paths,
                    "metrics": metrics,
                    "beatstep": beatstep,
                    "score": score,
                    "ok": ok,
                    "in_region": in_region,
                    "window": window,
                    "n_starts": step_count - window + 1,
                }
            )
    if not songs:
        print("no scored songs found: run the cleaner without --regions-only first")
        return

    rng = np.random.default_rng(seed)
    weights = np.array([s["n_starts"] for s in songs], dtype=float)
    weights /= weights.sum()
    chosen, seen = [], set()
    for _ in range(count * 500):
        if len(chosen) >= count:
            break
        index = int(rng.choice(len(songs), p=weights))
        song = songs[index]
        start = int(rng.integers(song["n_starts"]))
        key = (index, start // song["window"])
        if key in seen:
            continue
        fraction = float(song["in_region"][start : start + song["window"]].mean())
        if (
            (mode == "kept" and fraction < 0.999)
            or (mode == "rejected" and fraction > 0.0)
            or (mode == "mixed" and not 0.0 < fraction < 1.0)
        ):
            continue
        seen.add(key)
        chosen.append((song, start, fraction))
    if len(chosen) < count:
        print(f"only found {len(chosen)} windows matching --random-from {mode}")

    out_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "file", "song_id", "start_step", "end_step", "start_seconds", "end_seconds",
        "in_region_fraction", "kept", "score_mean", "passed_fraction",
        *METRIC_NAMES, "label", "comment",
    ]
    rows = []
    for k, (song, start, fraction) in enumerate(chosen):
        paths, beatstep, window = song["paths"], song["beatstep"], song["window"]
        end = start + window
        t0, t1 = float(beatstep[start]), float(beatstep[end])
        audio, _ = librosa.load(
            paths.song_path, sr=DEFAULT_SR, mono=True, offset=t0, duration=t1 - t0
        )
        clipped = _clip_midi(pretty_midi.PrettyMIDI(str(paths.midi_path)), t0, t1)
        mix = get_stereo(audio, clipped.fluidsynth(DEFAULT_SR), pop_scale=0.4)
        mix = np.asarray(np.asarray(mix).T, dtype=np.float32)
        name = f"{k:03d}__{paths.song_id}__{start}-{end}.wav"
        sf.write(str(out_dir / name), mix, DEFAULT_SR)

        row = {
            "file": name,
            "song_id": paths.song_id,
            "start_step": start,
            "end_step": end,
            "start_seconds": round(t0, 3),
            "end_seconds": round(t1, 3),
            "in_region_fraction": round(fraction, 3),
            "kept": fraction >= 0.999,
            "score_mean": round(float(song["score"][start:end].mean()), 4),
            "passed_fraction": round(float(song["ok"][start:end].mean()), 4),
            "label": "",
            "comment": "",
        }
        for metric in METRIC_NAMES:
            row[metric] = round(float(song["metrics"][metric][start:end].mean()), 4)
        rows.append(row)

    with open(out_dir / "samples.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"wrote {len(rows)} clips and {out_dir / 'samples.csv'}\n"
        "Listen, fill the 'label' column (1 = aligned, 0 = misaligned), then run:\n"
        f"  python cleaner.py --analyze {out_dir / 'samples.csv'}"
    )


# --------------------------------------------------------------------------- #
# Threshold analysis from listening labels
# --------------------------------------------------------------------------- #
def _auc(x: np.ndarray, y: np.ndarray) -> float:
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    ranks = rankdata(x)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _best_threshold(x: np.ndarray, y: np.ndarray):
    """Threshold maximising TPR - FPR for the rule ``x >= t`` -> good."""
    best = (-2.0, None, 0.0, 0.0)
    for t in np.unique(x):
        pred = x >= t
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        fn = int(np.sum(~pred & (y == 1)))
        tn = int(np.sum(~pred & (y == 0)))
        tpr = tp / (tp + fn) if tp + fn else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        if tpr - fpr > best[0]:
            best = (tpr - fpr, float(t), tp / (tp + fp) if tp + fp else 0.0, tpr)
    return best[1], best[2], best[3]


def analyze(csv_path: Path) -> None:
    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    labeled = [r for r in rows if r.get("label", "").strip() in ("0", "1")]
    y = np.array([int(r["label"]) for r in labeled])
    if len(labeled) < 10 or y.min() == y.max():
        print(
            f"{len(labeled)} labelled rows ({int(y.sum()) if len(y) else 0} good): "
            "need at least 10 rows with both labels 0 and 1."
        )
        return
    print(f"{len(labeled)} labelled clips, {int(y.sum())} good / {int((1 - y).sum())} bad\n")

    table = []
    for column in ("score_mean", "passed_fraction", *METRIC_NAMES):
        x = np.array([float(r[column]) for r in labeled])
        auc = _auc(x, y)
        threshold, precision, recall = _best_threshold(x, y)
        table.append((abs(auc - 0.5), column, auc, threshold, precision, recall))
    table.sort(reverse=True)

    print(f"{'metric':34s} {'AUC':>6s} {'best >=':>9s} {'precision':>10s} {'recall':>8s}")
    for _, column, auc, threshold, precision, recall in table:
        print(f"{column:34s} {auc:6.2f} {threshold:9.3f} {precision:10.2f} {recall:8.2f}")
    print(
        "\nAUC 0.5 = useless, 1.0 = perfect separation, far below 0.5 = the metric "
        "points the wrong way.\n'best >=' maximises recall minus false-positive rate."
    )

    kept = np.array([str(r["kept"]).lower() in ("true", "1") for r in labeled])
    tp = int(np.sum(kept & (y == 1)))
    fp = int(np.sum(kept & (y == 0)))
    fn = int(np.sum(~kept & (y == 1)))
    print(
        f"\ncurrent filter vs your labels: kept good {tp}, kept bad {fp}, "
        f"dropped good {fn}, dropped bad {int(np.sum(~kept & (y == 0)))} "
        f"(precision {tp / max(tp + fp, 1):.2f}, recall {tp / max(tp + fn, 1):.2f})"
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def thresholds_from_args(args) -> Thresholds:
    values = {f.name: getattr(args, f.name) for f in fields(Thresholds)}
    if args.threshold is not None:  # shorthand kept from the previous version
        values.update(
            min_chroma_similarity=args.threshold,
            min_vocal_support=args.threshold,
            enter_threshold=args.threshold,
            exit_threshold=0.8 * args.threshold,
        )
    return Thresholds(**values)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("data_dir", type=Path, nargs="?")
    parser.add_argument("--analyze", type=Path, metavar="CSV", help="analyse a labelled samples.csv and exit")

    scoring = parser.add_argument_group("scoring (slow stage)")
    scoring.add_argument("--window-seconds", type=float, default=8.0, help="width of the window each step is scored on")
    scoring.add_argument("--score-stride", type=int, default=1, help="score every Nth step and repeat (faster)")
    scoring.add_argument("--workers", type=int, default=1, help="songs processed in parallel")
    scoring.add_argument("--overwrite", action="store_true", help="recompute existing .scores.npz")
    scoring.add_argument("--regions-only", action="store_true", help="reuse saved scores, only re-extract regions")

    random_group = parser.add_argument_group("listening test")
    random_group.add_argument("--random", type=int, default=0, metavar="N", help="export N random clips with their metrics")
    random_group.add_argument("--random-from", choices=("all", "kept", "rejected", "mixed"), default="all", help="sample from any window, kept ones, rejected ones, or ones straddling a region boundary")
    random_group.add_argument("--random-seconds", type=float, default=8.0)
    random_group.add_argument("--random-dir", type=Path, default=Path("random_samples"))
    random_group.add_argument("--seed", type=int, default=0)

    thresholds = parser.add_argument_group("thresholds (cheap to change with --regions-only)")
    thresholds.add_argument("--threshold", type=float, default=None, help="shorthand: sets min-chroma-similarity, min-vocal-support and enter-threshold (exit = 0.8x)")
    for field in fields(Thresholds):
        flag = "--" + field.name.replace("_", "-")
        if field.name == "require":
            thresholds.add_argument(flag, choices=("music", "vocals", "either", "both"), default=field.default, help=THRESHOLD_HELP[field.name])
        else:
            thresholds.add_argument(flag, type=type(field.default), default=field.default, help=f"{THRESHOLD_HELP[field.name]} (default %(default)s)")
    args = parser.parse_args()

    if args.analyze is not None:
        analyze(args.analyze)
        return
    if args.data_dir is None:
        parser.error("data_dir is required (unless --analyze is used)")
    roots = dataset_roots(args.data_dir)
    if not roots:
        parser.error(f"No numbered dataset roots found below {args.data_dir}")

    th = thresholds_from_args(args)
    run_pipeline(roots, th, args)
    if args.random:
        export_random(
            roots, args.random, args.random_from, args.random_seconds,
            args.random_dir, th, args.seed,
        )


if __name__ == "__main__":
    main()