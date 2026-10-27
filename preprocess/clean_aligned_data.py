"""Create a filtered, chunked copy of a preprocessed Pop2Piano dataset.

The input must already have been processed by ``pop_align.py``,
``bpm_quantize.py``, and ``split_spleeter.py``.  Each retained output sample
has the same top-level metadata and song-directory layout as ``data/``.
"""

from __future__ import annotations

import argparse
import copy
import shutil
import sys
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
import soundfile as sf
from omegaconf import OmegaConf

sys.path.append(str(Path(__file__).resolve().parents[1]))
from evaluate.midi_melody_accuracy import get_highest_pitches_from_piano_roll
from utils.dsp import get_stereo
from data_roots import dataset_roots, mirrored_root


DEFAULT_SR = 44100
DEFAULT_HOP_LENGTH = 1024


def _local_metrics(
    f0: np.ndarray,
    midi_pitch: np.ndarray,
    start_frame: int,
    end_frame: int,
) -> dict[str, float]:
    """Calculate frame-level alignment metrics for one time interval."""
    ref = f0[start_frame:end_frame]
    est = midi_pitch[start_frame:end_frame]
    ref_voiced = np.isfinite(ref)
    est_voiced = np.isfinite(est)
    both_voiced = ref_voiced & est_voiced

    voiced_union = ref_voiced | est_voiced
    true_positive = np.sum(both_voiced)
    true_negative = np.sum(~ref_voiced & ~est_voiced)
    ref_count = np.sum(ref_voiced)
    unvoiced_count = np.sum(~ref_voiced)
    voicing_accuracy = (
        float(
            0.5
            * (
            (true_positive / ref_count if ref_count else 0.0)
            + (true_negative / unvoiced_count if unvoiced_count else 0.0)
            )
        )
        if voiced_union.size
        else 0.0
    )
    result = {
        "voicing_accuracy": voicing_accuracy,
        "voiced_fraction": float(np.mean(ref_voiced)) if ref.size else 0.0,
        "midi_voiced_fraction": float(np.mean(est_voiced)) if est.size else 0.0,
    }
    if not np.any(both_voiced):
        result.update(melody_chroma_accuracy=0.0, melody_pitch_accuracy=0.0)
        return result

    pitch_error = np.abs(librosa.hz_to_midi(ref[both_voiced]) - est[both_voiced])
    chroma_error = np.minimum(pitch_error % 12.0, 12.0 - pitch_error % 12.0)
    result.update({
        "melody_chroma_accuracy": float(np.mean(chroma_error < 0.5)),
        "melody_pitch_accuracy": float(np.mean(pitch_error < 0.5)),
    })
    return result


def _midi_chroma(piano_roll: np.ndarray) -> np.ndarray:
    chroma = np.zeros((12, piano_roll.shape[1]), dtype=np.float32)
    for pitch in range(piano_roll.shape[0]):
        chroma[pitch % 12] += piano_roll[pitch]
    return chroma


def _local_music_metrics(
    audio_chroma: np.ndarray,
    midi_chroma: np.ndarray,
    start_frame: int,
    end_frame: int,
) -> dict[str, float]:
    """Compare synchronized chroma against a time-shuffled null baseline."""
    audio = audio_chroma[:, start_frame:end_frame]
    midi = midi_chroma[:, start_frame:end_frame]
    length = min(audio.shape[1], midi.shape[1])
    if length == 0:
        return {
            "music_chroma_similarity": 0.0,
            "music_chroma_null_similarity": 0.0,
            "music_chroma_alignment_margin": 0.0,
            "music_chroma_coverage": 0.0,
            "voicing_accuracy": 0.0,
            "voiced_fraction": 0.0,
            "midi_voiced_fraction": 0.0,
        }
    audio = audio[:, :length]
    midi = midi[:, :length]
    audio_norm = np.linalg.norm(audio, axis=0)
    midi_norm = np.linalg.norm(midi, axis=0)
    audio_active = audio_norm > max(1e-6, np.percentile(audio_norm, 20))
    midi_active = midi_norm > 1e-6
    active = audio_active & midi_active
    if not np.any(active):
        return {
            "music_chroma_similarity": 0.0,
            "music_chroma_null_similarity": 0.0,
            "music_chroma_alignment_margin": 0.0,
            "music_chroma_coverage": 0.0,
            "voicing_accuracy": float(np.mean(audio_active == midi_active)),
            "voiced_fraction": float(np.mean(audio_active)),
            "midi_voiced_fraction": float(np.mean(midi_active)),
        }
    similarity = np.sum(audio[:, active] * midi[:, active], axis=0)
    similarity /= audio_norm[active] * midi_norm[active]
    synchronized_similarity = float(np.mean(np.clip(similarity, 0.0, 1.0)))

    rng = np.random.default_rng(0)
    null_scores = []
    active_indices = np.flatnonzero(active)
    for _ in range(8):
        shuffled = midi[:, rng.permutation(length)]
        shuffled_norm = np.linalg.norm(shuffled[:, active_indices], axis=0)
        valid = shuffled_norm > 1e-6
        if np.any(valid):
            null_similarity = np.sum(
                audio[:, active_indices[valid]] * shuffled[:, active_indices[valid]],
                axis=0,
            )
            null_similarity /= (
                audio_norm[active_indices[valid]] * shuffled_norm[valid]
            )
            null_scores.append(float(np.mean(np.clip(null_similarity, 0.0, 1.0))))
    null_mean = float(np.mean(null_scores)) if null_scores else 0.0
    return {
        "music_chroma_similarity": synchronized_similarity,
        "music_chroma_null_similarity": null_mean,
        "music_chroma_alignment_margin": synchronized_similarity - null_mean,
        "music_chroma_coverage": float(np.mean(active)),
        "voicing_accuracy": float(np.mean(audio_active == midi_active)),
        "voiced_fraction": float(np.mean(audio_active)),
        "midi_voiced_fraction": float(np.mean(midi_active)),
    }


def _activity_metrics(
    audio: np.ndarray,
    midi_chroma: np.ndarray,
    start_frame: int,
    end_frame: int,
) -> dict[str, float]:
    """Measure whether audio and MIDI are active for the same frames."""
    audio = audio[
        start_frame * DEFAULT_HOP_LENGTH : end_frame * DEFAULT_HOP_LENGTH
    ]
    midi = midi_chroma[:, start_frame:end_frame]
    if audio.size == 0 or midi.shape[1] == 0:
        return {"activity_iou": 0.0, "activity_ratio": 0.0}
    audio_rms = librosa.feature.rms(
        y=audio, frame_length=2048, hop_length=DEFAULT_HOP_LENGTH
    )[0]
    audio_active = audio_rms > max(1e-6, np.percentile(audio_rms, 20))
    midi_active = np.any(midi > 1e-6, axis=0)
    size = min(audio_active.size, midi_active.size)
    audio_active, midi_active = audio_active[:size], midi_active[:size]
    intersection = np.sum(audio_active & midi_active)
    union = np.sum(audio_active | midi_active)
    return {
        "activity_iou": float(intersection / union) if union else 0.0,
        "activity_ratio": float(
            min(np.mean(audio_active), np.mean(midi_active))
            / max(np.mean(audio_active), np.mean(midi_active), 1e-6)
        ),
    }


def _onset_metrics(
    audio: np.ndarray,
    piano_roll: np.ndarray,
    start_frame: int,
    end_frame: int,
    sr: int,
) -> dict[str, float]:
    """Compare song onset density and timing with MIDI note onsets."""
    audio_start = start_frame * DEFAULT_HOP_LENGTH
    audio_end = end_frame * DEFAULT_HOP_LENGTH
    audio_onsets = librosa.onset.onset_detect(
        y=audio[audio_start:audio_end],
        sr=sr,
        hop_length=DEFAULT_HOP_LENGTH,
        backtrack=False,
    )
    midi = np.any(piano_roll[:, start_frame:end_frame] > 1e-6, axis=0)
    midi_onsets = np.flatnonzero(midi & ~np.r_[False, midi[:-1]])
    if not audio_onsets.size and not midi_onsets.size:
        return {"onset_f1": 1.0, "note_density_ratio": 1.0}
    density_ratio = min(audio_onsets.size, midi_onsets.size) / max(
        audio_onsets.size, midi_onsets.size, 1
    )
    matched = sum(
        np.any(np.abs(midi_onsets - onset) <= 3) for onset in audio_onsets
    )
    precision = matched / max(audio_onsets.size, 1)
    recall = matched / max(midi_onsets.size, 1)
    onset_f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "onset_f1": float(onset_f1),
        "note_density_ratio": float(density_ratio),
    }


def _clip_midi(pm: pretty_midi.PrettyMIDI, start: float, end: float) -> pretty_midi.PrettyMIDI:
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
            event
            for event in instrument.control_changes
            if start <= event.time < end
        ]
        for event in instrument.control_changes:
            event.time -= start
    return clipped


def _chunk_notes(
    notes: np.ndarray, beatstep: np.ndarray, start_idx: int, end_idx: int
) -> np.ndarray:
    """Clip beat-indexed notes and make their indices relative to a chunk."""
    if notes.size == 0:
        return notes.reshape(0, 4)
    selected = notes[(notes[:, 0] < end_idx) & (notes[:, 1] > start_idx)].copy()
    if selected.size == 0:
        return np.empty((0, 4), dtype=notes.dtype)
    selected[:, 0] = np.maximum(selected[:, 0], start_idx) - start_idx
    selected[:, 1] = np.minimum(selected[:, 1], end_idx) - start_idx
    return selected[selected[:, 1] > selected[:, 0]]


def _write_chunk(
    source_meta: Path,
    output_dir: Path,
    sample_index: int,
    start_idx: int,
    end_idx: int,
    beatstep: np.ndarray,
    notes: np.ndarray,
    song_audio: np.ndarray,
    song_sr: int,
    aligned_midi: pretty_midi.PrettyMIDI,
    metrics: dict[str, float],
    minimal: bool,
) -> None:
    source = OmegaConf.load(source_meta)
    piano_id = str(source.piano.ytid)
    song_id = str(source.song.ytid)
    chunk_id = f"{piano_id}__{song_id}__chunk{sample_index:04d}"
    chunk_dir = output_dir / chunk_id
    chunk_dir.mkdir(parents=True)

    start = float(beatstep[start_idx])
    end = float(beatstep[end_idx])
    chunk_audio = song_audio[int(start * song_sr) : int(end * song_sr)]
    chunk_midi = _clip_midi(aligned_midi, start, end)
    chunk_midi_path = chunk_dir / f"{chunk_id}.qmidi.mid"
    chunk_midi.write(str(chunk_midi_path))
    sf.write(chunk_dir / f"{chunk_id}.pitchshift.wav", chunk_audio, song_sr)

    chunk_beatstep = beatstep[start_idx : end_idx + 1] - start
    np.save(chunk_dir / f"{chunk_id}.beatstep.npy", chunk_beatstep)
    np.save(chunk_dir / f"{chunk_id}.notes.npy", _chunk_notes(notes, beatstep, start_idx, end_idx))

    if not minimal:
        chunk_midi.write(str(chunk_dir / f"{chunk_id}.mid"))
        piano_audio = chunk_midi.fluidsynth(song_sr)
        qmix = get_stereo(chunk_audio, piano_audio, pop_scale=0.4)
        qmix = np.asarray(qmix.T, dtype=np.float32)
        sf.write(str(chunk_dir / f"{chunk_id}.qmix.flac"), qmix, song_sr, format="flac")
        np.save(chunk_dir / f"{chunk_id}.beattime.npy", chunk_beatstep[::2])
        np.save(
            chunk_dir / f"{chunk_id}.beatinterval.npy",
            np.diff(chunk_beatstep[::2]),
        )

    metadata = OmegaConf.create(source)
    metadata.piano.ytid = chunk_id
    metadata.piano.duration = end - start
    metadata.piano.title = f"{source.piano.title} [chunk {sample_index:04d}]"
    metadata.song.ytid = chunk_id
    metadata.song.duration = end - start
    metadata.song.title = f"{source.song.title} [chunk {sample_index:04d}]"
    metadata.eval = OmegaConf.create(metrics)
    metadata.eval.source_piano_ytid = piano_id
    metadata.eval.source_song_ytid = song_id
    metadata.eval.start_seconds = start
    metadata.eval.end_seconds = end
    OmegaConf.save(metadata, output_dir / f"{chunk_id}.yaml")

    if not minimal:
        # Preserve compatibility with older consumers expecting root-level files.
        sf.write(output_dir / f"{chunk_id}.wav", chunk_audio, song_sr)
        chunk_midi.write(str(output_dir / f"{chunk_id}.mid"))


def clean_sample(
    source_meta: Path,
    output_dir: Path,
    threshold: float,
    window_seconds: float,
    hop_seconds: float,
    min_notes: int,
    min_note_onsets: int,
    min_activity_iou: float,
    min_alignment_margin: float,
    min_onset_f1: float,
    sample_index_start: int,
    minimal: bool,
) -> int:
    source = OmegaConf.load(source_meta)
    source_dir = source_meta.parent / str(source.piano.ytid)
    song_id = str(source.song.ytid)
    song_path = source_dir / f"{song_id}.pitchshift.wav"
    midi_path = source_dir / f"{song_id}.qmidi.mid"
    vocal_path = source_dir / f"{song_id}.vocals.mp3"
    notes_path = source_dir / f"{song_id}.notes.npy"
    beatstep_path = source_dir / f"{song_id}.beatstep.npy"
    required = (song_path, midi_path, notes_path, beatstep_path)
    if not all(path.exists() for path in required):
        print(f"skip {source_meta}: missing required preprocessed file")
        return 0

    song_audio, song_sr = librosa.load(song_path, sr=DEFAULT_SR, mono=True)
    metric_audio = (
        librosa.load(vocal_path, sr=DEFAULT_SR, mono=True)[0]
        if vocal_path.exists()
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
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    frame_count = min(len(f0), song_chroma.shape[1])
    frame_times = np.arange(frame_count) * DEFAULT_HOP_LENGTH / DEFAULT_SR
    piano_roll = pm.get_piano_roll(
        fs=DEFAULT_SR / DEFAULT_HOP_LENGTH, times=frame_times
    )
    song_chroma = song_chroma[:, :frame_count]
    midi_chroma = _midi_chroma(piano_roll)
    midi_pitch = get_highest_pitches_from_piano_roll(piano_roll).astype(float)
    midi_pitch[midi_pitch == 0] = np.nan

    beatstep = np.load(beatstep_path)
    notes = np.load(notes_path)
    duration = min(float(beatstep[-1]), len(song_audio) / song_sr)
    window_steps = max(2, int(np.searchsorted(beatstep, window_seconds, side="left")))
    hop_steps = max(1, int(np.searchsorted(beatstep, hop_seconds, side="left")))
    kept = 0
    for start_idx in range(0, len(beatstep) - window_steps, hop_steps):
        end_idx = start_idx + window_steps
        if end_idx >= len(beatstep) or beatstep[end_idx] > duration:
            continue
        start_frame = int(np.searchsorted(frame_times, beatstep[start_idx], side="left"))
        end_frame = int(np.searchsorted(frame_times, beatstep[end_idx], side="left"))
        music_local = _local_music_metrics(
            song_chroma, midi_chroma, start_frame, end_frame
        )
        vocal_local = _local_metrics(f0, midi_pitch, start_frame, end_frame)
        local = dict(music_local)
        local["metric_source"] = "song_chroma"
        local["vocal_melody_chroma_accuracy"] = vocal_local[
            "melody_chroma_accuracy"
        ]
        local["vocal_melody_pitch_accuracy"] = vocal_local[
            "melody_pitch_accuracy"
        ]
        local["vocal_voicing_accuracy"] = vocal_local["voicing_accuracy"]
        local.update(_activity_metrics(song_audio, midi_chroma, start_frame, end_frame))
        local.update(
            _onset_metrics(song_audio, piano_roll, start_frame, end_frame, song_sr)
        )
        local["alignment_score"] = float(
            0.7 * local["music_chroma_alignment_margin"]
            + 0.2 * local["activity_iou"]
            + 0.1 * local["music_chroma_coverage"]
        )
        if (
            local["music_chroma_similarity"] < threshold
            or local["music_chroma_alignment_margin"] < min_alignment_margin
            or local["activity_iou"] < min_activity_iou
            or local["onset_f1"] < min_onset_f1
        ):
            continue
        chunk_notes = notes[
            (notes[:, 0] < end_idx) & (notes[:, 1] > start_idx)
        ]
        chunk_onsets = (
            np.unique(chunk_notes[:, 0].astype(int)).size
            if chunk_notes.size
            else 0
        )
        local["midi_note_count"] = int(chunk_notes.shape[0])
        local["midi_onset_count"] = int(chunk_onsets)
        if (
            local["midi_note_count"] < min_notes
            or local["midi_onset_count"] < min_note_onsets
        ):
            continue
        _write_chunk(
            source_meta,
            output_dir,
            sample_index_start + kept,
            start_idx,
            end_idx,
            beatstep,
            notes,
            song_audio,
            song_sr,
            pm,
            local,
            minimal,
        )
        kept += 1
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--window-seconds", type=float, default=8.0)
    parser.add_argument("--hop-seconds", type=float, default=8.0)
    parser.add_argument(
        "--min-notes",
        type=int,
        default=12,
        help="minimum MIDI notes per retained chunk",
    )
    parser.add_argument(
        "--min-note-onsets",
        type=int,
        default=4,
        help="minimum distinct MIDI onset times per retained chunk",
    )
    parser.add_argument("--min-activity-iou", type=float, default=0.2)
    parser.add_argument("--min-alignment-margin", type=float, default=0.05)
    parser.add_argument("--min-onset-f1", type=float, default=0.0)
    parser.add_argument(
        "--minimal",
        action="store_true",
        help=(
            "write only metadata, chunk audio, chunk qmidi, beatstep, and notes; "
            "use non-overlapping windows"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            parser.error(f"{args.output_dir} is not empty; use --overwrite to replace it")
        shutil.rmtree(args.output_dir)
    total = 0
    if args.minimal:
        args.hop_seconds = args.window_seconds
    roots = dataset_roots(args.data_dir)
    if not roots:
        parser.error(f"No numbered dataset roots found below {args.data_dir}")
    for root_index, source_root in enumerate(roots):
        output_root = mirrored_root(source_root, args.output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        for index, meta_file in enumerate(sorted(source_root.glob("*.yaml"))):
            total += clean_sample(
                meta_file,
                output_root,
                args.threshold,
                args.window_seconds,
                args.hop_seconds,
                args.min_notes,
                args.min_note_onsets,
                args.min_activity_iou,
                args.min_alignment_margin,
                args.min_onset_f1,
                (root_index * 100000) + index * 1000,
                args.minimal,
            )
    print(f"kept {total} aligned chunks in {args.output_dir}")


if __name__ == "__main__":
    main()
