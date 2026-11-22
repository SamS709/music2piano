"""Compare piano generation with ground-truth and Essentia-estimated beat grids."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
import mir_eval.transcription
from omegaconf import OmegaConf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_roots import metadata_files
from midiaudiopair import MidiAudioPair


def quantized_note_keys(pm: pretty_midi.PrettyMIDI, beatstep: np.ndarray):
    """Map MIDI note times to beat-grid onset/duration indices."""
    beatstep = np.asarray(beatstep, dtype=np.float64)
    if beatstep.size < 2:
        return set()

    midpoint = (beatstep[1:] + beatstep[:-1]) / 2.0
    key_map = {}
    for instrument in pm.instruments:
        for note in instrument.notes:
            on = int(np.digitize(note.start, midpoint))
            off = int(np.digitize(note.end, midpoint))
            on = int(np.clip(on, 0, len(beatstep) - 1))
            off = int(np.clip(off, 0, len(beatstep) - 1))
            if off <= on:
                off = min(on + 1, len(beatstep) - 1)
            key = (on, int(note.pitch), off - on)
            key_map[key] = max(key_map.get(key, 0), int(note.velocity))

    return set(key_map)


def f1_from_sets(predicted: set, target: set):
    tp = len(predicted & target)
    fp = len(predicted - target)
    fn = len(target - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "predicted": len(predicted),
        "target": len(target),
    }


def quantized_note_arrays(pm: pretty_midi.PrettyMIDI, beatstep: np.ndarray):
    """Return quantized MIDI intervals and pitches for mir_eval."""
    beatstep = np.asarray(beatstep, dtype=np.float64)
    if beatstep.size < 2:
        return np.zeros((0, 2)), np.zeros((0,))

    midpoint = (beatstep[1:] + beatstep[:-1]) / 2.0
    best = {}
    for instrument in pm.instruments:
        for note in instrument.notes:
            on = int(np.clip(np.digitize(note.start, midpoint), 0, len(beatstep) - 1))
            off = int(np.clip(np.digitize(note.end, midpoint), 0, len(beatstep) - 1))
            if off <= on:
                off = min(on + 1, len(beatstep) - 1)
            key = (on, int(note.pitch), off)
            best[key] = max(best.get(key, 0), int(note.velocity))

    ordered = sorted(best)
    intervals = np.asarray(
        [[beatstep[on], beatstep[off]] for on, _, off in ordered],
        dtype=np.float64,
    ).reshape(-1, 2)
    pitches = np.asarray([pretty_midi.note_number_to_hz(pitch) for _, pitch, _ in ordered])
    return intervals, pitches


def mir_eval_metrics(target_pm, predicted_pm, beatstep):
    ref_intervals, ref_pitches = quantized_note_arrays(target_pm, beatstep)
    est_intervals, est_pitches = quantized_note_arrays(predicted_pm, beatstep)
    if len(ref_intervals) == 0 or len(est_intervals) == 0:
        return {
            "onset_pitch": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "onset_pitch_duration": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        }

    scores = mir_eval.transcription.evaluate(
        ref_intervals,
        ref_pitches,
        est_intervals,
        est_pitches,
        onset_tolerance=0.05,
        offset_ratio=0.2,
        pitch_tolerance=50.0,
    )
    return {
        "onset_pitch": {
            "precision": float(scores.get("Onset_Precision", 0.0)),
            "recall": float(scores.get("Onset_Recall", 0.0)),
            "f1": float(scores.get("Onset_F-measure", 0.0)),
        },
        "onset_pitch_duration": {
            "precision": float(scores.get("Precision", 0.0)),
            "recall": float(scores.get("Recall", 0.0)),
            "f1": float(scores.get("F-measure", 0.0)),
        },
    }


def load_pair(meta_file: str):
    pair = MidiAudioPair(meta_file)
    if pair.error_code not in (MidiAudioPair.VALID,):
        return None
    if not os.path.isfile(pair.qmidi):
        return None
    if not os.path.isfile(pair.beatstep):
        return None
    if not os.path.isfile(pair.song):
        return None
    return pair


def estimated_beat_grid(pair: MidiAudioPair, sample_rate=44100):
    from preprocess.beat_quantizer import extract_rhythm, interpolate_beat_times

    y, sr = librosa.load(pair.song, sr=sample_rate)
    _, beat_times, _, _, _ = extract_rhythm(pair.song, y=y)
    beat_times = np.asarray(beat_times, dtype=np.float64)
    if beat_times.size < 2:
        raise ValueError(f"No usable estimated beats for {pair.song}")
    return interpolate_beat_times(beat_times, steps_per_beat=2, extend=True)


def evaluate_pair(pair: MidiAudioPair, wrapper, composer: str, model_id: str):
    ground_truth_grid = np.load(pair.beatstep)
    target_pm = pretty_midi.PrettyMIDI(pair.qmidi)

    y, sr = librosa.load(pair.song, sr=44100)
    estimated_grid = estimated_beat_grid(pair)

    grids = {
        "ground_truth": ground_truth_grid,
        "estimated": estimated_grid,
    }
    results = {}
    for name, beatstep in grids.items():
        midi_path = pair.generated(composer=composer, generated=model_id)
        if name == "ground_truth":
            midi_path = midi_path[:-4] + ".ground_truth.mid"
        else:
            midi_path = midi_path[:-4] + ".estimated.mid"
        os.makedirs(os.path.dirname(midi_path), exist_ok=True)

        pm, _, _, _ = wrapper.generate(
            audio_path=None,
            composer=composer,
            model=model_id,
            save_midi=True,
            save_mix=False,
            show_plot=False,
            midi_path=midi_path,
            beatsteps=beatstep,
            audio_y=y,
            audio_sr=sr,
        )

        metrics = mir_eval_metrics(target_pm, pm, beatstep)
        results[name] = {
            "onset_pitch_duration": metrics["onset_pitch_duration"],
            "onset_pitch": metrics["onset_pitch"],
            "beat_count": int(len(beatstep)),
            "beat_first": float(beatstep[0]),
            "beat_last": float(beatstep[-1]),
        }

    return results


def mean_results(results):
    names = ("ground_truth", "estimated")
    means = {}
    for name in names:
        usable = [r[name] for r in results if name in r]
        means[name] = {}
        for metric in ("onset_pitch_duration", "onset_pitch"):
            for field in ("precision", "recall", "f1"):
                means[name][f"{metric}.{field}"] = float(
                    np.mean([r[metric][field] for r in usable])
                )
    return means


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate note F1 using ground-truth versus estimated beat grids"
    )
    parser.add_argument("data_dir", help="Dataset directory")
    parser.add_argument("--ckpt", required=True, help="Model checkpoint")
    parser.add_argument("--config", default="config.yaml", help="Model config")
    parser.add_argument("--model-id", default="beat_eval")
    parser.add_argument("--composer", default=None)
    parser.add_argument("--output", default="beat_note_f1.csv")
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
        help="Fraction of song groups reserved for held-out evaluation",
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--all-songs",
        action="store_true",
        help="Evaluate every valid song instead of applying the held-out split",
    )
    args = parser.parse_args()

    import torch
    from dataset import split_by_song
    from transformer_wrapper import TransformerWrapper

    config = OmegaConf.load(args.config)
    wrapper = TransformerWrapper.load_from_checkpoint(args.ckpt, config=config)
    wrapper.eval()
    if torch.cuda.is_available():
        wrapper = wrapper.cuda()

    if args.composer is None:
        args.composer = str(next(iter(wrapper.composer_to_feature_token)))

    meta_files = [str(path) for path in metadata_files(args.data_dir)]
    if args.all_songs:
        eval_files = meta_files
        split_description = f"all {len(eval_files)} metadata files"
    else:
        _, eval_files = split_by_song(
            meta_files,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
        split_description = (
            f"held-out split: {len(eval_files)} of {len(meta_files)} metadata files "
            f"(by song, seed={args.seed})"
        )
    print(split_description)

    layer_weights = torch.softmax(
        wrapper.audio_encoder.layer_logits.detach().float().cpu(),
        dim=0,
    ).tolist()
    print("MERT layer weights:")
    for index, weight in enumerate(layer_weights):
        print(f"  layer {index:2d}: {weight:.6f}")

    rows = []
    per_song = []
    for meta_file in eval_files:
        pair = load_pair(str(meta_file))
        if pair is None:
            continue
        try:
            result = evaluate_pair(pair, wrapper, args.composer, args.model_id)
        except Exception as exc:
            print(f"Skipping {meta_file}: {exc}")
            continue
        per_song.append(result)
        for name in ("ground_truth", "estimated"):
            rows.append({
                "song_id": pair.yaml.song.ytid,
                "beat_grid": name,
                "precision_full": result[name]["onset_pitch_duration"]["precision"],
                "recall_full": result[name]["onset_pitch_duration"]["recall"],
                "f1_full": result[name]["onset_pitch_duration"]["f1"],
                "precision_onset_pitch": result[name]["onset_pitch"]["precision"],
                "recall_onset_pitch": result[name]["onset_pitch"]["recall"],
                "f1_onset_pitch": result[name]["onset_pitch"]["f1"],
            })

    if not rows:
        raise SystemExit("No valid songs were evaluated")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = mean_results(per_song)
    summary["layer_weights"] = layer_weights
    summary["held_out"] = not args.all_songs
    summary["evaluation_metadata_files"] = len(eval_files)
    summary_path = output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {output} and {summary_path}")


if __name__ == "__main__":
    main()
