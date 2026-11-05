"""Overfit test: can the whole pipeline memorise a handful of real segments?

This is a bug detector, not an evaluation. A model that cannot memorise ~16
segments will not learn from 500 songs either, so run this before a long
training. It uses the same config, dataset, tokenizer and model as training and
checks three things on the SAME segments it trained on:

1. the loss goes (close to) zero;
2. greedy generation, segment by segment, reproduces the target notes;
3. the real inference path (``single_inference``: audio windows, beat grids,
   chunking, decoding) reproduces the notes of a run of consecutive segments.

    python overfit_test.py data --config config.yaml

Dropout is disabled (the model stays in eval mode) so memorisation is fast.
The exit code is 0 when every check passes, 1 otherwise.
"""

import argparse
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import LambdaLR

from data_roots import metadata_files
from dataset import Pop2PianoDataset, collate_batch
from midi_tokenizer import EOS
from transformer_wrapper import TransformerWrapper


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def longest_run(samples, indices, step):
    """Longest list of dataset indices that are consecutive segments of one song.

    ``samples`` is ``dataset.samples`` ((song index, start step) pairs); two
    segments are consecutive when they are ``step`` beat steps apart.
    """
    best, current = [], []
    for i in sorted(indices):
        if (
            current
            and samples[i][0] == samples[current[-1]][0]
            and samples[i][1] - samples[current[-1]][1] == step
        ):
            current.append(i)
        else:
            current = [i]
        if len(current) > len(best):
            best = list(current)
    return best


def trim_after_eos(tokens):
    """Drop the decoder start token and everything after the first EOS (kept)."""
    tokens = np.asarray(tokens)[1:]
    eos = np.flatnonzero(tokens == EOS)
    return tokens[: eos[0] + 1] if eos.size else tokens


def note_keys(notes, with_duration=True):
    """Set of (onset, pitch[, duration]) tuples of an (N, 4) note array."""
    notes = np.asarray(notes).reshape(-1, 4)
    if with_duration:
        return {(int(a), int(c), int(b - a)) for a, b, c, _ in notes}
    return {(int(a), int(c)) for a, b, c, _ in notes}


def count_matches(pred, true):
    """(true positives, #predicted, #true) for two sets of note keys."""
    return len(pred & true), len(pred), len(true)


def f1_from_counts(tp, n_pred, n_true):
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_true if n_true else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #
def to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def train(model, batches, weights, args, config, device):
    """Full-batch training on the fixed set (gradient accumulation over micro-batches)."""
    params = [p for p in model.parameters() if p.requires_grad]
    name = config.training.optimizer.lower()
    lr = args.lr or (config.training.lr if name == "adafactor" else 3e-4)
    if name == "adafactor":
        from transformers.optimization import Adafactor

        optimizer = Adafactor(
            params,
            lr=lr,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )
    else:
        optimizer = torch.optim.AdamW(params, lr=lr)
    warmup = max(1, args.warmup)
    scheduler = LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / warmup))
    clip = float(config.training.gradient_clip_val)

    n_trainable = sum(p.numel() for p in params)
    print(f"training {n_trainable / 1e6:.1f} M parameters, {name}, lr {lr}, "
          f"{len(batches)} micro-batch(es), up to {args.steps} steps")

    history = []
    start = time.time()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for batch, weight in zip(batches, weights):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.bf16):
                loss = model(batch).loss
            (loss * weight).backward()
            total += loss.item() * weight
        grad_norm = torch.nn.utils.clip_grad_norm_(params, clip)
        optimizer.step()
        scheduler.step()
        history.append(total)
        if step == 0 or (step + 1) % 10 == 0:
            print(f"  step {step + 1:4d}  loss {total:8.4f}  grad norm {float(grad_norm):8.2f}  "
                  f"({time.time() - start:5.0f} s)")
        if total < args.target_loss:
            print(f"  loss below {args.target_loss} at step {step + 1}, stopping")
            break
    return history


@torch.no_grad()
def generate_segments(model, batches, max_length, args, device):
    """Greedy generation for every segment; returns a list of token arrays."""
    generated = []
    for batch in batches:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.bf16):
            embeds, mask = model.encode_batch(batch)
            out = model.transformer.generate(
                inputs_embeds=embeds,
                attention_mask=mask,
                max_length=max_length,
                do_sample=False,
                num_beams=1,
            )
        generated.extend(trim_after_eos(row) for row in out.cpu().numpy())
    return generated


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = OmegaConf.load(args.config)
    composer_value = config.composer_to_feature_token[args.composer]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    max_len = int(config.dataset.target_length) + 1  # + decoder start token

    # ---- data ------------------------------------------------------------
    files = sorted(metadata_files(args.data_dir))[: args.max_songs]
    dataset = Pop2PianoDataset(files, config, composer_value, stride_fraction=1.0)
    n = min(args.n_segments, len(dataset))
    if args.spread:
        indices = sorted(set(np.linspace(0, len(dataset) - 1, n).round().astype(int).tolist()))
    else:
        indices = list(range(n))
    items = [dataset[i] for i in indices]
    targets = [dataset.targets[i] for i in indices]
    print(f"\nusing {len(indices)} segments "
          f"({sum(len(t) for t in targets)} target tokens, max {max(len(t) for t in targets)})")

    micro = max(1, args.micro_batch)
    batches, weights = [], []
    total_tokens = sum(len(t) for t in targets)
    for start in range(0, len(items), micro):
        chunk = items[start : start + micro]
        batches.append(to_device(collate_batch(chunk), device))
        weights.append(sum(len(t) for t in targets[start : start + micro]) / total_tokens)

    # ---- model + training ----------------------------------------------------
    model = TransformerWrapper(config).to(device)
    model.eval()  # no dropout: memorisation should be fast
    history = train(model, batches, weights, args, config, device)
    final_loss = history[-1]
    results = {"loss": final_loss <= args.max_final_loss}
    print(f"\n1. final loss {final_loss:.4f} (first {history[0]:.2f})  "
          f"-> {'PASS' if results['loss'] else 'FAIL'} (needs <= {args.max_final_loss})")

    # ---- 2. greedy generation, segment by segment --------------------------------
    generated = generate_segments(model, batches, max_len, args, device)
    exact = sum(np.array_equal(g, t) for g, t in zip(generated, targets))
    tokenizer = dataset.tokenizer
    steps = dataset.steps_per_segment
    totals = {"full": [0, 0, 0], "pitch": [0, 0, 0]}
    for g, t in zip(generated, targets):
        pred = tokenizer.tokens_to_notes(g, 0, steps)
        true = tokenizer.tokens_to_notes(t, 0, steps)
        for key, with_duration in (("full", True), ("pitch", False)):
            counts = count_matches(note_keys(pred, with_duration), note_keys(true, with_duration))
            totals[key] = [a + b for a, b in zip(totals[key], counts)]
    f1_full, f1_pitch = f1_from_counts(*totals["full"]), f1_from_counts(*totals["pitch"])
    results["generation"] = f1_full >= args.min_f1
    print(f"2. greedy generation: {exact}/{len(targets)} segments identical token-by-token; "
          f"note F1 {f1_full:.3f} (onset+pitch+duration), {f1_pitch:.3f} (onset+pitch)  "
          f"-> {'PASS' if results['generation'] else 'FAIL'} (needs >= {args.min_f1})")
    shown = next((i for i, (g, t) in enumerate(zip(generated, targets)) if not np.array_equal(g, t)), 0)
    print(f"   segment {shown}: target    {tokenizer.to_string(targets[shown])[:14]}")
    print(f"   segment {shown}: generated {tokenizer.to_string(generated[shown])[:14]}")

    # ---- 3. the real inference path ---------------------------------------------------
    run = longest_run(dataset.samples, indices, steps)
    if len(run) < 2:
        results["inference"] = None
        print("3. inference path: skipped (no run of 2+ consecutive segments in the subset; "
              "do not use --spread)")
    else:
        song_index, first_step = dataset.samples[run[0]]
        song = dataset.songs[song_index]
        beatstep = song["beatstep"]
        last_step = first_step + len(run) * steps
        t0, t1 = float(beatstep[first_step]), float(beatstep[last_step])
        audio, _ = librosa.load(
            song["song_path"], sr=model.sample_rate, mono=True, offset=t0, duration=t1 - t0
        )
        truth = np.concatenate(
            [tokenizer.tokens_to_notes(dataset.targets[i], k * steps, steps) for k, i in enumerate(run)]
        )
        _, notes, _ = model.single_inference(
            torch.from_numpy(audio.astype(np.float32)).to(device),
            beatstep[first_step : last_step + 1] - t0,
            max_length=max_len,
            max_batch_size=micro,
            n_bars=dataset.n_bars,
            composer_value=composer_value,
            generation_kwargs={"do_sample": False},
        )
        notes = np.asarray(notes).reshape(-1, 4)
        f1_chunks = []
        for k in range(len(run)):
            lo, hi = k * steps, (k + 1) * steps
            pick = lambda a: a[(a[:, 0] >= lo) & (a[:, 0] < hi)]
            f1_chunks.append(
                f1_from_counts(*count_matches(note_keys(pick(notes)), note_keys(pick(truth))))
            )
        interior = f1_chunks[1:-1] or f1_chunks
        overall = f1_from_counts(*count_matches(note_keys(notes), note_keys(truth)))
        results["inference"] = float(np.mean(interior)) >= args.min_f1_inference
        print(f"3. inference path on {len(run)} consecutive segments ({t1 - t0:.1f} s): "
              f"note F1 {overall:.3f}, interior chunks {np.mean(interior):.3f}  "
              f"-> {'PASS' if results['inference'] else 'FAIL'} (needs >= {args.min_f1_inference})")
        print("   per-chunk F1:", " ".join(f"{f:.2f}" for f in f1_chunks),
              "(first/last chunk lack audio context in this test)")

    # ---- verdict ----------------------------------------------------------------------
    failed = [name for name, ok in results.items() if ok is False]
    if not failed:
        print("\nALL CHECKS PASSED: the pipeline can memorise and reproduce its training data.")
        return 0
    print("\nFAILED:", ", ".join(failed))
    if "loss" in failed:
        print("- loss too high: look at the loss curve above. Flat from the start -> labels or "
              "learning rate (print a decoded target, try --lr); NaN -> try --lr 1e-4; "
              "slowly decreasing -> raise --steps.")
    if "generation" in failed and "loss" not in failed:
        print("- low loss but wrong generation: the decoding path differs from training "
              "(attention mask, max_length, decoder start token) or the model is in the wrong mode.")
    if "inference" in failed and "generation" not in failed:
        print("- segments are fine but single_inference is not: the inference chunking "
              "(windows, offsets, beat grids, steps per beat, n_bars) disagrees with the dataset.")
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--composer", default="composer1")
    parser.add_argument("--n_segments", type=int, default=16)
    parser.add_argument("--max_songs", type=int, default=5, help="only the first songs are read")
    parser.add_argument("--spread", action="store_true",
                        help="spread the segments over the dataset (skips check 3)")
    parser.add_argument("--micro_batch", type=int, default=8, help="lower it on out-of-memory")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=None, help="default: config lr (Adafactor) / 3e-4 (AdamW)")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--target_loss", type=float, default=0.02, help="stop early below this loss")
    parser.add_argument("--max_final_loss", type=float, default=0.05)
    parser.add_argument("--min_f1", type=float, default=0.95)
    parser.add_argument("--min_f1_inference", type=float, default=0.8)
    parser.add_argument("--bf16", action="store_true", help="bf16 autocast (less memory)")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default=None)
    sys.exit(main(parser.parse_args()))